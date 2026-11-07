"""Independent exact-arithmetic adjudication for S26 prefix/recurrence audit.
The deployed recurrence, thresholds, and output bits are NEVER altered here.
Algebraically equivalent formulae need not give identical binary64 decisions.
Every disagreeing *whole trace* is verified with dyadic rational arithmetic,
rounding each original floating operation independently (nearest, ties-to-even).
A separately unrounded recurrence must also equal the exact prefix identity.
Non-finite values, numerical-distance failures, or emulator disagreement STOP.
"""
from fractions import Fraction as F
from pathlib import Path
import math, json, hashlib
import numpy as np

def round_dyadic(q):
    """Correctly round a finite dyadic Fraction to IEEE binary64, ties-to-even."""
    if q == 0:
        return 0.0
    n, den = q.numerator, q.denominator
    if den <= 0 or den & (den-1):
        raise ValueError("Non-dyadic input to binary64 emulator")
    negative = n < 0
    n = abs(n)
    e = -(den.bit_length()-1)
    # At most 53 significant bits and no bits below 2**-1074.
    shift = max(0, n.bit_length()-53, -1074-e)
    if shift:
        t, r = divmod(n, 1 << shift)
        half = 1 << (shift-1)
        if r > half or (r == half and t & 1):
            t += 1
        n = t
        e += shift
    try:
        ans = math.ldexp(float(n), e)
    except OverflowError as exc:
        raise ValueError("Overflow in exact binary64 emulator") from exc
    if not math.isfinite(ans):
        raise ValueError("Nonfinite emulated operation")
    return -ans if negative else ans

def same_values(a, b):
    """Exact binary64 values; signed zero is immaterial to max/strict >."""
    return np.array_equal(a, b)

def exact_trace_audit(seq, deployed, prefix, h, k, mismatch_samples):
    seq = np.asarray(seq, dtype=np.float64)
    target = np.asarray(deployed, dtype=np.float64)
    prefix = np.asarray(prefix, dtype=np.float64)
    if seq.ndim != 1 or target.shape != seq.shape or prefix.shape != seq.shape:
        raise ValueError("Bad exact-audit trace shape")
    if not all(np.isfinite(a).all() for a in (seq,target,prefix)) or not math.isfinite(h):
        raise ValueError("Nonfinite exact-audit value")
    fk, fh = F.from_float(float(k)), F.from_float(float(h))
    cp = cm = 0.0
    pp = pm = pmin = mmin = 0.0
    rcp = rcm = ep = em = emin_p = emin_m = F(0)
    rounded_rec, rounded_prefix = [], []
    records = []
    selected = set(int(i) for i in mismatch_samples)
    for t, x in enumerate(seq):
        fx = F.from_float(float(x))
        # Emulate EXACT operation order of core.recurrence:
        # fl(fl(cp + x) - k); fl(fl(cm - x) - k)
        cp = max(0.0, round_dyadic(F.from_float(round_dyadic(F.from_float(cp)+fx))-fk))
        cm = max(0.0, round_dyadic(F.from_float(round_dyadic(F.from_float(cm)-fx))-fk))
        actual = max(cp,cm)
        rounded_rec.append(actual)
        # Emulate EXACT operation order of np.cumsum(sign*u-k) and P-min(P).
        ip, im = round_dyadic(fx-fk), round_dyadic(-fx-fk)
        pp = round_dyadic(F.from_float(pp)+F.from_float(ip))
        pm = round_dyadic(F.from_float(pm)+F.from_float(im))
        pmin, mmin = min(pmin,pp), min(mmin,pm)
        alternative = max(round_dyadic(F.from_float(pp)-F.from_float(pmin)),
                          round_dyadic(F.from_float(pm)-F.from_float(mmin)))
        rounded_prefix.append(alternative)
        # True rational CUSUM, no intermediate rounding: verify algebraic identity.
        rcp, rcm = max(F(0),rcp+fx-fk), max(F(0),rcm-fx-fk)
        ep, em = ep+fx-fk, em-fx-fk
        emin_p, emin_m = min(emin_p,ep), min(emin_m,em)
        exact_c = max(rcp,rcm)
        if exact_c != max(ep-emin_p,em-emin_m):
            raise ValueError("Exact rational recurrence/prefix identity failed")
        if t in selected:
            records.append({
                "sample":t+1,
                "recurrence":float(target[t]),"prefix":float(prefix[t]),"threshold":float(h),
                "recurrence_hex":float(target[t]).hex(),"prefix_hex":float(prefix[t]).hex(),
                "threshold_hex":float(h).hex(),
                "recurrence_gt_h":bool(target[t]>h),"prefix_gt_h":bool(prefix[t]>h),
                "exact_real_cusum_gt_h":bool(exact_c>fh),
                "exact_real_cusum_numerator":str(exact_c.numerator),
                "exact_real_cusum_denominator":str(exact_c.denominator),
                "recurrence_error_exact":str(F.from_float(float(target[t]))-exact_c),
                "prefix_error_exact":str(F.from_float(float(prefix[t]))-exact_c),
                "absolute_formula_difference":abs(float(target[t])-float(prefix[t]))
            })
    if not same_values(np.asarray(rounded_rec),target):
        bad=np.flatnonzero(np.asarray(rounded_rec)!=target)
        raise ValueError("DEPLOYED recurrence differs from exact operation emulator at samples "+
                         repr((bad[:10]+1).tolist()))
    if not same_values(np.asarray(rounded_prefix),prefix):
        bad=np.flatnonzero(np.asarray(rounded_prefix)!=prefix)
        raise ValueError("PREFIX differs from exact operation emulator at samples "+
                         repr((bad[:10]+1).tolist()))
    return {"status":"EXACT_OPERATION_CERTIFIED_PRIMARY_UNCHANGED",
            "trace_samples_verified":len(seq),"records":records,
            "mathematical_identity_exact":True,"both_binary64_traces_exactly_reproduced":True,
            "threshold_changed":False,"deployed_decisions_changed":False}

def audit_prefix(u,target,alternative,h,*,k=.05,out=None,label="family",run_ids=None):
    """Retain the v1.0 closeness requirement and certify—not ignore—bit disagreements."""
    u=np.asarray(u,dtype=np.float64)
    target=np.asarray(target,dtype=np.float64)
    alternative=np.asarray(alternative,dtype=np.float64)
    h=np.asarray(h,dtype=np.float64)
    if u.ndim!=3 or target.shape!=u.shape or alternative.shape!=u.shape or h.shape!=(u.shape[2],):
        raise ValueError("Prefix audit shape mismatch")
    if not (np.isfinite(u).all() and np.isfinite(target).all() and
            np.isfinite(alternative).all() and np.isfinite(h).all()):
        raise ValueError("Nonfinite prefix-audit value")
    close=np.isclose(alternative,target,rtol=1e-9,atol=1e-8) # UNCHANGED v1.0 numeric criterion
    difference=(alternative>h)!=(target>h)
    numeric_bad=~close
    affected=np.any(difference|numeric_bad,axis=1)
    count=int(difference.sum())
    report={"audit_version":"1.0.1","family":label,"total_decisions":int(u.size),
            "strict_prefix_agreements":int(u.size-count),"decision_disagreements":count,
            "exactly_certified_disagreements":0,"exactly_replayed_traces":0,
            "prefix_max_error":float(np.max(np.abs(alternative-target))),
            "original_numeric_close_criterion_unchanged":True,
            "primary_operation_order_unchanged":True,"entries":[]}
    dest=None if out is None else Path(out)
    if dest:dest.mkdir(parents=True,exist_ok=True)
    def save():
        if dest:(dest/(label+"_AUDIT.json")).write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    for ir,ia in np.argwhere(affected):
        ir,ia=int(ir),int(ia)
        rid=int(run_ids[ir]) if run_ids is not None else ir
        entry={"run_id":rid,"run_position":ir,"axis":ia,
               "disagreement_samples":(np.flatnonzero(difference[ir,:,ia])+1).tolist()}
        report["entries"].append(entry)
        if dest:
            fname=f"{label}_run{rid:03}_axis{ia:02}.npz"
            np.savez_compressed(dest/fname,input=u[ir,:,ia],recurrence=target[ir,:,ia],
                  prefix=alternative[ir,:,ia],threshold=np.array(h[ia]),k=np.array(k))
            entry["trace_file"]=fname
            entry["trace_sha256"]=hashlib.sha256((dest/fname).read_bytes()).hexdigest()
        save() # retain evidence even if exact adjudication subsequently rejects
        if numeric_bad[ir,:,ia].any():
            report["status"]="FAIL_ORIGINAL_NUMERIC_CLOSE_REQUIREMENT"
            save()
            raise ValueError("Prefix formula discrepancy; trace saved, no tolerance relaxed")
        try:
            proof=exact_trace_audit(u[ir,:,ia],target[ir,:,ia],alternative[ir,:,ia],
                                   float(h[ia]),float(k),np.flatnonzero(difference[ir,:,ia]))
        except Exception as exc:
            report["status"]="FAIL_EXACT_OPERATION_CERTIFICATION"
            entry["error"]=type(exc).__name__+": "+str(exc)
            save()
            raise ValueError("Unexplained prefix disagreement; exact replay failed: "+str(exc)) from exc
        entry["certificate"]=proof
        report["exactly_replayed_traces"]+=1
        report["exactly_certified_disagreements"]+=len(entry["disagreement_samples"])
    if report["exactly_certified_disagreements"]!=count:
        raise ValueError("Not all divergent decisions certified")
    report["status"]="PASS_STRICT_EQUAL" if not count else "PASS_EXACT_OPERATION_CERTIFICATES"
    save()
    return report
