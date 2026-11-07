"""Fail-closed audit repair tests. Synthetic, not TEP performance confirmation."""
from pathlib import Path
import sys,json,tempfile,math,struct
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from fractions import Fraction as F
from numeric_audit import audit_prefix,round_dyadic,exact_trace_audit
from core import recurrence,prefix_cusum
from legacy_selftest import main as original_tests

def run(output):
    output=Path(output);tests=[]
    def test(name,fn):
        fn();tests.append({'test':name,'passed':True})
    def ensure(v,message):
        if not v:raise AssertionError(message)
    def reject(fn):
        try:fn()
        except (ValueError,OverflowError):return
        raise AssertionError('Invalid input/result accepted')
    def rounding():
        # Exact representation plus nearest-even, including normal/subnormal ties.
        for value in [0.,.05,.1,1.,-1.,math.ldexp(1.,-1074),math.ldexp(1.,-1022),
                      math.nextafter(1.,math.inf),sys.float_info.max]:
            ensure(round_dyadic(F.from_float(value))==value,'roundtrip')
        tiny=F(1,1<<1074)
        ensure(round_dyadic(tiny/2)==0.,'subnormal even tie')
        ensure(round_dyadic(3*tiny/2)==math.ldexp(1.,-1073),'subnormal odd tie')
        one=F(1);eps=F(1,1<<52)
        ensure(round_dyadic(one+eps/2)==1.,'normal even tie')
        ensure(round_dyadic(one+3*eps/2)==1+2**-51,'normal odd tie')
        rng=np.random.default_rng(260601)
        for _ in range(4000):
            x=math.ldexp(float(rng.uniform(-1.,1.)),int(rng.integers(-1000,1000)))
            y=math.ldexp(float(rng.uniform(-1.,1.)),int(rng.integers(-1000,1000)))
            for v,ref in [(F.from_float(x)+F.from_float(y),x+y),
                          (F.from_float(x)-F.from_float(y),x-y)]:
                ensure(round_dyadic(v)==ref,'operation')
    test('dyadic_rounding_8000_add_sub_operations_and_ties',rounding)
    u=np.array([[[.1],[.1]]]);a=recurrence(u);b=prefix_cusum(u);h=np.array([.1])
    def mismatch():
        ensure(not np.array_equal(a>h,b>h),'counterexample missing')
        acopy=a.copy();hcopy=h.copy()
        r=audit_prefix(u,a,b,h)
        ensure(r['exactly_certified_disagreements']==1,'certificate')
        ensure(np.array_equal(a,acopy) and np.array_equal(h,hcopy),'audit modified production')
    test('genuine_rounding_disagreement_certified_without_output_change',mismatch)
    test('opposite_signed_axis_same_certificate',lambda:ensure(
        audit_prefix(-u,recurrence(-u),prefix_cusum(-u),h)['exactly_certified_disagreements']==1,'negative'))
    rng=np.random.default_rng(260602)
    sample=rng.normal(size=(4,960,3))
    at=recurrence(sample);bt=prefix_cusum(sample)
    test('ordinary_full_horizon_strict_comparison',lambda:ensure(
        audit_prefix(sample,at,bt,np.array([1e6]*3))['decision_disagreements']==0,'strict'))
    def long_boundary():
        x=sample[:1,:,:1]
        aa=recurrence(x);bb=prefix_cusum(x)
        idx=np.flatnonzero((aa!=bb).ravel())
        ensure(len(idx)>0,'long mismatch available')
        hh=np.array([float(min(aa[0,idx[-1],0],bb[0,idx[-1],0]))])
        r=audit_prefix(x,aa,bb,hh)
        ensure(r['exact_operation_certified_decisions']>0 if 'exact_operation_certified_decisions' in r
            else r['exactly_certified_disagreements']>0,'full length certificate')
    test('full960_trace_exact_adjudication',long_boundary)
    def corrupt_target():
        bad=a.copy();bad[0,1,0]=np.nextafter(a[0,1,0],np.inf)
        reject(lambda:audit_prefix(u,bad,b,h))
    test('one_ULP_deployed_bug_is_NOT_excused_as_roundoff',corrupt_target)
    def corrupt_prefix():
        bad=b.copy();bad[0,1,0]=np.nextafter(b[0,1,0],-np.inf)
        reject(lambda:audit_prefix(u,a,bad,h))
    test('one_ULP_prefix_bug_is_NOT_excused_as_roundoff',corrupt_prefix)
    def numericfail():
        bad=b.copy();bad[0,1,0]=10.
        reject(lambda:audit_prefix(u,a,bad,h))
    test('original_numeric_closeness_requirement_retained',numericfail)
    test('NaN_input_rejected',lambda:reject(lambda:audit_prefix(u*np.nan,a,b,h)))
    test('infinite_threshold_rejected',lambda:reject(lambda:audit_prefix(u,a,b,np.array([np.inf]))))
    test('shape_mismatch_rejected',lambda:reject(lambda:audit_prefix(u,a,b,np.array([.1,.2]))))
    def extremes():
        x=np.array([.1]*3+[1e10]*200+[-1e10]*200+[.06,-.06]*200+[0.]*157,dtype=float)[:960]
        uu=x[None,:,None];aa=recurrence(uu);bb=prefix_cusum(uu)
        rr=exact_trace_audit(x,aa[0,:,0],bb[0,:,0],1.,.05,np.array([],int))
        ensure(rr['both_binary64_traces_exactly_reproduced'],'large')
    test('large_dynamic_range_both_operation_orders_exact',extremes)
    def diagnostics():
        with tempfile.TemporaryDirectory() as tmp:
            rr=audit_prefix(u,a,b,h,out=Path(tmp),label='squared_two_sided',run_ids=[123])
            meta=json.loads((Path(tmp)/'squared_two_sided_AUDIT.json').read_text())
            ensure(meta['entries'][0]['run_id']==123,'runID')
            trace=Path(tmp)/meta['entries'][0]['trace_file']
            ensure(trace.is_file(),'saved full input')
            with np.load(trace,allow_pickle=False) as v:
                ensure(np.array_equal(v['input'],u[0,:,0]),'trace bytes')
    test('full_disagreement_trace_and_hex_threshold_saved',diagnostics)
    original_tests(output.parent/'ORIGINAL_SELFTEST.json')
    old=json.loads((output.parent/'ORIGINAL_SELFTEST.json').read_text())
    results={'status':'PASS_AUTHOR_IMPLEMENTATION_TESTS_NOT_SCIENTIFIC_ACCEPTANCE',
      'repair_tests':tests,'repair_count':len(tests),'unchanged_original_tests':old['count'],
      'synthetic_regression_not_new_S28_process_evidence':True}
    output.write_text(json.dumps(results,indent=2))
    print('REPAIR_SELFTEST_PASS',len(tests),'ORIGINAL',old['count'],flush=True)
    return results
if __name__=='__main__':run(sys.argv[1])
