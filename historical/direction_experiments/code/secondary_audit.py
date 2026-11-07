"""Second implementation of arithmetic/result checks, not independent scientific review."""
from pathlib import Path
import sys,json,csv,math
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from core import sha
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8-sig")))
def check(x,msg):
    if not x:raise AssertionError(msg)
def main(run,out=None):
    run=Path(run);root=Path(__file__).resolve().parents[1]
    destination=Path(out) if out is not None else run/"SECONDARY_AUDIT.json"
    if destination.exists():raise FileExistsError("Choose a new check output file")
    destination.parent.mkdir(parents=True,exist_ok=True)
    df=json.loads((root/"STUDY_DEFINITION.json").read_text())
    roles=0;bits=0;selected=0;ranks=0;events=0;unit=0;maxerr=0.;matrix_checks=0;poison=0
    for sp in df["splits"].values():
        for typ in ["normal","fault"]:
            ss=[set(v) for v in sp[typ].values()]
            check(set.union(*ss)==set(range(1,251)),"run coverage")
            check(sum(map(len,ss))==250,"role overlap")
            roles+=1
    fd_cache={}
    for b in sorted((run/"models").glob("F*/D*/*")):
        ch=json.loads((b/"CHOICES.json").read_text());f=ch["fault"];tag=ch["split_id"];var=ch["variant"]
        m=load(b/"MODEL.npz");p=load(b/"TRANSFORM_STATE.npz");o=load(b/"OUTER_RECORDS.npz")
        ir=load(b/"INNER_RECORDS.npz");rr=load(b/"EVALUATION_RECORDS.npz");mk=load(b/"SELECTED_MASKS.npz")
        des=load(root/"inputs/normal_design"/tag/(var+".npz"))
        km,ka=ch["mean_count"],ch["moment_count"];k=km+ka
        evfield=list(map(str,ir["fields"]));risk=ir["selection"][:,:,evfield.index("event_loss_samples")].sum(1)/75
        old={fam:[g*k+j for j in range(km)] for g,fam in enumerate(["signed","absolute","squared"])}
        new={fam:[g*k+j for j in range(km,k)] for g,fam in enumerate(["signed","absolute","squared"])}
        def best(ids):
            low=min(float(risk[j]) for j in ids)
            return next(j for j in ids if float(risk[j])<=low+1e-12)
        expected={"fixed_mean_signed":ch["original_axis"],"fixed_mean_squared":2*k+ch["original_axis"]}
        for fam in old:expected["mean_pool_"+fam]=best(old[fam])
        for fam in new:expected["moment_pool_"+fam]=best(new[fam])
        allold=sum(old.values(),[]);allnew=sum(new.values(),[])
        expected.update(mean_pool_three=best(allold),moment_pool_three=best(allnew),
                        mean_pool_three_or_Q=best(allold+[3*k]),combined_three=best(allold+allnew),
                        combined_three_or_Q=best(allold+allnew+[3*k]),Q_CUSUM=3*k)
        check(expected==ch["choices"],"Selection identity")
        selected+=len(expected)
        for j in range(3*k+1):
            ar=sorted(float(x) for x in ir["anchor_peaks"][:,j]);position=.9*(len(ar)-1)
            lo=int(position);frac=position-lo;anchor=ar[lo]+frac*(ar[lo+1]-ar[lo])
            check(math.isclose(anchor,float(p["anchors"][j]),rel_tol=1e-14,abs_tol=1e-12),"Anchor interpolation")
            for peaks,h in [(ir["inner_peaks"],p["inner_thresholds"]),(o["peaks"],o["thresholds"])]:
                q=sorted(float(x)/float(p["anchors"][j]) for x in peaks[:,j])[56]
                hh=q*float(p["anchors"][j])
                check(hh==float(h[j]),"Rank threshold");ranks+=1
        F=m["fault_second_moment"];V=m["fault_within_window_covariance"];J=m["fault_endpoint_mean_outer"]
        check(np.allclose(F,V+J,rtol=1e-10,atol=1e-9),"Moment sum")
        C0=des["G"]*((40*960-1)/(40*960))
        check(np.allclose(C0,m["normal_second_moment"],rtol=1e-8,atol=1e-9),"Normal moment vs covariance")
        a=m["moment_axes"]
        check(np.allclose(np.einsum("ai,ij,aj->a",a,des["G"],a),1,atol=1e-9),"Unit normal variance")
        unit+=len(a)
        wins=rows(b/"FIT_WINDOWS.csv")
        check([int(w["run_id"]) for w in wins]==df["splits"][tag]["fault"]["fit"],"FIT run IDs")
        if (run/"work"/f"fault{f}.npz").exists():
            if f not in fd_cache:
                fd_cache.clear();fd_cache[f]=load(run/"work"/f"fault{f}.npz")
            zz=fd_cache[f]["z"]
            MM=[]
            for row in wins:
                rid=int(row["run_id"]);lo=int(row["early_first_sample"])-1;hi=int(row["early_last_sample"])
                lo2=int(row["late_first_sample"])-1;hi2=int(row["late_last_sample"])
                if int(row["record_suffix_samples"])>=160:
                    check(hi<int(row["record_suffix_start"]) and hi2<int(row["record_suffix_start"]),"Postmarker FIT leak")
                y1=(zz[rid-1,lo:hi]-des["centre"])@des["whitener"]
                y2=(zz[rid-1,lo2:hi2]-des["centre"])@des["whitener"]
                MM.append((np.einsum("ti,tj->ij",y1,y1)/len(y1)+np.einsum("ti,tj->ij",y2,y2)/len(y2))/2)
            err=float(np.max(np.abs(np.mean(MM,0)-F)));maxerr=max(maxerr,err)
            check(np.allclose(np.mean(MM,0),F,rtol=1e-9,atol=1e-8),"Independent moment values")
            matrix_checks+=1
        ms=rows(b/"MAIN_RESULTS.csv");main_by={x["method"]:x for x in ms}
        ff=list(map(str,rr["fault_fields"]));nf=list(map(str,rr["normal_fields"]))
        for q,method in enumerate(map(str,mk["methods"])):
            j=expected[method]
            fmask=np.unpackbits(mk["fault"][q],axis=1)[:,:960].astype(bool)
            nmask=np.unpackbits(mk["normal"][q],axis=1)[:,:960].astype(bool)
            bits+=fmask.size+nmask.size
            pre=fmask[:,:160].any(1);post=fmask[:,160:];det=post.any(1)
            first=np.where(det,post.argmax(1),800);loss=np.where(pre|~det,800,first)
            clean=(~pre)&det
            for field,val in [("prefault_alarm",pre.astype(int)),("clean_detection",clean.astype(int)),
                              ("event_loss_samples",loss),("detected",det.astype(int)),
                              ("occupancy_points",post.sum(1))]:
                check(np.array_equal(rr["fault"][j,:,ff.index(field)],val),"Mask fault metric "+field)
            check(np.array_equal(rr["normal"][j,:,nf.index("alarm")],nmask.any(1).astype(int)),"Normal mask")
            r=main_by[method]
            check(int(r["clean_detected_runs"])==int(clean.sum()),"Success summary")
            check(int(r["prefault_alarm_runs"])==int(pre.sum()),"Pre summary")
            check(int(r["normal_alarm_runs"])==int(nmask.any(1).sum()),"Normal summary")
            check(math.isclose(float(r["event_loss_min"]),float(loss.sum())/75*3,abs_tol=1e-10),"Loss summary")
            check(np.array_equal(rr["fault_run_id"],df["splits"][tag]["fault"]["audit"]),"Evaluation IDs")
        for r in rows(b/"EVENT_PHASES.csv"):
            if int(r["prefault_alarm"]):cat="prefault_alarm"
            elif not int(r["detected"]):cat="marked_not_detected" if int(r["tail_samples"])>=160 else "unmarked_not_detected"
            elif int(r["tail_samples"])<160:cat="unmarked_detected"
            else:cat="before_marker" if int(r["first_delay_samples_penalized"])+161<int(r["record_marker_start"]) else "at_or_after_marker"
            check(cat==r["category"],"Event category");events+=1
    out={"status":"PASS_AUTHOR_SECOND_IMPLEMENTATION_CHECKS_NOT_INDEPENDENT_SCIENCE",
      "role_partitions_verified":roles,"selected_systems":selected,"normal_ranks":ranks,
      "saved_mask_bits_recounted":bits,"event_rows_classified":events,"new_axis_unit_variances":unit,
      "independently_recomputed_fit_matrices":matrix_checks,"fit_matrix_max_absolute_difference":maxerr,
      "all20_actual_run":json.loads((run/"COMPLETE.json").read_text())["all20_complete"]}
    destination.write_text(json.dumps(out,indent=2))
    print(out)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError("Choose output outside the archive")
    main(a.root,a.out)
