
"""V1.5-D1: fixed-boundary accumulator-state diagnostic; no learning or production execution."""
from pathlib import Path
import csv, io, json, hashlib, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=None
FAMILIES=("signed","absolute","squared")

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load(z,n):
    with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}

def paths(v,reset=False):
    v=np.asarray(v,dtype=np.float64)
    cp=np.zeros(v.shape[0]);cm=cp.copy();out=np.zeros_like(v)
    for t in range(v.shape[1]):
        if reset and t==160:cp.fill(0);cm.fill(0)
        cp=np.maximum(0.0,(cp+v[:,t])-.05)
        cm=np.maximum(0.0,(cm-v[:,t])-.05)
        out[:,t]=np.maximum(cp,cm)
    return out

def scalar_paths(v,reset=False):
    result=np.empty_like(v)
    for i,row in enumerate(v):
        p=m=0.
        for t,x in enumerate(row):
            if reset and t==160:p=m=0.
            p=max(0.,(p+float(x))-.05);m=max(0.,(m-float(x))-.05)
            result[i,t]=max(p,m)
    return result

def fmetrics(stat,h,ids):
    mask=stat>h;pre=mask[:,:160].any(1);det=mask[:,160:].any(1)
    delay=np.where(det,mask[:,160:].argmax(1),800)
    clean=(~pre)&det;loss=np.where(clean,delay,800)
    return mask,dict(run_id=ids,prefault_alarm=pre.astype(int),detected=det.astype(int),
                     clean_detection=clean.astype(int),first_delay_samples_penalized=delay,
                     event_loss_samples=loss,occupancy_points=mask[:,160:].sum(1))
def nmetrics(stat,h,ids):
    mask=stat>h
    return mask,dict(run_id=ids,alarm=mask.any(1).astype(int),above_points=mask.sum(1),
                    first_alarm_sample=np.where(mask.any(1),mask.argmax(1),-1))
def main():
    global SOURCE
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    SOURCE=args.source
    contract=json.loads((ROOT/"audit/DIAGNOSTIC_CONTRACT.json").read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==contract["source"]["sha256"]
    rows=[];runrows=[];checked=0;scalar_checked=0
    with zipfile.ZipFile(SOURCE) as z:
        for split in contract["splits"]:
            role=json.loads(z.read(f"scores/{split}/AXIS_ROLES.json"))
            tr=load(z,f"scores/{split}/TRANSFORM_AND_THRESHOLDS.npz")
            orig=load(z,f"inputs/{split}/EVALUATION_RECORDS.npz")
            ts=load(z,f"inputs/{split}/TRANSFORM_STATE.npz")
            ai=role["axis_indices"].index(role["new_signed_axis"])
            paths0={s:load(z,f"scores/{split}/{s}.npz") for s in 
                    ("normal_outer_calibration","normal_audit","fault_audit")}
            for j,fam in enumerate(FAMILIES):
                ci=role["candidate_indices_family_major"][j*len(role["axis_indices"])+ai]
                h0=float(tr["thresholds"][j,ai]);anchor=float(ts["anchors"][ci])
                def transform(u):
                    x=u[:,:,ai]
                    if fam=="signed":return x
                    if fam=="absolute":return (np.abs(x)-tr["abs_mean"][ai])/tr["abs_sd"][ai]
                    return (x*x-tr["sq_mean"][ai])/tr["sq_sd"][ai]
                values={s:transform(a["u"]) for s,a in paths0.items()}
                for reset in (False,True):
                    stats={s:paths(a,reset) for s,a in values.items()}
                    # Separate scalar operation implementation, all rows on fault paths.
                    assert np.array_equal(stats["fault_audit"],scalar_paths(values["fault_audit"],reset))
                    scalar_checked+=stats["fault_audit"].size
                    peaks=stats["normal_outer_calibration"].max(1)
                    h1=float(np.sort(peaks/anchor)[56]*anchor)
                    if not reset:assert h1==h0
                    for route,h in [("original_threshold",h0),("matched_reset_calibration",h1)]:
                        fm,fd=fmetrics(stats["fault_audit"],h,paths0["fault_audit"]["run_id"])
                        nm,nd=nmetrics(stats["normal_audit"],h,paths0["normal_audit"]["run_id"])
                        if not reset:
                            for key,vec in fd.items():
                                assert np.array_equal(vec,orig["fault"][ci,:,list(orig["fault_fields"]).index(key)]),key
                                checked+=len(vec)
                            # Normal first_alarm_sample source uses zero based -1 as checked below.
                            for key in ("run_id","alarm","above_points"):
                                assert np.array_equal(nd[key],orig["normal"][ci,:,list(orig["normal_fields"]).index(key)]),key
                                checked+=len(nd[key])
                        record=dict(split=split,family=fam,reset=reset,route=route,h=h,
                            threshold_ratio=h/h0,fault_n=len(fd["run_id"]),normal_n=len(nd["run_id"]),
                            success_percent=100*fd["clean_detection"].mean(),
                            event_loss_min=3*fd["event_loss_samples"].mean(),
                            prefault_percent=100*fd["prefault_alarm"].mean(),
                            normal_far_percent=100*nd["alarm"].mean(),
                            detection_by480min_percent=100*np.mean((fd["clean_detection"]==1)&(fd["first_delay_samples_penalized"]<=160)))
                        rows.append(record)
                        for i,rid in enumerate(fd["run_id"]):
                            runrows.append(dict(split=split,family=fam,reset=reset,route=route,run_id=int(rid),
                                prefault=int(fd["prefault_alarm"][i]),success=int(fd["clean_detection"][i]),
                                loss_min=int(fd["event_loss_samples"][i]*3)))
    out=ROOT/"Supporting_Data/D1_boundary_reset"
    write_csv(out/"PER_SPLIT.csv",rows);write_csv(out/"PER_RUN.csv",runrows)
    summary=[]
    for fam in FAMILIES:
        for reset in (False,True):
            for route in ("original_threshold","matched_reset_calibration"):
                ss=[r for r in rows if r["family"]==fam and r["reset"]==reset and r["route"]==route]
                res=dict(family=fam,reset=reset,route=route)
                for key in ("success_percent","event_loss_min","normal_far_percent","prefault_percent","h","threshold_ratio","detection_by480min_percent"):
                    a=[r[key] for r in ss]
                    res.update({key+"_median":float(np.median(a)),key+"_min":float(min(a)),key+"_max":float(max(a))})
                summary.append(res)
    write_csv(out/"SUMMARY.csv",summary)
    diffs=[]
    for fam in FAMILIES:
        for route in ("original_threshold","matched_reset_calibration"):
            ds=[]
            for split in contract["splits"]:
                old=next(r for r in rows if r["family"]==fam and r["split"]==split and not r["reset"] and r["route"]==route)
                new=next(r for r in rows if r["family"]==fam and r["split"]==split and r["reset"] and r["route"]==route)
                ds.append(new["event_loss_min"]-old["event_loss_min"])
            diffs.append(dict(family=fam,route=route,reset_minus_original_median=float(np.median(ds)),
                              min=min(ds),max=max(ds),earlier=sum(d<0 for d in ds),tied=sum(d==0 for d in ds),later=sum(d>0 for d in ds)))
    write_csv(out/"PAIRED.csv",diffs)
    (ROOT/"audit/D1_AUDIT.json").write_text(json.dumps(dict(status="COMPLETED_EXPLORATORY_DIAGNOSTIC_NOT_CONFIRMATION",
        inherited_fields_checked=checked,scalar_path_values_checked=scalar_checked,configurations=len(rows),
        contract_sha256=hashlib.sha256((ROOT/"audit/DIAGNOSTIC_CONTRACT.json").read_bytes()).hexdigest(),
        source_sha256=contract["source"]["sha256"],new_training_or_production_execution=False),indent=2))
if __name__=="__main__":main()
