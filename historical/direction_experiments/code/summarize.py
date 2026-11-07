from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from core import js
def summarize_run(run,out=None):
    run=Path(run);out=Path(out) if out is not None else run/"reports";out.mkdir(parents=True,exist_ok=False)
    a=pd.read_csv(run/"MAIN_RESULTS.csv")
    a["success_percent"]=a["clean_detected_runs"]/a["audit_fault_n"]*100
    rows=[]
    for (f,v,m),g in a.groupby(["fault","variant","method"],sort=True):
        row={"fault":int(f),"variant":v,"method":m,"splits":len(g)}
        for c in ["event_loss_min","success_percent","normal_FAR_percent","occupancy_percent","prefault_alarm_runs"]:
            row[c+"_median"]=float(g[c].median());row[c+"_min"]=float(g[c].min());row[c+"_max"]=float(g[c].max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"ALL_METHODS_SUMMARY.csv",index=False)
    pairs=[("moment_pool_squared","mean_pool_squared"),
           ("moment_pool_signed","mean_pool_signed"),
           ("moment_pool_absolute","mean_pool_absolute"),
           ("moment_pool_three","mean_pool_three"),
           ("combined_three_or_Q","mean_pool_three_or_Q")]
    allpairs=[];summary=[]
    for (f,v),g in a.groupby(["fault","variant"]):
        for new,old in pairs:
            gg=g[g.method==new].merge(g[g.method==old],on="split_id",suffixes=("_new","_old"),validate="one_to_one")
            values=[]
            for _,r in gg.iterrows():
                delta=float(r.event_loss_min_new-r.event_loss_min_old);values.append(delta)
                allpairs.append({"fault":int(f),"variant":v,"split":r.split_id,"new":new,"old":old,
                    "loss_difference_min":delta,"success_difference_pp":float(r.success_percent_new-r.success_percent_old),
                    "normal_FAR_difference_pp":float(r.normal_FAR_percent_new-r.normal_FAR_percent_old)})
            summary.append({"fault":int(f),"variant":v,"new":new,"old":old,
                "paired_difference_median":float(np.median(values)),"paired_difference_min":float(min(values)),
                "paired_difference_max":float(max(values)),"earlier":int(sum(x<0 for x in values)),
                "same":int(sum(x==0 for x in values)),"later":int(sum(x>0 for x in values))})
    pd.DataFrame(allpairs).to_csv(out/"ALL_PAIRED_DIFFERENCES.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"PAIRED_SUMMARY.csv",index=False)
    primary=[r for r in rows if r["method"] in ["mean_pool_signed","mean_pool_squared","moment_pool_squared"]]
    pd.DataFrame(primary).to_csv(out/"PRIMARY_EQUAL_BUDGET.csv",index=False)
    choices=[];eigens=[];events=[];gains=[]
    for b in sorted((run/"models").glob("F*/D*/*")):
        c=json.loads((b/"CHOICES.json").read_text());km=c["mean_count"];k=km+c["moment_count"]
        for method,j in c["choices"].items():
            choices.append({"fault":c["fault"],"split":c["split_id"],"variant":c["variant"],"method":method,
               "candidate":c["candidate_names"][j],"source":"Q" if j==3*k else "mean" if j%k<km else "second_moment",
               "family":"Q" if j==3*k else ("signed","absolute","squared")[j//k],
               "mean_count":km,"moment_count":c["moment_count"]})
        for _,r in pd.read_csv(b/"EIGEN_DIAGNOSTICS.csv").iterrows():
            eigens.append({"fault":c["fault"],"split":c["split_id"],"variant":c["variant"],**r.to_dict()})
        for _,r in pd.read_csv(b/"INPUT_GAINS.csv").iterrows():
            gains.append({"fault":c["fault"],"split":c["split_id"],"variant":c["variant"],**r.to_dict()})
        ev=pd.read_csv(b/"EVENT_PHASES.csv")
        for (meth,cat),gg in ev.groupby(["method","category"]):
            events.append({"fault":c["fault"],"split":c["split_id"],"variant":c["variant"],
                  "method":meth,"category":cat,"n":len(gg)})
    pd.DataFrame(choices).to_csv(out/"ALL_CHOICES.csv",index=False)
    pd.DataFrame(eigens).to_csv(out/"ALL_EIGEN_DIAGNOSTICS.csv",index=False)
    pd.DataFrame(gains).to_csv(out/"ALL_GAINS.csv",index=False)
    pd.DataFrame(events).to_csv(out/"EVENT_COUNTS.csv",index=False)
    return len(a)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="Rebuild eight saved-result summaries into a new output directory")
    p.add_argument("root",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):raise ValueError("Choose output outside the archive")
    print("SUMMARIZED",summarize_run(a.root,a.out))
