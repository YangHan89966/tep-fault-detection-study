from pathlib import Path
import csv, numpy as np, matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
with open(HERE/"P2-SIF03_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
dev=[r for r in R if r["record_type"]=="development_variable"]
summ={r["fault_id"]:r for r in R if r["record_type"]=="gate_summary"}
top={}
for fid in ("9","15"):
    top[fid]=sorted([r for r in dev if r["fault_id"]==fid and r["display_top5_abs_g"]=="True"],
                    key=lambda r:float(r["abs_hedges_g"]),reverse=True)
plt.rcParams.update({"font.size":7.8,"axes.titlesize":8.8,"axes.labelsize":7.8,
                     "xtick.labelsize":6.9,"ytick.labelsize":6.9})
fig=plt.figure(figsize=(178/25.4,134/25.4))
fig.text(.07,.965,"Signed-level applicability failure for Faults 9 and 15",
         fontsize=9.6,fontweight="bold",ha="left",va="top")
axes=[fig.add_axes([.09,.58,.37,.28]),fig.add_axes([.55,.58,.37,.28])]
for ax,fid,panel in zip(axes,("9","15"),("A","B")):
    rr=list(reversed(top[fid])); y=np.arange(len(rr))
    g=np.array([float(r["hedges_g"]) for r in rr])
    lo=np.array([float(r["bootstrap_ci_low_g"]) for r in rr])
    hi=np.array([float(r["bootstrap_ci_high_g"]) for r in rr])
    ax.errorbar(g,y,xerr=np.vstack([g-lo,hi-g]),fmt="o",capsize=3,linewidth=1.15)
    ax.axvline(0,linestyle="--",linewidth=1.0)
    ax.set_yticks(y,[r["variable"] for r in rr]); ax.set_xlim(-.26,.27)
    ax.set_xlabel("Standardized run-mean shift (Hedges' g)")
    ax.set_title(f"{panel}  Fault {fid} development screen\n0/52 direction-stable signed residuals",
                 loc="left",fontweight="bold")
    ax.grid(True,axis="x",alpha=.20)
fig.text(.09,.515,"A-B show the five largest |g| coordinates for readability; all displayed 95% intervals cross 0.",
         fontsize=6.2,ha="left",va="top")
fig.text(.09,.492,"The full development screen yielded 0/52 direction-stable signed residuals for each fault.",
         fontsize=6.2,ha="left",va="top")
ax=fig.add_axes([.12,.18,.76,.21]); x=np.arange(2); width=.28
dev_counts=[int(summ["9"]["development_stable_variable_count"]),int(summ["15"]["development_stable_variable_count"])]
comp_counts=[int(summ["9"]["companion_stable_variable_count"]),int(summ["15"]["companion_stable_variable_count"])]
b1=ax.bar(x-width/2,dev_counts,width=width)
b2=ax.bar(x+width/2,comp_counts,width=width,hatch="//",alpha=.65)
ax.axhline(5,linestyle="--",linewidth=1.15)
ax.text(.02,5.18,"5 residuals required for a full Top-5 branch",
        transform=ax.get_yaxis_transform(),fontsize=6.3,ha="left",va="bottom")
ax.set_xticks(x,["Fault 9","Fault 15"]); ax.set_ylim(0,8.5)
ax.set_ylabel("Direction-stable signed residual count")
ax.set_title("C  Applicability-gate outcome",loc="left",fontweight="bold")
ax.grid(True,axis="y",alpha=.20)
ax.text(.02,.93,"development = prespecified screen",transform=ax.transAxes,fontsize=6.15,va="top")
ax.text(.53,.93,"hatched companion = post-freeze diagnostic only",transform=ax.transAxes,fontsize=6.15,va="top")
for rect,val in zip(b1,dev_counts):
    ax.text(rect.get_x()+rect.get_width()/2,.15,str(val),ha="center",va="bottom",fontsize=7.1,fontweight="bold")
for rect,val in zip(b2,comp_counts):
    ax.text(rect.get_x()+rect.get_width()/2,val+.12,str(val),ha="center",va="bottom",fontsize=7.1,fontweight="bold")
fig.text(.12,.105,"The prespecified signed-level gate abstained for both faults.",
         fontsize=6.25,ha="left",va="bottom")
fig.text(.12,.080,"Companion counts (7 for Fault 9; 6 for Fault 15) were inspected only after the freeze.",
         fontsize=6.25,ha="left",va="bottom")
fig.text(.12,.055,"They were not used to rescue, tune, or redefine the branch; abstention is a valid applicability outcome, not a code failure.",
         fontsize=6.25,ha="left",va="bottom")
fig.savefig(HERE/"P2-SIF03_signed_level_applicability_failure.pdf")
fig.savefig(HERE/"P2-SIF03_signed_level_applicability_failure.svg")
fig.savefig(HERE/"P2-SIF03_signed_level_applicability_failure.png",dpi=600)
plt.close(fig)
