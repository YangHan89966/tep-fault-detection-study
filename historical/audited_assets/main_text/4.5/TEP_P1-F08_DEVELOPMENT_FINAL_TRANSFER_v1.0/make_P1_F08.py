from pathlib import Path
import csv, numpy as np, matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
with open(HERE/"P1-F08_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
for r in R:
    r["fault_id"]=int(r["fault_id"])
    for k in ["development_occupancy_pct","final_occupancy_pct","abs_occupancy_change_pp",
              "abs_median_delay_change_min"]:
        r[k]=float(r[k])
mean_abs=float(np.mean([r["abs_occupancy_change_pp"] for r in R]))
med_delay=float(np.median([r["abs_median_delay_change_min"] for r in R]))
plt.rcParams.update({"font.size":8.2,"axes.titlesize":9.5,"axes.labelsize":8.5,
                     "xtick.labelsize":7.4,"ytick.labelsize":7.4,"legend.fontsize":7.0})
fig=plt.figure(figsize=(178/25.4,90/25.4)); ax=fig.add_axes([.08,.17,.58,.72])
lo,hi=59.0,100.5
ax.plot([lo,hi],[lo,hi],linestyle="--",linewidth=1.2,label="Identity")
markers={"signed_level":"o","absolute_deviation":"s","lag1_product":"^"}
labels={"signed_level":"Signed level","absolute_deviation":"Absolute deviation","lag1_product":"Lag-1 product"}
for fam in ["signed_level","absolute_deviation","lag1_product"]:
    rr=[r for r in R if r["evidence_family"]==fam]
    ax.scatter([r["development_occupancy_pct"] for r in rr],[r["final_occupancy_pct"] for r in rr],
               marker=markers[fam],s=48,label=labels[fam],zorder=3)
offs={8:(6,5),10:(6,-13),18:(7,5),20:(-25,-13)}
for r in R:
    if r["fault_id"] in offs:
        ax.annotate(f'Fault {r["fault_id"]}',(r["development_occupancy_pct"],r["final_occupancy_pct"]),
                    xytext=offs[r["fault_id"]],textcoords="offset points",fontsize=6.8)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal",adjustable="box")
ax.set_xlabel("Development runs 1–250 threshold occupancy (%)")
ax.set_ylabel("Frozen-final runs 251–500 threshold occupancy (%)")
ax.set_title("Frozen Proposed Top-5 development-to-final transfer",loc="left",fontweight="bold")
ax.grid(True,alpha=.22)
fig.text(.71,.83,"Internal transfer check",fontsize=8.7,fontweight="bold",ha="left")
fig.text(.71,.73,f"Mean absolute occupancy change\n{mean_abs:.3f} percentage points\n\n"
        f"Median absolute median-delay change\n{med_delay:.0f} min\n\n"
        "Run detection\n250/250 in both halves for all 17 faults",
        fontsize=7.0,ha="left",va="top",linespacing=1.25)
handles,leglabels=ax.get_legend_handles_labels()
fig.legend(handles,leglabels,loc="center left",bbox_to_anchor=(.70,.38),frameon=False)
fig.text(.71,.13,"All family, coordinate,\norientation, scaling, k,\nand limit states were frozen\nbefore final evaluation.",
        fontsize=6.6,ha="left",va="bottom",linespacing=1.15)
fig.savefig(HERE/"P1-F08_development_to_final_transfer.pdf")
fig.savefig(HERE/"P1-F08_development_to_final_transfer.svg")
fig.savefig(HERE/"P1-F08_development_to_final_transfer.png",dpi=600)
plt.close(fig)
