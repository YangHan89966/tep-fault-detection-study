from pathlib import Path
import csv, numpy as np, matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def read(fn):
    with open(HERE/fn,encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
summary=read("P2-SIF05_EXACT_25_detector_bank_summary.csv")[0]
normal=read("P2-SIF05_EXACT_25_normal_monitor_operating_points.csv")
matrix=read("P2-SIF05_EXACT_25_crossfault_alarm_matrix.csv")
diag=read("P2-SIF05_EXACT_25_target_monitor_diagnostics_by_fault.csv")
faults=[1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20]
labels=[str(x) for x in faults]; idx={str(x):i for i,x in enumerate(faults)}
x=np.arange(17)
far=np.array([100*float(r["normal_run_level_far"]) for r in normal])
bank_far=100*float(summary["normal_bank_run_level_far"]); point_far=100*float(summary["normal_bank_point_level_far"])
M=np.zeros((17,17))
for r in matrix: M[idx[r["monitor_fault_id"]],idx[r["actual_fault_id"]]]=100*float(r["post_fault_run_detection_rate"])
off=np.array([M[i,j] for i in range(17) for j in range(17) if i!=j])
D={int(r["actual_fault_id"]):r for r in diag}
ear=np.array([100*float(D[f]["target_among_earliest_rate"]) for f in faults])
uni=np.array([100*float(D[f]["target_uniquely_earliest_rate"]) for f in faults])
rank=np.array([float(D[f]["median_target_earliest_rank"]) for f in faults])
plt.rcParams.update({"font.size":8.0,"axes.titlesize":9.2,"axes.labelsize":8.1,
                     "xtick.labelsize":7.0,"ytick.labelsize":7.0,"legend.fontsize":6.7})

fig=plt.figure(figsize=(178/25.4,80/25.4)); ax=fig.add_axes([.08,.22,.88,.66])
ax.bar(x,far); ax.axhline(bank_far,linestyle="--",linewidth=1.4,label=f"Hypothetical 17-monitor OR bank = {bank_far:.1f}%")
ax.set_xticks(x,labels); ax.set_xlabel("Frozen target-specific monitor identity"); ax.set_ylabel("Normal run-level FAR (%)"); ax.set_ylim(0,42)
ax.set_title("Hypothetical OR-bank normal-operation false-alarm audit",loc="left",fontweight="bold")
ax.grid(True,axis="y",alpha=.2); ax.legend(loc="upper right",frameon=False)
ax.text(.02,.82,f"Local monitors: mean {far.mean():.3f}% | range {far.min():.1f}-{far.max():.1f}%\n"
        f"OR bank: {summary['normal_bank_alarm_runs']}/250 normal runs alarmed | point FAR {point_far:.3f}%",
        transform=ax.transAxes,ha="left",va="top",fontsize=6.6)
fig.savefig(HERE/"P2-SIF05a_normal_or_bank_false_alarm_audit.pdf")
fig.savefig(HERE/"P2-SIF05a_normal_or_bank_false_alarm_audit.svg")
fig.savefig(HERE/"P2-SIF05a_normal_or_bank_false_alarm_audit.png",dpi=600)
plt.close(fig)

fig=plt.figure(figsize=(178/25.4,150/25.4)); ax=fig.add_axes([.10,.16,.78,.75])
im=ax.imshow(M,vmin=0,vmax=100,aspect="equal")
ax.set_xticks(np.arange(17),labels,rotation=45,ha="right"); ax.set_yticks(np.arange(17),labels)
ax.set_xlabel("Actual reserved TEP fault"); ax.set_ylabel("Frozen monitor identity")
ax.set_title("Full 17 x 17 cross-trigger audit: post-fault run detection",loc="left",fontweight="bold")
for i in range(17): ax.text(i,i,"T",ha="center",va="center",fontsize=6.2,fontweight="bold")
cax=fig.add_axes([.90,.16,.025,.75]); cb=fig.colorbar(im,cax=cax); cb.set_label("Monitor run detection / cross-trigger rate (%)")
fig.text(.10,.075,"Diagonal T = target monitor. Every off-diagonal cell is a non-target cross-trigger on a real faulty run, not a false alarm.",
         fontsize=6.35,ha="left",va="bottom")
fig.text(.10,.045,f"Off-diagonal mean {off.mean():.2f}%, median {np.median(off):.1f}%; {int(np.sum(off>=90))}/272 cells are >=90%.",
         fontsize=6.35,ha="left",va="bottom")
fig.savefig(HERE/"P2-SIF05b_full_cross_trigger_matrix.pdf")
fig.savefig(HERE/"P2-SIF05b_full_cross_trigger_matrix.svg")
fig.savefig(HERE/"P2-SIF05b_full_cross_trigger_matrix.png",dpi=600)
plt.close(fig)

fig=plt.figure(figsize=(178/25.4,94/25.4)); ax=fig.add_axes([.08,.25,.77,.59]); ax2=ax.twinx()
p1,=ax.plot(x,ear,marker="o",linewidth=1.35,label="Target among earliest")
p2,=ax.plot(x,uni,marker="s",linestyle="--",linewidth=1.1,label="Target uniquely earliest")
p3,=ax2.plot(x,rank,marker="D",linestyle="-.",linewidth=1.1,label="Median target earliest rank")
ax.set_xticks(x,labels); ax.set_xlabel("Actual reserved TEP fault"); ax.set_ylabel("Runs (%)")
ax2.set_ylabel("Median earliest rank (1 = best)"); ax.set_ylim(-2,104); ax2.set_ylim(17.5,.5)
ax.set_title("Target-monitor earliest-alarm prominence under the hypothetical bank",loc="left",fontweight="bold")
ax.grid(True,axis="y",alpha=.2)
hs=[p1,p2,p3]; ax.legend(hs,[h.get_label() for h in hs],loc="upper center",bbox_to_anchor=(.5,1.0),ncol=3,frameon=False)
fig.text(.08,.085,f"Macro mean: target among earliest {100*float(summary['macro_mean_target_among_earliest_rate']):.2f}% | "
         f"uniquely earliest {100*float(summary['macro_mean_target_uniquely_earliest_rate']):.2f}% | "
         f"macro median earliest rank {float(summary['macro_median_target_earliest_rank']):.0f}/17.",
         fontsize=6.3,ha="left",va="bottom")
fig.text(.08,.058,"Largest-normalized-peak rates and peak ranks are retained in the accompanying compact table.",
         fontsize=6.3,ha="left",va="bottom")
fig.text(.08,.033,"Prominence and rank are descriptive localization diagnostics, not classification accuracy.",
         fontsize=6.3,ha="left",va="bottom")
fig.savefig(HERE/"P2-SIF05c_target_monitor_rank_diagnostics.pdf")
fig.savefig(HERE/"P2-SIF05c_target_monitor_rank_diagnostics.svg")
fig.savefig(HERE/"P2-SIF05c_target_monitor_rank_diagnostics.png",dpi=600)
plt.close(fig)
