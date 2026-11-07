from pathlib import Path
import csv, math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
with open(HERE/"P2-SIF02_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))

marker_map={
    "global_pca_statistic":"P",
    "dense_52_residual":"s",
    "single_residual":"^",
    "sparse_top5":"o",
}
offsets={
    "pca_q_cusum":(-30,16),
    "pca_t2_cusum":(-28,22),
    "pca_joint_cusum":(-20,42),
    "all52_signed_cusum":(-18,-32),
    "lda_residual_cusum":(-62,-30),
    "top1_signed_cusum":(8,-28),
    "proposed_top5_cusum":(8,-26),
}
def bubble_size(delay):
    return 20.0*math.sqrt(delay)

plt.rcParams.update({
    "font.size":8.0,"axes.titlesize":9.2,"axes.labelsize":8.2,
    "xtick.labelsize":7.2,"ytick.labelsize":7.2,"legend.fontsize":6.6
})
fig=plt.figure(figsize=(178/25.4,108/25.4))
fig.text(.08,.95,"Fault 3 clean-data sequential operating trade-offs",
         fontsize=9.4,fontweight="bold",ha="left",va="top")
ax=fig.add_axes([.08,.18,.66,.68])

for r in R:
    far=float(r["normal_far_pct"])
    occ=float(r["threshold_occupancy_pct"])
    delay=float(r["median_detected_run_delay_min"])
    ax.scatter([far],[occ],s=bubble_size(delay),
               marker=marker_map[r["complexity_class"]],alpha=.78,zorder=3)

for r in R:
    far=float(r["normal_far_pct"])
    occ=float(r["threshold_occupancy_pct"])
    det=float(r["run_detection_pct"])
    delay=float(r["median_detected_run_delay_min"])
    delay_txt=f"{delay:.0f}" if abs(delay-round(delay))<1e-9 else f"{delay:.1f}"
    ax.annotate(f'{r["method_short"]}\n{det:.1f}% det.; {delay_txt} min',
                (far,occ),xytext=offsets[r["method_key"]],
                textcoords="offset points",fontsize=6.25)

ax.set_xlim(2.5,8.7)
ax.set_ylim(-2,102)
ax.set_xlabel("Normal run-level FAR (%)")
ax.set_ylabel("Fault 3 post-fault threshold occupancy (%)")
ax.grid(True,alpha=.22)

fig.text(.78,.82,"Encoding",fontsize=8.5,fontweight="bold",ha="left")
handles=[
    Line2D([0],[0],marker="P",linestyle="None",markersize=7,label="Global PCA statistic"),
    Line2D([0],[0],marker="s",linestyle="None",markersize=7,label="Dense 52-residual fusion"),
    Line2D([0],[0],marker="^",linestyle="None",markersize=7,label="Single residual"),
    Line2D([0],[0],marker="o",linestyle="None",markersize=7,label="Sparse Top-5 residuals"),
]
fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.765,.78),frameon=False)

fig.text(.78,.53,"Bubble area",fontsize=7.1,fontweight="bold",ha="left")
fig.text(.78,.495,"Median detected-run delay\n(monotone scale; larger = later)",
         fontsize=6.45,ha="left",va="top",linespacing=1.2)
fig.text(.78,.36,"Detection context",fontsize=7.1,fontweight="bold",ha="left")
fig.text(.78,.325,
         "250/250: All-52, LDA,\nTop-1, Proposed\n\n"
         "33/250: Q-CUSUM\n"
         "19/250: T2-CUSUM\n"
         "22/250: Joint CUSUM",
         fontsize=6.45,ha="left",va="top",linespacing=1.18)

fig.text(.08,.055,
         "Delay is conditional on detected runs; for Q/T2/Joint CUSUM the detected subsets are small, so bubble size is not an unconditional speed ranking.",
         fontsize=6.25,ha="left",va="bottom")

fig.savefig(HERE/"P2-SIF02_fault3_clean_data_tradeoffs.pdf")
fig.savefig(HERE/"P2-SIF02_fault3_clean_data_tradeoffs.svg")
fig.savefig(HERE/"P2-SIF02_fault3_clean_data_tradeoffs.png",dpi=600)
plt.close(fig)
