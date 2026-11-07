from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with open(HERE/"P1-F02_source_data.csv", encoding="utf-8", newline="") as f:
    R = list(csv.DictReader(f))
identity = next(r for r in R if r["pairing_type"] == "identity")
rand = sorted(
    [r for r in R if r["pairing_type"] == "random_derangement"],
    key=lambda r: int(r["repeat"])
)
x = np.arange(1, len(rand)+1)
stable = np.array([int(r["n_direction_stable_variables"]) for r in rand])
overlap = np.array([int(r["original_top5_overlap_count"]) for r in rand])
max_ratio = np.array([float(r["maximum_direction_ratio"]) for r in rand])

fig = plt.figure(figsize=(178/25.4, 72/25.4))
ax1 = fig.add_axes([0.075, 0.20, 0.405, 0.70])
ax2 = fig.add_axes([0.57, 0.20, 0.405, 0.70])

ax1.scatter(x, stable, s=12, marker="o", label="100 random derangements")
ax1.scatter([0], [int(identity["n_direction_stable_variables"])], s=55, marker="D", label="Identity pairing")
ax1.set_xlim(-4, 104); ax1.set_ylim(-2, 51)
ax1.set_xticks([0,20,40,60,80,100], ["ID","20","40","60","80","100"])
ax1.set_xlabel("Pairing reference / derangement repeat")
ax1.set_ylabel("Direction-stable residual count")
ax1.set_title("A  Direction-stability screen", loc="left", fontweight="bold")
ax1.grid(True, alpha=0.22); ax1.legend(loc="upper right", frameon=False)
ax1.annotate("47 / 52 stable", (0,47), xytext=(10,-5), textcoords="offset points", va="top", fontsize=7.2)
ax1.text(0.03,0.08,
         f"Random maximum direction ratio:\n{max_ratio.min():.4f}-{max_ratio.max():.4f} < 0.95 threshold",
         transform=ax1.transAxes, fontsize=6.8, va="bottom")

ax2.scatter(x, overlap, s=12, marker="o", label="Overlap count")
ax2.scatter([0], [5], s=55, marker="D", label="Identity reference")
ax2.set_xlim(-4,104); ax2.set_ylim(-0.4,5.7)
ax2.set_xticks([0,20,40,60,80,100], ["ID","20","40","60","80","100"])
ax2.set_xlabel("Pairing reference / derangement repeat")
ax2.set_ylabel("Overlap with identity Top-5")
ax2.set_title("B  Historical Top-5 reproducibility", loc="left", fontweight="bold")
ax2.grid(True, alpha=0.22); ax2.legend(loc="upper right", frameon=False)
ax2.annotate("5 / 5 reference", (0,5), xytext=(10,-5), textcoords="offset points", va="top", fontsize=7.2)
ax2.text(0.03,0.08,
         "Random derangements:\n0 overlap in 100 / 100; exact recovery 0 / 100",
         transform=ax2.transAxes, fontsize=6.8, va="bottom")

fig.savefig(HERE/"P1-F02_identity_pairing_vs_random_derangements.pdf")
fig.savefig(HERE/"P1-F02_identity_pairing_vs_random_derangements.svg")
fig.savefig(HERE/"P1-F02_identity_pairing_vs_random_derangements.png", dpi=600)
plt.close(fig)
