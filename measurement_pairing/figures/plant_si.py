"""Draw submission SI Figures S6 and S7 using the two plant analyses."""
import argparse
from pathlib import Path
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, FuncFormatter
from matplotlib.scale import FuncScale
from PIL import Image

STUDY = Path(__file__).resolve().parents[1]
MM = 1 / 25.4
W2 = 190 * MM
RED, BLACK, GREY = "#C00000", "#000000", "#7F7F7F"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.labelweight": "bold", "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "axes.labelpad": 2.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
})
CHECK = []


def save(fig, stem):
    fig.savefig(os.path.join(outdir, stem + ".pdf"))
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=600)
    plt.close(fig)
    print("saved", stem)


def minus(v):
    return str(v).replace("-", "\u2212")


def logminor(ax):
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ pilot plant, four panels
def pilot():
    d = os.path.join(extdir, "2024_Pilot/results")
    tr = [r for r in csv.DictReader(open(os.path.join(d, "working_condition14_trace.csv")))
          if 100 <= int(r["row_zero_based"]) <= 124]
    byrun = list(csv.DictReader(open(os.path.join(d, "lag_metrics_by_run.csv"))))
    agg = list(csv.DictReader(open(os.path.join(d, "lag_metrics_aggregate.csv"))))
    res = list(csv.DictReader(open(os.path.join(d, "adjusted_output_residuals.csv"))))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 128 * MM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.95, wspace=0.27, hspace=0.42)
    # (a) recorded proportional component and 1.7 times the error of the same row
    ax = axs[0, 0]
    t = np.array([float(r["time_s"]) for r in tr])
    p = np.array([float(r["P2_uA"]) for r in tr])
    e = np.array([float(r["KP_times_E_same_uA"]) for r in tr])
    ax.step(t, e, where="post", color=GREY, ls=(0, (3, 1.5)), lw=1.1, label=r"$1.7\,E_2$, same row", zorder=2)
    ax.step(t, p, where="post", color=BLACK, lw=1.1, label=r"Stored $P_2$", zorder=3)
    ax.plot(t, p, "o", ms=2.2, color=BLACK, zorder=4)
    ax.set_xlabel("Time in working condition 14 (s)")
    ax.set_ylabel(r"Proportional component ($\mathbf{\mu}$A)")
    ax.set_title("(a)", loc="left", pad=4)
    ax.legend(loc="upper right", handlelength=2.2)
    CHECK.append(("pilot_a", len(t), t[0], t[-1]))
    # (b) per-record RMSE, same row against preceding row (level-based check)
    ax = axs[0, 1]
    runs = {}
    for r in byrun:
        if r["include_primary"] != "True" or r["metric"] != "q_physical_PV" or r["lag_rows"] not in ("0", "-1"):
            continue
        if int(r["n"]) == 0:
            continue
        runs.setdefault(r["run"], {})[r["lag_rows"]] = float(r["rmse"])
    runs = {k: v for k, v in runs.items() if "0" in v and "-1" in v}
    for k, v in runs.items():
        ax.plot([0, 1], [v["0"], v["-1"]], color="#BDBDBD", lw=0.6, zorder=1)
    ax.plot(np.zeros(len(runs)), [v["0"] for v in runs.values()], "o", ms=3.2, mfc=BLACK, mec=BLACK, zorder=3)
    ax.plot(np.ones(len(runs)), [v["-1"] for v in runs.values()], "o", ms=3.2, mfc=RED, mec=RED, zorder=3)
    ax.set_yscale("log")
    logminor(ax)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0.2, 200)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same row", "Preceding row"])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_ylabel(r"Output RMSE per record ($\mathbf{\mu}$A)")
    ax.set_title("(b)", loc="left", pad=4)
    ax.text(0.97, 0.95, f"{len(runs)} records", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    CHECK.append(("pilot_b", len(runs), round(min(v["-1"] for v in runs.values()), 4),
                  round(max(v["-1"] for v in runs.values()), 4), round(min(v["0"] for v in runs.values()), 2),
                  round(max(v["0"] for v in runs.values()), 2)))
    # (c) pooled RMSE of the clipped output relation against the row offset of the error
    ax = axs[1, 0]
    sel = sorted((int(r["lag_rows"]), float(r["rmse"]), int(r["n"])) for r in agg
                 if r["group"] == "unique_runs_primary" and r["metric"] == "output")
    xs = [s[0] for s in sel]
    ys = [s[1] for s in sel]
    ax.plot(xs, ys, color=GREY, lw=0.8, zorder=1)
    for x, y in zip(xs, ys):
        ax.plot(x, y, "o", ms=4.5, mfc=RED if x == -1 else BLACK, mec=RED if x == -1 else BLACK, zorder=3)
    ax.annotate(f"{dict(zip(xs, ys))[-1]:.3f}", (-1, dict(zip(xs, ys))[-1]), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", color=RED, fontsize=7.5)
    ax.annotate(f"same row\n{dict(zip(xs, ys))[0]:.1f}", (0, dict(zip(xs, ys))[0]), xytext=(5, -3),
                textcoords="offset points", ha="left", va="top", fontsize=7)
    ax.set_yscale("log")
    logminor(ax)
    ax.set_ylim(0.2, 200)
    ax.set_xticks(xs)
    ax.set_xticklabels([minus(x) for x in xs])
    ax.set_xlim(-3.5, 3.5)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    ax.set_xlabel(r"Row offset of error, $\boldsymbol{\ell}$")
    ax.set_ylabel(r"Clipped-output RMSE ($\mathbf{\mu}$A)")
    ax.set_title("(c)", loc="left", pad=4)
    CHECK.append(("pilot_c", sel[0][2]) + tuple(round(y, 4) for y in ys))
    # (d) cumulative distribution of the absolute level-based residual
    ax = axs[1, 1]
    same = np.sort(np.abs(np.array([float(r["q_physical_PV_same_residual_uA"]) for r in res])))
    prev = np.sort(np.abs(np.array([float(r["q_physical_PV_previous_residual_uA"]) for r in res])))
    n = len(same)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= 1, x, 1 + np.log10(np.maximum(x, 1e-12)))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= 1, y, 10 ** (y - 1))

    ax.set_xscale(FuncScale(ax, (fwd, inv)))
    yy = np.arange(1, n + 1) / n
    ax.plot(same, yy, color=BLACK, lw=1.1, label="Same row", zorder=2)
    ax.plot(prev, yy, color=RED, lw=1.1, label="Preceding row", zorder=3)
    ax.axvline(1.04, color=GREY, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 0.5, 1, 10, 100])
    ax.set_xticklabels(["0", "0.5", "1", "10", "100"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Absolute output residual ($\mathbf{\mu}$A)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(d)", loc="left", pad=4)
    ax.legend(loc="lower right")
    f_same = float(np.mean(same <= 1.04))
    f_prev = float(np.mean(prev <= 1.04))
    ax.text(1.25, 0.62, f"within 1.04 $\\mathrm{{\\mu}}$A:\n{100 * f_prev:.1f}% (preceding)\n{100 * f_same:.1f}% (same)",
            fontsize=7, va="top")
    CHECK.append(("pilot_d", n, round(f_prev, 4), round(f_same, 4), round(float(prev.max()), 4)))
    save(fig, "Figure_S6")


# ------------------------------------------------------------------ industrial level loop
def industrial():
    d = os.path.join(extdir, "2003_Industrial/results")
    comp = sorted((int(r["offset_seconds"]), float(r["rmse_refit"]))
                  for r in csv.DictReader(open(os.path.join(d, "primary_comparisons.csv"))))
    pred = list(csv.DictReader(open(os.path.join(d, "test_predictions.csv"))))
    eig = [r for r in csv.DictReader(open(os.path.join(d, "covariance_eigenvalues.csv"))) if r["rank_ascending"] == "1"]
    eig = sorted((int(r["offset_seconds"]), float(r["eigenvalue"])) for r in eig)
    if rawcsv:
        fig, axs = plt.subplots(2, 2, figsize=(W2, 118 * MM))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.95, wspace=0.27, hspace=0.42)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM))
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.88, wspace=0.42)
    k = 0
    labels = "abcd"
    if rawcsv:
        with open(rawcsv, encoding="utf-8-sig") as fh:
            raw = list(csv.DictReader(fh, delimiter=";"))
        cols = {c.strip().lower(): c for c in raw[0].keys()}
        tt = np.array([float(r[cols["time"]]) for r in raw]) / 60
        pv = [float(r[cols["pv"]]) for r in raw]
        op = [float(r[cols["op"]]) for r in raw]
        CHECK.append(("ind_raw", len(raw), tt[0], tt[-1], min(pv), max(pv), min(op), max(op)))
        ax = axs[0]
        ax.axvspan(tt[3], tt[447], color="#EEEEEE", lw=0, zorder=0)
        ax.axvline(tt[456], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.plot(tt, pv, color=BLACK, lw=0.8, label="PV", zorder=3)
        ax.plot(tt, op, color=GREY, lw=0.8, label="OP", zorder=2)
        ax.text((tt[3] + tt[447]) / 2, 66.5, "Training", ha="center", fontsize=7, color=GREY)
        ax.text((tt[456] + tt[-1]) / 2, 66.5, "Test", ha="center", fontsize=7, color=GREY)
        ax.set_xlim(0, tt[-1])
        ax.set_ylim(20, 70)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Recorded value (%)")
        ax.set_title("(a)", loc="left", pad=4)
        ax.legend(loc="lower right", ncol=2)
        k = 1
    # held-out RMSE against the constructed time offset
    ax = axs[k]
    xs = [c[0] for c in comp]
    ys = [c[1] for c in comp]
    ax.plot(xs, ys, color=GREY, lw=0.8, zorder=1)
    for x, y in zip(xs, ys):
        ax.plot(x, y, "o", ms=4.5, mfc=RED if x == 0 else BLACK, mec=RED if x == 0 else BLACK, zorder=3)
    ax.set_yscale("log")
    logminor(ax)
    ax.set_ylim(0.001, 3)
    ax.set_xticks(xs)
    ax.set_xticklabels([minus(x) for x in xs])
    ax.set_xlim(-5, 5)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    ax.set_xlabel("Time offset of level (s)")
    ax.set_ylabel("Held-out RMSE (pp)")
    ax.set_title(f"({labels[k]})", loc="left", pad=4)
    # observed and predicted increments, stored pairing and +2 s
    ax = axs[k + 1]
    for ell, col, mk, lbl in (("1", BLACK, "o", "Level +2 s"), ("0", RED, "o", "Stored pairing")):
        sub = [r for r in pred if r["ell_samples"] == ell]
        ox = [float(r["observed_delta_OP"]) for r in sub]
        py = [float(r["predicted_delta_OP_refit"]) for r in sub]
        if ell == "1":
            ax.plot(ox, py, mk, ms=2.2, mfc="none", mec=col, mew=0.5, label=lbl, zorder=2)
        else:
            ax.plot(ox, py, mk, ms=1.8, mfc=col, mec=col, label=lbl, zorder=3)
        CHECK.append(("ind_c", ell, len(sub)))
    lim = (-1.5, 1.3)
    ax.plot(lim, lim, color=GREY, lw=0.6, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observed $\mathbf{\Delta}$OP (pp)")
    ax.set_ylabel(r"Predicted $\mathbf{\Delta}$OP (pp)")
    ax.set_title(f"({labels[k + 1]})", loc="left", pad=4)
    ax.legend(loc="upper left", handletextpad=0.2)
    # smallest eigenvalue of the feature covariance
    ax = axs[k + 2]
    xs = [c[0] for c in eig]
    ys = [c[1] for c in eig]
    ax.plot(xs, ys, color=GREY, lw=0.8, zorder=1)
    for x, y in zip(xs, ys):
        ax.plot(x, y, "o", ms=4.5, mfc=RED if x == 0 else BLACK, mec=RED if x == 0 else BLACK, zorder=3)
    ax.set_yscale("log")
    logminor(ax)
    ax.set_ylim(1e-5, 0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([minus(x) for x in xs])
    ax.set_xlim(-5, 5)
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    ax.set_xlabel("Time offset of level (s)")
    ax.set_ylabel("Smallest covariance eigenvalue")
    ax.set_title(f"({labels[k + 2]})", loc="left", pad=4)
    CHECK.append(("ind_b", ys))
    CHECK.append(("ind_rmse", [c[1] for c in comp]))
    save(fig, "Figure_S7")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=STUDY / "plant_records",
                        help="Directory with the two case subdirectories and their results folders")
    parser.add_argument("--industrial-csv", type=Path,
                        default=STUDY / "plant_records/2003_Industrial/data/other-L-paper-horch-2003.csv")
    parser.add_argument("--output", type=Path, default=STUDY.parent / "replay_output" / "plant_si")
    args = parser.parse_args()
    extdir, rawcsv, outdir = args.results_root, args.industrial_csv, args.output
    outdir.mkdir(parents=True, exist_ok=True)
    pilot()
    industrial()
    for entry in CHECK:
        print(entry)
