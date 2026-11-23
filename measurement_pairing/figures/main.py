"""Draw the six manuscript figures from the distributed measurements and results."""
import argparse
import gzip
from pathlib import Path
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
from PIL import Image

STUDY = Path(__file__).resolve().parents[1]
ARCHIVE_TABLES = STUDY.parent / "paper" / "Data_S1"
FAMILY_B = STUDY / "family_b" / "statistics"
PLANT = STUDY / "plant_records"
INPUTS = Path(__file__).resolve().parent / "inputs"
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM
RED, BLACK, GREY, BLUE = "#C00000", "#000000", "#7F7F7F", "#1F5FA8"
LGREY = "#EDEDED"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.labelweight": "bold", "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "axes.edgecolor": BLACK, "axes.labelpad": 2.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "lines.linewidth": 1.2, "lines.markersize": 5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
})
CHECK = []
SIZES = []


def save(fig, stem):
    w, h = fig.get_size_inches()
    fig.savefig(os.path.join(outdir, stem + ".pdf"))
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=600)
    tmp = os.path.join(outdir, stem + "_1000.png")
    fig.savefig(tmp, dpi=1000)
    im = Image.open(tmp).convert("RGB")
    im.save(os.path.join(outdir, stem + ".tif"), compression="tiff_lzw", dpi=(1000, 1000))
    os.remove(tmp)
    SIZES.append((stem, round(w / MM, 1), round(h / MM, 1), im.size[0], im.size[1]))
    plt.close(fig)
    print("saved", stem, f"{w / MM:.0f} x {h / MM:.0f} mm", im.size)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------------------------ Fig. 1: flowsheet with the four loops (140 mm)
def fig1():
    S = 140 / 160  # layout drawn for 160 mm; all text and strokes scaled to keep proportions at 140 mm
    X0, X1, Y0, Y1 = 0.0, 15.1, -0.15, 8.2
    fig = plt.figure(figsize=(W15, W15 * (Y1 - Y0) / (X1 - X0)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")
    LW = 1.6 * S
    CLW = 1.2 * S
    DASH = (0, (3, 2))

    def line(pts, color=BLACK, lw=LW, ls="-", arrow=True, z=2):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", dash_capstyle="butt")
        if arrow:
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11 * S, lw=0,
                                         color=color, shrinkA=0, shrinkB=0, zorder=z + 1))

    def unit(x0, y0, w, h, label, size=10.5):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.28",
                                    fc=LGREY, ec=BLACK, lw=LW + 0.2 * S, zorder=4))
        ax.text(x0 + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=size * S, fontweight="bold", zorder=5)

    def valve(x, y, color=BLACK, s=0.24):
        for sg in (-1, 1):
            ax.add_patch(Polygon([(x + sg * s, y + s * 0.75), (x + sg * s, y - s * 0.75), (x, y)], closed=True,
                                 fc="white", ec=color, lw=1.4 * S, zorder=6))

    def ctrl(x, y, text, r=0.36):
        ax.add_patch(Circle((x, y), r, fc="white", ec=RED, lw=1.5 * S, zorder=7))
        ax.text(x, y, text, ha="center", va="center", color=RED, fontsize=(9.5 if len(text) > 1 else 10.5) * S,
                fontweight="bold", zorder=8)

    def tap(x, y):
        ax.add_patch(Circle((x, y), 0.09, fc=RED, ec=RED, zorder=8))

    def node(x, y):
        ax.add_patch(Circle((x, y), 0.07, fc=BLACK, zorder=5))

    def fault(x, y, n, stem_to, r=0.28):
        ax.plot([x, x], [y, stem_to], color=BLACK, lw=1.2 * S, zorder=3)
        ax.add_patch(Circle((x, y), r, fc=BLACK, ec=BLACK, zorder=7))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=9.5 * S, fontweight="bold", zorder=8)

    def lab(x, y, s, ha="center", va="center", size=8.5, color=BLACK, bold=False):
        ax.text(x, y, s, ha=ha, va=va, fontsize=max(size * S, 7.0), color=color,
                fontweight="bold" if bold else "normal", zorder=9)

    yF = 4.0
    xM, xS, xC, xR = 2.9, 3.25, 3.6, 3.95
    line([(0.15, yF), (xM, yF)], arrow=False)
    lab(0.15, yF - 0.38, "D feed", ha="left", size=9.5)
    fault(0.5, yF + 0.72, 3, yF)
    fault(1.12, yF + 0.72, 9, yF)
    tap(1.68, yF)
    valve(2.35, yF)
    lab(1.68, yF - 0.38, "XMEAS2")
    lab(2.35, yF - 0.72, "XMV1")
    ctrl(2.02, yF + 1.45, "P")
    line([(1.68, yF + 0.09), (1.68, yF + 1.45), (1.66, yF + 1.45)], color=RED, lw=CLW, ls=DASH, arrow=False)
    line([(2.38, yF + 1.45), (2.35, yF + 1.45), (2.35, yF + 0.2)], color=RED, lw=CLW, ls=DASH)
    line([(2.02, yF + 2.2), (2.02, yF + 1.83)], color=RED, lw=CLW, ls=DASH)
    lab(0.15, yF + 2.5, "setpoint from PI loop\non XMEAS26", ha="left", color=RED)
    line([(xM, 2.75), (xM, yF)], arrow=False)
    node(xM, yF)
    lab(xM, 2.5, "A, E feeds", size=9.5)
    line([(xM, yF), (xR, yF)])
    node(xS, yF)
    node(xC, yF)
    unit(xR, 3.05, 1.6, 1.9, "Reactor")
    line([(5.55, yF), (6.25, yF)])
    ax.add_patch(Circle((6.95, yF), 0.72, fc="white", ec=BLACK, lw=LW + 0.2 * S, zorder=4))
    ax.plot([6.62, 6.72, 6.82, 6.95, 7.08, 7.18, 7.28], [3.45, 4.45, 3.55, 4.45, 3.55, 4.45, 3.45],
            color=BLUE, lw=1.4 * S, zorder=5)
    lab(6.95, 5.05, "Condenser", size=10.5, bold=True)
    line([(4.25, 1.75), (6.62, 1.75), (6.62, 3.45)], color=BLUE, lw=1.4 * S)
    line([(7.28, 3.45), (7.28, 2.55), (8.05, 2.55)], color=BLUE, lw=1.4 * S)
    lab(4.15, 1.75, "CW", ha="right", size=9.5)
    valve(5.15, 1.75, color=BLUE)
    lab(4.92, 1.22, "XMV11", ha="right")
    fault(5.15, 2.55, 15, 1.95)
    line([(7.67, yF), (8.4, yF)])
    unit(8.4, 3.0, 1.75, 2.0, "Separator", size=10)
    line([(9.0, 5.0), (9.0, 6.12)])
    ax.add_patch(Circle((9.0, 6.62), 0.5, fc="white", ec=BLACK, lw=LW + 0.2 * S, zorder=4))
    ax.add_patch(Polygon([(9.28, 6.88), (9.28, 6.36), (8.74, 6.62)], closed=True, fc="white", ec=BLACK,
                         lw=1.3 * S, zorder=5))
    lab(9.65, 6.62, "Compressor", ha="left", size=10.5, bold=True)
    node(9.0, 5.6)
    line([(9.0, 5.6), (10.35, 5.6)])
    lab(10.45, 5.6, "Purge", ha="left", size=9.5)
    line([(8.5, 6.62), (xC, 6.62), (xC, yF + 0.07)])
    unit(12.2, 1.6, 1.4, 4.5, "Stripper")
    line([(12.9, 6.1), (12.9, 7.6), (xS, 7.6), (xS, yF + 0.07)])
    line([(10.15, 3.35), (12.2, 3.35)])
    valve(11.15, 3.35)
    lab(11.15, 2.86, "XMV7")
    tap(10.15, 4.45)
    lab(10.25, 5.12, "XMEAS12", ha="left")
    ctrl(11.15, 4.45, "P")
    line([(10.24, 4.45), (10.79, 4.45)], color=RED, lw=CLW, ls=DASH, arrow=False)
    line([(11.15, 4.09), (11.15, 3.55)], color=RED, lw=CLW, ls=DASH)
    line([(10.55, 2.25), (12.2, 2.25)])
    lab(10.45, 2.25, "A, C feed", ha="right", size=9.5)
    line([(12.9, 1.6), (12.9, 0.85), (14.9, 0.85)])
    valve(13.75, 0.85)
    lab(13.75, 0.42, "XMV8")
    lab(14.95, 1.2, "Product", ha="right", size=9.5)
    tap(13.6, 4.1)
    lab(13.72, 4.52, "XMEAS15", ha="left")
    ctrl(14.45, 3.2, "P")
    line([(13.69, 4.1), (14.45, 4.1), (14.45, 3.56)], color=RED, lw=CLW, ls=DASH, arrow=False)
    line([(14.45, 2.84), (14.45, 1.6), (13.75, 1.6), (13.75, 1.05)], color=RED, lw=CLW, ls=DASH)
    tap(12.9, 1.2)
    lab(12.78, 0.92, "XMEAS17", ha="right")
    line([(12.81, 1.2), (11.2, 1.2), (11.2, 0.3), (8.81, 0.3)], color=RED, lw=CLW, ls=DASH, arrow=False)
    ctrl(8.45, 0.3, "PI")
    line([(8.09, 0.3), (5.15, 0.3), (5.15, 1.52)], color=RED, lw=CLW, ls=DASH)
    lx, ly = 0.25, 1.85
    for k, (lbl, col, ls, kind) in enumerate([("Process stream", BLACK, "-", "line"),
                                              ("Cooling water (CW)", BLUE, "-", "line"),
                                              ("Control signal", RED, DASH, "line"),
                                              ("Fault number", BLACK, None, "fault")]):
        yy = ly - 0.5 * k
        if kind == "line":
            line([(lx, yy), (lx + 0.75, yy)], color=col, lw=1.3 * S if col != BLACK else LW, ls=ls)
        else:
            ax.add_patch(Circle((lx + 0.37, yy), 0.22, fc=BLACK, zorder=7))
            ax.text(lx + 0.37, yy, "n", ha="center", va="center", color="white", fontsize=7.4, fontweight="bold",
                    zorder=8)
        lab(lx + 0.95, yy, lbl, ha="left", size=9)
    save(fig, "Figure_1")


# ------------------------------------------------------------------ Fig. 3: both families, both records (140 mm)
def fig3():
    # S55 records counts on 120 fault runs. S51 reports the five-fit means in percent.
    rates = {}
    with (ARCHIVE_TABLES / "Table_S55.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["Fault"]:
                fault = int(row["Fault"])
            if row["Monitor"]:
                selected = row["Monitor"].startswith("Full-pool monitor")
            if selected:
                for horizon in ("8 h", "40 h"):
                    rates[fault, row["Recording"], horizon] = 100 * int(row[horizon]) / 120
    with (ARCHIVE_TABLES / "Table_S51.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["Pool"]:
                pool = row["Pool"]
            for horizon in ("8 h", "40 h"):
                if pool == "Second-moment" and row["Endpoint"] == f"Clean detection within {horizon} (%)":
                    for recording in ("Controller-input", "Post-update"):
                        rates[9, recording, horizon] = float(row[recording])
    A = {fault: tuple(rates[fault, recording, horizon]
                      for horizon in ("8 h", "40 h")
                      for recording in ("Controller-input", "Post-update")) for fault in (3, 9, 15)}
    B = {}
    rows = list(csv.DictReader(gzip.open(FAMILY_B / "E17" / "summary_by_horizon.csv.gz", "rt", encoding="utf-8")))
    for f in (3, 9, 15):
        vals = {}
        for r in rows:
            if (r["workflow"] == "Fixed246" and r["rank"] == "95" and r["fault"] == str(f)
                    and r["variant"] in ("aligned", "native") and r["horizon_h"] in ("8", "40")):
                vals[(r["variant"], r["horizon_h"])] = float(r["clean_rate_percent"])
        B[f] = (vals[("aligned", "8")], vals[("native", "8")], vals[("aligned", "40")], vals[("native", "40")])
    for f in (3, 9, 15):
        CHECK.append(("Fig3", "A", f) + tuple(round(v, 1) for v in A[f]))
        CHECK.append(("Fig3", "B", f) + tuple(round(v, 1) for v in B[f]))

    fig, axes = plt.subplots(1, 2, figsize=(W15, 62 * MM), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.30, top=0.91, wspace=0.07)
    centers = {3: 0.0, 9: 1.0, 15: 2.0}
    off = {"A": -0.17, "B": 0.17}
    mk = {"A": "o", "B": "^"}
    for ax, h, title in zip(axes, (0, 1), ("(a) Within 8 h of onset", "(b) Within 40 h of onset")):
        for fam, D in (("A", A), ("B", B)):
            for f in (3, 9, 15):
                x = centers[f] + off[fam]
                ci, pu = D[f][2 * h], D[f][2 * h + 1]
                ax.plot([x, x], [pu, ci], color=GREY, lw=0.8, zorder=1)
                ax.plot(x, ci, mk[fam], ms=5.2, mfc=RED, mec=RED, zorder=3)
                ax.plot(x, pu, mk[fam], ms=5.2, mfc="none", mec=BLACK, mew=1.0, zorder=4)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks([centers[f] + o for f in (3, 9, 15) for o in (off["A"], off["B"])])
        ax.set_xticklabels(["A", "B"] * 3)
        ax.tick_params(axis="x", which="both", top=False)
        for f in (3, 9, 15):
            ax.text(centers[f], -0.13, f"Fault {f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, fontweight="bold")
        ax.set_title(title, loc="left", pad=4)
    axes[0].set_ylabel("Clean detection (%)")
    handles = [Line2D([], [], ls="none", marker="s", ms=5, mfc=RED, mec=RED, label="Controller-input record"),
               Line2D([], [], ls="none", marker="s", ms=5, mfc="white", mec=BLACK, mew=1.0, label="Post-update record"),
               Line2D([], [], ls="none", marker="o", ms=5, mfc=GREY, mec=GREY, label="A: 175 labelled runs"),
               Line2D([], [], ls="none", marker="^", ms=5, mfc=GREY, mec=GREY, label="B: 16 labelled runs")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.0), handletextpad=0.25,
               columnspacing=1.0)
    save(fig, "Figure_3")


# ------------------------------------------------------------------ Fig. 4: single-measurement screen (190 mm)
def fig4():
    rows = list(csv.DictReader(open(FAMILY_B / "E16_S" / "screening_simultaneous_clean8h_rank95.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM), sharey=False)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.9, wspace=0.16)
    for ax, f, title in zip(axes, ("3", "9"), ("(a) Fault 3", "(b) Fault 9")):
        sub = [r for r in rows if r["fault"] == f]
        names = sorted({int(r["measurement_name"].split("(")[1].rstrip(")")) for r in sub})
        xpos = {n: i for i, n in enumerate(names)}
        for d, col, mk, dx in (("S1", RED, "o", -0.17), ("S2", BLACK, "s", 0.17)):
            for r in sub:
                if r["screen_direction"] != d:
                    continue
                n = int(r["measurement_name"].split("(")[1].rstrip(")"))
                est, lo, hi = float(r["estimate_pp"]), float(r["simultaneous_lo_pp"]), float(r["simultaneous_hi_pp"])
                x = xpos[n] + dx
                ax.plot([x, x], [lo, hi], color=col, lw=0.8, zorder=2)
                ax.plot(x, est, mk, ms=3.6, mfc=col if d == "S1" else "white", mec=col, mew=0.8, zorder=3)
                if r["simultaneous_excludes_zero"] == "1":
                    CHECK.append(("Fig4", f, d, n, est, lo, hi))
        ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([str(n) for n in names], fontsize=7)
        ax.tick_params(axis="x", which="both", top=False)
        ax.set_xlim(-0.7, len(names) - 0.3)
        ax.set_xlabel("XMEAS number")
        ax.set_title(title, loc="left", pad=4)
    axes[0].set_ylabel("Change in detection (pp)")
    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=RED, mec=RED, label="Controller-input value into PU+L"),
               Line2D([], [], ls="none", marker="s", ms=4.2, mfc="white", mec=BLACK, mew=0.8,
                      label="Post-update value into CI")]
    axes[1].legend(handles=handles, loc="upper right", handletextpad=0.3)
    save(fig, "Figure_4")


# ------------------------------------------------------------------ Fig. 5: partial noise sharing (140 mm), archived Table S61
def fig5():
    rows = list(csv.DictReader(open(ARCHIVE_TABLES / "Table_S61.csv", encoding="utf-8")))
    rho, var, d8, d40 = [], [], [], []
    for r in rows:
        if not r["Setting"].startswith("ρ"):
            CHECK.append(("Fig5", "PU", r["Normal score variance"], r["Clean ≤8 h"], r["Clean 40 h"]))
            continue
        rho.append(int(r["Shared components (of 12)"]) / 12)
        var.append(float(r["Normal score variance"]))
        d8.append(int(r["Clean ≤8 h"]))
        d40.append(int(r["Clean 40 h"]))
    n = 200
    for i in range(len(rho)):
        CHECK.append(("Fig5", rho[i], var[i], round(100 * d8[i] / n, 1), round(100 * d40[i] / n, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(W15, 52 * MM))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.21, top=0.9, wspace=0.27)
    ax = axes[0]
    ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.plot(rho, var, "-o", color=BLACK, ms=4.5, zorder=3)
    ax.set_ylim(0.6, 2.6)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.set_ylabel("Normal score variance")
    ax.set_title("(a)", loc="left", pad=4)
    ax = axes[1]
    for counts, col, mfc, lbl in ((d40, BLACK, BLACK, "Within 40 h"), (d8, GREY, "white", "Within 8 h")):
        pct = [100 * c / n for c in counts]
        lo = [pct[i] - wilson(c, n)[0] for i, c in enumerate(counts)]
        hi = [wilson(c, n)[1] - pct[i] for i, c in enumerate(counts)]
        ax.errorbar(rho, pct, yerr=[lo, hi], fmt="-o", color=col, mfc=mfc, mec=col, mew=1.0, ms=4.5,
                    elinewidth=0.8, capsize=2, capthick=0.8, label=lbl, zorder=3)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Clean detection (%)")
    ax.set_title("(b)", loc="left", pad=4)
    ax.legend(loc="upper left", handlelength=1.6)
    for ax in axes:
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel(r"Shared fraction, $\boldsymbol{\rho}$")
    save(fig, "Figure_5")


# ------------------------------------------------------------------ Fig. 6: public plant records (140 mm)
def fig6():
    pil = list(csv.DictReader(open(PLANT / "2024_Pilot" / "published" / "lag_metrics_aggregate.csv", encoding="utf-8")))
    sel = sorted(((int(r["lag_rows"]), float(r["rmse"]), float(r["max_abs"]), int(r["n"]), int(r["runs_with_rows"]))
                  for r in pil if r["group"] == "unique_runs_primary" and r["metric"] == "q_physical_PV"))
    ind = list(csv.DictReader(open(PLANT / "2003_Industrial" / "published" / "primary_comparisons.csv", encoding="utf-8")))
    ind = sorted((int(r["offset_seconds"]), float(r["rmse_refit"])) for r in ind)
    for s in sel:
        CHECK.append(("Fig6a",) + s)
    for s in ind:
        CHECK.append(("Fig6b",) + s)
    fig, axes = plt.subplots(1, 2, figsize=(W15, 54 * MM))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.2, top=0.9, wspace=0.3)
    # (a) pilot plant: output residual RMSE against the row offset of the level measurement
    ax = axes[0]
    xs = [s[0] for s in sel]
    ys = [s[1] for s in sel]
    ax.plot(xs, ys, color=GREY, lw=0.8, zorder=1)
    for x, y in zip(xs, ys):
        ok = x == -1
        ax.plot(x, y, "o", ms=4.8, mfc=RED if ok else BLACK, mec=RED if ok else BLACK, zorder=3)
    y_m1 = dict(zip(xs, ys))[-1]
    y_0 = dict(zip(xs, ys))[0]
    ax.annotate(f"{y_m1:.3f}", (-1, y_m1), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                color=RED, fontsize=7.5)
    ax.annotate(f"same row\n{y_0:.1f}", (0, y_0), xytext=(5, -3), textcoords="offset points", ha="left",
                va="top", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 200)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x).replace("-", "−") for x in xs])
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(r"Row offset of level, $\boldsymbol{\ell}$")
    ax.set_ylabel(r"Output RMSE ($\mathbf{\mu}$A)")
    ax.set_title("(a) Pilot plant, PID level loop", loc="left", pad=4)
    # (b) industrial level loop: held-out RMSE of a refitted PI-type relation against the level time offset
    ax = axes[1]
    xs = [s[0] for s in ind]
    ys = [s[1] for s in ind]
    ax.plot(xs, ys, color=GREY, lw=0.8, zorder=1)
    for x, y in zip(xs, ys):
        ok = x == 0
        ax.plot(x, y, "o", ms=4.8, mfc=RED if ok else BLACK, mec=RED if ok else BLACK, zorder=3)
    y0 = dict(zip(xs, ys))[0]
    ax.annotate(f"stored pairing\n{y0:.5f}".rstrip("0"), (0, y0), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", color=RED, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x).replace("-", "−") for x in xs])
    ax.set_xlim(-5, 5)
    ax.set_xlabel("Time offset of level (s)")
    ax.set_ylabel("Held-out RMSE (pp)")
    ax.set_title("(b) Industrial level loop", loc="left", pad=4)
    for ax in axes:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    save(fig, "Figure_6")



# ------------------------------------------------------------------ Fig. 2: the two conventions (190 mm)
def fig2():
    rows = list(csv.DictReader(gzip.open(INPUTS / "paired_measurements.csv.gz", "rt", encoding="utf-8-sig")))
    pts = {"CI": ([], []), "PU": ([], [])}
    for r in rows:
        if int(r["record"]) < 161:
            continue
        pts[r["recording"]][0].append(float(r["XMEAS12"]))
        pts[r["recording"]][1].append(float(r["XMV7"]))
    for k in ("CI", "PU"):
        x, y = np.array(pts[k][0]), np.array(pts[k][1])
        CHECK.append(("Fig2b", k, len(x), round(float(np.corrcoef(x, y)[0, 1]), 6)))
    fig = plt.figure(figsize=(W2, 66 * MM))
    # (a) schematic of one control interval
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.1)
    ax.axis("off")
    ax.text(0.05, 6.85, "(a)", fontsize=8.5, fontweight="bold", va="center")
    bw, bh, gap = 2.05, 1.45, 0.42

    def box(x, y, top, bottom, ec=BLACK, fc="white", lw=0.9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.16", fc=fc, ec=ec, lw=lw,
                                    zorder=3))
        ax.text(x + bw / 2, y + bh * 0.66, top, ha="center", va="center", fontsize=8,
                fontweight="bold" if bold else "normal", zorder=4)
        ax.text(x + bw / 2, y + bh * 0.3, bottom, ha="center", va="center", fontsize=8, zorder=4)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=BLACK,
                                     shrinkA=0, shrinkB=0, zorder=2))

    rows_def = [
        (4.25, "Controller-input recording", RED,
         [("Measure", r"$m_k$", None), ("Control", r"$u_k$", None), ("Record", r"$(m_k,\ u_k)$", "CI"),
          ("Integrate", "3 s", None)]),
        (1.0, "Post-update recording", BLACK,
         [("Measure", r"$m_k$", None), ("Control", r"$u_k$", None), ("Integrate", "3 s", None),
          ("Measure", r"$m^{\prime}_{k}$", None), ("Record", r"$(m^{\prime}_{k},\ u_k)$", "PU")]),
    ]
    for y, title, col, boxes in rows_def:
        ax.text(0.1, y + bh + 0.42, title, color=col, fontsize=8.5, fontweight="bold", va="center")
        for i, (top, bottom, kind) in enumerate(boxes):
            x = 0.1 + i * (bw + gap)
            if kind == "CI":
                box(x, y, top, bottom, ec=RED, fc="#FBE3E3", lw=1.6, bold=True)
            elif kind == "PU":
                box(x, y, top, bottom, ec=BLACK, fc="#E6E6E6", lw=1.6, bold=True)
            else:
                box(x, y, top, bottom)
            if i < len(boxes) - 1:
                arrow(x + bw + 0.04, x + bw + gap - 0.04, y + bh / 2)
    # (b) separator level against its valve output
    ax = fig.add_axes([0.715, 0.17, 0.27, 0.73])
    xp, yp = pts["PU"]
    xc, yc = pts["CI"]
    ax.scatter(xp, yp, s=1.0, c=BLACK, alpha=0.35, lw=0, rasterized=True, zorder=2)
    ax.scatter(xc, yc, s=1.0, c=RED, lw=0, rasterized=True, zorder=3)
    ax.set_xlabel("XMEAS12 (%)")
    ax.set_ylabel("XMV7 (%)")
    ax.set_xlim(45.5, 54.5)
    ax.set_ylim(24, 56)
    ax.set_xticks([46, 48, 50, 52, 54])
    ax.set_yticks([25, 30, 35, 40, 45, 50, 55])
    ax.set_title("(b)", loc="left", pad=4, x=-0.22)
    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=RED, mec=RED, label="Controller-input"),
               Line2D([], [], ls="none", marker="o", ms=3.5, mfc=GREY, mec=GREY, label="Post-update")]
    ax.legend(handles=handles, loc="upper left", handletextpad=0.2, borderaxespad=0.3)
    save(fig, "Figure_2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY.parent / "replay_output" / "main")
    outdir = parser.parse_args().output
    outdir.mkdir(parents=True, exist_ok=True)
    for draw in (fig1, fig3, fig4, fig5, fig6, fig2):
        draw()
    with (outdir / "check_values.tsv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(CHECK)
    with (outdir / "sizes.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("figure", "width_mm", "height_mm", "tiff_px_w", "tiff_px_h"))
        writer.writerows(SIZES)
