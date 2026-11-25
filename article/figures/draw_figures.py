"""Draw the six main figures from the accompanying plotting data."""
import argparse
import gzip
from pathlib import Path
import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
import fitz

INPUTS = Path(__file__).resolve().parent / "data"
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM
RED, BLACK, GREY, BLUE = "#C00000", "#000000", "#7F7F7F", "#1F5FA8"
LGREY = "#EDEDED"
STATE_FILL = "#F6C9C9"

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


def save(fig, stem):
    figure_path = outdir / f"{stem}.pdf"
    fig.savefig(figure_path, dpi=600)
    plt.close(fig)



def fig1():
    S = 140 / 160  # layout drawn for 160 mm; all text and strokes scaled to keep proportions at 140 mm
    X0, X1, Y0, Y1 = -1.15, 15.45, -0.55, 8.5
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

    def ctrl(x, y, text, r=0.36, state=False):
        ax.add_patch(Circle((x, y), r, fc=STATE_FILL if state else "white", ec=RED,
                            lw=(2.1 if state else 1.2) * S, zorder=7))
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
    line([(-0.79, yF), (xM, yF)], arrow=False)
    lab(-0.79, yF - 0.38, "D feed", ha="left", size=9.5)
    lab(-0.75, 6.22, "D-feed temperature", ha="left", bold=True)
    fault(-0.46, 5.61, 3, 4.96, r=0.25)
    fault(-0.46, 4.96, 9, yF, r=0.25)
    lab(-0.07, 5.61, "Step", ha="left")
    lab(-0.07, 4.96, "Random", ha="left")
    tap(1.68, yF)
    valve(2.35, yF)
    lab(1.68, yF - 0.38, "XMEAS2")
    lab(2.35, yF - 0.72, "XMV1")
    ctrl(2.02, yF + 1.45, "P", state=True)
    line([(1.68, yF + 0.09), (1.68, yF + 1.45), (1.66, yF + 1.45)], color=RED, lw=CLW, ls=DASH, arrow=False)
    line([(2.38, yF + 1.45), (2.35, yF + 1.45), (2.35, yF + 0.2)], color=RED, lw=CLW, ls=DASH)
    line([(2.02, 6.92), (2.02, yF + 1.83)], color=RED, lw=CLW, ls=DASH)
    lab(1.05, 7.37, "Setpoint from PI loop\non XMEAS26", color=RED)
    line([(xM, 2.75), (xM, yF)], arrow=False)
    node(xM, yF)
    lab(xM - 0.03, 2.5, "A, E feeds", ha="right", size=9.5)
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
    lab(4.15, 1.75, "CW in", ha="right", size=9.5, color=BLUE)
    lab(7.92, 2.20, "CW out", color=BLUE)
    valve(5.15, 1.75, color=BLUE)
    lab(4.92, 1.22, "XMV11", ha="right")
    fault(5.15, 2.55, 15, 1.95)
    lab(4.75, 2.55, "CW valve\nsticking", ha="right")
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
    unit(12.2, 1.6, 1.4, 4.5, "Stripper", size=9.7)
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
    ctrl(8.45, 0.3, "PI", state=True)
    line([(8.09, 0.3), (5.15, 0.3), (5.15, 1.52)], color=RED, lw=CLW, ls=DASH)
    lx, ly = -0.73, 2.05
    for k, (lbl, col, ls, kind) in enumerate([("Process stream", BLACK, "-", "line"),
                                              ("Cooling water (CW)", BLUE, "-", "line"),
                                              ("Control signal", RED, DASH, "line"),
                                              ("Uses a state not stored", RED, None, "state"),
                                              ("Static P controller", RED, None, "static"),
                                              ("Fault number", BLACK, None, "fault")]):
        yy = ly - 0.43 * k
        if kind == "line":
            line([(lx, yy), (lx + 0.75, yy)], color=col, lw=1.3 * S if col != BLACK else LW, ls=ls)
        elif kind in ("state", "static"):
            ax.add_patch(Circle((lx + 0.37, yy), 0.2, fc=STATE_FILL if kind == "state" else "white", ec=RED,
                                lw=(2.1 if kind == "state" else 1.2) * S, zorder=7))
        else:
            ax.add_patch(Circle((lx + 0.37, yy), 0.22, fc=BLACK, zorder=7))
            ax.text(lx + 0.37, yy, "n", ha="center", va="center", color="white", fontsize=7.4, fontweight="bold",
                    zorder=8)
        lab(lx + 0.95, yy, lbl, ha="left", size=9)
    save(fig, "Figure_1")


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
    for left, label in ((0.008, "(a)"), (0.715, "(b)")):
        fig.text(left, 0.975, label, ha="left", va="top", fontsize=8.5, fontweight="bold")
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
    ax.set_xlabel("Separator level, XMEAS12 (%)")
    ax.set_ylabel("Separator pot valve, XMV7 (%)")
    ax.set_xlim(45.5, 54.5)
    ax.set_ylim(24, 56)
    ax.set_xticks([46, 48, 50, 52, 54])
    ax.set_yticks([25, 30, 35, 40, 45, 50, 55])
    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=RED, mec=RED, label="Controller-input"),
               Line2D([], [], ls="none", marker="o", ms=3.5, mfc=GREY, mec=GREY, label="Post-update")]
    ax.legend(handles=handles, loc="upper left", handletextpad=0.2, borderaxespad=0.3)
    save(fig, "Figure_2")


def fig3():
    # Convert the 120-run counts to percentages; the five-fit means are already percentages.
    rates = {}
    with (INPUTS / "family_a_120_run_counts.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["Fault"]:
                fault = int(row["Fault"])
            if row["Monitor"]:
                selected = row["Monitor"].startswith("Full-pool monitor")
            if selected:
                for horizon in ("8 h", "40 h"):
                    rates[fault, row["Recording"], horizon] = 100 * int(row[horizon]) / 120
    with (INPUTS / "family_a_five_fit_rates.csv").open(encoding="utf-8") as stream:
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
    rows = list(csv.DictReader((INPUTS / "family_b_detection_rates.csv").open(encoding="utf-8")))
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
    fig.subplots_adjust(left=0.093, right=0.982, bottom=0.29, top=0.90, wspace=0.18)
    centers = {3: 0.0, 9: 1.0, 15: 2.0}
    off = {"A": -0.17, "B": 0.17}
    record_offset = 0.071
    for ax, h, title in zip(axes, (0, 1), ("(a) Within 8 h of onset", "(b) Within 40 h of onset")):
        for level in (25, 50, 75, 100):
            ax.axhline(level, color="#DDDDDD", lw=0.52, ls=(0, (3, 3)), zorder=0)
        for fam, D in (("A", A), ("B", B)):
            for f in (3, 9, 15):
                x = centers[f] + off[fam]
                ci, pu = D[f][2 * h], D[f][2 * h + 1]
                x_ci, x_pu = x - record_offset, x + record_offset
                ax.plot([x_ci, x_pu], [ci, pu], color="#949494", lw=0.7, zorder=1)
                ax.plot(x_ci, ci, "o", ms=4.6, mfc=RED, mec=RED, mew=0.85, zorder=3)
                ax.plot(x_pu, pu, "o", ms=4.6, mfc="white", mec=BLACK, mew=0.85, zorder=3)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xticks([centers[f] + o for f in (3, 9, 15) for o in (off["A"], off["B"])])
        ax.set_xticklabels(["A", "B"] * 3)
        ax.tick_params(axis="x", which="both", top=False)
        for f in (3, 9, 15):
            ax.text(centers[f], -0.13, f"Fault {f}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, fontweight="bold")
        ax.set_title(title, loc="left", pad=4)
    axes[0].set_ylabel("Clean detection (%)")
    handles = [Line2D([], [], ls="none", marker="o", ms=4.6, mfc=RED, mec=RED, mew=0.85,
                      label="Controller-input record"),
               Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white", mec=BLACK, mew=0.85,
                      label="Post-update record")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.008),
               handlelength=1.0, handletextpad=0.4, columnspacing=2.2)
    save(fig, "paired_detection_panels")


def make_panel(width, records):
    height = 134
    figure = plt.figure(figsize=(width / 72, height / 72))
    figure.patch.set_facecolor("white")
    # All layout coordinates below use PDF points at the official figure size.
    def label(x, top, value, **kwargs):
        figure.text(x / width, 1 - top / height, value, va="center", **kwargs)

    left, right = width * 0.545, width * 0.97
    axis_top, axis_bottom = 17.5, 107.5
    axis = figure.add_axes((left / width, 1 - axis_bottom / height,
                            (right - left) / width, (axis_bottom - axis_top) / height))
    axis.set_xlim(-10, 100)
    axis.set_ylim(axis_bottom, axis_top)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_yticks([])
    for side in ("top", "left", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.tick_params(axis="x", width=0.55, length=2.3, pad=2, labelsize=7.1)
    axis.axvline(0, color="#5D5D5D", linewidth=0.75, zorder=1)
    for value in (25, 50, 75, 100):
        axis.axvline(value, color="#DDDDDD", linewidth=0.42, linestyle=(0, (2, 3)), zorder=0)

    label(8, 6, "(c) Recording differences within 40 h", size=8.5, weight="bold")
    label(10, 24, "Family A, Fault 9", size=7.7, weight="bold")
    label(10, 61, "Normal-trained SFA", size=7.7, weight="bold")
    label(10, 32, "Five-fit panel", size=6.9, style="italic")
    label(10, 69, "Confirmation panel", size=6.9, style="italic")
    label(width * 0.462, 16, "CI / PU (%)", ha="center", size=7.0, weight="bold")

    label((left + right) / 2, 14.5, "Difference with 95% interval",
          ha="center", size=7.0, weight="bold")

    y_positions = [41, 51, 78, 88, 98]
    for record, y in zip(records, y_positions):
        display = record["representation"] if record["table"] == "S35" else f'Fault {record["fault"]}'
        label(width * 0.08, y, display, size=7.4)
        rates = " / ".join(f'{float(record[key]):g}' for key in ("controller_input_pct", "post_update_pct"))
        label(width * 0.462, y, rates, size=7.3, ha="center")
        difference = float(record["difference_pp"])
        lower, upper = float(record["lower_95_pp"]), float(record["upper_95_pp"])
        axis.errorbar(difference, y, xerr=[[difference - lower], [upper - difference]],
                      fmt="D", markersize=3.2, color="#262626", markeredgewidth=0.65,
                      capsize=2.4, elinewidth=0.9, zorder=3)
    label((left + right) / 2, 128, "CI − PU in clean detection (percentage points)", ha="center",
          weight="bold", size=7.6)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf")
    plt.close(figure)
    return buffer.getvalue()


def fig4():
    # (a) main effects of the 2 x 2 exchange, fixed template, rank 95, clean detection within 8 h. Point estimates
    #     are recomputed from summary_by_horizon.csv.gz; the crossed-bootstrap intervals are those of SI Table S23.
    rates = {}
    for r in csv.DictReader(gzip.open(INPUTS / "summary_by_horizon.csv.gz", "rt",
                                      encoding="utf-8")):
        if (r["workflow"] == "Fixed246" and r["rank"] == "95" and r["horizon_h"] == "8"
                and r["variant"] in ("aligned", "native", "X1", "X2")):
            rates[int(r["fault"]), r["variant"]] = float(r["clean_rate_percent"])
    ref = {}
    for r in csv.DictReader(open(INPUTS / "exchange_main_effects_rank95.csv", encoding="utf-8")):
        ref[int(r["fault"]), r["group"]] = (float(r["estimate_pp"]), float(r["lo_pp"]), float(r["hi_pp"]))
    effects = {}
    for f in (3, 9, 15):
        aa, nn, an, na = rates[f, "aligned"], rates[f, "native"], rates[f, "X1"], rates[f, "X2"]
        effects[f, "three_loop_measurements"] = (aa - na + an - nn) / 2
        effects[f, "other_19_measurements"] = (aa + na - an - nn) / 2
    for key, est in effects.items():
        if abs(est - ref[key][0]) > 0.006:
            raise SystemExit(f"main effect {key}: {est:.3f} recomputed, {ref[key][0]} tabulated")
        CHECK.append(("Fig4a",) + key + tuple(round(v, 2) for v in ref[key]))
    rows = list(csv.DictReader(open(INPUTS / "screening_simultaneous_clean8h_rank95.csv")))
    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 2, 2], left=0.062, right=0.988, bottom=0.25, top=0.9,
                          wspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i, f in enumerate((3, 9, 15)):
        for group, dx, mk, fc in (("three_loop_measurements", -0.16, "D", "#9E9E9E"),
                                  ("other_19_measurements", 0.16, "^", BLACK)):
            est, lo, hi = ref[f, group]
            ax.plot([i + dx, i + dx], [lo, hi], color=BLACK, lw=0.8, zorder=2)
            ax.plot(i + dx, est, mk, ms=4.6 if mk == "^" else 4.0, mfc=fc, mec=BLACK, mew=0.8, zorder=3)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["3", "9", "15"])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlabel("Fault")
    ax.set_ylim(-10, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Main effect on detection (pp)")
    ax.set_title("(a) Measurement groups", loc="left", pad=4)
    screen_axes = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    screen_axes[1].sharey(screen_axes[0])
    for ax, f, title in zip(screen_axes, ("3", "9"), ("(b) Fault 3, single measurements", "(c) Fault 9, single measurements")):
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
                if d == "S1" and n == 2:
                    ax.annotate(f"XMEAS2 (D-feed flow)\n{est:+.1f} pp", (x, est), xytext=(6, 0),
                                textcoords="offset points", ha="left", va="center", fontsize=7.6,
                                color=RED, linespacing=1.05)
                if d == "S1" and f == "3" and n in (9, 14):
                    label, dx_text, align = (("XMEAS9\nReactor temp.", -5, "right") if n == 9
                                             else ("XMEAS14\nSeparator underflow", 5, "left"))
                    ax.annotate(label, (x, hi), xytext=(dx_text, 4), textcoords="offset points",
                                ha=align, va="bottom", fontsize=6.8, color="#444444", linespacing=1.05,
                                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.55, shrinkA=1,
                                                shrinkB=1, relpos=(1 if n == 9 else 0, 0)))
                if r["simultaneous_excludes_zero"] == "1":
                    CHECK.append(("Fig4", f, d, n, est, lo, hi))
        ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(-60, 105)
        ax.set_yticks(range(-60, 101, 20))
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([str(n) for n in names], fontsize=6.6)
        ax.tick_params(axis="x", which="both", top=False)
        ax.set_xlim(-0.7, len(names) - 0.3)
        ax.set_xlabel("XMEAS number")
        ax.set_title(title, loc="left", pad=4)
    screen_axes[0].set_ylabel("Change in detection (pp)")
    handles = [Line2D([], [], ls="none", marker="D", ms=4.0, mfc="#9E9E9E", mec=BLACK, mew=0.8,
                      label="XMEAS12, 15 and 17 exchanged"),
               Line2D([], [], ls="none", marker="^", ms=4.6, mfc=BLACK, mec=BLACK, mew=0.8,
                      label="Other 19 measurements exchanged"),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc=RED, mec=RED,
                      label="Controller-input value into PU record with controller-input loop measurements"),
               Line2D([], [], ls="none", marker="s", ms=4.0, mfc="white", mec=BLACK, mew=0.8, color=BLACK,
                      label="Post-update value into CI record")]
    fig.legend(handles=handles[:2], loc="lower left", bbox_to_anchor=(0.005, 0.005), ncol=1,
               handlelength=0.9, handletextpad=0.4, labelspacing=0.3)
    fig.legend(handles=handles[2:], loc="lower left", bbox_to_anchor=(0.27, 0.005), ncol=1,
               handlelength=0.9, handletextpad=0.45, labelspacing=0.3)
    save(fig, "Figure_4")


def fig5():
    with (INPUTS / "dfeed_clean_detection.csv").open(encoding="utf-8") as stream:
        rows = {(int(row["fault"]), row["arm"], row["endpoint"]): row
                for row in csv.DictReader(stream)}

    fig, axes = plt.subplots(1, 2, figsize=(5.51, 2.44), sharey=True)
    fig.subplots_adjust(left=0.094, right=0.977, bottom=0.264, top=0.878, wspace=0.06)
    styles = [
        ("CI", "Controller-input record", 0.12, RED, "o", RED, "-"),
        ("RC", "Post-update record, XMEAS2 recomputed", 0.0, "#6E6E6E", "D", "#6E6E6E", "-"),
        ("PU", "Post-update record", -0.12, BLACK, "o", "white", "--"),
    ]
    horizons = (2, 8, 40)
    handles = []
    for ax, fault, panel in zip(axes, (3, 9), ("(a)", "(b)")):
        for arm, label, offset, color, marker, fill, line in styles:
            values = [rows[fault, arm, "clean" + str(h)] for h in horizons]
            means = np.array([float(row["estimate"]) for row in values])
            lower = np.array([float(row["lower95"]) for row in values])
            upper = np.array([float(row["upper95"]) for row in values])
            counts = np.array([int(row["events"]) for row in values])
            denominators = np.array([int(row["reconstructions"]) * int(row["physical_runs"])
                                     for row in values])
            np.testing.assert_allclose(100 * counts / denominators, means, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(denominators, np.full(3, 36 * 120))
            for h, count, mean, lo, hi in zip(horizons, counts, means, lower, upper):
                CHECK.append(("Fig5", fault, arm, h, int(count), mean, lo, hi))
            points = ax.errorbar(
                np.arange(3) + offset, means, yerr=np.stack((means - lower, upper - means)),
                fmt=marker, linestyle=line, lw=0.8, color=color, mfc=fill, mec=color,
                ms=4.6, mew=0.9, elinewidth=0.8, capsize=2.1, label=label, zorder=3,
            )
            if fault == 3:
                handles.append(points.lines[0])
        ax.set_title(f"{panel} Fault {fault}", loc="left", pad=6)
        ax.set_xlabel("Time after onset")
        ax.set_xlim(-0.45, 2.45)
        ax.set_xticks([0, 1, 2], ["2 h", "8 h", "40 h"])
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", top=False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9D9D9", linestyle="--", lw=0.5)
    axes[0].set_ylabel("Clean detection (%)")
    fig.legend(handles, [style[1] for style in styles], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.004), fontsize=8, handlelength=1.9,
               handletextpad=0.55, columnspacing=1.2)
    save(fig, "Figure_5")


def fig6():
    pil = list(csv.DictReader(open(INPUTS / "lag_metrics_aggregate.csv", encoding="utf-8")))
    sel = sorted(((int(r["lag_rows"]), float(r["rmse"]), float(r["max_abs"]), int(r["n"]), int(r["runs_with_rows"]))
                  for r in pil if r["group"] == "unique_runs_primary" and r["metric"] == "q_physical_PV"))
    ind = list(csv.DictReader(open(INPUTS / "primary_comparisons.csv", encoding="utf-8")))
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
    ax.annotate("row $t-1$\n" + f"{y_m1:.3f}", (-1, y_m1), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", color=RED, fontsize=7)
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


def append_contrast_panel():
    """Append Figure 3c below the two clean-detection panels."""
    with (INPUTS / "representation_and_sfa_contrasts.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if [record["table"] for record in records] != ["S35", "S35", "S36", "S36", "S36"]:
        raise ValueError("Figure 3c requires two representation rows and three SFA rows.")
    intermediate = outdir / "paired_detection_panels.pdf"
    with plt.rc_context():
        # The contrast panel uses the Matplotlib defaults with its own font settings.
        plt.rcdefaults()
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                             "pdf.fonttype": 42, "axes.unicode_minus": True})
        with fitz.open(intermediate) as original, fitz.open() as combined:
            top = original[0].rect
            with fitz.open(stream=make_panel(top.width, records), filetype="pdf") as panel:
                page = combined.new_page(width=top.width, height=top.height + panel[0].rect.height)
                page.show_pdf_page(top, original, 0)
                page.show_pdf_page(fitz.Rect(0, top.height, page.rect.width, page.rect.height), panel, 0)
                combined.set_metadata({"title": "Figure 3: paired recordings and 40 h detection differences"})
                combined.save(outdir / "Figure_3.pdf", garbage=4, deflate=True)
    intermediate.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("redrawn_figures"))
    outdir = parser.parse_args().output
    outdir.mkdir(parents=True, exist_ok=True)
    for draw in (fig1, fig2, fig3, fig4, fig5, fig6):
        draw()
    append_contrast_panel()
    with (outdir / "plotted_values.tsv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(CHECK)
