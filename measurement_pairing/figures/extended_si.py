"""Draw the fourteen archived Family A figures in the numbering of the extended SI."""
import argparse
import gzip
import zipfile
from pathlib import Path
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FuncFormatter

STUDY = Path(__file__).resolve().parents[1]
INPUTS = Path(__file__).resolve().parent / "inputs"
TABLES = STUDY.parent / "paper" / "Data_S1"
PAIRED = INPUTS / "paired_measurements.csv.gz"
MM = 1 / 25.4
W2 = 190 * MM
RED, BLACK, GREY, BLUE, LGREY = "#C00000", "#000000", "#7F7F7F", "#1F5FA8", "#BDBDBD"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.labelweight": "bold", "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.linewidth": 0.8, "axes.labelpad": 2.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2, "legend.frameon": False, "lines.linewidth": 1.1, "pdf.fonttype": 42,
})
LOG = []


def rd(path, **kw):
    stream = gzip.open(path, "rt", encoding="utf-8-sig") if path == PAIRED else path.open("r", encoding="utf-8-sig")
    with stream as fh:
        return list(csv.DictReader(fh, **kw))


def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=600)
    plt.close(fig)
    print("saved", stem)


def title(ax, s):
    ax.set_title(s, loc="left", pad=4)


def onset(ax, x=8.0):
    ax.axvline(x, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=0)


TG = np.linspace(0, 40, 801)


def curve_from_delays(delays_h, n):
    d = np.array([x for x in delays_h if x is not None])
    return np.array([100.0 * np.sum(d <= t) / n for t in TG])


def delay_from_record(rec):
    """clean delay in hours from a 1-based first-alarm record; None for no alarm or a pre-fault alarm"""
    if rec in (None, "", "0"):
        return None
    r = int(float(rec))
    if r < 161:
        return None
    return (r - 161) * 0.05


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ------------------------------------------------------------------ S2 and S3 (Validation 2 examples)
def s2s3(fault, perrun, paths_mon, styles, stem, legloc="lower right"):
    paths = [r for r in rd(SRC.joinpath("S2_S3/fig45_cusum_paths.csv")) if r["fault"] == str(fault)]
    runs = rd(SRC.joinpath("S2_S3", perrun))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.9, wspace=0.25)
    ax = axs[0]
    for mon, col, ls, lab in paths_mon:
        sub = sorted((int(r["record"]), max(float(r["c_plus"]), float(r["c_minus"])) / float(r["limit_h"]))
                     for r in paths if r["monitor"] == mon)
        ax.plot([s[0] * 0.05 for s in sub], [s[1] for s in sub], color=col, ls=ls, lw=1.0, label=lab)
        LOG.append((stem, "path", mon, len(sub), round(max(s[1] for s in sub), 3)))
    ax.axhline(1.0, color=GREY, lw=0.7, ls=(0, (4, 2)))
    onset(ax)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("CUSUM statistic / limit")
    ax.legend(loc="upper left", handlelength=2.2)
    title(ax, "(a)")
    ax = axs[1]
    for mon, (col, ls, lab) in styles.items():
        curves = []
        for rep in sorted({r["replicate"] for r in runs}):
            sub = [r for r in runs if r["monitor"] == mon and r["replicate"] == rep and r["run_type"] == "fault"]
            dl = [int(r["clean_delay_samples"]) * 0.05 if r["clean_detection"] == "1" else None for r in sub]
            curves.append(curve_from_delays(dl, len(sub)))
        c = np.mean(curves, axis=0)
        ax.plot(TG, c, color=col, ls=ls, lw=1.0, label=lab)
        LOG.append((stem, "curve", mon, round(float(c[np.searchsorted(TG, 8.0)]), 1), round(float(c[-1]), 1)))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_xlabel("Time after fault onset (h)")
    ax.set_ylabel("Clean detection (%)")
    ax.legend(loc=legloc, handlelength=2.2)
    title(ax, "(b)")
    save(fig, stem)


# ------------------------------------------------------------------ S5 (Validation 2, Fault 15 pools)
def s5():
    st = rd(SRC.joinpath("S5/FigureS3_step_curves.csv"))
    far = rd(SRC.joinpath("S5/FigureS3_normal_FAR.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 60 * MM), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.9, wspace=0.3)
    ax = axs[0]
    for conf, col, ls in (("Mean-shift pool, Squared", BLACK, (0, (3, 1.5))), ("Second-moment pool, Squared", RED, "-")):
        for s in sorted({r["state"] for r in st}):
            sub = sorted((float(r["delay_h"]), float(r["cumulative_clean_percent"])) for r in st
                         if r["state"] == s and r["configuration"] == conf)
            ax.step([x[0] for x in sub] + [40], [x[1] for x in sub] + [sub[-1][1]], where="post", color=col, ls=ls,
                    lw=0.9)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_xlabel("Time after fault onset (h)")
    ax.set_ylabel("Clean detection (%)")
    ax.legend(handles=[Line2D([], [], color=RED, label="Second-moment pool (full-pool monitor)"),
                       Line2D([], [], color=BLACK, ls=(0, (3, 1.5)), label="Mean-shift pool (C3 comparator)")],
              loc="upper left", handlelength=2.4)
    title(ax, "(a)")
    ax = axs[1]
    for conf, mfc, dx in (("Second-moment pool, Squared", RED, -0.12), ("Mean-shift pool, Squared", "white", 0.12)):
        sub = sorted((int(r["state"]), float(r["normal_FAR_percent"])) for r in far if r["configuration"] == conf)
        ax.plot([x[0] + dx for x in sub], [x[1] for x in sub], "o", ms=4.2, mfc=mfc,
                mec=RED if mfc == RED else BLACK, mew=0.9)
        LOG.append(("S5", conf, [x[1] for x in sub]))
    ax.axhline(5, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, 12)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlabel("Training replicate")
    ax.set_ylabel("Normal runs with alarm (%)")
    title(ax, "(b)")
    save(fig, "Figure_S5")


# ------------------------------------------------------------------ S6 (score moments)
def s6():
    r = rd(SRC.joinpath("S6/02_score_moments.csv"))
    cats = [("3", "full monitor", "Fault 3, full-pool"), ("3", "FDA-CUSUM", "Fault 3, FDA-CUSUM"),
            ("15", "full monitor", "Fault 15, full-pool"), ("15", "variance-ratio reference", "Fault 15, variance ratio"),
            ("15", "FDA-CUSUM", "Fault 15, FDA-CUSUM"), ("9", "full monitor", "Fault 9, full-pool")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 58 * MM), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.19, top=0.9, wspace=0.08)
    for ax, key, ref, lab, xl in ((axs[0], "delta", 0.05, r"|$\boldsymbol{\delta}$| (normal standard deviations)", (-0.01, 0.3)),
                                  (axs[1], "omega", 1 + 0.05 * math.sqrt(2), r"$\boldsymbol{\omega}$ (fault / normal variance)", (0.98, 1.22))):
        for i, (f, m, name) in enumerate(cats):
            v = [abs(float(x[key])) if key == "delta" else float(x[key]) for x in r if x["fault"] == f and x["monitor"] == m]
            y = len(cats) - 1 - i
            full = m == "full monitor"
            ax.plot(v, np.full(len(v), y) + np.linspace(-0.12, 0.12, len(v)), "o", ms=3.6,
                    mfc=RED if full else "white", mec=RED if full else BLUE, mew=0.9)
            LOG.append(("S6", key, name, round(min(v), 3), round(max(v), 3)))
        ax.axvline(ref, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
        ax.set_xlim(*xl)
        ax.set_xlabel(lab)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c[2] for c in cats][::-1])
        ax.tick_params(axis="y", which="both", right=False)
    title(axs[0], "(a)")
    title(axs[1], "(b)")
    axs[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.6, mfc=RED, mec=RED, label="Full-pool monitor"),
                           Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=BLUE, label="Supervised reference")],
                  loc="lower right")
    save(fig, "Figure_S6")


# ------------------------------------------------------------------ S10 (fewer fitting runs), from Table S31
def s10():
    rows = rd(TABLES.joinpath("Table_S31.csv"))
    blocks = {}
    cur = None
    for r in rows:
        k = r["Fault, reference"].strip()
        if k:
            cur = k
        med, iqr = r["40 h clean detection (%), median [IQR]"].split(" [")
        lo, hi = iqr.rstrip("]").split(", ")
        rlo, rhi = r["40 h range (%)"].split("–")
        blocks.setdefault(cur, []).append((int(r["n"]), float(med), float(lo), float(hi), float(rlo), float(rhi)))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 60 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.9, wspace=0.25)
    for ax, key, orig, ttl in ((axs[0], "3, FDA", 97.5, "(a) Fault 3, FDA-CUSUM"), (axs[1], "15, VR", 66.3, "(b) Fault 15, variance ratio")):
        b = sorted(blocks[key])
        n = [x[0] for x in b]
        ax.fill_between(n, [x[2] for x in b], [x[3] for x in b], color="#D6E2F2", lw=0, zorder=1)
        for x in b:
            ax.plot([x[0], x[0]], [x[4], x[5]], color=BLUE, lw=0.8, zorder=2)
        ax.plot(n, [x[1] for x in b], "-o", color=BLUE, ms=3.6, zorder=3)
        ax.plot([100], [orig], "s", ms=5, mfc="white", mec=BLACK, mew=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels([str(v) for v in n])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, 102)
        ax.set_xlabel(r"Fault fitting runs, $\boldsymbol{n}$")
        ax.set_ylabel("Clean detection within 40 h (%)")
        title(ax, ttl)
        LOG.append(("S10", key, b))
    axs[1].legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.6, label="Median"),
                           plt.Rectangle((0, 0), 1, 1, color="#D6E2F2", label="Interquartile range"),
                           Line2D([], [], color=BLUE, lw=0.8, label="Range"),
                           Line2D([], [], ls="none", marker="s", ms=5, mfc="white", mec=BLACK, label=r"Original, $n$ = 100")],
                  loc="lower right")
    save(fig, "Figure_S10")


# ------------------------------------------------------------------ S12 (long-run variance ratio)
def s12():
    r = rd(SRC.joinpath("S12/Figure5_block_variance_results.csv"))
    groups = {}
    for x in r:
        groups[(x["dataset"], x["run_type"])] = (float(x["L"]), float(x["ci95_lower"]), float(x["ci95_upper"]))
    names = sorted({k[0] for k in groups})
    LOG.append(("S12 datasets", names))
    order = [d for d in ("historical",) if d in names] + [d for d in names if d not in ("historical", "native")] + \
            [d for d in ("native",) if d in names]
    label = {"historical": "Historical data (first split)", "native": "Recording panel 2, post-update"}
    fig, ax = plt.subplots(figsize=(140 * MM, 48 * MM))
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.25, top=0.95)
    for i, d in enumerate(order):
        y = len(order) - 1 - i
        for rt, mfc in (("normal", "white"), ("fault", BLACK)):
            L, lo, hi = groups[(d, rt)]
            ax.plot([lo, hi], [y + (0.1 if rt == "normal" else -0.1)] * 2, color=BLACK, lw=0.8)
            ax.plot(L, y + (0.1 if rt == "normal" else -0.1), "o", ms=4.2, mfc=mfc, mec=BLACK, mew=0.9)
        LOG.append(("S12", d, groups[(d, "normal")], groups[(d, "fault")]))
    ax.axvline(1, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3.2)
    ax.set_xticks([0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label.get(d, "Recording panel 2, controller-input") for d in order][::-1])
    ax.tick_params(axis="y", which="both", right=False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"Long-run variance ratio, $\boldsymbol{L}$")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=BLACK, label="Normal runs"),
                       Line2D([], [], ls="none", marker="o", ms=4.2, mfc=BLACK, mec=BLACK, label="Fault 9 runs")],
              loc="lower right")
    save(fig, "Figure_S12")


# ------------------------------------------------------------------ S13 (Fault 9 and the recording convention)
def s13():
    fig, axs = plt.subplots(2, 2, figsize=(W2, 120 * MM))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.95, wspace=0.25, hspace=0.42)
    # (a) historical scores, 1 h backward moving average
    rows = rd(SRC.joinpath("S13/a_historical/figure7a_all_historical_scores.csv"))
    series = {}
    for r in rows:
        series.setdefault((r["run_type"], int(r["run_id"])), []).append((int(r["record"]), float(r["u_t"])))

    def ma(v):
        v = np.array([x[1] for x in sorted(v)])
        c = np.cumsum(np.insert(v, 0, 0.0))
        out = np.full(len(v), np.nan)
        out[19:] = (c[20:] - c[:-20]) / 20
        return out

    normals = np.array([ma(v) for k, v in series.items() if k[0] == "normal"])
    lo = np.nanpercentile(normals, 2.5, axis=0)
    hi = np.nanpercentile(normals, 97.5, axis=0)
    t = np.arange(1, normals.shape[1] + 1) * 0.05
    f = ma(series[("fault", 144)])
    ax = axs[0, 0]
    ax.fill_between(t, lo, hi, color="#DDDDDD", lw=0, zorder=1)
    ax.plot(t, f, color=BLACK, lw=0.9, zorder=2)
    ax.axhline(0, color=GREY, lw=0.5, zorder=0)
    onset(ax)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Score, 1-h moving average")
    title(ax, "(a)")
    LOG.append(("S13a", normals.shape, round(float(np.nanmax(f)), 3)))
    # (b) normal score variance under the recording policies
    pooled = {x["policy"]: float(x["score_variance"]) for x in rd(SRC.joinpath("S13/b_score_variance/POOLED_STATISTICS.csv"))
              if x["window"] == "post800"}
    hist = [float(x["variance_u_ddof1"]) for x in rd(SRC.joinpath("S13/b_score_variance/historical_and_panel2_moments.csv"))
            if x["dataset"] == "historical" and x["run_type"] == "normal"][0]
    items = [("Historical\nnormal runs", hist, "white", GREY), ("Post-update", pooled["native"], BLACK, BLACK),
             ("XMEAS12\nand 15", pooled["pair7_8_only"], GREY, GREY), ("XMEAS17", pooled["pair11_only"], GREY, GREY),
             ("Three\nloops", pooled["three_pairs_only"], GREY, GREY), ("All 41\n(controller-input)", pooled["all_fast_input"], RED, RED)]
    ax = axs[0, 1]
    for i, (lab, v, mfc, mec) in enumerate(items):
        ax.plot(i, v, "o", ms=5, mfc=mfc, mec=mec, mew=1.0)
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(1, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[0] for x in items], fontsize=6.8)
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylim(0, 4)
    ax.set_ylabel("Normal score variance")
    title(ax, "(b)")
    LOG.append(("S13b", [(x[0].replace("\n", " "), round(x[1], 3)) for x in items]))
    # (c) recording panel 2, run 071, relearned monitors
    rows = rd(SRC.joinpath("S13/c_recording_panel2/fig8a_cusum_recordwise.csv"))
    ax = axs[1, 0]
    for rec, col, ls, lab in (("controller-input", RED, "-", "Controller-input"), ("post-update", BLACK, (0, (3, 1.5)), "Post-update")):
        sub = sorted((int(r["record"]), float(r["c_max_over_h"])) for r in rows
                     if r["run_type"] == "fault" and r["recording"] == rec)
        ax.plot([s[0] * 0.05 for s in sub], [s[1] for s in sub], color=col, ls=ls, lw=1.0, label=lab)
    ax.axhline(1, color=GREY, lw=0.7, ls=(0, (4, 2)))
    onset(ax)
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("CUSUM statistic / limit")
    ax.legend(loc="upper left", handlelength=2.2)
    title(ax, "(c)")
    # (d) cumulative clean detection, panel 2
    rows = rd(SRC.joinpath("S13/d_detection/S13d_paired_first_alarm_records.csv"))
    ax = axs[1, 1]
    for rec, col, ls in (("controller-input", RED, "-"), ("post-update", BLACK, (0, (3, 1.5)))):
        sub = [r for r in rows if r["run_type"] == "fault" and r["recording"] == rec]
        c = curve_from_delays([delay_from_record(r["first_alarm_record"]) for r in sub], len(sub))
        ax.plot(TG, c, color=col, ls=ls, lw=1.0)
        LOG.append(("S13d", rec, len(sub), round(float(c[-1]) * len(sub) / 100)))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_xlabel("Time after fault onset (h)")
    ax.set_ylabel("Clean detection (%)")
    title(ax, "(d)")
    save(fig, "Figure_S13")


# ------------------------------------------------------------------ S14 (paired normal simulations)
def s14():
    rows = rd(PAIRED)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.9, wspace=0.28)
    ax = axs[0]
    for rec, col, al, z in (("PU", BLACK, 0.35, 2), ("CI", RED, 1.0, 3)):
        x = [float(r["XMEAS15"]) for r in rows if r["recording"] == rec and int(r["record"]) >= 161]
        y = [float(r["XMV8"]) for r in rows if r["recording"] == rec and int(r["record"]) >= 161]
        ax.scatter(x, y, s=1.0, c=col, alpha=al, lw=0, rasterized=True, zorder=z)
        LOG.append(("S14a", rec, len(x), round(float(np.corrcoef(x, y)[0, 1]), 4)))
    ax.set_xlabel("XMEAS15 (%)")
    ax.set_ylabel("XMV8 (%)")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.5, mfc=RED, mec=RED, label="Controller-input"),
                       Line2D([], [], ls="none", marker="o", ms=3.5, mfc=GREY, mec=GREY, label="Post-update")],
              loc="upper left", handletextpad=0.2)
    title(ax, "(a)")
    v = rd(SRC.joinpath("S14b/S14b_eight_runs_three_views_score_variance.csv"))
    views = []
    for r in v:
        if r["view"] not in views:
            views.append(r["view"])
    LOG.append(("S14b views", views))
    order = [w for w in ("native", "pair3") if w in views] + [w for w in views if w not in ("native", "pair3")]
    ax = axs[1]
    for run in sorted({r["run"] for r in v}):
        y = [float([r for r in v if r["run"] == run and r["view"] == w][0]["var"]) for w in order]
        ax.plot(range(len(order)), y, "-o", color=LGREY, mfc=GREY, mec=GREY, ms=3, lw=0.8)
    ax.axhline(1, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.set_xticks(range(len(order)))
    lab = {"native": "Post-update", "pair3": "Three loops\nreplaced"}
    ax.set_xticklabels([lab.get(w, "Controller-input") for w in order])
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlim(-0.4, len(order) - 0.6)
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("Normal score variance")
    title(ax, "(b)")
    save(fig, "Figure_S14")


# ------------------------------------------------------------------ S16 and S17 (cumulative clean detection, CI vs PU)
def ci_pu_panel(ax, rows, ttl, n_expected):
    for rec, col, ls, lab in (("controller-input", RED, "-", "Controller-input"), ("post-update", BLACK, (0, (3, 1.5)), "Post-update")):
        sub = [r for r in rows if r["run_type"] == "fault" and r["recording"] == rec]
        assert len(sub) == n_expected, (ttl, rec, len(sub))
        dl = [delay_from_record(r["first_alarm_record"]) for r in sub]
        c = curve_from_delays(dl, len(sub))
        ax.plot(TG, c, color=col, ls=ls, lw=1.0, label=lab)
        cnt = lambda h: sum(1 for d in dl if d is not None and d <= h + 1e-9)
        LOG.append((ttl, rec, cnt(2), cnt(8), cnt(40), len(sub)))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 102)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_xlabel("Time after fault onset (h)")
    ax.set_ylabel("Clean detection (%)")
    title(ax, ttl)


def s16():
    p1 = rd(SRC.joinpath("S16_S17/S16_Fault9_panel1_fixed_first_alarm.csv"))
    p2 = rd(SRC.joinpath("S16_S17/S16_Fault9_panel2_fixed_first_alarm.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 60 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.9, wspace=0.25)
    ci_pu_panel(axs[0], p1, "(a) Recording panel 1", 75)
    ci_pu_panel(axs[1], p2, "(b) Recording panel 2", 120)
    axs[0].legend(loc="upper left", handlelength=2.2)
    save(fig, "Figure_S16")


def s17():
    r = rd(SRC.joinpath("S16_S17/S17_Fault3_Fault15_first_alarm.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 112 * MM))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.95, wspace=0.25, hspace=0.45)
    spec = [("3", "full monitor", "(a) Fault 3, full-pool monitor"), ("3", "FDA-CUSUM", "(b) Fault 3, FDA-CUSUM"),
            ("15", "full monitor", "(c) Fault 15, full-pool monitor"), ("15", "variance-ratio reference", "(d) Fault 15, variance ratio")]
    for ax, (f, m, ttl) in zip(axs.flat, spec):
        ci_pu_panel(ax, [x for x in r if x["fault"] == f and x["monitor"] == m], ttl, 120)
    axs[0, 0].legend(loc="lower right", handlelength=2.2)
    save(fig, "Figure_S17")


# ------------------------------------------------------------------ S18 (partial sharing: normal alarms)
def s18():
    rows = rd(TABLES.joinpath("Table_S61.csv"))
    xs, ks = [], []
    for r in rows:
        if r["Setting"].startswith("ρ"):
            xs.append(int(r["Shared components (of 12)"]) / 12)
            ks.append(int(r["Normal alarms (of 400)"]))
    fig, ax = plt.subplots(figsize=(90 * MM, 58 * MM))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.2, top=0.95)
    p = [100 * k / 400 for k in ks]
    lo = [p[i] - wilson(k, 400)[0] for i, k in enumerate(ks)]
    hi = [wilson(k, 400)[1] - p[i] for i, k in enumerate(ks)]
    ax.errorbar(xs, p, yerr=[lo, hi], fmt="-o", color=BLACK, ms=4, elinewidth=0.8, capsize=2, capthick=0.8)
    ax.axhline(5, color=GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0, 16)
    ax.set_xlabel(r"Shared fraction, $\boldsymbol{\rho}$")
    ax.set_ylabel("Normal runs with alarm (%)")
    LOG.append(("S18", ks))
    save(fig, "Figure_S18")


# ------------------------------------------------------------------ S19–S21 (detection against false alarms)
def far_figure(panel, kind, contrasts, titles, n_fault, stem):
    rows = rd(SRC.joinpath("S19_S21/results/curves_mean.csv"))
    fig, axs = plt.subplots(2, 3, figsize=(W2, 110 * MM), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.95, wspace=0.27, hspace=0.25)
    for j, (con, ttl) in enumerate(zip(contrasts, titles)):
        for i, (col_y, ylab) in enumerate((("clean40_pct", "Clean detection within 40 h (%)"), ("clean8_pct", "Clean detection within 8 h (%)"))):
            ax = axs[i, j]
            for arm, col, ls in (("favoured", RED, "-"), ("comparator", BLACK, (0, (3, 1.5)))):
                sub = sorted((float(r["far_pct"]), float(r[col_y])) for r in rows
                             if r["panel"] == panel and r["contrast"] == con and r["arm"] == arm and r["kind"] == kind)
                sub = [s for s in sub if s[0] <= 15.0001]
                ax.plot([s[0] for s in sub], [s[1] for s in sub], color=col, ls=ls, lw=1.0)
                o = [r for r in rows if r["panel"] == panel and r["contrast"] == con and r["arm"] == arm and r["kind"] == "original"]
                if o and kind == "calibration_rank" or o and kind == "matched_evaluation":
                    ax.plot(float(o[0]["far_pct"]), float(o[0][col_y]), "o" if arm == "favoured" else "s", ms=4.5,
                            mfc=RED if arm == "favoured" else "white", mec=RED if arm == "favoured" else BLACK, mew=0.9, zorder=5)
                    LOG.append((stem, con, arm, col_y, o[0]["far_pct"], o[0][col_y]))
            ax.axvline(5, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
            ax.set_xlim(0, 15)
            ax.set_xticks([0, 5, 10, 15])
            ax.set_ylim(-2, 102)
            if i == 1:
                ax.set_xlabel("Normal runs with alarm (%)")
            if j == 0:
                ax.set_ylabel(ylab)
            title(ax, f"({'abcdef'[3 * i + j]}) {ttl}")
    axs[0, 0].legend(handles=[Line2D([], [], color=RED, label="Favoured variant"),
                              Line2D([], [], color=BLACK, ls=(0, (3, 1.5)), label="Comparator")],
                     loc="center right", handlelength=2.2)
    save(fig, stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY.parent / "replay_output" / "extended_si")
    OUT = parser.parse_args().output
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUTS / "supplementary_sources.zip") as archive:
        SRC = zipfile.Path(archive)
        s2s3(3, "figure4_all_monitors_per_run.csv",
             [("mean_pool_signed", RED, "-", "Full-pool monitor (signed)"), ("same_mean_signed_axis_squared", BLACK, (0, (3, 1.5)), "Squared, same direction (C1)")],
             {"full-pool signed": (RED, "-", "Full-pool monitor"), "C1 same-direction squared": (BLACK, (0, (3, 1.5)), "C1 comparator"),
              "FDA-CUSUM": (BLUE, "-", "FDA-CUSUM"), "selected multivariate chart": (GREY, "-", "Multivariate chart"),
              "Q-CUSUM": (GREY, (0, (1, 1.2)), "Q-CUSUM")}, "Figure_S2", legloc="center right")
        s2s3(15, "figure5_all_monitors_per_run.csv",
             [("moment_pool_squared", RED, "-", "Full-pool monitor (second moment)"), ("mean_pool_squared", BLACK, (0, (3, 1.5)), "Mean-shift direction (C3)")],
             {"full-pool moment squared": (RED, "-", "Full-pool monitor"), "C3 mean-direction squared": (BLACK, (0, (3, 1.5)), "C3 comparator"),
              "variance-ratio reference": (BLUE, "-", "Variance-ratio reference"), "FDA-CUSUM": (BLUE, (0, (3, 1.5)), "FDA-CUSUM"),
              "selected multivariate chart": (GREY, "-", "Multivariate chart"), "Q-CUSUM": (GREY, (0, (1, 1.2)), "Q-CUSUM")}, "Figure_S3", legloc="upper left")
        s5()
        s6()
        s10()
        s12()
        s13()
        s14()
        s16()
        s17()
        s18()
        far_figure("confirmation", "calibration_rank", ["C1", "C3", "F9_recording"],
                   ["C1, Fault 3", "C3, Fault 15", "Fault 9, recording"], 200, "Figure_S19")
        far_figure("validation2", "calibration_rank", ["C1", "C2", "C3"], ["C1, Fault 3", "C2, Fault 9", "C3, Fault 15"], 250, "Figure_S20")
        far_figure("confirmation", "matched_evaluation", ["C1", "C3", "F9_recording"],
                   ["C1, Fault 3", "C3, Fault 15", "Fault 9, recording"], 200, "Figure_S21")
        with open(os.path.join(OUT, "redraw_log.txt"), "w") as fh:
            for x in LOG:
                fh.write(repr(x) + "\n")
        print("log lines", len(LOG))
