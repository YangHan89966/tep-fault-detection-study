#!/usr/bin/env python3
"""Build P0-F05: Primary reserved-panel operating trade-offs.

Production-support only: reads existing frozen package tables and creates display assets.
No experiment execution, refitting, tuning, or detector-state changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

MM = 1.0 / 25.4
PT_PER_MM = 72.0 / 25.4
HERE = Path(__file__).resolve().parent

METHOD_ORDER = [
    "Q-CUSUM",
    "T2-CUSUM",
    "Joint CUSUM",
    "LDA residual",
    "Family All-52",
    "Family Top-1",
    "Proposed Top-5",
]

# Okabe-Ito-inspired, colorblind-safe method mapping. Proposed is vermillion and
# additionally distinguished by marker/line weight, so color is not the sole cue.
METHOD_COLOR = {
    "Q-CUSUM": "#0072B2",
    "T2-CUSUM": "#56B4E9",
    "Joint CUSUM": "#009E73",
    "LDA residual": "#CC79A7",
    "Family All-52": "#8C6D31",
    "Family Top-1": "#E69F00",
    "Proposed Top-5": "#D55E00",
}
METHOD_MARKER = {
    "Q-CUSUM": "o",
    "T2-CUSUM": "s",
    "Joint CUSUM": "D",
    "LDA residual": "P",
    "Family All-52": "v",
    "Family Top-1": "^",
    "Proposed Top-5": "*",
}

DISPLAY_RENAME = {"T²-CUSUM": "T2-CUSUM"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    if x is None or x == "":
        return None
    return float(x)


def load_data(root: Path):
    summary_path = root / "derived/PRIMARY_SEVEN_METHOD_EARLY_DETECTION_AND_DELAY_SUMMARY.csv"
    ci_path = root / "core_sources/E1H_statistics/results/tables/23_reserved_primary_summary_with_ci.csv"
    summary = read_csv(summary_path)
    ci_rows = read_csv(ci_path)

    ci_map = {}
    for r in ci_rows:
        short = DISPLAY_RENAME.get(r["method_short"], r["method_short"])
        if r["endpoint_key"] == "alarm_delay_median":
            ci_map[short] = (float(r["ci_95_low"]), float(r["ci_95_high"]))

    rows = []
    for r in summary:
        short = DISPLAY_RENAME.get(r["method_short"], r["method_short"])
        low, high = ci_map[short]
        rows.append({
            "method_key": r["method_key"],
            "method_short": short,
            "method_class": r["method_class"],
            "method_order": int(r["method_order"]),
            "n_variables": fnum(r["n_variables"]),
            "macro_normal_run_level_far_pct": 100.0 * float(r["macro_normal_run_level_far"]),
            "macro_detection_within_15_min_pct": 100.0 * float(r["macro_detection_within_15_min"]),
            "macro_detection_within_30_min_pct": 100.0 * float(r["macro_detection_within_30_min"]),
            "macro_detection_within_60_min_pct": 100.0 * float(r["macro_detection_within_60_min"]),
            "macro_detection_within_120_min_pct": 100.0 * float(r["macro_detection_within_120_min"]),
            "macro_full_horizon_run_detection_pct": 100.0 * float(r["macro_full_horizon_run_detection"]),
            "macro_mean_post_fault_threshold_occupancy_pct": 100.0 * float(r["macro_mean_post_fault_threshold_occupancy"]),
            "macro_median_post_fault_threshold_occupancy_pct": 100.0 * float(r["macro_median_post_fault_threshold_occupancy"]),
            "macro_median_of_fault_penalized_median_delay_min": float(r["macro_median_of_fault_penalized_median_delay_min"]),
            "macro_mean_all_run_penalized_delay_min": float(r["macro_mean_all_run_penalized_delay_min"]),
            "delay_median_ci_95_low_min": low,
            "delay_median_ci_95_high_min": high,
        })
    rows.sort(key=lambda r: r["method_order"])
    return rows, summary_path, ci_path


def load_local_source(path: Path):
    rows = read_csv(path)
    int_fields = {"method_order"}
    numeric_fields = {
        "n_variables", "macro_normal_run_level_far_pct", "macro_detection_within_15_min_pct",
        "macro_detection_within_30_min_pct", "macro_detection_within_60_min_pct",
        "macro_detection_within_120_min_pct", "macro_full_horizon_run_detection_pct",
        "macro_mean_post_fault_threshold_occupancy_pct", "macro_median_post_fault_threshold_occupancy_pct",
        "macro_median_of_fault_penalized_median_delay_min", "macro_mean_all_run_penalized_delay_min",
        "delay_median_ci_95_low_min", "delay_median_ci_95_high_min",
    }
    out=[]
    for r in rows:
        rr=dict(r)
        for k in int_fields: rr[k]=int(rr[k])
        for k in numeric_fields: rr[k]=fnum(rr.get(k))
        out.append(rr)
    out.sort(key=lambda r:r["method_order"])
    return out

def configure_mpl():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_three_formats(fig, stem: Path, dpi=600):
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)


def panel_a(rows, out: Path):
    # One standalone chart: FAR vs mean threshold occupancy.
    # Method mapping is carried by the shared colors/markers and the Panel-B legend;
    # omitting point labels here avoids overlap in the high-occupancy cluster.
    fig = plt.figure(figsize=(84 * MM, 72 * MM))
    ax = fig.add_axes([0.17, 0.17, 0.80, 0.75])
    for r in rows:
        m = r["method_short"]
        x = r["macro_normal_run_level_far_pct"]
        y = r["macro_mean_post_fault_threshold_occupancy_pct"]
        size = 80 if m == "Proposed Top-5" else 34
        lw = 0.9 if m == "Proposed Top-5" else 0.5
        ax.scatter(x, y, s=size, marker=METHOD_MARKER[m], c=METHOD_COLOR[m], edgecolors="black", linewidths=lw, zorder=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Macro mean normal run-level FAR (%)")
    ax.set_ylabel("Macro mean threshold occupancy (%)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.text(0.0, 1.04, "A", transform=ax.transAxes, fontweight="bold", fontsize=9.5, va="bottom")
    ax.text(1.0, 1.04, "method key in panel B", transform=ax.transAxes, fontsize=6.8, ha="right", va="bottom")
    save_three_formats(fig, out / "panel_A_far_vs_occupancy")

def panel_b(rows, out: Path):
    # One standalone chart: early detection profiles.
    times = np.array([15, 30, 60, 120], dtype=float)
    keys = [
        "macro_detection_within_15_min_pct",
        "macro_detection_within_30_min_pct",
        "macro_detection_within_60_min_pct",
        "macro_detection_within_120_min_pct",
    ]
    fig = plt.figure(figsize=(84 * MM, 72 * MM))
    ax = fig.add_axes([0.17, 0.17, 0.80, 0.75])
    for r in rows:
        m = r["method_short"]
        y = [r[k] for k in keys]
        lw = 2.15 if m == "Proposed Top-5" else 1.15
        ms = 6.5 if m == "Proposed Top-5" else 4.0
        ax.plot(times, y, color=METHOD_COLOR[m], marker=METHOD_MARKER[m], markersize=ms,
                markeredgecolor="black", markeredgewidth=0.35, linewidth=lw, label=m, zorder=4 if m == "Proposed Top-5" else 2)
    ax.set_xlim(10, 125)
    ax.set_ylim(0, 100)
    ax.set_xticks(times)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Time since fault onset (min)")
    ax.set_ylabel("Macro run detection within time (%)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.0, columnspacing=0.8,
              borderaxespad=0.2, labelspacing=0.25)
    ax.text(0.0, 1.04, "B", transform=ax.transAxes, fontweight="bold", fontsize=9.5, va="bottom")
    save_three_formats(fig, out / "panel_B_early_detection")


def panel_c(rows, out: Path):
    # One standalone chart: delay dot-and-interval plot.
    fig = plt.figure(figsize=(178 * MM, 58 * MM))
    ax = fig.add_axes([0.18, 0.22, 0.79, 0.68])
    # Top-to-bottom follows method order, Proposed last at bottom for emphasis.
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        m = r["method_short"]
        est = r["macro_median_of_fault_penalized_median_delay_min"]
        lo = r["delay_median_ci_95_low_min"]
        hi = r["delay_median_ci_95_high_min"]
        ms = 8.5 if m == "Proposed Top-5" else 5.2
        ax.errorbar(est, yi, xerr=np.array([[est - lo], [hi - est]]), fmt=METHOD_MARKER[m],
                    color=METHOD_COLOR[m], markeredgecolor="black", markeredgewidth=0.45,
                    markersize=ms, elinewidth=1.25 if m == "Proposed Top-5" else 1.0,
                    capsize=2.2, zorder=4 if m == "Proposed Top-5" else 2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["method_short"] for r in rows])
    ax.set_xlim(0, 300)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xlabel("Macro median of fault-specific all-run penalized median delay (min)")
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.6)
    ax.text(0.0, 1.05, "C", transform=ax.transAxes, fontweight="bold", fontsize=9.5, va="bottom")
    ax.text(1.0, 1.05, "horizontal bars: 95% fault-cluster bootstrap CI", transform=ax.transAxes,
            fontsize=6.8, ha="right", va="bottom")
    save_three_formats(fig, out / "panel_C_penalized_delay")


def combine_pdf(panel_dir: Path, out_pdf: Path):
    # Combine three independent one-chart PDFs on one publication-sized vector page.
    page_w_mm = 178.0
    gap_mm = 4.0
    top_h_mm = 72.0
    bottom_h_mm = 58.0
    page_h_mm = top_h_mm + gap_mm + bottom_h_mm
    doc = fitz.open()
    page = doc.new_page(width=page_w_mm * PT_PER_MM, height=page_h_mm * PT_PER_MM)
    half_w = (page_w_mm - gap_mm) / 2.0
    rect_a = fitz.Rect(0, 0, half_w * PT_PER_MM, top_h_mm * PT_PER_MM)
    rect_b = fitz.Rect((half_w + gap_mm) * PT_PER_MM, 0, page_w_mm * PT_PER_MM, top_h_mm * PT_PER_MM)
    rect_c = fitz.Rect(0, (top_h_mm + gap_mm) * PT_PER_MM, page_w_mm * PT_PER_MM, page_h_mm * PT_PER_MM)
    for pdf_name, rect in [
        ("panel_A_far_vs_occupancy.pdf", rect_a),
        ("panel_B_early_detection.pdf", rect_b),
        ("panel_C_penalized_delay.pdf", rect_c),
    ]:
        src = fitz.open(panel_dir / pdf_name)
        page.show_pdf_page(rect, src, 0, keep_proportion=True, overlay=True)
        src.close()
    doc.set_metadata({
        "title": "P0-F05 Primary reserved-panel operating trade-offs",
        "subject": "Frozen primary 17-fault TEP panel; production-support visualization only",
    })
    doc.save(out_pdf, garbage=4, deflate=True)
    doc.close()


def combine_png(panel_dir: Path, out_png: Path, dpi=600):
    # Composite the separately rendered charts without creating matplotlib subplots.
    page_w_mm = 178.0
    gap_mm = 4.0
    top_h_mm = 72.0
    bottom_h_mm = 58.0
    px_per_mm = dpi / 25.4
    W = round(page_w_mm * px_per_mm)
    gap = round(gap_mm * px_per_mm)
    top_h = round(top_h_mm * px_per_mm)
    bottom_h = round(bottom_h_mm * px_per_mm)
    H = top_h + gap + bottom_h
    half_w = (W - gap) // 2
    canvas = Image.new("RGB", (W, H), "white")
    for name, box in [
        ("panel_A_far_vs_occupancy.png", (0, 0, half_w, top_h)),
        ("panel_B_early_detection.png", (half_w + gap, 0, W, top_h)),
        ("panel_C_penalized_delay.png", (0, top_h + gap, W, H)),
    ]:
        img = Image.open(panel_dir / name).convert("RGB")
        target_w = box[2] - box[0]
        target_h = box[3] - box[1]
        scale = min(target_w / img.width, target_h / img.height)
        nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x = box[0] + (target_w - nw) // 2
        y = box[1] + (target_h - nh) // 2
        canvas.paste(img, (x, y))
    canvas.save(out_png, dpi=(dpi, dpi), optimize=True)


def write_source_csv(rows, out_csv: Path):
    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_readme(out: Path, input_paths, rows):
    proposed = next(r for r in rows if r["method_short"] == "Proposed Top-5")
    text = f"""# P0-F05 - Primary reserved-panel operating trade-offs

**Evidence class:** Primary frozen reserved-fault analysis (Step-22-R / Step-23 summaries)  
**Intended placement:** Main manuscript, Section 4.2  
**Production status:** Figure asset only; no manuscript overwrite

## Draft caption
**Figure X. Frozen primary reserved-panel operating trade-offs across seven prespecified methods.** (A) Macro mean normal run-level false-alarm rate (FAR) versus macro mean post-fault threshold occupancy across the fixed 17 reserved TEP faults. (B) Macro run detection within 15, 30, 60, and 120 min after fault onset. (C) Macro median of fault-specific all-run penalized median alarm delay; horizontal bars show 95% fault-cluster bootstrap intervals (20,000 resamples). Proposed Top-5 is marked distinctly for identification. The three panels report separate operating endpoints and are not a composite score or universal method ranking. Dynamic U4 CVA results are intentionally excluded because they are a separate post-freeze comparison.

## Provenance
- `{input_paths[0].as_posix()}`: seven-method frozen primary FAR, early-detection, full-horizon detection, threshold-occupancy, and delay summaries.
- `{input_paths[1].as_posix()}`: endpoint-specific 95% fault-cluster intervals for the median-delay panel.
- Display-only derivation: proportions in the first source were multiplied by 100 for percentage axes; no experiment was rerun and no detector state, tuning parameter, method ordering, or endpoint definition was changed.
- Visualization-only mapping: a fixed colorblind-safe method palette and method markers were assigned; Proposed Top-5 also receives a star / heavier stroke so color is not the sole identifier.

## Numerical anchor check for Proposed Top-5
- Macro normal run-level FAR: {proposed['macro_normal_run_level_far_pct']:.4f}%.
- Macro mean threshold occupancy: {proposed['macro_mean_post_fault_threshold_occupancy_pct']:.4f}%.
- Detection within 15 / 30 / 60 / 120 min: {proposed['macro_detection_within_15_min_pct']:.2f}% / {proposed['macro_detection_within_30_min_pct']:.2f}% / {proposed['macro_detection_within_60_min_pct']:.2f}% / {proposed['macro_detection_within_120_min_pct']:.2f}%.
- Macro median penalized delay: {proposed['macro_median_of_fault_penalized_median_delay_min']:.1f} min (95% CI {proposed['delay_median_ci_95_low_min']:.1f}-{proposed['delay_median_ci_95_high_min']:.1f}).

## Claim boundary
This figure supports an **endpoint-specific, TEP-scoped operating trade-off** across the seven frozen primary methods on the fixed 17-fault reserved panel. It can show that Proposed Top-5 combines sparse five-coordinate monitoring with complete full-horizon coverage and a competitive FAR/occupancy/delay profile, while other methods lead some endpoints. It **cannot** support a composite winner score, universal superiority, external industrial validation, zero-shot/unknown-fault generalization, physical-sensor robustness, or replacement of the frozen primary state by U1-U4 or system-layer analyses.

## Files
- `P0-F05_primary_reserved_panel_operating_tradeoffs.pdf`: final vector composite at 178 mm width.
- `P0-F05_primary_reserved_panel_operating_tradeoffs.png`: 600-dpi composite preview.
- `P0-F05_source_data.csv`: clean display-source table.
- `make_P0_F05.py`: reproducible build script.
- `panels/*.svg`: editable vector panel sources; matching PDF/PNG panel renders are retained for provenance/QA.
"""
    (out / "P0-F05_README_caption_provenance_claim_boundary.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", type=Path, default=None,
                    help="Optional exact source archive root. If omitted, rebuild from local P0-F05_source_data.csv.")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()
    out = args.outdir.resolve()
    panel_dir = out / "panels"
    out.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    configure_mpl()
    if args.package_root is None:
        rows = load_local_source(HERE / "P0-F05_source_data.csv")
        summary_path = HERE / "P0-F05_source_data.csv"
        ci_path = None
    else:
        root = args.package_root.resolve()
        rows, summary_path, ci_path = load_data(root)
    if [r["method_short"] for r in rows] != METHOD_ORDER:
        raise RuntimeError("Unexpected method ordering/names in source table")

    panel_a(rows, panel_dir)
    panel_b(rows, panel_dir)
    panel_c(rows, panel_dir)
    final_pdf = out / "P0-F05_primary_reserved_panel_operating_tradeoffs.pdf"
    final_png = out / "P0-F05_primary_reserved_panel_operating_tradeoffs.png"
    combine_pdf(panel_dir, final_pdf)
    combine_png(panel_dir, final_png, dpi=600)

    if args.package_root is not None:
        root = args.package_root.resolve()
        write_source_csv(rows, out / "P0-F05_source_data.csv")
        write_readme(out, [summary_path.relative_to(root), ci_path.relative_to(root)], rows)
        inputs = {
            "package_root": str(root),
            "input_sha256": {
                str(summary_path.relative_to(root)): sha256(summary_path),
                str(ci_path.relative_to(root)): sha256(ci_path),
            },
            "production_constraints": {"new_experiment": False,"retuning": False,"detector_state_change": False,"manuscript_overwrite": False},
        }
        (out / "P0-F05_provenance.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    files = sorted(q for q in out.rglob("*") if q.is_file() and q.name != "SHA256SUMS.txt")
    with (out / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for q in files: f.write(f"{sha256(q)}  {q.relative_to(out).as_posix()}\n")
    print(json.dumps({"outdir": str(out),"rows": len(rows),"local_source_mode": args.package_root is None,"final_pdf": str(final_pdf),"final_png": str(final_png)}, indent=2))

if __name__ == "__main__":
    main()
