"""Create the Step-18 applicability diagnostic without rereading raw RData.

This plotting-only companion is intentionally separate from Step 18.  The
prespecified Top-5 performance plot is undefined when fewer than five
direction-stable residuals are available.  In that case, this script documents
the applicability-gate result rather than drawing nonexistent performance.

Required inputs (created by Step 18):
    results/tables/18_faults9_15_screening_all_variables.csv
    results/tables/18_faults9_15_screening_transfer_summary.csv
    results/tables/18_faults9_15_all_method_results.csv

Outputs:
    results/figures/fig18a_faults9_15_applicability_diagnostic.png
    results/figures/fig18a_faults9_15_applicability_diagnostic.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

SCREEN_PATH = TABLE_DIR / "18_faults9_15_screening_all_variables.csv"
SUMMARY_PATH = TABLE_DIR / "18_faults9_15_screening_transfer_summary.csv"
RESULTS_PATH = TABLE_DIR / "18_faults9_15_all_method_results.csv"

FAULTS = [9, 15]
TOP_N_DISPLAY = 5


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step-18 table is missing: {path}\n"
            "Run Step 18 first, then run this plotting-only script."
        )
    return path


def standardized_bootstrap_interval(frame: pd.DataFrame) -> pd.DataFrame:
    """Express the raw-shift bootstrap interval on the Hedges-g scale."""
    output = frame.copy()
    pooled = output["pooled_run_mean_std"].to_numpy(dtype=float)
    cohen = output["cohen_d"].to_numpy(dtype=float)
    hedges = output["hedges_g"].to_numpy(dtype=float)

    correction = np.ones(len(output), dtype=float)
    usable = np.isfinite(cohen) & np.isfinite(hedges) & (np.abs(cohen) > 0)
    correction[usable] = hedges[usable] / cohen[usable]

    if np.any(~np.isfinite(pooled)) or np.any(pooled <= 0):
        raise ValueError("Invalid pooled run-mean standard deviation in screening table")

    output["ci_g_low"] = (
        output["bootstrap_ci_low"].to_numpy(dtype=float) / pooled * correction
    )
    output["ci_g_high"] = (
        output["bootstrap_ci_high"].to_numpy(dtype=float) / pooled * correction
    )
    return output


def plot_fault_screen(axis, screen: pd.DataFrame, fault_id: int, panel: str) -> None:
    current = screen[
        (screen["fault_id"] == fault_id)
        & (screen["screen_source"] == "faulty_testing_development")
    ].copy()
    if len(current) != 52:
        raise ValueError(
            f"Fault {fault_id}: expected 52 development screening rows, found {len(current)}"
        )

    current = standardized_bootstrap_interval(current)
    current = current.sort_values("abs_hedges_g", ascending=False).head(
        TOP_N_DISPLAY
    )
    current = current.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(current), dtype=float)
    point = current["hedges_g"].to_numpy(dtype=float)
    low = current["ci_g_low"].to_numpy(dtype=float)
    high = current["ci_g_high"].to_numpy(dtype=float)
    errors = np.vstack([point - low, high - point])
    stable = current["direction_stable_95ci"].astype(bool).to_numpy()
    colors = np.where(stable, "#D95F59", "#6F7C80")

    for index in range(len(current)):
        axis.errorbar(
            point[index],
            y[index],
            xerr=errors[:, index : index + 1],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            linewidth=1.6,
            markersize=5,
        )
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(current["variable"].astype(str))
    axis.set_xlabel("Standardized run-mean shift (Hedges' g)")
    stable_count = int(
        screen.loc[
            (screen["fault_id"] == fault_id)
            & (screen["screen_source"] == "faulty_testing_development"),
            "direction_stable_95ci",
        ].astype(bool).sum()
    )
    axis.set_title(
        f"({panel}) Fault {fault_id} development screen\n"
        f"{stable_count}/52 direction-stable residuals"
    )
    axis.grid(axis="x", alpha=0.22)


def plot_stable_counts(axis, summary: pd.DataFrame) -> None:
    summary = summary.set_index("fault_id").reindex(FAULTS)
    if summary.index.isna().any() or summary[
        ["development_stable_variable_count", "companion_stable_variable_count"]
    ].isna().any().any():
        raise ValueError("Screening transfer summary is incomplete for Faults 9 and 15")

    x = np.arange(len(FAULTS), dtype=float)
    width = 0.34
    development = summary["development_stable_variable_count"].to_numpy(dtype=float)
    companion = summary["companion_stable_variable_count"].to_numpy(dtype=float)

    bars_dev = axis.bar(
        x - width / 2,
        development,
        width,
        label="Testing development (prespecified)",
        color="#4C78A8",
    )
    bars_comp = axis.bar(
        x + width / 2,
        companion,
        width,
        label="Training companion (post-freeze diagnostic)",
        color="#D8A657",
        hatch="//",
        edgecolor="#8D6A2C",
    )
    axis.axhline(
        5,
        color="#B23A3A",
        linestyle="--",
        linewidth=1.3,
        label="Five residuals required",
    )
    axis.bar_label(bars_dev, fmt="%.0f", padding=3)
    axis.bar_label(bars_comp, fmt="%.0f", padding=3)
    axis.set_xticks(x)
    axis.set_xticklabels([f"Fault {fault_id}" for fault_id in FAULTS])
    axis.set_ylabel("Direction-stable residual count")
    axis.set_ylim(0, max(8.0, float(np.nanmax(companion)) + 1.5))
    axis.set_title("(c) Applicability-gate outcome")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="upper center", ncol=3, frameon=True, fontsize=8.5)


def validate_protocol_status(results: pd.DataFrame) -> None:
    rows = results[
        (results["fault_id"].isin(FAULTS))
        & (results["method_key"] == "proposed_top5_cusum")
    ]
    if len(rows) != len(FAULTS):
        raise ValueError("Expected one Proposed Top-5 status row per target fault")
    if not (rows["dataset"] == "protocol_not_estimable").all():
        raise ValueError(
            "Proposed Top-5 is estimable in this result set; use the main Step-18 "
            "performance plot instead of the abstention diagnostic."
        )


def main() -> None:
    screen = pd.read_csv(require_file(SCREEN_PATH), encoding="utf-8-sig")
    summary = pd.read_csv(require_file(SUMMARY_PATH), encoding="utf-8-sig")
    results = pd.read_csv(require_file(RESULTS_PATH), encoding="utf-8-sig")
    validate_protocol_status(results)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11.2, 8.2))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.40, wspace=0.42)
    axis_9 = figure.add_subplot(grid[0, 0])
    axis_15 = figure.add_subplot(grid[0, 1])
    axis_counts = figure.add_subplot(grid[1, :])

    plot_fault_screen(axis_9, screen, 9, "a")
    plot_fault_screen(axis_15, screen, 15, "b")
    plot_stable_counts(axis_counts, summary)

    figure.suptitle(
        "Prespecified signed-residual applicability check on TEP Faults 9 and 15",
        fontsize=14,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Neither development set met the five-residual gate; the proposed local "
        "Top-5 detector therefore abstained. Companion counts are diagnostic only "
        "and were not used for retuning.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.88, bottom=0.10)

    png_path = FIGURE_DIR / "fig18a_faults9_15_applicability_diagnostic.png"
    pdf_path = FIGURE_DIR / "fig18a_faults9_15_applicability_diagnostic.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print("Step 18b finished successfully.")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print("Protocol outcome: Fault 9 = abstained; Fault 15 = abstained.")


if __name__ == "__main__":
    main()
