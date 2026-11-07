"""
Step 16: Validation-calibrated operating curves for strong sequential methods.

Step 15 identified a genuine trade-off: the dense shrinkage-LDA residual score
and the Top-1 residual can alarm earlier than the frozen Top-5 method, but they
also produced more normal-evaluation run alarms at the primary 95th-percentile
control limit.  A single operating point cannot determine whether one method
dominates another.

This script therefore evaluates complete, prespecified control-limit curves.
The signal construction and CUSUM k values are frozen from Step 15.  Only the
percentile of the normal-validation run maxima changes:

    90, 92.5, 95, 97.5, and 99 percent.

These correspond nominally to validation run FAR targets of 10, 7.5, 5, 2.5,
and 1 percent.  Every point is reported.  No threshold is selected using the
normal-evaluation or Fault-3 final-test outcomes.

Compared methods
----------------
* PCA Q/SPE-CUSUM (global reference).
* All-52 unpaired signed residual CUSUM.
* Shrinkage-LDA residual CUSUM.
* Unpaired signed Top-1 residual CUSUM.
* Frozen proposed unpaired signed Top-5 residual CUSUM.

Place in:
    TEP_Fault3_Project/scripts/16_validation_far_operating_curves.py

Run from project root:
    python scripts/16_validation_far_operating_curves.py

Prerequisites: Steps 02, 11, and 15.  Keep both
``11_unpaired_runlevel_screening.py`` and
``15_fair_sequential_baselines.py`` in the scripts directory.
"""

from pathlib import Path
import gc
import importlib.util
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_LIMIT_PERCENTILES = [90.0, 92.5, 95.0, 97.5, 99.0]
PRIMARY_PERCENTILE = 95.0
PRIMARY_WINDOW = "early_160_primary"

METRIC_FIELDS_FOR_PARITY = [
    "normal_run_level_far",
    "fault3_run_level_detection_rate",
    "fault3_post_fault_point_detection_rate",
    "median_alarm_delay_min",
]


def load_step15_module():
    path = SCRIPT_DIR / "15_fair_sequential_baselines.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 15 script not found: {path}\n"
            "Keep 15_fair_sequential_baselines.py in the scripts directory."
        )
    spec = importlib.util.spec_from_file_location("step15_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "load_step11_module",
        "load_unpaired_definitions",
        "derive_retained_eigenvalues",
        "compute_pca_outputs",
        "fit_global_transforms",
        "transform_global_inputs",
        "fit_lda_score",
        "apply_lda_score",
        "fit_scalar_standardizer",
        "apply_scalar_standardizer",
        "signed_local_input",
        "compute_cusum",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Step 15 script is missing functions: {missing}")
    return module


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_step15_selected_k():
    path = require_file(
        TABLE_DIR / "15_fair_sequential_baseline_results.csv",
        "run scripts/15_fair_sequential_baselines.py",
    )
    table = pd.read_csv(path)
    required_keys = [
        "pca_q_cusum",
        "all52_signed_cusum",
        "lda_residual_cusum",
        "top1_signed_cusum",
        "proposed_top5_cusum",
    ]
    rows = table[table["method_key"].isin(required_keys)].copy()
    if len(rows) != len(required_keys) or rows["method_key"].duplicated().any():
        raise ValueError(
            "Step 15 results do not contain exactly one row for every "
            "required operating-curve method"
        )
    selected_k = rows.set_index("method_key")["cusum_k"].astype(float).to_dict()
    return selected_k, rows.set_index("method_key")


def percentile_compat(values, percentile):
    values = np.asarray(values, dtype=float)
    try:
        return float(np.percentile(values, percentile, method="linear"))
    except TypeError:
        return float(
            np.percentile(values, percentile, interpolation="linear")
        )


def validation_run_max(statistic_df):
    return (
        statistic_df.groupby("simulationRun")["monitoring_statistic"]
        .max()
        .reset_index(name="run_max_statistic")
    )


def annotate_run_results(
    frame,
    method_key,
    method,
    cusum_k,
    percentile,
    limit,
):
    out = frame.copy()
    out.insert(0, "method_key", method_key)
    out.insert(1, "method", method)
    out.insert(2, "cusum_k", float(cusum_k))
    out.insert(3, "control_limit_percentile", float(percentile))
    out.insert(4, "control_limit", float(limit))
    return out


def pareto_flags(curve):
    """Flag nondominated points using FAR, point detection, and delay."""
    flags = []
    for i, row in curve.iterrows():
        dominated = False
        for j, other in curve.iterrows():
            if i == j:
                continue
            no_worse = (
                other["normal_run_level_far"]
                <= row["normal_run_level_far"] + 1e-15
                and other["fault3_post_fault_point_detection_rate"]
                >= row["fault3_post_fault_point_detection_rate"] - 1e-15
                and other["median_alarm_delay_min"]
                <= row["median_alarm_delay_min"] + 1e-15
            )
            strictly_better = (
                other["normal_run_level_far"]
                < row["normal_run_level_far"] - 1e-15
                or other["fault3_post_fault_point_detection_rate"]
                > row["fault3_post_fault_point_detection_rate"] + 1e-15
                or other["median_alarm_delay_min"]
                < row["median_alarm_delay_min"] - 1e-15
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=curve.index, dtype=bool)


def compare_rows(current, reference, label):
    differences = {
        field: float(current[field]) - float(reference[field])
        for field in METRIC_FIELDS_FOR_PARITY
    }
    exact = all(
        np.isclose(value, 0.0, rtol=0.0, atol=1e-12, equal_nan=True)
        for value in differences.values()
    )
    return {
        "comparison": label,
        "exact_with_tolerance_1e-12": bool(exact),
        "differences": {key: float(value) for key, value in differences.items()},
    }


def plot_point_detection(curve):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method, group in curve.groupby("method", sort=False):
        group = group.sort_values("normal_run_level_far")
        ax.plot(
            100 * group["normal_run_level_far"],
            100 * group["fault3_post_fault_point_detection_rate"],
            marker="o",
            linewidth=1.8,
            label=group["method_short"].iloc[0],
        )
        primary = group[np.isclose(
            group["control_limit_percentile"], PRIMARY_PERCENTILE
        )]
        ax.scatter(
            100 * primary["normal_run_level_far"],
            100 * primary["fault3_post_fault_point_detection_rate"],
            marker="*",
            s=130,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
    ax.set_xlabel("Normal-evaluation run-level FAR (%)")
    ax.set_ylabel("Fault-3 post-fault point detection (%)")
    ax.set_title("Validation-calibrated detection operating curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png = FIGURE_DIR / "fig16a_point_detection_operating_curve.png"
    pdf = FIGURE_DIR / "fig16a_point_detection_operating_curve.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_delay(curve):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method, group in curve.groupby("method", sort=False):
        group = group.sort_values("normal_run_level_far")
        ax.plot(
            100 * group["normal_run_level_far"],
            group["median_alarm_delay_min"],
            marker="o",
            linewidth=1.8,
            label=group["method_short"].iloc[0],
        )
        primary = group[np.isclose(
            group["control_limit_percentile"], PRIMARY_PERCENTILE
        )]
        ax.scatter(
            100 * primary["normal_run_level_far"],
            primary["median_alarm_delay_min"],
            marker="*",
            s=130,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
    ax.set_xlabel("Normal-evaluation run-level FAR (%)")
    ax.set_ylabel("Median alarm delay among detected runs (min)")
    ax.set_title("Validation-calibrated delay operating curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png = FIGURE_DIR / "fig16b_delay_operating_curve.png"
    pdf = FIGURE_DIR / "fig16b_delay_operating_curve.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_run_detection(curve):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method, group in curve.groupby("method", sort=False):
        group = group.sort_values("normal_run_level_far")
        ax.plot(
            100 * group["normal_run_level_far"],
            100 * group["fault3_run_level_detection_rate"],
            marker="o",
            linewidth=1.8,
            label=group["method_short"].iloc[0],
        )
    ax.set_xlabel("Normal-evaluation run-level FAR (%)")
    ax.set_ylabel("Fault-3 run-level detection (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Run-level detection across validation-calibrated limits")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    png = FIGURE_DIR / "fig16c_run_detection_operating_curve.png"
    pdf = FIGURE_DIR / "fig16c_run_detection_operating_curve.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    print("=" * 80)
    print("Step 16: Validation-FAR operating curves")
    print("=" * 80)
    print(
        "Signal definitions and k values are frozen from Step 15; all "
        "prespecified normal-validation percentiles are reported."
    )

    step15 = load_step15_module()
    step11 = step15.load_step11_module()
    selected_k, step15_reference = load_step15_selected_k()
    variable_cols = step11.load_variable_columns()
    all_primary, _, top1, frozen_top5 = step15.load_unpaired_definitions(
        variable_cols
    )

    data = step11.load_data()
    model = step11.fit_pca_model(data["normal_train"], variable_cols)
    eigenvalues = step15.derive_retained_eigenvalues(
        data["normal_train"], variable_cols, model
    )
    residuals = {}
    statistics = {}
    residual_cols = None
    for dataset in [
        "normal_validation",
        "normal_evaluation",
        "fault3_development",
        "fault3_final_test",
    ]:
        residuals[dataset], statistics[dataset], current_cols = (
            step15.compute_pca_outputs(
                data[dataset], variable_cols, model, eigenvalues
            )
        )
        if residual_cols is None:
            residual_cols = current_cols
        elif current_cols != residual_cols:
            raise ValueError("Residual column order changed across datasets")
        print(f"Computed PCA outputs: {dataset}")
    del data
    gc.collect()

    global_parameters = step15.fit_global_transforms(
        statistics["normal_validation"]
    )
    global_inputs = {
        dataset: step15.transform_global_inputs(stats, global_parameters)
        for dataset, stats in statistics.items()
    }
    input_sets = {
        "pca_q_cusum": {
            dataset: values["q"] for dataset, values in global_inputs.items()
        }
    }

    local_tables = {
        "all52_signed_cusum": all_primary[
            ["variable", "residual_variable", "sign_for_positive_evidence"]
        ],
        "top1_signed_cusum": top1,
        "proposed_top5_cusum": frozen_top5,
    }
    for method_key, selected in local_tables.items():
        scaler = step11.fit_signed_scaler(
            residuals["normal_validation"], selected
        )
        input_sets[method_key] = {
            dataset: step15.signed_local_input(step11, frame, scaler)
            for dataset, frame in residuals.items()
        }

    lda_weights, _, lda_metadata = step15.fit_lda_score(
        step11,
        residuals["normal_validation"],
        residuals["fault3_development"],
        residual_cols,
        variable_cols,
    )
    lda_raw = {
        dataset: step15.apply_lda_score(frame, residual_cols, lda_weights)
        for dataset, frame in residuals.items()
    }
    lda_scaler = step15.fit_scalar_standardizer(
        lda_raw["normal_validation"]
    )
    input_sets["lda_residual_cusum"] = {
        dataset: step15.apply_scalar_standardizer(frame, lda_scaler)
        for dataset, frame in lda_raw.items()
    }

    method_specs = [
        ("pca_q_cusum", "PCA Q/SPE-CUSUM", "Q-CUSUM"),
        (
            "all52_signed_cusum",
            "All-52 unpaired signed residual CUSUM",
            "All-52 signed",
        ),
        (
            "lda_residual_cusum",
            "Shrinkage-LDA residual CUSUM",
            "LDA residual",
        ),
        (
            "top1_signed_cusum",
            "Unpaired signed Top-1 residual CUSUM",
            "Top-1 signed",
        ),
        (
            "proposed_top5_cusum",
            "Proposed unpaired signed Top-5 residual CUSUM",
            "Proposed Top-5",
        ),
    ]

    rows = []
    validation_max_tables = []
    normal_run_tables = []
    fault_run_tables = []
    parity_checks = []
    for method_key, method, method_short in method_specs:
        k = float(selected_k[method_key])
        inputs = input_sets[method_key]
        statistics_by_dataset = {
            dataset: step15.compute_cusum(frame, k)
            for dataset, frame in inputs.items()
            if dataset in {
                "normal_validation", "normal_evaluation", "fault3_final_test"
            }
        }
        run_max = validation_run_max(
            statistics_by_dataset["normal_validation"]
        )
        run_max.insert(0, "method_key", method_key)
        run_max.insert(1, "method", method)
        run_max.insert(2, "cusum_k", k)
        validation_max_tables.append(run_max)

        for percentile in CONTROL_LIMIT_PERCENTILES:
            limit = percentile_compat(
                run_max["run_max_statistic"].to_numpy(), percentile
            )
            validation_metrics, _ = step11.evaluate_normal(
                statistics_by_dataset["normal_validation"], limit
            )
            normal_metrics, normal_runs = step11.evaluate_normal(
                statistics_by_dataset["normal_evaluation"], limit
            )
            fault_metrics, fault_runs = step11.evaluate_fault(
                statistics_by_dataset["fault3_final_test"], limit
            )
            row = {
                "method_key": method_key,
                "method": method,
                "method_short": method_short,
                "cusum_k": k,
                "control_limit_percentile": float(percentile),
                "nominal_validation_run_far": 1.0 - percentile / 100.0,
                "control_limit": float(limit),
                "actual_validation_run_far": float(
                    validation_metrics["normal_run_level_far"]
                ),
            }
            row.update(normal_metrics)
            row.update(fault_metrics)
            rows.append(row)
            normal_run_tables.append(
                annotate_run_results(
                    normal_runs, method_key, method, k, percentile, limit
                )
            )
            fault_run_tables.append(
                annotate_run_results(
                    fault_runs, method_key, method, k, percentile, limit
                )
            )
            print(
                f"{method_short}, validation percentile={percentile:g}: "
                f"normal FAR={100*normal_metrics['normal_run_level_far']:.2f}%, "
                f"run detection="
                f"{100*fault_metrics['fault3_run_level_detection_rate']:.2f}%, "
                f"point="
                f"{100*fault_metrics['fault3_post_fault_point_detection_rate']:.2f}%, "
                f"delay={fault_metrics['median_alarm_delay_min']} min"
            )

        current_primary = pd.DataFrame(rows)
        current_primary = current_primary[
            (current_primary["method_key"] == method_key)
            & np.isclose(
                current_primary["control_limit_percentile"],
                PRIMARY_PERCENTILE,
            )
        ].iloc[0]
        parity_checks.append(
            compare_rows(
                current_primary,
                step15_reference.loc[method_key],
                f"Step 16 {method_key} 95th-percentile versus Step 15",
            )
        )

    curve = pd.DataFrame(rows)
    curve["pareto_nondominated_across_all_points"] = pareto_flags(curve)
    primary_table = curve[np.isclose(
        curve["control_limit_percentile"], PRIMARY_PERCENTILE
    )].copy()
    normal_run_results = pd.concat(normal_run_tables, ignore_index=True)
    fault_run_results = pd.concat(fault_run_tables, ignore_index=True)
    validation_max = pd.concat(validation_max_tables, ignore_index=True)

    paths = {
        "operating_curve": TABLE_DIR
        / "16_validation_far_operating_curve.csv",
        "primary_95th_percentile": TABLE_DIR
        / "16_primary_95th_percentile_comparison.csv",
        "validation_run_max": TABLE_DIR
        / "16_method_validation_run_maxima.csv",
        "normal_run_results": TABLE_DIR
        / "16_operating_curve_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / "16_operating_curve_fault3_run_results.csv",
    }
    curve.to_csv(paths["operating_curve"], index=False, encoding="utf-8-sig")
    primary_table.to_csv(
        paths["primary_95th_percentile"], index=False, encoding="utf-8-sig"
    )
    validation_max.to_csv(
        paths["validation_run_max"], index=False, encoding="utf-8-sig"
    )
    normal_run_results.to_csv(
        paths["normal_run_results"], index=False, encoding="utf-8-sig"
    )
    fault_run_results.to_csv(
        paths["fault3_run_results"], index=False, encoding="utf-8-sig"
    )

    figure_paths = (
        plot_point_detection(curve)
        + plot_delay(curve)
        + plot_run_detection(curve)
    )
    config = {
        "script": "scripts/16_validation_far_operating_curves.py",
        "analysis_role": (
            "Descriptive operating-characteristic analysis; no final-set "
            "threshold selection"
        ),
        "control_limit_percentiles_prespecified": CONTROL_LIMIT_PERCENTILES,
        "nominal_validation_far_targets": [
            1.0 - value / 100.0 for value in CONTROL_LIMIT_PERCENTILES
        ],
        "primary_percentile": PRIMARY_PERCENTILE,
        "method_k_values_frozen_from_step15": selected_k,
        "final_test_used_for_threshold_selection": False,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "lda": lda_metadata,
        "parity_checks": parity_checks,
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "fault3_validation_far_operating_curves.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 16 summary: primary 95th-percentile operating points")
    print("=" * 80)
    print(
        primary_table[
            [
                "method",
                "cusum_k",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print("\nNondominated operating points across the complete grid:")
    print(
        curve[curve["pareto_nondominated_across_all_points"]][
            [
                "method_short",
                "control_limit_percentile",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print(f"\nParity checks: {parity_checks}")
    print(f"Saved configuration: {config_path}")
    print("Step 16 finished successfully.")


if __name__ == "__main__":
    main()
