"""
Step 13: Parameter sensitivity for the frozen pairing-free method.

Purpose
-------
Step 11 replaced matched normal/fault subtraction with an independent,
run-level bootstrap screen.  Step 12 then verified the roles of direction
unification, Top-K compression, and CUSUM.  This script checks whether the
frozen primary choices (Top-5 and CUSUM reference value k=0.05) lie in a
reasonable robustness region.

The analysis is deliberately non-tuning:

* The primary method remains Top-5 with k=0.05, regardless of the sensitivity
  results.
* Every prespecified Top-K and k value is reported.
* For every setting, the control limit is recalibrated exclusively from the
  normal-validation run maxima at the unchanged 95th percentile.
* The pairing-free signs and Hedges-g ranking are read from the prespecified
  Step 11 ``early_160_primary`` development screen and never recomputed from
  final-test outcomes.

Place in:
    TEP_Fault3_Project/scripts/13_unpaired_parameter_sensitivity.py

Run from project root:
    python scripts/13_unpaired_parameter_sensitivity.py

Prerequisites: Steps 02 and 11.  Keep
``11_unpaired_runlevel_screening.py`` in the scripts directory.
"""

from pathlib import Path
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

PRIMARY_WINDOW = "early_160_primary"
PRIMARY_TOP_K = 5
PRIMARY_CUSUM_K = 0.05

# Broad range plus a dense neighborhood around the frozen Top-5 choice.
TOP_K_LIST = [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 30, 40]

# Same reference-value grid used by the historical Step 07 analysis, now
# applied to the pairing-free frozen Top-5 residual set.
CUSUM_K_LIST = [0.01, 0.03, 0.05, 0.07, 0.10]

METRIC_FIELDS_FOR_PARITY = [
    "normal_run_level_far",
    "fault3_run_level_detection_rate",
    "fault3_post_fault_point_detection_rate",
    "median_alarm_delay_min",
]


def load_step11_module():
    path = SCRIPT_DIR / "11_unpaired_runlevel_screening.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 11 script not found: {path}\n"
            "Keep 11_unpaired_runlevel_screening.py in the scripts directory."
        )
    spec = importlib.util.spec_from_file_location("step11_unpaired", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "load_variable_columns",
        "load_data",
        "fit_pca_model",
        "compute_residual_frame",
        "fit_signed_scaler",
        "construct_local_signed_index",
        "determine_limit",
        "evaluate_normal",
        "evaluate_fault",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Step 11 script is missing functions: {missing}")
    return module


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def parse_bool_series(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"Cannot parse Boolean column {name}: {unknown}")
    return normalized.map(mapping).astype(bool)


def load_ranked_stable_residuals(variable_cols):
    metrics_path = require_file(
        TABLE_DIR / "11_unpaired_screening_all_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    selected_path = require_file(
        TABLE_DIR / "11_unpaired_selected_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    metrics = pd.read_csv(metrics_path)
    selected_all = pd.read_csv(selected_path)

    primary = metrics[
        metrics["window"].astype(str) == PRIMARY_WINDOW
    ].copy()
    if len(primary) != len(variable_cols):
        raise ValueError(
            f"Expected {len(variable_cols)} primary-window rows, found "
            f"{len(primary)}"
        )
    if primary["variable"].astype(str).duplicated().any():
        raise ValueError("Primary Step 11 table contains duplicate variables")

    required = {
        "variable",
        "residual_variable",
        "direction_stable_95ci",
        "abs_hedges_g",
        "bootstrap_direction_confidence",
        "bootstrap_top5_selection_frequency",
        "sign_for_positive_evidence",
    }
    missing = sorted(required.difference(primary.columns))
    if missing:
        raise ValueError(f"Step 11 primary table is missing columns: {missing}")

    primary["direction_stable_95ci"] = parse_bool_series(
        primary["direction_stable_95ci"], "direction_stable_95ci"
    )
    primary["sign_for_positive_evidence"] = (
        primary["sign_for_positive_evidence"].astype(int)
    )
    invalid_signs = sorted(
        set(primary["sign_for_positive_evidence"]).difference({-1, 1})
    )
    if invalid_signs:
        raise ValueError(f"Invalid Step 11 signs: {invalid_signs}")

    stable = primary[primary["direction_stable_95ci"]].copy()
    ranked = stable.sort_values(
        [
            "abs_hedges_g",
            "bootstrap_direction_confidence",
            "bootstrap_top5_selection_frequency",
            "variable",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "sensitivity_rank", np.arange(1, len(ranked) + 1))

    if len(ranked) < max(TOP_K_LIST):
        raise ValueError(
            f"Top-K grid requires at least {max(TOP_K_LIST)} stable residuals, "
            f"but Step 11 contains {len(ranked)}"
        )

    frozen = selected_all[
        selected_all["window"].astype(str) == PRIMARY_WINDOW
    ].sort_values("selection_rank")
    if len(frozen) != PRIMARY_TOP_K:
        raise ValueError(
            f"Expected {PRIMARY_TOP_K} frozen primary residuals, found "
            f"{len(frozen)}"
        )
    ranked_top5 = ranked.head(PRIMARY_TOP_K)["variable"].astype(str).tolist()
    frozen_top5 = frozen["variable"].astype(str).tolist()
    if ranked_top5 != frozen_top5:
        raise ValueError(
            "The reconstructed Hedges-g Top-5 does not match Step 11: "
            f"ranked={ranked_top5}, frozen={frozen_top5}"
        )
    return ranked, frozen.copy()


def compute_cusum(local_df, k):
    """One-sided resetting CUSUM with an explicit reference value."""
    df = local_df.sort_values(["simulationRun", "sample"]).copy()
    values_out = np.empty(len(df), dtype=float)
    for positions in df.groupby("simulationRun", sort=True).indices.values():
        values = df.iloc[positions]["local_signed_residual"].to_numpy(
            dtype=float
        )
        run_cusum = np.empty(len(values), dtype=float)
        current = 0.0
        for i, value in enumerate(values):
            current = max(0.0, current + value - float(k))
            run_cusum[i] = current
        values_out[positions] = run_cusum
    df["monitoring_statistic"] = values_out
    return df


def evaluate_configuration(
    step11,
    selected,
    cusum_k,
    normal_validation_res,
    normal_evaluation_res,
    fault_final_res,
):
    scaler = step11.fit_signed_scaler(normal_validation_res, selected)
    validation_local = step11.construct_local_signed_index(
        normal_validation_res, scaler
    )
    normal_local = step11.construct_local_signed_index(
        normal_evaluation_res, scaler
    )
    fault_local = step11.construct_local_signed_index(fault_final_res, scaler)

    validation_cusum = compute_cusum(validation_local, cusum_k)
    normal_cusum = compute_cusum(normal_local, cusum_k)
    fault_cusum = compute_cusum(fault_local, cusum_k)

    control_limit, validation_run_max = step11.determine_limit(
        validation_cusum
    )
    normal_metrics, normal_runs = step11.evaluate_normal(
        normal_cusum, control_limit
    )
    fault_metrics, fault_runs = step11.evaluate_fault(
        fault_cusum, control_limit
    )
    metrics = {
        "n_selected_variables": int(len(selected)),
        "selected_variables": ";".join(selected["variable"].astype(str)),
        "selected_signs": ";".join(
            selected["sign_for_positive_evidence"].astype(int).astype(str)
        ),
        "cusum_k": float(cusum_k),
        "run_level_percentile": float(step11.RUN_LEVEL_PERCENTILE),
        "control_limit": float(control_limit),
    }
    metrics.update(normal_metrics)
    metrics.update(fault_metrics)
    return {
        "metrics": metrics,
        "scaler": scaler,
        "validation_run_max": validation_run_max,
        "normal_run_results": normal_runs,
        "fault3_run_results": fault_runs,
    }


def annotate_run_results(frame, axis, value, top_k, cusum_k, primary):
    out = frame.copy()
    out.insert(0, "sensitivity_axis", axis)
    out.insert(1, "parameter_value", float(value))
    out.insert(2, "top_k", int(top_k))
    out.insert(3, "cusum_k", float(cusum_k))
    out.insert(4, "is_frozen_primary", bool(primary))
    return out


def run_topk_sensitivity(
    step11,
    ranked_stable,
    normal_validation_res,
    normal_evaluation_res,
    fault_final_res,
):
    rows = []
    selections = []
    normal_runs_all = []
    fault_runs_all = []
    print("\n" + "=" * 80)
    print("Top-K sensitivity (fixed pairing-free ranking and k=0.05)")
    print("=" * 80)

    for top_k in TOP_K_LIST:
        selected = ranked_stable.head(top_k).copy()
        selected["selection_rank"] = np.arange(1, top_k + 1)
        result = evaluate_configuration(
            step11,
            selected,
            PRIMARY_CUSUM_K,
            normal_validation_res,
            normal_evaluation_res,
            fault_final_res,
        )
        is_primary = top_k == PRIMARY_TOP_K
        row = {
            "sensitivity_axis": "top_k",
            "top_k": int(top_k),
            "is_frozen_primary": is_primary,
            "n_bootstrap_stable_candidates": int(len(ranked_stable)),
        }
        row.update(result["metrics"])
        rows.append(row)

        selection_copy = selected.copy()
        selection_copy.insert(0, "top_k", int(top_k))
        selection_copy.insert(1, "is_frozen_primary", is_primary)
        selections.append(selection_copy)
        normal_runs_all.append(
            annotate_run_results(
                result["normal_run_results"],
                "top_k",
                top_k,
                top_k,
                PRIMARY_CUSUM_K,
                is_primary,
            )
        )
        fault_runs_all.append(
            annotate_run_results(
                result["fault3_run_results"],
                "top_k",
                top_k,
                top_k,
                PRIMARY_CUSUM_K,
                is_primary,
            )
        )
        m = result["metrics"]
        print(
            f"Top-{top_k:>2}: FAR={100*m['normal_run_level_far']:.2f}%, "
            f"run detection={100*m['fault3_run_level_detection_rate']:.2f}%, "
            f"point detection="
            f"{100*m['fault3_post_fault_point_detection_rate']:.2f}%, "
            f"median delay={m['median_alarm_delay_min']} min"
        )

    return (
        pd.DataFrame(rows),
        pd.concat(selections, ignore_index=True),
        normal_runs_all,
        fault_runs_all,
    )


def run_cusum_k_sensitivity(
    step11,
    frozen_top5,
    normal_validation_res,
    normal_evaluation_res,
    fault_final_res,
):
    rows = []
    normal_runs_all = []
    fault_runs_all = []
    print("\n" + "=" * 80)
    print("CUSUM reference-value sensitivity (fixed pairing-free Top-5)")
    print("=" * 80)

    for cusum_k in CUSUM_K_LIST:
        result = evaluate_configuration(
            step11,
            frozen_top5,
            cusum_k,
            normal_validation_res,
            normal_evaluation_res,
            fault_final_res,
        )
        is_primary = bool(np.isclose(cusum_k, PRIMARY_CUSUM_K))
        row = {
            "sensitivity_axis": "cusum_k",
            "top_k": PRIMARY_TOP_K,
            "is_frozen_primary": is_primary,
        }
        row.update(result["metrics"])
        rows.append(row)
        normal_runs_all.append(
            annotate_run_results(
                result["normal_run_results"],
                "cusum_k",
                cusum_k,
                PRIMARY_TOP_K,
                cusum_k,
                is_primary,
            )
        )
        fault_runs_all.append(
            annotate_run_results(
                result["fault3_run_results"],
                "cusum_k",
                cusum_k,
                PRIMARY_TOP_K,
                cusum_k,
                is_primary,
            )
        )
        m = result["metrics"]
        print(
            f"k={cusum_k:.2f}: FAR={100*m['normal_run_level_far']:.2f}%, "
            f"run detection={100*m['fault3_run_level_detection_rate']:.2f}%, "
            f"point detection="
            f"{100*m['fault3_post_fault_point_detection_rate']:.2f}%, "
            f"median delay={m['median_alarm_delay_min']} min"
        )
    return pd.DataFrame(rows), normal_runs_all, fault_runs_all


def compare_metric_rows(first, second, label):
    differences = {}
    for field in METRIC_FIELDS_FOR_PARITY:
        a, b = float(first[field]), float(second[field])
        differences[field] = a - b
    exact = all(
        np.isclose(value, 0.0, rtol=0.0, atol=1e-12, equal_nan=True)
        for value in differences.values()
    )
    return {
        "comparison": label,
        "exact_with_tolerance_1e-12": bool(exact),
        "differences": {k: float(v) for k, v in differences.items()},
    }


def build_parity_checks(topk_df, cusum_df):
    top5 = topk_df[topk_df["top_k"] == PRIMARY_TOP_K].iloc[0]
    k005 = cusum_df[np.isclose(cusum_df["cusum_k"], PRIMARY_CUSUM_K)].iloc[0]
    checks = [
        compare_metric_rows(
            top5,
            k005,
            "Top-K primary row versus CUSUM-k primary row",
        )
    ]

    step11_path = TABLE_DIR / "11_unpaired_cusum_performance.csv"
    if step11_path.exists():
        table = pd.read_csv(step11_path)
        reference = table[
            table["screening_window"].astype(str) == PRIMARY_WINDOW
        ]
        if len(reference) == 1:
            checks.append(
                compare_metric_rows(
                    top5,
                    reference.iloc[0],
                    "Step 13 primary versus Step 11 primary",
                )
            )

    step12_path = TABLE_DIR / "12_unpaired_factorized_ablation_results.csv"
    if step12_path.exists():
        table = pd.read_csv(step12_path)
        reference = table[
            table["method"].astype(str)
            == "Unpaired signed Top-5 residuals + CUSUM"
        ]
        if len(reference) == 1:
            checks.append(
                compare_metric_rows(
                    top5,
                    reference.iloc[0],
                    "Step 13 primary versus Step 12 full method",
                )
            )
    return checks


def plot_rate_sensitivity(df, x_col, x_label, title, stem):
    x = df[x_col].to_numpy(dtype=float)
    fig, axes = plt.subplots(
        2, 1, figsize=(8.8, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    axes[0].plot(
        x,
        100 * df["fault3_run_level_detection_rate"],
        marker="o",
        label="Fault 3 run-level detection",
        color="#D95F59",
    )
    axes[0].plot(
        x,
        100 * df["fault3_post_fault_point_detection_rate"],
        marker="s",
        label="Post-fault point detection",
        color="#4C78A8",
    )
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        x,
        100 * df["normal_run_level_far"],
        marker="o",
        color="#7A9E9F",
        label="Normal run-level FAR",
    )
    axes[1].set_ylabel("FAR (%)")
    axes[1].set_xlabel(x_label)
    axes[1].grid(axis="y", alpha=0.25)

    primary_x = PRIMARY_TOP_K if x_col == "top_k" else PRIMARY_CUSUM_K
    for ax in axes:
        ax.axvline(
            primary_x,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label="Frozen primary" if ax is axes[1] else None,
        )
    axes[1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_delay_sensitivity(df, x_col, x_label, title, stem):
    x = df[x_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(
        x,
        df["median_alarm_delay_min"],
        marker="o",
        color="#4C78A8",
    )
    primary_x = PRIMARY_TOP_K if x_col == "top_k" else PRIMARY_CUSUM_K
    ax.axvline(
        primary_x,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="Frozen primary",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Median alarm delay (min)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    print("=" * 80)
    print("Step 13: Pairing-free Top-K and CUSUM sensitivity")
    print("=" * 80)
    print(
        "Primary method is frozen at Top-5 and k=0.05; sensitivity results "
        "will not be used to retune it."
    )

    step11 = load_step11_module()
    variable_cols = step11.load_variable_columns()
    ranked_stable, frozen_top5 = load_ranked_stable_residuals(variable_cols)
    print(f"Bootstrap-stable primary candidates: {len(ranked_stable)}")
    print(f"Frozen Top-5: {frozen_top5['variable'].astype(str).tolist()}")

    data = step11.load_data()
    model = step11.fit_pca_model(data["normal_train"], variable_cols)
    print("Computing PCA residuals for validation and final evaluation...")
    normal_validation_res, _ = step11.compute_residual_frame(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = step11.compute_residual_frame(
        data["normal_evaluation"], variable_cols, model
    )
    fault_final_res, _ = step11.compute_residual_frame(
        data["fault3_final_test"], variable_cols, model
    )

    topk_df, selections_df, normal_topk, fault_topk = run_topk_sensitivity(
        step11,
        ranked_stable,
        normal_validation_res,
        normal_evaluation_res,
        fault_final_res,
    )
    cusum_df, normal_cusum, fault_cusum = run_cusum_k_sensitivity(
        step11,
        frozen_top5,
        normal_validation_res,
        normal_evaluation_res,
        fault_final_res,
    )

    paths = {
        "topk_metrics": TABLE_DIR / "13_unpaired_topk_sensitivity.csv",
        "topk_selections": TABLE_DIR
        / "13_unpaired_topk_selected_variables.csv",
        "cusum_metrics": TABLE_DIR / "13_unpaired_cusum_k_sensitivity.csv",
        "normal_run_results": TABLE_DIR
        / "13_unpaired_sensitivity_normal_run_results.csv",
        "fault_run_results": TABLE_DIR
        / "13_unpaired_sensitivity_fault3_run_results.csv",
    }
    topk_df.to_csv(paths["topk_metrics"], index=False, encoding="utf-8-sig")
    selections_df.to_csv(
        paths["topk_selections"], index=False, encoding="utf-8-sig"
    )
    cusum_df.to_csv(
        paths["cusum_metrics"], index=False, encoding="utf-8-sig"
    )
    pd.concat(normal_topk + normal_cusum, ignore_index=True).to_csv(
        paths["normal_run_results"], index=False, encoding="utf-8-sig"
    )
    pd.concat(fault_topk + fault_cusum, ignore_index=True).to_csv(
        paths["fault_run_results"], index=False, encoding="utf-8-sig"
    )

    figure_paths = []
    figure_paths += plot_rate_sensitivity(
        topk_df,
        "top_k",
        "Number of selected residual variables (K)",
        "Pairing-free Top-K sensitivity",
        "fig13a_unpaired_topk_sensitivity",
    )
    figure_paths += plot_delay_sensitivity(
        topk_df,
        "top_k",
        "Number of selected residual variables (K)",
        "Alarm delay across pairing-free Top-K settings",
        "fig13b_unpaired_topk_delay",
    )
    figure_paths += plot_rate_sensitivity(
        cusum_df,
        "cusum_k",
        "CUSUM reference value k",
        "Pairing-free CUSUM reference-value sensitivity",
        "fig13c_unpaired_cusum_k_sensitivity",
    )
    figure_paths += plot_delay_sensitivity(
        cusum_df,
        "cusum_k",
        "CUSUM reference value k",
        "Alarm delay across CUSUM reference values",
        "fig13d_unpaired_cusum_k_delay",
    )

    parity_checks = build_parity_checks(topk_df, cusum_df)
    config = {
        "script": "scripts/13_unpaired_parameter_sensitivity.py",
        "analysis_role": (
            "Robustness analysis only; no final-test-driven retuning"
        ),
        "primary_screening_window": PRIMARY_WINDOW,
        "selection_source": (
            "results/tables/11_unpaired_screening_all_variables.csv"
        ),
        "ranking_rule": (
            "95% run-bootstrap CI excludes zero, then descending absolute "
            "Hedges' g with the unchanged Step 11 tie-breaks"
        ),
        "primary_top_k_frozen": PRIMARY_TOP_K,
        "primary_cusum_k_frozen": PRIMARY_CUSUM_K,
        "top_k_grid_prespecified": TOP_K_LIST,
        "cusum_k_grid_prespecified": CUSUM_K_LIST,
        "control_limit_rule": (
            "95th percentile of normal-validation run maxima, recalibrated "
            "for every setting"
        ),
        "n_bootstrap_stable_candidates": int(len(ranked_stable)),
        "frozen_selected_variables": frozen_top5["variable"].astype(str).tolist(),
        "frozen_selected_signs": frozen_top5[
            "sign_for_positive_evidence"
        ].astype(int).tolist(),
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "parity_checks": parity_checks,
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "fault3_unpaired_parameter_sensitivity.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 13 summary")
    print("=" * 80)
    print("Top-K sensitivity:")
    print(
        topk_df[
            [
                "top_k",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print("\nCUSUM k sensitivity:")
    print(
        cusum_df[
            [
                "cusum_k",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print(f"\nParity checks: {parity_checks}")
    print(f"Saved configuration: {config_path}")
    print("Step 13 finished successfully.")


if __name__ == "__main__":
    main()
