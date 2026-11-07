"""
Step 15: Fair sequential baselines for the pairing-free local method.

Motivation
----------
The historical comparison gives the proposed method a CUSUM stage but compares
it mainly with non-sequential PCA Q/SPE and Hotelling T2 charts.  This script
tests whether the improvement persists when global and supervised competitors
receive the same run-level calibration and, where applicable, development-set
CUSUM tuning.

Final-test methods
------------------
1. Raw PCA Q/SPE.
2. Raw PCA Hotelling T2.
3. Raw joint PCA max-score chart.
4. PCA Q/SPE-CUSUM.
5. PCA T2-CUSUM.
6. Joint PCA max-score CUSUM.
7. All-52 unpaired signed residual CUSUM.
8. Run-mean shrinkage-LDA residual CUSUM.
9. Unpaired signed Top-1 residual CUSUM.
10. Frozen proposed unpaired signed Top-5 residual CUSUM.

Fairness rules
--------------
* Every scalar chart uses the 95th percentile of normal-validation run maxima.
* Global Q/T2 inputs use fixed log1p transforms and normal-validation scaling.
* Competing sequential methods may select k from a prespecified grid using
  Fault-3 development performance only.  The final test is evaluated once at
  that selected k.
* The proposed method receives no such advantage: K=5 and k=0.05 remain frozen.
* The LDA baseline uses the same pairing-free early-160 run means and a
  Ledoit-Wolf pooled within-class covariance estimate.

This script does not claim to reproduce any particular published CUSUM-PCA
algorithm.  The global sequential baselines are named descriptively.

Place in:
    TEP_Fault3_Project/scripts/15_fair_sequential_baselines.py

Run from project root:
    python scripts/15_fair_sequential_baselines.py

Prerequisites: Steps 02, 11, 12, and 13.  Keep
``11_unpaired_runlevel_screening.py`` in the scripts directory.
"""

from pathlib import Path
import gc
import importlib.util
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_WINDOW = "early_160_primary"
WINDOW_START_SAMPLE = 161
WINDOW_END_SAMPLE = 320
PRIMARY_TOP_K = 5
PRIMARY_CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95

# Competitors receive development-set tuning over this complete grid.  The
# proposed method is intentionally left at its frozen k=0.05.
CUSUM_K_GRID = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]

PARITY_FIELDS = [
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
        "calculate_run_means",
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


def load_unpaired_definitions(variable_cols):
    all_path = require_file(
        TABLE_DIR / "11_unpaired_screening_all_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    selected_path = require_file(
        TABLE_DIR / "11_unpaired_selected_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    all_metrics = pd.read_csv(all_path)
    selected_all = pd.read_csv(selected_path)
    primary = all_metrics[
        all_metrics["window"].astype(str) == PRIMARY_WINDOW
    ].copy()
    if len(primary) != len(variable_cols):
        raise ValueError(
            f"Expected {len(variable_cols)} primary rows, found {len(primary)}"
        )
    primary = primary.set_index("variable").loc[variable_cols].reset_index()
    primary["direction_stable_95ci"] = parse_bool_series(
        primary["direction_stable_95ci"], "direction_stable_95ci"
    )
    primary["sign_for_positive_evidence"] = primary[
        "sign_for_positive_evidence"
    ].astype(int)
    invalid = set(primary["sign_for_positive_evidence"]).difference({-1, 1})
    if invalid:
        raise ValueError(f"Invalid unpaired signs: {sorted(invalid)}")

    stable_ranked = primary[primary["direction_stable_95ci"]].sort_values(
        [
            "abs_hedges_g",
            "bootstrap_direction_confidence",
            "bootstrap_top5_selection_frequency",
            "variable",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    frozen_top5 = selected_all[
        selected_all["window"].astype(str) == PRIMARY_WINDOW
    ].sort_values("selection_rank")
    if len(frozen_top5) != PRIMARY_TOP_K:
        raise ValueError(
            f"Expected {PRIMARY_TOP_K} frozen primary variables, found "
            f"{len(frozen_top5)}"
        )
    reconstructed = stable_ranked.head(PRIMARY_TOP_K)["variable"].tolist()
    frozen = frozen_top5["variable"].tolist()
    if reconstructed != frozen:
        raise ValueError(
            f"Reconstructed Top-5 {reconstructed} differs from frozen {frozen}"
        )
    top1 = stable_ranked.head(1).copy()
    return primary, stable_ranked, top1, frozen_top5.copy()


def compute_pca_outputs(frame, variable_cols, model, eigenvalues):
    x = frame[variable_cols].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    components = model["components"]
    scores = x_std @ components.T
    reconstruction = scores @ components
    residual = x_std - reconstruction
    identifiers = frame[
        ["faultNumber", "simulationRun", "sample"]
    ].reset_index(drop=True)
    residual_cols = [f"res_{v}" for v in variable_cols]
    residual_df = pd.concat(
        [
            identifiers,
            pd.DataFrame(residual, columns=residual_cols),
        ],
        axis=1,
    )
    statistics = identifiers.copy()
    statistics["Q"] = np.einsum("ij,ij->i", residual, residual)
    statistics["T2"] = np.sum((scores * scores) / eigenvalues, axis=1)
    return residual_df, statistics, residual_cols


def derive_retained_eigenvalues(normal_train, variable_cols, model):
    x = normal_train[variable_cols].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    scores = x_std @ model["components"].T
    eigenvalues = scores.var(axis=0, ddof=1)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("Retained PCA eigenvalues must be finite and positive")
    return eigenvalues


def scalar_frame(identifiers, values):
    out = identifiers[
        ["faultNumber", "simulationRun", "sample"]
    ].copy()
    out["scalar_input"] = np.asarray(values, dtype=float)
    return out


def fit_global_transforms(normal_validation_stats):
    q_log = np.log1p(normal_validation_stats["Q"].to_numpy(dtype=float))
    t2_log = np.log1p(normal_validation_stats["T2"].to_numpy(dtype=float))
    q_mean, q_std = float(q_log.mean()), float(q_log.std(ddof=1))
    t2_mean, t2_std = float(t2_log.mean()), float(t2_log.std(ddof=1))
    if q_std <= 0 or t2_std <= 0:
        raise ValueError("Global PCA transformed statistics have zero variance")
    q_z = (q_log - q_mean) / q_std
    t2_z = (t2_log - t2_mean) / t2_std
    joint = np.maximum(q_z, t2_z)
    joint_mean = float(joint.mean())
    joint_std = float(joint.std(ddof=1))
    if joint_std <= 0:
        raise ValueError("Joint PCA max score has zero variance")
    return {
        "q_log_mean": q_mean,
        "q_log_std": q_std,
        "t2_log_mean": t2_mean,
        "t2_log_std": t2_std,
        "joint_max_mean": joint_mean,
        "joint_max_std": joint_std,
    }


def transform_global_inputs(stats, parameters):
    q_z = (
        np.log1p(stats["Q"].to_numpy(dtype=float))
        - parameters["q_log_mean"]
    ) / parameters["q_log_std"]
    t2_z = (
        np.log1p(stats["T2"].to_numpy(dtype=float))
        - parameters["t2_log_mean"]
    ) / parameters["t2_log_std"]
    joint_raw = np.maximum(q_z, t2_z)
    joint_z = (
        joint_raw - parameters["joint_max_mean"]
    ) / parameters["joint_max_std"]
    return {
        "q": scalar_frame(stats, q_z),
        "t2": scalar_frame(stats, t2_z),
        "joint": scalar_frame(stats, joint_z),
    }


def fit_lda_score(
    step11,
    normal_validation_res,
    fault_development_res,
    residual_cols,
    variable_cols,
):
    normal_means = step11.calculate_run_means(
        normal_validation_res,
        residual_cols,
        WINDOW_START_SAMPLE,
        WINDOW_END_SAMPLE,
    )
    fault_means = step11.calculate_run_means(
        fault_development_res,
        residual_cols,
        WINDOW_START_SAMPLE,
        WINDOW_END_SAMPLE,
    )
    normal = normal_means.to_numpy(dtype=float)
    fault = fault_means.to_numpy(dtype=float)
    normal_mean = normal.mean(axis=0)
    fault_mean = fault.mean(axis=0)
    delta = fault_mean - normal_mean
    within_centered = np.vstack(
        [normal - normal_mean, fault - fault_mean]
    )
    estimator = LedoitWolf(assume_centered=True).fit(within_centered)
    covariance = estimator.covariance_
    weights = np.linalg.solve(covariance, delta)
    norm = float(np.linalg.norm(weights))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Shrinkage-LDA produced invalid zero weights")
    weights = weights / norm
    if float(delta @ weights) < 0:
        weights = -weights
    weight_table = pd.DataFrame(
        {
            "variable": variable_cols,
            "residual_variable": residual_cols,
            "normal_run_mean": normal_mean,
            "fault3_run_mean": fault_mean,
            "unpaired_mean_shift": delta,
            "lda_weight": weights,
            "abs_lda_weight": np.abs(weights),
        }
    )
    weight_table["abs_weight_rank"] = (
        weight_table["abs_lda_weight"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    weight_table = weight_table.sort_values("abs_weight_rank")
    metadata = {
        "ledoit_wolf_shrinkage": float(estimator.shrinkage_),
        "n_features": int(len(weights)),
        "n_normal_development_runs": int(len(normal)),
        "n_fault3_development_runs": int(len(fault)),
    }
    return weights, weight_table, metadata


def apply_lda_score(residual_df, residual_cols, weights):
    values = residual_df[residual_cols].to_numpy(dtype=float) @ weights
    return scalar_frame(residual_df, values)


def fit_scalar_standardizer(normal_validation_scalar):
    values = normal_validation_scalar["scalar_input"].to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Scalar input has invalid normal-validation variance")
    return {"mean": mean, "std": std}


def apply_scalar_standardizer(frame, parameters):
    out = frame.copy()
    out["scalar_input"] = (
        out["scalar_input"] - parameters["mean"]
    ) / parameters["std"]
    return out


def signed_local_input(step11, residual_df, scaler):
    local = step11.construct_local_signed_index(residual_df, scaler)
    return scalar_frame(local, local["local_signed_residual"])


def compute_cusum(input_df, k):
    df = input_df.sort_values(
        ["simulationRun", "sample"]
    ).reset_index(drop=True).copy()
    output = np.empty(len(df), dtype=float)
    for positions in df.groupby("simulationRun", sort=True).indices.values():
        values = df.iloc[positions]["scalar_input"].to_numpy(dtype=float)
        current = 0.0
        run_values = np.empty(len(values), dtype=float)
        for i, value in enumerate(values):
            current = max(0.0, current + value - float(k))
            run_values[i] = current
        output[positions] = run_values
    result = df[
        ["faultNumber", "simulationRun", "sample"]
    ].copy()
    result["monitoring_statistic"] = output
    return result


def raw_monitoring_statistic(input_df):
    out = input_df[
        ["faultNumber", "simulationRun", "sample"]
    ].copy()
    out["monitoring_statistic"] = input_df["scalar_input"].to_numpy(
        dtype=float
    )
    return out


def evaluate_final_statistic(
    step11,
    validation_stat,
    normal_stat,
    fault_stat,
):
    limit, validation_run_max = step11.determine_limit(validation_stat)
    validation_metrics, _ = step11.evaluate_normal(validation_stat, limit)
    normal_metrics, normal_runs = step11.evaluate_normal(normal_stat, limit)
    fault_metrics, fault_runs = step11.evaluate_fault(fault_stat, limit)
    metrics = {
        "control_limit": float(limit),
        "validation_run_level_far": float(
            validation_metrics["normal_run_level_far"]
        ),
    }
    metrics.update(normal_metrics)
    metrics.update(fault_metrics)
    return metrics, validation_run_max, normal_runs, fault_runs


def evaluate_raw_method(step11, inputs):
    return evaluate_final_statistic(
        step11,
        raw_monitoring_statistic(inputs["normal_validation"]),
        raw_monitoring_statistic(inputs["normal_evaluation"]),
        raw_monitoring_statistic(inputs["fault3_final_test"]),
    )


def tune_cusum_k_on_development(step11, method_key, inputs):
    rows = []
    for k in CUSUM_K_GRID:
        validation_stat = compute_cusum(inputs["normal_validation"], k)
        development_stat = compute_cusum(inputs["fault3_development"], k)
        limit, _ = step11.determine_limit(validation_stat)
        validation_metrics, _ = step11.evaluate_normal(validation_stat, limit)
        development_metrics, _ = step11.evaluate_fault(
            development_stat, limit
        )
        row = {
            "method_key": method_key,
            "cusum_k": float(k),
            "control_limit_on_normal_validation": float(limit),
            "normal_validation_run_far": float(
                validation_metrics["normal_run_level_far"]
            ),
        }
        for name, value in development_metrics.items():
            row[f"development_{name}"] = value
        rows.append(row)
    table = pd.DataFrame(rows)
    sortable = table.copy()
    sortable["delay_for_sort"] = sortable[
        "development_median_alarm_delay_min"
    ].fillna(np.inf)
    sortable["distance_from_primary_k"] = np.abs(
        sortable["cusum_k"] - PRIMARY_CUSUM_K
    )
    chosen = sortable.sort_values(
        [
            "development_fault3_run_level_detection_rate",
            "development_fault3_post_fault_point_detection_rate",
            "delay_for_sort",
            "distance_from_primary_k",
            "cusum_k",
        ],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    chosen_k = float(chosen["cusum_k"])
    table["selected_on_development"] = np.isclose(
        table["cusum_k"], chosen_k
    )
    return chosen_k, table


def evaluate_cusum_method(step11, inputs, k):
    return evaluate_final_statistic(
        step11,
        compute_cusum(inputs["normal_validation"], k),
        compute_cusum(inputs["normal_evaluation"], k),
        compute_cusum(inputs["fault3_final_test"], k),
    )


def annotate_run_results(frame, method_key, method, k):
    out = frame.copy()
    out.insert(0, "method_key", method_key)
    out.insert(1, "method", method)
    out.insert(2, "cusum_k", np.nan if k is None else float(k))
    return out


def compare_rows(first, second, label):
    differences = {}
    for field in PARITY_FIELDS:
        differences[field] = float(first[field]) - float(second[field])
    exact = all(
        np.isclose(value, 0.0, rtol=0.0, atol=1e-12, equal_nan=True)
        for value in differences.values()
    )
    return {
        "comparison": label,
        "exact_with_tolerance_1e-12": bool(exact),
        "differences": {key: float(value) for key, value in differences.items()},
    }


def build_parity_checks(
    step11,
    input_sets,
    proposed_result_row,
):
    checks = []
    step11_path = TABLE_DIR / "11_unpaired_cusum_performance.csv"
    if step11_path.exists():
        table = pd.read_csv(step11_path)
        reference = table[
            table["screening_window"].astype(str) == PRIMARY_WINDOW
        ]
        if len(reference) == 1:
            checks.append(
                compare_rows(
                    proposed_result_row,
                    reference.iloc[0],
                    "Step 15 proposed versus Step 11 primary",
                )
            )

    diagnostic_specs = [
        (
            "all52_signed",
            "All residuals with unpaired signs + CUSUM",
            "Step 15 all-52 k=0.05 versus Step 12",
        ),
        (
            "top1_signed",
            None,
            "Step 15 Top-1 k=0.05 versus Step 13 Top-1",
        ),
    ]
    for key, step12_method, label in diagnostic_specs:
        diagnostic, _, _, _ = evaluate_cusum_method(
            step11, input_sets[key], PRIMARY_CUSUM_K
        )
        if step12_method is not None:
            path = TABLE_DIR / "12_unpaired_factorized_ablation_results.csv"
            if path.exists():
                table = pd.read_csv(path)
                reference = table[
                    table["method"].astype(str) == step12_method
                ]
                if len(reference) == 1:
                    checks.append(
                        compare_rows(diagnostic, reference.iloc[0], label)
                    )
        else:
            path = TABLE_DIR / "13_unpaired_topk_sensitivity.csv"
            if path.exists():
                table = pd.read_csv(path)
                reference = table[table["top_k"] == 1]
                if len(reference) == 1:
                    checks.append(
                        compare_rows(diagnostic, reference.iloc[0], label)
                    )
    return checks


def plot_rates(results):
    labels = results["method_short"].tolist()
    x = np.arange(len(results))
    fig, axes = plt.subplots(
        2, 1, figsize=(12.5, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6]}
    )
    axes[0].bar(
        x,
        100 * results["normal_run_level_far"],
        color="#7A9E9F",
    )
    axes[0].axhline(5.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Normal run FAR (%)")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.38
    axes[1].bar(
        x - width / 2,
        100 * results["fault3_run_level_detection_rate"],
        width,
        label="Run-level detection",
        color="#D95F59",
    )
    axes[1].bar(
        x + width / 2,
        100 * results["fault3_post_fault_point_detection_rate"],
        width,
        label="Post-fault point detection",
        color="#4C78A8",
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Detection rate (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Fair run-level-calibrated baselines for TEP Fault 3")
    fig.tight_layout()
    png = FIGURE_DIR / "fig15a_fair_baseline_rates.png"
    pdf = FIGURE_DIR / "fig15a_fair_baseline_rates.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_delays(results):
    labels = results["method_short"].tolist()
    x = np.arange(len(results))
    values = results["median_alarm_delay_min"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    bars = ax.bar(x, np.nan_to_num(values, nan=0.0), color="#4C78A8")
    for bar, value, detected in zip(
        bars, values, results["detected_runs"].astype(int)
    ):
        label = "No detection" if np.isnan(value) else f"n={detected}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if np.isnan(value) else 0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Median alarm delay among detected runs (min)")
    ax.set_title("Conditional alarm delay under fair baseline calibration")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png = FIGURE_DIR / "fig15b_fair_baseline_delays.png"
    pdf = FIGURE_DIR / "fig15b_fair_baseline_delays.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    print("=" * 80)
    print("Step 15: Fair sequential baseline comparison")
    print("=" * 80)
    print(
        "Competing CUSUM methods tune k on Fault-3 development data only; "
        "the proposed method remains frozen at K=5, k=0.05."
    )

    step11 = load_step11_module()
    variable_cols = step11.load_variable_columns()
    all_primary, stable_ranked, top1, frozen_top5 = (
        load_unpaired_definitions(variable_cols)
    )
    print(f"Frozen proposed Top-5: {frozen_top5['variable'].tolist()}")
    print(f"Frozen Top-1 comparator: {top1['variable'].tolist()}")

    data = step11.load_data()
    model = step11.fit_pca_model(data["normal_train"], variable_cols)
    eigenvalues = derive_retained_eigenvalues(
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
            compute_pca_outputs(
                data[dataset], variable_cols, model, eigenvalues
            )
        )
        if residual_cols is None:
            residual_cols = current_cols
        elif residual_cols != current_cols:
            raise ValueError("Residual column order changed across datasets")
        print(f"Computed PCA outputs: {dataset}")
    del data
    gc.collect()

    global_parameters = fit_global_transforms(
        statistics["normal_validation"]
    )
    global_inputs = {
        dataset: transform_global_inputs(stats, global_parameters)
        for dataset, stats in statistics.items()
    }

    # Build all-52 and frozen local signed inputs.
    all52_table = all_primary[
        ["variable", "residual_variable", "sign_for_positive_evidence"]
    ].copy()
    local_tables = {
        "all52_signed": all52_table,
        "top1_signed": top1,
        "proposed_top5": frozen_top5,
    }
    input_sets = {
        "pca_q": {
            dataset: global_inputs[dataset]["q"] for dataset in global_inputs
        },
        "pca_t2": {
            dataset: global_inputs[dataset]["t2"] for dataset in global_inputs
        },
        "pca_joint": {
            dataset: global_inputs[dataset]["joint"]
            for dataset in global_inputs
        },
    }
    for key, selected_table in local_tables.items():
        scaler = step11.fit_signed_scaler(
            residuals["normal_validation"], selected_table
        )
        input_sets[key] = {
            dataset: signed_local_input(step11, frame, scaler)
            for dataset, frame in residuals.items()
        }

    # Supervised global linear comparator using the same early-160 run means.
    lda_weights, lda_weight_table, lda_metadata = fit_lda_score(
        step11,
        residuals["normal_validation"],
        residuals["fault3_development"],
        residual_cols,
        variable_cols,
    )
    lda_raw = {
        dataset: apply_lda_score(frame, residual_cols, lda_weights)
        for dataset, frame in residuals.items()
    }
    lda_scaler = fit_scalar_standardizer(lda_raw["normal_validation"])
    input_sets["lda_residual"] = {
        dataset: apply_scalar_standardizer(frame, lda_scaler)
        for dataset, frame in lda_raw.items()
    }

    method_specs = [
        {
            "method_key": "pca_q_raw",
            "method": "PCA Q/SPE (raw)",
            "method_short": "Q raw",
            "input_key": "pca_q",
            "sequential": False,
            "tune_k": False,
            "uses_fault_labels": False,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "pca_t2_raw",
            "method": "PCA Hotelling T2 (raw)",
            "method_short": "T2 raw",
            "input_key": "pca_t2",
            "sequential": False,
            "tune_k": False,
            "uses_fault_labels": False,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "pca_joint_raw",
            "method": "PCA joint max-score (raw)",
            "method_short": "Joint raw",
            "input_key": "pca_joint",
            "sequential": False,
            "tune_k": False,
            "uses_fault_labels": False,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "pca_q_cusum",
            "method": "PCA Q/SPE-CUSUM",
            "method_short": "Q-CUSUM",
            "input_key": "pca_q",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "pca_t2_cusum",
            "method": "PCA T2-CUSUM",
            "method_short": "T2-CUSUM",
            "input_key": "pca_t2",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "pca_joint_cusum",
            "method": "PCA joint max-score CUSUM",
            "method_short": "Joint CUSUM",
            "input_key": "pca_joint",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "global PCA statistic",
        },
        {
            "method_key": "all52_signed_cusum",
            "method": "All-52 unpaired signed residual CUSUM",
            "method_short": "All-52 signed",
            "input_key": "all52_signed",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "global residual vector",
        },
        {
            "method_key": "lda_residual_cusum",
            "method": "Shrinkage-LDA residual CUSUM",
            "method_short": "LDA residual",
            "input_key": "lda_residual",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "global supervised residual score",
        },
        {
            "method_key": "top1_signed_cusum",
            "method": "Unpaired signed Top-1 residual CUSUM",
            "method_short": "Top-1 signed",
            "input_key": "top1_signed",
            "sequential": True,
            "tune_k": True,
            "uses_fault_labels": True,
            "scope": "local residual",
        },
        {
            "method_key": "proposed_top5_cusum",
            "method": "Proposed unpaired signed Top-5 residual CUSUM",
            "method_short": "Proposed Top-5",
            "input_key": "proposed_top5",
            "sequential": True,
            "tune_k": False,
            "fixed_k": PRIMARY_CUSUM_K,
            "uses_fault_labels": True,
            "scope": "local residual subset",
        },
    ]

    result_rows = []
    tuning_tables = []
    normal_run_tables = []
    fault_run_tables = []
    for spec in method_specs:
        inputs = input_sets[spec["input_key"]]
        if not spec["sequential"]:
            selected_k = None
            metrics, _, normal_runs, fault_runs = evaluate_raw_method(
                step11, inputs
            )
            tuning_source = "not applicable (raw chart)"
        else:
            if spec["tune_k"]:
                selected_k, tuning = tune_cusum_k_on_development(
                    step11, spec["method_key"], inputs
                )
                tuning.insert(1, "method", spec["method"])
                tuning_tables.append(tuning)
                tuning_source = "Fault-3 development set"
            else:
                selected_k = float(spec["fixed_k"])
                tuning_source = "frozen primary value"
            metrics, _, normal_runs, fault_runs = evaluate_cusum_method(
                step11, inputs, selected_k
            )
        row = {
            "method_key": spec["method_key"],
            "method": spec["method"],
            "method_short": spec["method_short"],
            "scope": spec["scope"],
            "sequential": bool(spec["sequential"]),
            "uses_fault3_development_labels": bool(
                spec["uses_fault_labels"]
            ),
            "cusum_k": np.nan if selected_k is None else float(selected_k),
            "cusum_k_source": tuning_source,
            "run_level_percentile": RUN_LEVEL_PERCENTILE,
        }
        row.update(metrics)
        result_rows.append(row)
        normal_run_tables.append(
            annotate_run_results(
                normal_runs,
                spec["method_key"],
                spec["method"],
                selected_k,
            )
        )
        fault_run_tables.append(
            annotate_run_results(
                fault_runs,
                spec["method_key"],
                spec["method"],
                selected_k,
            )
        )
        print(
            f"{spec['method']}: k={selected_k}, FAR="
            f"{100*metrics['normal_run_level_far']:.2f}%, detection="
            f"{100*metrics['fault3_run_level_detection_rate']:.2f}%, point="
            f"{100*metrics['fault3_post_fault_point_detection_rate']:.4f}%, "
            f"delay={metrics['median_alarm_delay_min']} min"
        )

    results = pd.DataFrame(result_rows)
    tuning_results = (
        pd.concat(tuning_tables, ignore_index=True)
        if tuning_tables
        else pd.DataFrame()
    )
    normal_run_results = pd.concat(normal_run_tables, ignore_index=True)
    fault_run_results = pd.concat(fault_run_tables, ignore_index=True)
    definitions = pd.DataFrame(
        [
            {
                "method_key": spec["method_key"],
                "method": spec["method"],
                "method_short": spec["method_short"],
                "scope": spec["scope"],
                "sequential": spec["sequential"],
                "k_tuned_on_development": spec["tune_k"],
                "uses_fault3_development_labels": spec[
                    "uses_fault_labels"
                ],
                "input_transform": (
                    "log1p, normal-validation z-score"
                    if spec["input_key"] in {"pca_q", "pca_t2"}
                    else (
                        "max of standardized log Q and log T2, then "
                        "normal-validation z-score"
                        if spec["input_key"] == "pca_joint"
                        else "method-specific normal-validation standardization"
                    )
                ),
            }
            for spec in method_specs
        ]
    )

    proposed_row = results[
        results["method_key"] == "proposed_top5_cusum"
    ].iloc[0]
    parity_checks = build_parity_checks(
        step11, input_sets, proposed_row
    )

    parameter_rows = [
        {"parameter": key, "value": value}
        for key, value in global_parameters.items()
    ]
    parameter_rows += [
        {"parameter": "lda_score_normal_mean", "value": lda_scaler["mean"]},
        {"parameter": "lda_score_normal_std", "value": lda_scaler["std"]},
        {
            "parameter": "ledoit_wolf_shrinkage",
            "value": lda_metadata["ledoit_wolf_shrinkage"],
        },
    ]
    transform_parameters = pd.DataFrame(parameter_rows)

    paths = {
        "results": TABLE_DIR / "15_fair_sequential_baseline_results.csv",
        "definitions": TABLE_DIR / "15_fair_baseline_definitions.csv",
        "development_tuning": TABLE_DIR
        / "15_cusum_k_development_tuning.csv",
        "normal_run_results": TABLE_DIR
        / "15_fair_baseline_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / "15_fair_baseline_fault3_run_results.csv",
        "lda_weights": TABLE_DIR / "15_shrinkage_lda_residual_weights.csv",
        "transform_parameters": TABLE_DIR
        / "15_scalar_transform_parameters.csv",
    }
    results.to_csv(paths["results"], index=False, encoding="utf-8-sig")
    definitions.to_csv(
        paths["definitions"], index=False, encoding="utf-8-sig"
    )
    tuning_results.to_csv(
        paths["development_tuning"], index=False, encoding="utf-8-sig"
    )
    normal_run_results.to_csv(
        paths["normal_run_results"], index=False, encoding="utf-8-sig"
    )
    fault_run_results.to_csv(
        paths["fault3_run_results"], index=False, encoding="utf-8-sig"
    )
    lda_weight_table.to_csv(
        paths["lda_weights"], index=False, encoding="utf-8-sig"
    )
    transform_parameters.to_csv(
        paths["transform_parameters"], index=False, encoding="utf-8-sig"
    )

    figure_paths = plot_rates(results) + plot_delays(results)
    config = {
        "script": "scripts/15_fair_sequential_baselines.py",
        "purpose": (
            "Fair global/sequential/supervised comparisons under common "
            "run-level calibration"
        ),
        "primary_method_frozen": {
            "top_k": PRIMARY_TOP_K,
            "cusum_k": PRIMARY_CUSUM_K,
            "selected_variables": frozen_top5["variable"].tolist(),
            "selected_signs": frozen_top5[
                "sign_for_positive_evidence"
            ].astype(int).tolist(),
        },
        "competitor_k_grid": CUSUM_K_GRID,
        "competitor_k_selection_data": "Fault-3 development runs 1-250",
        "competitor_k_selection_rule": (
            "lexicographic: maximize development run detection, maximize "
            "development post-fault point detection, minimize median delay, "
            "then prefer k nearest 0.05"
        ),
        "final_test_used_for_k_selection": False,
        "run_level_control_limit_rule": (
            "95th percentile of normal-validation run maxima for every method"
        ),
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "global_input_transform": (
            "fixed log1p followed by normal-validation standardization"
        ),
        "lda": lda_metadata,
        "parity_checks": parity_checks,
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "fault3_fair_sequential_baselines.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 15 summary")
    print("=" * 80)
    print(
        results[
            [
                "method",
                "cusum_k",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
                "detected_runs",
            ]
        ].to_string(index=False)
    )
    print(f"\nParity checks: {parity_checks}")
    print("\nLargest absolute shrinkage-LDA weights:")
    print(
        lda_weight_table[
            ["abs_weight_rank", "variable", "lda_weight", "unpaired_mean_shift"]
        ].head(10).to_string(index=False)
    )
    print(f"\nSaved configuration: {config_path}")
    print("Step 15 finished successfully.")


if __name__ == "__main__":
    main()
