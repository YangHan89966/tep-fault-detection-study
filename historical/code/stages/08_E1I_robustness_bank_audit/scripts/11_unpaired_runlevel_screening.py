"""
Step 11: Pairing-free run-level residual screening and final evaluation.

Motivation
----------
Step 10 diagnoses whether pointwise matched subtraction is essential to the
original Step 04 direction-ratio screen.  This script provides a replacement
that never pairs a normal run with a Fault 3 run:

1. PCA residuals are averaged within each independent simulation run over a
   prespecified post-fault development window.
2. Normal and Fault 3 run means are resampled independently (run-level
   bootstrap; no sample is treated as an independent replicate).
3. A residual is direction-stable when its two-sided 95% bootstrap percentile
   interval for the difference in run means excludes zero.
4. Stable residuals are ranked by absolute Hedges' g and the Top-5 are frozen.
5. The selected signed residuals are evaluated with the unchanged CUSUM rule
   on untouched normal-evaluation and Fault-3 final-test runs.

Three development windows are evaluated without choosing among them using the
final test: early 80 samples, early 160 samples (primary), and the full
post-fault period.  The original matched Top-5 is recomputed as a reference.

Place in:
    TEP_Fault3_Project/scripts/11_unpaired_runlevel_screening.py

Run from project root:
    python scripts/11_unpaired_runlevel_screening.py

Only files beginning with ``11_`` are written.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
PCA_CUMVAR_TARGET = 0.95
N_SELECTED_VARIABLES = 5

CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BATCH_SIZE = 200
BOOTSTRAP_CONFIDENCE = 0.95
RANDOM_SEED = 20260714

# Early-160 is the prespecified primary unpaired screen.  The other two are
# reported as window-sensitivity analyses, not alternatives selected by final
# test performance.
SCREENING_WINDOWS = {
    "early_80": (161, 240),
    "early_160_primary": (161, 320),
    "full_postfault": (161, 960),
}


def percentile_compat(values, q):
    try:
        return np.percentile(values, q, method="linear")
    except TypeError:
        return np.percentile(values, q, interpolation="linear")


def clopper_pearson(successes, trials, confidence=0.95):
    successes, trials = int(successes), int(trials)
    if trials <= 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    high = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return low, high


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_variable_columns():
    path = require_file(
        TABLE_DIR / "02_variable_list.csv",
        "run scripts/02_prepare_fault3_data.py",
    )
    table = pd.read_csv(path)
    variables = table["variable"].astype(str).tolist()
    if len(variables) != 52:
        raise ValueError(f"Expected 52 variables, found {len(variables)}")
    return variables


def load_data():
    paths = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR
        / "normal_validation_runs_1_250.pkl",
        "normal_evaluation": PROCESSED_DIR
        / "normal_evaluation_runs_251_500.pkl",
        "fault3_development": PROCESSED_DIR
        / "fault3_development_runs_1_250.pkl",
        "fault3_final_test": PROCESSED_DIR
        / "fault3_final_test_runs_251_500.pkl",
    }
    for path in paths.values():
        require_file(path, "run scripts/02_prepare_fault3_data.py")

    print("Loading processed datasets...")
    data = {name: pd.read_pickle(path) for name, path in paths.items()}
    for name, frame in data.items():
        print(f"  {name}: shape={frame.shape}")
    return data


def fit_pca_model(normal_train, variable_cols):
    x = normal_train[variable_cols].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    x_std = (x - mean) / std

    pca = PCA()
    pca.fit(x_std)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumulative, PCA_CUMVAR_TARGET) + 1)
    model = {
        "mean": mean,
        "std": std,
        "components": pca.components_[:n_pcs, :],
        "n_pcs": n_pcs,
        "cum_var": float(cumulative[n_pcs - 1]),
    }
    print(
        f"PCA model: {n_pcs} PCs, cumulative explained variance "
        f"{model['cum_var']:.6f}"
    )
    return model


def compute_residual_frame(frame, variable_cols, model):
    x = frame[variable_cols].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    loadings = model["components"]
    residual = x_std - (x_std @ loadings.T) @ loadings
    residual_cols = [f"res_{v}" for v in variable_cols]

    ids = frame[["faultNumber", "simulationRun", "sample"]].reset_index(
        drop=True
    )
    values = pd.DataFrame(residual, columns=residual_cols, index=ids.index)
    return pd.concat([ids, values], axis=1), residual_cols


def calculate_run_means(residual_df, residual_cols, start_sample, end_sample):
    window = residual_df[
        residual_df["sample"].between(start_sample, end_sample)
    ]
    if window.empty:
        raise ValueError(f"No samples found in window {start_sample}-{end_sample}")

    expected_count = end_sample - start_sample + 1
    counts = window.groupby("simulationRun").size()
    if not (counts == expected_count).all():
        raise ValueError(
            f"Incomplete runs in screening window {start_sample}-{end_sample}"
        )
    return window.groupby("simulationRun")[residual_cols].mean().sort_index()


def independent_run_bootstrap(normal_run_means, fault_run_means, seed):
    """Bootstrap differences in means by independently resampling whole runs."""
    normal = normal_run_means.to_numpy(dtype=float)
    fault = fault_run_means.to_numpy(dtype=float)
    n_normal, n_variables = normal.shape
    n_fault = fault.shape[0]
    rng = np.random.default_rng(seed)
    bootstrap_delta = np.empty(
        (BOOTSTRAP_REPLICATES, n_variables), dtype=float
    )

    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPLICATES)
        batch = stop - start
        normal_indices = rng.integers(0, n_normal, size=(batch, n_normal))
        fault_indices = rng.integers(0, n_fault, size=(batch, n_fault))
        normal_boot_mean = normal[normal_indices].mean(axis=1)
        fault_boot_mean = fault[fault_indices].mean(axis=1)
        bootstrap_delta[start:stop] = fault_boot_mean - normal_boot_mean

    return bootstrap_delta


def screen_unpaired_runlevel(
    normal_run_means,
    fault_run_means,
    variable_cols,
    window_name,
    start_sample,
    end_sample,
    seed,
):
    normal = normal_run_means.to_numpy(dtype=float)
    fault = fault_run_means.to_numpy(dtype=float)
    n_normal, n_fault = normal.shape[0], fault.shape[0]

    mean_normal = normal.mean(axis=0)
    mean_fault = fault.mean(axis=0)
    mean_delta = mean_fault - mean_normal
    var_normal = normal.var(axis=0, ddof=1)
    var_fault = fault.var(axis=0, ddof=1)
    pooled_variance = (
        (n_normal - 1) * var_normal + (n_fault - 1) * var_fault
    ) / (n_normal + n_fault - 2)
    pooled_std = np.sqrt(pooled_variance)
    cohen_d = np.divide(
        mean_delta,
        pooled_std,
        out=np.full_like(mean_delta, np.nan),
        where=pooled_std > 0,
    )
    hedges_correction = 1.0 - 3.0 / (4.0 * (n_normal + n_fault) - 9.0)
    hedges_g = hedges_correction * cohen_d

    bootstrap_delta = independent_run_bootstrap(
        normal_run_means, fault_run_means, seed
    )
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    ci_low = np.quantile(bootstrap_delta, alpha / 2.0, axis=0)
    ci_high = np.quantile(bootstrap_delta, 1.0 - alpha / 2.0, axis=0)
    positive_probability = (bootstrap_delta > 0).mean(axis=0)
    negative_probability = (bootstrap_delta < 0).mean(axis=0)
    direction_confidence = np.maximum(
        positive_probability, negative_probability
    )
    direction_stable = (ci_low > 0) | (ci_high < 0)
    sign = np.where(mean_delta >= 0, 1, -1)

    # Selection frequency is calculated within the prespecified stable set.
    stable_indices = np.flatnonzero(direction_stable)
    selection_counts = np.zeros(len(variable_cols), dtype=int)
    if len(stable_indices):
        safe_pooled_std = pooled_std.copy()
        safe_pooled_std[safe_pooled_std == 0] = np.nan
        boot_effect = np.abs(bootstrap_delta / safe_pooled_std)
        n_take = min(N_SELECTED_VARIABLES, len(stable_indices))
        for row in boot_effect:
            scores = row[stable_indices]
            finite_scores = np.nan_to_num(scores, nan=-np.inf)
            order = np.argsort(finite_scores)[::-1][:n_take]
            selection_counts[stable_indices[order]] += 1

    metrics = pd.DataFrame(
        {
            "window": window_name,
            "window_start_sample": start_sample,
            "window_end_sample": end_sample,
            "window_n_samples": end_sample - start_sample + 1,
            "variable": variable_cols,
            "residual_variable": [f"res_{v}" for v in variable_cols],
            "normal_run_mean": mean_normal,
            "fault3_run_mean": mean_fault,
            "unpaired_mean_shift": mean_delta,
            "pooled_run_mean_std": pooled_std,
            "cohen_d": cohen_d,
            "hedges_g": hedges_g,
            "abs_hedges_g": np.abs(hedges_g),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "bootstrap_positive_probability": positive_probability,
            "bootstrap_negative_probability": negative_probability,
            "bootstrap_direction_confidence": direction_confidence,
            "direction_stable_95ci": direction_stable,
            "dominant_direction": np.where(
                mean_delta >= 0, "positive", "negative"
            ),
            "sign_for_positive_evidence": sign,
            "bootstrap_top5_selection_frequency": selection_counts
            / BOOTSTRAP_REPLICATES,
        }
    )

    stable = metrics[metrics["direction_stable_95ci"]].copy()
    selected = stable.sort_values(
        [
            "abs_hedges_g",
            "bootstrap_direction_confidence",
            "bootstrap_top5_selection_frequency",
            "variable",
        ],
        ascending=[False, False, False, True],
    ).head(N_SELECTED_VARIABLES).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))

    return metrics, stable, selected


def fit_signed_scaler(normal_validation_res, selected):
    rows = []
    for _, selected_row in selected.iterrows():
        col = selected_row["residual_variable"]
        sign = int(selected_row["sign_for_positive_evidence"])
        signed = sign * normal_validation_res[col].to_numpy(dtype=float)
        mean = float(signed.mean())
        std = float(signed.std(ddof=1))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        rows.append(
            {
                "variable": selected_row["variable"],
                "residual_variable": col,
                "sign_for_positive_evidence": sign,
                "normal_validation_signed_mean": mean,
                "normal_validation_signed_std": std,
            }
        )
    return pd.DataFrame(rows)


def construct_local_signed_index(residual_df, scaler):
    out = residual_df[["faultNumber", "simulationRun", "sample"]].copy()
    z_values = []
    for _, row in scaler.iterrows():
        signed = (
            int(row["sign_for_positive_evidence"])
            * residual_df[row["residual_variable"]].to_numpy(dtype=float)
        )
        z = (
            signed - float(row["normal_validation_signed_mean"])
        ) / float(row["normal_validation_signed_std"])
        z_values.append(z)
    out["local_signed_residual"] = np.column_stack(z_values).mean(axis=1)
    return out


def compute_cusum(local_df):
    df = local_df.sort_values(["simulationRun", "sample"]).copy()
    values_out = np.empty(len(df), dtype=float)
    for positions in df.groupby("simulationRun", sort=True).indices.values():
        values = df.iloc[positions]["local_signed_residual"].to_numpy(dtype=float)
        run_cusum = np.empty(len(values), dtype=float)
        current = 0.0
        for i, value in enumerate(values):
            current = max(0.0, current + value - CUSUM_K)
            run_cusum[i] = current
        values_out[positions] = run_cusum
    df["monitoring_statistic"] = values_out
    return df


def determine_limit(normal_validation_cusum):
    run_max = (
        normal_validation_cusum.groupby("simulationRun")[
            "monitoring_statistic"
        ]
        .max()
        .reset_index(name="run_max_statistic")
    )
    limit = float(
        percentile_compat(
            run_max["run_max_statistic"].to_numpy(), RUN_LEVEL_PERCENTILE
        )
    )
    return limit, run_max


def evaluate_normal(cusum_df, limit):
    df = cusum_df.copy()
    df["alarm"] = df["monitoring_statistic"] > limit
    run_results = (
        df.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="run_alarm")
    )
    alarm_runs = int(run_results["run_alarm"].sum())
    n_runs = int(len(run_results))
    ci_low, ci_high = clopper_pearson(alarm_runs, n_runs)
    return {
        "normal_n_runs": n_runs,
        "normal_alarm_runs": alarm_runs,
        "normal_run_level_far": alarm_runs / n_runs,
        "normal_run_level_far_ci_low": ci_low,
        "normal_run_level_far_ci_high": ci_high,
        "normal_point_level_far": float(df["alarm"].mean()),
    }, run_results


def evaluate_fault(cusum_df, limit):
    df = cusum_df.copy()
    df["alarm"] = df["monitoring_statistic"] > limit
    pre = df[df["sample"] < FAULT_START_SAMPLE].copy()
    post = df[df["sample"] >= FAULT_START_SAMPLE].copy()

    pre_alarm = (
        pre.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="pre_fault_run_alarm")
    )
    post_alarm = (
        post.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="post_fault_run_alarm")
    )
    first_alarm = (
        post[post["alarm"]]
        .groupby("simulationRun")["sample"]
        .min()
        .reset_index(name="first_alarm_sample")
    )
    runs = pd.DataFrame(
        {"simulationRun": sorted(df["simulationRun"].unique())}
    )
    run_results = runs.merge(pre_alarm, on="simulationRun", how="left")
    run_results = run_results.merge(post_alarm, on="simulationRun", how="left")
    run_results = run_results.merge(first_alarm, on="simulationRun", how="left")
    for col in ["pre_fault_run_alarm", "post_fault_run_alarm"]:
        run_results[col] = run_results[col].fillna(False).astype(bool)
    run_results["detected"] = run_results["post_fault_run_alarm"]
    run_results["alarm_delay_min"] = (
        run_results["first_alarm_sample"] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN

    n_runs = int(len(run_results))
    pre_alarm_runs = int(run_results["pre_fault_run_alarm"].sum())
    detected_runs = int(run_results["detected"].sum())
    pre_low, pre_high = clopper_pearson(pre_alarm_runs, n_runs)
    det_low, det_high = clopper_pearson(detected_runs, n_runs)
    delays = run_results.loc[run_results["detected"], "alarm_delay_min"].dropna()
    metrics = {
        "fault3_n_runs": n_runs,
        "fault3_pre_fault_alarm_runs": pre_alarm_runs,
        "fault3_pre_fault_run_far": pre_alarm_runs / n_runs,
        "fault3_pre_fault_run_far_ci_low": pre_low,
        "fault3_pre_fault_run_far_ci_high": pre_high,
        "fault3_pre_fault_point_far": float(pre["alarm"].mean()),
        "detected_runs": detected_runs,
        "fault3_run_level_detection_rate": detected_runs / n_runs,
        "fault3_run_level_detection_ci_low": det_low,
        "fault3_run_level_detection_ci_high": det_high,
        "fault3_post_fault_point_detection_rate": float(post["alarm"].mean()),
        "median_alarm_delay_min": float(delays.median())
        if len(delays)
        else np.nan,
        "mean_alarm_delay_min": float(delays.mean()) if len(delays) else np.nan,
        "alarm_delay_q1_min": float(delays.quantile(0.25))
        if len(delays)
        else np.nan,
        "alarm_delay_q3_min": float(delays.quantile(0.75))
        if len(delays)
        else np.nan,
    }
    return metrics, run_results


def evaluate_selected_method(
    method_name,
    window_name,
    selected,
    normal_validation_res,
    normal_evaluation_res,
    fault_final_res,
):
    if len(selected) == 0:
        raise ValueError(f"{method_name} selected no residual variables")
    scaler = fit_signed_scaler(normal_validation_res, selected)
    validation_cusum = compute_cusum(
        construct_local_signed_index(normal_validation_res, scaler)
    )
    normal_cusum = compute_cusum(
        construct_local_signed_index(normal_evaluation_res, scaler)
    )
    fault_cusum = compute_cusum(
        construct_local_signed_index(fault_final_res, scaler)
    )
    limit, validation_run_max = determine_limit(validation_cusum)
    normal_metrics, normal_runs = evaluate_normal(normal_cusum, limit)
    fault_metrics, fault_runs = evaluate_fault(fault_cusum, limit)

    metrics = {
        "method": method_name,
        "screening_window": window_name,
        "n_selected_variables": int(len(selected)),
        "selected_variables": ";".join(selected["variable"].astype(str)),
        "selected_signs": ";".join(
            selected["sign_for_positive_evidence"].astype(int).astype(str)
        ),
        "cusum_k": CUSUM_K,
        "control_limit": limit,
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


def save_method_outputs(slug, result):
    outputs = {
        "scaler": TABLE_DIR / f"11_{slug}_scaler.csv",
        "validation_run_max": TABLE_DIR
        / f"11_{slug}_validation_run_max.csv",
        "normal_run_results": TABLE_DIR
        / f"11_{slug}_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / f"11_{slug}_fault3_run_results.csv",
    }
    for key, path in outputs.items():
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
    return {key: str(path.relative_to(PROJECT_ROOT)) for key, path in outputs.items()}


def plot_selection_stability(selected_all):
    primary = selected_all[
        selected_all["window"] == "early_160_primary"
    ].copy()
    primary = primary.sort_values("selection_rank")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        primary["variable"],
        100 * primary["bootstrap_top5_selection_frequency"],
        color="#4C78A8",
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Bootstrap Top-5 selection frequency (%)")
    ax.set_title("Pairing-free run-level screening: primary early window")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    png = FIGURE_DIR / "fig11a_unpaired_selection_stability.png"
    pdf = FIGURE_DIR / "fig11a_unpaired_selection_stability.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_performance(performance):
    labels = performance["screening_window"].replace(
        {"matched_original": "Matched original"}
    )
    x = np.arange(len(performance))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(
        x - width / 2,
        100 * performance["normal_run_level_far"],
        width,
        label="Normal run-level FAR",
        color="#7A9E9F",
    )
    ax.bar(
        x + width / 2,
        100 * performance["fault3_run_level_detection_rate"],
        width,
        label="Fault 3 run-level detection",
        color="#D95F59",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Matched and pairing-free screening: final-test performance")
    ax.legend()
    fig.tight_layout()
    png = FIGURE_DIR / "fig11b_unpaired_screening_performance.png"
    pdf = FIGURE_DIR / "fig11b_unpaired_screening_performance.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    print("=" * 80)
    print("Step 11: Pairing-free run-level residual screening")
    print("=" * 80)

    variable_cols = load_variable_columns()
    data = load_data()
    model = fit_pca_model(data["normal_train"], variable_cols)

    print("Computing PCA residuals for four development/evaluation datasets...")
    normal_validation_res, residual_cols = compute_residual_frame(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = compute_residual_frame(
        data["normal_evaluation"], variable_cols, model
    )
    fault_development_res, _ = compute_residual_frame(
        data["fault3_development"], variable_cols, model
    )
    fault_final_res, _ = compute_residual_frame(
        data["fault3_final_test"], variable_cols, model
    )

    all_metrics = []
    all_selected = []
    stable_counts = {}
    selected_by_window = {}
    for i, (window_name, (start_sample, end_sample)) in enumerate(
        SCREENING_WINDOWS.items()
    ):
        print("\n" + "=" * 80)
        print(
            f"Unpaired screen: {window_name}, samples "
            f"{start_sample}-{end_sample}"
        )
        print("=" * 80)
        normal_run_means = calculate_run_means(
            normal_validation_res,
            residual_cols,
            start_sample,
            end_sample,
        )
        fault_run_means = calculate_run_means(
            fault_development_res,
            residual_cols,
            start_sample,
            end_sample,
        )
        metrics, stable, selected = screen_unpaired_runlevel(
            normal_run_means,
            fault_run_means,
            variable_cols,
            window_name,
            start_sample,
            end_sample,
            RANDOM_SEED + i,
        )
        stable_counts[window_name] = int(len(stable))
        selected_by_window[window_name] = selected
        all_metrics.append(metrics)
        all_selected.append(selected)
        print(f"Stable residuals (95% bootstrap CI excludes zero): {len(stable)}")
        print(f"Selected Top-5: {selected['variable'].tolist()}")
        print(
            selected[
                [
                    "selection_rank",
                    "variable",
                    "dominant_direction",
                    "hedges_g",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "bootstrap_top5_selection_frequency",
                ]
            ].to_string(index=False)
        )

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    selected_all = pd.concat(all_selected, ignore_index=True)
    screening_metrics_path = TABLE_DIR / "11_unpaired_screening_all_variables.csv"
    selected_path = TABLE_DIR / "11_unpaired_selected_variables.csv"
    metrics_all.to_csv(
        screening_metrics_path, index=False, encoding="utf-8-sig"
    )
    selected_all.to_csv(selected_path, index=False, encoding="utf-8-sig")

    existing_selected_path = require_file(
        TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv",
        "run scripts/04_local_residual_screening.py",
    )
    matched_selected = pd.read_csv(existing_selected_path)
    matched_selected = matched_selected.head(N_SELECTED_VARIABLES).copy()

    method_specs = [
        (
            "Original matched Top-5 signed residual CUSUM",
            "matched_original",
            "matched_original",
            matched_selected,
        )
    ]
    for window_name, selected in selected_by_window.items():
        method_specs.append(
            (
                f"Unpaired run-level Top-5 ({window_name})",
                window_name,
                f"unpaired_{window_name}",
                selected,
            )
        )

    performance_rows = []
    method_outputs = {}
    original_set = set(matched_selected["variable"].astype(str))
    print("\n" + "=" * 80)
    print("Final evaluation of frozen selected sets")
    print("=" * 80)
    for method_name, window_name, slug, selected in method_specs:
        if len(selected) == 0:
            print(
                f"{window_name}: no residual passed the unpaired 95% bootstrap "
                "stability rule; final CUSUM evaluation skipped"
            )
            performance_rows.append(
                {
                    "method": method_name,
                    "screening_window": window_name,
                    "n_selected_variables": 0,
                    "selected_variables": "",
                    "selected_signs": "",
                    "overlap_with_original_top5": 0,
                    "jaccard_with_original_top5": 0.0,
                    "normal_run_level_far": np.nan,
                    "fault3_run_level_detection_rate": np.nan,
                    "fault3_post_fault_point_detection_rate": np.nan,
                    "median_alarm_delay_min": np.nan,
                }
            )
            method_outputs[window_name] = {}
            continue
        result = evaluate_selected_method(
            method_name,
            window_name,
            selected,
            normal_validation_res,
            normal_evaluation_res,
            fault_final_res,
        )
        metrics = result["metrics"]
        selected_set = set(selected["variable"].astype(str))
        metrics["overlap_with_original_top5"] = len(
            selected_set & original_set
        )
        metrics["jaccard_with_original_top5"] = len(
            selected_set & original_set
        ) / len(selected_set | original_set)
        performance_rows.append(metrics)
        method_outputs[window_name] = save_method_outputs(slug, result)
        print(
            f"{window_name}: selected={sorted(selected_set)}, "
            f"normal FAR={100 * metrics['normal_run_level_far']:.2f}%, "
            f"Fault 3 detection="
            f"{100 * metrics['fault3_run_level_detection_rate']:.2f}%, "
            f"post-fault point detection="
            f"{100 * metrics['fault3_post_fault_point_detection_rate']:.2f}%, "
            f"median delay={metrics['median_alarm_delay_min']} min"
        )

    performance = pd.DataFrame(performance_rows)
    performance_path = TABLE_DIR / "11_unpaired_cusum_performance.csv"
    performance.to_csv(performance_path, index=False, encoding="utf-8-sig")

    overlap_rows = []
    for left_name, left_selected in [
        ("matched_original", matched_selected),
        *list(selected_by_window.items()),
    ]:
        left_set = set(left_selected["variable"].astype(str))
        for right_name, right_selected in [
            ("matched_original", matched_selected),
            *list(selected_by_window.items()),
        ]:
            right_set = set(right_selected["variable"].astype(str))
            union = left_set | right_set
            overlap_rows.append(
                {
                    "set_a": left_name,
                    "set_b": right_name,
                    "overlap_count": len(left_set & right_set),
                    "jaccard": len(left_set & right_set) / len(union)
                    if union
                    else 1.0,
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap_path = TABLE_DIR / "11_unpaired_selection_overlap.csv"
    overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")

    fig11a = plot_selection_stability(selected_all)
    fig11b = plot_performance(performance)

    config_path = CONFIG_DIR / "fault3_unpaired_runlevel_screening_config.json"
    config = {
        "script": "scripts/11_unpaired_runlevel_screening.py",
        "method": (
            "Independent run-level bootstrap screening; no normal-fault run "
            "pairing and no sample-level bootstrap"
        ),
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "pca_cumulative_variance_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "screening_windows": SCREENING_WINDOWS,
        "primary_window": "early_160_primary",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "independent simulation run",
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_seed": RANDOM_SEED,
        "direction_stability_rule": (
            "two-sided 95% percentile bootstrap CI for the unpaired difference "
            "in run means excludes zero"
        ),
        "ranking_rule": "descending absolute Hedges' g among stable residuals",
        "top_k": N_SELECTED_VARIABLES,
        "cusum_k": CUSUM_K,
        "run_level_threshold_percentile": RUN_LEVEL_PERCENTILE,
        "stable_counts": stable_counts,
        "selected_variables": {
            name: selected["variable"].tolist()
            for name, selected in selected_by_window.items()
        },
        "method_outputs": method_outputs,
        "summary_outputs": {
            "all_screening_metrics": str(
                screening_metrics_path.relative_to(PROJECT_ROOT)
            ),
            "selected_variables": str(selected_path.relative_to(PROJECT_ROOT)),
            "selection_overlap": str(overlap_path.relative_to(PROJECT_ROOT)),
            "final_performance": str(performance_path.relative_to(PROJECT_ROOT)),
            "figures": [
                str(path.relative_to(PROJECT_ROOT))
                for path in [*fig11a, *fig11b]
            ],
        },
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 11 summary")
    print("=" * 80)
    print(
        performance[
            [
                "screening_window",
                "selected_variables",
                "overlap_with_original_top5",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print(f"Saved screening metrics: {screening_metrics_path}")
    print(f"Saved selected variables: {selected_path}")
    print(f"Saved final performance: {performance_path}")
    print(f"Saved configuration: {config_path}")
    print("Step 11 finished successfully.")


if __name__ == "__main__":
    main()
