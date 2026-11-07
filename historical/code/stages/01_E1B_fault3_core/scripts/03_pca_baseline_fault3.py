from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
PCA_CUMVAR_TARGET = 0.95
RUN_LEVEL_PERCENTILE = 95


def percentile_compat(x, q):
    """Compatible percentile function for different NumPy versions."""
    try:
        return np.percentile(x, q, method="linear")
    except TypeError:
        return np.percentile(x, q, interpolation="linear")


def load_variable_columns():
    variable_path = TABLE_DIR / "02_variable_list.csv"
    if not variable_path.exists():
        raise FileNotFoundError(
            f"Variable list not found: {variable_path}. "
            "Please run scripts/02_prepare_fault3_data.py first."
        )

    variable_info = pd.read_csv(variable_path)
    variable_cols = variable_info["variable"].tolist()
    return variable_cols


def load_processed_data():
    files = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR / "normal_validation_runs_1_250.pkl",
        "normal_evaluation": PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl",
        "fault3_development": PROCESSED_DIR / "fault3_development_runs_1_250.pkl",
        "fault3_final_test": PROCESSED_DIR / "fault3_final_test_runs_251_500.pkl",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {path}. "
                "Please run scripts/02_prepare_fault3_data.py first."
            )

    print("Loading processed data...")
    data = {name: pd.read_pickle(path) for name, path in files.items()}

    for name, df in data.items():
        print(f"{name}: shape={df.shape}")

    return data


def fit_pca_model(df_train, variable_cols):
    """
    Fit PCA model using only normal training data.

    PCA is built in standardized variable space.
    The number of retained PCs is selected by 95% cumulative explained variance.
    """
    print("\n" + "=" * 80)
    print("Fitting PCA model on normal training data")
    print("=" * 80)

    X_train = df_train[variable_cols].to_numpy(dtype=float)

    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0, ddof=1)
    std_[std_ == 0] = 1.0

    X_train_std = (X_train - mean_) / std_

    pca_full = PCA()
    pca_full.fit(X_train_std)

    explained_ratio = pca_full.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    n_pcs = int(np.searchsorted(cumulative_ratio, PCA_CUMVAR_TARGET) + 1)
    cum_var = float(cumulative_ratio[n_pcs - 1])

    components = pca_full.components_[:n_pcs, :]
    eigenvalues = pca_full.explained_variance_[:n_pcs]

    print(f"Target cumulative variance: {PCA_CUMVAR_TARGET}")
    print(f"Selected PCs: {n_pcs}")
    print(f"Cumulative explained variance: {cum_var:.6f}")

    model = {
        "mean": mean_,
        "std": std_,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio,
        "n_pcs": n_pcs,
        "cum_var": cum_var,
    }

    model_summary = pd.DataFrame(
        [
            {
                "pca_cumvar_target": PCA_CUMVAR_TARGET,
                "n_variables": len(variable_cols),
                "n_pcs": n_pcs,
                "cumulative_explained_variance": cum_var,
            }
        ]
    )

    model_summary_path = TABLE_DIR / "03_pca_model_summary.csv"
    model_summary.to_csv(model_summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved PCA model summary: {model_summary_path}")

    explained_df = pd.DataFrame(
        {
            "pc": np.arange(1, len(explained_ratio) + 1),
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance": cumulative_ratio,
        }
    )
    explained_path = TABLE_DIR / "03_pca_explained_variance.csv"
    explained_df.to_csv(explained_path, index=False, encoding="utf-8-sig")
    print(f"Saved explained variance: {explained_path}")

    return model


def compute_pca_statistics(df, variable_cols, model):
    """
    Compute PCA Q/SPE and Hotelling's T2 for a dataframe.
    """
    X = df[variable_cols].to_numpy(dtype=float)
    X_std = (X - model["mean"]) / model["std"]

    P = model["components"]
    lambdas = model["eigenvalues"]

    scores = X_std @ P.T
    X_recon = scores @ P

    residual = X_std - X_recon

    Q = np.sum(residual ** 2, axis=1)
    T2 = np.sum((scores ** 2) / lambdas, axis=1)

    stats = df[["faultNumber", "simulationRun", "sample"]].copy()
    stats["Q"] = Q
    stats["T2"] = T2

    return stats


def determine_run_level_limits(normal_validation_stats):
    """
    Determine run-level PCA thresholds from normal validation data.

    For each normal validation run, calculate the maximum Q and maximum T2.
    The control limit is the 95th percentile of these run-level maxima.
    """
    print("\n" + "=" * 80)
    print("Determining run-level PCA control limits")
    print("=" * 80)

    run_max = (
        normal_validation_stats.groupby("simulationRun")[["Q", "T2"]]
        .max()
        .reset_index()
    )

    q_limit = float(percentile_compat(run_max["Q"].to_numpy(), RUN_LEVEL_PERCENTILE))
    t2_limit = float(percentile_compat(run_max["T2"].to_numpy(), RUN_LEVEL_PERCENTILE))

    print(f"Run-level percentile: {RUN_LEVEL_PERCENTILE}%")
    print(f"Q limit: {q_limit:.6f}")
    print(f"T2 limit: {t2_limit:.6f}")

    thresholds = pd.DataFrame(
        [
            {
                "threshold_source": "normal_validation_runs_1_250",
                "run_level_percentile": RUN_LEVEL_PERCENTILE,
                "Q_limit": q_limit,
                "T2_limit": t2_limit,
            }
        ]
    )

    threshold_path = TABLE_DIR / "03_pca_run_level_thresholds.csv"
    thresholds.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    print(f"Saved PCA thresholds: {threshold_path}")

    return q_limit, t2_limit


def evaluate_normal_dataset(stats, q_limit, t2_limit, dataset_name):
    """
    Evaluate false alarms on normal data.
    """
    df = stats.copy()

    df["alarm_Q"] = df["Q"] > q_limit
    df["alarm_T2"] = df["T2"] > t2_limit
    df["alarm_total"] = df["alarm_Q"] | df["alarm_T2"]

    run_alarm = (
        df.groupby("simulationRun")[["alarm_Q", "alarm_T2", "alarm_total"]]
        .any()
        .reset_index()
    )

    metrics = {
        "dataset": dataset_name,
        "n_runs": int(run_alarm.shape[0]),
        "normal_run_level_far_Q": float(run_alarm["alarm_Q"].mean()),
        "normal_run_level_far_T2": float(run_alarm["alarm_T2"].mean()),
        "normal_run_level_far_total": float(run_alarm["alarm_total"].mean()),
        "normal_point_level_far_Q": float(df["alarm_Q"].mean()),
        "normal_point_level_far_T2": float(df["alarm_T2"].mean()),
        "normal_point_level_far_total": float(df["alarm_total"].mean()),
    }

    return metrics, run_alarm, df


def evaluate_fault3_dataset(stats, q_limit, t2_limit, dataset_name):
    """
    Evaluate Fault 3 detection performance.

    Detection is calculated on post-fault samples only.
    Pre-fault alarms are also recorded separately.
    """
    df = stats.copy()

    df["alarm_Q"] = df["Q"] > q_limit
    df["alarm_T2"] = df["T2"] > t2_limit
    df["alarm_total"] = df["alarm_Q"] | df["alarm_T2"]

    pre_fault = df[df["sample"] < FAULT_START_SAMPLE].copy()
    post_fault = df[df["sample"] >= FAULT_START_SAMPLE].copy()

    pre_run_alarm = (
        pre_fault.groupby("simulationRun")[["alarm_Q", "alarm_T2", "alarm_total"]]
        .any()
        .reset_index()
    )

    post_run_alarm = (
        post_fault.groupby("simulationRun")[["alarm_Q", "alarm_T2", "alarm_total"]]
        .any()
        .reset_index()
    )

    first_alarm_rows = (
        post_fault[post_fault["alarm_total"]]
        .groupby("simulationRun")["sample"]
        .min()
        .reset_index()
        .rename(columns={"sample": "first_alarm_sample"})
    )

    all_runs = pd.DataFrame({"simulationRun": sorted(df["simulationRun"].unique())})
    run_results = all_runs.merge(post_run_alarm, on="simulationRun", how="left")
    run_results = run_results.merge(first_alarm_rows, on="simulationRun", how="left")

    run_results[["alarm_Q", "alarm_T2", "alarm_total"]] = run_results[
        ["alarm_Q", "alarm_T2", "alarm_total"]
    ].fillna(False)

    run_results["detected"] = run_results["alarm_total"]
    run_results["alarm_delay_min"] = (
        run_results["first_alarm_sample"] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN

    detected_delays = run_results.loc[
        run_results["detected"], "alarm_delay_min"
    ].dropna()

    metrics = {
        "dataset": dataset_name,
        "n_runs": int(run_results.shape[0]),
        "fault3_pre_fault_run_far_Q": float(pre_run_alarm["alarm_Q"].mean()),
        "fault3_pre_fault_run_far_T2": float(pre_run_alarm["alarm_T2"].mean()),
        "fault3_pre_fault_run_far_total": float(pre_run_alarm["alarm_total"].mean()),
        "fault3_pre_fault_point_far_Q": float(pre_fault["alarm_Q"].mean()),
        "fault3_pre_fault_point_far_T2": float(pre_fault["alarm_T2"].mean()),
        "fault3_pre_fault_point_far_total": float(pre_fault["alarm_total"].mean()),
        "fault3_run_level_detection_rate_Q": float(post_run_alarm["alarm_Q"].mean()),
        "fault3_run_level_detection_rate_T2": float(post_run_alarm["alarm_T2"].mean()),
        "fault3_run_level_detection_rate_total": float(post_run_alarm["alarm_total"].mean()),
        "fault3_post_fault_point_detection_rate_Q": float(post_fault["alarm_Q"].mean()),
        "fault3_post_fault_point_detection_rate_T2": float(post_fault["alarm_T2"].mean()),
        "fault3_post_fault_point_detection_rate_total": float(post_fault["alarm_total"].mean()),
        "median_alarm_delay_min_detected_runs": float(detected_delays.median())
        if len(detected_delays) > 0
        else np.nan,
        "mean_alarm_delay_min_detected_runs": float(detected_delays.mean())
        if len(detected_delays) > 0
        else np.nan,
        "detected_runs": int(run_results["detected"].sum()),
    }

    return metrics, run_results, df


def plot_representative_failure(fault3_stats_with_alarm, q_limit, t2_limit):
    """
    Plot Q and T2 trajectories for a representative Fault 3 run that is not detected.
    """
    post_fault = fault3_stats_with_alarm[
        fault3_stats_with_alarm["sample"] >= FAULT_START_SAMPLE
    ].copy()

    run_detect = (
        post_fault.groupby("simulationRun")["alarm_total"]
        .any()
        .reset_index()
        .rename(columns={"alarm_total": "detected"})
    )

    undetected_runs = run_detect.loc[~run_detect["detected"], "simulationRun"].tolist()

    if len(undetected_runs) > 0:
        representative_run = int(undetected_runs[0])
    else:
        representative_run = int(run_detect["simulationRun"].iloc[0])

    one_run = fault3_stats_with_alarm[
        fault3_stats_with_alarm["simulationRun"] == representative_run
    ].sort_values("sample")

    print(f"Representative run for Fig. 1: {representative_run}")

    # Q figure
    plt.figure(figsize=(9, 4.8))
    plt.plot(one_run["sample"], one_run["Q"], label="PCA Q/SPE")
    plt.axhline(q_limit, linestyle="--", label="Run-level Q limit")
    plt.axvline(FAULT_START_SAMPLE, linestyle=":", label="Fault start")
    plt.xlabel("Sample")
    plt.ylabel("Q/SPE")
    plt.title(f"Fault 3 PCA Q/SPE response, run {representative_run}")
    plt.legend()
    plt.tight_layout()

    q_png = FIGURE_DIR / "fig1a_fault3_pca_Q_failure.png"
    q_pdf = FIGURE_DIR / "fig1a_fault3_pca_Q_failure.pdf"
    plt.savefig(q_png, dpi=300)
    plt.savefig(q_pdf)
    plt.close()

    # T2 figure
    plt.figure(figsize=(9, 4.8))
    plt.plot(one_run["sample"], one_run["T2"], label="PCA T²")
    plt.axhline(t2_limit, linestyle="--", label="Run-level T² limit")
    plt.axvline(FAULT_START_SAMPLE, linestyle=":", label="Fault start")
    plt.xlabel("Sample")
    plt.ylabel("Hotelling's T²")
    plt.title(f"Fault 3 PCA T² response, run {representative_run}")
    plt.legend()
    plt.tight_layout()

    t2_png = FIGURE_DIR / "fig1b_fault3_pca_T2_failure.png"
    t2_pdf = FIGURE_DIR / "fig1b_fault3_pca_T2_failure.pdf"
    plt.savefig(t2_png, dpi=300)
    plt.savefig(t2_pdf)
    plt.close()

    print(f"Saved figure: {q_png}")
    print(f"Saved figure: {q_pdf}")
    print(f"Saved figure: {t2_png}")
    print(f"Saved figure: {t2_pdf}")

    return representative_run


def main():
    print("=" * 80)
    print("Step 03: PCA baseline for TEP Fault 3")
    print("=" * 80)

    variable_cols = load_variable_columns()
    data = load_processed_data()

    # 1. Fit PCA only on normal training data.
    model = fit_pca_model(data["normal_train"], variable_cols)

    # 2. Compute PCA statistics.
    print("\nComputing PCA statistics...")
    normal_validation_stats = compute_pca_statistics(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_stats = compute_pca_statistics(
        data["normal_evaluation"], variable_cols, model
    )
    fault3_final_stats = compute_pca_statistics(
        data["fault3_final_test"], variable_cols, model
    )

    # 3. Determine run-level thresholds using normal validation data.
    q_limit, t2_limit = determine_run_level_limits(normal_validation_stats)

    # 4. Evaluate normal evaluation and Fault 3 final test data.
    normal_metrics, normal_run_results, normal_stats_with_alarm = evaluate_normal_dataset(
        normal_evaluation_stats, q_limit, t2_limit, "normal_evaluation_runs_251_500"
    )

    fault3_metrics, fault3_run_results, fault3_stats_with_alarm = evaluate_fault3_dataset(
        fault3_final_stats, q_limit, t2_limit, "fault3_final_test_runs_251_500"
    )

    # 5. Save metrics.
    metrics_df = pd.DataFrame([normal_metrics, fault3_metrics])
    metrics_path = TABLE_DIR / "03_pca_baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved PCA baseline metrics: {metrics_path}")

    normal_run_path = TABLE_DIR / "03_pca_normal_evaluation_run_results.csv"
    normal_run_results.to_csv(normal_run_path, index=False, encoding="utf-8-sig")
    print(f"Saved normal run results: {normal_run_path}")

    fault3_run_path = TABLE_DIR / "03_pca_fault3_final_test_run_results.csv"
    fault3_run_results.to_csv(fault3_run_path, index=False, encoding="utf-8-sig")
    print(f"Saved Fault 3 run results: {fault3_run_path}")

    # 6. Plot representative failure case.
    representative_run = plot_representative_failure(
        fault3_stats_with_alarm, q_limit, t2_limit
    )

    # 7. Save compact config for this step.
    step3_config = {
        "script": "scripts/03_pca_baseline_fault3.py",
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "Q_limit": q_limit,
        "T2_limit": t2_limit,
        "representative_run_for_fig1": representative_run,
    }

    config_path = CONFIG_DIR / "fault3_pca_baseline_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(step3_config, f, indent=4, ensure_ascii=False)

    print(f"Saved Step 03 config: {config_path}")

    print("\n" + "=" * 80)
    print("Key PCA baseline results")
    print("=" * 80)
    print(f"Selected PCs: {model['n_pcs']}")
    print(f"Cumulative explained variance: {model['cum_var']:.6f}")
    print(f"Q limit: {q_limit:.6f}")
    print(f"T2 limit: {t2_limit:.6f}")
    print(
        "Normal run-level FAR total: "
        f"{normal_metrics['normal_run_level_far_total'] * 100:.2f}%"
    )
    print(
        "Fault 3 run-level detection rate total: "
        f"{fault3_metrics['fault3_run_level_detection_rate_total'] * 100:.2f}%"
    )
    print(
        "Fault 3 post-fault point detection rate total: "
        f"{fault3_metrics['fault3_post_fault_point_detection_rate_total'] * 100:.4f}%"
    )
    print(
        "Median alarm delay among detected runs: "
        f"{fault3_metrics['median_alarm_delay_min_detected_runs']:.2f} min"
    )

    print("\nStep 03 finished successfully.")


if __name__ == "__main__":
    main()