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
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Main settings
# ======================================================================
FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
PCA_CUMVAR_TARGET = 0.95

CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95


def percentile_compat(x, q):
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
    return variable_info["variable"].tolist()


def load_selected_residuals():
    selected_path = TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Selected residual file not found: {selected_path}. "
            "Please run scripts/04_local_residual_screening.py first."
        )

    selected = pd.read_csv(selected_path)

    required_cols = [
        "variable",
        "residual_variable",
        "dominant_direction",
        "sign_for_positive_evidence",
    ]

    missing = [c for c in required_cols if c not in selected.columns]
    if missing:
        raise ValueError(f"Missing columns in selected residual file: {missing}")

    print("\nSelected local residual variables:")
    print(
        selected[
            [
                "variable",
                "residual_variable",
                "dominant_direction",
                "sign_for_positive_evidence",
                "mean_delta",
                "direction_ratio",
            ]
        ]
    )

    return selected


def load_data():
    files = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR / "normal_validation_runs_1_250.pkl",
        "normal_evaluation": PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl",
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

    model = {
        "mean": mean_,
        "std": std_,
        "components": pca_full.components_[:n_pcs, :],
        "eigenvalues": pca_full.explained_variance_[:n_pcs],
        "n_pcs": n_pcs,
        "cum_var": cum_var,
    }

    print(f"Selected PCs: {n_pcs}")
    print(f"Cumulative explained variance: {cum_var:.6f}")

    return model


def compute_variable_residuals(df, variable_cols, model):
    """
    Compute variable-wise PCA reconstruction residuals in standardized space.
    """
    X = df[variable_cols].to_numpy(dtype=float)
    X_std = (X - model["mean"]) / model["std"]

    P = model["components"]
    scores = X_std @ P.T
    X_recon = scores @ P

    residual = X_std - X_recon
    residual_cols = [f"res_{v}" for v in variable_cols]

    id_df = df[["faultNumber", "simulationRun", "sample"]].reset_index(drop=True)
    residual_values_df = pd.DataFrame(
        residual,
        columns=residual_cols,
        index=id_df.index,
    )

    residual_df = pd.concat([id_df, residual_values_df], axis=1)
    return residual_df


def fit_local_signed_residual_scaler(normal_validation_res, selected):
    """
    Use normal validation data to estimate mean and std for each selected
    direction-unified residual variable.
    """
    rows = []

    for _, row in selected.iterrows():
        variable = row["variable"]
        residual_variable = row["residual_variable"]
        sign = int(row["sign_for_positive_evidence"])

        signed_value = sign * normal_validation_res[residual_variable].to_numpy(dtype=float)

        mean_ = float(np.mean(signed_value))
        std_ = float(np.std(signed_value, ddof=1))
        if std_ == 0:
            std_ = 1.0

        rows.append(
            {
                "variable": variable,
                "residual_variable": residual_variable,
                "sign_for_positive_evidence": sign,
                "normal_validation_signed_mean": mean_,
                "normal_validation_signed_std": std_,
            }
        )

    scaler = pd.DataFrame(rows)

    scaler_path = TABLE_DIR / "05_local_signed_residual_scaler.csv"
    scaler.to_csv(scaler_path, index=False, encoding="utf-8-sig")
    print(f"Saved local signed residual scaler: {scaler_path}")

    return scaler


def construct_local_signed_residual(residual_df, scaler):
    """
    Construct local signed residual:

    1. Direction-unify each selected residual variable.
    2. Standardize using normal validation mean and std.
    3. Average selected standardized residuals.
    """
    out = residual_df[["faultNumber", "simulationRun", "sample"]].copy()

    z_cols = []

    for _, row in scaler.iterrows():
        variable = row["variable"]
        residual_variable = row["residual_variable"]
        sign = int(row["sign_for_positive_evidence"])
        mean_ = float(row["normal_validation_signed_mean"])
        std_ = float(row["normal_validation_signed_std"])

        signed_value = sign * residual_df[residual_variable].to_numpy(dtype=float)
        z = (signed_value - mean_) / std_

        z_col = f"signed_z_{variable}"
        out[z_col] = z
        z_cols.append(z_col)

    out["local_signed_residual"] = out[z_cols].mean(axis=1)

    return out, z_cols


def compute_one_sided_cusum(local_df, k=CUSUM_K):
    """
    Compute one-sided upper CUSUM for each simulation run.

    C_t = max(0, C_{t-1} + local_signed_residual_t - k)
    """
    df = local_df.sort_values(["simulationRun", "sample"]).copy()
    cusum_values = np.zeros(df.shape[0], dtype=float)

    start = 0
    for run, group in df.groupby("simulationRun", sort=True):
        values = group["local_signed_residual"].to_numpy(dtype=float)
        c = 0.0
        run_cusum = []

        for x in values:
            c = max(0.0, c + x - k)
            run_cusum.append(c)

        end = start + len(group)
        cusum_values[start:end] = run_cusum
        start = end

    df["local_signed_cusum"] = cusum_values
    return df


def determine_cusum_limit(normal_validation_cusum):
    """
    Determine CUSUM control limit using run-level maximum CUSUM
    from normal validation data.
    """
    print("\n" + "=" * 80)
    print("Determining local signed CUSUM control limit")
    print("=" * 80)

    run_max = (
        normal_validation_cusum.groupby("simulationRun")["local_signed_cusum"]
        .max()
        .reset_index()
        .rename(columns={"local_signed_cusum": "run_max_cusum"})
    )

    control_limit = float(
        percentile_compat(run_max["run_max_cusum"].to_numpy(), RUN_LEVEL_PERCENTILE)
    )

    print(f"CUSUM k: {CUSUM_K}")
    print(f"Run-level percentile: {RUN_LEVEL_PERCENTILE}%")
    print(f"CUSUM control limit: {control_limit:.6f}")

    threshold_df = pd.DataFrame(
        [
            {
                "threshold_source": "normal_validation_runs_1_250",
                "cusum_k": CUSUM_K,
                "run_level_percentile": RUN_LEVEL_PERCENTILE,
                "cusum_control_limit": control_limit,
            }
        ]
    )

    threshold_path = TABLE_DIR / "05_local_signed_cusum_threshold.csv"
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    print(f"Saved CUSUM threshold: {threshold_path}")

    run_max_path = TABLE_DIR / "05_normal_validation_run_max_cusum.csv"
    run_max.to_csv(run_max_path, index=False, encoding="utf-8-sig")
    print(f"Saved normal validation run max CUSUM: {run_max_path}")

    return control_limit


def evaluate_normal_cusum(cusum_df, control_limit, dataset_name):
    df = cusum_df.copy()
    df["alarm"] = df["local_signed_cusum"] > control_limit

    run_alarm = (
        df.groupby("simulationRun")["alarm"]
        .any()
        .reset_index()
        .rename(columns={"alarm": "run_alarm"})
    )

    metrics = {
        "dataset": dataset_name,
        "n_runs": int(run_alarm.shape[0]),
        "normal_run_level_far": float(run_alarm["run_alarm"].mean()),
        "normal_point_level_far": float(df["alarm"].mean()),
    }

    return metrics, run_alarm, df


def evaluate_fault3_cusum(cusum_df, control_limit, dataset_name):
    df = cusum_df.copy()
    df["alarm"] = df["local_signed_cusum"] > control_limit

    pre_fault = df[df["sample"] < FAULT_START_SAMPLE].copy()
    post_fault = df[df["sample"] >= FAULT_START_SAMPLE].copy()

    pre_run_alarm = (
        pre_fault.groupby("simulationRun")["alarm"]
        .any()
        .reset_index()
        .rename(columns={"alarm": "pre_fault_run_alarm"})
    )

    post_run_alarm = (
        post_fault.groupby("simulationRun")["alarm"]
        .any()
        .reset_index()
        .rename(columns={"alarm": "post_fault_run_alarm"})
    )

    first_alarm_rows = (
        post_fault[post_fault["alarm"]]
        .groupby("simulationRun")["sample"]
        .min()
        .reset_index()
        .rename(columns={"sample": "first_alarm_sample"})
    )

    all_runs = pd.DataFrame({"simulationRun": sorted(df["simulationRun"].unique())})
    run_results = all_runs.merge(pre_run_alarm, on="simulationRun", how="left")
    run_results = run_results.merge(post_run_alarm, on="simulationRun", how="left")
    run_results = run_results.merge(first_alarm_rows, on="simulationRun", how="left")

    run_results["pre_fault_run_alarm"] = run_results["pre_fault_run_alarm"].fillna(False)
    run_results["post_fault_run_alarm"] = run_results["post_fault_run_alarm"].fillna(False)

    run_results["detected"] = run_results["post_fault_run_alarm"]
    run_results["alarm_delay_min"] = (
        run_results["first_alarm_sample"] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN

    detected_delays = run_results.loc[
        run_results["detected"], "alarm_delay_min"
    ].dropna()

    metrics = {
        "dataset": dataset_name,
        "n_runs": int(run_results.shape[0]),
        "fault3_pre_fault_run_far": float(run_results["pre_fault_run_alarm"].mean()),
        "fault3_pre_fault_point_far": float(pre_fault["alarm"].mean()),
        "fault3_run_level_detection_rate": float(run_results["detected"].mean()),
        "fault3_post_fault_point_detection_rate": float(post_fault["alarm"].mean()),
        "median_alarm_delay_min": float(detected_delays.median())
        if len(detected_delays) > 0
        else np.nan,
        "mean_alarm_delay_min": float(detected_delays.mean())
        if len(detected_delays) > 0
        else np.nan,
        "detected_runs": int(run_results["detected"].sum()),
    }

    return metrics, run_results, df


def plot_representative_cusum(fault3_cusum_with_alarm, control_limit):
    """
    Plot a representative Fault 3 run.

    Use the run whose alarm delay is closest to the median delay.
    """
    df = fault3_cusum_with_alarm.copy()

    post_fault = df[(df["sample"] >= FAULT_START_SAMPLE) & (df["alarm"])].copy()

    first_alarm = (
        post_fault.groupby("simulationRun")["sample"]
        .min()
        .reset_index()
        .rename(columns={"sample": "first_alarm_sample"})
    )

    first_alarm["alarm_delay_min"] = (
        first_alarm["first_alarm_sample"] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN

    if len(first_alarm) > 0:
        median_delay = first_alarm["alarm_delay_min"].median()
        first_alarm["distance_to_median"] = (
            first_alarm["alarm_delay_min"] - median_delay
        ).abs()
        representative_run = int(
            first_alarm.sort_values("distance_to_median")["simulationRun"].iloc[0]
        )
    else:
        representative_run = int(df["simulationRun"].iloc[0])

    one_run = df[df["simulationRun"] == representative_run].sort_values("sample")

    first_alarm_sample = one_run.loc[one_run["alarm"], "sample"]
    if len(first_alarm_sample) > 0:
        first_alarm_sample = int(first_alarm_sample.iloc[0])
        alarm_delay_min = (first_alarm_sample - FAULT_START_SAMPLE) * SAMPLING_INTERVAL_MIN
    else:
        first_alarm_sample = None
        alarm_delay_min = None

    print(f"Representative run for Fig. 3: {representative_run}")
    print(f"First alarm sample: {first_alarm_sample}")
    print(f"Alarm delay: {alarm_delay_min} min")

    plt.figure(figsize=(9, 4.8))
    plt.plot(
        one_run["sample"],
        one_run["local_signed_cusum"],
        label="Local signed residual CUSUM",
    )
    plt.axhline(control_limit, linestyle="--", label="CUSUM control limit")
    plt.axvline(FAULT_START_SAMPLE, linestyle=":", label="Fault start")

    if first_alarm_sample is not None:
        plt.axvline(first_alarm_sample, linestyle="-.", label="First alarm")

    plt.xlabel("Sample")
    plt.ylabel("CUSUM")
    plt.title(f"Fault 3 local signed residual CUSUM, run {representative_run}")
    plt.legend()
    plt.tight_layout()

    fig_png = FIGURE_DIR / "fig3_fault3_local_signed_cusum_alarm.png"
    fig_pdf = FIGURE_DIR / "fig3_fault3_local_signed_cusum_alarm.pdf"

    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    print(f"Saved figure: {fig_png}")
    print(f"Saved figure: {fig_pdf}")

    return representative_run, first_alarm_sample, alarm_delay_min


def main():
    print("=" * 80)
    print("Step 05: Local signed residual CUSUM final validation")
    print("=" * 80)

    variable_cols = load_variable_columns()
    selected = load_selected_residuals()
    data = load_data()

    # 1. Fit PCA model using normal training data only.
    model = fit_pca_model(data["normal_train"], variable_cols)

    # 2. Compute variable-wise PCA residuals.
    print("\nComputing variable-wise PCA residuals...")
    normal_validation_res = compute_variable_residuals(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res = compute_variable_residuals(
        data["normal_evaluation"], variable_cols, model
    )
    fault3_final_res = compute_variable_residuals(
        data["fault3_final_test"], variable_cols, model
    )

    # 3. Fit local signed residual standardization using normal validation data.
    scaler = fit_local_signed_residual_scaler(normal_validation_res, selected)

    # 4. Construct local signed residual.
    print("\nConstructing local signed residual...")
    normal_validation_local, z_cols = construct_local_signed_residual(
        normal_validation_res, scaler
    )
    normal_evaluation_local, _ = construct_local_signed_residual(
        normal_evaluation_res, scaler
    )
    fault3_final_local, _ = construct_local_signed_residual(
        fault3_final_res, scaler
    )

    # 5. Compute CUSUM.
    print("\nComputing local signed residual CUSUM...")
    normal_validation_cusum = compute_one_sided_cusum(normal_validation_local, CUSUM_K)
    normal_evaluation_cusum = compute_one_sided_cusum(normal_evaluation_local, CUSUM_K)
    fault3_final_cusum = compute_one_sided_cusum(fault3_final_local, CUSUM_K)

    # 6. Determine CUSUM control limit from normal validation data.
    control_limit = determine_cusum_limit(normal_validation_cusum)

    # 7. Final evaluation.
    normal_metrics, normal_run_results, normal_eval_with_alarm = evaluate_normal_cusum(
        normal_evaluation_cusum,
        control_limit,
        "normal_evaluation_runs_251_500",
    )

    fault3_metrics, fault3_run_results, fault3_with_alarm = evaluate_fault3_cusum(
        fault3_final_cusum,
        control_limit,
        "fault3_final_test_runs_251_500",
    )

    # 8. Save metrics and run results.
    metrics_df = pd.DataFrame([normal_metrics, fault3_metrics])
    metrics_path = TABLE_DIR / "05_local_signed_cusum_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved local signed CUSUM metrics: {metrics_path}")

    normal_run_path = TABLE_DIR / "05_local_signed_cusum_normal_run_results.csv"
    normal_run_results.to_csv(normal_run_path, index=False, encoding="utf-8-sig")
    print(f"Saved normal run results: {normal_run_path}")

    fault3_run_path = TABLE_DIR / "05_local_signed_cusum_fault3_run_results.csv"
    fault3_run_results.to_csv(fault3_run_path, index=False, encoding="utf-8-sig")
    print(f"Saved Fault 3 run results: {fault3_run_path}")

    # Save local signed residual sample means for interpretation.
    fault3_sample_mean = (
        fault3_with_alarm.groupby("sample")[["local_signed_residual", "local_signed_cusum"]]
        .mean()
        .reset_index()
    )
    sample_mean_path = TABLE_DIR / "05_fault3_local_signed_cusum_by_sample.csv"
    fault3_sample_mean.to_csv(sample_mean_path, index=False, encoding="utf-8-sig")
    print(f"Saved Fault 3 sample-wise CUSUM summary: {sample_mean_path}")

    # 9. Plot representative run.
    representative_run, first_alarm_sample, alarm_delay_min = plot_representative_cusum(
        fault3_with_alarm,
        control_limit,
    )

    # 10. Save config.
    config = {
        "script": "scripts/05_local_signed_cusum.py",
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "selected_variables": selected["variable"].tolist(),
        "selected_residual_variables": selected["residual_variable"].tolist(),
        "selected_signs": selected["sign_for_positive_evidence"].astype(int).tolist(),
        "local_signed_residual_standardization": "mean and std estimated from normal_validation_runs_1_250",
        "cusum_k": CUSUM_K,
        "cusum_control_limit_source": "normal_validation_runs_1_250",
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "cusum_control_limit": control_limit,
        "normal_evaluation_data": "normal_evaluation_runs_251_500.pkl",
        "fault3_final_test_data": "fault3_final_test_runs_251_500.pkl",
        "representative_run_for_fig3": representative_run,
        "representative_first_alarm_sample": first_alarm_sample,
        "representative_alarm_delay_min": alarm_delay_min,
    }

    config_path = CONFIG_DIR / "fault3_local_signed_cusum_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Saved Step 05 config: {config_path}")

    print("\n" + "=" * 80)
    print("Key local signed CUSUM results")
    print("=" * 80)
    print(f"Selected PCs: {model['n_pcs']}")
    print(f"Cumulative explained variance: {model['cum_var']:.6f}")
    print(f"Selected variables: {selected['variable'].tolist()}")
    print(f"CUSUM k: {CUSUM_K}")
    print(f"CUSUM control limit: {control_limit:.6f}")
    print(
        "Normal run-level FAR: "
        f"{normal_metrics['normal_run_level_far'] * 100:.2f}%"
    )
    print(
        "Normal point-level FAR: "
        f"{normal_metrics['normal_point_level_far'] * 100:.4f}%"
    )
    print(
        "Fault 3 pre-fault run-level FAR: "
        f"{fault3_metrics['fault3_pre_fault_run_far'] * 100:.2f}%"
    )
    print(
        "Fault 3 pre-fault point-level FAR: "
        f"{fault3_metrics['fault3_pre_fault_point_far'] * 100:.4f}%"
    )
    print(
        "Fault 3 run-level detection rate: "
        f"{fault3_metrics['fault3_run_level_detection_rate'] * 100:.2f}%"
    )
    print(
        "Fault 3 post-fault point-level detection rate: "
        f"{fault3_metrics['fault3_post_fault_point_detection_rate'] * 100:.4f}%"
    )
    print(
        "Median alarm delay: "
        f"{fault3_metrics['median_alarm_delay_min']:.2f} min"
    )
    print(
        "Mean alarm delay: "
        f"{fault3_metrics['mean_alarm_delay_min']:.2f} min"
    )

    print("\nStep 05 finished successfully.")


if __name__ == "__main__":
    main()