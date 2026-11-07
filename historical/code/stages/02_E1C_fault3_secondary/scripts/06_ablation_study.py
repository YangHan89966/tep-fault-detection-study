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
    return residual_df, residual_cols


def fit_scaler_from_normal_validation(normal_validation_res, residual_cols, signs_dict=None):
    """
    Fit mean/std scaler using normal validation data.

    signs_dict:
    - None: no direction unification, sign = +1 for all residuals.
    - dict: residual_col -> sign.
    """
    rows = []

    for col in residual_cols:
        sign = 1
        if signs_dict is not None:
            sign = int(signs_dict.get(col, 1))

        signed_value = sign * normal_validation_res[col].to_numpy(dtype=float)

        mean_ = float(np.mean(signed_value))
        std_ = float(np.std(signed_value, ddof=1))
        if std_ == 0:
            std_ = 1.0

        rows.append(
            {
                "residual_variable": col,
                "sign": sign,
                "normal_validation_mean": mean_,
                "normal_validation_std": std_,
            }
        )

    return pd.DataFrame(rows)


def construct_monitoring_index(residual_df, scaler, index_name):
    """
    Construct a monitoring index by:
    1. applying sign;
    2. standardizing each residual variable using normal validation scaler;
    3. averaging standardized residual variables.
    """
    out = residual_df[["faultNumber", "simulationRun", "sample"]].copy()

    z_values = []

    for _, row in scaler.iterrows():
        col = row["residual_variable"]
        sign = int(row["sign"])
        mean_ = float(row["normal_validation_mean"])
        std_ = float(row["normal_validation_std"])

        signed_value = sign * residual_df[col].to_numpy(dtype=float)
        z = (signed_value - mean_) / std_
        z_values.append(z)

    z_matrix = np.column_stack(z_values)
    out[index_name] = z_matrix.mean(axis=1)

    return out


def compute_one_sided_cusum(index_df, index_col, k=CUSUM_K):
    df = index_df.sort_values(["simulationRun", "sample"]).copy()
    cusum_values = np.zeros(df.shape[0], dtype=float)

    start = 0
    for run, group in df.groupby("simulationRun", sort=True):
        values = group[index_col].to_numpy(dtype=float)
        c = 0.0
        run_cusum = []

        for x in values:
            c = max(0.0, c + x - k)
            run_cusum.append(c)

        end = start + len(group)
        cusum_values[start:end] = run_cusum
        start = end

    df["monitoring_statistic"] = cusum_values
    return df


def use_raw_index_without_cusum(index_df, index_col):
    df = index_df.sort_values(["simulationRun", "sample"]).copy()
    df["monitoring_statistic"] = df[index_col]
    return df


def determine_run_level_limit(normal_validation_stat):
    """
    Use run-level max monitoring statistic from normal validation data.
    """
    run_max = (
        normal_validation_stat.groupby("simulationRun")["monitoring_statistic"]
        .max()
        .reset_index()
        .rename(columns={"monitoring_statistic": "run_max_statistic"})
    )

    limit = float(
        percentile_compat(run_max["run_max_statistic"].to_numpy(), RUN_LEVEL_PERCENTILE)
    )

    return limit, run_max


def evaluate_normal(stat_df, limit, method_name):
    df = stat_df.copy()
    df["alarm"] = df["monitoring_statistic"] > limit

    run_alarm = (
        df.groupby("simulationRun")["alarm"]
        .any()
        .reset_index()
        .rename(columns={"alarm": "run_alarm"})
    )

    metrics = {
        "method": method_name,
        "normal_n_runs": int(run_alarm.shape[0]),
        "normal_run_level_far": float(run_alarm["run_alarm"].mean()),
        "normal_point_level_far": float(df["alarm"].mean()),
    }

    return metrics, run_alarm, df


def evaluate_fault3(stat_df, limit, method_name):
    df = stat_df.copy()
    df["alarm"] = df["monitoring_statistic"] > limit

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
        "method": method_name,
        "fault3_n_runs": int(run_results.shape[0]),
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


def run_one_method(
    method_name,
    residual_cols_for_method,
    signs_dict,
    use_cusum,
    normal_validation_res,
    normal_evaluation_res,
    fault3_final_res,
):
    """
    Run one ablation method.

    residual_cols_for_method:
    - selected residual variables or all residual variables.

    signs_dict:
    - None: no direction unification.
    - dict: residual_col -> sign.

    use_cusum:
    - True: apply CUSUM to the constructed index.
    - False: use raw constructed index directly.
    """
    print("\n" + "=" * 80)
    print(f"Running ablation method: {method_name}")
    print("=" * 80)

    index_col = "ablation_index"

    scaler = fit_scaler_from_normal_validation(
        normal_validation_res,
        residual_cols_for_method,
        signs_dict=signs_dict,
    )

    val_index = construct_monitoring_index(normal_validation_res, scaler, index_col)
    normal_eval_index = construct_monitoring_index(normal_evaluation_res, scaler, index_col)
    fault3_index = construct_monitoring_index(fault3_final_res, scaler, index_col)

    if use_cusum:
        val_stat = compute_one_sided_cusum(val_index, index_col, CUSUM_K)
        normal_eval_stat = compute_one_sided_cusum(normal_eval_index, index_col, CUSUM_K)
        fault3_stat = compute_one_sided_cusum(fault3_index, index_col, CUSUM_K)
    else:
        val_stat = use_raw_index_without_cusum(val_index, index_col)
        normal_eval_stat = use_raw_index_without_cusum(normal_eval_index, index_col)
        fault3_stat = use_raw_index_without_cusum(fault3_index, index_col)

    limit, val_run_max = determine_run_level_limit(val_stat)

    normal_metrics, normal_run_results, normal_eval_with_alarm = evaluate_normal(
        normal_eval_stat, limit, method_name
    )

    fault3_metrics, fault3_run_results, fault3_with_alarm = evaluate_fault3(
        fault3_stat, limit, method_name
    )

    combined_metrics = {
        "method": method_name,
        "n_residual_variables": len(residual_cols_for_method),
        "direction_unification": signs_dict is not None,
        "use_cusum": use_cusum,
        "cusum_k": CUSUM_K if use_cusum else np.nan,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "control_limit": limit,
    }

    combined_metrics.update(normal_metrics)
    combined_metrics.update(fault3_metrics)

    print(f"Control limit: {limit:.6f}")
    print(
        "Normal run-level FAR: "
        f"{combined_metrics['normal_run_level_far'] * 100:.2f}%"
    )
    print(
        "Fault 3 run-level detection rate: "
        f"{combined_metrics['fault3_run_level_detection_rate'] * 100:.2f}%"
    )
    print(
        "Fault 3 post-fault point detection rate: "
        f"{combined_metrics['fault3_post_fault_point_detection_rate'] * 100:.4f}%"
    )
    print(
        "Median alarm delay: "
        f"{combined_metrics['median_alarm_delay_min']:.2f} min"
    )

    return {
        "metrics": combined_metrics,
        "scaler": scaler,
        "normal_run_results": normal_run_results,
        "fault3_run_results": fault3_run_results,
        "normal_eval_with_alarm": normal_eval_with_alarm,
        "fault3_with_alarm": fault3_with_alarm,
        "validation_run_max": val_run_max,
    }


def plot_ablation_results(metrics_df):
    """
    Generate simple ablation comparison figures.
    """

    # Figure 4a: FAR and FDR comparison.
    plot_df = metrics_df.copy()
    x = np.arange(plot_df.shape[0])
    width = 0.35

    plt.figure(figsize=(11, 5.2))
    plt.bar(
        x - width / 2,
        plot_df["normal_run_level_far"] * 100,
        width,
        label="Normal run-level FAR (%)",
    )
    plt.bar(
        x + width / 2,
        plot_df["fault3_run_level_detection_rate"] * 100,
        width,
        label="Fault 3 run-level detection rate (%)",
    )
    plt.xticks(x, plot_df["method"], rotation=25, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title("Ablation study: run-level FAR and Fault 3 detection rate")
    plt.legend()
    plt.tight_layout()

    fig1_png = FIGURE_DIR / "fig4a_ablation_far_detection.png"
    fig1_pdf = FIGURE_DIR / "fig4a_ablation_far_detection.pdf"
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(fig1_pdf)
    plt.close()

    # Figure 4b: median delay comparison.
    plt.figure(figsize=(11, 5.2))
    plt.bar(x, plot_df["median_alarm_delay_min"])
    plt.xticks(x, plot_df["method"], rotation=25, ha="right")
    plt.ylabel("Median alarm delay (min)")
    plt.title("Ablation study: median alarm delay")
    plt.tight_layout()

    fig2_png = FIGURE_DIR / "fig4b_ablation_alarm_delay.png"
    fig2_pdf = FIGURE_DIR / "fig4b_ablation_alarm_delay.pdf"
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()

    print(f"Saved figure: {fig1_png}")
    print(f"Saved figure: {fig1_pdf}")
    print(f"Saved figure: {fig2_png}")
    print(f"Saved figure: {fig2_pdf}")


def main():
    print("=" * 80)
    print("Step 06: Ablation study for local signed residual CUSUM")
    print("=" * 80)

    variable_cols = load_variable_columns()
    selected = load_selected_residuals()
    data = load_data()

    selected_variables = selected["variable"].tolist()
    selected_residual_cols = selected["residual_variable"].tolist()

    selected_signs_dict = {
        row["residual_variable"]: int(row["sign_for_positive_evidence"])
        for _, row in selected.iterrows()
    }

    print("\nSelected variables from Step 04:")
    print(selected_variables)
    print("Selected signs:")
    print(selected_signs_dict)

    # 1. Fit PCA and compute residuals.
    model = fit_pca_model(data["normal_train"], variable_cols)

    print("\nComputing variable-wise PCA residuals...")
    normal_validation_res, all_residual_cols = compute_variable_residuals(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = compute_variable_residuals(
        data["normal_evaluation"], variable_cols, model
    )
    fault3_final_res, _ = compute_variable_residuals(
        data["fault3_final_test"], variable_cols, model
    )

    # 2. Define ablation methods.
    methods = [
        {
            "method_name": "All residuals + CUSUM",
            "residual_cols": all_residual_cols,
            "signs_dict": None,
            "use_cusum": True,
            "purpose": "Test whether simply averaging all PCA residuals and applying CUSUM is sufficient.",
        },
        {
            "method_name": "Selected residuals w/o sign + CUSUM",
            "residual_cols": selected_residual_cols,
            "signs_dict": None,
            "use_cusum": True,
            "purpose": "Test whether selected variables alone are sufficient without direction unification.",
        },
        {
            "method_name": "Selected signed residuals w/o CUSUM",
            "residual_cols": selected_residual_cols,
            "signs_dict": selected_signs_dict,
            "use_cusum": False,
            "purpose": "Test whether direction-unified local residuals are sufficient without sequential accumulation.",
        },
        {
            "method_name": "Selected signed residuals + CUSUM",
            "residual_cols": selected_residual_cols,
            "signs_dict": selected_signs_dict,
            "use_cusum": True,
            "purpose": "Full proposed method.",
        },
    ]

    all_results = []
    method_records = []

    for method in methods:
        result = run_one_method(
            method_name=method["method_name"],
            residual_cols_for_method=method["residual_cols"],
            signs_dict=method["signs_dict"],
            use_cusum=method["use_cusum"],
            normal_validation_res=normal_validation_res,
            normal_evaluation_res=normal_evaluation_res,
            fault3_final_res=fault3_final_res,
        )

        all_results.append(result["metrics"])

        safe_name = (
            method["method_name"]
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("/", "without")
        )

        result["fault3_run_results"].to_csv(
            TABLE_DIR / f"06_ablation_{safe_name}_fault3_run_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        result["normal_run_results"].to_csv(
            TABLE_DIR / f"06_ablation_{safe_name}_normal_run_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        method_records.append(
            {
                "method": method["method_name"],
                "purpose": method["purpose"],
                "n_residual_variables": len(method["residual_cols"]),
                "direction_unification": method["signs_dict"] is not None,
                "use_cusum": method["use_cusum"],
            }
        )

    # 3. Save ablation metrics.
    metrics_df = pd.DataFrame(all_results)

    preferred_cols = [
        "method",
        "n_residual_variables",
        "direction_unification",
        "use_cusum",
        "cusum_k",
        "control_limit",
        "normal_run_level_far",
        "normal_point_level_far",
        "fault3_pre_fault_run_far",
        "fault3_pre_fault_point_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
        "detected_runs",
    ]

    existing_cols = [c for c in preferred_cols if c in metrics_df.columns]
    metrics_df = metrics_df[existing_cols]

    metrics_path = TABLE_DIR / "06_ablation_results.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved ablation results: {metrics_path}")

    method_record_df = pd.DataFrame(method_records)
    method_record_path = TABLE_DIR / "06_ablation_method_definitions.csv"
    method_record_df.to_csv(method_record_path, index=False, encoding="utf-8-sig")
    print(f"Saved ablation method definitions: {method_record_path}")

    # 4. Plot comparison.
    plot_ablation_results(metrics_df)

    # 5. Save config.
    config = {
        "script": "scripts/06_ablation_study.py",
        "purpose": "Ablation study to verify the necessity of local residual selection, direction unification, and CUSUM accumulation.",
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "selected_variables": selected_variables,
        "selected_residual_variables": selected_residual_cols,
        "selected_signs": selected["sign_for_positive_evidence"].astype(int).tolist(),
        "cusum_k": CUSUM_K,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "methods": method_records,
        "output_table": "results/tables/06_ablation_results.csv",
        "output_figures": [
            "results/figures/fig4a_ablation_far_detection.png",
            "results/figures/fig4b_ablation_alarm_delay.png",
        ],
    }

    config_path = CONFIG_DIR / "fault3_ablation_study_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Saved Step 06 config: {config_path}")

    print("\n" + "=" * 80)
    print("Key ablation study results")
    print("=" * 80)

    display_cols = [
        "method",
        "normal_run_level_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
        "median_alarm_delay_min",
    ]

    print(metrics_df[display_cols])

    print("\nStep 06 finished successfully.")


if __name__ == "__main__":
    main()