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

N_SELECTED_VARIABLES = 5
DEFAULT_CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95

DIRECTION_RATIO_LIST = [0.90, 0.95, 0.98]
CUSUM_K_LIST = [0.01, 0.03, 0.05, 0.07, 0.10]


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


def load_step04_selected_residuals():
    selected_path = TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Selected residual file not found: {selected_path}. "
            "Please run scripts/04_local_residual_screening.py first."
        )

    return pd.read_csv(selected_path)


def load_data():
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


def screen_selected_residuals_for_threshold(
    normal_validation_res,
    fault3_development_res,
    variable_cols,
    residual_cols,
    direction_ratio_threshold,
):
    key_cols = ["simulationRun", "sample"]

    normal_post = (
        normal_validation_res[normal_validation_res["sample"] >= FAULT_START_SAMPLE]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    fault_post = (
        fault3_development_res[fault3_development_res["sample"] >= FAULT_START_SAMPLE]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    if normal_post[key_cols].equals(fault_post[key_cols]) is False:
        raise ValueError(
            "Normal validation data and Fault 3 development data are not exactly paired "
            "by simulationRun and sample."
        )

    normal_matrix = normal_post[residual_cols].to_numpy(dtype=float)
    fault_matrix = fault_post[residual_cols].to_numpy(dtype=float)
    delta_matrix = fault_matrix - normal_matrix

    rows = []

    for j, var in enumerate(variable_cols):
        delta = delta_matrix[:, j]
        normal_residual = normal_matrix[:, j]

        mean_delta = float(np.mean(delta))
        normal_std = float(np.std(normal_residual, ddof=1))
        positive_ratio = float(np.mean(delta > 0))
        negative_ratio = float(np.mean(delta < 0))
        direction_ratio = max(positive_ratio, negative_ratio)

        if mean_delta >= 0:
            direction = "positive"
            sign_for_positive_evidence = 1
        else:
            direction = "negative"
            sign_for_positive_evidence = -1

        effect_size = mean_delta / normal_std if normal_std > 0 else np.nan

        rows.append(
            {
                "variable": var,
                "residual_variable": f"res_{var}",
                "mean_delta": mean_delta,
                "abs_mean_delta": abs(mean_delta),
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "direction_ratio": direction_ratio,
                "dominant_direction": direction,
                "sign_for_positive_evidence": sign_for_positive_evidence,
                "effect_size": effect_size,
                "abs_effect_size": abs(effect_size),
            }
        )

    metrics = pd.DataFrame(rows)

    stable = metrics[
        metrics["direction_ratio"] >= direction_ratio_threshold
    ].copy()

    selected = stable.sort_values(
        by=["abs_mean_delta", "direction_ratio", "abs_effect_size"],
        ascending=[False, False, False],
    ).head(N_SELECTED_VARIABLES).reset_index(drop=True)

    return metrics, stable, selected


def fit_scaler_from_normal_validation(normal_validation_res, selected):
    rows = []

    for _, row in selected.iterrows():
        residual_variable = row["residual_variable"]
        sign = int(row["sign_for_positive_evidence"])

        signed_value = sign * normal_validation_res[residual_variable].to_numpy(dtype=float)

        mean_ = float(np.mean(signed_value))
        std_ = float(np.std(signed_value, ddof=1))
        if std_ == 0:
            std_ = 1.0

        rows.append(
            {
                "variable": row["variable"],
                "residual_variable": residual_variable,
                "sign_for_positive_evidence": sign,
                "normal_validation_signed_mean": mean_,
                "normal_validation_signed_std": std_,
            }
        )

    return pd.DataFrame(rows)


def construct_local_signed_residual(residual_df, scaler):
    out = residual_df[["faultNumber", "simulationRun", "sample"]].copy()
    z_values = []

    for _, row in scaler.iterrows():
        residual_variable = row["residual_variable"]
        sign = int(row["sign_for_positive_evidence"])
        mean_ = float(row["normal_validation_signed_mean"])
        std_ = float(row["normal_validation_signed_std"])

        signed_value = sign * residual_df[residual_variable].to_numpy(dtype=float)
        z = (signed_value - mean_) / std_
        z_values.append(z)

    z_matrix = np.column_stack(z_values)
    out["local_signed_residual"] = z_matrix.mean(axis=1)

    return out


def compute_one_sided_cusum(local_df, k):
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
    run_max = (
        normal_validation_cusum.groupby("simulationRun")["local_signed_cusum"]
        .max()
        .reset_index()
        .rename(columns={"local_signed_cusum": "run_max_cusum"})
    )

    control_limit = float(
        percentile_compat(run_max["run_max_cusum"].to_numpy(), RUN_LEVEL_PERCENTILE)
    )

    return control_limit, run_max


def evaluate_normal(cusum_df, control_limit):
    df = cusum_df.copy()
    df["alarm"] = df["local_signed_cusum"] > control_limit

    run_alarm = (
        df.groupby("simulationRun")["alarm"]
        .any()
        .reset_index()
        .rename(columns={"alarm": "run_alarm"})
    )

    metrics = {
        "normal_run_level_far": float(run_alarm["run_alarm"].mean()),
        "normal_point_level_far": float(df["alarm"].mean()),
    }

    return metrics


def evaluate_fault3(cusum_df, control_limit):
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

    return metrics


def evaluate_full_method_with_selected(
    selected,
    cusum_k,
    normal_validation_res,
    normal_evaluation_res,
    fault3_final_res,
):
    scaler = fit_scaler_from_normal_validation(normal_validation_res, selected)

    normal_validation_local = construct_local_signed_residual(normal_validation_res, scaler)
    normal_evaluation_local = construct_local_signed_residual(normal_evaluation_res, scaler)
    fault3_final_local = construct_local_signed_residual(fault3_final_res, scaler)

    normal_validation_cusum = compute_one_sided_cusum(normal_validation_local, cusum_k)
    normal_evaluation_cusum = compute_one_sided_cusum(normal_evaluation_local, cusum_k)
    fault3_final_cusum = compute_one_sided_cusum(fault3_final_local, cusum_k)

    control_limit, _ = determine_cusum_limit(normal_validation_cusum)

    normal_metrics = evaluate_normal(normal_evaluation_cusum, control_limit)
    fault3_metrics = evaluate_fault3(fault3_final_cusum, control_limit)

    metrics = {
        "cusum_k": cusum_k,
        "cusum_control_limit": control_limit,
    }
    metrics.update(normal_metrics)
    metrics.update(fault3_metrics)

    return metrics


def run_direction_ratio_sensitivity(
    variable_cols,
    residual_cols,
    normal_validation_res,
    fault3_development_res,
    normal_evaluation_res,
    fault3_final_res,
):
    print("\n" + "=" * 80)
    print("Direction ratio sensitivity analysis")
    print("=" * 80)

    rows = []
    selected_records = []

    for threshold in DIRECTION_RATIO_LIST:
        print(f"\nTesting direction_ratio_threshold = {threshold}")

        metrics_all, stable, selected = screen_selected_residuals_for_threshold(
            normal_validation_res=normal_validation_res,
            fault3_development_res=fault3_development_res,
            variable_cols=variable_cols,
            residual_cols=residual_cols,
            direction_ratio_threshold=threshold,
        )

        if len(selected) == 0:
            print("No variables selected. Skip this threshold.")
            rows.append(
                {
                    "direction_ratio_threshold": threshold,
                    "n_direction_stable_variables": len(stable),
                    "n_selected_variables": 0,
                    "selected_variables": "",
                    "cusum_k": DEFAULT_CUSUM_K,
                    "cusum_control_limit": np.nan,
                    "normal_run_level_far": np.nan,
                    "normal_point_level_far": np.nan,
                    "fault3_pre_fault_run_far": np.nan,
                    "fault3_pre_fault_point_far": np.nan,
                    "fault3_run_level_detection_rate": np.nan,
                    "fault3_post_fault_point_detection_rate": np.nan,
                    "median_alarm_delay_min": np.nan,
                    "mean_alarm_delay_min": np.nan,
                    "detected_runs": np.nan,
                }
            )
            continue

        eval_metrics = evaluate_full_method_with_selected(
            selected=selected,
            cusum_k=DEFAULT_CUSUM_K,
            normal_validation_res=normal_validation_res,
            normal_evaluation_res=normal_evaluation_res,
            fault3_final_res=fault3_final_res,
        )

        row = {
            "direction_ratio_threshold": threshold,
            "n_direction_stable_variables": len(stable),
            "n_selected_variables": len(selected),
            "selected_variables": ",".join(selected["variable"].tolist()),
            "selected_signs": ",".join(
                selected["sign_for_positive_evidence"].astype(str).tolist()
            ),
        }
        row.update(eval_metrics)
        rows.append(row)

        selected_copy = selected.copy()
        selected_copy["direction_ratio_threshold"] = threshold
        selected_records.append(selected_copy)

        print(f"Stable variables: {len(stable)}")
        print(f"Selected variables: {selected['variable'].tolist()}")
        print(
            "Normal FAR: "
            f"{eval_metrics['normal_run_level_far'] * 100:.2f}% | "
            "FDR: "
            f"{eval_metrics['fault3_run_level_detection_rate'] * 100:.2f}% | "
            "Median delay: "
            f"{eval_metrics['median_alarm_delay_min']:.2f} min"
        )

    result_df = pd.DataFrame(rows)

    result_path = TABLE_DIR / "07_direction_ratio_sensitivity.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved direction ratio sensitivity table: {result_path}")

    if len(selected_records) > 0:
        selected_all = pd.concat(selected_records, axis=0, ignore_index=True)
    else:
        selected_all = pd.DataFrame()

    selected_path = TABLE_DIR / "07_direction_ratio_selected_variables.csv"
    selected_all.to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"Saved selected variables under different direction ratios: {selected_path}")

    return result_df


def run_cusum_k_sensitivity(
    step04_selected,
    normal_validation_res,
    normal_evaluation_res,
    fault3_final_res,
):
    print("\n" + "=" * 80)
    print("CUSUM k sensitivity analysis")
    print("=" * 80)

    rows = []

    for k in CUSUM_K_LIST:
        print(f"\nTesting CUSUM k = {k}")

        eval_metrics = evaluate_full_method_with_selected(
            selected=step04_selected,
            cusum_k=k,
            normal_validation_res=normal_validation_res,
            normal_evaluation_res=normal_evaluation_res,
            fault3_final_res=fault3_final_res,
        )

        row = {"cusum_k": k}
        row.update(eval_metrics)
        rows.append(row)

        print(
            "Control limit: "
            f"{eval_metrics['cusum_control_limit']:.6f} | "
            "Normal FAR: "
            f"{eval_metrics['normal_run_level_far'] * 100:.2f}% | "
            "FDR: "
            f"{eval_metrics['fault3_run_level_detection_rate'] * 100:.2f}% | "
            "Median delay: "
            f"{eval_metrics['median_alarm_delay_min']:.2f} min"
        )

    result_df = pd.DataFrame(rows)

    result_path = TABLE_DIR / "07_cusum_k_sensitivity.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved CUSUM k sensitivity table: {result_path}")

    return result_df


def plot_direction_ratio_sensitivity(result_df):
    x_labels = result_df["direction_ratio_threshold"].astype(str)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(
        x_labels,
        result_df["normal_run_level_far"] * 100,
        marker="o",
        label="Normal run-level FAR (%)",
    )
    plt.plot(
        x_labels,
        result_df["fault3_run_level_detection_rate"] * 100,
        marker="o",
        label="Fault 3 run-level detection rate (%)",
    )
    plt.xlabel("Direction ratio threshold")
    plt.ylabel("Percentage (%)")
    plt.title("Sensitivity to direction ratio threshold")
    plt.legend()
    plt.tight_layout()

    fig_png = FIGURE_DIR / "fig5a_direction_ratio_sensitivity.png"
    fig_pdf = FIGURE_DIR / "fig5a_direction_ratio_sensitivity.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(
        x_labels,
        result_df["median_alarm_delay_min"],
        marker="o",
        label="Median alarm delay",
    )
    plt.xlabel("Direction ratio threshold")
    plt.ylabel("Median alarm delay (min)")
    plt.title("Alarm delay under different direction ratio thresholds")
    plt.legend()
    plt.tight_layout()

    fig2_png = FIGURE_DIR / "fig5b_direction_ratio_delay.png"
    fig2_pdf = FIGURE_DIR / "fig5b_direction_ratio_delay.pdf"
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()

    print(f"Saved figure: {fig_png}")
    print(f"Saved figure: {fig_pdf}")
    print(f"Saved figure: {fig2_png}")
    print(f"Saved figure: {fig2_pdf}")


def plot_cusum_k_sensitivity(result_df):
    x_labels = result_df["cusum_k"].astype(str)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(
        x_labels,
        result_df["normal_run_level_far"] * 100,
        marker="o",
        label="Normal run-level FAR (%)",
    )
    plt.plot(
        x_labels,
        result_df["fault3_run_level_detection_rate"] * 100,
        marker="o",
        label="Fault 3 run-level detection rate (%)",
    )
    plt.xlabel("CUSUM k")
    plt.ylabel("Percentage (%)")
    plt.title("Sensitivity to CUSUM reference value k")
    plt.legend()
    plt.tight_layout()

    fig_png = FIGURE_DIR / "fig5c_cusum_k_sensitivity.png"
    fig_pdf = FIGURE_DIR / "fig5c_cusum_k_sensitivity.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(
        x_labels,
        result_df["median_alarm_delay_min"],
        marker="o",
        label="Median alarm delay",
    )
    plt.xlabel("CUSUM k")
    plt.ylabel("Median alarm delay (min)")
    plt.title("Alarm delay under different CUSUM k values")
    plt.legend()
    plt.tight_layout()

    fig2_png = FIGURE_DIR / "fig5d_cusum_k_delay.png"
    fig2_pdf = FIGURE_DIR / "fig5d_cusum_k_delay.pdf"
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()

    print(f"Saved figure: {fig_png}")
    print(f"Saved figure: {fig_pdf}")
    print(f"Saved figure: {fig2_png}")
    print(f"Saved figure: {fig2_pdf}")


def main():
    print("=" * 80)
    print("Step 07: Parameter sensitivity analysis")
    print("=" * 80)

    variable_cols = load_variable_columns()
    step04_selected = load_step04_selected_residuals()
    data = load_data()

    print("\nStep 04 selected variables:")
    print(step04_selected["variable"].tolist())

    model = fit_pca_model(data["normal_train"], variable_cols)

    print("\nComputing variable-wise PCA residuals...")
    normal_validation_res, residual_cols = compute_variable_residuals(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = compute_variable_residuals(
        data["normal_evaluation"], variable_cols, model
    )
    fault3_development_res, _ = compute_variable_residuals(
        data["fault3_development"], variable_cols, model
    )
    fault3_final_res, _ = compute_variable_residuals(
        data["fault3_final_test"], variable_cols, model
    )

    direction_ratio_df = run_direction_ratio_sensitivity(
        variable_cols=variable_cols,
        residual_cols=residual_cols,
        normal_validation_res=normal_validation_res,
        fault3_development_res=fault3_development_res,
        normal_evaluation_res=normal_evaluation_res,
        fault3_final_res=fault3_final_res,
    )

    cusum_k_df = run_cusum_k_sensitivity(
        step04_selected=step04_selected,
        normal_validation_res=normal_validation_res,
        normal_evaluation_res=normal_evaluation_res,
        fault3_final_res=fault3_final_res,
    )

    plot_direction_ratio_sensitivity(direction_ratio_df)
    plot_cusum_k_sensitivity(cusum_k_df)

    config = {
        "script": "scripts/07_parameter_sensitivity.py",
        "purpose": "Parameter sensitivity analysis for direction_ratio_threshold and CUSUM k.",
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "n_selected_variables": N_SELECTED_VARIABLES,
        "direction_ratio_list": DIRECTION_RATIO_LIST,
        "cusum_k_list": CUSUM_K_LIST,
        "default_cusum_k_for_direction_ratio_sensitivity": DEFAULT_CUSUM_K,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "output_tables": [
            "results/tables/07_direction_ratio_sensitivity.csv",
            "results/tables/07_direction_ratio_selected_variables.csv",
            "results/tables/07_cusum_k_sensitivity.csv",
        ],
        "output_figures": [
            "results/figures/fig5a_direction_ratio_sensitivity.png",
            "results/figures/fig5b_direction_ratio_delay.png",
            "results/figures/fig5c_cusum_k_sensitivity.png",
            "results/figures/fig5d_cusum_k_delay.png",
        ],
    }

    config_path = CONFIG_DIR / "fault3_parameter_sensitivity_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\nSaved Step 07 config: {config_path}")

    print("\n" + "=" * 80)
    print("Key direction ratio sensitivity results")
    print("=" * 80)
    print(
        direction_ratio_df[
            [
                "direction_ratio_threshold",
                "n_direction_stable_variables",
                "n_selected_variables",
                "selected_variables",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ]
    )

    print("\n" + "=" * 80)
    print("Key CUSUM k sensitivity results")
    print("=" * 80)
    print(
        cusum_k_df[
            [
                "cusum_k",
                "cusum_control_limit",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ]
    )

    print("\nStep 07 finished successfully.")


if __name__ == "__main__":
    main()