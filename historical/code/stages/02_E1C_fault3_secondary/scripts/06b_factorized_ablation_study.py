"""
Step 06b: Factorized ablation for the local signed PCA-residual CUSUM.

Purpose
-------
The original Step 06 does not compare the proposed Top-K signed-residual
CUSUM against an all-52-residual counterpart using the same direction
unification.  This script adds that missing controlled comparison and a
direction-stable-set intermediate comparison, while preserving every original
Step 06 method.

Place this file in:
    TEP_Fault3_Project/scripts/06b_factorized_ablation_study.py

Run from the project root with:
    python scripts/06b_factorized_ablation_study.py

Prerequisites
-------------
Run Steps 02 and 04 first.  This script reads their existing processed data and
screening tables; it does not overwrite the original Step 06 outputs.
"""

from pathlib import Path
import json
import re

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
CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95
CONFIDENCE_LEVEL = 0.95


def percentile_compat(x, q):
    """NumPy-version-compatible linear percentile."""
    try:
        return np.percentile(x, q, method="linear")
    except TypeError:
        return np.percentile(x, q, interpolation="linear")


def clopper_pearson(successes, trials, confidence=CONFIDENCE_LEVEL):
    """Exact two-sided Clopper-Pearson interval for a binomial proportion."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return np.nan, np.nan
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")

    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def safe_slug(text):
    """Stable filename slug independent of punctuation such as 'w/o'."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug


def require_file(path, prerequisite):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Please {prerequisite} before running Step 06b."
        )
    return path


def load_variable_columns():
    path = require_file(
        TABLE_DIR / "02_variable_list.csv",
        "run scripts/02_prepare_fault3_data.py",
    )
    info = pd.read_csv(path)
    if "variable" not in info.columns:
        raise ValueError(f"Missing 'variable' column in {path}")
    variables = info["variable"].astype(str).tolist()
    if len(variables) != 52:
        raise ValueError(f"Expected 52 process variables, found {len(variables)}")
    return variables


def load_screening_tables():
    paths = {
        "all_metrics": TABLE_DIR / "04_fault3_residual_direction_metrics.csv",
        "stable": TABLE_DIR / "04_fault3_direction_stable_residuals.csv",
        "selected": TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv",
    }
    for path in paths.values():
        require_file(path, "run scripts/04_local_residual_screening.py")

    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    required = {
        "variable",
        "residual_variable",
        "sign_for_positive_evidence",
    }
    for name, table in tables.items():
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{paths[name]} is missing columns: {missing}")
        if table["residual_variable"].duplicated().any():
            raise ValueError(f"Duplicate residual variables in {paths[name]}")
        signs = set(table["sign_for_positive_evidence"].astype(int).unique())
        if not signs.issubset({-1, 1}):
            raise ValueError(f"Invalid signs in {paths[name]}: {sorted(signs)}")

    return tables


def load_processed_data():
    files = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR
        / "normal_validation_runs_1_250.pkl",
        "normal_evaluation": PROCESSED_DIR
        / "normal_evaluation_runs_251_500.pkl",
        "fault3_final_test": PROCESSED_DIR
        / "fault3_final_test_runs_251_500.pkl",
    }
    for path in files.values():
        require_file(path, "run scripts/02_prepare_fault3_data.py")

    print("Loading processed data...")
    data = {name: pd.read_pickle(path) for name, path in files.items()}
    for name, frame in data.items():
        print(f"  {name}: shape={frame.shape}")
    return data


def fit_pca_model(normal_train, variable_cols):
    """Fit the same standardized PCA model used by Steps 03-06."""
    x = normal_train[variable_cols].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0

    x_std = (x - mean) / std
    pca_full = PCA()
    pca_full.fit(x_std)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumulative, PCA_CUMVAR_TARGET) + 1)

    model = {
        "mean": mean,
        "std": std,
        "components": pca_full.components_[:n_pcs, :],
        "n_pcs": n_pcs,
        "cum_var": float(cumulative[n_pcs - 1]),
    }
    print(
        f"PCA model: {n_pcs} PCs, cumulative explained variance "
        f"{model['cum_var']:.6f}"
    )
    return model


def compute_variable_residuals(frame, variable_cols, model):
    """Return variable-wise PCA reconstruction residuals in standardized space."""
    x = frame[variable_cols].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    loadings = model["components"]
    scores = x_std @ loadings.T
    reconstruction = scores @ loadings
    residual = x_std - reconstruction

    residual_cols = [f"res_{name}" for name in variable_cols]
    ids = frame[["faultNumber", "simulationRun", "sample"]].reset_index(
        drop=True
    )
    values = pd.DataFrame(residual, columns=residual_cols, index=ids.index)
    return pd.concat([ids, values], axis=1), residual_cols


def make_sign_dict(table):
    return {
        str(row["residual_variable"]): int(row["sign_for_positive_evidence"])
        for _, row in table.iterrows()
    }


def validate_residual_set(name, residual_cols, available_cols, signs_dict):
    missing = sorted(set(residual_cols).difference(available_cols))
    if missing:
        raise ValueError(f"{name}: residual columns absent from PCA output: {missing}")
    if not residual_cols:
        raise ValueError(f"{name}: empty residual set")
    if signs_dict is not None:
        missing_signs = sorted(set(residual_cols).difference(signs_dict))
        if missing_signs:
            raise ValueError(f"{name}: missing development signs: {missing_signs}")


def fit_normal_scaler(normal_validation_res, residual_cols, signs_dict):
    """Fit signed residual means/SDs using normal validation data only."""
    rows = []
    for col in residual_cols:
        sign = 1 if signs_dict is None else int(signs_dict[col])
        signed = sign * normal_validation_res[col].to_numpy(dtype=float)
        mean = float(np.mean(signed))
        std = float(np.std(signed, ddof=1))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        rows.append(
            {
                "residual_variable": col,
                "sign": sign,
                "normal_validation_mean": mean,
                "normal_validation_std": std,
            }
        )
    return pd.DataFrame(rows)


def construct_index(residual_df, scaler, index_col="ablation_index"):
    out = residual_df[["faultNumber", "simulationRun", "sample"]].copy()
    z_values = []
    for _, row in scaler.iterrows():
        col = row["residual_variable"]
        signed = int(row["sign"]) * residual_df[col].to_numpy(dtype=float)
        z = (
            signed - float(row["normal_validation_mean"])
        ) / float(row["normal_validation_std"])
        z_values.append(z)
    out[index_col] = np.column_stack(z_values).mean(axis=1)
    return out


def compute_one_sided_cusum(index_df, index_col, k=CUSUM_K):
    """Reset the one-sided upper CUSUM at the start of every run."""
    df = index_df.sort_values(["simulationRun", "sample"]).copy()
    result = np.empty(df.shape[0], dtype=float)

    for positions in df.groupby("simulationRun", sort=True).indices.values():
        values = df.iloc[positions][index_col].to_numpy(dtype=float)
        c = 0.0
        run_result = np.empty(len(values), dtype=float)
        for i, value in enumerate(values):
            c = max(0.0, c + value - k)
            run_result[i] = c
        result[positions] = run_result

    df["monitoring_statistic"] = result
    return df


def use_raw_index(index_df, index_col):
    df = index_df.sort_values(["simulationRun", "sample"]).copy()
    df["monitoring_statistic"] = df[index_col]
    return df


def determine_run_level_limit(normal_validation_stat):
    run_max = (
        normal_validation_stat.groupby("simulationRun")["monitoring_statistic"]
        .max()
        .reset_index(name="run_max_statistic")
    )
    limit = float(
        percentile_compat(
            run_max["run_max_statistic"].to_numpy(), RUN_LEVEL_PERCENTILE
        )
    )
    return limit, run_max


def evaluate_normal(stat_df, limit):
    df = stat_df.copy()
    df["alarm"] = df["monitoring_statistic"] > limit
    run_results = (
        df.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="run_alarm")
    )

    alarm_runs = int(run_results["run_alarm"].sum())
    n_runs = int(len(run_results))
    ci_low, ci_high = clopper_pearson(alarm_runs, n_runs)
    metrics = {
        "normal_n_runs": n_runs,
        "normal_alarm_runs": alarm_runs,
        "normal_run_level_far": alarm_runs / n_runs,
        "normal_run_level_far_ci_low": ci_low,
        "normal_run_level_far_ci_high": ci_high,
        "normal_point_level_far": float(df["alarm"].mean()),
    }
    return metrics, run_results


def evaluate_fault3(stat_df, limit):
    df = stat_df.copy()
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

    all_runs = pd.DataFrame(
        {"simulationRun": sorted(df["simulationRun"].unique())}
    )
    run_results = all_runs.merge(pre_alarm, on="simulationRun", how="left")
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
    pre_ci_low, pre_ci_high = clopper_pearson(pre_alarm_runs, n_runs)
    det_ci_low, det_ci_high = clopper_pearson(detected_runs, n_runs)
    delays = run_results.loc[run_results["detected"], "alarm_delay_min"].dropna()

    metrics = {
        "fault3_n_runs": n_runs,
        "fault3_pre_fault_alarm_runs": pre_alarm_runs,
        "fault3_pre_fault_run_far": pre_alarm_runs / n_runs,
        "fault3_pre_fault_run_far_ci_low": pre_ci_low,
        "fault3_pre_fault_run_far_ci_high": pre_ci_high,
        "fault3_pre_fault_point_far": float(pre["alarm"].mean()),
        "detected_runs": detected_runs,
        "fault3_run_level_detection_rate": detected_runs / n_runs,
        "fault3_run_level_detection_ci_low": det_ci_low,
        "fault3_run_level_detection_ci_high": det_ci_high,
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


def run_method(
    method,
    normal_validation_res,
    normal_evaluation_res,
    fault3_final_res,
    available_residual_cols,
):
    name = method["method"]
    residual_cols = method["residual_cols"]
    signs_dict = method["signs_dict"]
    use_cusum = bool(method["use_cusum"])
    validate_residual_set(
        name, residual_cols, available_residual_cols, signs_dict
    )

    print("\n" + "=" * 80)
    print(f"Running: {name}")
    print("=" * 80)

    scaler = fit_normal_scaler(
        normal_validation_res, residual_cols, signs_dict
    )
    validation_index = construct_index(normal_validation_res, scaler)
    normal_index = construct_index(normal_evaluation_res, scaler)
    fault_index = construct_index(fault3_final_res, scaler)

    if use_cusum:
        validation_stat = compute_one_sided_cusum(
            validation_index, "ablation_index"
        )
        normal_stat = compute_one_sided_cusum(normal_index, "ablation_index")
        fault_stat = compute_one_sided_cusum(fault_index, "ablation_index")
    else:
        validation_stat = use_raw_index(validation_index, "ablation_index")
        normal_stat = use_raw_index(normal_index, "ablation_index")
        fault_stat = use_raw_index(fault_index, "ablation_index")

    limit, validation_run_max = determine_run_level_limit(validation_stat)
    normal_metrics, normal_run_results = evaluate_normal(normal_stat, limit)
    fault_metrics, fault_run_results = evaluate_fault3(fault_stat, limit)

    metrics = {
        "method": name,
        "method_short": method["method_short"],
        "slug": method["slug"],
        "residual_set": method["residual_set"],
        "n_residual_variables": len(residual_cols),
        "direction_unification": signs_dict is not None,
        "sign_source": method["sign_source"],
        "use_cusum": use_cusum,
        "cusum_k": CUSUM_K if use_cusum else np.nan,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "control_limit": limit,
    }
    metrics.update(normal_metrics)
    metrics.update(fault_metrics)

    print(f"Control limit: {limit:.6f}")
    print(
        "Normal run FAR: "
        f"{100 * metrics['normal_run_level_far']:.2f}% "
        f"({metrics['normal_alarm_runs']}/{metrics['normal_n_runs']})"
    )
    print(
        "Fault 3 detection: "
        f"{100 * metrics['fault3_run_level_detection_rate']:.2f}% "
        f"({metrics['detected_runs']}/{metrics['fault3_n_runs']})"
    )
    print(f"Median delay: {metrics['median_alarm_delay_min']} min")

    return {
        "metrics": metrics,
        "scaler": scaler,
        "validation_run_max": validation_run_max,
        "normal_run_results": normal_run_results,
        "fault3_run_results": fault_run_results,
    }


def make_contrast_table(metrics_df):
    """Create reviewer-facing controlled contrasts (target minus comparator)."""
    contrast_specs = [
        {
            "contrast": "Direction unification within Top-5",
            "comparator": "Selected residuals w/o sign + CUSUM",
            "target": "Selected signed residuals + CUSUM",
            "isolated_component": "direction unification",
        },
        {
            "contrast": "CUSUM within signed Top-5",
            "comparator": "Selected signed residuals w/o CUSUM",
            "target": "Selected signed residuals + CUSUM",
            "isolated_component": "sequential accumulation",
        },
        {
            "contrast": "Top-5 selection after sign unification",
            "comparator": "All signed residuals + CUSUM",
            "target": "Selected signed residuals + CUSUM",
            "isolated_component": "local Top-K selection",
        },
        {
            "contrast": "Direction-stability filtering",
            "comparator": "All signed residuals + CUSUM",
            "target": "Direction-stable signed residuals + CUSUM",
            "isolated_component": "direction-ratio filter",
        },
        {
            "contrast": "Top-5 compression within stable set",
            "comparator": "Direction-stable signed residuals + CUSUM",
            "target": "Selected signed residuals + CUSUM",
            "isolated_component": "magnitude ranking and Top-K compression",
        },
    ]

    indexed = metrics_df.set_index("method")
    rows = []
    for spec in contrast_specs:
        comparator = indexed.loc[spec["comparator"]]
        target = indexed.loc[spec["target"]]
        row = dict(spec)
        row.update(
            {
                "delta_normal_run_far_percentage_points": 100.0
                * (
                    target["normal_run_level_far"]
                    - comparator["normal_run_level_far"]
                ),
                "delta_fault3_detection_percentage_points": 100.0
                * (
                    target["fault3_run_level_detection_rate"]
                    - comparator["fault3_run_level_detection_rate"]
                ),
                "delta_postfault_point_detection_percentage_points": 100.0
                * (
                    target["fault3_post_fault_point_detection_rate"]
                    - comparator["fault3_post_fault_point_detection_rate"]
                ),
                "delta_median_delay_min": target["median_alarm_delay_min"]
                - comparator["median_alarm_delay_min"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_method_outputs(method, result):
    slug = method["slug"]
    outputs = {
        "scaler": TABLE_DIR / f"06b_{slug}_scaler.csv",
        "validation_run_max": TABLE_DIR
        / f"06b_{slug}_validation_run_max.csv",
        "normal_run_results": TABLE_DIR
        / f"06b_{slug}_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / f"06b_{slug}_fault3_run_results.csv",
    }
    for key, path in outputs.items():
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
    return {key: str(path.relative_to(PROJECT_ROOT)) for key, path in outputs.items()}


def plot_results(metrics_df):
    x = np.arange(len(metrics_df))
    labels = metrics_df["method_short"].tolist()
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        x - width / 2,
        100 * metrics_df["normal_run_level_far"],
        width,
        label="Normal run-level FAR",
        color="#7A9E9F",
    )
    ax.bar(
        x + width / 2,
        100 * metrics_df["fault3_run_level_detection_rate"],
        width,
        label="Fault 3 run-level detection",
        color="#D95F59",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Factorized ablation: false alarms and detection")
    ax.legend()
    fig.tight_layout()

    fig1_png = FIGURE_DIR / "fig4a_factorized_ablation_far_detection.png"
    fig1_pdf = FIGURE_DIR / "fig4a_factorized_ablation_far_detection.pdf"
    fig.savefig(fig1_png, dpi=300)
    fig.savefig(fig1_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars = ax.bar(
        x,
        metrics_df["median_alarm_delay_min"],
        color="#4C78A8",
    )
    for bar, value in zip(bars, metrics_df["median_alarm_delay_min"]):
        if pd.isna(value):
            bar.set_height(0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                "No detection",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Median alarm delay (min)")
    ax.set_title("Factorized ablation: alarm delay among detected runs")
    fig.tight_layout()

    fig2_png = FIGURE_DIR / "fig4b_factorized_ablation_alarm_delay.png"
    fig2_pdf = FIGURE_DIR / "fig4b_factorized_ablation_alarm_delay.pdf"
    fig.savefig(fig2_png, dpi=300)
    fig.savefig(fig2_pdf)
    plt.close(fig)

    return [fig1_png, fig1_pdf, fig2_png, fig2_pdf]


def main():
    print("=" * 80)
    print("Step 06b: Factorized ablation study")
    print("=" * 80)

    variable_cols = load_variable_columns()
    screening = load_screening_tables()
    data = load_processed_data()

    model = fit_pca_model(data["normal_train"], variable_cols)
    print("Computing variable-wise PCA residuals...")
    normal_validation_res, all_residual_cols = compute_variable_residuals(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = compute_variable_residuals(
        data["normal_evaluation"], variable_cols, model
    )
    fault3_final_res, _ = compute_variable_residuals(
        data["fault3_final_test"], variable_cols, model
    )

    all_signs = make_sign_dict(screening["all_metrics"])
    stable_cols = screening["stable"]["residual_variable"].astype(str).tolist()
    stable_signs = make_sign_dict(screening["stable"])
    selected_cols = (
        screening["selected"]["residual_variable"].astype(str).tolist()
    )
    selected_signs = make_sign_dict(screening["selected"])

    if set(all_signs) != set(all_residual_cols):
        missing = sorted(set(all_residual_cols).difference(all_signs))
        extra = sorted(set(all_signs).difference(all_residual_cols))
        raise ValueError(
            "Step 04 all-residual sign table does not match the 52 PCA residuals. "
            f"Missing={missing}; extra={extra}"
        )

    method_specs = [
        {
            "method": "All residuals + CUSUM",
            "method_short": "All / no sign / CUSUM",
            "residual_set": "all_52",
            "residual_cols": all_residual_cols,
            "signs_dict": None,
            "sign_source": "none",
            "use_cusum": True,
            "purpose": "Original all-residual, no-sign Step 06 comparator.",
        },
        {
            "method": "All signed residuals + CUSUM",
            "method_short": "All / signed / CUSUM",
            "residual_set": "all_52",
            "residual_cols": all_residual_cols,
            "signs_dict": all_signs,
            "sign_source": "Step 04 development mean paired shift",
            "use_cusum": True,
            "purpose": (
                "Missing controlled comparator: all 52 residuals use the same "
                "development-derived direction unification and CUSUM as the proposal."
            ),
        },
        {
            "method": "Direction-stable signed residuals + CUSUM",
            "method_short": "Stable set / signed / CUSUM",
            "residual_set": "direction_stable",
            "residual_cols": stable_cols,
            "signs_dict": stable_signs,
            "sign_source": "Step 04 development mean paired shift",
            "use_cusum": True,
            "purpose": (
                "Intermediate comparator after direction-ratio filtering but before "
                "Top-K magnitude ranking."
            ),
        },
        {
            "method": "Selected residuals w/o sign + CUSUM",
            "method_short": "Top-5 / no sign / CUSUM",
            "residual_set": "selected_top_k",
            "residual_cols": selected_cols,
            "signs_dict": None,
            "sign_source": "none",
            "use_cusum": True,
            "purpose": "Original sign-unification ablation.",
        },
        {
            "method": "Selected signed residuals w/o CUSUM",
            "method_short": "Top-5 / signed / no CUSUM",
            "residual_set": "selected_top_k",
            "residual_cols": selected_cols,
            "signs_dict": selected_signs,
            "sign_source": "Step 04 development mean paired shift",
            "use_cusum": False,
            "purpose": "Original sequential-accumulation ablation.",
        },
        {
            "method": "Selected signed residuals + CUSUM",
            "method_short": "Top-5 / signed / CUSUM",
            "residual_set": "selected_top_k",
            "residual_cols": selected_cols,
            "signs_dict": selected_signs,
            "sign_source": "Step 04 development mean paired shift",
            "use_cusum": True,
            "purpose": "Full proposed method.",
        },
    ]
    for method in method_specs:
        method["slug"] = safe_slug(method["method"])

    metrics_rows = []
    method_outputs = {}
    for method in method_specs:
        result = run_method(
            method,
            normal_validation_res,
            normal_evaluation_res,
            fault3_final_res,
            all_residual_cols,
        )
        metrics_rows.append(result["metrics"])
        method_outputs[method["method"]] = save_method_outputs(method, result)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = TABLE_DIR / "06b_factorized_ablation_results.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    definition_rows = []
    for method in method_specs:
        definition_rows.append(
            {
                "method": method["method"],
                "method_short": method["method_short"],
                "residual_set": method["residual_set"],
                "n_residual_variables": len(method["residual_cols"]),
                "direction_unification": method["signs_dict"] is not None,
                "sign_source": method["sign_source"],
                "use_cusum": method["use_cusum"],
                "purpose": method["purpose"],
            }
        )
    definitions_df = pd.DataFrame(definition_rows)
    definitions_path = TABLE_DIR / "06b_factorized_ablation_definitions.csv"
    definitions_df.to_csv(definitions_path, index=False, encoding="utf-8-sig")

    contrast_df = make_contrast_table(metrics_df)
    contrast_path = TABLE_DIR / "06b_factorized_ablation_contrasts.csv"
    contrast_df.to_csv(contrast_path, index=False, encoding="utf-8-sig")

    figure_paths = plot_results(metrics_df)

    config = {
        "script": "scripts/06b_factorized_ablation_study.py",
        "purpose": (
            "Controlled factorized ablation of residual-set size, direction "
            "unification, and CUSUM accumulation."
        ),
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "cusum_k": CUSUM_K,
        "run_level_percentile": RUN_LEVEL_PERCENTILE,
        "confidence_interval": (
            f"{int(CONFIDENCE_LEVEL * 100)}% exact Clopper-Pearson for "
            "run-level binomial rates"
        ),
        "normal_threshold_data": "normal_validation_runs_1_250.pkl",
        "normal_final_evaluation_data": (
            "normal_evaluation_runs_251_500.pkl"
        ),
        "fault_final_test_data": "fault3_final_test_runs_251_500.pkl",
        "all_direction_sign_source": (
            "04_fault3_residual_direction_metrics.csv generated from "
            "Fault 3 development runs 1-250 and matched normal validation runs 1-250"
        ),
        "n_all_residuals": len(all_residual_cols),
        "n_direction_stable_residuals": len(stable_cols),
        "n_selected_residuals": len(selected_cols),
        "selected_residuals": selected_cols,
        "methods": definition_rows,
        "method_outputs": method_outputs,
        "summary_outputs": {
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "definitions": str(definitions_path.relative_to(PROJECT_ROOT)),
            "contrasts": str(contrast_path.relative_to(PROJECT_ROOT)),
            "figures": [str(p.relative_to(PROJECT_ROOT)) for p in figure_paths],
        },
    }
    config_path = CONFIG_DIR / "fault3_factorized_ablation_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    display_cols = [
        "method",
        "n_residual_variables",
        "normal_run_level_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
        "median_alarm_delay_min",
    ]
    print("\n" + "=" * 80)
    print("Step 06b summary")
    print("=" * 80)
    print(metrics_df[display_cols].to_string(index=False))
    print(f"\nSaved main results: {metrics_path}")
    print(f"Saved controlled contrasts: {contrast_path}")
    print(f"Saved configuration: {config_path}")
    print("Step 06b finished successfully.")


if __name__ == "__main__":
    main()
