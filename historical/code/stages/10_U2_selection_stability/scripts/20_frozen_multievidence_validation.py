"""Step 20: Frozen validation of the Step-19 multi-evidence branches.

Scientific role
---------------
Step 19 was a development-only audit over five prespecified residual-evidence
families.  Its fixed rule selected one family and five residual variables for
each target fault.  This script freezes those exact decisions and evaluates
them without any Step-20 retuning:

* Fault 9: lag-one centered residual product, Top-5
* Fault 15: squared centered residual deviation, Top-5
* CUSUM reference value: k = 0.05 for both faults
* control limit: 95th percentile of normal-validation run maxima

Before loading any target final or companion cache, the script reconstructs
the selected Step-19 development feature means and stops unless every value
matches the saved Step-19 table within 1e-10.  It then evaluates the frozen
method on normal evaluation, target-fault final runs, the onset/horizon-matched
segment, and the Faulty-Training companion set.

Study-history caveat
--------------------
Step 18 previously accessed aggregate final/companion results for other
methods.  Therefore Step 20 is a secondary frozen validation of a newly
defined branch, not a pristine external validation.  The new branch itself is
not selected or tuned using Step-20 outcomes.  A later, separately frozen
multi-fault study is required for broad external-generalization claims.

Place in:
    TEP_Fault3_Project/scripts/20_frozen_multievidence_validation.py

Run from project root:
    python scripts/20_frozen_multievidence_validation.py

Prerequisites:
    Steps 02, 11, 15, 18, and 19.

Only files beginning with ``20_`` and ``fig20`` are written.
"""

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in [TABLE_DIR, FIGURE_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


TARGET_FAULTS = [9, 15]
FAULT_TESTING_START = 161
FAULT_TRAINING_START = 21
MATCHED_ORIGINAL_FIRST_SAMPLE = 141
MATCHED_ORIGINAL_LAST_SAMPLE = 640
MATCHED_SAMPLE_OFFSET = 140
SAMPLING_INTERVAL_MIN = 3
TOP_K = 5
CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95
PARITY_TOLERANCE = 1e-10

EXPECTED_STEP19_SCRIPT_SHA256 = (
    "0a19a1c0210669c201d7cbe42c5824aed2a38cb8a08405c68879cc415a30ee6b"
)

FROZEN_PROTOCOL = {
    9: {
        "family_key": "lag1_product",
        "family_label": "Lag-1 product",
        "family_formula": (
            "(r(t)-normal_center)*(r(t-1)-normal_center)"
        ),
        "variables": ["xmeas_2", "xmv_1", "xmeas_21", "xmv_10", "xmeas_3"],
        "orientation_signs": [1, 1, 1, 1, 1],
    },
    15: {
        "family_key": "squared_deviation",
        "family_label": "Squared deviation",
        "family_formula": "(r(t)-normal_center)^2",
        "variables": ["xmeas_17", "xmeas_11", "xmeas_22", "xmv_7", "xmeas_12"],
        "orientation_signs": [1, 1, 1, 1, 1],
    },
}

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"

STEP19_CONFIG_PATH = CONFIG_DIR / "faults9_15_multievidence_development_audit.json"
STEP19_SELECTED_PATH = TABLE_DIR / "19_multievidence_selected_features.csv"
STEP18_BASELINE_PATH = TABLE_DIR / "18_faults9_15_all_method_results.csv"

METHOD_KEY = "frozen_multievidence_top5_cusum"
METHOD = "Frozen applicability-gated multi-evidence Top-5 residual CUSUM"
METHOD_SHORT = "Multi-evidence Top-5"

BASELINE_METHODS = ["Q-CUSUM", "All-52 signed", "LDA residual"]
COMPARISON_METHODS = [*BASELINE_METHODS, METHOD_SHORT]


def require_file(path: Path, instruction: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    require_file(path, f"place {path.name} in the scripts directory")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not construct an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    step15 = load_module(
        SCRIPT_DIR / "15_fair_sequential_baselines.py",
        "step15_for_frozen_multievidence",
    )
    step11 = step15.load_step11_module()
    step18 = load_module(
        SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
        "step18_for_frozen_multievidence",
    )
    required11 = [
        "load_variable_columns",
        "fit_pca_model",
        "compute_residual_frame",
        "determine_limit",
        "evaluate_normal",
    ]
    required15 = ["compute_cusum"]
    required18 = [
        "load_fault_cache",
        "make_matched_original_segment",
        "evaluate_generic_fault",
    ]
    missing11 = [name for name in required11 if not hasattr(step11, name)]
    missing15 = [name for name in required15 if not hasattr(step15, name)]
    missing18 = [name for name in required18 if not hasattr(step18, name)]
    if missing11 or missing15 or missing18:
        raise AttributeError(
            "Missing dependency functions: "
            f"Step11={missing11}, Step15={missing15}, Step18={missing18}"
        )
    if int(step11.RUN_LEVEL_PERCENTILE) != RUN_LEVEL_PERCENTILE:
        raise ValueError("Step-11 run-level percentile is no longer the frozen 95")
    return step11, step15, step18


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError("Could not parse a required boolean column")
    return mapped.astype(bool)


def verify_and_load_step19_definition() -> tuple[pd.DataFrame, dict]:
    config_path = require_file(STEP19_CONFIG_PATH, "run Step 19")
    selected_path = require_file(STEP19_SELECTED_PATH, "run Step 19")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    checks = {
        "script hash": config.get("script_sha256_at_execution")
        == EXPECTED_STEP19_SCRIPT_SHA256,
        "protocol frozen": config.get("candidate_protocol_frozen_in_source") is True,
        "development only": config.get("development_only") is True,
        "no final performance": config.get("final_performance_evaluated") is False,
        "no companion": config.get("companion_data_loaded") is False,
        "no normal evaluation": config.get("normal_evaluation_loaded") is False,
        "260 candidates": int(config.get("candidate_tests_per_fault", -1)) == 260,
        "2000 bootstraps": int(config.get("bootstrap_replicates", -1)) == 2000,
        "2000 permutations": int(config.get("permutation_replicates", -1)) == 2000,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-19 frozen configuration checks failed: {failed}")

    recommended = {
        str(key): value for key, value in config["recommended_family_by_fault"].items()
    }
    expected_recommended = {
        str(fault_id): protocol["family_key"]
        for fault_id, protocol in FROZEN_PROTOCOL.items()
    }
    if recommended != expected_recommended:
        raise ValueError(
            f"Step-19 recommendations changed: {recommended} != {expected_recommended}"
        )

    selected = pd.read_csv(selected_path, encoding="utf-8-sig")
    selected["recommended_by_fixed_development_rule"] = parse_bool(
        selected["recommended_by_fixed_development_rule"]
    )
    selected["primary_stable_global_fdr"] = parse_bool(
        selected["primary_stable_global_fdr"]
    )
    selected["strict_stable_max_t_fwer"] = parse_bool(
        selected["strict_stable_max_t_fwer"]
    )
    frozen_rows = selected[
        selected["recommended_by_fixed_development_rule"]
    ].copy()

    verified = []
    for fault_id in TARGET_FAULTS:
        protocol = FROZEN_PROTOCOL[fault_id]
        current = frozen_rows[frozen_rows["fault_id"] == fault_id].sort_values(
            "selection_rank"
        )
        if len(current) != TOP_K:
            raise ValueError(f"Fault {fault_id}: frozen Step-19 selection is not Top-5")
        if set(current["family_key"].astype(str)) != {protocol["family_key"]}:
            raise ValueError(f"Fault {fault_id}: frozen evidence family changed")
        variables = current["variable"].astype(str).tolist()
        signs = current["orientation_sign"].astype(int).tolist()
        if variables != protocol["variables"]:
            raise ValueError(
                f"Fault {fault_id}: variable order changed: {variables}"
            )
        if signs != protocol["orientation_signs"]:
            raise ValueError(f"Fault {fault_id}: orientation signs changed")
        if not current["primary_stable_global_fdr"].all():
            raise ValueError(f"Fault {fault_id}: a selected feature failed the primary gate")
        if not current["strict_stable_max_t_fwer"].all():
            raise ValueError(
                f"Fault {fault_id}: a selected feature failed strict FWER"
            )
        if (current["bh_q_global_260"] > 0.05).any():
            raise ValueError(f"Fault {fault_id}: selected global q-value exceeds 0.05")
        if (current["max_t_fwer_adjusted_p"] > 0.05).any():
            raise ValueError(f"Fault {fault_id}: selected max-|t| p-value exceeds 0.05")
        verified.append(current)

    definition = pd.concat(verified, ignore_index=True)
    definition["frozen_cusum_k"] = CUSUM_K
    definition["control_limit_percentile"] = RUN_LEVEL_PERCENTILE
    definition["step20_retuning_allowed"] = False
    return definition, config


def validate_normal_frame(
    frame: pd.DataFrame,
    variable_cols: list[str],
    dataset_name: str,
    expected_runs: np.ndarray,
) -> pd.DataFrame:
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")
    output = frame[required].copy()
    for column in ["simulationRun", "sample"]:
        values = pd.to_numeric(output[column], errors="raise").to_numpy(dtype=float)
        rounded = np.rint(values)
        if np.any(~np.isfinite(values)) or not np.allclose(
            values, rounded, rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"{dataset_name}: {column} is not integer-valued")
        output[column] = rounded.astype(np.int64)
    output = output.sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    actual_runs = np.sort(output["simulationRun"].unique())
    if not np.array_equal(actual_runs, expected_runs):
        raise ValueError(f"{dataset_name}: unexpected simulation-run IDs")
    if int(output["sample"].min()) != 1 or int(output["sample"].max()) != 960:
        raise ValueError(f"{dataset_name}: unexpected sample range")
    counts = output.groupby("simulationRun")["sample"].nunique()
    if not (counts == 960).all():
        raise ValueError(f"{dataset_name}: incomplete runs")
    if output.duplicated(["simulationRun", "sample"]).any():
        raise ValueError(f"{dataset_name}: duplicate run/sample rows")
    values = output[variable_cols].to_numpy(dtype=float, copy=False)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{dataset_name}: non-finite process values")
    return output


def residual_centers(
    normal_validation_residuals: pd.DataFrame,
    residual_cols: list[str],
) -> dict[str, float]:
    values = normal_validation_residuals[residual_cols].mean(axis=0)
    return {column: float(values[column]) for column in residual_cols}


def feature_values(
    residual_frame: pd.DataFrame,
    residual_column: str,
    family_key: str,
    center: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual = residual_frame[residual_column].to_numpy(dtype=float)
    n_rows = len(residual_frame)
    valid = np.ones(n_rows, dtype=bool)

    if family_key == "signed_level":
        values = residual.copy()
    elif family_key == "absolute_deviation":
        values = np.abs(residual - center)
    elif family_key == "squared_deviation":
        values = np.square(residual - center)
    elif family_key in {"absolute_increment", "lag1_product"}:
        values = np.zeros(n_rows, dtype=float)
        valid[:] = False
        for positions in residual_frame.groupby(
            "simulationRun", sort=True
        ).indices.values():
            positions = np.asarray(positions, dtype=np.int64)
            if len(positions) < 2:
                raise ValueError("Dynamic evidence requires at least two samples per run")
            current = residual[positions]
            if family_key == "absolute_increment":
                values[positions[1:]] = np.abs(current[1:] - current[:-1])
            else:
                values[positions[1:]] = (
                    (current[1:] - center) * (current[:-1] - center)
                )
            valid[positions[1:]] = True
    else:
        raise ValueError(f"Unsupported frozen evidence family: {family_key}")

    if np.any(~np.isfinite(values[valid])):
        raise ValueError(f"Non-finite transformed values for {residual_column}")
    return values, valid


def fit_frozen_scaler(
    normal_validation_residuals: pd.DataFrame,
    centers: dict[str, float],
    definition: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, selected in definition.sort_values("selection_rank").iterrows():
        residual_column = str(selected["residual_variable"])
        family_key = str(selected["family_key"])
        values, valid = feature_values(
            normal_validation_residuals,
            residual_column,
            family_key,
            centers[residual_column],
        )
        mean = float(values[valid].mean())
        std = float(values[valid].std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid normal-validation feature scale for {residual_column}"
            )
        rows.append(
            {
                "fault_id": int(selected["fault_id"]),
                "selection_rank": int(selected["selection_rank"]),
                "family_key": family_key,
                "family_label": str(selected["family_label"]),
                "family_formula": str(selected["family_formula"]),
                "variable": str(selected["variable"]),
                "residual_variable": residual_column,
                "orientation_sign": int(selected["orientation_sign"]),
                "normal_residual_center": centers[residual_column],
                "normal_validation_feature_mean": mean,
                "normal_validation_feature_std": std,
                "step19_normal_run_feature_mean": float(
                    selected["normal_run_feature_mean"]
                ),
                "step19_fault_run_feature_mean": float(
                    selected["fault_run_feature_mean"]
                ),
                "step19_hedges_g": float(selected["hedges_g"]),
                "step19_global_bh_q": float(selected["bh_q_global_260"]),
                "step19_max_t_fwer_p": float(
                    selected["max_t_fwer_adjusted_p"]
                ),
                "step19_bootstrap_top5_frequency": float(
                    selected["bootstrap_top5_selection_frequency"]
                ),
            }
        )
    return pd.DataFrame(rows)


def construct_scalar_input(
    residual_frame: pd.DataFrame,
    scaler: pd.DataFrame,
) -> pd.DataFrame:
    frame = residual_frame.sort_values(
        ["simulationRun", "sample"]
    ).reset_index(drop=True)
    standardized = []
    for _, row in scaler.sort_values("selection_rank").iterrows():
        values, valid = feature_values(
            frame,
            str(row["residual_variable"]),
            str(row["family_key"]),
            float(row["normal_residual_center"]),
        )
        z = np.zeros(len(frame), dtype=float)
        z[valid] = int(row["orientation_sign"]) * (
            values[valid] - float(row["normal_validation_feature_mean"])
        ) / float(row["normal_validation_feature_std"])
        standardized.append(z)
    output = frame[["faultNumber", "simulationRun", "sample"]].copy()
    output["scalar_input"] = np.column_stack(standardized).mean(axis=1)
    if np.any(~np.isfinite(output["scalar_input"].to_numpy(dtype=float))):
        raise ValueError("Frozen scalar input contains non-finite values")
    return output


def selected_window_feature_means(
    residual_frame: pd.DataFrame,
    scaler: pd.DataFrame,
    start_sample: int,
    end_sample: int,
) -> dict[str, float]:
    frame = residual_frame.sort_values(
        ["simulationRun", "sample"]
    ).reset_index(drop=True)
    result = {}
    for _, row in scaler.sort_values("selection_rank").iterrows():
        values, valid = feature_values(
            frame,
            str(row["residual_variable"]),
            str(row["family_key"]),
            float(row["normal_residual_center"]),
        )
        mask = (
            frame["sample"].between(start_sample, end_sample).to_numpy()
            & valid
        )
        temporary = frame.loc[mask, ["simulationRun", "sample"]].copy()
        temporary["feature"] = values[mask]
        expected_count = end_sample - start_sample + 1
        counts = temporary.groupby("simulationRun").size()
        if len(counts) != 250 or not (counts == expected_count).all():
            raise ValueError(
                f"Incomplete reconstruction window for {row['variable']}"
            )
        run_means = temporary.groupby("simulationRun")["feature"].mean()
        result[str(row["variable"])] = float(run_means.mean())
    return result


def verify_step19_reconstruction(
    fault_id: int,
    normal_validation_residuals: pd.DataFrame,
    development_residuals: pd.DataFrame,
    scaler: pd.DataFrame,
) -> pd.DataFrame:
    normal_means = selected_window_feature_means(
        normal_validation_residuals,
        scaler,
        FAULT_TESTING_START,
        320,
    )
    fault_means = selected_window_feature_means(
        development_residuals,
        scaler,
        FAULT_TESTING_START,
        320,
    )
    rows = []
    for _, row in scaler.sort_values("selection_rank").iterrows():
        variable = str(row["variable"])
        normal_difference = (
            normal_means[variable] - float(row["step19_normal_run_feature_mean"])
        )
        fault_difference = (
            fault_means[variable] - float(row["step19_fault_run_feature_mean"])
        )
        rows.append(
            {
                "fault_id": fault_id,
                "family_key": str(row["family_key"]),
                "selection_rank": int(row["selection_rank"]),
                "variable": variable,
                "reconstructed_normal_run_feature_mean": normal_means[variable],
                "step19_normal_run_feature_mean": float(
                    row["step19_normal_run_feature_mean"]
                ),
                "normal_difference": normal_difference,
                "reconstructed_fault_run_feature_mean": fault_means[variable],
                "step19_fault_run_feature_mean": float(
                    row["step19_fault_run_feature_mean"]
                ),
                "fault_difference": fault_difference,
                "exact_with_tolerance_1e_10": bool(
                    abs(normal_difference) <= PARITY_TOLERANCE
                    and abs(fault_difference) <= PARITY_TOLERANCE
                ),
            }
        )
    parity = pd.DataFrame(rows)
    if not parity["exact_with_tolerance_1e_10"].all():
        raise ValueError(
            f"Fault {fault_id}: Step-19 feature reconstruction parity failed"
        )
    return parity


def annotate_runs(
    frame: pd.DataFrame,
    fault_id: int,
    dataset: str,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "dataset", dataset)
    output.insert(0, "method", METHOD)
    output.insert(0, "method_key", METHOD_KEY)
    output.insert(0, "fault_id", fault_id)
    return output


def evaluate_frozen_branch(
    step11,
    step15,
    step18,
    fault_id: int,
    scaler: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_stat = step15.compute_cusum(inputs["normal_validation"], CUSUM_K)
    limit, validation_run_max = step11.determine_limit(validation_stat)
    validation_metrics, validation_runs = step11.evaluate_normal(
        validation_stat, limit
    )

    normal_stat = step15.compute_cusum(inputs["normal_evaluation"], CUSUM_K)
    normal_metrics, normal_runs = step11.evaluate_normal(normal_stat, limit)

    common = {
        "fault_id": fault_id,
        "method_key": METHOD_KEY,
        "method": METHOD,
        "method_short": METHOD_SHORT,
        "evidence_family": str(scaler.iloc[0]["family_key"]),
        "selected_variables": ";".join(scaler["variable"].astype(str)),
        "selected_orientations": ";".join(
            scaler["orientation_sign"].astype(int).astype(str)
        ),
        "n_selected_variables": int(len(scaler)),
        "sequential": True,
        "cusum_k": CUSUM_K,
        "cusum_k_source": "frozen before Step-20 evaluation",
        "control_limit": float(limit),
        "control_limit_percentile": RUN_LEVEL_PERCENTILE,
        "normal_validation_run_level_far": float(
            validation_metrics["normal_run_level_far"]
        ),
        "normal_run_level_far": float(normal_metrics["normal_run_level_far"]),
        "normal_run_level_far_ci_low": float(
            normal_metrics["normal_run_level_far_ci_low"]
        ),
        "normal_run_level_far_ci_high": float(
            normal_metrics["normal_run_level_far_ci_high"]
        ),
        "protocol_status": "frozen after Step 19; no Step-20 retuning",
    }

    dataset_specs = [
        (
            "development_fit_runs_1_250",
            FAULT_TESTING_START,
            960,
            "development",
        ),
        (
            "faulty_testing_final_full_960",
            FAULT_TESTING_START,
            960,
            "final",
        ),
        (
            "original_onset_horizon_matched",
            FAULT_TRAINING_START,
            500,
            "matched",
        ),
        (
            "faulty_training_companion",
            FAULT_TRAINING_START,
            500,
            "companion",
        ),
    ]

    rows = []
    run_tables = []
    for dataset, start, run_length, input_key in dataset_specs:
        statistic = step15.compute_cusum(inputs[input_key], CUSUM_K)
        metrics, runs = step18.evaluate_generic_fault(
            step11, statistic, float(limit), start
        )
        row = {
            "dataset": dataset,
            "fault_start_sample": start,
            "pre_fault_samples_per_run": start - 1,
            "post_fault_samples_per_run": run_length - start + 1,
        }
        row.update(common)
        row.update(metrics)
        rows.append(row)
        run_tables.append(annotate_runs(runs, fault_id, dataset))

    validation_run_max = validation_run_max.copy()
    validation_run_max.insert(0, "fault_id", fault_id)
    validation_run_max.insert(1, "evidence_family", common["evidence_family"])
    validation_run_max.insert(2, "cusum_k", CUSUM_K)
    validation_run_max.insert(3, "control_limit", float(limit))

    normal_runs = normal_runs.copy()
    normal_runs.insert(0, "fault_id", fault_id)
    normal_runs.insert(1, "method_key", METHOD_KEY)
    normal_runs.insert(2, "method", METHOD)
    normal_runs.insert(3, "dataset", "normal_evaluation")

    return (
        pd.DataFrame(rows),
        pd.concat(run_tables, ignore_index=True),
        validation_run_max,
        normal_runs,
    )


def build_comparison(results: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(
        require_file(STEP18_BASELINE_PATH, "run Step 18"),
        encoding="utf-8-sig",
    )
    baseline = baseline[
        baseline["fault_id"].isin(TARGET_FAULTS)
        & baseline["method_short"].isin(BASELINE_METHODS)
        & baseline["dataset"].isin(
            [
                "faulty_testing_final_full_960",
                "original_onset_horizon_matched",
                "faulty_training_companion",
            ]
        )
    ].copy()
    new = results[
        results["dataset"] != "development_fit_runs_1_250"
    ].copy()
    common_columns = [
        "fault_id",
        "dataset",
        "method_key",
        "method",
        "method_short",
        "cusum_k",
        "normal_run_level_far",
        "fault_pre_fault_run_alarm_rate",
        "fault_run_level_detection_rate",
        "fault_post_fault_point_detection_rate",
        "median_alarm_delay_min",
        "detected_runs",
        "right_censored_runs",
    ]
    comparison = pd.concat(
        [baseline[common_columns], new[common_columns]], ignore_index=True
    )
    comparison["method_order"] = comparison["method_short"].map(
        {name: index + 1 for index, name in enumerate(COMPARISON_METHODS)}
    )
    return comparison.sort_values(
        ["fault_id", "dataset", "method_order"]
    ).reset_index(drop=True)


def plot_frozen_validation(results: pd.DataFrame) -> list[Path]:
    subset = results[
        results["dataset"].isin(
            [
                "faulty_testing_final_full_960",
                "original_onset_horizon_matched",
                "faulty_training_companion",
            ]
        )
    ].copy()
    datasets = [
        ("faulty_testing_final_full_960", "Final full", "#4C78A8"),
        ("original_onset_horizon_matched", "Original matched", "#7A9E9F"),
        ("faulty_training_companion", "Companion", "#D95F59"),
    ]
    x = np.arange(len(TARGET_FAULTS), dtype=float)
    width = 0.24
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 9.2), sharex=True)
    for position, (dataset, label, color) in enumerate(datasets):
        current = subset[subset["dataset"] == dataset].set_index(
            "fault_id"
        ).reindex(TARGET_FAULTS)
        offset = (position - 1) * width
        axes[0].bar(
            x + offset,
            100 * current["fault_run_level_detection_rate"].to_numpy(dtype=float),
            width,
            color=color,
            label=label,
        )
        axes[1].bar(
            x + offset,
            100
            * current["fault_post_fault_point_detection_rate"].to_numpy(
                dtype=float
            ),
            width,
            color=color,
        )
        axes[2].bar(
            x + offset,
            current["median_alarm_delay_min"].to_numpy(dtype=float),
            width,
            color=color,
        )
    axes[0].set_ylabel("Run detection (%)")
    axes[1].set_ylabel("Post-fault point detection (%)")
    axes[2].set_ylabel("Median delay (min)")
    axes[0].legend(loc="best")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Fault {fault_id}" for fault_id in TARGET_FAULTS])
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Frozen multi-evidence Top-5 validation")
    figure.tight_layout()
    png = FIGURE_DIR / "fig20a_frozen_multievidence_validation.png"
    pdf = FIGURE_DIR / "fig20a_frozen_multievidence_validation.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def plot_final_comparison(comparison: pd.DataFrame) -> list[Path]:
    current = comparison[
        comparison["dataset"] == "faulty_testing_final_full_960"
    ].copy()
    x = np.arange(len(COMPARISON_METHODS), dtype=float)
    width = 0.36
    colors = {9: "#4C78A8", 15: "#D95F59"}
    metrics = [
        ("fault_run_level_detection_rate", "Run detection (%)", 100.0),
        ("fault_post_fault_point_detection_rate", "Post-fault point detection (%)", 100.0),
        ("median_alarm_delay_min", "Median delay (min)", 1.0),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.3), sharex=True)
    for axis, (column, ylabel, multiplier) in zip(axes, metrics):
        for position, fault_id in enumerate(TARGET_FAULTS):
            values = (
                current[current["fault_id"] == fault_id]
                .set_index("method_short")
                .reindex(COMPARISON_METHODS)[column]
                .to_numpy(dtype=float)
                * multiplier
            )
            offset = (position - 0.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                color=colors[fault_id],
                label=f"Fault {fault_id}" if axis is axes[0] else None,
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(loc="best")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [name.replace(" ", "\n") for name in COMPARISON_METHODS]
    )
    figure.suptitle("Final-set comparison under frozen protocols")
    figure.tight_layout()
    png = FIGURE_DIR / "fig20b_frozen_final_comparison.png"
    pdf = FIGURE_DIR / "fig20b_frozen_final_comparison.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def main() -> None:
    print("=" * 88)
    print("Step 20: Frozen multi-evidence validation")
    print("=" * 88)
    print("No Step-20 variable selection, family selection, k tuning, or limit tuning.")
    print(f"Frozen k={CUSUM_K:.2f}; normal-validation limit percentile={RUN_LEVEL_PERCENTILE}.")

    definition, step19_config = verify_and_load_step19_definition()
    print("Step-19 configuration and all 10 frozen selected features verified.")
    step11, step15, step18 = load_dependencies()
    variable_cols = step11.load_variable_columns()

    normal_train = pd.read_pickle(require_file(NORMAL_TRAIN_PATH, "run Step 02"))
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_validation = validate_normal_frame(
        pd.read_pickle(require_file(NORMAL_VALIDATION_PATH, "run Step 02")),
        variable_cols,
        "normal_validation",
        np.arange(1, 251, dtype=np.int64),
    )
    normal_evaluation = validate_normal_frame(
        pd.read_pickle(require_file(NORMAL_EVALUATION_PATH, "run Step 02")),
        variable_cols,
        "normal_evaluation",
        np.arange(251, 501, dtype=np.int64),
    )
    print("Computing frozen-evaluation normal PCA residuals...")
    normal_validation_res, residual_cols = step11.compute_residual_frame(
        normal_validation, variable_cols, model
    )
    normal_evaluation_res, evaluation_cols = step11.compute_residual_frame(
        normal_evaluation, variable_cols, model
    )
    if evaluation_cols != residual_cols:
        raise ValueError("Normal residual column order changed")
    centers = residual_centers(normal_validation_res, residual_cols)
    del normal_validation, normal_evaluation
    gc.collect()

    all_results = []
    all_runs = []
    all_validation_max = []
    all_normal_runs = []
    all_scalers = []
    all_parity = []
    audit_rows = []

    for fault_id in TARGET_FAULTS:
        print("\n" + "=" * 88)
        print(
            f"Fault {fault_id}: {FROZEN_PROTOCOL[fault_id]['family_key']} Top-5, "
            f"k={CUSUM_K:.2f}"
        )
        print("=" * 88)
        current_definition = definition[definition["fault_id"] == fault_id].copy()
        scaler = fit_frozen_scaler(
            normal_validation_res, centers, current_definition
        )
        all_scalers.append(scaler)

        # Development reconstruction is completed before any final/companion
        # cache for this fault is loaded.
        development, audit = step18.load_fault_cache(
            fault_id, "development", variable_cols
        )
        audit_rows.append(audit)
        development_res, current_cols = step11.compute_residual_frame(
            development, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: development residual order changed")
        parity = verify_step19_reconstruction(
            fault_id,
            normal_validation_res,
            development_res,
            scaler,
        )
        all_parity.append(parity)
        print(
            "Step-19 reconstruction parity passed; maximum absolute difference="
            f"{max(parity['normal_difference'].abs().max(), parity['fault_difference'].abs().max()):.3e}"
        )
        development_input = construct_scalar_input(development_res, scaler)
        del development, development_res
        gc.collect()

        # Frozen target evaluation starts only after parity succeeds.
        final, audit = step18.load_fault_cache(fault_id, "final", variable_cols)
        audit_rows.append(audit)
        matched = step18.make_matched_original_segment(final)
        final_res, current_cols = step11.compute_residual_frame(
            final, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: final residual order changed")
        final_input = construct_scalar_input(final_res, scaler)
        del final, final_res
        gc.collect()

        matched_res, current_cols = step11.compute_residual_frame(
            matched, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: matched residual order changed")
        matched_input = construct_scalar_input(matched_res, scaler)
        del matched, matched_res
        gc.collect()

        companion, audit = step18.load_fault_cache(
            fault_id, "companion", variable_cols
        )
        audit_rows.append(audit)
        companion_res, current_cols = step11.compute_residual_frame(
            companion, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: companion residual order changed")
        companion_input = construct_scalar_input(companion_res, scaler)
        del companion, companion_res
        gc.collect()

        inputs = {
            "normal_validation": construct_scalar_input(
                normal_validation_res, scaler
            ),
            "normal_evaluation": construct_scalar_input(
                normal_evaluation_res, scaler
            ),
            "development": development_input,
            "final": final_input,
            "matched": matched_input,
            "companion": companion_input,
        }
        results, runs, validation_max, normal_runs = evaluate_frozen_branch(
            step11, step15, step18, fault_id, scaler, inputs
        )
        all_results.append(results)
        all_runs.append(runs)
        all_validation_max.append(validation_max)
        all_normal_runs.append(normal_runs)

        print(
            results[
                [
                    "dataset",
                    "normal_run_level_far",
                    "fault_run_level_detection_rate",
                    "fault_post_fault_point_detection_rate",
                    "median_alarm_delay_min",
                ]
            ].to_string(index=False)
        )
        del inputs, development_input, final_input, matched_input, companion_input
        gc.collect()

    results = pd.concat(all_results, ignore_index=True)
    runs = pd.concat(all_runs, ignore_index=True)
    validation_max = pd.concat(all_validation_max, ignore_index=True)
    normal_runs = pd.concat(all_normal_runs, ignore_index=True)
    scalers = pd.concat(all_scalers, ignore_index=True)
    parity = pd.concat(all_parity, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    comparison = build_comparison(results)

    output_paths = {
        "definition": TABLE_DIR / "20_frozen_multievidence_method_definition.csv",
        "scalers": TABLE_DIR / "20_frozen_multievidence_scalers.csv",
        "parity": TABLE_DIR / "20_step19_reconstruction_parity.csv",
        "results": TABLE_DIR / "20_frozen_multievidence_results.csv",
        "comparison": TABLE_DIR / "20_frozen_multievidence_comparison.csv",
        "runs": TABLE_DIR / "20_frozen_multievidence_run_results.csv",
        "normal_runs": TABLE_DIR / "20_frozen_multievidence_normal_run_results.csv",
        "validation_max": TABLE_DIR / "20_frozen_multievidence_validation_run_max.csv",
        "data_audit": TABLE_DIR / "20_frozen_multievidence_data_audit.csv",
    }
    definition.to_csv(output_paths["definition"], index=False, encoding="utf-8-sig")
    scalers.to_csv(output_paths["scalers"], index=False, encoding="utf-8-sig")
    parity.to_csv(output_paths["parity"], index=False, encoding="utf-8-sig")
    results.to_csv(output_paths["results"], index=False, encoding="utf-8-sig")
    comparison.to_csv(output_paths["comparison"], index=False, encoding="utf-8-sig")
    runs.to_csv(output_paths["runs"], index=False, encoding="utf-8-sig")
    normal_runs.to_csv(output_paths["normal_runs"], index=False, encoding="utf-8-sig")
    validation_max.to_csv(
        output_paths["validation_max"], index=False, encoding="utf-8-sig"
    )
    audits.to_csv(output_paths["data_audit"], index=False, encoding="utf-8-sig")

    figure_paths = [
        *plot_frozen_validation(results),
        *plot_final_comparison(comparison),
    ]

    config = {
        "script": "scripts/20_frozen_multievidence_validation.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "step19_script_sha256_expected": EXPECTED_STEP19_SCRIPT_SHA256,
        "step19_config_sha256": sha256_file(STEP19_CONFIG_PATH),
        "step19_selected_table_sha256": sha256_file(STEP19_SELECTED_PATH),
        "step19_protocol_verified": True,
        "step19_reconstruction_parity_all_passed": bool(
            parity["exact_with_tolerance_1e_10"].all()
        ),
        "step19_reconstruction_tolerance": PARITY_TOLERANCE,
        "target_faults": TARGET_FAULTS,
        "frozen_protocol": FROZEN_PROTOCOL,
        "top_k": TOP_K,
        "cusum_k": CUSUM_K,
        "cusum_k_tuned_in_step20": False,
        "variables_or_families_changed_in_step20": False,
        "run_level_control_limit_percentile": RUN_LEVEL_PERCENTILE,
        "dynamic_feature_first_sample_handling": (
            "zero standardized evidence at the first sample of each run; no "
            "cross-run lag is used"
        ),
        "normal_feature_center_and_scaler_source": "normal validation only",
        "target_final_or_companion_used_for_scaling_or_tuning": False,
        "study_history": {
            "step18_previously_accessed_target_final_and_companion": True,
            "step18_evaluated_this_new_multievidence_branch": False,
            "interpretation": (
                "secondary frozen validation of a newly defined branch, not a "
                "pristine external validation"
            ),
            "broad_generalization_claim_requires_future_reserved_faults": True,
        },
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "faults9_15_frozen_multievidence_validation.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("Step 20 summary: frozen method")
    print("=" * 88)
    print(
        results[
            [
                "fault_id",
                "evidence_family",
                "dataset",
                "normal_run_level_far",
                "fault_run_level_detection_rate",
                "fault_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print("\nFinal-set comparison:")
    print(
        comparison[
            comparison["dataset"] == "faulty_testing_final_full_960"
        ][
            [
                "fault_id",
                "method_short",
                "cusum_k",
                "normal_run_level_far",
                "fault_run_level_detection_rate",
                "fault_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print(f"Saved configuration: {config_path}")
    print("Step 20 finished successfully.")


if __name__ == "__main__":
    main()
