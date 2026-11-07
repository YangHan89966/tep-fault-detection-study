"""Step 19: Development-only multi-evidence residual audit.

Purpose
-------
Step 18 showed that the signed run-mean gate abstains on TEP Faults 9 and 15.
This script asks a narrower development question: do other *prespecified*
sample-level residual transforms provide stable, sparse evidence?

The candidate families are fixed in this file before Step-19 results:

1. signed residual level:                 r(t)
2. absolute deviation from normal center: |r(t) - c|
3. squared deviation from normal center:  (r(t) - c)^2
4. absolute one-step increment:            |r(t) - r(t-1)|
5. lag-one centered product:               (r(t)-c)(r(t-1)-c)

All screening uses independent simulation-run summaries over samples 161--320.
Inference uses 2,000 independent run bootstraps plus 2,000 studentized label
permutations.  The primary gate requires both a two-sided 95% bootstrap
interval excluding zero and a global Benjamini-Hochberg q <= 0.05 across all
5 x 52 candidates for the fault.  Westfall-Young-style max-|t| adjusted
permutation p-values are also reported as a stricter FWER diagnostic.

Data firewall
-------------
Only the following are read:

* FaultFree Training (for the unchanged PCA model)
* normal-validation runs 1--250
* Fault-9 development runs 1--250 cached by Step 18
* Fault-15 development runs 1--250 cached by Step 18

No final runs, normal-evaluation runs, or Faulty-Training companion runs are
loaded.  This script performs no detector tuning and no final performance
evaluation.

Place in:
    TEP_Fault3_Project/scripts/19_multievidence_development_audit.py

Run from project root:
    python scripts/19_multievidence_development_audit.py

Prerequisites:
    Steps 02, 11, and 18 (Step 18 must have created the development caches).

Only files beginning with ``19_`` and ``fig19`` are written.
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
EXPECTED_RUNS = np.arange(1, 251, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SCREEN_END_SAMPLE = 320
BASE_WINDOW_START = FAULT_START_SAMPLE - 1
SCREEN_WINDOW_LENGTH = SCREEN_END_SAMPLE - FAULT_START_SAMPLE + 1

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BATCH_SIZE = 100
PERMUTATION_REPLICATES = 2000
CONFIDENCE = 0.95
GLOBAL_FDR_ALPHA = 0.05
FWER_ALPHA = 0.05
TOP_K = 5
RANDOM_SEED = 20260714


FAMILY_SPECS = [
    {
        "key": "signed_level",
        "label": "Signed level",
        "formula": "r(t)",
        "priority": 1,
    },
    {
        "key": "absolute_deviation",
        "label": "Absolute deviation",
        "formula": "abs(r(t) - normal_center)",
        "priority": 2,
    },
    {
        "key": "squared_deviation",
        "label": "Squared deviation",
        "formula": "(r(t) - normal_center)^2",
        "priority": 3,
    },
    {
        "key": "absolute_increment",
        "label": "Absolute increment",
        "formula": "abs(r(t) - r(t-1))",
        "priority": 4,
    },
    {
        "key": "lag1_product",
        "label": "Lag-1 product",
        "formula": "(r(t)-normal_center)*(r(t-1)-normal_center)",
        "priority": 5,
    },
]

FAMILY_KEYS = [spec["key"] for spec in FAMILY_SPECS]
FAMILY_LABELS = {spec["key"]: spec["label"] for spec in FAMILY_SPECS}
FAMILY_PRIORITY = {spec["key"]: spec["priority"] for spec in FAMILY_SPECS}


NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
DEVELOPMENT_PATHS = {
    fault_id: PROCESSED_DIR / f"fault{fault_id}_development_runs_1_250.pkl"
    for fault_id in TARGET_FAULTS
}

# These paths are documented as a firewall and are never passed to a reader.
FORBIDDEN_EVALUATION_PATHS = [
    PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl",
    *[
        PROCESSED_DIR / f"fault{fault_id}_final_test_runs_251_500.pkl"
        for fault_id in TARGET_FAULTS
    ],
    *[
        PROCESSED_DIR / f"fault{fault_id}_companion_training_runs_1_500.pkl"
        for fault_id in TARGET_FAULTS
    ],
]


def require_file(path: Path, instruction: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_module(path: Path, module_name: str):
    require_file(path, f"place {path.name} in the scripts directory")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not construct import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_step11_module():
    step11 = load_module(
        SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
        "step11_for_multievidence_audit",
    )
    required = [
        "load_variable_columns",
        "fit_pca_model",
        "compute_residual_frame",
    ]
    missing = [name for name in required if not hasattr(step11, name)]
    if missing:
        raise AttributeError(f"Step 11 is missing required functions: {missing}")
    if float(step11.PCA_CUMVAR_TARGET) != 0.95:
        raise ValueError("Step-11 PCA cumulative-variance target is no longer 0.95")
    return step11


def normalize_integer_column(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    rounded = np.rint(values)
    if np.any(~np.isfinite(values)) or not np.allclose(
        values, rounded, rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"Column {column} is not finite integer-valued")
    frame[column] = rounded.astype(np.int64)


def validate_run_frame(
    frame: pd.DataFrame,
    variable_cols: list[str],
    dataset_name: str,
    expected_fault: int | None,
) -> pd.DataFrame:
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")
    output = frame[required].copy()
    for column in ["faultNumber", "simulationRun", "sample"]:
        normalize_integer_column(output, column)
    output = output.sort_values(["simulationRun", "sample"]).reset_index(drop=True)

    actual_runs = np.sort(output["simulationRun"].unique())
    if not np.array_equal(actual_runs, EXPECTED_RUNS):
        raise ValueError(
            f"{dataset_name}: expected runs 1--250, found "
            f"{len(actual_runs)} runs from {actual_runs.min()} to {actual_runs.max()}"
        )
    if (
        int(output["sample"].min()) != EXPECTED_SAMPLE_MIN
        or int(output["sample"].max()) != EXPECTED_SAMPLE_MAX
    ):
        raise ValueError(f"{dataset_name}: unexpected sample range")
    counts = output.groupby("simulationRun")["sample"].nunique()
    if not (counts == EXPECTED_SAMPLE_MAX).all():
        raise ValueError(f"{dataset_name}: incomplete simulation runs")
    if output.duplicated(["simulationRun", "sample"]).any():
        raise ValueError(f"{dataset_name}: duplicate run/sample identifiers")
    if expected_fault is not None:
        faults = set(output["faultNumber"].unique())
        if faults != {int(expected_fault)}:
            raise ValueError(
                f"{dataset_name}: expected only Fault {expected_fault}, found {faults}"
            )
    values = output[variable_cols].to_numpy(dtype=float, copy=False)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{dataset_name}: non-finite process values")
    return output


def extract_base_tensor(
    residual_frame: pd.DataFrame,
    residual_cols: list[str],
    dataset_name: str,
) -> np.ndarray:
    window = residual_frame[
        residual_frame["sample"].between(BASE_WINDOW_START, SCREEN_END_SAMPLE)
    ].sort_values(["simulationRun", "sample"])
    expected_per_run = SCREEN_END_SAMPLE - BASE_WINDOW_START + 1
    counts = window.groupby("simulationRun").size()
    if len(counts) != len(EXPECTED_RUNS) or not (counts == expected_per_run).all():
        raise ValueError(f"{dataset_name}: incomplete Step-19 screening window")
    for run_id, group in window.groupby("simulationRun", sort=True):
        expected_samples = np.arange(
            BASE_WINDOW_START, SCREEN_END_SAMPLE + 1, dtype=np.int64
        )
        if not np.array_equal(group["sample"].to_numpy(dtype=np.int64), expected_samples):
            raise ValueError(
                f"{dataset_name}: sample order mismatch in simulation run {run_id}"
            )
    tensor = window[residual_cols].to_numpy(dtype=float).reshape(
        len(EXPECTED_RUNS), expected_per_run, len(residual_cols)
    )
    if tensor.shape != (250, 161, 52):
        raise ValueError(f"{dataset_name}: unexpected residual tensor shape {tensor.shape}")
    return tensor


def build_feature_run_means(
    base_tensor: np.ndarray,
    normal_center: np.ndarray,
) -> dict[str, np.ndarray]:
    if base_tensor.ndim != 3 or base_tensor.shape[1] != 161:
        raise ValueError("Feature construction requires a runs x 161 x variables tensor")
    if normal_center.shape != (base_tensor.shape[2],):
        raise ValueError("Normal residual center has the wrong length")

    previous = base_tensor[:, :-1, :]
    current = base_tensor[:, 1:, :]
    if current.shape[1] != SCREEN_WINDOW_LENGTH:
        raise AssertionError("Static/dynamic feature windows are not aligned")
    centered_current = current - normal_center[None, None, :]
    centered_previous = previous - normal_center[None, None, :]

    feature_means = {
        "signed_level": current.mean(axis=1),
        "absolute_deviation": np.abs(centered_current).mean(axis=1),
        "squared_deviation": np.square(centered_current).mean(axis=1),
        "absolute_increment": np.abs(current - previous).mean(axis=1),
        "lag1_product": (centered_current * centered_previous).mean(axis=1),
    }
    for family_key, matrix in feature_means.items():
        if matrix.shape != (base_tensor.shape[0], base_tensor.shape[2]):
            raise ValueError(f"{family_key}: unexpected run-mean feature shape")
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"{family_key}: non-finite run-mean features")
    return feature_means


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)):
        raise ValueError("BH adjustment requires a finite one-dimensional p-value array")
    p = np.clip(p, 0.0, 1.0)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def welch_t_statistic(normal: np.ndarray, fault: np.ndarray) -> np.ndarray:
    normal_mean = normal.mean(axis=0)
    fault_mean = fault.mean(axis=0)
    normal_var = normal.var(axis=0, ddof=1)
    fault_var = fault.var(axis=0, ddof=1)
    standard_error = np.sqrt(
        normal_var / normal.shape[0] + fault_var / fault.shape[0]
    )
    return np.divide(
        fault_mean - normal_mean,
        standard_error,
        out=np.zeros_like(normal_mean),
        where=standard_error > 0,
    )


def run_studentized_permutations(
    normal: np.ndarray,
    fault: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normal.shape != fault.shape:
        raise ValueError("Step-19 permutation test expects equal normal/fault run counts")
    observed_abs_t = np.abs(welch_t_statistic(normal, fault))
    combined = np.vstack([normal, fault])
    n_normal = normal.shape[0]
    n_total = combined.shape[0]
    rng = np.random.default_rng(seed)
    marginal_exceedances = np.zeros(normal.shape[1], dtype=np.int64)
    max_statistics = np.empty(PERMUTATION_REPLICATES, dtype=float)

    for replicate in range(PERMUTATION_REPLICATES):
        permutation = rng.permutation(n_total)
        perm_normal = combined[permutation[:n_normal], :]
        perm_fault = combined[permutation[n_normal:], :]
        perm_abs_t = np.abs(welch_t_statistic(perm_normal, perm_fault))
        marginal_exceedances += perm_abs_t >= observed_abs_t
        max_statistics[replicate] = float(np.nanmax(perm_abs_t))
        if (replicate + 1) % 500 == 0:
            print(
                f"    permutation {replicate + 1}/{PERMUTATION_REPLICATES}"
            )

    denominator = PERMUTATION_REPLICATES + 1.0
    marginal_p = (marginal_exceedances + 1.0) / denominator
    max_t_p = (
        (max_statistics[:, None] >= observed_abs_t[None, :]).sum(axis=0) + 1.0
    ) / denominator
    return observed_abs_t, marginal_p, max_t_p


def independent_run_bootstrap(
    normal: np.ndarray,
    fault: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(
        (BOOTSTRAP_REPLICATES, normal.shape[1]), dtype=np.float64
    )
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPLICATES)
        batch = stop - start
        normal_indices = rng.integers(
            0, normal.shape[0], size=(batch, normal.shape[0])
        )
        fault_indices = rng.integers(
            0, fault.shape[0], size=(batch, fault.shape[0])
        )
        output[start:stop, :] = (
            fault[fault_indices, :].mean(axis=1)
            - normal[normal_indices, :].mean(axis=1)
        )
    return output


def concatenate_families(
    feature_means: dict[str, np.ndarray],
    variable_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    matrices = []
    metadata_rows = []
    for family_index, spec in enumerate(FAMILY_SPECS):
        key = spec["key"]
        matrix = feature_means[key]
        if matrix.shape[1] != len(variable_cols):
            raise ValueError(f"{key}: variable count changed")
        matrices.append(matrix)
        for variable_index, variable in enumerate(variable_cols):
            metadata_rows.append(
                {
                    "family_index": family_index,
                    "family_key": key,
                    "family_label": spec["label"],
                    "family_formula": spec["formula"],
                    "family_priority": spec["priority"],
                    "variable_index": variable_index,
                    "variable": variable,
                    "residual_variable": f"res_{variable}",
                }
            )
    combined = np.concatenate(matrices, axis=1)
    metadata = pd.DataFrame(metadata_rows)
    if combined.shape[1] != len(FAMILY_SPECS) * len(variable_cols):
        raise AssertionError("Candidate family concatenation is incomplete")
    return combined, metadata


def compute_selection_frequencies(
    metrics: pd.DataFrame,
    bootstrap_delta: np.ndarray,
    pooled_std: np.ndarray,
) -> np.ndarray:
    frequencies = np.zeros(len(metrics), dtype=float)
    for family_key in FAMILY_KEYS:
        family_positions = np.flatnonzero(
            metrics["family_key"].to_numpy() == family_key
        )
        eligible_positions = family_positions[
            metrics.iloc[family_positions]["primary_stable_global_fdr"]
            .astype(bool)
            .to_numpy()
        ]
        if len(eligible_positions) == 0:
            continue
        safe_scale = pooled_std[eligible_positions].copy()
        safe_scale[safe_scale <= 0] = np.nan
        scores = np.abs(
            bootstrap_delta[:, eligible_positions] / safe_scale[None, :]
        )
        selection_counts = np.zeros(len(eligible_positions), dtype=np.int64)
        n_take = min(TOP_K, len(eligible_positions))
        for row in scores:
            finite = np.nan_to_num(row, nan=-np.inf)
            order = np.argsort(finite)[::-1][:n_take]
            selection_counts[order] += 1
        frequencies[eligible_positions] = (
            selection_counts / float(BOOTSTRAP_REPLICATES)
        )
    return frequencies


def analyze_fault(
    fault_id: int,
    normal_features: dict[str, np.ndarray],
    fault_features: dict[str, np.ndarray],
    variable_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    normal, metadata = concatenate_families(normal_features, variable_cols)
    fault, fault_metadata = concatenate_families(fault_features, variable_cols)
    if not metadata.equals(fault_metadata):
        raise ValueError("Normal/fault candidate metadata do not align")
    if normal.shape != fault.shape or normal.shape != (250, 260):
        raise ValueError(
            f"Fault {fault_id}: expected paired matrix dimensions 250 x 260, "
            f"found {normal.shape} and {fault.shape}"
        )

    print("  Running independent run bootstrap...")
    bootstrap_delta = independent_run_bootstrap(
        normal, fault, RANDOM_SEED + fault_id * 1000 + 1
    )
    alpha = 1.0 - CONFIDENCE
    ci_low = np.quantile(bootstrap_delta, alpha / 2.0, axis=0)
    ci_high = np.quantile(bootstrap_delta, 1.0 - alpha / 2.0, axis=0)
    positive_probability = (bootstrap_delta > 0).mean(axis=0)
    negative_probability = (bootstrap_delta < 0).mean(axis=0)

    print("  Running global studentized permutation audit...")
    observed_abs_t, permutation_p, max_t_p = run_studentized_permutations(
        normal, fault, RANDOM_SEED + fault_id * 1000 + 2
    )
    bh_q_global = bh_adjust(permutation_p)

    normal_mean = normal.mean(axis=0)
    fault_mean = fault.mean(axis=0)
    mean_shift = fault_mean - normal_mean
    normal_var = normal.var(axis=0, ddof=1)
    fault_var = fault.var(axis=0, ddof=1)
    pooled_var = (
        (normal.shape[0] - 1) * normal_var
        + (fault.shape[0] - 1) * fault_var
    ) / (normal.shape[0] + fault.shape[0] - 2)
    pooled_std = np.sqrt(pooled_var)
    cohen_d = np.divide(
        mean_shift,
        pooled_std,
        out=np.full_like(mean_shift, np.nan),
        where=pooled_std > 0,
    )
    hedges_correction = 1.0 - 3.0 / (
        4.0 * (normal.shape[0] + fault.shape[0]) - 9.0
    )
    hedges_g = hedges_correction * cohen_d
    bootstrap_ci_excludes_zero = (ci_low > 0) | (ci_high < 0)
    primary_stable = bootstrap_ci_excludes_zero & (
        bh_q_global <= GLOBAL_FDR_ALPHA
    )
    fwer_stable = max_t_p <= FWER_ALPHA

    metrics = metadata.copy()
    metrics.insert(0, "fault_id", fault_id)
    metrics["normal_run_feature_mean"] = normal_mean
    metrics["fault_run_feature_mean"] = fault_mean
    metrics["unpaired_mean_shift"] = mean_shift
    metrics["pooled_run_feature_std"] = pooled_std
    metrics["cohen_d"] = cohen_d
    metrics["hedges_g"] = hedges_g
    metrics["abs_hedges_g"] = np.abs(hedges_g)
    metrics["bootstrap_ci_low"] = ci_low
    metrics["bootstrap_ci_high"] = ci_high
    metrics["bootstrap_positive_probability"] = positive_probability
    metrics["bootstrap_negative_probability"] = negative_probability
    metrics["bootstrap_direction_confidence"] = np.maximum(
        positive_probability, negative_probability
    )
    metrics["bootstrap_ci_excludes_zero"] = bootstrap_ci_excludes_zero
    metrics["observed_abs_welch_t"] = observed_abs_t
    metrics["permutation_p_two_sided"] = permutation_p
    metrics["bh_q_global_260"] = bh_q_global
    metrics["max_t_fwer_adjusted_p"] = max_t_p
    metrics["primary_stable_global_fdr"] = primary_stable
    metrics["strict_stable_max_t_fwer"] = fwer_stable
    metrics["dominant_direction"] = np.where(
        mean_shift >= 0, "positive", "negative"
    )
    metrics["orientation_sign"] = np.where(mean_shift >= 0, 1, -1)
    metrics["bootstrap_top5_selection_frequency"] = (
        compute_selection_frequencies(metrics, bootstrap_delta, pooled_std)
    )

    selected_tables = []
    summary_rows = []
    for family_key in FAMILY_KEYS:
        current = metrics[metrics["family_key"] == family_key].copy()
        stable = current[current["primary_stable_global_fdr"]].copy()
        selected = stable.sort_values(
            [
                "abs_hedges_g",
                "bootstrap_direction_confidence",
                "bootstrap_top5_selection_frequency",
                "variable",
            ],
            ascending=[False, False, False, True],
        ).head(TOP_K)
        selected = selected.reset_index(drop=True)
        selected.insert(1, "selection_rank", np.arange(1, len(selected) + 1))
        family_eligible = len(stable) >= TOP_K and len(selected) == TOP_K
        selected["family_eligible"] = family_eligible
        selected_tables.append(selected)

        summary_rows.append(
            {
                "fault_id": fault_id,
                "family_key": family_key,
                "family_label": FAMILY_LABELS[family_key],
                "family_priority": FAMILY_PRIORITY[family_key],
                "n_bootstrap_ci_stable": int(
                    current["bootstrap_ci_excludes_zero"].sum()
                ),
                "n_primary_stable_global_fdr": int(
                    current["primary_stable_global_fdr"].sum()
                ),
                "n_strict_stable_max_t_fwer": int(
                    current["strict_stable_max_t_fwer"].sum()
                ),
                "family_eligible_primary_gate": bool(family_eligible),
                "selected_count": int(len(selected)),
                "selected_variables": ";".join(
                    selected["variable"].astype(str)
                ),
                "selected_orientations": ";".join(
                    selected["orientation_sign"].astype(int).astype(str)
                ),
                "median_selected_abs_hedges_g": (
                    float(selected["abs_hedges_g"].median())
                    if len(selected)
                    else np.nan
                ),
                "minimum_selected_bootstrap_frequency": (
                    float(
                        selected["bootstrap_top5_selection_frequency"].min()
                    )
                    if len(selected)
                    else np.nan
                ),
            }
        )

    selected_all = pd.concat(selected_tables, ignore_index=True)
    family_summary = pd.DataFrame(summary_rows)
    eligible = family_summary[
        family_summary["family_eligible_primary_gate"]
    ].sort_values(
        ["median_selected_abs_hedges_g", "family_priority"],
        ascending=[False, True],
    )
    recommended_family = (
        str(eligible.iloc[0]["family_key"]) if len(eligible) else "none"
    )
    family_summary["recommended_by_fixed_development_rule"] = (
        family_summary["family_key"] == recommended_family
    )
    if len(selected_all):
        selected_all["recommended_by_fixed_development_rule"] = (
            selected_all["family_key"] == recommended_family
        )
    return metrics, selected_all, family_summary, recommended_family


def build_crossfault_summary(
    family_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for family_key in FAMILY_KEYS:
        current = family_summary[family_summary["family_key"] == family_key]
        selected_9 = set(
            selected.loc[
                (selected["fault_id"] == 9)
                & (selected["family_key"] == family_key),
                "variable",
            ].astype(str)
        )
        selected_15 = set(
            selected.loc[
                (selected["fault_id"] == 15)
                & (selected["family_key"] == family_key),
                "variable",
            ].astype(str)
        )
        lookup = current.set_index("fault_id")
        rows.append(
            {
                "family_key": family_key,
                "family_label": FAMILY_LABELS[family_key],
                "fault9_primary_stable_count": int(
                    lookup.loc[9, "n_primary_stable_global_fdr"]
                ),
                "fault15_primary_stable_count": int(
                    lookup.loc[15, "n_primary_stable_global_fdr"]
                ),
                "fault9_family_eligible": bool(
                    lookup.loc[9, "family_eligible_primary_gate"]
                ),
                "fault15_family_eligible": bool(
                    lookup.loc[15, "family_eligible_primary_gate"]
                ),
                "selected_top5_overlap_count": int(
                    len(selected_9 & selected_15)
                ),
                "selected_top5_overlap_variables": ";".join(
                    sorted(selected_9 & selected_15)
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_development_audit(family_summary: pd.DataFrame) -> list[Path]:
    x = np.arange(len(FAMILY_KEYS), dtype=float)
    width = 0.36
    labels = [FAMILY_LABELS[key].replace(" ", "\n") for key in FAMILY_KEYS]
    colors = {9: "#4C78A8", 15: "#D95F59"}

    figure, axes = plt.subplots(2, 1, figsize=(11.2, 8.3), sharex=True)
    for axis, count_column, title in [
        (
            axes[0],
            "n_primary_stable_global_fdr",
            "(a) Primary gate: bootstrap CI and global BH-FDR",
        ),
        (
            axes[1],
            "n_strict_stable_max_t_fwer",
            "(b) Strict diagnostic: max-|t| permutation FWER",
        ),
    ]:
        for position, fault_id in enumerate(TARGET_FAULTS):
            current = (
                family_summary[family_summary["fault_id"] == fault_id]
                .set_index("family_key")
                .reindex(FAMILY_KEYS)
            )
            offset = (position - 0.5) * width
            bars = axis.bar(
                x + offset,
                current[count_column].to_numpy(dtype=float),
                width,
                label=f"Fault {fault_id}",
                color=colors[fault_id],
            )
            axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
        axis.axhline(
            TOP_K,
            color="#B23A3A",
            linestyle="--",
            linewidth=1.2,
            label="Five features required" if axis is axes[0] else None,
        )
        axis.set_ylabel("Stable residual-feature count")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)

    axes[0].legend(loc="upper right", ncol=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    figure.suptitle(
        "Development-only multi-evidence residual audit for TEP Faults 9 and 15",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.012,
        "Only runs 1-250 were used. No final or companion data were loaded; "
        "these counts are feature-applicability evidence, not detection performance.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.05, 1, 0.95])

    png = FIGURE_DIR / "fig19a_multievidence_development_audit.png"
    pdf = FIGURE_DIR / "fig19a_multievidence_development_audit.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def main() -> None:
    print("=" * 88)
    print("Step 19: Development-only multi-evidence residual audit")
    print("=" * 88)
    print("DATA FIREWALL: final, normal-evaluation, and companion files are not loaded.")
    print(
        f"Candidate universe: {len(FAMILY_SPECS)} families x 52 residuals "
        f"= {len(FAMILY_SPECS) * 52} tests per fault."
    )

    step11 = load_step11_module()
    variable_cols = step11.load_variable_columns()
    if variable_cols != [
        *[f"xmeas_{index}" for index in range(1, 42)],
        *[f"xmv_{index}" for index in range(1, 12)],
    ]:
        raise ValueError("The expected 52-variable order has changed")

    normal_train = pd.read_pickle(
        require_file(NORMAL_TRAIN_PATH, "run Step 02")
    )
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_validation = pd.read_pickle(
        require_file(NORMAL_VALIDATION_PATH, "run Step 02")
    )
    normal_validation = validate_run_frame(
        normal_validation,
        variable_cols,
        "normal_validation_runs_1_250",
        expected_fault=None,
    )
    print(f"Computing normal-validation PCA residuals: {normal_validation.shape}")
    normal_residuals, residual_cols = step11.compute_residual_frame(
        normal_validation, variable_cols, model
    )
    normal_center = normal_residuals[residual_cols].mean(axis=0).to_numpy(
        dtype=float
    )
    normal_tensor = extract_base_tensor(
        normal_residuals, residual_cols, "normal_validation"
    )
    normal_features = build_feature_run_means(normal_tensor, normal_center)
    del normal_validation, normal_residuals, normal_tensor
    gc.collect()

    all_metrics = []
    all_selected = []
    all_family_summaries = []
    recommended = {}

    for fault_id in TARGET_FAULTS:
        print("\n" + "=" * 88)
        print(f"Fault {fault_id}: development runs 1--250 only")
        print("=" * 88)
        development = pd.read_pickle(
            require_file(
                DEVELOPMENT_PATHS[fault_id],
                "run Step 18 once to create the target-fault caches",
            )
        )
        development = validate_run_frame(
            development,
            variable_cols,
            f"fault{fault_id}_development_runs_1_250",
            expected_fault=fault_id,
        )
        development_residuals, current_cols = step11.compute_residual_frame(
            development, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: residual column order changed")
        development_tensor = extract_base_tensor(
            development_residuals,
            residual_cols,
            f"fault{fault_id}_development",
        )
        fault_features = build_feature_run_means(
            development_tensor, normal_center
        )
        del development, development_residuals, development_tensor
        gc.collect()

        metrics, selected, summary, recommended_family = analyze_fault(
            fault_id,
            normal_features,
            fault_features,
            variable_cols,
        )
        all_metrics.append(metrics)
        all_selected.append(selected)
        all_family_summaries.append(summary)
        recommended[str(fault_id)] = recommended_family

        print("\n  Development family summary:")
        print(
            summary[
                [
                    "family_label",
                    "n_primary_stable_global_fdr",
                    "n_strict_stable_max_t_fwer",
                    "family_eligible_primary_gate",
                    "selected_variables",
                ]
            ].to_string(index=False)
        )
        print(f"  Fixed-rule recommendation: {recommended_family}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    family_summary = pd.concat(all_family_summaries, ignore_index=True)
    crossfault = build_crossfault_summary(family_summary, selected)

    output_paths = {
        "all_features": TABLE_DIR / "19_multievidence_all_features.csv",
        "selected_features": TABLE_DIR / "19_multievidence_selected_features.csv",
        "family_summary": TABLE_DIR / "19_multievidence_family_summary.csv",
        "crossfault_summary": TABLE_DIR / "19_multievidence_crossfault_summary.csv",
    }
    metrics.to_csv(output_paths["all_features"], index=False, encoding="utf-8-sig")
    selected.to_csv(
        output_paths["selected_features"], index=False, encoding="utf-8-sig"
    )
    family_summary.to_csv(
        output_paths["family_summary"], index=False, encoding="utf-8-sig"
    )
    crossfault.to_csv(
        output_paths["crossfault_summary"], index=False, encoding="utf-8-sig"
    )
    figure_paths = plot_development_audit(family_summary)

    config = {
        "script": "scripts/19_multievidence_development_audit.py",
        "script_sha256_at_execution": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "candidate_protocol_frozen_in_source": True,
        "study_status": "exploratory method extension after Step-18 abstention",
        "development_only": True,
        "final_performance_evaluated": False,
        "companion_data_loaded": False,
        "normal_evaluation_loaded": False,
        "allowed_input_files": [
            str(NORMAL_TRAIN_PATH.relative_to(PROJECT_ROOT)),
            str(NORMAL_VALIDATION_PATH.relative_to(PROJECT_ROOT)),
            *[
                str(path.relative_to(PROJECT_ROOT))
                for path in DEVELOPMENT_PATHS.values()
            ],
            "results/tables/02_variable_list.csv",
            "scripts/11_unpaired_runlevel_screening.py",
        ],
        "forbidden_evaluation_files_not_loaded": [
            str(path.relative_to(PROJECT_ROOT))
            for path in FORBIDDEN_EVALUATION_PATHS
        ],
        "target_faults": TARGET_FAULTS,
        "development_runs": [1, 250],
        "screening_samples": [FAULT_START_SAMPLE, SCREEN_END_SAMPLE],
        "screening_window_length": SCREEN_WINDOW_LENGTH,
        "candidate_families": FAMILY_SPECS,
        "candidate_tests_per_fault": len(FAMILY_SPECS) * len(variable_cols),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "primary_feature_gate": (
            "two-sided 95% independent-run bootstrap CI excludes zero AND "
            "global BH q<=0.05 across all 260 candidates for the fault"
        ),
        "strict_diagnostic": "studentized max-|t| permutation FWER p<=0.05",
        "family_eligibility": "at least five primary-gate-stable residual features",
        "family_choice_rule": (
            "among eligible families, maximize median absolute Hedges g of the "
            "selected Top-5; break exact ties by the fixed family priority"
        ),
        "top_k": TOP_K,
        "random_seed": RANDOM_SEED,
        "recommended_family_by_fault": recommended,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
        "next_step_constraint": (
            "freeze the branch construction before any new method is applied "
            "to final or reserved validation data"
        ),
    }
    config_path = CONFIG_DIR / "faults9_15_multievidence_development_audit.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("Step 19 summary")
    print("=" * 88)
    print(
        family_summary[
            [
                "fault_id",
                "family_label",
                "n_primary_stable_global_fdr",
                "n_strict_stable_max_t_fwer",
                "family_eligible_primary_gate",
                "selected_variables",
                "recommended_by_fixed_development_rule",
            ]
        ].to_string(index=False)
    )
    print(f"\nFixed-rule recommendations: {recommended}")
    print("No final, normal-evaluation, or companion data were loaded.")
    print(f"Saved configuration: {config_path}")
    print("Step 19 finished successfully.")


if __name__ == "__main__":
    main()
