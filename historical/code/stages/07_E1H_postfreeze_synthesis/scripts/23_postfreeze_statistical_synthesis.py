"""Step 23: Post-freeze statistical synthesis and reporting diagnostics.

Scientific role
---------------
This script does not fit, select, tune, or re-evaluate any detector.  It reads
only already-frozen Step-15, Step-20, Step-21, and Step-22 result files.

The 17 Step-22 reserved faults form the primary cross-fault validation set.
Faults 3, 9, and 15 are method-development case studies and are kept in a
separate output table; they are never pooled into the reserved-fault macro
estimates.

The analyses are explicitly post-freeze secondary analyses:

* fault-cluster bootstrap confidence intervals for macro summaries;
* paired, fault-level comparisons of Proposed Top-5 with each comparator;
* Holm correction within each endpoint family;
* descriptive family-choice and Top-5 bootstrap-stability diagnostics;
* publication-oriented figures that do not change any performance result.

Two-stage use
-------------
Place in:
    TEP_Fault3_Project/scripts/23_postfreeze_statistical_synthesis.py

Register the immutable analysis specification first:
    python scripts/23_postfreeze_statistical_synthesis.py --register

Then run:
    python scripts/23_postfreeze_statistical_synthesis.py --run

Registration occurs after Step-22 performance generation and therefore must
be described as a post-freeze analysis specification, not as a prospective
performance preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in [TABLE_DIR, FIGURE_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


STEP15_RESULTS_PATH = TABLE_DIR / "15_fair_sequential_baseline_results.csv"
STEP20_COMPARISON_PATH = TABLE_DIR / "20_frozen_multievidence_comparison.csv"
STEP21_ALL_FEATURES_PATH = TABLE_DIR / "21_reserved_multifault_all_features.csv"
STEP21_FAMILY_SUMMARY_PATH = TABLE_DIR / "21_reserved_multifault_family_summary.csv"
STEP21_PROTOCOL_PATH = TABLE_DIR / "21_reserved_multifault_frozen_protocol.csv"
STEP22_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
STEP22_RUN_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_run_results.csv"
STEP22_CONFIG_PATH = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"

SPECIFICATION_PATH = CONFIG_DIR / "23_postfreeze_analysis_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "23_postfreeze_statistical_synthesis.json"


REQUIRED_INPUTS = {
    "step15_results": STEP15_RESULTS_PATH,
    "step20_comparison": STEP20_COMPARISON_PATH,
    "step21_all_features": STEP21_ALL_FEATURES_PATH,
    "step21_family_summary": STEP21_FAMILY_SUMMARY_PATH,
    "step21_protocol": STEP21_PROTOCOL_PATH,
    "step22_results": STEP22_RESULTS_PATH,
    "step22_run_results": STEP22_RUN_RESULTS_PATH,
    "step22_config": STEP22_CONFIG_PATH,
}

RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
DEVELOPMENT_CASE_FAULTS = [3, 9, 15]
FINAL_DATASET = "faulty_testing_final_full_960"
DEVELOPMENT_DATASET = "development_fit_runs_1_250"
NORMAL_DATASET = "normal_evaluation"

METHOD_ORDER = [
    "pca_q_cusum",
    "pca_t2_cusum",
    "pca_joint_cusum",
    "lda_residual_cusum",
    "chosen_family_all52_cusum",
    "chosen_family_top1_cusum",
    "frozen_multievidence_top5_cusum",
]
PROPOSED_KEY = "frozen_multievidence_top5_cusum"

METHOD_LABELS = {
    "pca_q_cusum": "Q-CUSUM",
    "pca_t2_cusum": "T²-CUSUM",
    "pca_joint_cusum": "Joint CUSUM",
    "lda_residual_cusum": "LDA residual",
    "chosen_family_all52_cusum": "Family All-52",
    "chosen_family_top1_cusum": "Family Top-1",
    "frozen_multievidence_top5_cusum": "Proposed Top-5",
}

METHOD_COMPLEXITY = {
    "pca_q_cusum": "global PCA statistic",
    "pca_t2_cusum": "global PCA statistic",
    "pca_joint_cusum": "global PCA statistic",
    "lda_residual_cusum": "dense supervised, 52 residuals",
    "chosen_family_all52_cusum": "chosen family, 52 residuals",
    "chosen_family_top1_cusum": "chosen family, 1 residual",
    "frozen_multievidence_top5_cusum": "chosen family, 5 residuals",
}

METHOD_COLORS = {
    "pca_q_cusum": "#4C78A8",
    "pca_t2_cusum": "#72B7B2",
    "pca_joint_cusum": "#54A24B",
    "lda_residual_cusum": "#B279A2",
    "chosen_family_all52_cusum": "#9D755D",
    "chosen_family_top1_cusum": "#F2CF5B",
    "frozen_multievidence_top5_cusum": "#E45756",
}

FAMILY_COLORS = {
    "signed_level": "#4C78A8",
    "absolute_deviation": "#E45756",
    "squared_deviation": "#B279A2",
    "absolute_increment": "#72B7B2",
    "lag1_product": "#F2CF5B",
}

FAMILY_LABELS = {
    "signed_level": "Signed level",
    "absolute_deviation": "Absolute deviation",
    "squared_deviation": "Squared deviation",
    "absolute_increment": "Absolute increment",
    "lag1_product": "Lag-1 product",
}

BOOTSTRAP_REPLICATES = 20000
RANDOM_SEED = 20260715
ALPHA = 0.05
TOLERANCE = 1e-12


def require_file(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Required to {purpose}.")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    return {
        key: sha256_file(require_file(path, f"hash frozen input {key}"))
        for key, path in REQUIRED_INPUTS.items()
    }


def specification_payload(script_hash: str, hashes: dict[str, str]) -> dict:
    return {
        "registration_name": "Step 23 post-freeze statistical synthesis",
        "registration_status": (
            "post-freeze secondary analysis specification; Step-22 results already exist"
        ),
        "script": "scripts/23_postfreeze_statistical_synthesis.py",
        "script_sha256": script_hash,
        "input_sha256": hashes,
        "process_data_loaded": False,
        "detector_fitting_allowed": False,
        "variable_selection_allowed": False,
        "family_selection_allowed": False,
        "cusum_k_tuning_allowed": False,
        "control_limit_tuning_allowed": False,
        "performance_rows_changed_or_recomputed": False,
        "primary_validation_population": {
            "role": "reserved-fault frozen validation",
            "faults": RESERVED_FAULTS,
            "source": "Step 22 final runs 251--500",
            "unit_for_crossfault_inference": "TEP fault",
            "n_faults": len(RESERVED_FAULTS),
        },
        "method_development_cases": {
            "faults": DEVELOPMENT_CASE_FAULTS,
            "pooled_with_primary_validation": False,
            "fault3_source": "Step 15 frozen final comparison",
            "faults9_15_source": "Step 20 frozen final comparison",
        },
        "macro_endpoints": [
            "mean normal run-level FAR",
            "mean and median run-level detection",
            "mean and median post-fault point detection",
            "median alarm delay",
        ],
        "uncertainty": {
            "method": "nonparametric fault-cluster percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "confidence_level": 0.95,
        },
        "paired_comparisons": {
            "reference_method": PROPOSED_KEY,
            "comparators": [key for key in METHOD_ORDER if key != PROPOSED_KEY],
            "unit": "fault",
            "endpoints": [
                "normal FAR (positive utility difference means lower proposed FAR)",
                "run detection (positive means higher proposed detection)",
                "point detection (positive means higher proposed detection)",
                "alarm delay (positive means faster proposed detection)",
            ],
            "test": "two-sided paired Wilcoxon signed-rank with Pratt zeros",
            "multiplicity": "Holm adjustment separately within each endpoint family",
            "equivalence_claim_allowed": False,
        },
        "selection_stability": {
            "source": "Step-21 development-only bootstrap frequencies and frozen family scores",
            "inferential_status": "descriptive secondary diagnostic",
            "family_margin": (
                "chosen-family median Top-5 |Hedges g| minus best eligible runner-up"
            ),
            "top5_stability": "marginal bootstrap Top-5 selection frequencies",
            "diagnostic_guides_only": {
                "relative_family_margin": 0.05,
                "minimum_top5_frequency": 0.80,
            },
        },
        "reporting_guardrails": [
            "Do not pool Faults 3, 9, and 15 into the 17-fault primary macro estimates.",
            "Do not claim universal superiority or statistical equivalence.",
            "Report every Step-22 method and every reserved fault.",
            "Describe the numerical repair and retain both Step-22 registrations.",
            "Report Fault 18 as the principal sustained point-detection limitation.",
        ],
    }


def register_specification() -> None:
    script_hash = sha256_file(Path(__file__))
    hashes = input_hashes()
    payload = specification_payload(script_hash, hashes)
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()

    if SPECIFICATION_PATH.exists():
        existing = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
        checks = {
            "script hash": existing.get("script_sha256") == script_hash,
            "input hashes": existing.get("input_sha256") == hashes,
            "reserved faults": existing.get("primary_validation_population", {}).get(
                "faults"
            )
            == RESERVED_FAULTS,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Existing Step-23 specification differs from current files: "
                f"{failed}. Preserve it and investigate rather than overwriting it."
            )
        print("Step 23 post-freeze analysis specification already exists and matches.")
        print(f"Registered script SHA-256: {script_hash}")
        print(f"Specification: {SPECIFICATION_PATH}")
        return

    SPECIFICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 23 post-freeze analysis specification registered.")
    print("This is a secondary analysis registry, not a prospective performance preregistration.")
    print(f"Registered script SHA-256: {script_hash}")
    print(f"Specification: {SPECIFICATION_PATH}")


def verify_specification() -> dict:
    require_file(
        SPECIFICATION_PATH,
        "run this same script once with --register before --run",
    )
    specification = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
    current_script_hash = sha256_file(Path(__file__))
    current_input_hashes = input_hashes()
    checks = {
        "script hash": specification.get("script_sha256") == current_script_hash,
        "input hashes": specification.get("input_sha256") == current_input_hashes,
        "reserved faults": specification.get("primary_validation_population", {}).get(
            "faults"
        )
        == RESERVED_FAULTS,
        "development cases not pooled": specification.get(
            "method_development_cases", {}
        ).get("pooled_with_primary_validation")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-23 specification verification failed: {failed}")
    return specification


def load_and_validate_inputs() -> dict[str, object]:
    frames = {
        "step15": pd.read_csv(STEP15_RESULTS_PATH),
        "step20": pd.read_csv(STEP20_COMPARISON_PATH),
        "all_features": pd.read_csv(STEP21_ALL_FEATURES_PATH),
        "family_summary": pd.read_csv(STEP21_FAMILY_SUMMARY_PATH),
        "protocol": pd.read_csv(STEP21_PROTOCOL_PATH),
        "results": pd.read_csv(STEP22_RESULTS_PATH),
        "run_results": pd.read_csv(STEP22_RUN_RESULTS_PATH),
    }
    config = json.loads(STEP22_CONFIG_PATH.read_text(encoding="utf-8"))

    results = frames["results"]
    run_results = frames["run_results"]
    family_summary = frames["family_summary"]
    protocol = frames["protocol"]
    all_features = frames["all_features"]

    expected_final_rows = len(RESERVED_FAULTS) * len(METHOD_ORDER)
    final = results[results["dataset"] == FINAL_DATASET]
    development = results[results["dataset"] == DEVELOPMENT_DATASET]
    if len(final) != expected_final_rows or len(development) != expected_final_rows:
        raise ValueError("Step-22 result row counts changed")
    if sorted(final["fault_id"].unique().tolist()) != RESERVED_FAULTS:
        raise ValueError("Step-22 reserved fault set changed")
    if set(final["method_key"]) != set(METHOD_ORDER):
        raise ValueError("Step-22 method set changed")
    if final.duplicated(["fault_id", "method_key"]).any():
        raise ValueError("Duplicate Step-22 final method rows")
    if development.duplicated(["fault_id", "method_key"]).any():
        raise ValueError("Duplicate Step-22 development method rows")

    expected_run_rows = len(RESERVED_FAULTS) * len(METHOD_ORDER) * 250 * 3
    if len(run_results) != expected_run_rows:
        raise ValueError("Step-22 run-result row count changed")
    if run_results.duplicated(
        ["fault_id", "method_key", "dataset", "simulationRun"]
    ).any():
        raise ValueError("Duplicate Step-22 run-result identifiers")
    if set(run_results["dataset"]) != {
        DEVELOPMENT_DATASET,
        FINAL_DATASET,
        NORMAL_DATASET,
    }:
        raise ValueError("Unexpected Step-22 run-result dataset")

    if sorted(protocol["fault_id"].tolist()) != RESERVED_FAULTS or len(protocol) != 17:
        raise ValueError("Step-21 frozen protocol changed")
    if not protocol["branch_status"].eq("estimable_frozen_after_step21").all():
        raise ValueError("A Step-21 reserved branch is no longer estimable")
    if len(family_summary) != len(RESERVED_FAULTS) * 5:
        raise ValueError("Step-21 family summary shape changed")
    if len(all_features) != len(RESERVED_FAULTS) * 5 * 52:
        raise ValueError("Step-21 all-feature table shape changed")

    config_checks = {
        "all states frozen": config.get("all_states_frozen_before_any_final_subset")
        is True,
        "final not used for tuning": config.get("final_used_for_tuning") is False,
        "normal evaluation not used for tuning": config.get(
            "normal_evaluation_used_for_tuning"
        )
        is False,
        "companion not loaded": config.get("companion_loaded") is False,
        "repair only": config.get("numerical_repair_only") is True,
        "scientific protocol unchanged": config.get("scientific_protocol_changed")
        is False,
        "reserved faults": config.get("reserved_faults") == RESERVED_FAULTS,
    }
    failed = [name for name, passed in config_checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-22 configuration checks failed: {failed}")

    return {**frames, "config": config}


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def bootstrap_statistic(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
    statistic: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(RESERVED_FAULTS):
        raise ValueError("Fault-cluster bootstrap requires one value per reserved fault")
    observed = float(statistic(values[None, :], 1)[0])
    bootstrap_values = statistic(values[bootstrap_indices], 1)
    low, high = percentile_interval(bootstrap_values)
    return observed, low, high


def mean_stat(values: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(values, axis=axis)


def median_stat(values: np.ndarray, axis: int) -> np.ndarray:
    return np.median(values, axis=axis)


def build_primary_summary(
    final: pd.DataFrame,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    endpoint_definitions = [
        (
            "normal_run_far",
            "Macro mean normal run-level FAR",
            "normal_run_level_far",
            mean_stat,
            100.0,
            "percent",
            "lower is better",
        ),
        (
            "run_detection_mean",
            "Macro mean run-level detection",
            "fault_run_level_detection_rate",
            mean_stat,
            100.0,
            "percent",
            "higher is better",
        ),
        (
            "run_detection_median",
            "Macro median run-level detection",
            "fault_run_level_detection_rate",
            median_stat,
            100.0,
            "percent",
            "higher is better",
        ),
        (
            "point_detection_mean",
            "Macro mean post-fault point detection",
            "fault_post_fault_point_detection_rate",
            mean_stat,
            100.0,
            "percent",
            "higher is better",
        ),
        (
            "point_detection_median",
            "Macro median post-fault point detection",
            "fault_post_fault_point_detection_rate",
            median_stat,
            100.0,
            "percent",
            "higher is better",
        ),
        (
            "alarm_delay_median",
            "Macro median alarm delay",
            "median_alarm_delay_min",
            median_stat,
            1.0,
            "minutes",
            "lower is better",
        ),
    ]

    rows = []
    for method_order, method_key in enumerate(METHOD_ORDER):
        group = final[final["method_key"] == method_key].set_index("fault_id").loc[
            RESERVED_FAULTS
        ]
        for (
            endpoint_key,
            endpoint_label,
            column,
            statistic,
            scale,
            unit,
            direction,
        ) in endpoint_definitions:
            estimate, low, high = bootstrap_statistic(
                group[column].to_numpy(dtype=float),
                bootstrap_indices,
                statistic,
            )
            rows.append(
                {
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "method_order": method_order,
                    "complexity": METHOD_COMPLEXITY[method_key],
                    "endpoint_key": endpoint_key,
                    "endpoint": endpoint_label,
                    "direction": direction,
                    "estimate": scale * estimate,
                    "ci_95_low": scale * low,
                    "ci_95_high": scale * high,
                    "unit": unit,
                    "n_faults": len(RESERVED_FAULTS),
                    "inference_unit": "fault",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    summary = pd.DataFrame(rows)

    rank_rules = {
        "normal_run_far": True,
        "run_detection_mean": False,
        "run_detection_median": False,
        "point_detection_mean": False,
        "point_detection_median": False,
        "alarm_delay_median": True,
    }
    summary["endpoint_rank"] = np.nan
    for endpoint, ascending in rank_rules.items():
        mask = summary["endpoint_key"] == endpoint
        summary.loc[mask, "endpoint_rank"] = summary.loc[mask, "estimate"].rank(
            method="min", ascending=ascending
        )
    summary["endpoint_rank"] = summary["endpoint_rank"].astype(int)
    return summary.sort_values(["endpoint_key", "method_order"]).reset_index(drop=True)


def paired_wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if np.all(np.abs(values) <= TOLERANCE):
        return 0.0, 1.0
    result = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="pratt",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    m = len(values)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, p_value in enumerate(sorted_values):
        candidate = min(1.0, (m - rank) * p_value)
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def build_pairwise_statistics(
    final: pd.DataFrame,
    bootstrap_indices: np.ndarray,
) -> pd.DataFrame:
    endpoint_definitions = [
        (
            "normal_run_far",
            "Normal run-level FAR",
            "normal_run_level_far",
            -1.0,
            100.0,
            "percentage points",
            "baseline minus proposed; positive favors proposed",
        ),
        (
            "run_detection",
            "Run-level detection",
            "fault_run_level_detection_rate",
            1.0,
            100.0,
            "percentage points",
            "proposed minus baseline; positive favors proposed",
        ),
        (
            "point_detection",
            "Post-fault point detection",
            "fault_post_fault_point_detection_rate",
            1.0,
            100.0,
            "percentage points",
            "proposed minus baseline; positive favors proposed",
        ),
        (
            "alarm_delay",
            "Median alarm delay",
            "median_alarm_delay_min",
            -1.0,
            1.0,
            "minutes",
            "baseline minus proposed; positive favors proposed",
        ),
    ]

    pivots = {
        column: final.pivot(index="fault_id", columns="method_key", values=column).loc[
            RESERVED_FAULTS
        ]
        for _, _, column, _, _, _, _ in endpoint_definitions
    }
    rows = []
    for endpoint_order, (
        endpoint_key,
        endpoint_label,
        column,
        direction_sign,
        scale,
        unit,
        effect_definition,
    ) in enumerate(endpoint_definitions):
        pivot = pivots[column]
        for baseline_order, baseline_key in enumerate(
            [key for key in METHOD_ORDER if key != PROPOSED_KEY]
        ):
            utility_difference = direction_sign * (
                pivot[PROPOSED_KEY].to_numpy(dtype=float)
                - pivot[baseline_key].to_numpy(dtype=float)
            )
            observed_mean, mean_low, mean_high = bootstrap_statistic(
                utility_difference,
                bootstrap_indices,
                mean_stat,
            )
            observed_median, median_low, median_high = bootstrap_statistic(
                utility_difference,
                bootstrap_indices,
                median_stat,
            )
            statistic, raw_p = paired_wilcoxon(utility_difference)
            rows.append(
                {
                    "endpoint_key": endpoint_key,
                    "endpoint": endpoint_label,
                    "endpoint_order": endpoint_order,
                    "baseline_key": baseline_key,
                    "baseline_short": METHOD_LABELS[baseline_key],
                    "baseline_order": baseline_order,
                    "reference_key": PROPOSED_KEY,
                    "reference_short": METHOD_LABELS[PROPOSED_KEY],
                    "effect_definition": effect_definition,
                    "mean_utility_difference": scale * observed_mean,
                    "mean_utility_difference_ci_95_low": scale * mean_low,
                    "mean_utility_difference_ci_95_high": scale * mean_high,
                    "median_utility_difference": scale * observed_median,
                    "median_utility_difference_ci_95_low": scale * median_low,
                    "median_utility_difference_ci_95_high": scale * median_high,
                    "unit": unit,
                    "faults_favoring_proposed": int(
                        np.sum(utility_difference > TOLERANCE)
                    ),
                    "faults_tied": int(np.sum(np.abs(utility_difference) <= TOLERANCE)),
                    "faults_favoring_baseline": int(
                        np.sum(utility_difference < -TOLERANCE)
                    ),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_raw_p": raw_p,
                    "n_faults": len(RESERVED_FAULTS),
                    "inference_unit": "fault",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    pairwise = pd.DataFrame(rows)
    pairwise["wilcoxon_holm_p_within_endpoint"] = np.nan
    for endpoint, indices in pairwise.groupby("endpoint_key").groups.items():
        pairwise.loc[indices, "wilcoxon_holm_p_within_endpoint"] = holm_adjust(
            pairwise.loc[indices, "wilcoxon_raw_p"]
        )
    pairwise["holm_reject_0_05"] = (
        pairwise["wilcoxon_holm_p_within_endpoint"] < ALPHA
    )
    return pairwise.sort_values(["endpoint_order", "baseline_order"]).reset_index(
        drop=True
    )


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    )


def build_run_detection_discordance(run_results: pd.DataFrame) -> pd.DataFrame:
    final_runs = run_results[run_results["dataset"] == FINAL_DATASET].copy()
    final_runs["detected_boolean"] = to_bool(final_runs["detected"])
    pivot = final_runs.pivot(
        index=["fault_id", "simulationRun"],
        columns="method_key",
        values="detected_boolean",
    )
    rows = []
    for baseline_key in [key for key in METHOD_ORDER if key != PROPOSED_KEY]:
        proposed_only = int((pivot[PROPOSED_KEY] & ~pivot[baseline_key]).sum())
        baseline_only = int((~pivot[PROPOSED_KEY] & pivot[baseline_key]).sum())
        both = int((pivot[PROPOSED_KEY] & pivot[baseline_key]).sum())
        neither = int((~pivot[PROPOSED_KEY] & ~pivot[baseline_key]).sum())
        rows.append(
            {
                "reference_key": PROPOSED_KEY,
                "reference_short": METHOD_LABELS[PROPOSED_KEY],
                "baseline_key": baseline_key,
                "baseline_short": METHOD_LABELS[baseline_key],
                "both_detected": both,
                "proposed_only_detected": proposed_only,
                "baseline_only_detected": baseline_only,
                "neither_detected": neither,
                "paired_runs": len(pivot),
                "status": (
                    "descriptive run-level discordance; fault-level analysis is primary"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_selection_stability(
    protocol: pd.DataFrame,
    family_summary: pd.DataFrame,
    all_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fault_id in RESERVED_FAULTS:
        frozen = protocol[protocol["fault_id"] == fault_id].iloc[0]
        fault_families = family_summary[family_summary["fault_id"] == fault_id]
        chosen = fault_families[
            fault_families["recommended_by_fixed_development_rule"]
        ]
        if len(chosen) != 1:
            raise ValueError(f"Fault {fault_id}: expected one chosen family")
        chosen = chosen.iloc[0]
        eligible_runners = fault_families[
            fault_families["family_eligible_primary_gate"]
            & ~fault_families["recommended_by_fixed_development_rule"]
        ].sort_values(
            ["median_selected_abs_hedges_g", "family_priority"],
            ascending=[False, True],
        )
        if eligible_runners.empty:
            runner_key = None
            runner_label = None
            runner_score = math.nan
            absolute_margin = math.nan
            relative_margin = math.nan
        else:
            runner = eligible_runners.iloc[0]
            runner_key = runner["family_key"]
            runner_label = runner["family_label"]
            runner_score = float(runner["median_selected_abs_hedges_g"])
            absolute_margin = float(chosen["median_selected_abs_hedges_g"]) - runner_score
            denominator = abs(float(chosen["median_selected_abs_hedges_g"]))
            relative_margin = absolute_margin / denominator if denominator else math.nan

        selected_variables = str(frozen["selected_variables"]).split(";")
        selected_rows = all_features[
            (all_features["fault_id"] == fault_id)
            & (all_features["family_key"] == frozen["family_key"])
            & (all_features["variable"].isin(selected_variables))
        ].copy()
        if len(selected_rows) != 5:
            raise ValueError(f"Fault {fault_id}: selected Top-5 reconstruction failed")
        frequency_map = selected_rows.set_index("variable")[
            "bootstrap_top5_selection_frequency"
        ]
        frequencies = np.array(
            [float(frequency_map[variable]) for variable in selected_variables]
        )
        family_margin_sensitive = bool(
            np.isfinite(relative_margin) and relative_margin < 0.05
        )
        top5_frequency_sensitive = bool(float(frequencies.min()) < 0.80)
        if family_margin_sensitive and top5_frequency_sensitive:
            diagnostic_label = "family-and-variable-sensitive"
        elif family_margin_sensitive:
            diagnostic_label = "family-choice-sensitive"
        elif top5_frequency_sensitive:
            diagnostic_label = "variable-selection-sensitive"
        else:
            diagnostic_label = "descriptively-stable"
        rows.append(
            {
                "fault_id": fault_id,
                "chosen_family_key": frozen["family_key"],
                "chosen_family_label": frozen["family_label"],
                "chosen_family_score_median_top5_abs_hedges_g": float(
                    chosen["median_selected_abs_hedges_g"]
                ),
                "runner_up_family_key": runner_key,
                "runner_up_family_label": runner_label,
                "runner_up_family_score": runner_score,
                "absolute_family_score_margin": absolute_margin,
                "relative_family_score_margin": relative_margin,
                "selected_variables": ";".join(selected_variables),
                "rank1_bootstrap_top5_frequency": frequencies[0],
                "minimum_selected_bootstrap_top5_frequency": frequencies.min(),
                "mean_selected_bootstrap_top5_frequency": frequencies.mean(),
                "median_selected_bootstrap_top5_frequency": np.median(frequencies),
                "selected_variables_frequency_ge_0_80": int(
                    np.sum(frequencies >= 0.80)
                ),
                "selected_variables_frequency_ge_0_60": int(
                    np.sum(frequencies >= 0.60)
                ),
                "family_margin_below_descriptive_0_05_guide": family_margin_sensitive,
                "minimum_frequency_below_descriptive_0_80_guide": (
                    top5_frequency_sensitive
                ),
                "descriptive_stability_diagnostic": diagnostic_label,
                "inferential_status": "secondary descriptive diagnostic only",
            }
        )
    return pd.DataFrame(rows)


def build_development_case_studies(
    step15: pd.DataFrame,
    step20: pd.DataFrame,
) -> pd.DataFrame:
    step15_key_map = {
        "pca_q_cusum": "pca_q_cusum",
        "pca_t2_cusum": "pca_t2_cusum",
        "pca_joint_cusum": "pca_joint_cusum",
        "lda_residual_cusum": "lda_residual_cusum",
        "all52_signed_cusum": "chosen_family_all52_cusum",
        "top1_signed_cusum": "chosen_family_top1_cusum",
        "proposed_top5_cusum": PROPOSED_KEY,
    }
    fault3 = step15[step15["method_key"].isin(step15_key_map)].copy()
    if len(fault3) != 7:
        raise ValueError("Expected seven sequential Fault-3 case-study methods")
    fault3["canonical_method_key"] = fault3["method_key"].map(step15_key_map)
    fault3_rows = pd.DataFrame(
        {
            "fault_id": 3,
            "case_role": "original signed-level method-development target",
            "proposed_evidence_family": "signed_level",
            "method_key": fault3["canonical_method_key"],
            "method_short": fault3["canonical_method_key"].map(METHOD_LABELS),
            "cusum_k": fault3["cusum_k"],
            "normal_run_level_far": fault3["normal_run_level_far"],
            "fault_run_level_detection_rate": fault3[
                "fault3_run_level_detection_rate"
            ],
            "fault_post_fault_point_detection_rate": fault3[
                "fault3_post_fault_point_detection_rate"
            ],
            "median_alarm_delay_min": fault3["median_alarm_delay_min"],
            "source_step": 15,
            "pooled_with_reserved_primary": False,
        }
    )

    fault9_15 = step20[
        (step20["dataset"] == FINAL_DATASET)
        & (step20["fault_id"].isin([9, 15]))
    ].copy()
    if len(fault9_15) != 8:
        raise ValueError("Expected eight frozen Fault-9/15 comparison rows")
    step20_key_map = {
        "pca_q_cusum": "pca_q_cusum",
        "all52_signed_cusum": "chosen_family_all52_cusum",
        "lda_residual_cusum": "lda_residual_cusum",
        "frozen_multievidence_top5_cusum": PROPOSED_KEY,
    }
    if not set(fault9_15["method_key"]).issubset(step20_key_map):
        raise ValueError("Unexpected Step-20 case-study method")
    fault9_15["canonical_method_key"] = fault9_15["method_key"].map(step20_key_map)
    evidence_map = {9: "lag1_product", 15: "squared_deviation"}
    role_map = {
        9: "multi-evidence family-development target",
        15: "multi-evidence family-development target",
    }
    other_rows = pd.DataFrame(
        {
            "fault_id": fault9_15["fault_id"],
            "case_role": fault9_15["fault_id"].map(role_map),
            "proposed_evidence_family": fault9_15["fault_id"].map(evidence_map),
            "method_key": fault9_15["canonical_method_key"],
            "method_short": fault9_15["canonical_method_key"].map(METHOD_LABELS),
            "cusum_k": fault9_15["cusum_k"],
            "normal_run_level_far": fault9_15["normal_run_level_far"],
            "fault_run_level_detection_rate": fault9_15[
                "fault_run_level_detection_rate"
            ],
            "fault_post_fault_point_detection_rate": fault9_15[
                "fault_post_fault_point_detection_rate"
            ],
            "median_alarm_delay_min": fault9_15["median_alarm_delay_min"],
            "source_step": 20,
            "pooled_with_reserved_primary": False,
        }
    )
    cases = pd.concat([fault3_rows, other_rows], ignore_index=True)
    cases["normal_run_level_far_percent"] = 100 * cases["normal_run_level_far"]
    cases["fault_run_level_detection_percent"] = (
        100 * cases["fault_run_level_detection_rate"]
    )
    cases["fault_post_fault_point_detection_percent"] = (
        100 * cases["fault_post_fault_point_detection_rate"]
    )
    cases["method_order"] = cases["method_key"].map(
        {key: index for index, key in enumerate(METHOD_ORDER)}
    )
    return cases.sort_values(["fault_id", "method_order"]).reset_index(drop=True)


def build_guardrails(final: pd.DataFrame) -> pd.DataFrame:
    proposed = final[final["method_key"] == PROPOSED_KEY].set_index("fault_id")
    fault18 = proposed.loc[18]
    return pd.DataFrame(
        [
            {
                "guardrail_id": "primary_population",
                "manuscript_instruction": (
                    "Use only the 17 Step-22 reserved faults for primary cross-fault macro inference."
                ),
                "evidence": "Faults 3, 9, and 15 were used during method development.",
            },
            {
                "guardrail_id": "claim_scope",
                "manuscript_instruction": (
                    "Claim competitive sparse and interpretable trade-offs, not universal superiority."
                ),
                "evidence": (
                    "Q-CUSUM and same-family controls lead different endpoints; no composite rank is prespecified."
                ),
            },
            {
                "guardrail_id": "generalization_scope",
                "manuscript_instruction": (
                    "Describe broad cross-fault applicability within TEP, not external-process universality."
                ),
                "evidence": "All primary validation faults come from the TEP benchmark.",
            },
            {
                "guardrail_id": "fault18_limitation",
                "manuscript_instruction": (
                    "Report Fault 18 as the main sustained point-detection limitation."
                ),
                "evidence": (
                    f"Proposed run detection={100*fault18['fault_run_level_detection_rate']:.1f}%, "
                    f"point detection={100*fault18['fault_post_fault_point_detection_rate']:.1f}%, "
                    f"median delay={fault18['median_alarm_delay_min']:.1f} min."
                ),
            },
            {
                "guardrail_id": "repair_transparency",
                "manuscript_instruction": (
                    "Retain the original Step-22 failure record and the Step-22-R numerical amendment."
                ),
                "evidence": (
                    "The repair disabled an unused precision pseudo-inverse without changing covariance or protocol."
                ),
            },
        ]
    )


def summary_wide(primary_summary: pd.DataFrame) -> pd.DataFrame:
    estimates = primary_summary.pivot(
        index=["method_key", "method_short", "method_order", "complexity"],
        columns="endpoint_key",
        values="estimate",
    ).reset_index()
    return estimates.sort_values("method_order").reset_index(drop=True)


def plot_performance_tradeoffs(primary_summary: pd.DataFrame) -> list[Path]:
    wide = summary_wide(primary_summary).set_index("method_key")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)

    annotation_offsets_left = {
        "pca_q_cusum": (8, 8),
        "pca_t2_cusum": (8, -15),
        "pca_joint_cusum": (8, 8),
        "lda_residual_cusum": (8, -15),
        "chosen_family_all52_cusum": (8, 8),
        "chosen_family_top1_cusum": (8, -15),
        PROPOSED_KEY: (8, 8),
    }
    annotation_offsets_right = {
        "pca_q_cusum": (8, 8),
        "pca_t2_cusum": (8, -15),
        "pca_joint_cusum": (8, -15),
        "lda_residual_cusum": (8, 8),
        "chosen_family_all52_cusum": (8, 8),
        "chosen_family_top1_cusum": (8, -15),
        PROPOSED_KEY: (8, 8),
    }

    panels = [
        (
            axes[0],
            "normal_run_far",
            "point_detection_mean",
            "(a) Point detection versus normal FAR",
            "Macro mean normal run-level FAR (%)",
            "Macro mean post-fault point detection (%)",
            annotation_offsets_left,
        ),
        (
            axes[1],
            "alarm_delay_median",
            "point_detection_median",
            "(b) Point detection versus alarm delay",
            "Macro median alarm delay (min)",
            "Macro median post-fault point detection (%)",
            annotation_offsets_right,
        ),
    ]
    for ax, x_key, y_key, title, xlabel, ylabel, offsets in panels:
        for method_key in METHOD_ORDER:
            row = wide.loc[method_key]
            marker = "*" if method_key == PROPOSED_KEY else "o"
            size = 260 if method_key == PROPOSED_KEY else 95
            edge = "black" if method_key == PROPOSED_KEY else "white"
            ax.scatter(
                row[x_key],
                row[y_key],
                s=size,
                marker=marker,
                color=METHOD_COLORS[method_key],
                edgecolor=edge,
                linewidth=1.1,
                zorder=4,
            )
            ax.annotate(
                METHOD_LABELS[method_key],
                (row[x_key], row[y_key]),
                xytext=offsets[method_key],
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.8)
    fig.suptitle(
        "Frozen performance trade-offs on 17 reserved TEP faults",
        fontsize=15,
        fontweight="semibold",
    )
    return save_figure(fig, "fig23a_reserved_performance_tradeoffs")


def plot_proposed_by_fault(final: pd.DataFrame, protocol: pd.DataFrame) -> list[Path]:
    proposed = final[final["method_key"] == PROPOSED_KEY].set_index("fault_id").loc[
        RESERVED_FAULTS
    ]
    family_by_fault = protocol.set_index("fault_id")["family_key"].loc[RESERVED_FAULTS]
    colors = [FAMILY_COLORS[key] for key in family_by_fault]
    x = np.arange(len(RESERVED_FAULTS))
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)

    point = 100 * proposed["fault_post_fault_point_detection_rate"].to_numpy()
    delay = proposed["median_alarm_delay_min"].to_numpy()
    far = 100 * proposed["normal_run_level_far"].to_numpy()
    panels = [
        (axes[0], point, "Post-fault point detection (%)", (55, 101)),
        (axes[1], delay, "Median alarm delay (min)", None),
        (axes[2], far, "Normal run-level FAR (%)", (0, max(10.0, far.max() + 1))),
    ]
    for ax, values, ylabel, ylim in panels:
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].axhline(np.median(point), color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(np.median(delay), color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(np.mean(far), color="black", linestyle="--", linewidth=1.0)
    axes[0].annotate(
        "Fault 18: sustained point-detection limitation",
        xy=(RESERVED_FAULTS.index(18), point[RESERVED_FAULTS.index(18)]),
        xytext=(-150, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=9,
    )
    axes[2].set_xticks(x, [str(value) for value in RESERVED_FAULTS])
    axes[2].set_xlabel("Reserved TEP fault")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=FAMILY_COLORS[key],
            markeredgecolor="none",
            markersize=10,
            label=FAMILY_LABELS[key],
        )
        for key in ["signed_level", "absolute_deviation", "lag1_product"]
    ]
    axes[0].legend(handles=legend_handles, ncol=3, loc="lower left", frameon=True)
    fig.suptitle(
        "Frozen Proposed Top-5 performance by reserved fault and evidence family",
        fontsize=15,
        fontweight="semibold",
    )
    return save_figure(fig, "fig23b_reserved_proposed_by_fault")


def plot_selection_stability(stability: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    label_offsets = {
        4: (-12, 7),
        6: (6, 7),
        11: (5, 7),
        12: (6, -14),
        14: (5, 7),
        16: (6, 7),
    }
    for family_key, group in stability.groupby("chosen_family_key"):
        ax.scatter(
            100 * group["relative_family_score_margin"],
            100 * group["minimum_selected_bootstrap_top5_frequency"],
            s=90,
            color=FAMILY_COLORS[family_key],
            edgecolor="white",
            linewidth=0.8,
            label=FAMILY_LABELS[family_key],
            zorder=3,
        )
        for _, row in group.iterrows():
            fault_id = int(row["fault_id"])
            ax.annotate(
                str(fault_id),
                (
                    100 * row["relative_family_score_margin"],
                    100 * row["minimum_selected_bootstrap_top5_frequency"],
                ),
                xytext=label_offsets.get(fault_id, (5, 5)),
                textcoords="offset points",
                fontsize=8.5,
            )
    ax.axvline(5.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(80.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.text(
        5.2,
        51,
        "5% family-margin guide",
        rotation=90,
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    ax.text(
        ax.get_xlim()[1] * 0.62,
        80.7,
        "80% minimum-frequency guide",
        fontsize=8,
        color="#666666",
    )
    ax.set_xlabel("Relative chosen-family score margin (%)")
    ax.set_ylabel("Minimum selected-variable bootstrap Top-5 frequency (%)")
    ax.set_title(
        "Development-only selection stability diagnostics\n"
        "Guides are descriptive and do not alter the frozen protocol"
    )
    ax.grid(alpha=0.22)
    ax.legend(title="Frozen evidence family", frameon=True)
    return save_figure(fig, "fig23c_selection_stability_diagnostic")


def plot_development_final_transfer(
    results: pd.DataFrame,
    protocol: pd.DataFrame,
) -> list[Path]:
    proposed = results[results["method_key"] == PROPOSED_KEY]
    dev = proposed[proposed["dataset"] == DEVELOPMENT_DATASET].set_index("fault_id").loc[
        RESERVED_FAULTS
    ]
    final = proposed[proposed["dataset"] == FINAL_DATASET].set_index("fault_id").loc[
        RESERVED_FAULTS
    ]
    family = protocol.set_index("fault_id")["family_key"].loc[RESERVED_FAULTS]
    x = 100 * dev["fault_post_fault_point_detection_rate"].to_numpy()
    y = 100 * final["fault_post_fault_point_detection_rate"].to_numpy()
    lower = max(0.0, min(float(x.min()), float(y.min())) - 4.0)
    upper = 100.5

    fig, ax = plt.subplots(figsize=(8.2, 7.4), constrained_layout=True)
    for family_key in family.unique():
        mask = family.to_numpy() == family_key
        ax.scatter(
            x[mask],
            y[mask],
            s=95,
            color=FAMILY_COLORS[family_key],
            edgecolor="white",
            linewidth=0.8,
            label=FAMILY_LABELS[family_key],
            zorder=3,
        )
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    label_offsets = {
        10: (7, -14),
        13: (-16, 8),
        17: (7, 7),
        18: (7, 7),
        20: (-18, -14),
    }
    for fault_id, x_value, y_value in zip(RESERVED_FAULTS, x, y):
        if fault_id not in label_offsets:
            continue
        ax.annotate(
            str(fault_id),
            (x_value, y_value),
            xytext=label_offsets[fault_id],
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.text(
        0.04,
        0.90,
        "Development and final point-detection rates\nremain close for all 17 frozen branches",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Development runs 1–250 point detection (%)")
    ax.set_ylabel("Frozen final runs 251–500 point detection (%)")
    ax.set_title(
        "Frozen Proposed Top-5 development-to-final transfer\n"
        "Each point is one reserved TEP fault"
    )
    ax.grid(alpha=0.22)
    ax.legend(title="Frozen evidence family", frameon=True)
    return save_figure(fig, "fig23d_development_final_transfer")


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def run_analysis() -> None:
    specification = verify_specification()
    data = load_and_validate_inputs()
    results = data["results"]
    final = results[results["dataset"] == FINAL_DATASET].copy()
    final["method_order"] = final["method_key"].map(
        {key: index for index, key in enumerate(METHOD_ORDER)}
    )
    final = final.sort_values(["fault_id", "method_order"])

    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(RESERVED_FAULTS),
        size=(BOOTSTRAP_REPLICATES, len(RESERVED_FAULTS)),
    )

    primary_summary = build_primary_summary(final, bootstrap_indices)
    pairwise = build_pairwise_statistics(final, bootstrap_indices)
    discordance = build_run_detection_discordance(data["run_results"])
    stability = build_selection_stability(
        data["protocol"],
        data["family_summary"],
        data["all_features"],
    )
    cases = build_development_case_studies(data["step15"], data["step20"])
    guardrails = build_guardrails(final)

    output_tables = {
        "primary_summary": TABLE_DIR / "23_reserved_primary_summary_with_ci.csv",
        "pairwise": TABLE_DIR / "23_reserved_pairwise_statistics.csv",
        "discordance": TABLE_DIR / "23_reserved_run_detection_discordance.csv",
        "stability": TABLE_DIR / "23_selection_stability_diagnostic.csv",
        "cases": TABLE_DIR / "23_method_development_case_studies.csv",
        "guardrails": TABLE_DIR / "23_reporting_guardrails.csv",
    }
    primary_summary.to_csv(output_tables["primary_summary"], index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_tables["pairwise"], index=False, encoding="utf-8-sig")
    discordance.to_csv(output_tables["discordance"], index=False, encoding="utf-8-sig")
    stability.to_csv(output_tables["stability"], index=False, encoding="utf-8-sig")
    cases.to_csv(output_tables["cases"], index=False, encoding="utf-8-sig")
    guardrails.to_csv(output_tables["guardrails"], index=False, encoding="utf-8-sig")

    output_figures = []
    output_figures.extend(plot_performance_tradeoffs(primary_summary))
    output_figures.extend(plot_proposed_by_fault(final, data["protocol"]))
    output_figures.extend(plot_selection_stability(stability))
    output_figures.extend(plot_development_final_transfer(results, data["protocol"]))

    config = {
        "script": "scripts/23_postfreeze_statistical_synthesis.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "specification": str(SPECIFICATION_PATH.relative_to(PROJECT_ROOT)),
        "specification_registered_at_utc": specification["registered_at_utc"],
        "specification_verified": True,
        "input_sha256_verified": input_hashes(),
        "analysis_status": "post-freeze secondary synthesis",
        "process_data_loaded": False,
        "detector_fitting_performed": False,
        "selection_or_tuning_performed": False,
        "performance_rows_changed_or_recomputed": False,
        "primary_reserved_faults": RESERVED_FAULTS,
        "method_development_cases_not_pooled": DEVELOPMENT_CASE_FAULTS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "multiplicity": "Holm within endpoint",
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_tables.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_figures
        ],
        "next_step_constraint": (
            "Use these tables to rewrite the manuscript without changing frozen performance; "
            "any new repeated-split experiment must be labelled secondary and development-only."
        ),
    }
    FINAL_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    proposed_summary = primary_summary[
        primary_summary["method_key"] == PROPOSED_KEY
    ][["endpoint", "estimate", "ci_95_low", "ci_95_high", "unit", "endpoint_rank"]]
    sensitive = stability[
        stability["descriptive_stability_diagnostic"] != "descriptively-stable"
    ][
        [
            "fault_id",
            "chosen_family_label",
            "relative_family_score_margin",
            "minimum_selected_bootstrap_top5_frequency",
            "descriptive_stability_diagnostic",
        ]
    ]

    print("=" * 96)
    print("Step 23: Post-freeze statistical synthesis")
    print("=" * 96)
    print("No process data, fitting, selection, tuning, or performance recomputation occurred.")
    print("Primary inference population: 17 Step-22 reserved faults.")
    print("Faults 3, 9, and 15 remain separate method-development cases.\n")
    print("Proposed Top-5 primary summary with fault-cluster 95% CIs:")
    print(proposed_summary.to_string(index=False))
    print("\nHolm-adjusted paired comparisons:")
    print(
        pairwise[
            [
                "endpoint",
                "baseline_short",
                "mean_utility_difference",
                "median_utility_difference",
                "faults_favoring_proposed",
                "faults_tied",
                "faults_favoring_baseline",
                "wilcoxon_holm_p_within_endpoint",
                "holm_reject_0_05",
            ]
        ].to_string(index=False)
    )
    print("\nSelection-stability diagnostics requiring transparent discussion:")
    if sensitive.empty:
        print("None under the registered descriptive guides.")
    else:
        print(sensitive.to_string(index=False))
    print("\nSaved tables:")
    for path in output_tables.values():
        print(f"  {path}")
    print("Saved figures:")
    for path in output_figures:
        print(f"  {path}")
    print(f"Saved configuration: {FINAL_CONFIG_PATH}")
    print("Step 23 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--register", action="store_true")
    group.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_specification()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
