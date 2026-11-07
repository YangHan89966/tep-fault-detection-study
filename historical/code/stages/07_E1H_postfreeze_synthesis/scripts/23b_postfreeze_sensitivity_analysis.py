"""Step 23B: Post-freeze finite-run and dependence-aware sensitivity analysis.

Scientific role
---------------
This script is a secondary analysis of the already-frozen Step-22 outputs.
It does not load process data and it does not fit, select, tune, or re-evaluate
any detector.  In particular, variables, evidence families, Top-K, CUSUM k,
and control limits remain exactly as frozen in Steps 21 and 22.

The script addresses three reporting questions that are not fully answered by
detected-run medians and a fault-only bootstrap:

1. What exact finite-run uncertainty accompanies each run-detection rate?
2. Do alarm-delay conclusions change when missed runs receive a prespecified
   full-horizon penalty instead of being omitted from the delay summary?
3. How uncertain is macro normal run-level FAR when both fault-specific
   detector heterogeneity and reuse of the same 250 normal runs are respected?

Two-stage use
-------------
Place in:
    TEP_Fault3_Project/scripts/23b_postfreeze_sensitivity_analysis.py

Register the immutable post-freeze analysis specification:
    python scripts/23b_postfreeze_sensitivity_analysis.py --register

Then run the registered analysis:
    python scripts/23b_postfreeze_sensitivity_analysis.py --run

Registration is a post-freeze secondary-analysis record, not a prospective
performance preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in (TABLE_DIR, FIGURE_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


STEP22_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
STEP22_RUN_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_run_results.csv"
STEP22_CONFIG_PATH = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"

SPECIFICATION_PATH = CONFIG_DIR / "23b_postfreeze_sensitivity_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "23b_postfreeze_sensitivity_analysis.json"

OUTPUT_PATHS = {
    "detection_delay_by_fault": (
        TABLE_DIR / "23b_run_detection_and_allrun_delay_by_fault.csv"
    ),
    "delay_macro": TABLE_DIR / "23b_allrun_delay_macro_summary.csv",
    "far_bootstrap": TABLE_DIR / "23b_crossed_normal_far_bootstrap.csv",
    "far_pairwise": TABLE_DIR / "23b_crossed_normal_far_pairwise.csv",
    "integrity": TABLE_DIR / "23b_frozen_run_result_integrity_audit.csv",
    "delay_figure_png": FIGURE_DIR / "fig23b_allrun_delay_sensitivity.png",
    "delay_figure_pdf": FIGURE_DIR / "fig23b_allrun_delay_sensitivity.pdf",
    "far_figure_png": FIGURE_DIR / "fig23b_crossed_far_uncertainty.png",
    "far_figure_pdf": FIGURE_DIR / "fig23b_crossed_far_uncertainty.pdf",
}

REQUIRED_INPUTS = {
    "step22_results": STEP22_RESULTS_PATH,
    "step22_run_results": STEP22_RUN_RESULTS_PATH,
    "step22_config": STEP22_CONFIG_PATH,
}

RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
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
METHOD_COLORS = {
    "pca_q_cusum": "#4C78A8",
    "pca_t2_cusum": "#72B7B2",
    "pca_joint_cusum": "#54A24B",
    "lda_residual_cusum": "#B279A2",
    "chosen_family_all52_cusum": "#9D755D",
    "chosen_family_top1_cusum": "#F2CF5B",
    "frozen_multievidence_top5_cusum": "#E45756",
}

DEVELOPMENT_DATASET = "development_fit_runs_1_250"
FINAL_DATASET = "faulty_testing_final_full_960"
NORMAL_DATASET = "normal_evaluation"

EXPECTED_RUNS_PER_CELL = 250
EXPECTED_DEVELOPMENT_RUN_IDS = list(range(1, 251))
EXPECTED_FINAL_RUN_IDS = list(range(251, 501))
EXPECTED_NORMAL_RUN_IDS = list(range(251, 501))

FAULT_START_SAMPLE = 161
FINAL_SAMPLE = 960
SAMPLING_INTERVAL_MIN = 3
POSTFAULT_SAMPLES = FINAL_SAMPLE - FAULT_START_SAMPLE + 1
# A missed run is assigned one full monitored post-fault horizon.  The latest
# observable alarm has delay 2397 min; the miss penalty is therefore 2400 min.
MISS_PENALTY_MIN = POSTFAULT_SAMPLES * SAMPLING_INTERVAL_MIN
LATEST_OBSERVABLE_DELAY_MIN = (
    FINAL_SAMPLE - FAULT_START_SAMPLE
) * SAMPLING_INTERVAL_MIN

BOOTSTRAP_REPLICATES = 20000
RANDOM_SEED = 20260716
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
        "registration_name": "Step 23B post-freeze sensitivity analysis",
        "registration_status": (
            "post-freeze secondary analysis; Step-22 performance already exists"
        ),
        "script": "scripts/23b_postfreeze_sensitivity_analysis.py",
        "script_sha256": script_hash,
        "input_sha256": hashes,
        "process_data_loaded": False,
        "detector_performance_recomputed": False,
        "detector_fitting_allowed": False,
        "variable_selection_allowed": False,
        "evidence_family_selection_allowed": False,
        "top_k_tuning_allowed": False,
        "cusum_k_tuning_allowed": False,
        "control_limit_tuning_allowed": False,
        "primary_population": {
            "faults": RESERVED_FAULTS,
            "n_faults": len(RESERVED_FAULTS),
            "final_runs_per_fault": EXPECTED_RUNS_PER_CELL,
            "normal_runs": EXPECTED_RUNS_PER_CELL,
        },
        "finite_run_detection_intervals": {
            "method": "two-sided Clopper-Pearson exact binomial interval",
            "confidence_level": 1 - ALPHA,
            "unit": "run within fault and method",
        },
        "all_run_delay_sensitivity": {
            "detected_run_delay": (
                "(first post-fault alarm sample - 161) * 3 minutes"
            ),
            "miss_penalty_minutes": MISS_PENALTY_MIN,
            "miss_penalty_definition": (
                "one full 800-sample post-fault monitoring horizon"
            ),
            "latest_observable_delay_minutes": LATEST_OBSERVABLE_DELAY_MIN,
            "primary_cross_fault_unit": "fault",
            "uncertainty": "fault-cluster percentile bootstrap",
        },
        "normal_far_uncertainty": {
            "method": (
                "crossed nonparametric bootstrap resampling the 17 fault-specific "
                "detectors and the 250 shared normal runs independently"
            ),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "confidence_level": 1 - ALPHA,
            "additional_decompositions": ["fault-only", "normal-run-only"],
        },
        "reporting_guardrails": [
            "Detected-run delay remains reported but is explicitly conditional on detection.",
            "All-run penalized delay is a sensitivity endpoint, not a newly tuned objective.",
            "The 2400-minute miss penalty was fixed before reading Step-22 run rows.",
            "Do not describe post-freeze registration as prospective preregistration.",
            "Do not claim equivalence from a non-significant comparison.",
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
            "miss penalty": existing.get("all_run_delay_sensitivity", {}).get(
                "miss_penalty_minutes"
            )
            == MISS_PENALTY_MIN,
            "reserved faults": existing.get("primary_population", {}).get("faults")
            == RESERVED_FAULTS,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Existing Step-23B specification differs from current files: "
                f"{failed}. Preserve it and investigate rather than overwriting it."
            )
        print("Step 23B specification already exists and matches current frozen files.")
        print(f"Registered script SHA-256: {script_hash}")
        print(f"Specification: {SPECIFICATION_PATH}")
        return

    SPECIFICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 23B post-freeze analysis specification registered.")
    print("This is a secondary analysis record, not a performance preregistration.")
    print(f"Registered script SHA-256: {script_hash}")
    print(f"Specification: {SPECIFICATION_PATH}")


def verify_specification() -> dict:
    require_file(
        SPECIFICATION_PATH,
        "run this same script once with --register before --run",
    )
    specification = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
    checks = {
        "script hash": specification.get("script_sha256")
        == sha256_file(Path(__file__)),
        "input hashes": specification.get("input_sha256") == input_hashes(),
        "miss penalty": specification.get("all_run_delay_sensitivity", {}).get(
            "miss_penalty_minutes"
        )
        == MISS_PENALTY_MIN,
        "reserved faults": specification.get("primary_population", {}).get("faults")
        == RESERVED_FAULTS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-23B specification verification failed: {failed}")
    return specification


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{label} contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(normalized[invalid].unique().tolist())[:5]
        raise ValueError(f"{label} contains non-Boolean values: {examples}")
    return normalized.map(mapping).astype(bool)


def dataframe_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    canonical = frame.sort_values(sort_columns).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("\n".join(canonical.columns).encode("utf-8"))
    digest.update("\n".join(map(str, canonical.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(canonical, index=False).values.tobytes())
    return digest.hexdigest()


def audit_row(check: str, passed: bool, observed: object, expected: object) -> dict:
    return {
        "check": check,
        "passed": bool(passed),
        "observed": str(observed),
        "expected": str(expected),
    }


def load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    results = pd.read_csv(require_file(STEP22_RESULTS_PATH, "read Step-22 results"))
    runs = pd.read_csv(
        require_file(STEP22_RUN_RESULTS_PATH, "read Step-22 run-level results")
    )
    config = json.loads(
        require_file(STEP22_CONFIG_PATH, "read Step-22 configuration").read_text(
            encoding="utf-8"
        )
    )

    require_columns(
        results,
        [
            "fault_id",
            "method_key",
            "dataset",
            "normal_run_level_far",
            "fault_run_level_detection_rate",
            "median_alarm_delay_min",
        ],
        "Step-22 result table",
    )
    require_columns(
        runs,
        [
            "fault_id",
            "method_key",
            "dataset",
            "simulationRun",
            "detected",
            "alarm_delay_min",
            "run_alarm",
        ],
        "Step-22 run-result table",
    )

    for frame in (results, runs):
        frame["fault_id"] = pd.to_numeric(frame["fault_id"], errors="raise").astype(int)
        frame["method_key"] = frame["method_key"].astype(str)
        frame["dataset"] = frame["dataset"].astype(str)
    runs["simulationRun"] = pd.to_numeric(
        runs["simulationRun"], errors="raise"
    ).astype(int)

    expected_result_rows = len(RESERVED_FAULTS) * len(METHOD_ORDER) * 2
    expected_run_rows = (
        len(RESERVED_FAULTS)
        * len(METHOD_ORDER)
        * EXPECTED_RUNS_PER_CELL
        * 3
    )
    audit = []
    audit.append(
        audit_row(
            "Step-22 aggregate result row count",
            len(results) == expected_result_rows,
            len(results),
            expected_result_rows,
        )
    )
    audit.append(
        audit_row(
            "Step-22 run-result row count",
            len(runs) == expected_run_rows,
            len(runs),
            expected_run_rows,
        )
    )
    duplicate_count = int(
        runs.duplicated(["fault_id", "method_key", "dataset", "simulationRun"]).sum()
    )
    audit.append(
        audit_row(
            "Unique run-result identifiers",
            duplicate_count == 0,
            duplicate_count,
            0,
        )
    )
    audit.append(
        audit_row(
            "Reserved fault identities",
            sorted(runs["fault_id"].unique().tolist()) == RESERVED_FAULTS,
            sorted(runs["fault_id"].unique().tolist()),
            RESERVED_FAULTS,
        )
    )
    audit.append(
        audit_row(
            "Frozen method identities",
            set(runs["method_key"]) == set(METHOD_ORDER),
            sorted(runs["method_key"].unique().tolist()),
            METHOD_ORDER,
        )
    )
    expected_datasets = {DEVELOPMENT_DATASET, FINAL_DATASET, NORMAL_DATASET}
    audit.append(
        audit_row(
            "Run-result dataset identities",
            set(runs["dataset"]) == expected_datasets,
            sorted(runs["dataset"].unique().tolist()),
            sorted(expected_datasets),
        )
    )

    group_sizes = runs.groupby(["fault_id", "method_key", "dataset"]).size()
    audit.append(
        audit_row(
            "Every fault-method-dataset cell has 250 runs",
            bool((group_sizes == EXPECTED_RUNS_PER_CELL).all()),
            sorted(group_sizes.unique().tolist()),
            [EXPECTED_RUNS_PER_CELL],
        )
    )

    expected_id_map = {
        DEVELOPMENT_DATASET: EXPECTED_DEVELOPMENT_RUN_IDS,
        FINAL_DATASET: EXPECTED_FINAL_RUN_IDS,
        NORMAL_DATASET: EXPECTED_NORMAL_RUN_IDS,
    }
    for dataset, expected_ids in expected_id_map.items():
        matches = True
        for _, group in runs[runs["dataset"] == dataset].groupby(
            ["fault_id", "method_key"]
        ):
            if sorted(group["simulationRun"].tolist()) != expected_ids:
                matches = False
                break
        audit.append(
            audit_row(
                f"{dataset} run identifiers",
                matches,
                "all cells match" if matches else "at least one mismatch",
                f"{expected_ids[0]}--{expected_ids[-1]}",
            )
        )

    config_checks = {
        "All fault states frozen before final subsetting": config.get(
            "all_states_frozen_before_any_final_subset"
        )
        is True,
        "Final rows not used for tuning": config.get("final_used_for_tuning") is False,
        "Normal evaluation not used for tuning": config.get(
            "normal_evaluation_used_for_tuning"
        )
        is False,
        "Companion data not loaded": config.get("companion_loaded") is False,
        "Numerical repair only": config.get("numerical_repair_only") is True,
        "Scientific protocol unchanged": config.get("scientific_protocol_changed")
        is False,
        "Reserved fault list unchanged": config.get("reserved_faults")
        == RESERVED_FAULTS,
    }
    for name, passed in config_checks.items():
        audit.append(audit_row(name, passed, passed, True))

    final = runs[runs["dataset"] == FINAL_DATASET].copy()
    normal = runs[runs["dataset"] == NORMAL_DATASET].copy()
    final["detected_bool"] = parse_bool(final["detected"], "final detected")
    normal["run_alarm_bool"] = parse_bool(normal["run_alarm"], "normal run_alarm")
    final["alarm_delay_min"] = pd.to_numeric(
        final["alarm_delay_min"], errors="coerce"
    )
    detected_delay_valid = (
        final.loc[final["detected_bool"], "alarm_delay_min"].notna().all()
        and final.loc[final["detected_bool"], "alarm_delay_min"].between(
            0, LATEST_OBSERVABLE_DELAY_MIN
        ).all()
    )
    missed_delay_missing = final.loc[
        ~final["detected_bool"], "alarm_delay_min"
    ].isna().all()
    audit.append(
        audit_row(
            "Detected-run delay is finite and inside monitored horizon",
            detected_delay_valid,
            detected_delay_valid,
            True,
        )
    )
    audit.append(
        audit_row(
            "Missed-run delay is right-censored/missing before sensitivity penalty",
            missed_delay_missing,
            missed_delay_missing,
            True,
        )
    )

    final_results = results[results["dataset"] == FINAL_DATASET].copy()
    parity_failures = []
    for (fault_id, method_key), group in final.groupby(["fault_id", "method_key"]):
        row = final_results[
            (final_results["fault_id"] == fault_id)
            & (final_results["method_key"] == method_key)
        ]
        if len(row) != 1:
            parity_failures.append((fault_id, method_key, "missing aggregate row"))
            continue
        row = row.iloc[0]
        detection = float(group["detected_bool"].mean())
        delay = float(group.loc[group["detected_bool"], "alarm_delay_min"].median())
        expected_delay = float(row["median_alarm_delay_min"])
        if not np.isclose(
            detection,
            float(row["fault_run_level_detection_rate"]),
            atol=TOLERANCE,
        ) or not (
            (np.isnan(delay) and np.isnan(expected_delay))
            or np.isclose(delay, expected_delay, atol=TOLERANCE)
        ):
            parity_failures.append((fault_id, method_key, "fault metric mismatch"))
        normal_group = normal[
            (normal["fault_id"] == fault_id)
            & (normal["method_key"] == method_key)
        ]
        normal_far = float(normal_group["run_alarm_bool"].mean())
        if not np.isclose(
            normal_far, float(row["normal_run_level_far"]), atol=TOLERANCE
        ):
            parity_failures.append((fault_id, method_key, "normal FAR mismatch"))
    audit.append(
        audit_row(
            "Run rows exactly reconstruct Step-22 aggregate metrics",
            len(parity_failures) == 0,
            parity_failures[:5] if parity_failures else "all cells match",
            "all cells match",
        )
    )

    audit.append(
        audit_row(
            "Canonical final run-result hash",
            True,
            dataframe_sha256(
                final,
                ["fault_id", "method_key", "simulationRun"],
            ),
            "recorded for reproducibility",
        )
    )
    audit.append(
        audit_row(
            "Canonical normal run-result hash",
            True,
            dataframe_sha256(
                normal,
                ["fault_id", "method_key", "simulationRun"],
            ),
            "recorded for reproducibility",
        )
    )
    audit.append(
        audit_row(
            "Raw-process duplicate-content scope",
            True,
            "not tested here because Step 23B intentionally loads no process data",
            "identifier and frozen-result integrity only; raw-content audit belongs to Step 24",
        )
    )

    audit_table = pd.DataFrame(audit)
    failed = audit_table[~audit_table["passed"]]
    if not failed.empty:
        raise ValueError(
            "Step-23B frozen-input audit failed: "
            + "; ".join(failed["check"].tolist())
        )
    return results, runs, config, audit_table


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Invalid exact-binomial counts")
    low = 0.0 if successes == 0 else float(
        beta.ppf(ALPHA / 2, successes, trials - successes + 1)
    )
    high = 1.0 if successes == trials else float(
        beta.ppf(1 - ALPHA / 2, successes + 1, trials - successes)
    )
    return low, high


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def fault_bootstrap_interval(
    values: np.ndarray,
    indices: np.ndarray,
    statistic: str,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (len(RESERVED_FAULTS),):
        raise ValueError("Fault bootstrap requires one value per reserved fault")
    if statistic == "mean":
        observed = float(values.mean())
        replicates = values[indices].mean(axis=1)
    elif statistic == "median":
        observed = float(np.median(values))
        replicates = np.median(values[indices], axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    low, high = percentile_interval(replicates)
    return observed, low, high


def build_detection_delay_table(runs: pd.DataFrame) -> pd.DataFrame:
    final = runs[runs["dataset"] == FINAL_DATASET].copy()
    final["detected_bool"] = parse_bool(final["detected"], "final detected")
    final["alarm_delay_min"] = pd.to_numeric(
        final["alarm_delay_min"], errors="coerce"
    )
    final["allrun_penalized_delay_min"] = np.where(
        final["detected_bool"], final["alarm_delay_min"], MISS_PENALTY_MIN
    )

    rows = []
    for method_order, method_key in enumerate(METHOD_ORDER):
        method = final[final["method_key"] == method_key]
        for fault_id in RESERVED_FAULTS:
            group = method[method["fault_id"] == fault_id].sort_values(
                "simulationRun"
            )
            n_runs = len(group)
            detected = int(group["detected_bool"].sum())
            low, high = clopper_pearson(detected, n_runs)
            detected_delays = group.loc[
                group["detected_bool"], "alarm_delay_min"
            ].to_numpy(dtype=float)
            penalized = group["allrun_penalized_delay_min"].to_numpy(dtype=float)
            rows.append(
                {
                    "fault_id": fault_id,
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "method_order": method_order,
                    "n_runs": n_runs,
                    "detected_runs": detected,
                    "right_censored_runs": n_runs - detected,
                    "run_detection_rate": detected / n_runs,
                    "run_detection_exact_ci_95_low": low,
                    "run_detection_exact_ci_95_high": high,
                    "detected_only_median_delay_min": (
                        float(np.median(detected_delays))
                        if len(detected_delays)
                        else np.nan
                    ),
                    "detected_only_mean_delay_min": (
                        float(np.mean(detected_delays))
                        if len(detected_delays)
                        else np.nan
                    ),
                    "allrun_penalized_median_delay_min": float(
                        np.median(penalized)
                    ),
                    "allrun_penalized_mean_delay_min": float(np.mean(penalized)),
                    "allrun_penalized_q90_delay_min": float(
                        np.quantile(penalized, 0.90)
                    ),
                    "miss_penalty_min": MISS_PENALTY_MIN,
                    "penalty_definition": (
                        "missed run assigned one full 800-sample post-fault horizon"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["fault_id", "method_order"])


def build_delay_macro_summary(
    by_fault: pd.DataFrame,
    fault_indices: np.ndarray,
) -> pd.DataFrame:
    endpoints = [
        (
            "run_detection_mean",
            "Macro mean run-level detection",
            "run_detection_rate",
            "mean",
            100.0,
            "percent",
            "higher is better",
        ),
        (
            "detected_only_delay_median",
            "Macro median detected-run median delay",
            "detected_only_median_delay_min",
            "median",
            1.0,
            "minutes",
            "lower is better; conditional on detection",
        ),
        (
            "allrun_penalized_delay_mean",
            "Macro mean all-run penalized mean delay",
            "allrun_penalized_mean_delay_min",
            "mean",
            1.0,
            "minutes",
            "lower is better",
        ),
        (
            "allrun_penalized_delay_median",
            "Macro median all-run penalized median delay",
            "allrun_penalized_median_delay_min",
            "median",
            1.0,
            "minutes",
            "lower is better",
        ),
    ]
    rows = []
    for method_order, method_key in enumerate(METHOD_ORDER):
        group = by_fault[by_fault["method_key"] == method_key].set_index(
            "fault_id"
        ).loc[RESERVED_FAULTS]
        for endpoint_order, (
            endpoint_key,
            endpoint,
            column,
            statistic,
            scale,
            unit,
            direction,
        ) in enumerate(endpoints):
            values = group[column].to_numpy(dtype=float)
            estimate, low, high = fault_bootstrap_interval(
                values, fault_indices, statistic
            )
            rows.append(
                {
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "method_order": method_order,
                    "endpoint_key": endpoint_key,
                    "endpoint": endpoint,
                    "endpoint_order": endpoint_order,
                    "direction": direction,
                    "estimate": scale * estimate,
                    "ci_95_low": scale * low,
                    "ci_95_high": scale * high,
                    "unit": unit,
                    "n_faults": len(RESERVED_FAULTS),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "miss_penalty_min": MISS_PENALTY_MIN,
                }
            )
    return pd.DataFrame(rows).sort_values(["endpoint_order", "method_order"])


def crossed_matrix_bootstrap(
    matrix: np.ndarray,
    fault_indices: np.ndarray,
    run_indices: np.ndarray,
    chunk_size: int = 250,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (len(RESERVED_FAULTS), EXPECTED_RUNS_PER_CELL)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Crossed bootstrap matrix has shape {matrix.shape}, expected {expected_shape}"
        )
    output = np.empty(len(fault_indices), dtype=float)
    for start in range(0, len(output), chunk_size):
        stop = min(start + chunk_size, len(output))
        sampled = matrix[
            fault_indices[start:stop, :, None],
            run_indices[start:stop, None, :],
        ]
        output[start:stop] = sampled.mean(axis=(1, 2))
    return output


def normal_alarm_matrices(runs: pd.DataFrame) -> dict[str, np.ndarray]:
    normal = runs[runs["dataset"] == NORMAL_DATASET].copy()
    normal["run_alarm_bool"] = parse_bool(normal["run_alarm"], "normal run_alarm")
    matrices = {}
    for method_key in METHOD_ORDER:
        group = normal[normal["method_key"] == method_key]
        pivot = group.pivot(
            index="fault_id", columns="simulationRun", values="run_alarm_bool"
        ).loc[RESERVED_FAULTS, EXPECTED_NORMAL_RUN_IDS]
        if pivot.isna().any().any():
            raise ValueError(f"Normal alarm matrix is incomplete for {method_key}")
        matrices[method_key] = pivot.to_numpy(dtype=float)
    return matrices


def build_crossed_far_tables(
    matrices: dict[str, np.ndarray],
    fault_indices: np.ndarray,
    run_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    replicate_map = {}
    for method_order, method_key in enumerate(METHOD_ORDER):
        matrix = matrices[method_key]
        crossed = crossed_matrix_bootstrap(matrix, fault_indices, run_indices)
        replicate_map[method_key] = crossed
        fault_only = matrix.mean(axis=1)[fault_indices].mean(axis=1)
        run_only = matrix.mean(axis=0)[run_indices].mean(axis=1)
        crossed_low, crossed_high = percentile_interval(crossed)
        fault_low, fault_high = percentile_interval(fault_only)
        run_low, run_high = percentile_interval(run_only)
        rows.append(
            {
                "method_key": method_key,
                "method_short": METHOD_LABELS[method_key],
                "method_order": method_order,
                "macro_normal_run_far": float(matrix.mean()),
                "crossed_ci_95_low": crossed_low,
                "crossed_ci_95_high": crossed_high,
                "fault_only_ci_95_low": fault_low,
                "fault_only_ci_95_high": fault_high,
                "normal_run_only_ci_95_low": run_low,
                "normal_run_only_ci_95_high": run_high,
                "n_fault_specific_detectors": len(RESERVED_FAULTS),
                "n_shared_normal_runs": EXPECTED_RUNS_PER_CELL,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "crossed_bootstrap_definition": (
                    "fault-specific detectors and shared normal runs resampled independently"
                ),
            }
        )

    pairwise_rows = []
    proposed_matrix = matrices[PROPOSED_KEY]
    for baseline_order, baseline_key in enumerate(
        [key for key in METHOD_ORDER if key != PROPOSED_KEY]
    ):
        # Positive difference means the baseline alarms more often than the
        # proposed detector and therefore favors the proposed method.
        difference_matrix = matrices[baseline_key] - proposed_matrix
        replicates = crossed_matrix_bootstrap(
            difference_matrix, fault_indices, run_indices
        )
        low, high = percentile_interval(replicates)
        pairwise_rows.append(
            {
                "reference_key": PROPOSED_KEY,
                "reference_short": METHOD_LABELS[PROPOSED_KEY],
                "baseline_key": baseline_key,
                "baseline_short": METHOD_LABELS[baseline_key],
                "baseline_order": baseline_order,
                "macro_far_difference_baseline_minus_proposed": float(
                    difference_matrix.mean()
                ),
                "crossed_ci_95_low": low,
                "crossed_ci_95_high": high,
                "bootstrap_probability_difference_above_zero": float(
                    np.mean(replicates > 0)
                ),
                "effect_definition": (
                    "baseline FAR minus proposed FAR; positive favors proposed"
                ),
                "inferential_status": (
                    "post-freeze dependence-aware sensitivity comparison"
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pairwise_rows)


def plot_delay_sensitivity(delay_macro: pd.DataFrame) -> None:
    detected = delay_macro[
        delay_macro["endpoint_key"] == "detected_only_delay_median"
    ].set_index("method_key").loc[METHOD_ORDER]
    penalized = delay_macro[
        delay_macro["endpoint_key"] == "allrun_penalized_delay_median"
    ].set_index("method_key").loc[METHOD_ORDER]

    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        x - width / 2,
        detected["estimate"],
        width,
        color="#9ECAE1",
        edgecolor="#4C78A8",
        label="Detected-run median (conditional)",
    )
    ax.bar(
        x + width / 2,
        penalized["estimate"],
        width,
        color=[METHOD_COLORS[key] for key in METHOD_ORDER],
        alpha=0.90,
        label=f"All-run penalized median (miss = {MISS_PENALTY_MIN} min)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[key] for key in METHOD_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Macro median of fault-level median alarm delay (min)")
    ax.set_title("Detected-run and all-run alarm-delay summaries")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATHS["delay_figure_png"], dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PATHS["delay_figure_pdf"], bbox_inches="tight")
    plt.close(fig)


def plot_far_uncertainty(far_table: pd.DataFrame) -> None:
    table = far_table.set_index("method_key").loc[METHOD_ORDER]
    estimate = 100 * table["macro_normal_run_far"].to_numpy(dtype=float)
    low = 100 * table["crossed_ci_95_low"].to_numpy(dtype=float)
    high = 100 * table["crossed_ci_95_high"].to_numpy(dtype=float)
    y = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for index, method_key in enumerate(METHOD_ORDER):
        ax.errorbar(
            estimate[index],
            y[index],
            xerr=np.array(
                [[estimate[index] - low[index]], [high[index] - estimate[index]]]
            ),
            fmt="o",
            markersize=7,
            capsize=4,
            color=METHOD_COLORS[method_key],
            ecolor=METHOD_COLORS[method_key],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[key] for key in METHOD_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Macro normal run-level FAR (%)")
    ax.set_title("Crossed-bootstrap uncertainty for normal run-level FAR")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATHS["far_figure_png"], dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PATHS["far_figure_pdf"], bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> None:
    specification = verify_specification()
    results, runs, step22_config, integrity = load_and_validate_inputs()

    rng = np.random.default_rng(RANDOM_SEED)
    fault_indices = rng.integers(
        0,
        len(RESERVED_FAULTS),
        size=(BOOTSTRAP_REPLICATES, len(RESERVED_FAULTS)),
        dtype=np.int16,
    )
    run_indices = rng.integers(
        0,
        EXPECTED_RUNS_PER_CELL,
        size=(BOOTSTRAP_REPLICATES, EXPECTED_RUNS_PER_CELL),
        dtype=np.int16,
    )

    by_fault = build_detection_delay_table(runs)
    delay_macro = build_delay_macro_summary(by_fault, fault_indices)
    matrices = normal_alarm_matrices(runs)
    far_table, far_pairwise = build_crossed_far_tables(
        matrices, fault_indices, run_indices
    )

    by_fault.to_csv(
        OUTPUT_PATHS["detection_delay_by_fault"], index=False, encoding="utf-8-sig"
    )
    delay_macro.to_csv(
        OUTPUT_PATHS["delay_macro"], index=False, encoding="utf-8-sig"
    )
    far_table.to_csv(
        OUTPUT_PATHS["far_bootstrap"], index=False, encoding="utf-8-sig"
    )
    far_pairwise.to_csv(
        OUTPUT_PATHS["far_pairwise"], index=False, encoding="utf-8-sig"
    )
    integrity.to_csv(
        OUTPUT_PATHS["integrity"], index=False, encoding="utf-8-sig"
    )
    plot_delay_sensitivity(delay_macro)
    plot_far_uncertainty(far_table)

    proposed_fault = by_fault[by_fault["method_key"] == PROPOSED_KEY]
    proposed_macro = delay_macro[delay_macro["method_key"] == PROPOSED_KEY]
    proposed_far = far_table[far_table["method_key"] == PROPOSED_KEY].iloc[0]
    final_config = {
        "analysis": "Step 23B post-freeze sensitivity analysis",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "specification": specification,
        "step22_config_sha256": sha256_file(STEP22_CONFIG_PATH),
        "step22_numerical_repair_only": step22_config.get("numerical_repair_only"),
        "input_sha256": input_hashes(),
        "output_sha256": {
            key: sha256_file(path)
            for key, path in OUTPUT_PATHS.items()
            if path.exists()
        },
        "proposed_top5_summary": {
            "faults_with_250_of_250_detected": int(
                (
                    (proposed_fault["detected_runs"] == 250)
                    & (proposed_fault["n_runs"] == 250)
                ).sum()
            ),
            "minimum_exact_detection_ci_95_low": float(
                proposed_fault["run_detection_exact_ci_95_low"].min()
            ),
            "macro_normal_far": float(proposed_far["macro_normal_run_far"]),
            "macro_normal_far_crossed_ci_95": [
                float(proposed_far["crossed_ci_95_low"]),
                float(proposed_far["crossed_ci_95_high"]),
            ],
        },
        "no_process_data_loaded": True,
        "no_detector_recomputed_or_retuned": True,
    }
    FINAL_CONFIG_PATH.write_text(
        json.dumps(final_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    exact_low = 100 * proposed_fault["run_detection_exact_ci_95_low"].min()
    detected_delay = proposed_macro[
        proposed_macro["endpoint_key"] == "detected_only_delay_median"
    ].iloc[0]
    penalized_delay = proposed_macro[
        proposed_macro["endpoint_key"] == "allrun_penalized_delay_median"
    ].iloc[0]

    print("\n" + "=" * 88)
    print("Step 23B summary")
    print("=" * 88)
    print(
        "Proposed Top-5 faults with 250/250 final runs detected: "
        f"{int((proposed_fault['detected_runs'] == 250).sum())}/{len(RESERVED_FAULTS)}"
    )
    print(
        "Exact 95% lower bound for a 250/250 fault-specific result: "
        f"{exact_low:.3f}%"
    )
    print(
        "Proposed macro median detected-run delay: "
        f"{detected_delay['estimate']:.1f} min "
        f"[{detected_delay['ci_95_low']:.1f}, {detected_delay['ci_95_high']:.1f}]"
    )
    print(
        "Proposed macro median all-run penalized delay: "
        f"{penalized_delay['estimate']:.1f} min "
        f"[{penalized_delay['ci_95_low']:.1f}, {penalized_delay['ci_95_high']:.1f}]"
    )
    print(
        "Proposed macro normal run-level FAR with crossed 95% CI: "
        f"{100*proposed_far['macro_normal_run_far']:.3f}% "
        f"[{100*proposed_far['crossed_ci_95_low']:.3f}%, "
        f"{100*proposed_far['crossed_ci_95_high']:.3f}%]"
    )
    print("\nSaved tables:")
    for key in (
        "detection_delay_by_fault",
        "delay_macro",
        "far_bootstrap",
        "far_pairwise",
        "integrity",
    ):
        print(f"  {OUTPUT_PATHS[key]}")
    print("Saved figures:")
    for key in (
        "delay_figure_png",
        "delay_figure_pdf",
        "far_figure_png",
        "far_figure_pdf",
    ):
        print(f"  {OUTPUT_PATHS[key]}")
    print(f"Saved configuration: {FINAL_CONFIG_PATH}")
    print("Step 23B finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="register the immutable post-freeze secondary-analysis specification",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify the registration and run the sensitivity analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_specification()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
