"""Step 25: Frozen detector-bank false-alarm and cross-trigger audit.

Scientific role
---------------
Step 22 evaluated one fault-specific, frozen multi-evidence Top-5 detector at
a time.  Step 25 asks a different deployment question: what happens if all 17
reserved-fault detectors are operated simultaneously as a detector bank?

This is a post-freeze secondary system audit.  It does not select a family or
variable, fit a detector, tune CUSUM k, change a control limit, or define a new
diagnosis rule.  Every monitor is reconstructed from the frozen Step-22
parameters and must reproduce its own Step-22 normal and target-fault metrics
before any bank-level result is accepted.

The registered analyses are:

1. Normal bank audit.  On the common normal-evaluation runs, report the
   run-level FAR of the hypothetical OR bank, point-level OR FAR, the number of
   distinct monitors that alarm per run, maximum simultaneous multiplicity,
   and pairwise co-alarm rates.
2. Cross-fault audit.  Apply every frozen fault-specific monitor to every
   reserved final fault set and report the 17 x 17 cross-trigger matrix.
3. Target-monitor diagnostics.  For each final run, report whether the target
   monitor is among the earliest alarms, is uniquely earliest, and has the
   largest frozen-limit-normalized CUSUM peak.  These are descriptive
   diagnostics, not a trained or validated fault-classification rule.

Two-stage use
-------------
Place this file in the project ``scripts`` directory, then run:

    python scripts/25_frozen_detector_bank_audit.py --register
    python scripts/25_frozen_detector_bank_audit.py --run

Registration records the exact post-freeze audit before process data are
parsed.  It is not a prospective performance preregistration because the
Step-22 results were already known.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in (TABLE_DIR, FIGURE_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
EXPECTED_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (
    EXPECTED_SAMPLE_MAX - FAULT_START_SAMPLE + 1
) * SAMPLING_INTERVAL_MIN

TOP5_KEY = "frozen_multievidence_top5_cusum"
TOP5_LABEL = "Proposed Top-5"
PARITY_TOLERANCE = 1e-12

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"

STEP22_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
STEP22_PARAMETERS_PATH = TABLE_DIR / "22_reserved_multifault_method_parameters.csv"
STEP22_SCALERS_PATH = TABLE_DIR / "22_reserved_multifault_local_scalers.csv"
STEP22_CONFIG_PATH = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"

SPECIFICATION_PATH = CONFIG_DIR / "25_frozen_detector_bank_audit_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "25_frozen_detector_bank_audit.json"

OUTPUT_PATHS = {
    "clean_parity": TABLE_DIR / "25_monitor_clean_reconstruction_parity.csv",
    "normal_monitor": TABLE_DIR / "25_normal_monitor_operating_points.csv",
    "normal_run": TABLE_DIR / "25_normal_bank_run_metrics.csv",
    "normal_coalarm": TABLE_DIR / "25_normal_monitor_coalarm.csv",
    "crossfault": TABLE_DIR / "25_crossfault_alarm_matrix.csv",
    "target_run": TABLE_DIR / "25_target_monitor_run_diagnostics.csv",
    "target_fault": TABLE_DIR / "25_target_monitor_diagnostics_by_fault.csv",
    "bank_fault": TABLE_DIR / "25_bank_detection_by_fault.csv",
    "summary": TABLE_DIR / "25_detector_bank_summary.csv",
    "guardrails": TABLE_DIR / "25_reporting_guardrails.csv",
    "normal_png": FIGURE_DIR / "fig25a_normal_bank_false_alarm_audit.png",
    "normal_pdf": FIGURE_DIR / "fig25a_normal_bank_false_alarm_audit.pdf",
    "crossfault_png": FIGURE_DIR / "fig25b_crossfault_alarm_matrix.png",
    "crossfault_pdf": FIGURE_DIR / "fig25b_crossfault_alarm_matrix.pdf",
    "target_png": FIGURE_DIR / "fig25c_target_monitor_rank_diagnostics.png",
    "target_pdf": FIGURE_DIR / "fig25c_target_monitor_rank_diagnostics.pdf",
}


# Registration reads byte hashes only.  No process table or RData object is
# parsed until --run verifies the complete registration.
REQUIRED_INPUTS = {
    "step11_script": SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
    "step15_script": SCRIPT_DIR / "15_fair_sequential_baselines.py",
    "step18_script": SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
    "step20_script": SCRIPT_DIR / "20_frozen_multievidence_validation.py",
    "step24_script": SCRIPT_DIR / "24_frozen_evidence_channel_robustness.py",
    "step22_results": STEP22_RESULTS_PATH,
    "step22_parameters": STEP22_PARAMETERS_PATH,
    "step22_scalers": STEP22_SCALERS_PATH,
    "step22_config": STEP22_CONFIG_PATH,
    "variable_list": TABLE_DIR / "02_variable_list.csv",
    "normal_train": NORMAL_TRAIN_PATH,
    "normal_evaluation": NORMAL_EVALUATION_PATH,
    "faulty_testing_rdata": RAW_TESTING_PATH,
}


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
        key: sha256_file(require_file(path, f"freeze input {key}"))
        for key, path in REQUIRED_INPUTS.items()
    }


def specification_payload(script_hash: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "registration_name": "Step 25 frozen detector-bank system audit",
        "registration_status": (
            "post-freeze secondary system audit; Step-22 performance was "
            "known before registration"
        ),
        "script": "scripts/25_frozen_detector_bank_audit.py",
        "script_sha256": script_hash,
        "input_sha256": hashes,
        "registration_reads": "byte hashes only; no process dataframe is parsed",
        "process_data_loaded_during_registration": False,
        "frozen_population": {
            "monitor_faults": RESERVED_FAULTS,
            "actual_faults": RESERVED_FAULTS,
            "normal_evaluation_runs": [251, 500],
            "final_fault_runs": [251, 500],
            "samples_per_run": EXPECTED_SAMPLE_MAX,
            "fault_start_sample": FAULT_START_SAMPLE,
            "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        },
        "frozen_method": {
            "method_key": TOP5_KEY,
            "label": TOP5_LABEL,
            "channels_per_monitor": 5,
            "monitor_count": len(RESERVED_FAULTS),
            "family_selection_allowed": False,
            "variable_selection_allowed": False,
            "orientation_change_allowed": False,
            "feature_scaler_refitting_allowed": False,
            "cusum_k_tuning_allowed": False,
            "control_limit_tuning_allowed": False,
            "final_data_arbitration_tuning_allowed": False,
        },
        "normal_bank_audit": {
            "bank_rule": "logical OR of all 17 frozen monitor alarms",
            "reported_endpoints": [
                "bank run-level false-alarm rate",
                "bank point-level false-alarm rate",
                "number of distinct monitors alarming per run",
                "maximum simultaneous alarm multiplicity per run",
                "pairwise run-level co-alarm rate and Jaccard index",
            ],
        },
        "crossfault_audit": {
            "matrix_rows": "frozen monitor fault identity",
            "matrix_columns": "actual reserved final fault",
            "reported_endpoints": [
                "post-fault run detection",
                "post-fault point detection",
                "median detected-run alarm delay",
                "all-run penalized mean delay",
                "median frozen-limit-normalized post-fault CUSUM peak",
            ],
        },
        "target_monitor_diagnostics": {
            "earliest_rank_definition": (
                "1 plus the number of monitors with a strictly earlier "
                "post-fault first alarm; ties share the best rank"
            ),
            "peak_rank_definition": (
                "1 plus the number of monitors with a strictly larger "
                "maximum post-fault CUSUM divided by its frozen limit"
            ),
            "diagnosis_rule_trained": False,
            "confirmatory_inference": False,
        },
        "reporting_guardrails": [
            "The OR bank is a hypothetical deployment audit, not the Step-22 primary operating rule.",
            "A cross-trigger is not a false alarm when a real fault is present.",
            "The cross-fault matrix describes detector selectivity, not validated multiclass diagnosis.",
            "Target rank and normalized peak are descriptive and cannot be used to retune an arbitration rule.",
            "Do not use Step-25 results to alter any Step-22 frozen state or primary endpoint.",
            "Retain unfavorable bank-level FAR or cross-trigger results without post hoc threshold changes.",
        ],
    }


def register_specification() -> None:
    script_hash = sha256_file(Path(__file__))
    print("Hashing frozen scripts, outputs, and data containers...")
    hashes = input_hashes()
    payload = specification_payload(script_hash, hashes)
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()

    if SPECIFICATION_PATH.exists():
        existing = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
        checks = {
            "script hash": existing.get("script_sha256") == script_hash,
            "input hashes": existing.get("input_sha256") == hashes,
            "monitor faults": existing.get("frozen_population", {}).get(
                "monitor_faults"
            ) == RESERVED_FAULTS,
            "bank rule": existing.get("normal_bank_audit", {}).get("bank_rule")
            == "logical OR of all 17 frozen monitor alarms",
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Existing Step-25 specification differs from the current "
                f"script or frozen inputs: {failed}. Preserve it and "
                "investigate rather than overwriting it."
            )
        print("Step 25 specification already exists and matches all frozen files.")
        print(f"Registered script SHA-256: {script_hash}")
        print(f"Specification: {SPECIFICATION_PATH}")
        return

    SPECIFICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 25 post-freeze detector-bank specification registered.")
    print("No process dataframe was parsed during registration.")
    print(f"Registered script SHA-256: {script_hash}")
    print(f"Specification: {SPECIFICATION_PATH}")


def verify_specification() -> dict[str, Any]:
    require_file(
        SPECIFICATION_PATH,
        "run this same script with --register before --run",
    )
    specification = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
    print("Re-hashing frozen inputs before process data are loaded...")
    hashes = input_hashes()
    checks = {
        "script hash": specification.get("script_sha256")
        == sha256_file(Path(__file__)),
        "input hashes": specification.get("input_sha256") == hashes,
        "monitor faults": specification.get("frozen_population", {}).get(
            "monitor_faults"
        ) == RESERVED_FAULTS,
        "no arbitration tuning": specification.get("frozen_method", {}).get(
            "final_data_arbitration_tuning_allowed"
        )
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-25 specification verification failed: {failed}")
    return specification


def load_module(path: Path, name: str):
    require_file(path, f"load frozen dependency {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_number(left: Any, right: Any, tolerance: float = PARITY_TOLERANCE) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=tolerance))


def binary_phi(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    n11 = float(np.sum(left_bool & right_bool))
    n10 = float(np.sum(left_bool & ~right_bool))
    n01 = float(np.sum(~left_bool & right_bool))
    n00 = float(np.sum(~left_bool & ~right_bool))
    denominator = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / denominator) if denominator > 0 else np.nan


def first_alarm_samples(alarm: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    if alarm.ndim != 2 or alarm.shape[1] != len(sample_ids):
        raise ValueError("Alarm matrix and sample grid are inconsistent")
    detected = alarm.any(axis=1)
    output = np.full(alarm.shape[0], np.inf, dtype=float)
    if detected.any():
        first_index = alarm[detected].argmax(axis=1)
        output[detected] = sample_ids[first_index]
    return output


def delay_from_first_sample(first_sample: np.ndarray) -> np.ndarray:
    first = np.asarray(first_sample, dtype=float)
    output = np.full(len(first), np.nan, dtype=float)
    finite = np.isfinite(first)
    output[finite] = (first[finite] - FAULT_START_SAMPLE) * SAMPLING_INTERVAL_MIN
    return output


def metric_from_alarm(
    alarm: np.ndarray,
    stat: np.ndarray,
    sample_ids: np.ndarray,
    limit: float,
) -> dict[str, Any]:
    pre_mask = sample_ids < FAULT_START_SAMPLE
    post_mask = sample_ids >= FAULT_START_SAMPLE
    if not pre_mask.any() or not post_mask.any():
        raise ValueError("Both pre- and post-fault samples are required")
    pre_alarm = alarm[:, pre_mask]
    post_alarm = alarm[:, post_mask]
    detected = post_alarm.any(axis=1)
    first = first_alarm_samples(post_alarm, sample_ids[post_mask])
    delay = delay_from_first_sample(first)
    penalized = np.where(detected, delay, float(MISS_PENALTY_MIN))
    observed = delay[detected]
    normalized_peak = stat[:, post_mask].max(axis=1) / float(limit)
    return {
        "pre_fault_run_alarm_rate": float(pre_alarm.any(axis=1).mean()),
        "pre_fault_point_alarm_rate": float(pre_alarm.mean()),
        "post_fault_run_detection_rate": float(detected.mean()),
        "post_fault_point_detection_rate": float(post_alarm.mean()),
        "detected_runs": int(detected.sum()),
        "right_censored_runs": int((~detected).sum()),
        "median_alarm_delay_min": float(np.median(observed)) if len(observed) else np.nan,
        "mean_alarm_delay_min": float(np.mean(observed)) if len(observed) else np.nan,
        "allrun_penalized_median_delay_min": float(np.median(penalized)),
        "allrun_penalized_mean_delay_min": float(np.mean(penalized)),
        "median_normalized_peak": float(np.median(normalized_peak)),
        "mean_normalized_peak": float(np.mean(normalized_peak)),
        "first_sample": first,
        "delay": delay,
        "penalized_delay": penalized,
        "normalized_peak": normalized_peak,
        "detected": detected,
        "pre_alarm": pre_alarm,
        "post_alarm": post_alarm,
    }


def build_normal_bank_tables(
    monitor_alarm: np.ndarray,
    monitor_stats: np.ndarray,
    sample_ids: np.ndarray,
    monitor_limits: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # Arrays are monitor x run x sample.
    if monitor_alarm.shape != monitor_stats.shape:
        raise ValueError("Normal alarm/stat arrays differ")
    if monitor_alarm.shape[0] != len(RESERVED_FAULTS):
        raise ValueError("Normal detector bank is incomplete")

    ever = monitor_alarm.any(axis=2)
    bank_alarm = monitor_alarm.any(axis=0)
    bank_run = bank_alarm.any(axis=1)
    distinct = ever.sum(axis=0)
    simultaneous = monitor_alarm.sum(axis=0)
    max_simultaneous = simultaneous.max(axis=1)
    first_bank = first_alarm_samples(bank_alarm, sample_ids)

    run_rows = []
    for run_index, run_id in enumerate(EXPECTED_RUNS):
        active = [
            str(RESERVED_FAULTS[index])
            for index in np.flatnonzero(ever[:, run_index])
        ]
        run_rows.append(
            {
                "simulationRun": int(run_id),
                "bank_run_alarm": bool(bank_run[run_index]),
                "first_bank_alarm_sample": (
                    float(first_bank[run_index])
                    if np.isfinite(first_bank[run_index])
                    else np.nan
                ),
                "distinct_monitors_ever_alarming": int(distinct[run_index]),
                "maximum_simultaneous_monitors": int(max_simultaneous[run_index]),
                "monitor_faults_ever_alarming": ";".join(active),
            }
        )
    run_table = pd.DataFrame(run_rows)

    monitor_rows = []
    for monitor_index, fault_id in enumerate(RESERVED_FAULTS):
        ratio = monitor_stats[monitor_index] / monitor_limits[monitor_index]
        monitor_rows.append(
            {
                "monitor_fault_id": fault_id,
                "normal_run_level_far": float(ever[monitor_index].mean()),
                "normal_point_level_far": float(monitor_alarm[monitor_index].mean()),
                "normal_alarm_runs": int(ever[monitor_index].sum()),
                "normal_median_max_normalized_cusum": float(
                    np.median(ratio.max(axis=1))
                ),
            }
        )
    monitor_table = pd.DataFrame(monitor_rows)

    coalarm_rows = []
    for left_index, left_fault in enumerate(RESERVED_FAULTS):
        for right_index, right_fault in enumerate(RESERVED_FAULTS):
            left = ever[left_index]
            right = ever[right_index]
            both = left & right
            either = left | right
            coalarm_rows.append(
                {
                    "left_monitor_fault_id": left_fault,
                    "right_monitor_fault_id": right_fault,
                    "left_run_alarm_rate": float(left.mean()),
                    "right_run_alarm_rate": float(right.mean()),
                    "both_run_alarm_rate": float(both.mean()),
                    "either_run_alarm_rate": float(either.mean()),
                    "jaccard_run_alarm": (
                        float(both.sum() / either.sum()) if either.any() else np.nan
                    ),
                    "phi_run_alarm": binary_phi(left, right),
                }
            )
    coalarm_table = pd.DataFrame(coalarm_rows)

    summary = {
        "normal_bank_run_level_far": float(bank_run.mean()),
        "normal_bank_point_level_far": float(bank_alarm.mean()),
        "normal_bank_alarm_runs": int(bank_run.sum()),
        "normal_median_distinct_monitors_ever_alarming": float(np.median(distinct)),
        "normal_mean_distinct_monitors_ever_alarming": float(np.mean(distinct)),
        "normal_median_distinct_conditional_on_bank_alarm": (
            float(np.median(distinct[bank_run])) if bank_run.any() else 0.0
        ),
        "normal_maximum_distinct_monitors_ever_alarming": int(distinct.max()),
        "normal_median_maximum_simultaneous_monitors": float(
            np.median(max_simultaneous)
        ),
        "normal_maximum_simultaneous_monitors": int(max_simultaneous.max()),
    }
    return monitor_table, run_table, coalarm_table, summary


def target_run_records(
    actual_fault_id: int,
    first_matrix: np.ndarray,
    peak_matrix: np.ndarray,
    pre_alarm_stack: np.ndarray,
    post_alarm_stack: np.ndarray,
) -> list[dict[str, Any]]:
    target_index = RESERVED_FAULTS.index(actual_fault_id)
    bank_post_alarm = post_alarm_stack.any(axis=0)
    bank_first = first_alarm_samples(
        bank_post_alarm,
        np.arange(FAULT_START_SAMPLE, EXPECTED_SAMPLE_MAX + 1, dtype=np.int64),
    )
    target_first = first_matrix[target_index]
    target_peak = peak_matrix[target_index]
    target_delay = delay_from_first_sample(target_first)
    bank_delay = delay_from_first_sample(bank_first)
    bank_pre_alarm = pre_alarm_stack.any(axis=0)
    bank_pre_run_alarm = bank_pre_alarm.any(axis=1)
    distinct_pre = pre_alarm_stack.any(axis=2).sum(axis=0)
    distinct = post_alarm_stack.any(axis=2).sum(axis=0)
    maximum_simultaneous = post_alarm_stack.sum(axis=0).max(axis=1)

    rows = []
    for run_index, run_id in enumerate(EXPECTED_RUNS):
        current_first = first_matrix[:, run_index]
        current_peak = peak_matrix[:, run_index]
        finite = np.isfinite(current_first)
        minimum = np.min(current_first[finite]) if finite.any() else np.inf
        n_at_minimum = int(np.sum(current_first == minimum)) if np.isfinite(minimum) else 0
        target_finite = bool(np.isfinite(target_first[run_index]))
        target_earliest = bool(target_finite and target_first[run_index] == minimum)
        earliest_rank = (
            1 + int(np.sum(current_first < target_first[run_index]))
            if target_finite
            else np.nan
        )
        target_peak_value = float(target_peak[run_index])
        peak_rank = 1 + int(np.sum(current_peak > target_peak_value))
        max_peak = float(np.max(current_peak))
        peak_ties = int(np.sum(np.isclose(current_peak, max_peak, rtol=0.0, atol=1e-12)))
        rows.append(
            {
                "actual_fault_id": actual_fault_id,
                "simulationRun": int(run_id),
                "bank_prefault_run_alarm": bool(bank_pre_run_alarm[run_index]),
                "eligible_no_bank_prefault_alarm": bool(
                    not bank_pre_run_alarm[run_index]
                ),
                "distinct_monitors_alarming_prefault": int(
                    distinct_pre[run_index]
                ),
                "bank_detected": bool(np.isfinite(bank_first[run_index])),
                "bank_first_alarm_sample": (
                    float(bank_first[run_index])
                    if np.isfinite(bank_first[run_index])
                    else np.nan
                ),
                "bank_alarm_delay_min": (
                    float(bank_delay[run_index])
                    if np.isfinite(bank_delay[run_index])
                    else np.nan
                ),
                "target_monitor_detected": target_finite,
                "target_first_alarm_sample": (
                    float(target_first[run_index]) if target_finite else np.nan
                ),
                "target_alarm_delay_min": (
                    float(target_delay[run_index]) if target_finite else np.nan
                ),
                "target_among_earliest_including_ties": target_earliest,
                "target_uniquely_earliest": bool(target_earliest and n_at_minimum == 1),
                "target_earliest_rank": earliest_rank,
                "number_monitors_tied_at_earliest": n_at_minimum,
                "distinct_monitors_detecting_postfault": int(distinct[run_index]),
                "maximum_simultaneous_postfault_monitors": int(
                    maximum_simultaneous[run_index]
                ),
                "target_normalized_peak": target_peak_value,
                "largest_bank_normalized_peak": max_peak,
                "target_peak_margin_over_best_nontarget": float(
                    target_peak_value
                    - np.max(np.delete(current_peak, target_index))
                ),
                "target_has_largest_normalized_peak_including_ties": bool(
                    np.isclose(target_peak_value, max_peak, rtol=0.0, atol=1e-12)
                ),
                "target_uniquely_largest_normalized_peak": bool(
                    np.isclose(target_peak_value, max_peak, rtol=0.0, atol=1e-12)
                    and peak_ties == 1
                ),
                "target_normalized_peak_rank": int(peak_rank),
            }
        )
    return rows


def aggregate_target_diagnostics(target_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fault_id, current in target_run.groupby("actual_fault_id", sort=True):
        target_detected = current["target_monitor_detected"].astype(bool)
        bank_detected = current["bank_detected"].astype(bool)
        target_delay = current["target_alarm_delay_min"].to_numpy(dtype=float)
        bank_delay = current["bank_alarm_delay_min"].to_numpy(dtype=float)
        no_pre = current["eligible_no_bank_prefault_alarm"].astype(bool)

        def conditional_mean(column: str) -> float:
            return (
                float(current.loc[no_pre, column].mean())
                if no_pre.any()
                else np.nan
            )

        rows.append(
            {
                "actual_fault_id": int(fault_id),
                "n_runs": int(len(current)),
                "bank_run_detection_rate": float(bank_detected.mean()),
                "target_run_detection_rate": float(target_detected.mean()),
                "bank_prefault_run_alarm_rate": float(
                    current["bank_prefault_run_alarm"].mean()
                ),
                "n_runs_without_bank_prefault_alarm": int(no_pre.sum()),
                "bank_median_alarm_delay_min": float(np.nanmedian(bank_delay)),
                "target_median_alarm_delay_min": float(np.nanmedian(target_delay)),
                "target_among_earliest_rate": float(
                    current["target_among_earliest_including_ties"].mean()
                ),
                "target_uniquely_earliest_rate": float(
                    current["target_uniquely_earliest"].mean()
                ),
                "target_among_earliest_rate_no_bank_prefault_alarm": (
                    conditional_mean("target_among_earliest_including_ties")
                ),
                "target_uniquely_earliest_rate_no_bank_prefault_alarm": (
                    conditional_mean("target_uniquely_earliest")
                ),
                "median_target_earliest_rank": float(
                    current["target_earliest_rank"].median()
                ),
                "mean_target_earliest_rank": float(
                    current["target_earliest_rank"].mean()
                ),
                "target_largest_normalized_peak_rate": float(
                    current["target_has_largest_normalized_peak_including_ties"].mean()
                ),
                "target_largest_normalized_peak_rate_no_bank_prefault_alarm": (
                    conditional_mean(
                        "target_has_largest_normalized_peak_including_ties"
                    )
                ),
                "target_uniquely_largest_normalized_peak_rate": float(
                    current["target_uniquely_largest_normalized_peak"].mean()
                ),
                "median_target_normalized_peak_rank": float(
                    current["target_normalized_peak_rank"].median()
                ),
                "median_distinct_monitors_detecting_postfault": float(
                    current["distinct_monitors_detecting_postfault"].median()
                ),
                "median_maximum_simultaneous_postfault_monitors": float(
                    current["maximum_simultaneous_postfault_monitors"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_fault_id").reset_index(drop=True)


def plot_normal_bank(
    monitor_table: pd.DataFrame,
    run_table: pd.DataFrame,
    bank_far: float,
) -> list[Path]:
    faults = monitor_table["monitor_fault_id"].to_numpy(dtype=int)
    fars = 100.0 * monitor_table["normal_run_level_far"].to_numpy(dtype=float)
    distinct = run_table["distinct_monitors_ever_alarming"].to_numpy(dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.5), constrained_layout=True)
    axes[0].bar(np.arange(len(faults)), fars, color="#4C78A8", alpha=0.9)
    axes[0].axhline(
        100.0 * bank_far,
        color="#E45756",
        linestyle="--",
        linewidth=2.0,
        label=f"17-monitor OR bank: {100.0 * bank_far:.1f}%",
    )
    axes[0].set_xticks(np.arange(len(faults)), faults)
    axes[0].set_ylabel("Normal run-level FAR (%)")
    axes[0].set_title("Frozen monitor and hypothetical OR-bank false alarms")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.25)

    values, counts = np.unique(distinct, return_counts=True)
    axes[1].bar(values, counts, color="#72B7B2", width=0.8)
    axes[1].set_xticks(np.arange(0, len(RESERVED_FAULTS) + 1))
    axes[1].set_xlim(-0.6, len(RESERVED_FAULTS) + 0.6)
    axes[1].set_xlabel("Distinct fault-specific monitors alarming in a normal run")
    axes[1].set_ylabel("Number of normal runs")
    axes[1].set_title("Normal-run false-alarm multiplicity")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Step 25: frozen detector-bank normal-operation audit",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_PATHS["normal_png"], dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PATHS["normal_pdf"], bbox_inches="tight")
    plt.close(fig)
    return [OUTPUT_PATHS["normal_png"], OUTPUT_PATHS["normal_pdf"]]


def matrix_from_crossfault(crossfault: pd.DataFrame, value: str) -> np.ndarray:
    pivot = crossfault.pivot(
        index="monitor_fault_id", columns="actual_fault_id", values=value
    ).reindex(index=RESERVED_FAULTS, columns=RESERVED_FAULTS)
    if pivot.isna().any().any():
        raise ValueError(f"Cross-fault matrix {value} is incomplete")
    return pivot.to_numpy(dtype=float)


def plot_crossfault(crossfault: pd.DataFrame) -> list[Path]:
    detection = 100.0 * matrix_from_crossfault(
        crossfault, "post_fault_run_detection_rate"
    )
    delay = matrix_from_crossfault(crossfault, "allrun_penalized_mean_delay_min")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.7), constrained_layout=True)
    panels = [
        (axes[0], detection, "Run detection (%)", "viridis", 0.0, 100.0),
        (
            axes[1],
            delay,
            "All-run penalized mean delay (min)",
            "magma_r",
            0.0,
            float(np.nanmax(delay)),
        ),
    ]
    for axis, values, title, cmap, lower, upper in panels:
        image = axis.imshow(values, cmap=cmap, vmin=lower, vmax=max(upper, lower + 1e-9))
        axis.set_xticks(np.arange(len(RESERVED_FAULTS)), RESERVED_FAULTS, rotation=45)
        axis.set_yticks(np.arange(len(RESERVED_FAULTS)), RESERVED_FAULTS)
        axis.set_xlabel("Actual reserved TEP fault")
        axis.set_ylabel("Frozen monitor fault identity")
        axis.set_title(title)
        for diagonal in range(len(RESERVED_FAULTS)):
            axis.add_patch(
                plt.Rectangle(
                    (diagonal - 0.5, diagonal - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.1,
                )
            )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Frozen 17 × 17 cross-trigger audit (diagonal = target monitor)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_PATHS["crossfault_png"], dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PATHS["crossfault_pdf"], bbox_inches="tight")
    plt.close(fig)
    return [OUTPUT_PATHS["crossfault_png"], OUTPUT_PATHS["crossfault_pdf"]]


def plot_target_diagnostics(target_fault: pd.DataFrame) -> list[Path]:
    faults = target_fault["actual_fault_id"].to_numpy(dtype=int)
    x = np.arange(len(faults))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), constrained_layout=True)
    axes[0].plot(
        x,
        100.0 * target_fault["target_among_earliest_rate"],
        marker="o",
        color="#4C78A8",
        label="Target among earliest (ties allowed)",
    )
    axes[0].plot(
        x,
        100.0 * target_fault["target_largest_normalized_peak_rate"],
        marker="s",
        color="#E45756",
        label="Target has largest normalized peak (ties allowed)",
    )
    axes[0].set_ylim(-2, 102)
    axes[0].set_ylabel("Runs (%)")
    axes[0].set_title("Descriptive target-monitor prominence")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    axes[1].plot(
        x,
        target_fault["median_target_earliest_rank"],
        marker="o",
        color="#54A24B",
        label="Median first-alarm rank",
    )
    axes[1].plot(
        x,
        target_fault["median_target_normalized_peak_rank"],
        marker="s",
        color="#B279A2",
        label="Median normalized-peak rank",
    )
    axes[1].set_ylim(0.5, len(RESERVED_FAULTS) + 0.5)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Target monitor rank (1 = best)")
    axes[1].set_xlabel("Actual reserved TEP fault")
    axes[1].set_title("Target rank among 17 frozen monitors")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=True)
    for axis in axes:
        axis.set_xticks(x, faults)

    fig.suptitle(
        "Step 25 target-monitor diagnostics — not a trained diagnosis rule",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_PATHS["target_png"], dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PATHS["target_pdf"], bbox_inches="tight")
    plt.close(fig)
    return [OUTPUT_PATHS["target_png"], OUTPUT_PATHS["target_pdf"]]


def reporting_guardrails() -> pd.DataFrame:
    rows = [
        (
            "operating rule",
            "The OR bank is hypothetical and is not the Step-22 primary per-fault operating rule.",
            True,
        ),
        (
            "false alarms",
            "Use false-alarm terminology only for normal-operation results; use cross-trigger for real-fault rows.",
            True,
        ),
        (
            "diagnosis",
            "The cross-fault matrix and target ranks do not constitute a trained or validated multiclass diagnosis rule.",
            True,
        ),
        (
            "normalization",
            "CUSUM peaks are divided only by each monitor's already-frozen control limit.",
            True,
        ),
        (
            "retuning",
            "No Step-25 result may change a family, variable, sign, scaler, k, limit, or arbitration rule.",
            True,
        ),
        (
            "scope",
            "Conclusions are limited to the 17 reserved TEP fault sets and the evaluated frozen detector bank.",
            True,
        ),
        (
            "interpretation",
            "Report unfavorable bank FAR and weak selectivity as deployment limitations rather than hiding them.",
            True,
        ),
    ]
    return pd.DataFrame(rows, columns=["topic", "statement", "required"])


def run_analysis() -> None:
    registration = verify_specification()
    step24 = load_module(
        SCRIPT_DIR / "24_frozen_evidence_channel_robustness.py", "step24_for_step25"
    )
    dependencies = step24.load_dependencies()
    step11 = dependencies["step11"]
    step18 = dependencies["step18"]
    step20 = dependencies["step20"]
    frozen_results, parameters, scalers, _ = step24.load_frozen_state()

    variable_cols = step11.load_variable_columns()
    step24.validate_variable_order(variable_cols)

    print("=" * 96)
    print("Step 25: Frozen detector-bank false-alarm and cross-trigger audit")
    print("=" * 96)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print("All Step-22 frozen-state hashes passed.")
    print("No family, variable, sign, scaler, k, limit, or arbitration tuning is allowed.")

    normal_train = pd.read_pickle(NORMAL_TRAIN_PATH)
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_evaluation = step24.validate_normal_evaluation(
        pd.read_pickle(NORMAL_EVALUATION_PATH), variable_cols
    )
    normal_residuals, residual_cols = step11.compute_residual_frame(
        normal_evaluation, variable_cols, model
    )
    del normal_evaluation
    gc.collect()

    top5_parameters = parameters[
        parameters["method_key"].astype(str) == TOP5_KEY
    ].copy()
    top5_scalers = scalers[scalers["method_key"].astype(str) == TOP5_KEY].copy()
    top5_frozen = frozen_results[
        frozen_results["method_key"].astype(str) == TOP5_KEY
    ].copy()

    monitor_specs: dict[int, dict[str, Any]] = {}
    normal_alarm_list = []
    normal_stat_list = []
    parity_rows = []
    sample_ids_reference: np.ndarray | None = None

    print("Reconstructing all 17 frozen monitors on the common normal set...")
    for fault_id in RESERVED_FAULTS:
        scaler = top5_scalers[
            top5_scalers["fault_id"].astype(int) == fault_id
        ].sort_values("selection_rank")
        parameter = top5_parameters[
            top5_parameters["fault_id"].astype(int) == fault_id
        ].iloc[0]
        frozen = top5_frozen[
            top5_frozen["fault_id"].astype(int) == fault_id
        ].iloc[0]
        runs, sample_ids, components, metadata = step24.build_standardized_components(
            normal_residuals, scaler, step20
        )
        if not np.array_equal(runs, EXPECTED_RUNS):
            raise ValueError(f"Fault {fault_id}: normal run IDs changed")
        if sample_ids_reference is None:
            sample_ids_reference = sample_ids
        elif not np.array_equal(sample_ids_reference, sample_ids):
            raise ValueError("Normal sample grid differs between monitors")
        scalar = components.mean(axis=2)
        k = float(parameter["cusum_k"])
        limit = float(parameter["control_limit"])
        stat = step24.compute_cusum_matrix(scalar, k)
        alarm = stat > limit
        reconstructed_far = float(alarm.any(axis=1).mean())
        frozen_far = float(frozen["normal_run_level_far"])
        if not same_number(reconstructed_far, frozen_far):
            raise ValueError(
                f"Fault {fault_id}: normal FAR parity failed: "
                f"{reconstructed_far} versus {frozen_far}"
            )
        monitor_specs[fault_id] = {
            "scaler": scaler,
            "k": k,
            "limit": limit,
            "family": str(parameter["evidence_family"]),
            "variables": ";".join(item["variable"] for item in metadata),
        }
        normal_alarm_list.append(alarm)
        normal_stat_list.append(stat)
        parity_rows.append(
            {
                "fault_id": fault_id,
                "method_key": TOP5_KEY,
                "normal_run_level_far_reconstructed": reconstructed_far,
                "normal_run_level_far_step22": frozen_far,
                "normal_far_difference": reconstructed_far - frozen_far,
                "normal_parity_exact_with_tolerance_1e_12": True,
                "target_final_parity_checked": False,
            }
        )

    if sample_ids_reference is None:
        raise RuntimeError("No normal monitor was reconstructed")
    normal_alarm_stack = np.stack(normal_alarm_list, axis=0)
    normal_stat_stack = np.stack(normal_stat_list, axis=0)
    limits = np.array([monitor_specs[fault]["limit"] for fault in RESERVED_FAULTS])
    normal_monitor, normal_run, normal_coalarm, normal_summary = (
        build_normal_bank_tables(
            normal_alarm_stack, normal_stat_stack, sample_ids_reference, limits
        )
    )
    del normal_alarm_stack, normal_stat_stack, normal_alarm_list, normal_stat_list
    gc.collect()

    testing, testing_object, testing_names = step18.read_rdata_frame(
        RAW_TESTING_PATH, "faulty_testing"
    )
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    step24.require_columns(testing, required, "Faulty-Testing RData object")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    crossfault_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []

    for actual_fault in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"Actual Fault {actual_fault}: applying all 17 frozen monitors")
        print("=" * 96)
        final_mask = (fault_numbers == actual_fault) & run_numbers.between(251, 500)
        final = testing.loc[final_mask, required].copy()
        final, _ = step18.validate_fault_frame(
            final,
            variable_cols,
            actual_fault,
            EXPECTED_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{actual_fault}_step25_final",
        )
        final_residuals, current_residual_cols = step11.compute_residual_frame(
            final, variable_cols, model
        )
        if current_residual_cols != residual_cols:
            raise ValueError(f"Fault {actual_fault}: residual order changed")
        del final
        gc.collect()

        alarm_post_list = []
        alarm_pre_list = []
        first_list = []
        peak_list = []
        target_metric: dict[str, Any] | None = None
        for monitor_fault in RESERVED_FAULTS:
            spec = monitor_specs[monitor_fault]
            runs, sample_ids, components, _ = step24.build_standardized_components(
                final_residuals, spec["scaler"], step20
            )
            if not np.array_equal(runs, EXPECTED_RUNS):
                raise ValueError(
                    f"Actual Fault {actual_fault}, monitor {monitor_fault}: run IDs changed"
                )
            if not np.array_equal(sample_ids_reference, sample_ids):
                raise ValueError(
                    f"Actual Fault {actual_fault}, monitor {monitor_fault}: sample grid changed"
                )
            scalar = components.mean(axis=2)
            stat = step24.compute_cusum_matrix(scalar, spec["k"])
            alarm = stat > spec["limit"]
            metric = metric_from_alarm(alarm, stat, sample_ids, spec["limit"])
            alarm_pre_list.append(metric["pre_alarm"])
            alarm_post_list.append(metric["post_alarm"])
            first_list.append(metric["first_sample"])
            peak_list.append(metric["normalized_peak"])
            crossfault_rows.append(
                {
                    "monitor_fault_id": monitor_fault,
                    "actual_fault_id": actual_fault,
                    "target_monitor": monitor_fault == actual_fault,
                    "method_key": TOP5_KEY,
                    "evidence_family": spec["family"],
                    "selected_variables": spec["variables"],
                    "cusum_k": spec["k"],
                    "control_limit": spec["limit"],
                    "pre_fault_run_alarm_rate": metric["pre_fault_run_alarm_rate"],
                    "pre_fault_point_alarm_rate": metric["pre_fault_point_alarm_rate"],
                    "post_fault_run_detection_rate": metric[
                        "post_fault_run_detection_rate"
                    ],
                    "post_fault_point_detection_rate": metric[
                        "post_fault_point_detection_rate"
                    ],
                    "detected_runs": metric["detected_runs"],
                    "right_censored_runs": metric["right_censored_runs"],
                    "median_alarm_delay_min": metric["median_alarm_delay_min"],
                    "mean_alarm_delay_min": metric["mean_alarm_delay_min"],
                    "allrun_penalized_median_delay_min": metric[
                        "allrun_penalized_median_delay_min"
                    ],
                    "allrun_penalized_mean_delay_min": metric[
                        "allrun_penalized_mean_delay_min"
                    ],
                    "median_normalized_peak": metric["median_normalized_peak"],
                    "mean_normalized_peak": metric["mean_normalized_peak"],
                }
            )
            if monitor_fault == actual_fault:
                target_metric = metric
                frozen = top5_frozen[
                    top5_frozen["fault_id"].astype(int) == actual_fault
                ].iloc[0]
                parity_mapping = {
                    "post_fault_run_detection_rate": "fault_run_level_detection_rate",
                    "post_fault_point_detection_rate": "fault_post_fault_point_detection_rate",
                    "pre_fault_run_alarm_rate": "fault_pre_fault_run_alarm_rate",
                    "median_alarm_delay_min": "median_alarm_delay_min",
                    "mean_alarm_delay_min": "mean_alarm_delay_min",
                }
                differences = {}
                for reconstructed_name, frozen_name in parity_mapping.items():
                    left = float(metric[reconstructed_name])
                    right = float(frozen[frozen_name])
                    differences[reconstructed_name] = left - right
                    if not same_number(left, right):
                        raise ValueError(
                            f"Fault {actual_fault}: target-monitor parity failed for "
                            f"{reconstructed_name}: {left} versus {right}"
                        )
                parity_index = RESERVED_FAULTS.index(actual_fault)
                parity_rows[parity_index].update(
                    {
                        "target_final_parity_checked": True,
                        "target_final_maximum_metric_abs_difference": float(
                            np.nanmax(np.abs(list(differences.values())))
                        ),
                        "target_final_parity_exact_with_tolerance_1e_12": True,
                    }
                )

        if target_metric is None:
            raise RuntimeError(f"Fault {actual_fault}: target monitor missing")
        pre_alarm_stack = np.stack(alarm_pre_list, axis=0)
        post_alarm_stack = np.stack(alarm_post_list, axis=0)
        first_matrix = np.stack(first_list, axis=0)
        peak_matrix = np.stack(peak_list, axis=0)
        current_target_rows = target_run_records(
            actual_fault,
            first_matrix,
            peak_matrix,
            pre_alarm_stack,
            post_alarm_stack,
        )
        target_rows.extend(current_target_rows)

        bank_post = post_alarm_stack.any(axis=0)
        bank_detected = bank_post.any(axis=1)
        bank_first = first_alarm_samples(
            bank_post,
            np.arange(FAULT_START_SAMPLE, EXPECTED_SAMPLE_MAX + 1, dtype=np.int64),
        )
        bank_delay = delay_from_first_sample(bank_first)
        bank_penalized = np.where(bank_detected, bank_delay, float(MISS_PENALTY_MIN))
        current_target = pd.DataFrame(current_target_rows)
        bank_rows.append(
            {
                "actual_fault_id": actual_fault,
                "bank_rule": "OR of 17 frozen Top-5 monitors",
                "bank_prefault_run_alarm_rate": float(
                    pre_alarm_stack.any(axis=0).any(axis=1).mean()
                ),
                "bank_run_detection_rate": float(bank_detected.mean()),
                "bank_post_fault_point_detection_rate": float(bank_post.mean()),
                "bank_median_alarm_delay_min": float(np.nanmedian(bank_delay)),
                "bank_mean_alarm_delay_min": float(np.nanmean(bank_delay)),
                "bank_allrun_penalized_mean_delay_min": float(np.mean(bank_penalized)),
                "median_distinct_monitors_detecting_postfault": float(
                    current_target["distinct_monitors_detecting_postfault"].median()
                ),
                "median_maximum_simultaneous_postfault_monitors": float(
                    current_target["maximum_simultaneous_postfault_monitors"].median()
                ),
            }
        )
        print(
            f"  target DR={100*target_metric['post_fault_run_detection_rate']:.1f}%, "
            f"bank DR={100*bank_detected.mean():.1f}%, "
            f"median active monitors={current_target['distinct_monitors_detecting_postfault'].median():.1f}"
        )
        del final_residuals, pre_alarm_stack, post_alarm_stack, first_matrix, peak_matrix
        gc.collect()

    del testing, fault_numbers, run_numbers, normal_residuals
    gc.collect()

    clean_parity = pd.DataFrame(parity_rows).sort_values("fault_id")
    if len(clean_parity) != len(RESERVED_FAULTS):
        raise ValueError("Clean parity table is incomplete")
    if not clean_parity["normal_parity_exact_with_tolerance_1e_12"].all():
        raise ValueError("A normal parity check failed")
    if not clean_parity["target_final_parity_exact_with_tolerance_1e_12"].all():
        raise ValueError("A target final parity check failed")

    crossfault = pd.DataFrame(crossfault_rows).sort_values(
        ["monitor_fault_id", "actual_fault_id"]
    )
    if len(crossfault) != len(RESERVED_FAULTS) ** 2:
        raise ValueError("The 17 x 17 cross-fault matrix is incomplete")
    target_run = pd.DataFrame(target_rows).sort_values(
        ["actual_fault_id", "simulationRun"]
    )
    if len(target_run) != len(RESERVED_FAULTS) * len(EXPECTED_RUNS):
        raise ValueError("Target-monitor run diagnostics are incomplete")
    target_fault = aggregate_target_diagnostics(target_run)
    bank_fault = pd.DataFrame(bank_rows).sort_values("actual_fault_id")

    summary = pd.DataFrame(
        [
            {
                **normal_summary,
                "n_monitors": len(RESERVED_FAULTS),
                "n_actual_faults": len(RESERVED_FAULTS),
                "n_crossfault_cells": len(crossfault),
                "macro_mean_bank_run_detection": float(
                    bank_fault["bank_run_detection_rate"].mean()
                ),
                "macro_median_bank_alarm_delay_min": float(
                    bank_fault["bank_median_alarm_delay_min"].median()
                ),
                "macro_mean_target_among_earliest_rate": float(
                    target_fault["target_among_earliest_rate"].mean()
                ),
                "macro_mean_target_uniquely_earliest_rate": float(
                    target_fault["target_uniquely_earliest_rate"].mean()
                ),
                "macro_mean_target_among_earliest_rate_no_bank_prefault_alarm": float(
                    target_fault[
                        "target_among_earliest_rate_no_bank_prefault_alarm"
                    ].mean()
                ),
                "macro_median_target_earliest_rank": float(
                    target_fault["median_target_earliest_rank"].median()
                ),
                "macro_mean_target_largest_normalized_peak_rate": float(
                    target_fault["target_largest_normalized_peak_rate"].mean()
                ),
                "macro_median_target_normalized_peak_rank": float(
                    target_fault["median_target_normalized_peak_rank"].median()
                ),
                "interpretation": (
                    "post-freeze deployment/selectivity audit; not a trained diagnosis rule"
                ),
            }
        ]
    )
    guardrails = reporting_guardrails()

    tables = {
        "clean_parity": clean_parity,
        "normal_monitor": normal_monitor,
        "normal_run": normal_run,
        "normal_coalarm": normal_coalarm,
        "crossfault": crossfault,
        "target_run": target_run,
        "target_fault": target_fault,
        "bank_fault": bank_fault,
        "summary": summary,
        "guardrails": guardrails,
    }
    for key, table in tables.items():
        table.to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")

    figure_paths = []
    figure_paths.extend(
        plot_normal_bank(
            normal_monitor,
            normal_run,
            float(summary.iloc[0]["normal_bank_run_level_far"]),
        )
    )
    figure_paths.extend(plot_crossfault(crossfault))
    figure_paths.extend(plot_target_diagnostics(target_fault))

    config = {
        "analysis_name": "Step 25 frozen detector-bank system audit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_script_sha256": registration["script_sha256"],
        "registered_input_sha256": registration["input_sha256"],
        "all_17_normal_parity_checks_passed": True,
        "all_17_target_final_parity_checks_passed": True,
        "monitor_faults": RESERVED_FAULTS,
        "actual_faults": RESERVED_FAULTS,
        "method_key": TOP5_KEY,
        "bank_rule": "logical OR of 17 frozen monitor alarms",
        "diagnosis_rule_trained": False,
        "final_data_used_for_tuning": False,
        "summary": summary.iloc[0].to_dict(),
        "outputs": {
            key: str(path)
            for key, path in OUTPUT_PATHS.items()
        },
        "output_sha256": {
            key: sha256_file(path)
            for key, path in OUTPUT_PATHS.items()
            if path.exists()
        },
        "rdata_object": testing_object,
        "rdata_object_names": testing_names,
    }
    FINAL_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    row = summary.iloc[0]
    print("\n" + "=" * 96)
    print("Step 25 summary")
    print("=" * 96)
    print(
        f"Hypothetical 17-monitor OR-bank normal run FAR: "
        f"{100*row['normal_bank_run_level_far']:.2f}%"
    )
    print(
        f"Normal point-level OR FAR: {100*row['normal_bank_point_level_far']:.3f}%"
    )
    print(
        f"Macro mean bank final run detection: "
        f"{100*row['macro_mean_bank_run_detection']:.2f}%"
    )
    print(
        f"Macro mean target-among-earliest rate: "
        f"{100*row['macro_mean_target_among_earliest_rate']:.2f}%"
    )
    print(
        f"Macro median target first-alarm rank: "
        f"{row['macro_median_target_earliest_rank']:.2f} of 17"
    )
    print("Saved tables:")
    for key in tables:
        print(f"  {OUTPUT_PATHS[key]}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Saved configuration: {FINAL_CONFIG_PATH}")
    print("Step 25 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="register the immutable post-freeze detector-bank audit",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify the registration and run the frozen audit",
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
