"""Step 26: Post-freeze joint detector-bank gate.

Scientific role
---------------
Step 25 showed that the logical OR of 17 individually frozen Top-5 monitors
has strong fault coverage but an excessive normal run-level false-alarm rate.
Step 26 is a post-freeze, secondary system-engineering study.  It adds one
bank-level gate without changing any fault-specific evidence family, variable,
orientation, scaler, CUSUM reference value, or local control limit.

For monitor j, let C_j(t) be its frozen CUSUM and h_j its frozen local limit.
The registered bank score is

    G(t) = max_j C_j(t) / h_j.

The single bank threshold H_G is the empirical ``higher`` 95th percentile of
the per-run maximum of G(t) on normal-validation runs 1--250.  A strict
``G(t) > H_G`` rule is then evaluated once on normal-evaluation runs 251--500
and on the final 250 runs of each of the 17 reserved TEP faults.  Final fault
data and normal-evaluation data are not used to choose or alter H_G.

This analysis is not a prospective confirmation: Step-22 and Step-25 results
were already known.  It is an explicitly labelled exploratory deployment
layer.  Local monitor rankings at the first bank alarm are descriptive
attribution diagnostics, not a validated fault classifier.

Two-stage use
-------------
Place this file in the project ``scripts`` directory, then run:

    python scripts/26_postfreeze_joint_bank_gate.py --register
    python scripts/26_postfreeze_joint_bank_gate.py --run
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
from scipy.stats import beta


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
VALIDATION_RUNS = np.arange(1, 251, dtype=np.int64)
EVALUATION_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (
    EXPECTED_SAMPLE_MAX - FAULT_START_SAMPLE + 1
) * SAMPLING_INTERVAL_MIN

TOP5_KEY = "frozen_multievidence_top5_cusum"
TOP5_LABEL = "Proposed Top-5"
N_MONITORS = len(RESERVED_FAULTS)
JOINT_GATE_ALPHA = 0.05
JOINT_GATE_QUANTILE = 1.0 - JOINT_GATE_ALPHA
QUANTILE_METHOD = "higher"
PARITY_TOLERANCE = 1e-12
RANK_TOLERANCE = 1e-12

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"

STEP22_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
STEP22_PARAMETERS_PATH = TABLE_DIR / "22_reserved_multifault_method_parameters.csv"
STEP22_SCALERS_PATH = TABLE_DIR / "22_reserved_multifault_local_scalers.csv"
STEP22_CONFIG_PATH = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"

STEP25_PARITY_PATH = TABLE_DIR / "25_monitor_clean_reconstruction_parity.csv"
STEP25_NORMAL_MONITOR_PATH = TABLE_DIR / "25_normal_monitor_operating_points.csv"
STEP25_CROSSFAULT_PATH = TABLE_DIR / "25_crossfault_alarm_matrix.csv"
STEP25_BANK_FAULT_PATH = TABLE_DIR / "25_bank_detection_by_fault.csv"
STEP25_SUMMARY_PATH = TABLE_DIR / "25_detector_bank_summary.csv"
STEP25_CONFIG_PATH = CONFIG_DIR / "25_frozen_detector_bank_audit.json"

SPECIFICATION_PATH = CONFIG_DIR / "26_postfreeze_joint_bank_gate_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "26_postfreeze_joint_bank_gate.json"

OUTPUT_PATHS = {
    "normal_runs": TABLE_DIR / "26_joint_gate_normal_evaluation_runs.csv",
    "fault_runs": TABLE_DIR / "26_joint_gate_fault_run_attribution.csv",
    "by_fault": TABLE_DIR / "26_joint_gate_by_fault.csv",
    "comparison": TABLE_DIR / "26_joint_gate_comparison_with_step25.csv",
    "summary": TABLE_DIR / "26_joint_gate_summary.csv",
    "guardrails": TABLE_DIR / "26_reporting_guardrails.csv",
    "operating_png": FIGURE_DIR / "fig26a_joint_gate_operating_point.png",
    "operating_pdf": FIGURE_DIR / "fig26a_joint_gate_operating_point.pdf",
    "fault_png": FIGURE_DIR / "fig26b_joint_gate_detection_by_fault.png",
    "fault_pdf": FIGURE_DIR / "fig26b_joint_gate_detection_by_fault.pdf",
    "rank_png": FIGURE_DIR / "fig26c_joint_gate_attribution_rank.png",
    "rank_pdf": FIGURE_DIR / "fig26c_joint_gate_attribution_rank.pdf",
}


# Registration reads byte hashes only.  No process dataframe or RData object is
# parsed until --run has verified the immutable specification.
REQUIRED_INPUTS = {
    "step11_script": SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
    "step18_script": SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
    "step20_script": SCRIPT_DIR / "20_frozen_multievidence_validation.py",
    "step24_script": SCRIPT_DIR / "24_frozen_evidence_channel_robustness.py",
    "step25_script": SCRIPT_DIR / "25_frozen_detector_bank_audit.py",
    "step22_results": STEP22_RESULTS_PATH,
    "step22_parameters": STEP22_PARAMETERS_PATH,
    "step22_scalers": STEP22_SCALERS_PATH,
    "step22_config": STEP22_CONFIG_PATH,
    "step25_parity": STEP25_PARITY_PATH,
    "step25_normal_monitor": STEP25_NORMAL_MONITOR_PATH,
    "step25_crossfault": STEP25_CROSSFAULT_PATH,
    "step25_bank_fault": STEP25_BANK_FAULT_PATH,
    "step25_summary": STEP25_SUMMARY_PATH,
    "step25_config": STEP25_CONFIG_PATH,
    "variable_list": TABLE_DIR / "02_variable_list.csv",
    "normal_train": NORMAL_TRAIN_PATH,
    "normal_validation": NORMAL_VALIDATION_PATH,
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
        "registration_name": "Step 26 post-freeze joint detector-bank gate",
        "registration_status": (
            "post-freeze exploratory deployment study; Step-22 and Step-25 "
            "results were known before registration"
        ),
        "script": "scripts/26_postfreeze_joint_bank_gate.py",
        "script_sha256": script_hash,
        "input_sha256": hashes,
        "registration_reads": "byte hashes only; no process dataframe is parsed",
        "process_data_loaded_during_registration": False,
        "frozen_population": {
            "monitor_faults": RESERVED_FAULTS,
            "actual_faults": RESERVED_FAULTS,
            "normal_validation_runs": [1, 250],
            "normal_evaluation_runs": [251, 500],
            "final_fault_runs": [251, 500],
            "samples_per_run": EXPECTED_SAMPLE_MAX,
            "fault_start_sample": FAULT_START_SAMPLE,
            "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        },
        "unchanged_local_monitors": {
            "method_key": TOP5_KEY,
            "monitor_count": N_MONITORS,
            "evidence_family_change_allowed": False,
            "variable_change_allowed": False,
            "orientation_change_allowed": False,
            "scaler_refitting_allowed": False,
            "cusum_k_change_allowed": False,
            "local_control_limit_change_allowed": False,
        },
        "single_registered_system_gate": {
            "sample_score": "max_j frozen_CUSUM_j / frozen_local_limit_j",
            "run_calibration_score": "maximum sample score within each run",
            "calibration_data": "normal-validation runs 1--250 only",
            "quantile": JOINT_GATE_QUANTILE,
            "quantile_method": QUANTILE_METHOD,
            "alarm_inequality": "strictly greater than the calibrated threshold",
            "normal_evaluation_used_for_calibration": False,
            "fault_development_used_for_calibration": False,
            "fault_final_used_for_calibration": False,
            "alternative_gate_selection_after_final_allowed": False,
        },
        "registered_endpoints": {
            "primary": [
                "normal-evaluation joint-gate run-level false-alarm rate",
                "final joint-gate run-level detection by fault",
                "all-run penalized mean alarm delay by fault",
            ],
            "secondary": [
                "normal-evaluation point-level false-alarm rate",
                "post-fault point detection",
                "target monitor top-1 and top-3 rank at first joint-gate alarm",
            ],
            "miss_penalty_min": MISS_PENALTY_MIN,
        },
        "reporting_guardrails": [
            "Call Step 26 an exploratory post-freeze deployment layer, not a prospective confirmation.",
            "Do not describe the joint gate as an unchanged Step-22 primary method.",
            "Do not alter the registered gate after viewing normal-evaluation or final-fault results.",
            "Report the normal-evaluation FAR even if the 5% validation target does not transfer.",
            "Report every reserved fault, including misses and long delays.",
            "Monitor rank at gate time is descriptive attribution, not validated multiclass diagnosis.",
            "Retain Step-25 OR-bank results as the uncalibrated deployment comparator.",
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
            "faults": existing.get("frozen_population", {}).get("actual_faults")
            == RESERVED_FAULTS,
            "gate score": existing.get("single_registered_system_gate", {}).get(
                "sample_score"
            )
            == "max_j frozen_CUSUM_j / frozen_local_limit_j",
            "quantile": existing.get("single_registered_system_gate", {}).get(
                "quantile"
            )
            == JOINT_GATE_QUANTILE,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Existing Step-26 specification differs from the current "
                f"script or frozen inputs: {failed}. Preserve it and investigate "
                "instead of overwriting it."
            )
        print("Step 26 specification already exists and matches all frozen files.")
        print(f"Registered script SHA-256: {script_hash}")
        print(f"Specification: {SPECIFICATION_PATH}")
        return

    SPECIFICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 26 post-freeze joint-gate specification registered.")
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
        "faults": specification.get("frozen_population", {}).get("actual_faults")
        == RESERVED_FAULTS,
        "normal evaluation not calibrated": specification.get(
            "single_registered_system_gate", {}
        ).get("normal_evaluation_used_for_calibration")
        is False,
        "final not calibrated": specification.get(
            "single_registered_system_gate", {}
        ).get("fault_final_used_for_calibration")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-26 specification verification failed: {failed}")
    return specification


def load_module(path: Path, name: str):
    require_file(path, f"load frozen dependency {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def same_number(left: Any, right: Any, tolerance: float = PARITY_TOLERANCE) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=tolerance))


def first_alarm_samples(alarm: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    detected = alarm.any(axis=1)
    output = np.full(alarm.shape[0], np.nan, dtype=float)
    if detected.any():
        first_index = alarm[detected].argmax(axis=1)
        output[detected] = sample_ids[first_index]
    return output


def alarm_delay(first_sample: np.ndarray) -> np.ndarray:
    output = np.full(len(first_sample), np.nan, dtype=float)
    detected = np.isfinite(first_sample)
    output[detected] = (
        first_sample[detected] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN
    return output


def build_monitor_specs(
    parameters: pd.DataFrame,
    scalers: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    top5_parameters = parameters[
        parameters["method_key"].astype(str) == TOP5_KEY
    ].copy()
    top5_scalers = scalers[
        scalers["method_key"].astype(str) == TOP5_KEY
    ].copy()
    specs: dict[int, dict[str, Any]] = {}
    for fault_id in RESERVED_FAULTS:
        parameter_rows = top5_parameters[
            top5_parameters["fault_id"].astype(int) == fault_id
        ]
        scaler = top5_scalers[
            top5_scalers["fault_id"].astype(int) == fault_id
        ].sort_values("selection_rank")
        if len(parameter_rows) != 1 or len(scaler) != 5:
            raise ValueError(f"Fault {fault_id}: incomplete frozen Top-5 state")
        parameter = parameter_rows.iloc[0]
        limit = float(parameter["control_limit"])
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(f"Fault {fault_id}: invalid frozen local limit")
        specs[fault_id] = {
            "scaler": scaler,
            "k": float(parameter["cusum_k"]),
            "limit": limit,
            "family": str(parameter["evidence_family"]),
            "normal_validation_run_far": float(
                parameter["normal_validation_run_level_far"]
            ),
        }
    return specs


def reconstruct_normalized_scores(
    residuals: pd.DataFrame,
    monitor_specs: dict[int, dict[str, Any]],
    step20: Any,
    step24: Any,
    expected_runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_list = []
    run_reference: np.ndarray | None = None
    sample_reference: np.ndarray | None = None
    for fault_id in RESERVED_FAULTS:
        spec = monitor_specs[fault_id]
        runs, sample_ids, components, _ = step24.build_standardized_components(
            residuals, spec["scaler"], step20
        )
        if not np.array_equal(runs, expected_runs):
            raise ValueError(f"Monitor {fault_id}: unexpected run IDs")
        if run_reference is None:
            run_reference = runs
            sample_reference = sample_ids
        elif not np.array_equal(run_reference, runs) or not np.array_equal(
            sample_reference, sample_ids
        ):
            raise ValueError("Monitor score grids differ")
        scalar = components.mean(axis=2)
        statistic = step24.compute_cusum_matrix(scalar, spec["k"])
        score = statistic / spec["limit"]
        if not np.isfinite(score).all():
            raise ValueError(f"Monitor {fault_id}: non-finite normalized score")
        score_list.append(score)
    if run_reference is None or sample_reference is None:
        raise RuntimeError("No monitor scores were reconstructed")
    return run_reference, sample_reference, np.stack(score_list, axis=0)


def calibrate_joint_threshold(score_stack: np.ndarray) -> tuple[float, pd.DataFrame]:
    bank_score = score_stack.max(axis=0)
    run_max = bank_score.max(axis=1)
    threshold = float(
        np.quantile(run_max, JOINT_GATE_QUANTILE, method=QUANTILE_METHOD)
    )
    alarms = run_max > threshold
    if float(alarms.mean()) > JOINT_GATE_ALPHA + 1e-12:
        raise ValueError("Registered higher-quantile rule exceeded validation alpha")
    calibration = pd.DataFrame(
        {
            "simulationRun": VALIDATION_RUNS,
            "joint_gate_run_max_score": run_max,
            "joint_gate_alarm": alarms,
        }
    )
    return threshold, calibration


def normal_gate_table(
    score_stack: np.ndarray,
    runs: np.ndarray,
    sample_ids: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    bank_score = score_stack.max(axis=0)
    alarm = bank_score > threshold
    run_alarm = alarm.any(axis=1)
    first = first_alarm_samples(alarm, sample_ids)
    run_max = bank_score.max(axis=1)
    low, high = exact_binomial_interval(int(run_alarm.sum()), len(run_alarm))
    table = pd.DataFrame(
        {
            "simulationRun": runs,
            "joint_gate_run_max_score": run_max,
            "joint_gate_alarm": run_alarm,
            "joint_gate_first_alarm_sample": first,
            "maximum_simultaneous_local_alarms": (score_stack > 1.0).sum(axis=0).max(axis=1),
        }
    )
    summary = {
        "normal_evaluation_alarm_runs": int(run_alarm.sum()),
        "normal_evaluation_run_far": float(run_alarm.mean()),
        "normal_evaluation_run_far_ci_low": low,
        "normal_evaluation_run_far_ci_high": high,
        "normal_evaluation_point_far": float(alarm.mean()),
    }
    return table, summary


def verify_normal_step25_parity(
    score_stack: np.ndarray,
    step25_normal: pd.DataFrame,
) -> float:
    differences = []
    for monitor_index, fault_id in enumerate(RESERVED_FAULTS):
        reconstructed = float((score_stack[monitor_index] > 1.0).any(axis=1).mean())
        row = step25_normal[
            step25_normal["monitor_fault_id"].astype(int) == fault_id
        ]
        if len(row) != 1:
            raise ValueError(f"Step-25 normal monitor row missing for fault {fault_id}")
        frozen = float(row.iloc[0]["normal_run_level_far"])
        differences.append(reconstructed - frozen)
        if not same_number(reconstructed, frozen):
            raise ValueError(
                f"Fault {fault_id}: Step-25 normal parity failed: "
                f"{reconstructed} versus {frozen}"
            )
    return float(np.max(np.abs(differences)))


def verify_validation_local_far(
    score_stack: np.ndarray,
    monitor_specs: dict[int, dict[str, Any]],
) -> float:
    differences = []
    for monitor_index, fault_id in enumerate(RESERVED_FAULTS):
        reconstructed = float((score_stack[monitor_index] > 1.0).any(axis=1).mean())
        frozen = float(monitor_specs[fault_id]["normal_validation_run_far"])
        differences.append(reconstructed - frozen)
        if not same_number(reconstructed, frozen):
            raise ValueError(
                f"Fault {fault_id}: normal-validation local FAR parity failed: "
                f"{reconstructed} versus {frozen}"
            )
    return float(np.max(np.abs(differences)))


def verify_crossfault_step25_parity(
    score_stack: np.ndarray,
    sample_ids: np.ndarray,
    actual_fault: int,
    step25_crossfault: pd.DataFrame,
) -> float:
    post_mask = sample_ids >= FAULT_START_SAMPLE
    differences = []
    for monitor_index, monitor_fault in enumerate(RESERVED_FAULTS):
        alarm = score_stack[monitor_index][:, post_mask] > 1.0
        run_detection = float(alarm.any(axis=1).mean())
        point_detection = float(alarm.mean())
        row = step25_crossfault[
            (step25_crossfault["monitor_fault_id"].astype(int) == monitor_fault)
            & (step25_crossfault["actual_fault_id"].astype(int) == actual_fault)
        ]
        if len(row) != 1:
            raise ValueError(
                f"Step-25 cross-fault row missing: monitor {monitor_fault}, "
                f"actual fault {actual_fault}"
            )
        frozen = row.iloc[0]
        current_differences = [
            run_detection - float(frozen["post_fault_run_detection_rate"]),
            point_detection - float(frozen["post_fault_point_detection_rate"]),
        ]
        differences.extend(current_differences)
        if not all(abs(value) <= PARITY_TOLERANCE for value in current_differences):
            raise ValueError(
                f"Step-25 cross-fault parity failed: monitor {monitor_fault}, "
                f"actual fault {actual_fault}"
            )
    return float(np.max(np.abs(differences)))


def evaluate_fault_gate(
    score_stack: np.ndarray,
    runs: np.ndarray,
    sample_ids: np.ndarray,
    threshold: float,
    actual_fault: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pre_mask = sample_ids < FAULT_START_SAMPLE
    post_mask = sample_ids >= FAULT_START_SAMPLE
    post_samples = sample_ids[post_mask]
    bank_score = score_stack.max(axis=0)
    pre_alarm = bank_score[:, pre_mask] > threshold
    post_alarm = bank_score[:, post_mask] > threshold
    pre_run_alarm = pre_alarm.any(axis=1)
    detected = post_alarm.any(axis=1)
    first = first_alarm_samples(post_alarm, post_samples)
    delay = alarm_delay(first)
    penalized = np.where(detected, delay, float(MISS_PENALTY_MIN))

    target_index = RESERVED_FAULTS.index(actual_fault)
    run_rows = []
    for run_index, run_id in enumerate(runs):
        if detected[run_index]:
            first_position = int(np.flatnonzero(post_alarm[run_index])[0])
            absolute_position = int(np.flatnonzero(post_mask)[first_position])
            values = score_stack[:, run_index, absolute_position]
            target_value = float(values[target_index])
            rank = 1 + int(np.sum(values > target_value + RANK_TOLERANCE))
            best = float(values.max())
            top1 = bool(target_value >= best - RANK_TOLERANCE)
            winner_count = int(np.sum(values >= best - RANK_TOLERANCE))
            winning_faults = ";".join(
                str(RESERVED_FAULTS[index])
                for index in np.flatnonzero(values >= best - RANK_TOLERANCE)
            )
        else:
            target_value = float("nan")
            rank = np.nan
            top1 = False
            winner_count = 0
            winning_faults = ""
        run_rows.append(
            {
                "actual_fault_id": actual_fault,
                "simulationRun": int(run_id),
                "joint_gate_prefault_alarm": bool(pre_run_alarm[run_index]),
                "joint_gate_detected": bool(detected[run_index]),
                "joint_gate_first_alarm_sample": float(first[run_index]),
                "joint_gate_alarm_delay_min": float(delay[run_index]),
                "target_monitor_score_at_gate": target_value,
                "target_monitor_rank_at_gate": rank,
                "target_monitor_top1_at_gate": top1,
                "target_monitor_top3_at_gate": bool(
                    np.isfinite(rank) and rank <= 3
                ),
                "winner_tie_count_at_gate": winner_count,
                "winning_monitor_faults_at_gate": winning_faults,
            }
        )

    run_table = pd.DataFrame(run_rows)
    detected_ranks = run_table.loc[
        run_table["joint_gate_detected"], "target_monitor_rank_at_gate"
    ].to_numpy(dtype=float)
    low, high = exact_binomial_interval(int(detected.sum()), len(detected))
    summary = {
        "actual_fault_id": actual_fault,
        "joint_gate_threshold": threshold,
        "prefault_run_alarm_rate": float(pre_run_alarm.mean()),
        "detected_runs": int(detected.sum()),
        "right_censored_runs": int((~detected).sum()),
        "run_detection_rate": float(detected.mean()),
        "run_detection_ci_low": low,
        "run_detection_ci_high": high,
        "post_fault_point_detection_rate": float(post_alarm.mean()),
        "median_alarm_delay_min": float(np.nanmedian(delay)) if detected.any() else float("nan"),
        "mean_alarm_delay_min": float(np.nanmean(delay)) if detected.any() else float("nan"),
        "allrun_penalized_mean_delay_min": float(np.mean(penalized)),
        "target_top1_at_gate_rate_all_runs": float(
            run_table["target_monitor_top1_at_gate"].mean()
        ),
        "target_top3_at_gate_rate_all_runs": float(
            run_table["target_monitor_top3_at_gate"].mean()
        ),
        "median_target_rank_at_gate_detected_runs": float(
            np.nanmedian(detected_ranks)
        )
        if len(detected_ranks)
        else float("nan"),
    }
    return run_table, summary


def reporting_guardrails() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topic": "analysis status",
                "required_language": "exploratory post-freeze deployment layer",
                "prohibited_language": "prospectively confirmed final method",
            },
            {
                "topic": "local detectors",
                "required_language": "all 17 local states remain exactly frozen",
                "prohibited_language": "jointly retuned local monitors",
            },
            {
                "topic": "system threshold",
                "required_language": "one normal-validation-calibrated joint gate",
                "prohibited_language": "unchanged Step-22 threshold",
            },
            {
                "topic": "normal performance",
                "required_language": "report normal-evaluation FAR with exact 95% CI",
                "prohibited_language": "guaranteed 5% deployment FAR",
            },
            {
                "topic": "attribution",
                "required_language": "descriptive target-monitor rank at gate time",
                "prohibited_language": "validated fault diagnosis accuracy",
            },
            {
                "topic": "Step-25 comparator",
                "required_language": "uncalibrated OR bank retained as a limitation audit",
                "prohibited_language": "Step-25 invalid or discarded",
            },
        ]
    )


def plot_operating_point(summary: pd.Series) -> list[Path]:
    labels = ["Step 25\nOR bank", "Step 26\njoint gate"]
    far = [
        100 * float(summary["step25_or_bank_normal_run_far"]),
        100 * float(summary["normal_evaluation_run_far"]),
    ]
    detection = [
        100 * float(summary["step25_macro_mean_bank_run_detection"]),
        100 * float(summary["macro_mean_run_detection"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    colors = ["#9b9b9b", "#d95f02"]
    axes[0].bar(labels, far, color=colors)
    axes[0].axhline(5.0, color="#1b9e77", linestyle="--", linewidth=1.5, label="5% target")
    axes[0].set_ylabel("Normal run-level FAR (%)")
    axes[0].set_title("System false-alarm burden")
    axes[0].legend(frameon=False)
    axes[1].bar(labels, detection, color=colors)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Macro mean run detection (%)")
    axes[1].set_title("Reserved-fault detection")
    for axis, values in zip(axes, [far, detection]):
        for index, value in enumerate(values):
            axis.text(index, value + 1.2, f"{value:.1f}%", ha="center", va="bottom")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Uncalibrated OR bank versus normal-validation-calibrated joint gate")
    fig.tight_layout()
    paths = [OUTPUT_PATHS["operating_png"], OUTPUT_PATHS["operating_pdf"]]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_fault_results(by_fault: pd.DataFrame) -> list[Path]:
    frame = by_fault.sort_values("actual_fault_id")
    x = np.arange(len(frame))
    labels = frame["actual_fault_id"].astype(int).astype(str).tolist()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    axes[0].plot(
        x,
        100 * frame["run_detection_rate"],
        marker="o",
        color="#d95f02",
        label="Step 26 joint gate",
    )
    axes[0].plot(
        x,
        100 * frame["step25_or_bank_run_detection_rate"],
        marker="s",
        color="#7570b3",
        linestyle="--",
        label="Step 25 OR bank",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Run detection (%)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(
        x,
        frame["allrun_penalized_mean_delay_min"],
        marker="o",
        color="#d95f02",
        label="Step 26 joint gate",
    )
    axes[1].plot(
        x,
        frame["step25_or_bank_allrun_penalized_mean_delay_min"],
        marker="s",
        color="#7570b3",
        linestyle="--",
        label="Step 25 OR bank",
    )
    axes[1].set_ylabel("All-run penalized mean delay (min)")
    axes[1].set_xlabel("Reserved TEP fault")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Joint-gate fault coverage and delay after system-level calibration")
    fig.tight_layout()
    paths = [OUTPUT_PATHS["fault_png"], OUTPUT_PATHS["fault_pdf"]]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_ranks(by_fault: pd.DataFrame) -> list[Path]:
    frame = by_fault.sort_values("actual_fault_id")
    x = np.arange(len(frame))
    labels = frame["actual_fault_id"].astype(int).astype(str).tolist()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
    axes[0].plot(
        x,
        100 * frame["target_top1_at_gate_rate_all_runs"],
        marker="o",
        label="Target is top-1",
        color="#1b9e77",
    )
    axes[0].plot(
        x,
        100 * frame["target_top3_at_gate_rate_all_runs"],
        marker="s",
        label="Target is top-3",
        color="#7570b3",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("All-run attribution rate (%)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(
        x,
        frame["median_target_rank_at_gate_detected_runs"],
        marker="o",
        color="#d95f02",
    )
    axes[1].axhline(3.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylim(17.5, 0.5)
    axes[1].set_ylabel("Median target rank at gate")
    axes[1].set_xlabel("Reserved TEP fault")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Descriptive local-monitor attribution at first joint-gate alarm")
    fig.tight_layout()
    paths = [OUTPUT_PATHS["rank_png"], OUTPUT_PATHS["rank_pdf"]]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def run_analysis() -> None:
    registration = verify_specification()
    step24 = load_module(
        SCRIPT_DIR / "24_frozen_evidence_channel_robustness.py",
        "step24_for_step26",
    )
    dependencies = step24.load_dependencies()
    step11 = dependencies["step11"]
    step18 = dependencies["step18"]
    step20 = dependencies["step20"]
    _, parameters, scalers, _ = step24.load_frozen_state()

    variable_cols = step11.load_variable_columns()
    step24.validate_variable_order(variable_cols)
    monitor_specs = build_monitor_specs(parameters, scalers)

    step25_parity = pd.read_csv(STEP25_PARITY_PATH, encoding="utf-8-sig")
    normal_parity_flags = (
        step25_parity["normal_parity_exact_with_tolerance_1e_12"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    target_parity_flags = (
        step25_parity["target_final_parity_exact_with_tolerance_1e_12"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    if (
        len(step25_parity) != N_MONITORS
        or not normal_parity_flags.all()
        or not target_parity_flags.all()
    ):
        raise ValueError("Step-25 parity audit is incomplete or failed")
    step25_normal = pd.read_csv(STEP25_NORMAL_MONITOR_PATH, encoding="utf-8-sig")
    step25_crossfault = pd.read_csv(STEP25_CROSSFAULT_PATH, encoding="utf-8-sig")
    step25_bank_fault = pd.read_csv(STEP25_BANK_FAULT_PATH, encoding="utf-8-sig")
    step25_summary = pd.read_csv(STEP25_SUMMARY_PATH, encoding="utf-8-sig").iloc[0]

    print("=" * 96)
    print("Step 26: Post-freeze joint detector-bank gate")
    print("=" * 96)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print("All Step-22 and Step-25 hashes and parity checks passed.")
    print("No local family, variable, sign, scaler, k, or limit may change.")
    print(
        "One joint threshold will be calibrated on normal-validation runs only; "
        "no final or normal-evaluation result may retune it."
    )

    normal_train = pd.read_pickle(NORMAL_TRAIN_PATH)
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    print("Reconstructing 17 frozen monitors on normal validation...")
    normal_validation = step20.validate_normal_frame(
        pd.read_pickle(NORMAL_VALIDATION_PATH),
        variable_cols,
        "normal_validation_step26",
        VALIDATION_RUNS,
    )
    validation_residuals, residual_cols = step11.compute_residual_frame(
        normal_validation, variable_cols, model
    )
    del normal_validation
    validation_runs, validation_samples, validation_scores = (
        reconstruct_normalized_scores(
            validation_residuals,
            monitor_specs,
            step20,
            step24,
            VALIDATION_RUNS,
        )
    )
    del validation_residuals
    validation_local_parity_difference = verify_validation_local_far(
        validation_scores, monitor_specs
    )
    joint_threshold, calibration = calibrate_joint_threshold(validation_scores)
    validation_gate_alarm = calibration["joint_gate_alarm"].to_numpy(dtype=bool)
    validation_point_far = float(
        (validation_scores.max(axis=0) > joint_threshold).mean()
    )
    print(
        f"Frozen joint-gate threshold H_G={joint_threshold:.6f}; "
        f"validation run FAR={100*validation_gate_alarm.mean():.2f}%"
    )
    del validation_scores
    gc.collect()

    print("Evaluating the frozen joint gate on normal evaluation...")
    normal_evaluation = step24.validate_normal_evaluation(
        pd.read_pickle(NORMAL_EVALUATION_PATH), variable_cols
    )
    evaluation_residuals, current_residual_cols = step11.compute_residual_frame(
        normal_evaluation, variable_cols, model
    )
    del normal_evaluation
    if current_residual_cols != residual_cols:
        raise ValueError("Normal-evaluation residual order changed")
    evaluation_runs, evaluation_samples, evaluation_scores = (
        reconstruct_normalized_scores(
            evaluation_residuals,
            monitor_specs,
            step20,
            step24,
            EVALUATION_RUNS,
        )
    )
    del evaluation_residuals
    normal_parity_difference = verify_normal_step25_parity(
        evaluation_scores, step25_normal
    )
    normal_run_table, normal_summary = normal_gate_table(
        evaluation_scores,
        evaluation_runs,
        evaluation_samples,
        joint_threshold,
    )
    del evaluation_scores
    gc.collect()
    print(
        f"Normal-evaluation joint-gate FAR={100*normal_summary['normal_evaluation_run_far']:.2f}% "
        f"[{100*normal_summary['normal_evaluation_run_far_ci_low']:.2f}%, "
        f"{100*normal_summary['normal_evaluation_run_far_ci_high']:.2f}%]"
    )

    testing, testing_object, testing_names = step18.read_rdata_frame(
        RAW_TESTING_PATH, "faulty_testing"
    )
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    step24.require_columns(testing, required, "Faulty-Testing RData object")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    fault_run_tables = []
    by_fault_rows = []
    crossfault_parity_differences = []
    for actual_fault in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"Frozen joint-gate final evaluation: Fault {actual_fault}")
        print("=" * 96)
        mask = (fault_numbers == actual_fault) & run_numbers.between(251, 500)
        final = testing.loc[mask, required].copy()
        final, _ = step18.validate_fault_frame(
            final,
            variable_cols,
            actual_fault,
            EVALUATION_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{actual_fault}_step26_final",
        )
        residuals, current_residual_cols = step11.compute_residual_frame(
            final, variable_cols, model
        )
        del final
        if current_residual_cols != residual_cols:
            raise ValueError(f"Fault {actual_fault}: residual order changed")
        runs, sample_ids, scores = reconstruct_normalized_scores(
            residuals,
            monitor_specs,
            step20,
            step24,
            EVALUATION_RUNS,
        )
        del residuals
        crossfault_parity_differences.append(
            verify_crossfault_step25_parity(
                scores, sample_ids, actual_fault, step25_crossfault
            )
        )
        run_table, fault_summary = evaluate_fault_gate(
            scores,
            runs,
            sample_ids,
            joint_threshold,
            actual_fault,
        )
        del scores
        step25_row = step25_bank_fault[
            step25_bank_fault["actual_fault_id"].astype(int) == actual_fault
        ]
        if len(step25_row) != 1:
            raise ValueError(f"Step-25 bank result missing for fault {actual_fault}")
        step25_row = step25_row.iloc[0]
        fault_summary.update(
            {
                "step25_or_bank_run_detection_rate": float(
                    step25_row["bank_run_detection_rate"]
                ),
                "step25_or_bank_post_fault_point_detection_rate": float(
                    step25_row["bank_post_fault_point_detection_rate"]
                ),
                "step25_or_bank_median_alarm_delay_min": float(
                    step25_row["bank_median_alarm_delay_min"]
                ),
                "step25_or_bank_allrun_penalized_mean_delay_min": float(
                    step25_row["bank_allrun_penalized_mean_delay_min"]
                ),
            }
        )
        fault_run_tables.append(run_table)
        by_fault_rows.append(fault_summary)
        print(
            f"  DR={100*fault_summary['run_detection_rate']:.1f}%, "
            f"point={100*fault_summary['post_fault_point_detection_rate']:.1f}%, "
            f"penalized mean delay={fault_summary['allrun_penalized_mean_delay_min']:.1f} min, "
            f"target top-3={100*fault_summary['target_top3_at_gate_rate_all_runs']:.1f}%"
        )
        gc.collect()

    del testing, fault_numbers, run_numbers
    gc.collect()

    fault_runs = pd.concat(fault_run_tables, ignore_index=True).sort_values(
        ["actual_fault_id", "simulationRun"]
    )
    by_fault = pd.DataFrame(by_fault_rows).sort_values("actual_fault_id")
    if len(fault_runs) != N_MONITORS * len(EVALUATION_RUNS):
        raise ValueError("Joint-gate run-attribution table is incomplete")
    if len(by_fault) != N_MONITORS:
        raise ValueError("Joint-gate fault summary is incomplete")

    step25_or_far = float(step25_summary["normal_bank_run_level_far"])
    step25_macro_detection = float(step25_summary["macro_mean_bank_run_detection"])
    current_far = normal_summary["normal_evaluation_run_far"]
    relative_far_reduction = (
        float((step25_or_far - current_far) / step25_or_far)
        if step25_or_far > 0
        else float("nan")
    )
    faults_all_detected = int((by_fault["detected_runs"] == len(EVALUATION_RUNS)).sum())
    summary = pd.DataFrame(
        [
            {
                "analysis_status": "post-freeze exploratory deployment layer",
                "n_monitors": N_MONITORS,
                "n_reserved_faults": N_MONITORS,
                "joint_gate_score": "max frozen CUSUM/local limit",
                "joint_gate_threshold": joint_threshold,
                "joint_gate_calibration_quantile": JOINT_GATE_QUANTILE,
                "joint_gate_quantile_method": QUANTILE_METHOD,
                "normal_validation_alarm_runs": int(validation_gate_alarm.sum()),
                "normal_validation_run_far": float(validation_gate_alarm.mean()),
                "normal_validation_point_far": validation_point_far,
                **normal_summary,
                "step25_or_bank_normal_run_far": step25_or_far,
                "relative_normal_run_far_reduction_vs_step25": relative_far_reduction,
                "step25_macro_mean_bank_run_detection": step25_macro_detection,
                "faults_with_250_of_250_detected": faults_all_detected,
                "macro_mean_run_detection": float(by_fault["run_detection_rate"].mean()),
                "macro_median_run_detection": float(by_fault["run_detection_rate"].median()),
                "macro_mean_post_fault_point_detection": float(
                    by_fault["post_fault_point_detection_rate"].mean()
                ),
                "macro_median_allrun_penalized_mean_delay_min": float(
                    by_fault["allrun_penalized_mean_delay_min"].median()
                ),
                "macro_mean_target_top1_at_gate_rate": float(
                    by_fault["target_top1_at_gate_rate_all_runs"].mean()
                ),
                "macro_mean_target_top3_at_gate_rate": float(
                    by_fault["target_top3_at_gate_rate_all_runs"].mean()
                ),
                "macro_median_target_rank_at_gate": float(
                    by_fault["median_target_rank_at_gate_detected_runs"].median()
                ),
                "normal_step25_parity_max_abs_difference": normal_parity_difference,
                "normal_validation_local_far_parity_max_abs_difference": (
                    validation_local_parity_difference
                ),
                "crossfault_step25_parity_max_abs_difference": float(
                    max(crossfault_parity_differences)
                ),
                "normal_evaluation_used_for_calibration": False,
                "fault_final_used_for_calibration": False,
                "confirmatory_inference": False,
            }
        ]
    )

    comparison_rows = []
    for _, row in by_fault.iterrows():
        comparison_rows.append(
            {
                "actual_fault_id": int(row["actual_fault_id"]),
                "step25_or_bank_run_detection_rate": row[
                    "step25_or_bank_run_detection_rate"
                ],
                "step26_joint_gate_run_detection_rate": row["run_detection_rate"],
                "run_detection_difference_step26_minus_step25": row[
                    "run_detection_rate"
                ]
                - row["step25_or_bank_run_detection_rate"],
                "step25_or_bank_allrun_penalized_mean_delay_min": row[
                    "step25_or_bank_allrun_penalized_mean_delay_min"
                ],
                "step26_joint_gate_allrun_penalized_mean_delay_min": row[
                    "allrun_penalized_mean_delay_min"
                ],
                "delay_difference_step26_minus_step25_min": row[
                    "allrun_penalized_mean_delay_min"
                ]
                - row["step25_or_bank_allrun_penalized_mean_delay_min"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    guardrails = reporting_guardrails()

    tables = {
        "normal_runs": normal_run_table,
        "fault_runs": fault_runs,
        "by_fault": by_fault,
        "comparison": comparison,
        "summary": summary,
        "guardrails": guardrails,
    }
    for key, table in tables.items():
        table.to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")

    figure_paths = []
    figure_paths.extend(plot_operating_point(summary.iloc[0]))
    figure_paths.extend(plot_fault_results(by_fault))
    figure_paths.extend(plot_ranks(by_fault))

    config = {
        "analysis_name": "Step 26 post-freeze joint detector-bank gate",
        "analysis_status": "post-freeze exploratory deployment layer",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_script_sha256": registration["script_sha256"],
        "registered_input_sha256": registration["input_sha256"],
        "monitor_faults": RESERVED_FAULTS,
        "actual_faults": RESERVED_FAULTS,
        "method_key": TOP5_KEY,
        "joint_gate_threshold": joint_threshold,
        "joint_gate_calibration_quantile": JOINT_GATE_QUANTILE,
        "joint_gate_quantile_method": QUANTILE_METHOD,
        "normal_validation_used_for_system_gate_calibration": True,
        "normal_evaluation_used_for_calibration": False,
        "fault_final_used_for_calibration": False,
        "all_17_step25_normal_monitor_parity_checks_passed": True,
        "all_289_step25_crossfault_parity_checks_passed": True,
        "summary": summary.iloc[0].to_dict(),
        "outputs": {key: str(path) for key, path in OUTPUT_PATHS.items()},
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
    print("Step 26 summary")
    print("=" * 96)
    print(
        f"Frozen joint-gate threshold: {row['joint_gate_threshold']:.6f} "
        f"(normal-validation higher 95th percentile)"
    )
    print(
        f"Normal-evaluation run FAR: {100*row['normal_evaluation_run_far']:.2f}% "
        f"[{100*row['normal_evaluation_run_far_ci_low']:.2f}%, "
        f"{100*row['normal_evaluation_run_far_ci_high']:.2f}%]"
    )
    print(
        f"Step-25 OR-bank run FAR: {100*row['step25_or_bank_normal_run_far']:.2f}%"
    )
    print(
        f"Reserved faults with 250/250 detected: "
        f"{int(row['faults_with_250_of_250_detected'])}/{N_MONITORS}"
    )
    print(
        f"Macro mean run detection: {100*row['macro_mean_run_detection']:.2f}%"
    )
    print(
        f"Macro median all-run penalized mean delay: "
        f"{row['macro_median_allrun_penalized_mean_delay_min']:.1f} min"
    )
    print(
        f"Macro target top-3-at-gate rate: "
        f"{100*row['macro_mean_target_top3_at_gate_rate']:.2f}%"
    )
    print("Saved tables:")
    for key in tables:
        print(f"  {OUTPUT_PATHS[key]}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Saved configuration: {FINAL_CONFIG_PATH}")
    print("Step 26 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="register the immutable post-freeze joint-gate study",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify the registration and run the joint-gate study",
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
