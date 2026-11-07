#!/usr/bin/env python3
"""U3: Matched frozen residual-evidence robustness audit.

Scientific role
---------------
This is a post-freeze secondary audit of the already-frozen Step-22-R
same-family Top-1 and Proposed Top-5 monitors. It directly addresses two
limitations of the earlier Step-24 robustness analysis:

1. Top-5 is subjected to simultaneous loss of 1, 2, or 3 selected standardized
   residual-evidence coordinates, enumerating every combination while retaining
   the original denominator and frozen control limit.
2. Top-1 and Top-5 receive matched perturbations under two explicit designs:

   * exact matched total channel-space energy: the Top-5 perturbation vector has
     the same per-sample squared L2 energy as the Top-1 scalar perturbation;
   * exact matched aggregate-index perturbation: the same common-mode Gaussian
     sequence is applied to all Top-5 coordinates, so its averaged perturbation
     is exactly the Top-1 perturbation.

No PCA model, evidence family, selected coordinate, orientation, scaler, CUSUM
reference value, or control limit is refit or retuned. Perturbations occur after
PCA reconstruction and frozen normal-validation standardization. They are tests
of standardized residual-evidence coordinates, not simulations of physical
sensor loss, raw-input noise, plant drift, or missing measurements.

Two-stage use
-------------
Run registration before process-data links exist, then run the frozen audit:

    python scripts/U3_matched_robustness_audit.py --register
    python scripts/U3_matched_robustness_audit.py --run
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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
FAULT_START_SAMPLE = 161
SAMPLE_MIN = 1
SAMPLE_MAX = 960
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (SAMPLE_MAX - FAULT_START_SAMPLE + 1) * SAMPLING_INTERVAL_MIN
EARLY_WINDOWS_MIN = (15, 30, 60, 120)

TOP1_KEY = "chosen_family_top1_cusum"
TOP5_KEY = "frozen_multievidence_top5_cusum"
METHOD_LABELS = {TOP1_KEY: "Family Top-1", TOP5_KEY: "Proposed Top-5"}
LOSS_COUNTS = (1, 2, 3)
CONTAMINATION_SIGMAS = (0.5, 1.0, 2.0)
PRIMARY_SIGMA = 1.0
MONTE_CARLO_REPLICATES = 20
MONTE_CARLO_SEED = 20260821031
ALPHA = 0.05
PARITY_TOLERANCE = 1e-12
ENERGY_TOLERANCE = 1e-10

STEP24_SCRIPT_PATH = SCRIPT_DIR / "24_frozen_evidence_channel_robustness.py"
STEP24_EXPECTED_SHA256 = "afd2dfd6873291ecb8d4426d9bee8179fe51c91ed1678ebf966d5c357202cb82"

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"

SPECIFICATION_PATH = CONFIG_DIR / "U3_matched_robustness_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "U3_matched_robustness_audit.json"

OUTPUT_PATHS = {
    "clean_parity": TABLE_DIR / "U3_clean_reconstruction_parity.csv",
    "loss_combination": TABLE_DIR / "U3_multichannel_loss_by_combination.csv",
    "loss_fault": TABLE_DIR / "U3_multichannel_loss_by_fault.csv",
    "loss_crossfault": TABLE_DIR / "U3_multichannel_loss_crossfault_summary.csv",
    "contamination_replicate": TABLE_DIR / "U3_matched_contamination_replicate_metrics.csv",
    "contamination_fault": TABLE_DIR / "U3_matched_contamination_by_fault.csv",
    "contamination_crossfault": TABLE_DIR / "U3_matched_contamination_crossfault_summary.csv",
    "paired_comparisons": TABLE_DIR / "U3_matched_contamination_paired_comparisons.csv",
    "guardrails": TABLE_DIR / "U3_reporting_guardrails.csv",
    "loss_png": FIGURE_DIR / "figU3a_multichannel_loss_robustness.png",
    "loss_pdf": FIGURE_DIR / "figU3a_multichannel_loss_robustness.pdf",
    "contamination_png": FIGURE_DIR / "figU3b_matched_contamination_robustness.png",
    "contamination_pdf": FIGURE_DIR / "figU3b_matched_contamination_robustness.pdf",
}

# These are small, frozen inputs. Process data are deliberately absent during
# registration and are linked only after the immutable U3 specification exists.
REQUIRED_SMALL_INPUTS = {
    "step11_script": SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
    "step15_script": SCRIPT_DIR / "15_fair_sequential_baselines.py",
    "step18_script": SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
    "step20_script": SCRIPT_DIR / "20_frozen_multievidence_validation.py",
    "step22r_script": SCRIPT_DIR / "22r_reserved_multifault_frozen_validation.py",
    "step24_script": STEP24_SCRIPT_PATH,
    "step22_results": TABLE_DIR / "22_reserved_multifault_results.csv",
    "step22_run_results": TABLE_DIR / "22_reserved_multifault_run_results.csv",
    "step22_parameters": TABLE_DIR / "22_reserved_multifault_method_parameters.csv",
    "step22_scalers": TABLE_DIR / "22_reserved_multifault_local_scalers.csv",
    "step22_config": CONFIG_DIR / "22_reserved_multifault_frozen_validation.json",
    "step24_config": CONFIG_DIR / "24_frozen_evidence_channel_robustness.json",
    "step24_specification": CONFIG_DIR / "24_frozen_evidence_channel_robustness_specification.json",
    "step24_clean_parity": TABLE_DIR / "24_clean_reconstruction_parity.csv",
    "step24_loss_summary": TABLE_DIR / "24_channel_loss_crossfault_summary.csv",
    "step24_contamination_summary": TABLE_DIR / "24_channel_contamination_crossfault_summary.csv",
    "step24_guardrails": TABLE_DIR / "24_reporting_guardrails.csv",
    "u2_config": CONFIG_DIR / "U2_selection_stability_simplification.json",
    "u2_crossrepeat_summary": TABLE_DIR / "U2_crossrepeat_summary.csv",
    "u2_fault_stability": TABLE_DIR / "U2_fault_stability_summary.csv",
    "u2_guardrails": TABLE_DIR / "U2_reporting_guardrails.csv",
    "variable_list": TABLE_DIR / "02_variable_list.csv",
}

BASE_METRICS = [
    "normal_run_level_far",
    "normal_point_level_far",
    "fault_pre_fault_run_alarm_rate",
    "fault_pre_fault_point_alarm_rate",
    "fault_run_level_detection_rate",
    "fault_post_fault_point_detection_rate",
    "median_alarm_delay_min",
    "mean_alarm_delay_min",
    "allrun_penalized_median_delay_min",
    "allrun_penalized_mean_delay_min",
]
EARLY_METRICS = [f"fault_detection_within_{window}_min" for window in EARLY_WINDOWS_MIN]
METRIC_COLUMNS = [*BASE_METRICS, *EARLY_METRICS]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_INVALID:{path}")
    return path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def small_input_hashes() -> dict[str, str]:
    return {
        key: sha256_file(require_regular(path, f"U3_SMALL_INPUT_{key}"))
        for key, path in REQUIRED_SMALL_INPUTS.items()
    }


def specification_payload(script_hash: str, input_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "registration_name": "U3 matched frozen residual-evidence robustness audit",
        "registration_status": "post-freeze secondary critical robustness audit",
        "script": "scripts/U3_matched_robustness_audit.py",
        "script_sha256": script_hash,
        "small_input_sha256": input_hashes,
        "process_data_present_during_registration": False,
        "process_data_loaded_during_registration": False,
        "primary_step22r_state_changed": False,
        "detector_fitting_allowed": False,
        "family_reselection_allowed": False,
        "coordinate_reselection_allowed": False,
        "orientation_change_allowed": False,
        "scaler_change_allowed": False,
        "cusum_k_change_allowed": False,
        "control_limit_change_allowed": False,
        "loss_protocol": {
            "top1": "sole coordinate loss retained only as a structural reference",
            "top5_loss_counts": list(LOSS_COUNTS),
            "all_combinations_enumerated": True,
            "replacement_standardized_value": 0.0,
            "original_denominator_retained": True,
            "frozen_control_limit_retained": True,
        },
        "contamination_protocol": {
            "sigma_grid": list(CONTAMINATION_SIGMAS),
            "primary_sigma": PRIMARY_SIGMA,
            "replicates": MONTE_CARLO_REPLICATES,
            "seed": MONTE_CARLO_SEED,
            "top1_reference": (
                "one standardized coordinate receives a Gaussian sequence n(t)"
            ),
            "top5_exact_total_energy": (
                "all five coordinates receive n(t) times a random unit vector; "
                "the per-sample sum of squared coordinate perturbations equals n(t)^2 exactly"
            ),
            "top5_exact_aggregate_common_mode": (
                "the same Gaussian sequence n(t) is added to all five coordinates; "
                "the averaged index perturbation equals n(t) exactly"
            ),
            "dynamic_family_first_sample_noise": 0.0,
            "frozen_denominator_and_limit": True,
        },
        "endpoints": {
            "normal_far": True,
            "early_detection_minutes": list(EARLY_WINDOWS_MIN),
            "full_horizon_run_detection": True,
            "post_fault_threshold_occupancy": True,
            "allrun_miss_penalty_minutes": MISS_PENALTY_MIN,
        },
        "inference_unit": "reserved fault after Monte Carlo averaging within fault",
        "multiplicity": "Holm within each sigma and matched contamination contrast",
        "mandatory_disclosures": [
            "U1 found no general panel-level superiority of five-family adaptation",
            "U2 found high family recovery but moderate coordinate-set variability",
            "Top-1 sole-coordinate loss is structurally implied and not a novel empirical result",
            "all perturbations occur in standardized residual-evidence space",
            "no physical-sensor, raw-input, plant-drift, missingness, or deployment claim is allowed",
        ],
        "scope": "NON_QUALIFICATION_EXPLORATORY",
    }


def register_specification() -> None:
    data_root = PROJECT_ROOT / "data"
    if data_root.exists():
        raise RuntimeError("U3_PROCESS_DATA_PATH_PRESENT_DURING_REGISTRATION")
    script_hash = sha256_file(Path(__file__))
    hashes = small_input_hashes()
    payload = specification_payload(script_hash, hashes)
    payload["registered_at_utc"] = utc_now()
    if SPECIFICATION_PATH.exists():
        existing = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
        existing_core = {k: v for k, v in existing.items() if k != "registered_at_utc"}
        new_core = {k: v for k, v in payload.items() if k != "registered_at_utc"}
        if existing_core != new_core:
            raise RuntimeError("U3_DIFFERENT_SPECIFICATION_ALREADY_EXISTS")
        print(f"Identical U3 specification already exists: {SPECIFICATION_PATH}")
        return
    atomic_json(SPECIFICATION_PATH, payload)
    print("U3 matched robustness specification registered without process data.")
    print(f"U3_SPECIFICATION={SPECIFICATION_PATH}")


def verify_specification() -> dict[str, Any]:
    require_regular(SPECIFICATION_PATH, "U3_SPECIFICATION")
    existing = json.loads(SPECIFICATION_PATH.read_text(encoding="utf-8"))
    expected = specification_payload(sha256_file(Path(__file__)), small_input_hashes())
    failures = []
    for key, value in expected.items():
        if existing.get(key) != value:
            failures.append(key)
    if failures:
        raise RuntimeError(f"U3_SPECIFICATION_MISMATCH:{failures}")
    return existing


def load_module(path: Path, name: str):
    require_regular(path, f"U3_MODULE_{name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"U3_MODULE_SPEC_FAIL:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_step24():
    observed = sha256_file(require_regular(STEP24_SCRIPT_PATH, "U3_STEP24_SOURCE"))
    if observed != STEP24_EXPECTED_SHA256:
        raise RuntimeError(f"U3_STEP24_SHA256_MISMATCH:{observed}")
    return load_module(STEP24_SCRIPT_PATH, "u3_exact_step24")


def evaluate_scalar_pair(
    scalar_normal: np.ndarray,
    scalar_fault: np.ndarray,
    sample_ids: np.ndarray,
    k: float,
    limit: float,
) -> dict[str, float | int]:
    if scalar_normal.ndim != 2 or scalar_fault.ndim != 2:
        raise ValueError("U3 scalar inputs must be runs by samples")
    if scalar_normal.shape[1] != scalar_fault.shape[1] or scalar_fault.shape[1] != len(sample_ids):
        raise ValueError("U3 scalar sample dimensions differ")

    def cusum(values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values, dtype=float)
        current = np.zeros(values.shape[0], dtype=float)
        for sample_index in range(values.shape[1]):
            current = np.maximum(0.0, current + values[:, sample_index] - float(k))
            output[:, sample_index] = current
        return output

    normal_alarm = cusum(scalar_normal) > float(limit)
    normal_run_alarm = normal_alarm.any(axis=1)

    fault_alarm = cusum(scalar_fault) > float(limit)
    pre_mask = sample_ids < FAULT_START_SAMPLE
    post_mask = sample_ids >= FAULT_START_SAMPLE
    if not pre_mask.any() or not post_mask.any():
        raise ValueError("U3 requires both pre- and post-fault samples")
    pre_alarm = fault_alarm[:, pre_mask]
    post_alarm = fault_alarm[:, post_mask]
    pre_run_alarm = pre_alarm.any(axis=1)
    detected = post_alarm.any(axis=1)
    post_samples = sample_ids[post_mask]
    first_relative = post_alarm.argmax(axis=1)
    detected_delay = np.full(len(detected), np.nan, dtype=float)
    detected_delay[detected] = (
        post_samples[first_relative[detected]] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN
    penalized = np.where(detected, detected_delay, float(MISS_PENALTY_MIN))
    observed = detected_delay[detected]

    result: dict[str, float | int] = {
        "normal_n_runs": int(scalar_normal.shape[0]),
        "normal_alarm_runs": int(normal_run_alarm.sum()),
        "normal_run_level_far": float(normal_run_alarm.mean()),
        "normal_point_level_far": float(normal_alarm.mean()),
        "fault_n_runs": int(scalar_fault.shape[0]),
        "fault_pre_fault_alarm_runs": int(pre_run_alarm.sum()),
        "fault_pre_fault_run_alarm_rate": float(pre_run_alarm.mean()),
        "fault_pre_fault_point_alarm_rate": float(pre_alarm.mean()),
        "detected_runs": int(detected.sum()),
        "right_censored_runs": int((~detected).sum()),
        "fault_run_level_detection_rate": float(detected.mean()),
        "fault_post_fault_point_detection_rate": float(post_alarm.mean()),
        "median_alarm_delay_min": float(np.median(observed)) if len(observed) else np.nan,
        "mean_alarm_delay_min": float(np.mean(observed)) if len(observed) else np.nan,
        "allrun_penalized_median_delay_min": float(np.median(penalized)),
        "allrun_penalized_mean_delay_min": float(np.mean(penalized)),
    }
    for window in EARLY_WINDOWS_MIN:
        result[f"fault_detection_within_{window}_min"] = float(
            np.mean(detected & (detected_delay <= float(window)))
        )
    return result


def same_number(left: Any, right: Any, tolerance: float = PARITY_TOLERANCE) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=tolerance))


def add_clean_deltas(record: dict[str, Any], clean: dict[str, Any]) -> None:
    for metric in METRIC_COLUMNS:
        current = float(record[metric])
        baseline = float(clean[metric])
        record[f"clean_{metric}"] = baseline
        record[f"delta_{metric}"] = current - baseline


def dynamic_family(family_key: str) -> bool:
    return family_key in {"absolute_increment", "lag1_product"}


def seed_sequence(
    fault_id: int,
    sigma: float,
    replicate: int,
    dataset_code: int,
    purpose_code: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [
            MONTE_CARLO_SEED,
            int(fault_id),
            int(round(1000.0 * sigma)),
            int(replicate),
            int(dataset_code),
            int(purpose_code),
        ]
    )


def matched_noise_arrays(
    shape: tuple[int, int],
    n_channels: int,
    sigma: float,
    fault_id: int,
    replicate: int,
    dataset_code: int,
    zero_first_sample: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return base Top-1 noise and exact-energy Top-5 noise cube.

    For every run/sample, sum_j cube[...,j]^2 == base^2 up to floating error.
    """
    base_rng = np.random.default_rng(
        seed_sequence(fault_id, sigma, replicate, dataset_code, purpose_code=1)
    )
    base = base_rng.normal(0.0, float(sigma), size=shape)
    direction_rng = np.random.default_rng(
        seed_sequence(fault_id, sigma, replicate, dataset_code, purpose_code=2)
    )
    direction = direction_rng.normal(0.0, 1.0, size=(*shape, n_channels))
    norms = np.linalg.norm(direction, axis=2, keepdims=True)
    if np.any(norms == 0.0):
        raise RuntimeError("U3_ZERO_RANDOM_DIRECTION_NORM")
    unit = direction / norms
    cube = base[:, :, None] * unit
    if zero_first_sample:
        base[:, 0] = 0.0
        cube[:, 0, :] = 0.0
    energy_error = float(
        np.max(np.abs(np.sum(cube * cube, axis=2) - base * base))
    )
    aggregate_common_error = 0.0  # common mode is added as base directly
    if energy_error > ENERGY_TOLERANCE:
        raise RuntimeError(f"U3_MATCHED_ENERGY_ERROR:{energy_error}")
    return base, cube, energy_error, aggregate_common_error


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    statistic, p_value = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
    return float(statistic), float(p_value)


def aggregate_loss_by_fault(loss_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["fault_id", "method_key", "method_short", "loss_count"]
    for keys, current in loss_table.groupby(group_cols, sort=True):
        fault_id, method_key, method_short, loss_count = keys
        row: dict[str, Any] = {
            "fault_id": int(fault_id),
            "method_key": str(method_key),
            "method_short": str(method_short),
            "loss_count": int(loss_count),
            "n_combinations": int(len(current)),
        }
        for metric in METRIC_COLUMNS:
            values = current[metric].to_numpy(dtype=float)
            row[f"combination_mean_{metric}"] = float(np.nanmean(values))
        worst = current.sort_values(
            [
                "fault_run_level_detection_rate",
                "fault_post_fault_point_detection_rate",
                "allrun_penalized_mean_delay_min",
                "combination_key",
            ],
            ascending=[True, True, False, True],
        ).iloc[0]
        row["worst_combination_key"] = str(worst["combination_key"])
        row["worst_combination_variables"] = str(worst["lost_variables"])
        for metric in METRIC_COLUMNS:
            row[f"combination_worst_{metric}"] = float(worst[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_key", "loss_count", "fault_id"])


def aggregate_crossfault(table: pd.DataFrame, grouping: list[str], prefixes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, current in table.groupby(grouping, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row["n_faults"] = int(current["fault_id"].nunique())
        for prefix in prefixes:
            for metric in METRIC_COLUMNS:
                column = f"{prefix}{metric}"
                values = current[column].to_numpy(dtype=float)
                row[f"macro_mean_{column}"] = float(np.nanmean(values))
                row[f"macro_median_{column}"] = float(np.nanmedian(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping)


def aggregate_contamination_by_fault(replicate_table: pd.DataFrame) -> pd.DataFrame:
    identity = ["fault_id", "sigma", "arm_key", "arm_label"]
    rows: list[dict[str, Any]] = []
    for keys, current in replicate_table.groupby(identity, sort=True):
        row = dict(zip(identity, keys))
        row["n_replicates"] = int(len(current))
        row["maximum_energy_match_abs_error"] = float(
            current["energy_match_abs_error"].max()
        )
        row["maximum_aggregate_match_abs_error"] = float(
            current["aggregate_match_abs_error"].max()
        )
        for metric in METRIC_COLUMNS:
            values = current[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sigma", "arm_key", "fault_id"])


def contamination_crossfault(by_fault: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, current in by_fault.groupby(["sigma", "arm_key", "arm_label"], sort=True):
        sigma, arm_key, arm_label = keys
        row: dict[str, Any] = {
            "sigma": float(sigma),
            "arm_key": str(arm_key),
            "arm_label": str(arm_label),
            "n_faults": int(current["fault_id"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = current[metric].to_numpy(dtype=float)
            row[f"macro_mean_{metric}"] = float(np.nanmean(values))
            row[f"macro_median_{metric}"] = float(np.nanmedian(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sigma", "arm_key"])


def build_paired_comparisons(by_fault: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        (
            "top5_exact_total_energy_vs_top1",
            "top5_exact_total_energy",
            "top1_reference",
            "Top-5 exact total-energy versus Top-1",
        ),
        (
            "top5_common_mode_vs_top1",
            "top5_common_mode_equal_index",
            "top1_reference",
            "Top-5 common-mode equal-index versus Top-1",
        ),
        (
            "top5_common_mode_vs_total_energy",
            "top5_common_mode_equal_index",
            "top5_exact_total_energy",
            "Top-5 common-mode versus Top-5 exact total-energy",
        ),
    ]
    endpoints = [
        ("normal_run_level_far", "lower"),
        ("fault_detection_within_15_min", "higher"),
        ("fault_detection_within_60_min", "higher"),
        ("fault_detection_within_120_min", "higher"),
        ("fault_run_level_detection_rate", "higher"),
        ("fault_post_fault_point_detection_rate", "higher"),
        ("allrun_penalized_mean_delay_min", "lower"),
    ]
    rows: list[dict[str, Any]] = []
    for sigma in CONTAMINATION_SIGMAS:
        current_sigma = by_fault[np.isclose(by_fault["sigma"], sigma)]
        for contrast_key, candidate_key, reference_key, label in definitions:
            candidate = current_sigma[current_sigma["arm_key"] == candidate_key].set_index("fault_id")
            reference = current_sigma[current_sigma["arm_key"] == reference_key].set_index("fault_id")
            if sorted(candidate.index.tolist()) != RESERVED_FAULTS or sorted(reference.index.tolist()) != RESERVED_FAULTS:
                raise RuntimeError(f"U3_PAIRED_FAULT_SET_MISMATCH:{sigma}:{contrast_key}")
            family_rows: list[dict[str, Any]] = []
            for endpoint, direction in endpoints:
                cand = candidate.loc[RESERVED_FAULTS, endpoint].to_numpy(dtype=float)
                ref = reference.loc[RESERVED_FAULTS, endpoint].to_numpy(dtype=float)
                raw_difference = cand - ref
                utility = raw_difference if direction == "higher" else -raw_difference
                statistic, p_value = paired_wilcoxon(utility)
                family_rows.append(
                    {
                        "sigma": float(sigma),
                        "contrast_key": contrast_key,
                        "contrast_label": label,
                        "candidate_arm": candidate_key,
                        "reference_arm": reference_key,
                        "endpoint": endpoint,
                        "direction": direction,
                        "n_faults": len(RESERVED_FAULTS),
                        "candidate_mean": float(np.mean(cand)),
                        "reference_mean": float(np.mean(ref)),
                        "raw_candidate_minus_reference_mean": float(np.mean(raw_difference)),
                        "mean_utility_difference_positive_favors_candidate": float(np.mean(utility)),
                        "median_utility_difference_positive_favors_candidate": float(np.median(utility)),
                        "candidate_better_faults": int(np.sum(utility > 0)),
                        "ties": int(np.sum(np.isclose(utility, 0.0, rtol=0.0, atol=1e-15))),
                        "reference_better_faults": int(np.sum(utility < 0)),
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                    }
                )
            adjusted = holm_adjust([row["raw_p_value"] for row in family_rows])
            for row, value in zip(family_rows, adjusted):
                row["holm_adjusted_p_value_within_sigma_contrast"] = value
                row["significant_0_05"] = bool(value < 0.05)
            rows.extend(family_rows)
    return pd.DataFrame(rows).sort_values(["sigma", "contrast_key", "endpoint"])


def reporting_guardrails() -> pd.DataFrame:
    rows = [
        (
            "U3-G1",
            "Top-1 sole-coordinate loss",
            "Treat the 0-channel Top-1 result as a structural availability reference, not a nontrivial empirical discovery.",
        ),
        (
            "U3-G2",
            "Perturbation domain",
            "Describe all U3 perturbations as standardized residual-evidence-coordinate tests; never call them physical sensor failures or raw-input robustness.",
        ),
        (
            "U3-G3",
            "Matched total energy",
            "State that exact channel-space L2 energy is matched per run and sample; this does not match aggregate-index variance.",
        ),
        (
            "U3-G4",
            "Matched aggregate index",
            "State that common-mode contamination exactly matches the Top-1 aggregate-index perturbation and is a severe perfectly correlated evidence-coordinate stress test.",
        ),
        (
            "U3-G5",
            "Frozen state",
            "No family, coordinate, orientation, scaler, k, or control limit was changed after Step 22-R.",
        ),
        (
            "U3-G6",
            "Primary hierarchy",
            "U3 is post-freeze secondary evidence and cannot replace the Step-22-R primary reserved-fault analysis.",
        ),
        (
            "U3-G7",
            "U1/U2 disclosure",
            "Retain the U1 simplification result and U2 coordinate-variability result; U3 cannot be used to restore a universal family-adaptation claim.",
        ),
        (
            "U3-G8",
            "Deployment",
            "Do not claim real-plant, open-ended ARL, missing-data, sensor-drift, or prospective deployment robustness.",
        ),
    ]
    return pd.DataFrame(rows, columns=["guardrail_id", "topic", "mandatory_wording"])


def plot_loss(crossfault: pd.DataFrame, clean: pd.DataFrame) -> list[Path]:
    top5_clean = clean[clean["method_key"] == TOP5_KEY]
    top1_clean = clean[clean["method_key"] == TOP1_KEY]
    x_top5 = [0]
    run_top5 = [float(top5_clean["fault_run_level_detection_rate"].mean())]
    point_top5 = [float(top5_clean["fault_post_fault_point_detection_rate"].mean())]
    delay_top5 = [float(top5_clean["allrun_penalized_mean_delay_min"].median())]
    for loss_count in LOSS_COUNTS:
        row = crossfault[
            (crossfault["method_key"] == TOP5_KEY)
            & (crossfault["loss_count"] == loss_count)
        ].iloc[0]
        x_top5.append(loss_count)
        run_top5.append(float(row["macro_mean_combination_worst_fault_run_level_detection_rate"]))
        point_top5.append(float(row["macro_mean_combination_worst_fault_post_fault_point_detection_rate"]))
        delay_top5.append(float(row["macro_median_combination_worst_allrun_penalized_mean_delay_min"]))

    top1_loss = crossfault[
        (crossfault["method_key"] == TOP1_KEY) & (crossfault["loss_count"] == 1)
    ].iloc[0]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    axes[0].plot(x_top5, 100 * np.asarray(run_top5), marker="o", label="Top-5 worst combination")
    axes[0].scatter([0, 1], [100 * float(top1_clean["fault_run_level_detection_rate"].mean()), 100 * float(top1_loss["macro_mean_combination_worst_fault_run_level_detection_rate"])], marker="x", label="Top-1 structural reference")
    axes[0].set_ylabel("Macro run detection (%)")
    axes[1].plot(x_top5, 100 * np.asarray(point_top5), marker="o")
    axes[1].set_ylabel("Macro post-fault occupancy (%)")
    axes[2].plot(x_top5, delay_top5, marker="o")
    axes[2].set_ylabel("Macro median penalized mean delay (min)")
    for axis in axes:
        axis.set_xlabel("Simultaneously lost selected coordinates")
        axis.set_xticks([0, 1, 2, 3])
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Frozen Top-5 robustness to simultaneous residual-evidence-coordinate loss")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATHS["loss_png"], dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PATHS["loss_pdf"], bbox_inches="tight")
    plt.close(figure)
    return [OUTPUT_PATHS["loss_png"], OUTPUT_PATHS["loss_pdf"]]


def plot_contamination(crossfault: pd.DataFrame) -> list[Path]:
    current = crossfault[np.isclose(crossfault["sigma"], PRIMARY_SIGMA)].copy()
    order = ["top1_reference", "top5_exact_total_energy", "top5_common_mode_equal_index"]
    current["order"] = current["arm_key"].map({key: i for i, key in enumerate(order)})
    current = current.sort_values("order")
    labels = ["Top-1\nreference", "Top-5\nmatched energy", "Top-5\ncommon mode"]
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
    axes[0].bar(labels, 100 * current["macro_mean_normal_run_level_far"].to_numpy(dtype=float))
    axes[0].set_ylabel("Macro normal FAR (%)")
    axes[1].bar(labels, 100 * current["macro_mean_fault_post_fault_point_detection_rate"].to_numpy(dtype=float))
    axes[1].set_ylabel("Macro post-fault occupancy (%)")
    axes[2].bar(labels, current["macro_median_allrun_penalized_mean_delay_min"].to_numpy(dtype=float))
    axes[2].set_ylabel("Macro median penalized mean delay (min)")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(f"Matched residual-evidence contamination at σ={PRIMARY_SIGMA:.1f}")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATHS["contamination_png"], dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PATHS["contamination_pdf"], bbox_inches="tight")
    plt.close(figure)
    return [OUTPUT_PATHS["contamination_png"], OUTPUT_PATHS["contamination_pdf"]]


def run_analysis() -> None:
    registration = verify_specification()
    step24 = load_step24()
    modules = step24.load_dependencies()
    step11 = modules["step11"]
    step18 = modules["step18"]
    step20 = modules["step20"]
    frozen_results, parameters, scalers, step22_config = step24.load_frozen_state()

    u2_config = json.loads((CONFIG_DIR / "U2_selection_stability_simplification.json").read_text(encoding="utf-8"))
    if u2_config.get("result") != "PASS" or u2_config.get("u1_result_disclosed") is not True:
        raise RuntimeError("U3_U2_DISCLOSURE_BINDING_FAILED")

    variable_cols = step11.load_variable_columns()
    step24.validate_variable_order(variable_cols)

    print("=" * 96)
    print("U3 matched frozen residual-evidence robustness audit")
    print("=" * 96)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print("No family, coordinate, orientation, scaler, k, or limit retuning is allowed.")

    normal_train = pd.read_pickle(NORMAL_TRAIN_PATH)
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

    testing, testing_object, testing_names = step18.read_rdata_frame(
        RAW_TESTING_PATH, "faulty_testing"
    )
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    step24.require_columns(testing, required, "Faulty-Testing RData object")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    step24_clean = pd.read_csv(TABLE_DIR / "24_clean_reconstruction_parity.csv", encoding="utf-8-sig")

    clean_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    contamination_rows: list[dict[str, Any]] = []
    max_energy_error = 0.0
    max_aggregate_error = 0.0

    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"U3 reconstructing frozen Top-1 and Top-5 for Fault {fault_id}")
        print("=" * 96)
        final_mask = (fault_numbers == fault_id) & run_numbers.between(251, 500)
        final = testing.loc[final_mask, required].copy()
        final, _ = step18.validate_fault_frame(
            final,
            variable_cols,
            fault_id,
            EXPECTED_RUNS,
            SAMPLE_MIN,
            SAMPLE_MAX,
            f"fault{fault_id}_u3_final",
        )
        final_residuals, current_cols = step11.compute_residual_frame(final, variable_cols, model)
        if current_cols != residual_cols:
            raise RuntimeError(f"U3_RESIDUAL_ORDER_CHANGED:{fault_id}")
        del final
        gc.collect()

        method_inputs: dict[str, dict[str, Any]] = {}
        for method_key in (TOP1_KEY, TOP5_KEY):
            current_scaler = scalers[
                (scalers["fault_id"].astype(int) == fault_id)
                & (scalers["method_key"].astype(str) == method_key)
            ].sort_values("selection_rank")
            normal_runs, sample_ids, normal_components, channel_metadata = step24.build_standardized_components(
                normal_residuals, current_scaler, step20
            )
            final_runs, final_samples, final_components, final_metadata = step24.build_standardized_components(
                final_residuals, current_scaler, step20
            )
            if not np.array_equal(normal_runs, EXPECTED_RUNS) or not np.array_equal(final_runs, EXPECTED_RUNS):
                raise RuntimeError(f"U3_RUN_IDS_CHANGED:{fault_id}:{method_key}")
            if not np.array_equal(sample_ids, final_samples) or channel_metadata != final_metadata:
                raise RuntimeError(f"U3_COMPONENT_METADATA_CHANGED:{fault_id}:{method_key}")
            parameter_row = parameters[
                (parameters["fault_id"].astype(int) == fault_id)
                & (parameters["method_key"].astype(str) == method_key)
            ].iloc[0]
            method_inputs[method_key] = {
                "normal_components": normal_components,
                "fault_components": final_components,
                "sample_ids": sample_ids,
                "metadata": channel_metadata,
                "k": float(parameter_row["cusum_k"]),
                "limit": float(parameter_row["control_limit"]),
                "family_key": str(parameter_row["evidence_family"]),
            }

        shared_normal = float(np.max(np.abs(
            method_inputs[TOP1_KEY]["normal_components"][:, :, 0]
            - method_inputs[TOP5_KEY]["normal_components"][:, :, 0]
        )))
        shared_fault = float(np.max(np.abs(
            method_inputs[TOP1_KEY]["fault_components"][:, :, 0]
            - method_inputs[TOP5_KEY]["fault_components"][:, :, 0]
        )))
        if max(shared_normal, shared_fault) > PARITY_TOLERANCE:
            raise RuntimeError(f"U3_SHARED_TOP1_CHANNEL_PARITY_FAILED:{fault_id}")

        clean_by_method: dict[str, dict[str, Any]] = {}
        for method_key in (TOP1_KEY, TOP5_KEY):
            state = method_inputs[method_key]
            clean_normal = state["normal_components"].mean(axis=2)
            clean_fault = state["fault_components"].mean(axis=2)
            metrics = evaluate_scalar_pair(
                clean_normal, clean_fault, state["sample_ids"], state["k"], state["limit"]
            )
            step24_metrics = step24.evaluate_scalar_pair(
                clean_normal, clean_fault, state["sample_ids"], state["k"], state["limit"]
            )
            for metric in BASE_METRICS:
                if not same_number(metrics[metric], step24_metrics[metric]):
                    raise RuntimeError(f"U3_STEP24_EVALUATOR_PARITY_FAILED:{fault_id}:{method_key}:{metric}")
            frozen_row = frozen_results[
                (frozen_results["fault_id"].astype(int) == fault_id)
                & (frozen_results["method_key"].astype(str) == method_key)
            ].iloc[0]
            mapping = {
                "normal_run_level_far": "normal_run_level_far",
                "fault_pre_fault_run_alarm_rate": "fault_pre_fault_run_alarm_rate",
                "fault_run_level_detection_rate": "fault_run_level_detection_rate",
                "fault_post_fault_point_detection_rate": "fault_post_fault_point_detection_rate",
                "median_alarm_delay_min": "median_alarm_delay_min",
                "mean_alarm_delay_min": "mean_alarm_delay_min",
            }
            for reconstructed, frozen in mapping.items():
                if not same_number(metrics[reconstructed], frozen_row[frozen]):
                    raise RuntimeError(f"U3_STEP22_PARITY_FAILED:{fault_id}:{method_key}:{reconstructed}")
            previous = step24_clean[
                (step24_clean["fault_id"].astype(int) == fault_id)
                & (step24_clean["method_key"].astype(str) == method_key)
            ].iloc[0]
            for metric in BASE_METRICS:
                if metric in previous and not same_number(metrics[metric], previous[metric]):
                    raise RuntimeError(f"U3_STEP24_TABLE_PARITY_FAILED:{fault_id}:{method_key}:{metric}")
            clean_by_method[method_key] = {
                **metrics,
                "normal_scalar": clean_normal,
                "fault_scalar": clean_fault,
            }
            clean_rows.append(
                {
                    "fault_id": fault_id,
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "evidence_family": state["family_key"],
                    "n_channels": state["normal_components"].shape[2],
                    "selected_variables": ";".join(item["variable"] for item in state["metadata"]),
                    "shared_rank1_normal_max_abs_difference": shared_normal,
                    "shared_rank1_fault_max_abs_difference": shared_fault,
                    "all_step22_and_step24_metrics_exact_with_tolerance_1e_12": True,
                    **metrics,
                }
            )

        # Structural Top-1 sole-coordinate loss reference.
        state = method_inputs[TOP1_KEY]
        clean = clean_by_method[TOP1_KEY]
        top1_loss_normal = np.zeros_like(clean["normal_scalar"])
        top1_loss_fault = np.zeros_like(clean["fault_scalar"])
        metrics = evaluate_scalar_pair(
            top1_loss_normal, top1_loss_fault, state["sample_ids"], state["k"], state["limit"]
        )
        record: dict[str, Any] = {
            "fault_id": fault_id,
            "method_key": TOP1_KEY,
            "method_short": METHOD_LABELS[TOP1_KEY],
            "loss_count": 1,
            "combination_key": "rank1",
            "lost_ranks": "1",
            "lost_variables": state["metadata"][0]["variable"],
            "n_frozen_channels": 1,
            "replacement_standardized_value": 0.0,
            "original_denominator_retained": True,
            "structural_reference_only": True,
            **metrics,
        }
        add_clean_deltas(record, clean)
        loss_rows.append(record)

        # Every simultaneous Top-5 loss combination for m=1,2,3.
        state = method_inputs[TOP5_KEY]
        clean = clean_by_method[TOP5_KEY]
        for loss_count in LOSS_COUNTS:
            for indices in itertools.combinations(range(5), loss_count):
                removed_normal = state["normal_components"][:, :, indices].sum(axis=2) / 5.0
                removed_fault = state["fault_components"][:, :, indices].sum(axis=2) / 5.0
                perturbed_normal = clean["normal_scalar"] - removed_normal
                perturbed_fault = clean["fault_scalar"] - removed_fault
                metrics = evaluate_scalar_pair(
                    perturbed_normal, perturbed_fault, state["sample_ids"], state["k"], state["limit"]
                )
                ranks = [int(state["metadata"][index]["selection_rank"]) for index in indices]
                variables = [str(state["metadata"][index]["variable"]) for index in indices]
                record = {
                    "fault_id": fault_id,
                    "method_key": TOP5_KEY,
                    "method_short": METHOD_LABELS[TOP5_KEY],
                    "loss_count": int(loss_count),
                    "combination_key": "ranks_" + "_".join(map(str, ranks)),
                    "lost_ranks": ";".join(map(str, ranks)),
                    "lost_variables": ";".join(variables),
                    "n_frozen_channels": 5,
                    "replacement_standardized_value": 0.0,
                    "original_denominator_retained": True,
                    "structural_reference_only": False,
                    **metrics,
                }
                add_clean_deltas(record, clean)
                loss_rows.append(record)

        # Matched contamination. The same base Gaussian sequence is used for the
        # Top-1 reference and Top-5 common-mode arm. The exact-energy Top-5 arm
        # uses the same base magnitude with a random unit direction in R^5.
        top1 = method_inputs[TOP1_KEY]
        top5 = method_inputs[TOP5_KEY]
        top1_clean = clean_by_method[TOP1_KEY]
        top5_clean = clean_by_method[TOP5_KEY]
        if not same_number(top1["k"], top5["k"]):
            # Different frozen k values are allowed scientifically, but the
            # audit must state that each method retains its own Step-22 state.
            pass
        for sigma in CONTAMINATION_SIGMAS:
            print(f"  Fault {fault_id}: matched contamination sigma={sigma:.1f}")
            for replicate in range(1, MONTE_CARLO_REPLICATES + 1):
                zero_first = dynamic_family(top5["family_key"])
                normal_base, normal_cube, normal_energy_error, _ = matched_noise_arrays(
                    top5_clean["normal_scalar"].shape, 5, sigma, fault_id, replicate, 0, zero_first
                )
                fault_base, fault_cube, fault_energy_error, _ = matched_noise_arrays(
                    top5_clean["fault_scalar"].shape, 5, sigma, fault_id, replicate, 1, zero_first
                )
                max_energy_error = max(max_energy_error, normal_energy_error, fault_energy_error)

                # Top-1 reference. If its family is dynamic, enforce its own
                # deterministic first-sample placeholder as well.
                top1_normal_noise = normal_base.copy()
                top1_fault_noise = fault_base.copy()
                if dynamic_family(top1["family_key"]):
                    top1_normal_noise[:, 0] = 0.0
                    top1_fault_noise[:, 0] = 0.0
                top1_metrics = evaluate_scalar_pair(
                    top1_clean["normal_scalar"] + top1_normal_noise,
                    top1_clean["fault_scalar"] + top1_fault_noise,
                    top1["sample_ids"], top1["k"], top1["limit"],
                )
                top1_record = {
                    "fault_id": fault_id,
                    "sigma": float(sigma),
                    "replicate": replicate,
                    "arm_key": "top1_reference",
                    "arm_label": "Top-1 reference",
                    "method_key": TOP1_KEY,
                    "n_channels": 1,
                    "perturbation_definition": "single standardized coordinate Gaussian sequence n(t)",
                    "energy_match_abs_error": 0.0,
                    "aggregate_match_abs_error": 0.0,
                    **top1_metrics,
                }
                add_clean_deltas(top1_record, top1_clean)
                contamination_rows.append(top1_record)

                # Top-5 exact total channel-space energy.
                normal_total_scalar_noise = normal_cube.mean(axis=2)
                fault_total_scalar_noise = fault_cube.mean(axis=2)
                total_metrics = evaluate_scalar_pair(
                    top5_clean["normal_scalar"] + normal_total_scalar_noise,
                    top5_clean["fault_scalar"] + fault_total_scalar_noise,
                    top5["sample_ids"], top5["k"], top5["limit"],
                )
                total_record = {
                    "fault_id": fault_id,
                    "sigma": float(sigma),
                    "replicate": replicate,
                    "arm_key": "top5_exact_total_energy",
                    "arm_label": "Top-5 exact total-energy",
                    "method_key": TOP5_KEY,
                    "n_channels": 5,
                    "perturbation_definition": "per-sample channel-space L2 energy exactly equals Top-1 n(t)^2",
                    "energy_match_abs_error": max(normal_energy_error, fault_energy_error),
                    "aggregate_match_abs_error": math.nan,
                    **total_metrics,
                }
                add_clean_deltas(total_record, top5_clean)
                contamination_rows.append(total_record)

                # Top-5 exact aggregate-index common-mode perturbation.
                common_normal = top5_clean["normal_scalar"] + normal_base
                common_fault = top5_clean["fault_scalar"] + fault_base
                aggregate_error = float(max(
                    np.max(np.abs((np.repeat(normal_base[:, :, None], 5, axis=2)).mean(axis=2) - normal_base)),
                    np.max(np.abs((np.repeat(fault_base[:, :, None], 5, axis=2)).mean(axis=2) - fault_base)),
                ))
                max_aggregate_error = max(max_aggregate_error, aggregate_error)
                if aggregate_error > ENERGY_TOLERANCE:
                    raise RuntimeError(f"U3_COMMON_MODE_AGGREGATE_MATCH_ERROR:{aggregate_error}")
                common_metrics = evaluate_scalar_pair(
                    common_normal, common_fault, top5["sample_ids"], top5["k"], top5["limit"]
                )
                common_record = {
                    "fault_id": fault_id,
                    "sigma": float(sigma),
                    "replicate": replicate,
                    "arm_key": "top5_common_mode_equal_index",
                    "arm_label": "Top-5 common-mode equal-index",
                    "method_key": TOP5_KEY,
                    "n_channels": 5,
                    "perturbation_definition": "same n(t) added to all five coordinates; averaged index perturbation equals Top-1 exactly",
                    "energy_match_abs_error": math.nan,
                    "aggregate_match_abs_error": aggregate_error,
                    **common_metrics,
                }
                add_clean_deltas(common_record, top5_clean)
                contamination_rows.append(common_record)

        del final_residuals, method_inputs, clean_by_method
        gc.collect()

    del testing, fault_numbers, run_numbers, normal_residuals
    gc.collect()

    clean_table = pd.DataFrame(clean_rows).sort_values(["fault_id", "method_key"])
    if len(clean_table) != len(RESERVED_FAULTS) * 2:
        raise RuntimeError("U3_CLEAN_PARITY_TABLE_INCOMPLETE")

    loss_table = pd.DataFrame(loss_rows).sort_values(
        ["fault_id", "method_key", "loss_count", "combination_key"]
    )
    expected_loss_rows = len(RESERVED_FAULTS) * (1 + 5 + 10 + 10)
    if len(loss_table) != expected_loss_rows:
        raise RuntimeError(f"U3_LOSS_TABLE_INCOMPLETE:{len(loss_table)}:{expected_loss_rows}")
    if loss_table.duplicated(["fault_id", "method_key", "loss_count", "combination_key"]).any():
        raise RuntimeError("U3_DUPLICATE_LOSS_COMBINATION")
    loss_fault = aggregate_loss_by_fault(loss_table)
    loss_crossfault = aggregate_crossfault(
        loss_fault,
        ["method_key", "method_short", "loss_count"],
        ["combination_mean_", "combination_worst_"],
    )

    contamination_replicate = pd.DataFrame(contamination_rows).sort_values(
        ["sigma", "arm_key", "fault_id", "replicate"]
    )
    expected_contamination_rows = (
        len(RESERVED_FAULTS)
        * len(CONTAMINATION_SIGMAS)
        * MONTE_CARLO_REPLICATES
        * 3
    )
    if len(contamination_replicate) != expected_contamination_rows:
        raise RuntimeError(
            f"U3_CONTAMINATION_TABLE_INCOMPLETE:{len(contamination_replicate)}:{expected_contamination_rows}"
        )
    if contamination_replicate.duplicated(["fault_id", "sigma", "replicate", "arm_key"]).any():
        raise RuntimeError("U3_DUPLICATE_CONTAMINATION_ARM")
    contamination_fault = aggregate_contamination_by_fault(contamination_replicate)
    contamination_summary = contamination_crossfault(contamination_fault)
    comparisons = build_paired_comparisons(contamination_fault)
    guardrails = reporting_guardrails()

    tables = {
        "clean_parity": clean_table,
        "loss_combination": loss_table,
        "loss_fault": loss_fault,
        "loss_crossfault": loss_crossfault,
        "contamination_replicate": contamination_replicate,
        "contamination_fault": contamination_fault,
        "contamination_crossfault": contamination_summary,
        "paired_comparisons": comparisons,
        "guardrails": guardrails,
    }
    for key, frame in tables.items():
        frame.to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")

    figure_paths = [*plot_loss(loss_crossfault, clean_table), *plot_contamination(contamination_summary)]

    final_config = {
        "record_type": "TEP_NON_QUALIFICATION_U3_MATCHED_ROBUSTNESS_AUDIT",
        "record_version": "1.0",
        "result": "PASS",
        "completed_at_utc": utc_now(),
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "registration": registration,
        "reserved_faults": RESERVED_FAULTS,
        "loss_counts": list(LOSS_COUNTS),
        "contamination_sigmas": list(CONTAMINATION_SIGMAS),
        "monte_carlo_replicates": MONTE_CARLO_REPLICATES,
        "top1_loss_structural_reference_only": True,
        "matched_total_energy_max_abs_error": max_energy_error,
        "matched_aggregate_index_max_abs_error": max_aggregate_error,
        "primary_step22r_state_changed": False,
        "detector_retuning": False,
        "physical_sensor_robustness_claim_allowed": False,
        "raw_input_robustness_claim_allowed": False,
        "u1_negative_ablation_disclosure_required": True,
        "u2_coordinate_variability_disclosure_required": True,
        "output_sha256": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in [*OUTPUT_PATHS.values()]
        },
        "step22_config_hash": sha256_file(CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"),
        "step24_source_hash": sha256_file(STEP24_SCRIPT_PATH),
        "u2_config_hash": sha256_file(CONFIG_DIR / "U2_selection_stability_simplification.json"),
        "process_data_loaded": True,
        "process_data_modified": False,
    }
    atomic_json(FINAL_CONFIG_PATH, final_config)

    print("\nU3 matched robustness audit finished successfully.")
    print(f"U3_CLEAN_PARITY_ROWS={len(clean_table)}")
    print(f"U3_MULTICHANNEL_LOSS_ROWS={len(loss_table)}")
    print(f"U3_CONTAMINATION_REPLICATE_ROWS={len(contamination_replicate)}")
    print(f"U3_MATCHED_TOTAL_ENERGY_MAX_ERROR={max_energy_error:.3e}")
    print(f"U3_MATCHED_AGGREGATE_INDEX_MAX_ERROR={max_aggregate_error:.3e}")
    print("U3_TOP1_SOLE_LOSS_STRUCTURAL_REFERENCE=YES")
    print("U3_PHYSICAL_SENSOR_ROBUSTNESS_CLAIM=NO")
    print("U3_MATCHED_ROBUSTNESS_AUDIT=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--register", action="store_true")
    actions.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_specification()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
