"""Step 24: Frozen evidence-channel loss and contamination robustness.

Scientific role
---------------
This is a post-freeze secondary robustness analysis of the two sparse local
detectors already evaluated in Step 22:

* the same-family Top-1 comparator; and
* the proposed multi-evidence Top-5 detector.

No detector is fitted, selected, or tuned in Step 24.  The evidence family,
selected residual variables, orientations, normal-validation centering and
scaling, CUSUM reference value, and control limit are read from the frozen
Step-22 outputs.  Clean inputs are reconstructed and must reproduce every
Step-22 final metric before a perturbation is evaluated.

The perturbations are applied after PCA reconstruction and after the frozen
normal-validation standardization.  They therefore represent loss or
contamination of a *standardized residual-evidence channel*.  They are not a
physical sensor-failure simulation, do not recompute PCA after a sensor is
removed, and must not be described as such.

Two prespecified stress tests are used:

1. Permanent single-channel loss: one standardized channel is replaced by its
   neutral value zero for the complete run.  The original 1- or 5-channel
   denominator is retained and the frozen control limit is unchanged.
2. Persistent single-channel contamination: independent zero-mean Gaussian
   noise is added to one standardized channel at every valid evidence sample.
   For dynamic evidence families, the unavailable first sample of each run
   retains its deterministic zero placeholder.  The primary
   standard deviation is 1.0 frozen normal-validation SD; 0.5 and 2.0 are
   sensitivity severities.  Results are averaged over 20 registered Monte
   Carlo replicates and worst-channel results are also reported.

Two-stage use
-------------
Place this file in the project ``scripts`` directory, then run:

    python scripts/24_frozen_evidence_channel_robustness.py --register
    python scripts/24_frozen_evidence_channel_robustness.py --run

Registration is a post-freeze perturbation-protocol record, not a prospective
performance preregistration.
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
EXPECTED_FINAL_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_NORMAL_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (
    EXPECTED_SAMPLE_MAX - FAULT_START_SAMPLE + 1
) * SAMPLING_INTERVAL_MIN

TOP1_KEY = "chosen_family_top1_cusum"
TOP5_KEY = "frozen_multievidence_top5_cusum"
METHOD_KEYS = [TOP1_KEY, TOP5_KEY]
METHOD_LABELS = {
    TOP1_KEY: "Family Top-1",
    TOP5_KEY: "Proposed Top-5",
}
METHOD_COLORS = {
    TOP1_KEY: "#4C78A8",
    TOP5_KEY: "#E45756",
}
EXPECTED_CHANNEL_COUNTS = {TOP1_KEY: 1, TOP5_KEY: 5}

LOSS_NEUTRAL_VALUE = 0.0
FIXED_DENOMINATOR = True
CONTAMINATION_SIGMAS = (0.5, 1.0, 2.0)
PRIMARY_CONTAMINATION_SIGMA = 1.0
MONTE_CARLO_REPLICATES = 20
MONTE_CARLO_SEED = 2026072401
FAULT_BOOTSTRAP_REPLICATES = 20000
FAULT_BOOTSTRAP_SEED = 2026072402
ALPHA = 0.05
PARITY_TOLERANCE = 1e-12


NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_EVALUATION_PATH = (
    PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
)
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"

STEP22_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
STEP22_RUN_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_run_results.csv"
STEP22_PARAMETERS_PATH = (
    TABLE_DIR / "22_reserved_multifault_method_parameters.csv"
)
STEP22_SCALERS_PATH = TABLE_DIR / "22_reserved_multifault_local_scalers.csv"
STEP22_CONFIG_PATH = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"

SPECIFICATION_PATH = (
    CONFIG_DIR / "24_frozen_evidence_channel_robustness_specification.json"
)
FINAL_CONFIG_PATH = (
    CONFIG_DIR / "24_frozen_evidence_channel_robustness.json"
)

OUTPUT_PATHS = {
    "clean_parity": TABLE_DIR / "24_clean_reconstruction_parity.csv",
    "loss_channel": TABLE_DIR / "24_channel_loss_by_fault_channel.csv",
    "loss_fault": TABLE_DIR / "24_channel_loss_by_fault.csv",
    "loss_summary": TABLE_DIR / "24_channel_loss_crossfault_summary.csv",
    "contamination_replicate": (
        TABLE_DIR / "24_channel_contamination_replicate_metrics.csv"
    ),
    "contamination_channel": (
        TABLE_DIR / "24_channel_contamination_by_fault_channel.csv"
    ),
    "contamination_fault": (
        TABLE_DIR / "24_channel_contamination_by_fault.csv"
    ),
    "contamination_summary": (
        TABLE_DIR / "24_channel_contamination_crossfault_summary.csv"
    ),
    "contrasts": TABLE_DIR / "24_top5_vs_top1_robustness_contrasts.csv",
    "guardrails": TABLE_DIR / "24_reporting_guardrails.csv",
    "loss_png": FIGURE_DIR / "fig24a_single_channel_loss_robustness.png",
    "loss_pdf": FIGURE_DIR / "fig24a_single_channel_loss_robustness.pdf",
    "contamination_png": (
        FIGURE_DIR / "fig24b_single_channel_contamination_robustness.png"
    ),
    "contamination_pdf": (
        FIGURE_DIR / "fig24b_single_channel_contamination_robustness.pdf"
    ),
}


# Every item is byte-hashed at registration.  Process-data files are hashed but
# not parsed, subsetted, or statistically inspected during registration.
REQUIRED_INPUTS = {
    "step11_script": SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
    "step15_script": SCRIPT_DIR / "15_fair_sequential_baselines.py",
    "step18_script": SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
    "step20_script": SCRIPT_DIR / "20_frozen_multievidence_validation.py",
    "step22r_script": SCRIPT_DIR / "22r_reserved_multifault_frozen_validation.py",
    "step22_results": STEP22_RESULTS_PATH,
    "step22_run_results": STEP22_RUN_RESULTS_PATH,
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
    hashes = {}
    for key, path in REQUIRED_INPUTS.items():
        hashes[key] = sha256_file(require_file(path, f"freeze input {key}"))
    return hashes


def specification_payload(script_hash: str, hashes: dict[str, str]) -> dict:
    return {
        "registration_name": "Step 24 frozen evidence-channel robustness",
        "registration_status": (
            "post-freeze secondary robustness protocol; Step-22 performance "
            "was known before registration"
        ),
        "script": "scripts/24_frozen_evidence_channel_robustness.py",
        "script_sha256": script_hash,
        "input_sha256": hashes,
        "registration_reads": (
            "byte hashes only; no process dataframe is parsed or subsetted"
        ),
        "process_data_loaded_during_registration": False,
        "detector_fitting_allowed": False,
        "variable_or_family_selection_allowed": False,
        "top_k_tuning_allowed": False,
        "cusum_k_tuning_allowed": False,
        "control_limit_tuning_allowed": False,
        "frozen_population": {
            "faults": RESERVED_FAULTS,
            "normal_evaluation_runs": [251, 500],
            "final_fault_runs": [251, 500],
            "samples_per_run": 960,
            "fault_start_sample": FAULT_START_SAMPLE,
            "sampling_interval_min": SAMPLING_INTERVAL_MIN,
        },
        "methods": {
            TOP1_KEY: {
                "label": METHOD_LABELS[TOP1_KEY],
                "frozen_channels": 1,
            },
            TOP5_KEY: {
                "label": METHOD_LABELS[TOP5_KEY],
                "frozen_channels": 5,
            },
        },
        "perturbation_domain": (
            "standardized oriented residual-evidence channels after frozen "
            "PCA reconstruction and Step-22 normal-validation scaling"
        ),
        "single_channel_loss": {
            "replacement_value": LOSS_NEUTRAL_VALUE,
            "replacement_interpretation": (
                "the frozen normal-validation mean after standardization"
            ),
            "each_selected_channel_tested_once": True,
            "original_denominator_retained": FIXED_DENOMINATOR,
            "control_limit_unchanged": True,
            "primary_reports": [
                "mean over possible lost channels within fault",
                "worst channel within fault",
            ],
        },
        "single_channel_contamination": {
            "distribution": (
                "independent additive Gaussian noise in standardized "
                "evidence units at every valid evidence sample; the first "
                "sample of dynamic families retains its zero placeholder"
            ),
            "mean": 0.0,
            "sigma_grid": list(CONTAMINATION_SIGMAS),
            "primary_sigma": PRIMARY_CONTAMINATION_SIGMA,
            "monte_carlo_replicates": MONTE_CARLO_REPLICATES,
            "seed": MONTE_CARLO_SEED,
            "each_selected_channel_tested": True,
            "original_denominator_retained": FIXED_DENOMINATOR,
            "control_limit_unchanged": True,
            "primary_reports": [
                "mean over possible contaminated channels within fault",
                "worst channel within fault",
            ],
        },
        "endpoints": {
            "primary": [
                "final run-level detection",
                "all-run penalized mean alarm delay",
                "normal run-level false-alarm rate",
            ],
            "secondary": [
                "post-fault point detection",
                "all-run penalized median alarm delay",
            ],
            "miss_penalty_min": MISS_PENALTY_MIN,
        },
        "paired_crossfault_inference": {
            "unit": "reserved fault",
            "replicates": FAULT_BOOTSTRAP_REPLICATES,
            "seed": FAULT_BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95% confidence interval",
            "monte_carlo_order": (
                "average registered contamination replicates within each "
                "fault-channel case before channel and fault aggregation"
            ),
            "confirmatory_multiplicity_control": False,
        },
        "reporting_guardrails": [
            "Describe perturbations as evidence-channel tests, not physical sensor failures.",
            "The Top-1 loss result is structurally implied and is descriptive, not a novel inferential result.",
            "Do not claim universal robustness from 17 TEP faults.",
            "Do not retune or renormalize after a channel is perturbed.",
            "Report both mean-channel and worst-channel Top-5 results.",
            "If Top-5 has no measured advantage, retain only the multivariable-attribution rationale.",
            "Treat paired confidence intervals as secondary effect summaries, not multiplicity-controlled confirmatory tests.",
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
            "faults": existing.get("frozen_population", {}).get("faults")
            == RESERVED_FAULTS,
            "primary sigma": existing.get(
                "single_channel_contamination", {}
            ).get("primary_sigma")
            == PRIMARY_CONTAMINATION_SIGMA,
            "replicates": existing.get(
                "single_channel_contamination", {}
            ).get("monte_carlo_replicates")
            == MONTE_CARLO_REPLICATES,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Existing Step-24 specification differs from the current "
                f"script or frozen inputs: {failed}. Preserve it and "
                "investigate instead of overwriting it."
            )
        print("Step 24 specification already exists and matches all frozen files.")
        print(f"Registered script SHA-256: {script_hash}")
        print(f"Specification: {SPECIFICATION_PATH}")
        return

    SPECIFICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 24 post-freeze perturbation specification registered.")
    print("No process dataframe was parsed or subsetted during registration.")
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
        "faults": specification.get("frozen_population", {}).get("faults")
        == RESERVED_FAULTS,
        "loss denominator": specification.get("single_channel_loss", {}).get(
            "original_denominator_retained"
        )
        is True,
        "contamination denominator": specification.get(
            "single_channel_contamination", {}
        ).get("original_denominator_retained")
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-24 specification verification failed: {failed}")
    return specification


def load_module(path: Path, name: str):
    require_file(path, f"load frozen dependency {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies() -> dict[str, Any]:
    return {
        "step11": load_module(
            SCRIPT_DIR / "11_unpaired_runlevel_screening.py", "step11_for_step24"
        ),
        "step15": load_module(
            SCRIPT_DIR / "15_fair_sequential_baselines.py", "step15_for_step24"
        ),
        "step18": load_module(
            SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
            "step18_for_step24",
        ),
        "step20": load_module(
            SCRIPT_DIR / "20_frozen_multievidence_validation.py",
            "step20_for_step24",
        ),
    }


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_variable_order(variable_cols: list[str]) -> None:
    expected = [
        *[f"xmeas_{index}" for index in range(1, 42)],
        *[f"xmv_{index}" for index in range(1, 12)],
    ]
    if variable_cols != expected:
        raise ValueError("The expected 52-variable order changed")


def same_number(left: Any, right: Any, tolerance: float = PARITY_TOLERANCE) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=tolerance))


def load_frozen_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    results = pd.read_csv(STEP22_RESULTS_PATH, encoding="utf-8-sig")
    parameters = pd.read_csv(STEP22_PARAMETERS_PATH, encoding="utf-8-sig")
    scalers = pd.read_csv(STEP22_SCALERS_PATH, encoding="utf-8-sig")
    config = json.loads(STEP22_CONFIG_PATH.read_text(encoding="utf-8"))

    require_columns(
        results,
        [
            "fault_id",
            "method_key",
            "dataset",
            "normal_run_level_far",
            "fault_pre_fault_run_alarm_rate",
            "fault_run_level_detection_rate",
            "fault_post_fault_point_detection_rate",
            "median_alarm_delay_min",
            "mean_alarm_delay_min",
        ],
        "Step-22 results",
    )
    require_columns(
        parameters,
        [
            "fault_id",
            "method_key",
            "cusum_k",
            "control_limit",
            "control_limit_percentile",
            "final_retuning_allowed",
            "evidence_family",
        ],
        "Step-22 method parameters",
    )
    require_columns(
        scalers,
        [
            "fault_id",
            "method_key",
            "selection_rank",
            "family_key",
            "variable",
            "residual_variable",
            "orientation_sign",
            "normal_residual_center",
            "normal_validation_feature_mean",
            "normal_validation_feature_std",
        ],
        "Step-22 local scalers",
    )

    checks = {
        "reserved faults": config.get("reserved_faults") == RESERVED_FAULTS,
        "all states frozen": config.get("all_states_frozen_before_any_final_subset")
        is True,
        "final not tuned": config.get("final_used_for_tuning") is False,
        "normal evaluation not tuned": config.get(
            "normal_evaluation_used_for_tuning"
        )
        is False,
        "Step21 reconstruction": config.get("step21_reconstruction_all_passed")
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-22 frozen-state checks failed: {failed}")

    final = results[
        (results["dataset"].astype(str) == "faulty_testing_final_full_960")
        & results["method_key"].astype(str).isin(METHOD_KEYS)
    ].copy()
    params = parameters[
        parameters["method_key"].astype(str).isin(METHOD_KEYS)
    ].copy()
    local = scalers[scalers["method_key"].astype(str).isin(METHOD_KEYS)].copy()

    if len(final) != len(RESERVED_FAULTS) * len(METHOD_KEYS):
        raise ValueError("Step-22 Top-1/Top-5 final result matrix is incomplete")
    if len(params) != len(RESERVED_FAULTS) * len(METHOD_KEYS):
        raise ValueError("Step-22 Top-1/Top-5 parameter matrix is incomplete")
    if final.duplicated(["fault_id", "method_key"]).any():
        raise ValueError("Duplicate Step-22 final result row")
    if params.duplicated(["fault_id", "method_key"]).any():
        raise ValueError("Duplicate Step-22 parameter row")

    for fault_id in RESERVED_FAULTS:
        for method_key in METHOD_KEYS:
            current = local[
                (local["fault_id"].astype(int) == fault_id)
                & (local["method_key"].astype(str) == method_key)
            ].sort_values("selection_rank")
            expected_count = EXPECTED_CHANNEL_COUNTS[method_key]
            if len(current) != expected_count:
                raise ValueError(
                    f"Fault {fault_id} {method_key}: expected {expected_count} "
                    f"frozen channels, found {len(current)}"
                )
            if current["selection_rank"].astype(int).tolist() != list(
                range(1, expected_count + 1)
            ):
                raise ValueError(
                    f"Fault {fault_id} {method_key}: channel ranks changed"
                )
            if not np.isfinite(
                current["normal_validation_feature_std"].to_numpy(dtype=float)
            ).all() or not (
                current["normal_validation_feature_std"].to_numpy(dtype=float) > 0
            ).all():
                raise ValueError(
                    f"Fault {fault_id} {method_key}: invalid frozen scales"
                )

        top1 = local[
            (local["fault_id"].astype(int) == fault_id)
            & (local["method_key"].astype(str) == TOP1_KEY)
        ].sort_values("selection_rank").iloc[0]
        top5_first = local[
            (local["fault_id"].astype(int) == fault_id)
            & (local["method_key"].astype(str) == TOP5_KEY)
        ].sort_values("selection_rank").iloc[0]
        identity_columns = [
            "family_key",
            "variable",
            "residual_variable",
            "orientation_sign",
        ]
        if any(str(top1[column]) != str(top5_first[column]) for column in identity_columns):
            raise ValueError(
                f"Fault {fault_id}: Top-1 is not the first frozen Top-5 channel"
            )
        numeric_columns = [
            "normal_residual_center",
            "normal_validation_feature_mean",
            "normal_validation_feature_std",
        ]
        if any(
            not same_number(top1[column], top5_first[column])
            for column in numeric_columns
        ):
            raise ValueError(
                f"Fault {fault_id}: shared Top-1/Top-5 scaling differs"
            )

        fault_params = params[params["fault_id"].astype(int) == fault_id]
        if (fault_params["control_limit"].to_numpy(dtype=float) <= 0).any():
            raise ValueError(f"Fault {fault_id}: non-positive frozen control limit")
        if (fault_params["cusum_k"].to_numpy(dtype=float) < 0).any():
            raise ValueError(f"Fault {fault_id}: negative frozen CUSUM k")

    return final, params, local, config


def validate_normal_evaluation(
    frame: pd.DataFrame, variable_cols: list[str]
) -> pd.DataFrame:
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    require_columns(frame, required, "normal evaluation")
    output = frame[required].copy()
    for column in ("faultNumber", "simulationRun", "sample"):
        values = pd.to_numeric(output[column], errors="raise").to_numpy(dtype=float)
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
            raise ValueError(f"Normal evaluation {column} is not integer-valued")
        output[column] = rounded.astype(np.int64)
    output = output.sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    runs = np.sort(output["simulationRun"].unique())
    if not np.array_equal(runs, EXPECTED_NORMAL_RUNS):
        raise ValueError("Normal-evaluation run IDs changed")
    if output["sample"].min() != EXPECTED_SAMPLE_MIN or output["sample"].max() != EXPECTED_SAMPLE_MAX:
        raise ValueError("Normal-evaluation sample range changed")
    counts = output.groupby("simulationRun")["sample"].nunique()
    if not (counts == EXPECTED_SAMPLE_MAX).all():
        raise ValueError("Normal-evaluation runs are incomplete")
    if output.duplicated(["simulationRun", "sample"]).any():
        raise ValueError("Normal evaluation has duplicate run/sample rows")
    process = output[variable_cols].to_numpy(dtype=float, copy=False)
    if not np.isfinite(process).all():
        raise ValueError("Normal evaluation contains non-finite process values")
    return output


def build_standardized_components(
    residual_frame: pd.DataFrame,
    scaler: pd.DataFrame,
    step20: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame = residual_frame.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )
    arrays = []
    metadata = []
    for _, row in scaler.sort_values("selection_rank").iterrows():
        values, valid = step20.feature_values(
            frame,
            str(row["residual_variable"]),
            str(row["family_key"]),
            float(row["normal_residual_center"]),
        )
        z = np.zeros(len(frame), dtype=float)
        z[valid] = int(row["orientation_sign"]) * (
            values[valid] - float(row["normal_validation_feature_mean"])
        ) / float(row["normal_validation_feature_std"])
        if not np.isfinite(z).all():
            raise ValueError("A reconstructed evidence channel is non-finite")
        arrays.append(z)
        metadata.append(
            {
                "selection_rank": int(row["selection_rank"]),
                "family_key": str(row["family_key"]),
                "variable": str(row["variable"]),
                "residual_variable": str(row["residual_variable"]),
                "orientation_sign": int(row["orientation_sign"]),
            }
        )

    components = np.column_stack(arrays)
    run_ids = np.sort(frame["simulationRun"].unique())
    sample_ids = np.sort(frame["sample"].unique())
    expected_rows = len(run_ids) * len(sample_ids)
    if len(frame) != expected_rows:
        raise ValueError("Evidence frame is not a complete run-by-sample grid")
    counts = frame.groupby("simulationRun")["sample"].nunique()
    if not (counts == len(sample_ids)).all():
        raise ValueError("Evidence frame has incomplete runs")
    expected_run_column = np.repeat(run_ids, len(sample_ids))
    expected_sample_column = np.tile(sample_ids, len(run_ids))
    if not np.array_equal(
        frame["simulationRun"].to_numpy(dtype=np.int64), expected_run_column
    ) or not np.array_equal(
        frame["sample"].to_numpy(dtype=np.int64), expected_sample_column
    ):
        raise ValueError("Evidence frame run/sample order is not rectangular")
    components = components.reshape(len(run_ids), len(sample_ids), len(arrays))
    return run_ids, sample_ids, components, metadata


def compute_cusum_matrix(scalar: np.ndarray, k: float) -> np.ndarray:
    if scalar.ndim != 2:
        raise ValueError("CUSUM scalar input must be runs by samples")
    output = np.empty_like(scalar, dtype=float)
    current = np.zeros(scalar.shape[0], dtype=float)
    for sample_index in range(scalar.shape[1]):
        current = np.maximum(
            0.0, current + scalar[:, sample_index] - float(k)
        )
        output[:, sample_index] = current
    return output


def evaluate_scalar_pair(
    normal_scalar: np.ndarray,
    fault_scalar: np.ndarray,
    sample_ids: np.ndarray,
    k: float,
    limit: float,
) -> dict[str, float | int]:
    normal_stat = compute_cusum_matrix(normal_scalar, k)
    normal_alarm = normal_stat > float(limit)
    normal_run_alarm = normal_alarm.any(axis=1)

    fault_stat = compute_cusum_matrix(fault_scalar, k)
    fault_alarm = fault_stat > float(limit)
    pre_mask = sample_ids < FAULT_START_SAMPLE
    post_mask = sample_ids >= FAULT_START_SAMPLE
    if not pre_mask.any() or not post_mask.any():
        raise ValueError("Fault evaluation requires pre- and post-fault samples")
    pre_alarm = fault_alarm[:, pre_mask]
    post_alarm = fault_alarm[:, post_mask]
    pre_run_alarm = pre_alarm.any(axis=1)
    detected = post_alarm.any(axis=1)
    first_relative = post_alarm.argmax(axis=1)
    post_samples = sample_ids[post_mask]
    detected_delay = np.full(len(detected), np.nan, dtype=float)
    detected_delay[detected] = (
        post_samples[first_relative[detected]] - FAULT_START_SAMPLE
    ) * SAMPLING_INTERVAL_MIN
    penalized_delay = np.where(
        detected, detected_delay, float(MISS_PENALTY_MIN)
    )
    observed = detected_delay[detected]

    return {
        "normal_n_runs": int(normal_scalar.shape[0]),
        "normal_alarm_runs": int(normal_run_alarm.sum()),
        "normal_run_level_far": float(normal_run_alarm.mean()),
        "normal_point_level_far": float(normal_alarm.mean()),
        "fault_n_runs": int(fault_scalar.shape[0]),
        "fault_pre_fault_alarm_runs": int(pre_run_alarm.sum()),
        "fault_pre_fault_run_alarm_rate": float(pre_run_alarm.mean()),
        "fault_pre_fault_point_alarm_rate": float(pre_alarm.mean()),
        "detected_runs": int(detected.sum()),
        "right_censored_runs": int((~detected).sum()),
        "fault_run_level_detection_rate": float(detected.mean()),
        "fault_post_fault_point_detection_rate": float(post_alarm.mean()),
        "median_alarm_delay_min": (
            float(np.median(observed)) if len(observed) else np.nan
        ),
        "mean_alarm_delay_min": (
            float(np.mean(observed)) if len(observed) else np.nan
        ),
        "allrun_penalized_median_delay_min": float(np.median(penalized_delay)),
        "allrun_penalized_mean_delay_min": float(np.mean(penalized_delay)),
    }


METRIC_COLUMNS = [
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


def add_clean_deltas(record: dict[str, Any], clean: dict[str, Any]) -> None:
    for metric in METRIC_COLUMNS:
        current_value = float(record[metric])
        clean_value = float(clean[metric])
        record[f"clean_{metric}"] = clean_value
        record[f"delta_{metric}"] = current_value - clean_value


def noise_seed_sequence(
    fault_id: int,
    selection_rank: int,
    sigma: float,
    replicate: int,
    dataset_code: int,
) -> np.random.SeedSequence:
    return np.random.SeedSequence(
        [
            MONTE_CARLO_SEED,
            int(fault_id),
            int(selection_rank),
            int(round(1000.0 * sigma)),
            int(replicate),
            int(dataset_code),
        ]
    )


def aggregate_channel_level(
    channel_table: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, current in channel_table.groupby(grouping_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping_columns, keys))
        row["n_channel_cases"] = int(len(current))
        row["selected_variables"] = ";".join(
            current.sort_values("selection_rank")["variable"].astype(str)
        )
        for metric in METRIC_COLUMNS:
            values = current[metric].to_numpy(dtype=float)
            row[f"channel_mean_{metric}"] = float(np.nanmean(values))
        detection_worst = current.sort_values(
            [
                "fault_run_level_detection_rate",
                "allrun_penalized_mean_delay_min",
                "selection_rank",
            ],
            ascending=[True, False, True],
        ).iloc[0]
        delay_worst = current.sort_values(
            ["allrun_penalized_mean_delay_min", "selection_rank"],
            ascending=[False, True],
        ).iloc[0]
        far_worst = current.sort_values(
            ["normal_run_level_far", "selection_rank"],
            ascending=[False, True],
        ).iloc[0]
        point_worst = current.sort_values(
            ["fault_post_fault_point_detection_rate", "selection_rank"],
            ascending=[True, True],
        ).iloc[0]
        row["worst_detection_channel"] = str(detection_worst["variable"])
        row["worst_delay_channel"] = str(delay_worst["variable"])
        row["worst_far_channel"] = str(far_worst["variable"])
        row["worst_point_channel"] = str(point_worst["variable"])
        for metric in METRIC_COLUMNS:
            if metric in {
                "fault_run_level_detection_rate",
            }:
                source = detection_worst
            elif metric == "fault_post_fault_point_detection_rate":
                source = point_worst
            elif metric in {
                "median_alarm_delay_min",
                "mean_alarm_delay_min",
                "allrun_penalized_median_delay_min",
                "allrun_penalized_mean_delay_min",
            }:
                source = delay_worst
            elif metric in {"normal_run_level_far", "normal_point_level_far"}:
                source = far_worst
            else:
                source = detection_worst
            row[f"channel_worst_{metric}"] = float(source[metric])
        clean_columns = [column for column in current if column.startswith("clean_")]
        for column in clean_columns:
            unique = current[column].drop_duplicates()
            if len(unique) != 1:
                raise ValueError(
                    f"Clean reference changed within {grouping_columns}: {column}"
                )
            row[column] = float(unique.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def crossfault_summary(
    by_fault: pd.DataFrame,
    scenario: str,
    extra_grouping: list[str] | None = None,
) -> pd.DataFrame:
    extra_grouping = extra_grouping or []
    rows = []
    grouping = [*extra_grouping, "method_key"]
    for keys, current in by_fault.groupby(grouping, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))
        if len(current) != len(RESERVED_FAULTS):
            raise ValueError(f"Incomplete {scenario} cross-fault group: {base}")
        for aggregation in ("mean", "worst"):
            row = {
                "scenario": scenario,
                **base,
                "method_short": METHOD_LABELS[str(base["method_key"])],
                "channel_aggregation": aggregation,
                "n_faults": int(len(current)),
            }
            prefix = f"channel_{aggregation}_"
            for metric in METRIC_COLUMNS:
                values = current[f"{prefix}{metric}"].to_numpy(dtype=float)
                row[f"macro_mean_{metric}"] = float(np.nanmean(values))
                row[f"macro_median_{metric}"] = float(np.nanmedian(values))
            row["faults_with_250_of_250_detected"] = int(
                (
                    current[f"{prefix}fault_run_level_detection_rate"]
                    >= 1.0 - PARITY_TOLERANCE
                ).sum()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_interval(
    differences: np.ndarray, seed_offset: int
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    if values.shape != (len(RESERVED_FAULTS),):
        raise ValueError("Paired contrast must contain one value per fault")
    rng = np.random.default_rng(FAULT_BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(
        0,
        len(values),
        size=(FAULT_BOOTSTRAP_REPLICATES, len(values)),
    )
    bootstrap = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(bootstrap, ALPHA / 2.0)),
        float(np.quantile(bootstrap, 1.0 - ALPHA / 2.0)),
    )


def build_contrasts(
    loss_by_fault: pd.DataFrame,
    contamination_by_fault: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    datasets: list[tuple[str, float | None, pd.DataFrame]] = [
        ("single_channel_loss", None, loss_by_fault)
    ]
    for sigma in CONTAMINATION_SIGMAS:
        datasets.append(
            (
                "single_channel_contamination",
                float(sigma),
                contamination_by_fault[
                    np.isclose(
                        contamination_by_fault["contamination_sigma"], sigma
                    )
                ].copy(),
            )
        )

    metrics = [
        (
            "fault_run_level_detection_rate",
            100.0,
            "percentage points",
            "Top-5 minus Top-1",
        ),
        (
            "fault_post_fault_point_detection_rate",
            100.0,
            "percentage points",
            "Top-5 minus Top-1",
        ),
        (
            "allrun_penalized_mean_delay_min",
            1.0,
            "minutes",
            "Top-1 minus Top-5",
        ),
        (
            "normal_run_level_far",
            100.0,
            "percentage points",
            "Top-1 minus Top-5",
        ),
    ]
    seed_offset = 0
    for scenario, sigma, table in datasets:
        for aggregation in ("mean", "worst"):
            top1 = table[table["method_key"] == TOP1_KEY].set_index("fault_id")
            top5 = table[table["method_key"] == TOP5_KEY].set_index("fault_id")
            if top1.index.tolist() != RESERVED_FAULTS or top5.index.tolist() != RESERVED_FAULTS:
                top1 = top1.reindex(RESERVED_FAULTS)
                top5 = top5.reindex(RESERVED_FAULTS)
            if top1.isna().all(axis=1).any() or top5.isna().all(axis=1).any():
                raise ValueError(f"Incomplete paired contrast for {scenario}")
            for metric, multiplier, unit, definition in metrics:
                column = f"channel_{aggregation}_{metric}"
                if definition == "Top-5 minus Top-1":
                    differences = (
                        top5[column].to_numpy(dtype=float)
                        - top1[column].to_numpy(dtype=float)
                    ) * multiplier
                else:
                    differences = (
                        top1[column].to_numpy(dtype=float)
                        - top5[column].to_numpy(dtype=float)
                    ) * multiplier
                estimate, low, high = paired_bootstrap_interval(
                    differences, seed_offset
                )
                seed_offset += 1
                rows.append(
                    {
                        "scenario": scenario,
                        "contamination_sigma": sigma,
                        "channel_aggregation": aggregation,
                        "endpoint": metric,
                        "effect_definition": definition,
                        "positive_favors": "Proposed Top-5",
                        "effect_unit": unit,
                        "paired_macro_mean_effect": estimate,
                        "fault_bootstrap_ci_95_low": low,
                        "fault_bootstrap_ci_95_high": high,
                        "faults_favoring_top5": int((differences > 0).sum()),
                        "faults_tied": int(
                            np.isclose(differences, 0.0, atol=1e-12).sum()
                        ),
                        "faults_favoring_top1": int((differences < 0).sum()),
                        "n_faults": len(differences),
                    }
                )
    return pd.DataFrame(rows)


def plot_loss(by_fault: pd.DataFrame) -> list[Path]:
    x = np.arange(len(RESERVED_FAULTS))
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    for method_key in METHOD_KEYS:
        current = (
            by_fault[by_fault["method_key"] == method_key]
            .set_index("fault_id")
            .loc[RESERVED_FAULTS]
        )
        axes[0].plot(
            x,
            100.0
            * current["channel_worst_fault_run_level_detection_rate"].to_numpy(
                dtype=float
            ),
            marker="o",
            linewidth=1.8,
            label=METHOD_LABELS[method_key],
            color=METHOD_COLORS[method_key],
        )
        axes[1].plot(
            x,
            current[
                "channel_worst_allrun_penalized_mean_delay_min"
            ].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            label=METHOD_LABELS[method_key],
            color=METHOD_COLORS[method_key],
        )
    axes[0].set_ylabel("Worst-channel run detection (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].legend(loc="lower left", ncol=2)
    axes[1].set_ylabel("Worst-channel all-run mean delay (min)")
    axes[1].set_xlabel("Reserved TEP fault")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(fault) for fault in RESERVED_FAULTS])
    figure.suptitle(
        "Frozen single evidence-channel loss (fixed denominator and threshold)"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATHS["loss_png"], dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PATHS["loss_pdf"], bbox_inches="tight")
    plt.close(figure)
    return [OUTPUT_PATHS["loss_png"], OUTPUT_PATHS["loss_pdf"]]


def plot_contamination(by_fault: pd.DataFrame) -> list[Path]:
    primary = by_fault[
        np.isclose(
            by_fault["contamination_sigma"], PRIMARY_CONTAMINATION_SIGMA
        )
    ].copy()
    x = np.arange(len(RESERVED_FAULTS))
    figure, axes = plt.subplots(3, 1, figsize=(12.5, 10.2), sharex=True)
    specifications = [
        (
            "channel_worst_fault_run_level_detection_rate",
            100.0,
            "Worst-channel run detection (%)",
        ),
        (
            "channel_worst_allrun_penalized_mean_delay_min",
            1.0,
            "Worst-channel all-run mean delay (min)",
        ),
        (
            "channel_worst_normal_run_level_far",
            100.0,
            "Worst-channel normal FAR (%)",
        ),
    ]
    for method_key in METHOD_KEYS:
        current = (
            primary[primary["method_key"] == method_key]
            .set_index("fault_id")
            .loc[RESERVED_FAULTS]
        )
        for axis, (column, multiplier, _) in zip(axes, specifications):
            axis.plot(
                x,
                multiplier * current[column].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                label=METHOD_LABELS[method_key],
                color=METHOD_COLORS[method_key],
            )
    for axis, (_, _, label) in zip(axes, specifications):
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(-3, 103)
    axes[0].legend(loc="best", ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(fault) for fault in RESERVED_FAULTS])
    axes[-1].set_xlabel("Reserved TEP fault")
    figure.suptitle(
        "Frozen single evidence-channel contamination "
        f"(primary σ={PRIMARY_CONTAMINATION_SIGMA:.1f} SD; 20 replicates)"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_PATHS["contamination_png"], dpi=300, bbox_inches="tight"
    )
    figure.savefig(OUTPUT_PATHS["contamination_pdf"], bbox_inches="tight")
    plt.close(figure)
    return [
        OUTPUT_PATHS["contamination_png"],
        OUTPUT_PATHS["contamination_pdf"],
    ]


def reporting_guardrails() -> pd.DataFrame:
    rows = [
        (
            "scope",
            "Perturbations act on standardized residual-evidence channels.",
            True,
        ),
        (
            "sensor_claim",
            "Claim physical sensor-failure robustness from Step 24.",
            False,
        ),
        (
            "pca_recomputation",
            "Claim PCA was recomputed after a channel was lost.",
            False,
        ),
        (
            "retuning",
            "Claim any Step-24 variable, k, scale, or limit was retuned.",
            False,
        ),
        (
            "loss_interpretation",
            "Describe Top-1 sole-channel loss as an availability stress test.",
            True,
        ),
        (
            "top5_reporting",
            "Report both average-channel and worst-channel Top-5 results.",
            True,
        ),
        (
            "fallback",
            "If no advantage is measured, retain only multivariable attribution as the Top-5 rationale.",
            True,
        ),
        (
            "inference",
            "Describe paired bootstrap intervals as secondary effect summaries rather than multiplicity-controlled confirmatory tests.",
            True,
        ),
    ]
    return pd.DataFrame(rows, columns=["topic", "statement", "allowed_or_required"])


def run_analysis() -> None:
    registration = verify_specification()
    modules = load_dependencies()
    step11 = modules["step11"]
    step18 = modules["step18"]
    step20 = modules["step20"]
    frozen_results, parameters, scalers, step22_config = load_frozen_state()

    variable_cols = step11.load_variable_columns()
    validate_variable_order(variable_cols)

    print("=" * 96)
    print("Step 24: Frozen evidence-channel loss and contamination robustness")
    print("=" * 96)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print("All Step-22 hashes and frozen-state checks passed.")
    print("No selection, fitting, scaling, k tuning, or limit tuning is allowed.")

    normal_train = pd.read_pickle(NORMAL_TRAIN_PATH)
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_evaluation = validate_normal_evaluation(
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
    require_columns(testing, required, "Faulty-Testing RData object")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    clean_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    contamination_rows: list[dict[str, Any]] = []

    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"Fault {fault_id}: reconstructing frozen Top-1 and Top-5 inputs")
        print("=" * 96)
        final_mask = (
            (fault_numbers == fault_id)
            & run_numbers.between(251, 500)
        )
        final = testing.loc[final_mask, required].copy()
        final, _ = step18.validate_fault_frame(
            final,
            variable_cols,
            fault_id,
            EXPECTED_FINAL_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{fault_id}_step24_final",
        )
        final_residuals, current_cols = step11.compute_residual_frame(
            final, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: residual order changed")
        del final
        gc.collect()

        method_inputs: dict[str, dict[str, Any]] = {}
        for method_key in METHOD_KEYS:
            current_scaler = scalers[
                (scalers["fault_id"].astype(int) == fault_id)
                & (scalers["method_key"].astype(str) == method_key)
            ].sort_values("selection_rank")
            normal_runs, sample_ids, normal_components, channel_metadata = (
                build_standardized_components(
                    normal_residuals, current_scaler, step20
                )
            )
            final_runs, final_samples, final_components, final_metadata = (
                build_standardized_components(
                    final_residuals, current_scaler, step20
                )
            )
            if not np.array_equal(normal_runs, EXPECTED_NORMAL_RUNS):
                raise ValueError("Normal run IDs changed during reconstruction")
            if not np.array_equal(final_runs, EXPECTED_FINAL_RUNS):
                raise ValueError(f"Fault {fault_id}: final run IDs changed")
            if not np.array_equal(sample_ids, final_samples):
                raise ValueError(f"Fault {fault_id}: sample grids differ")
            if channel_metadata != final_metadata:
                raise ValueError(f"Fault {fault_id}: channel metadata differs by dataset")
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

        # The shared first channel must also reconstruct identically.
        shared_normal_difference = np.max(
            np.abs(
                method_inputs[TOP1_KEY]["normal_components"][:, :, 0]
                - method_inputs[TOP5_KEY]["normal_components"][:, :, 0]
            )
        )
        shared_fault_difference = np.max(
            np.abs(
                method_inputs[TOP1_KEY]["fault_components"][:, :, 0]
                - method_inputs[TOP5_KEY]["fault_components"][:, :, 0]
            )
        )
        if max(shared_normal_difference, shared_fault_difference) > PARITY_TOLERANCE:
            raise ValueError(
                f"Fault {fault_id}: shared Top-1 channel reconstruction differs"
            )

        clean_by_method: dict[str, dict[str, Any]] = {}
        for method_key in METHOD_KEYS:
            state = method_inputs[method_key]
            n_channels = state["normal_components"].shape[2]
            clean_normal = state["normal_components"].mean(axis=2)
            clean_fault = state["fault_components"].mean(axis=2)
            clean_metrics = evaluate_scalar_pair(
                clean_normal,
                clean_fault,
                state["sample_ids"],
                state["k"],
                state["limit"],
            )
            clean_by_method[method_key] = {
                **clean_metrics,
                "normal_scalar": clean_normal,
                "fault_scalar": clean_fault,
            }
            frozen_row = frozen_results[
                (frozen_results["fault_id"].astype(int) == fault_id)
                & (frozen_results["method_key"].astype(str) == method_key)
            ].iloc[0]
            parity_mapping = {
                "normal_run_level_far": "normal_run_level_far",
                "fault_pre_fault_run_alarm_rate": "fault_pre_fault_run_alarm_rate",
                "fault_run_level_detection_rate": "fault_run_level_detection_rate",
                "fault_post_fault_point_detection_rate": (
                    "fault_post_fault_point_detection_rate"
                ),
                "median_alarm_delay_min": "median_alarm_delay_min",
                "mean_alarm_delay_min": "mean_alarm_delay_min",
            }
            parity_differences = {}
            for reconstructed, frozen in parity_mapping.items():
                left = float(clean_metrics[reconstructed])
                right = float(frozen_row[frozen])
                difference = left - right
                parity_differences[reconstructed] = difference
                if not same_number(left, right):
                    raise ValueError(
                        f"Fault {fault_id} {method_key}: clean parity failed "
                        f"for {reconstructed}: {left} versus {right}"
                    )
            clean_rows.append(
                {
                    "fault_id": fault_id,
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "evidence_family": state["family_key"],
                    "n_channels": n_channels,
                    "selected_variables": ";".join(
                        item["variable"] for item in state["metadata"]
                    ),
                    "cusum_k": state["k"],
                    "control_limit": state["limit"],
                    "shared_rank1_normal_max_abs_difference": float(
                        shared_normal_difference
                    ),
                    "shared_rank1_fault_max_abs_difference": float(
                        shared_fault_difference
                    ),
                    "maximum_step22_metric_abs_difference": float(
                        np.nanmax(np.abs(list(parity_differences.values())))
                    ),
                    "all_step22_metrics_exact_with_tolerance_1e_12": True,
                    **clean_metrics,
                }
            )
            print(
                f"  {METHOD_LABELS[method_key]} clean parity passed: "
                f"FAR={100*clean_metrics['normal_run_level_far']:.1f}%, "
                f"detection={100*clean_metrics['fault_run_level_detection_rate']:.1f}%, "
                f"delay={clean_metrics['allrun_penalized_mean_delay_min']:.1f} min"
            )

        # Prespecified permanent single-channel loss.
        for method_key in METHOD_KEYS:
            state = method_inputs[method_key]
            clean = clean_by_method[method_key]
            n_channels = state["normal_components"].shape[2]
            for channel_index, metadata in enumerate(state["metadata"]):
                normal_loss = clean["normal_scalar"] - (
                    state["normal_components"][:, :, channel_index] / n_channels
                )
                fault_loss = clean["fault_scalar"] - (
                    state["fault_components"][:, :, channel_index] / n_channels
                )
                metrics = evaluate_scalar_pair(
                    normal_loss,
                    fault_loss,
                    state["sample_ids"],
                    state["k"],
                    state["limit"],
                )
                record: dict[str, Any] = {
                    "fault_id": fault_id,
                    "method_key": method_key,
                    "method_short": METHOD_LABELS[method_key],
                    "scenario": "single_channel_loss",
                    "evidence_family": state["family_key"],
                    "n_frozen_channels": n_channels,
                    "selection_rank": metadata["selection_rank"],
                    "variable": metadata["variable"],
                    "residual_variable": metadata["residual_variable"],
                    "orientation_sign": metadata["orientation_sign"],
                    "shared_top1_channel": metadata["selection_rank"] == 1,
                    "replacement_standardized_value": LOSS_NEUTRAL_VALUE,
                    "original_denominator_retained": FIXED_DENOMINATOR,
                    "cusum_k": state["k"],
                    "control_limit": state["limit"],
                    **metrics,
                }
                add_clean_deltas(record, clean)
                loss_rows.append(record)

        # Prespecified persistent one-channel Gaussian contamination.
        for method_key in METHOD_KEYS:
            state = method_inputs[method_key]
            clean = clean_by_method[method_key]
            n_channels = state["normal_components"].shape[2]
            for sigma in CONTAMINATION_SIGMAS:
                print(
                    f"  {METHOD_LABELS[method_key]} contamination σ={sigma:.1f}: "
                    f"{n_channels} channel(s) × {MONTE_CARLO_REPLICATES} replicates"
                )
                for channel_index, metadata in enumerate(state["metadata"]):
                    rank = int(metadata["selection_rank"])
                    for replicate in range(1, MONTE_CARLO_REPLICATES + 1):
                        normal_rng = np.random.default_rng(
                            noise_seed_sequence(
                                fault_id, rank, sigma, replicate, dataset_code=0
                            )
                        )
                        fault_rng = np.random.default_rng(
                            noise_seed_sequence(
                                fault_id, rank, sigma, replicate, dataset_code=1
                            )
                        )
                        normal_noise = normal_rng.normal(
                            0.0, sigma, size=clean["normal_scalar"].shape
                        )
                        fault_noise = fault_rng.normal(
                            0.0, sigma, size=clean["fault_scalar"].shape
                        )
                        if state["family_key"] in {
                            "absolute_increment",
                            "lag1_product",
                        }:
                            normal_noise[:, 0] = 0.0
                            fault_noise[:, 0] = 0.0
                        contaminated_normal = (
                            clean["normal_scalar"] + normal_noise / n_channels
                        )
                        contaminated_fault = (
                            clean["fault_scalar"] + fault_noise / n_channels
                        )
                        metrics = evaluate_scalar_pair(
                            contaminated_normal,
                            contaminated_fault,
                            state["sample_ids"],
                            state["k"],
                            state["limit"],
                        )
                        record = {
                            "fault_id": fault_id,
                            "method_key": method_key,
                            "method_short": METHOD_LABELS[method_key],
                            "scenario": "single_channel_contamination",
                            "evidence_family": state["family_key"],
                            "n_frozen_channels": n_channels,
                            "selection_rank": rank,
                            "variable": metadata["variable"],
                            "residual_variable": metadata["residual_variable"],
                            "orientation_sign": metadata["orientation_sign"],
                            "shared_top1_channel": rank == 1,
                            "contamination_sigma": float(sigma),
                            "monte_carlo_replicate": replicate,
                            "original_denominator_retained": FIXED_DENOMINATOR,
                            "cusum_k": state["k"],
                            "control_limit": state["limit"],
                            **metrics,
                        }
                        add_clean_deltas(record, clean)
                        contamination_rows.append(record)

        del final_residuals, method_inputs, clean_by_method
        gc.collect()

    del testing, fault_numbers, run_numbers, normal_residuals
    gc.collect()

    clean_table = pd.DataFrame(clean_rows).sort_values(
        ["fault_id", "method_key"]
    )
    if len(clean_table) != len(RESERVED_FAULTS) * 2:
        raise ValueError("Clean parity table is incomplete")
    if not clean_table["all_step22_metrics_exact_with_tolerance_1e_12"].all():
        raise ValueError("A clean Step-22 parity check failed")

    loss_channel = pd.DataFrame(loss_rows).sort_values(
        ["fault_id", "method_key", "selection_rank"]
    )
    expected_loss_rows = len(RESERVED_FAULTS) * (1 + 5)
    if len(loss_channel) != expected_loss_rows:
        raise ValueError("Single-channel-loss table is incomplete")
    loss_fault = aggregate_channel_level(
        loss_channel, ["fault_id", "method_key", "method_short", "scenario"]
    ).sort_values(["fault_id", "method_key"])
    loss_summary = crossfault_summary(loss_fault, "single_channel_loss")

    contamination_replicate = pd.DataFrame(contamination_rows).sort_values(
        [
            "fault_id",
            "contamination_sigma",
            "method_key",
            "selection_rank",
            "monte_carlo_replicate",
        ]
    )
    expected_contamination_rows = (
        len(RESERVED_FAULTS)
        * (1 + 5)
        * len(CONTAMINATION_SIGMAS)
        * MONTE_CARLO_REPLICATES
    )
    if len(contamination_replicate) != expected_contamination_rows:
        raise ValueError("Contamination replicate table is incomplete")

    identity_columns = [
        "fault_id",
        "method_key",
        "method_short",
        "scenario",
        "evidence_family",
        "n_frozen_channels",
        "selection_rank",
        "variable",
        "residual_variable",
        "orientation_sign",
        "shared_top1_channel",
        "contamination_sigma",
        "original_denominator_retained",
        "cusum_k",
        "control_limit",
    ]
    numeric_columns = [
        column
        for column in contamination_replicate.columns
        if column not in identity_columns
        and column != "monte_carlo_replicate"
        and pd.api.types.is_numeric_dtype(contamination_replicate[column])
    ]
    contamination_channel = (
        contamination_replicate.groupby(identity_columns, as_index=False, dropna=False)[
            numeric_columns
        ]
        .mean()
        .sort_values(
            ["fault_id", "contamination_sigma", "method_key", "selection_rank"]
        )
    )
    contamination_fault = aggregate_channel_level(
        contamination_channel,
        [
            "fault_id",
            "method_key",
            "method_short",
            "scenario",
            "contamination_sigma",
        ],
    ).sort_values(["fault_id", "contamination_sigma", "method_key"])
    contamination_summary = crossfault_summary(
        contamination_fault,
        "single_channel_contamination",
        extra_grouping=["contamination_sigma"],
    )

    contrasts = build_contrasts(loss_fault, contamination_fault)
    guardrails = reporting_guardrails()

    tables = {
        "clean_parity": clean_table,
        "loss_channel": loss_channel,
        "loss_fault": loss_fault,
        "loss_summary": loss_summary,
        "contamination_replicate": contamination_replicate,
        "contamination_channel": contamination_channel,
        "contamination_fault": contamination_fault,
        "contamination_summary": contamination_summary,
        "contrasts": contrasts,
        "guardrails": guardrails,
    }
    for key, table in tables.items():
        table.to_csv(OUTPUT_PATHS[key], index=False, encoding="utf-8-sig")

    figure_paths = [*plot_loss(loss_fault), *plot_contamination(contamination_fault)]

    primary_contrasts = contrasts[
        (
            contrasts["scenario"] == "single_channel_loss"
        )
        | (
            (contrasts["scenario"] == "single_channel_contamination")
            & np.isclose(
                contrasts["contamination_sigma"].fillna(-1.0),
                PRIMARY_CONTAMINATION_SIGMA,
            )
        )
    ].copy()
    config = {
        "script": "scripts/24_frozen_evidence_channel_robustness.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "specification": str(SPECIFICATION_PATH.relative_to(PROJECT_ROOT)),
        "specification_registered_at_utc": registration["registered_at_utc"],
        "specification_verified": True,
        "postfreeze_secondary_analysis": True,
        "process_domain": "standardized residual-evidence channels",
        "physical_sensor_failure_simulation": False,
        "pca_recomputed_after_perturbation": False,
        "variable_or_family_selection_performed": False,
        "cusum_k_or_limit_tuning_performed": False,
        "original_channel_denominator_retained": True,
        "loss_neutral_value": LOSS_NEUTRAL_VALUE,
        "contamination_sigmas": list(CONTAMINATION_SIGMAS),
        "primary_contamination_sigma": PRIMARY_CONTAMINATION_SIGMA,
        "monte_carlo_replicates": MONTE_CARLO_REPLICATES,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "fault_bootstrap_replicates": FAULT_BOOTSTRAP_REPLICATES,
        "fault_bootstrap_seed": FAULT_BOOTSTRAP_SEED,
        "miss_penalty_min": MISS_PENALTY_MIN,
        "reserved_faults": RESERVED_FAULTS,
        "clean_parity_all_passed": True,
        "clean_parity_maximum_absolute_metric_difference": float(
            clean_table["maximum_step22_metric_abs_difference"].max()
        ),
        "shared_rank1_maximum_absolute_component_difference": float(
            max(
                clean_table["shared_rank1_normal_max_abs_difference"].max(),
                clean_table["shared_rank1_fault_max_abs_difference"].max(),
            )
        ),
        "raw_testing_object": testing_object,
        "raw_testing_object_names": testing_names,
        "step22_config_script_hash": step22_config.get(
            "script_sha256_at_execution"
        ),
        "primary_contrasts": primary_contrasts.to_dict(orient="records"),
        "outputs": {
            key: str(path.relative_to(PROJECT_ROOT))
            for key, path in OUTPUT_PATHS.items()
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    FINAL_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 96)
    print("Step 24 summary")
    print("=" * 96)
    print("All 34 clean Top-1/Top-5 rows reproduced Step 22 exactly.")
    print("\nSingle-channel loss cross-fault summary:")
    print(
        loss_summary[
            [
                "method_short",
                "channel_aggregation",
                "macro_mean_fault_run_level_detection_rate",
                "macro_median_allrun_penalized_mean_delay_min",
                "faults_with_250_of_250_detected",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nPrimary contamination summary (σ={PRIMARY_CONTAMINATION_SIGMA:.1f}):"
    )
    print(
        contamination_summary[
            np.isclose(
                contamination_summary["contamination_sigma"],
                PRIMARY_CONTAMINATION_SIGMA,
            )
        ][
            [
                "method_short",
                "channel_aggregation",
                "macro_mean_normal_run_level_far",
                "macro_mean_fault_run_level_detection_rate",
                "macro_median_allrun_penalized_mean_delay_min",
            ]
        ].to_string(index=False)
    )
    print("\nSaved tables:")
    for key in tables:
        print(f"  {OUTPUT_PATHS[key]}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Saved configuration: {FINAL_CONFIG_PATH}")
    print("Step 24 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="register the immutable post-freeze perturbation specification",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify registration and run the frozen robustness analysis",
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
