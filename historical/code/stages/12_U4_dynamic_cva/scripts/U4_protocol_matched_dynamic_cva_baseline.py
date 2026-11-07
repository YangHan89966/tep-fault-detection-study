#!/usr/bin/env python3
"""U4: protocol-matched dynamic canonical-variate residual baselines.

Scientific role
---------------
This post-U3 analysis adds one independently implemented dynamic external
baseline family under the same normal-training / fault-development /
normal-validation / held-out-final protocol as the frozen TEP study.

The baseline uses a normal-only PCA score representation followed by
canonical-variate reduced-rank one-step prediction from lagged score history.
Two prespecified residual statistics are monitored with one-sided CUSUM:

* CVA-RR Q residual: mean squared standardized prediction residual.
* CVA-RR L1 residual: mean absolute standardized prediction residual.

The implementation is protocol-matched and inspired by standard CVA dynamic
monitoring and canonical-variate residual literature. It is not claimed to be
an exact reproduction of any single published implementation.

Phases
------
--register  writes an immutable specification without reading process data.
--freeze    fits normal-only dynamic models and selects lag/rank/k on fault
            development runs before any final run is subsetted.
--evaluate  verifies the frozen state, then evaluates normal-evaluation and
            reserved final runs without retuning.

Scope: NON_QUALIFICATION_EXPLORATORY only.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
CONFIG_DIR = PROJECT_ROOT / "configs"
for directory in (TABLE_DIR, FIGURE_DIR, MODEL_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

VARIABLES = [*[f"xmeas_{i}" for i in range(1, 42)], *[f"xmv_{i}" for i in range(1, 12)]]
IDENTIFIERS = ["faultNumber", "simulationRun", "sample"]
RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
EXPECTED_DEVELOPMENT_RUNS = np.arange(1, 251, dtype=np.int64)
EXPECTED_FINAL_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_NORMAL_VALIDATION_RUNS = np.arange(1, 251, dtype=np.int64)
EXPECTED_NORMAL_EVALUATION_RUNS = np.arange(251, 501, dtype=np.int64)
FAULT_START_SAMPLE = 161
SAMPLE_INTERVAL_MIN = 3
MISS_PENALTY_MIN = 2400.0
EARLY_WINDOWS_MIN = [15, 30, 60, 120]
EARLY_WINDOWS_SAMPLES = {m: int(m // SAMPLE_INTERVAL_MIN) for m in EARLY_WINDOWS_MIN}
CONTROL_LIMIT_PERCENTILE = 95.0
PCA_VARIANCE_FRACTION = 0.95
LAG_GRID = [1, 2, 3]
CANONICAL_ENERGY_FRACTIONS = [0.90, 0.95]
CUSUM_K_GRID = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]
RIDGE_RELATIVE = 1e-6
METHODS = [
    {
        "method_key": "cva_rr_q_residual_cusum",
        "method_short": "CVA-RR Q",
        "score_arm": "q",
    },
    {
        "method_key": "cva_rr_l1_residual_cusum",
        "method_short": "CVA-RR L1",
        "score_arm": "l1",
    },
]
PROPOSED_KEY = "frozen_multievidence_top5_cusum"

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"
PROPOSED_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_results.csv"
PROPOSED_RUN_RESULTS_PATH = TABLE_DIR / "22_reserved_multifault_run_results.csv"

SPEC_PATH = CONFIG_DIR / "U4_dynamic_cva_baseline_specification.json"
FREEZE_CONFIG_PATH = CONFIG_DIR / "U4_dynamic_cva_baseline_freeze.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "U4_dynamic_cva_baseline_evaluation.json"
MODEL_PATH = MODEL_DIR / "U4_dynamic_cva_models.npz"
MODEL_META_PATH = MODEL_DIR / "U4_dynamic_cva_models.json"
FROZEN_PARAMETERS_PATH = TABLE_DIR / "U4_dynamic_cva_frozen_parameters.csv"
TUNING_PATH = TABLE_DIR / "U4_dynamic_cva_development_tuning.csv"
RESULTS_PATH = TABLE_DIR / "U4_dynamic_cva_results.csv"
RUN_RESULTS_PATH = TABLE_DIR / "U4_dynamic_cva_run_results.csv"
CROSSFAULT_PATH = TABLE_DIR / "U4_dynamic_cva_crossfault_summary.csv"
PAIRED_PATH = TABLE_DIR / "U4_dynamic_cva_paired_comparisons.csv"
MODEL_AUDIT_PATH = TABLE_DIR / "U4_dynamic_cva_model_audit.csv"
GUARDRAILS_PATH = TABLE_DIR / "U4_dynamic_cva_reporting_guardrails.csv"
FIG_A_PNG = FIGURE_DIR / "figU4a_dynamic_baseline_tradeoffs.png"
FIG_A_PDF = FIGURE_DIR / "figU4a_dynamic_baseline_tradeoffs.pdf"
FIG_B_PNG = FIGURE_DIR / "figU4b_dynamic_baseline_early_detection.png"
FIG_B_PDF = FIGURE_DIR / "figU4b_dynamic_baseline_early_detection.pdf"

EXPECTED_SOURCE_REFERENCES = [
    {
        "title": "Dynamic latent variable modelling and fault detection of Tennessee Eastman challenge process",
        "authors": "Cao and Samuel",
        "year": 2016,
        "doi": "10.1109/ICIT.2016.7474861",
        "role": "canonical-variate dynamic process-monitoring precedent",
    },
    {
        "title": "Canonical variate residual analysis for industrial processes fault detection",
        "authors": "Li, Li, and Liu",
        "year": 2024,
        "doi": "10.1002/cjce.25399",
        "role": "recent canonical-variate residual precedent; U4 is not an exact reproduction",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_INVALID:{path}")
    return path


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def specification_payload(current_hash: str) -> dict[str, Any]:
    return {
        "record_type": "TEP_NON_QUALIFICATION_U4_DYNAMIC_CVA_BASELINE_SPECIFICATION",
        "record_version": "1.0",
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "script": "scripts/U4_protocol_matched_dynamic_cva_baseline.py",
        "script_sha256": current_hash,
        "scientific_role": "post-U3 protocol-matched external dynamic baseline",
        "implementation_status": "independent implementation; not an exact reproduction of a single paper",
        "literature_basis": EXPECTED_SOURCE_REFERENCES,
        "reserved_faults": RESERVED_FAULTS,
        "data_roles": {
            "normal_train": "fit standardization, PCA score representation, and CVA reduced-rank predictors",
            "normal_validation": "fit scalar score standardizers and 95th-percentile run-maximum control limits",
            "fault_development_runs_1_250": "select lag, canonical rank, and CUSUM k separately for each baseline arm and fault",
            "normal_evaluation_runs_251_500": "held-out false-alarm evaluation after all states frozen",
            "fault_final_runs_251_500": "held-out fault evaluation after all states frozen",
        },
        "pca_variance_fraction": PCA_VARIANCE_FRACTION,
        "lag_grid": LAG_GRID,
        "canonical_energy_fractions": CANONICAL_ENERGY_FRACTIONS,
        "cusum_k_grid": CUSUM_K_GRID,
        "ridge_relative": RIDGE_RELATIVE,
        "methods": METHODS,
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "fault_start_sample": FAULT_START_SAMPLE,
        "sampling_interval_min": SAMPLE_INTERVAL_MIN,
        "early_detection_windows_min": EARLY_WINDOWS_MIN,
        "miss_penalty_min": MISS_PENALTY_MIN,
        "development_selection_order": [
            "full_horizon_run_detection_desc",
            "detection_within_120_min_desc",
            "post_fault_threshold_occupancy_desc",
            "all_run_penalized_median_delay_asc",
            "normal_validation_run_far_asc",
            "lag_asc",
            "canonical_rank_asc",
            "canonical_energy_fraction_asc",
            "distance_from_k_0.05_asc",
            "cusum_k_asc",
        ],
        "final_or_normal_evaluation_used_for_tuning": False,
        "proposed_detector_retuned": False,
        "cross_paper_numeric_ranking_claim": False,
        "physical_sensor_robustness_claim": False,
        "formal_qualification_evidence": False,
    }


def register() -> None:
    if (PROJECT_ROOT / "data").exists():
        raise RuntimeError("U4_REGISTER_REQUIRES_PROCESS_DATA_PATH_ABSENT")
    payload = specification_payload(script_sha256())
    payload["registered_at_utc"] = utc_now()
    if SPEC_PATH.exists():
        current = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        old = {k: v for k, v in current.items() if k != "registered_at_utc"}
        new = {k: v for k, v in payload.items() if k != "registered_at_utc"}
        if old != new:
            raise RuntimeError("U4_DIFFERENT_SPECIFICATION_ALREADY_EXISTS")
        print(f"U4 identical specification already exists: {SPEC_PATH}")
        return
    atomic_json(SPEC_PATH, payload)
    print("U4 protocol-matched dynamic baseline specification registered without process data.")
    print(f"U4 specification: {SPEC_PATH}")


def verify_specification() -> dict[str, Any]:
    require_file(SPEC_PATH, "U4_SPECIFICATION")
    actual = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = specification_payload(script_sha256())
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"U4_SPECIFICATION_DRIFT:{key}")
    return actual


def parse_integral(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values != np.rint(values)):
        raise RuntimeError(f"U4_{label}_NONINTEGRAL")
    return np.rint(values).astype(np.int64)


def validate_process_columns(frame: pd.DataFrame, label: str) -> None:
    if list(frame.columns) != [*IDENTIFIERS, *VARIABLES]:
        raise RuntimeError(f"U4_{label}_COLUMN_ORDER_DRIFT")
    values = frame[VARIABLES].to_numpy(dtype=float, copy=False)
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"U4_{label}_NONFINITE_PROCESS_VALUES")


def validate_frame(
    frame: pd.DataFrame,
    label: str,
    expected_fault: int,
    expected_runs: np.ndarray,
    samples_per_run: int,
) -> pd.DataFrame:
    validate_process_columns(frame, label)
    fault = parse_integral(frame["faultNumber"], f"{label}_FAULT")
    runs = parse_integral(frame["simulationRun"], f"{label}_RUN")
    samples = parse_integral(frame["sample"], f"{label}_SAMPLE")
    if set(np.unique(fault).tolist()) != {int(expected_fault)}:
        raise RuntimeError(f"U4_{label}_FAULT_SET_DRIFT:{np.unique(fault).tolist()}")
    if not np.array_equal(np.unique(runs), expected_runs):
        raise RuntimeError(f"U4_{label}_RUN_SET_DRIFT:{runs.min()}:{runs.max()}:{len(np.unique(runs))}")
    grouped = frame.assign(_run=runs, _sample=samples).groupby("_run", sort=True)
    counts = grouped.size().to_numpy(dtype=int)
    if not np.all(counts == samples_per_run):
        raise RuntimeError(f"U4_{label}_ROWS_PER_RUN_DRIFT")
    if samples.min() != 1 or samples.max() != samples_per_run:
        raise RuntimeError(f"U4_{label}_SAMPLE_RANGE_DRIFT:{samples.min()}:{samples.max()}")
    if frame.duplicated(["faultNumber", "simulationRun", "sample"]).any():
        raise RuntimeError(f"U4_{label}_DUPLICATE_IDENTIFIERS")
    return frame.sort_values(["simulationRun", "sample"]).reset_index(drop=True)


def load_normal(path: Path, label: str, expected_runs: np.ndarray, samples_per_run: int) -> pd.DataFrame:
    require_file(path, label)
    frame = pd.read_pickle(path)
    return validate_frame(frame, label, 0, expected_runs, samples_per_run)


def load_full_testing() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    require_file(RAW_TESTING_PATH, "U4_RAW_TESTING")
    import pyreadr
    result = pyreadr.read_r(str(RAW_TESTING_PATH))
    if len(result) != 1:
        raise RuntimeError("U4_TESTING_R_OBJECT_COUNT_DRIFT")
    frame = next(iter(result.values()))
    validate_process_columns(frame, "FULL_TESTING")
    if tuple(frame.shape) != (9_600_000, 55):
        raise RuntimeError(f"U4_FULL_TESTING_SHAPE_DRIFT:{frame.shape}")
    fault = parse_integral(frame["faultNumber"], "FULL_TESTING_FAULT")
    runs = parse_integral(frame["simulationRun"], "FULL_TESTING_RUN")
    samples = parse_integral(frame["sample"], "FULL_TESTING_SAMPLE")
    if set(np.unique(fault).tolist()) != set(range(1, 21)):
        raise RuntimeError("U4_FULL_TESTING_FAULT_SET_DRIFT")
    if runs.min() != 1 or runs.max() != 500 or samples.min() != 1 or samples.max() != 960:
        raise RuntimeError("U4_FULL_TESTING_IDENTIFIER_RANGE_DRIFT")
    return frame, fault, runs


def standardize_values(frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (frame[VARIABLES].to_numpy(dtype=float, copy=False) - mean) / std


def fit_normal_pca(normal_train: pd.DataFrame) -> dict[str, np.ndarray | int | float]:
    x = normal_train[VARIABLES].to_numpy(dtype=float, copy=False)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError("U4_NORMAL_TRAIN_INVALID_STANDARD_DEVIATION")
    z = (x - mean) / std
    cov = (z.T @ z) / (len(z) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    cumulative = np.cumsum(eigvals) / eigvals.sum()
    q = int(np.searchsorted(cumulative, PCA_VARIANCE_FRACTION, side="left") + 1)
    q = max(2, min(q, len(VARIABLES)))
    loadings = eigvecs[:, :q]
    return {
        "mean": mean,
        "std": std,
        "loadings": loadings,
        "q": q,
        "explained_fraction": float(cumulative[q - 1]),
        "eigenvalues": eigvals,
    }


def project_runs(frame: pd.DataFrame, pca: dict[str, Any]) -> dict[int, np.ndarray]:
    mean = np.asarray(pca["mean"], dtype=float)
    std = np.asarray(pca["std"], dtype=float)
    loadings = np.asarray(pca["loadings"], dtype=float)
    out: dict[int, np.ndarray] = {}
    for run_id, group in frame.groupby("simulationRun", sort=True):
        x = standardize_values(group, mean, std)
        out[int(run_id)] = x @ loadings
    return out


def build_past(scores: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    n, q = scores.shape
    if n <= lag:
        raise RuntimeError("U4_RUN_SHORTER_THAN_LAG")
    rows = n - lag
    past = np.empty((rows, q * lag), dtype=float)
    for offset in range(lag):
        past[:, offset * q : (offset + 1) * q] = scores[lag - 1 - offset : n - 1 - offset]
    future = scores[lag:]
    return past, future


def covariance_sufficient_statistics(run_scores: dict[int, np.ndarray], lag: int) -> dict[str, np.ndarray | int]:
    q = next(iter(run_scores.values())).shape[1]
    dp = q * lag
    n_total = 0
    sum_p = np.zeros(dp)
    sum_f = np.zeros(q)
    pp = np.zeros((dp, dp))
    ff = np.zeros((q, q))
    pf = np.zeros((dp, q))
    for scores in run_scores.values():
        p, f = build_past(scores, lag)
        n_total += len(p)
        sum_p += p.sum(axis=0)
        sum_f += f.sum(axis=0)
        pp += p.T @ p
        ff += f.T @ f
        pf += p.T @ f
    if n_total <= dp + 1:
        raise RuntimeError("U4_CVA_INSUFFICIENT_NORMAL_PAIRS")
    mean_p = sum_p / n_total
    mean_f = sum_f / n_total
    cov_pp = (pp - n_total * np.outer(mean_p, mean_p)) / (n_total - 1)
    cov_ff = (ff - n_total * np.outer(mean_f, mean_f)) / (n_total - 1)
    cov_pf = (pf - n_total * np.outer(mean_p, mean_f)) / (n_total - 1)
    return {"n": n_total, "mean_p": mean_p, "mean_f": mean_f, "cov_pp": cov_pp, "cov_ff": cov_ff, "cov_pf": cov_pf}


def symmetric_sqrt(matrix: np.ndarray, inverse: bool) -> tuple[np.ndarray, np.ndarray, float]:
    scale = float(np.trace(matrix) / matrix.shape[0])
    ridge = RIDGE_RELATIVE * max(scale, np.finfo(float).eps)
    regularized = matrix + ridge * np.eye(matrix.shape[0])
    vals, vecs = np.linalg.eigh(regularized)
    if np.any(~np.isfinite(vals)) or vals.min() <= 0:
        raise RuntimeError("U4_CVA_COVARIANCE_NOT_POSITIVE")
    powers = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    transform = (vecs * powers) @ vecs.T
    return transform, vals, ridge


@dataclass
class CVAModel:
    config_key: str
    lag: int
    rank: int
    energy_fraction: float
    p_mean: np.ndarray
    f_mean: np.ndarray
    a: np.ndarray
    c: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray
    canonical_correlations: np.ndarray
    ridge_pp: float
    ridge_ff: float


def fit_cva_models(run_scores: dict[int, np.ndarray]) -> tuple[list[CVAModel], list[dict[str, Any]]]:
    models: list[CVAModel] = []
    audit_rows: list[dict[str, Any]] = []
    for lag in LAG_GRID:
        print(f"U4 fitting dynamic canonical-variate models for lag {lag}")
        stats = covariance_sufficient_statistics(run_scores, lag)
        cpp_inv, eval_pp, ridge_pp = symmetric_sqrt(np.asarray(stats["cov_pp"]), inverse=True)
        cff_inv, eval_ff, ridge_ff = symmetric_sqrt(np.asarray(stats["cov_ff"]), inverse=True)
        cff_sqrt, _, _ = symmetric_sqrt(np.asarray(stats["cov_ff"]), inverse=False)
        h = cpp_inv @ np.asarray(stats["cov_pf"]) @ cff_inv
        u, singular, vt = np.linalg.svd(h, full_matrices=False)
        singular = np.clip(singular, 0.0, 1.0)
        energy = singular ** 2
        if energy.sum() <= 0:
            raise RuntimeError(f"U4_CVA_ZERO_CANONICAL_ENERGY:lag={lag}")
        ranks: list[tuple[float, int]] = []
        for fraction in CANONICAL_ENERGY_FRACTIONS:
            rank = int(np.searchsorted(np.cumsum(energy) / energy.sum(), fraction, side="left") + 1)
            rank = max(1, min(rank, len(singular)))
            ranks.append((fraction, rank))
        # Preserve separate registered fractions even if ranks coincide, but use a unique key.
        max_rank = max(rank for _, rank in ranks)
        a_max = cpp_inv @ u[:, :max_rank]
        c_max = (np.diag(singular[:max_rank]) @ vt[:max_rank, :]) @ cff_sqrt
        residual_acc: dict[tuple[float, int], dict[str, Any]] = {}
        for fraction, rank in ranks:
            residual_acc[(fraction, rank)] = {"n": 0, "sum": np.zeros(c_max.shape[1]), "sumsq": np.zeros(c_max.shape[1])}
        for scores in run_scores.values():
            p, f = build_past(scores, lag)
            z = (p - np.asarray(stats["mean_p"])) @ a_max
            for fraction, rank in ranks:
                pred = np.asarray(stats["mean_f"]) + z[:, :rank] @ c_max[:rank, :]
                residual = f - pred
                acc = residual_acc[(fraction, rank)]
                acc["n"] += len(residual)
                acc["sum"] += residual.sum(axis=0)
                acc["sumsq"] += np.square(residual).sum(axis=0)
        for fraction, rank in ranks:
            acc = residual_acc[(fraction, rank)]
            n = int(acc["n"])
            rmean = acc["sum"] / n
            variance = (acc["sumsq"] - n * np.square(rmean)) / (n - 1)
            rstd = np.sqrt(np.maximum(variance, np.finfo(float).eps))
            if np.any(~np.isfinite(rstd)) or np.any(rstd <= 0):
                raise RuntimeError(f"U4_CVA_RESIDUAL_STD_INVALID:lag={lag}:rank={rank}")
            key = f"lag{lag}_energy{int(round(100*fraction))}_rank{rank}"
            model = CVAModel(
                config_key=key,
                lag=lag,
                rank=rank,
                energy_fraction=float(fraction),
                p_mean=np.asarray(stats["mean_p"]),
                f_mean=np.asarray(stats["mean_f"]),
                a=a_max[:, :rank],
                c=c_max[:rank, :],
                residual_mean=rmean,
                residual_std=rstd,
                canonical_correlations=singular.copy(),
                ridge_pp=float(ridge_pp),
                ridge_ff=float(ridge_ff),
            )
            models.append(model)
            print(f"U4_CVA_MODEL=PASS:config={key}:lag={lag}:rank={rank}")
            audit_rows.append({
                "config_key": key,
                "lag": lag,
                "canonical_rank": rank,
                "canonical_energy_fraction_target": fraction,
                "canonical_energy_fraction_achieved": float(energy[:rank].sum() / energy.sum()),
                "canonical_correlation_min_retained": float(singular[rank - 1]),
                "canonical_correlation_max": float(singular[0]),
                "normal_training_pairs": int(stats["n"]),
                "past_dimension": int(len(model.p_mean)),
                "future_dimension": int(len(model.f_mean)),
                "ridge_pp": model.ridge_pp,
                "ridge_ff": model.ridge_ff,
            })
    if len({m.config_key for m in models}) != len(models):
        raise RuntimeError("U4_DUPLICATE_MODEL_CONFIG_KEYS")
    return models, audit_rows


def score_runs(run_scores: dict[int, np.ndarray], model: CVAModel) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_id, scores in run_scores.items():
        n = len(scores)
        p, f = build_past(scores, model.lag)
        pred = model.f_mean + ((p - model.p_mean) @ model.a) @ model.c
        residual = f - pred
        rz = (residual - model.residual_mean) / model.residual_std
        q = np.full(n, np.nan)
        l1 = np.full(n, np.nan)
        q[model.lag:] = np.mean(np.square(rz), axis=1)
        l1[model.lag:] = np.mean(np.abs(rz), axis=1)
        rows.append(pd.DataFrame({
            "simulationRun": int(run_id),
            "sample": np.arange(1, n + 1, dtype=np.int64),
            "q_raw": q,
            "l1_raw": l1,
        }))
    return pd.concat(rows, ignore_index=True)


def fit_scalar_standardizer(score_frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = score_frame[column].dropna().to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"U4_SCALAR_STANDARDIZER_INVALID:{column}")
    return {"mean": mean, "std": std}


def scalar_input(score_frame: pd.DataFrame, column: str, scaler: dict[str, float]) -> pd.DataFrame:
    values = score_frame[column].to_numpy(dtype=float)
    standardized = np.zeros(len(values), dtype=float)
    valid = np.isfinite(values)
    standardized[valid] = (values[valid] - scaler["mean"]) / scaler["std"]
    return pd.DataFrame({
        "simulationRun": score_frame["simulationRun"].to_numpy(dtype=np.int64),
        "sample": score_frame["sample"].to_numpy(dtype=np.int64),
        "scalar_input": standardized,
    })


def compute_cusum(input_frame: pd.DataFrame, k: float) -> pd.DataFrame:
    frame = input_frame.sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    output = np.empty(len(frame), dtype=float)
    for positions in frame.groupby("simulationRun", sort=True).indices.values():
        values = frame.iloc[positions]["scalar_input"].to_numpy(dtype=float)
        cumulative = np.cumsum(values - float(k))
        minimum_prefix = np.minimum.accumulate(np.concatenate(([0.0], cumulative)))
        output[positions] = cumulative - minimum_prefix[1:]
    return pd.DataFrame({
        "simulationRun": frame["simulationRun"].to_numpy(dtype=np.int64),
        "sample": frame["sample"].to_numpy(dtype=np.int64),
        "monitoring_statistic": output,
    })


def percentile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:
        return float(np.percentile(values, q, interpolation="linear"))


def determine_limit(cusum: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    run_max = cusum.groupby("simulationRun", sort=True)["monitoring_statistic"].max().reset_index(name="run_max_statistic")
    return percentile(run_max["run_max_statistic"].to_numpy(dtype=float), CONTROL_LIMIT_PERCENTILE), run_max


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    alpha = 1 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def evaluate_normal(cusum: pd.DataFrame, limit: float) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = cusum.copy()
    frame["alarm"] = frame["monitoring_statistic"] > limit
    runs = frame.groupby("simulationRun", sort=True)["alarm"].any().reset_index(name="run_alarm")
    alarm_runs = int(runs["run_alarm"].sum())
    n = len(runs)
    low, high = clopper_pearson(alarm_runs, n)
    return {
        "normal_n_runs": n,
        "normal_alarm_runs": alarm_runs,
        "normal_run_level_far": alarm_runs / n,
        "normal_run_level_far_ci_low": low,
        "normal_run_level_far_ci_high": high,
        "normal_point_level_far": float(frame["alarm"].mean()),
    }, runs


def evaluate_fault(cusum: pd.DataFrame, limit: float) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = cusum.copy()
    frame["alarm"] = frame["monitoring_statistic"] > limit
    run_rows = []
    for run_id, group in frame.groupby("simulationRun", sort=True):
        pre = group[group["sample"] < FAULT_START_SAMPLE]
        post = group[group["sample"] >= FAULT_START_SAMPLE]
        alarm_samples = post.loc[post["alarm"], "sample"].to_numpy(dtype=int)
        detected = len(alarm_samples) > 0
        first_sample = int(alarm_samples[0]) if detected else np.nan
        delay_min = float((first_sample - FAULT_START_SAMPLE) * SAMPLE_INTERVAL_MIN) if detected else np.nan
        row = {
            "simulationRun": int(run_id),
            "pre_fault_run_alarm": bool(pre["alarm"].any()),
            "post_fault_run_alarm": bool(detected),
            "first_alarm_sample": first_sample,
            "alarm_delay_min": delay_min,
            "all_run_penalized_delay_min": delay_min if detected else MISS_PENALTY_MIN,
            "post_fault_threshold_occupancy": float(post["alarm"].mean()),
        }
        for minutes, width in EARLY_WINDOWS_SAMPLES.items():
            row[f"detected_within_{minutes}_min"] = bool(detected and first_sample <= FAULT_START_SAMPLE + width)
        run_rows.append(row)
    runs = pd.DataFrame(run_rows)
    n = len(runs)
    detected = int(runs["post_fault_run_alarm"].sum())
    low, high = clopper_pearson(detected, n)
    metrics: dict[str, Any] = {
        "fault_n_runs": n,
        "fault_detected_runs": detected,
        "fault_run_level_detection_rate": detected / n,
        "fault_run_level_detection_ci_low": low,
        "fault_run_level_detection_ci_high": high,
        "fault_pre_fault_run_far": float(runs["pre_fault_run_alarm"].mean()),
        "fault_post_fault_threshold_occupancy": float(runs["post_fault_threshold_occupancy"].mean()),
        "median_alarm_delay_min": float(runs.loc[runs["post_fault_run_alarm"], "alarm_delay_min"].median()) if detected else math.nan,
        "mean_alarm_delay_min": float(runs.loc[runs["post_fault_run_alarm"], "alarm_delay_min"].mean()) if detected else math.nan,
        "all_run_penalized_median_delay_min": float(runs["all_run_penalized_delay_min"].median()),
        "all_run_penalized_mean_delay_min": float(runs["all_run_penalized_delay_min"].mean()),
    }
    for minutes in EARLY_WINDOWS_MIN:
        metrics[f"fault_detection_within_{minutes}_min"] = float(runs[f"detected_within_{minutes}_min"].mean())
    return metrics, runs


def rank_configs(models: list[CVAModel]) -> dict[str, CVAModel]:
    return {model.config_key: model for model in models}


def save_models(pca: dict[str, Any], models: list[CVAModel], validation: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "pca_mean": np.asarray(pca["mean"]),
        "pca_std": np.asarray(pca["std"]),
        "pca_loadings": np.asarray(pca["loadings"]),
        "pca_eigenvalues": np.asarray(pca["eigenvalues"]),
    }
    meta_models = []
    for index, model in enumerate(models):
        prefix = f"m{index}"
        arrays[f"{prefix}_p_mean"] = model.p_mean
        arrays[f"{prefix}_f_mean"] = model.f_mean
        arrays[f"{prefix}_a"] = model.a
        arrays[f"{prefix}_c"] = model.c
        arrays[f"{prefix}_residual_mean"] = model.residual_mean
        arrays[f"{prefix}_residual_std"] = model.residual_std
        arrays[f"{prefix}_canonical_correlations"] = model.canonical_correlations
        meta_models.append({
            "prefix": prefix,
            "config_key": model.config_key,
            "lag": model.lag,
            "rank": model.rank,
            "energy_fraction": model.energy_fraction,
            "ridge_pp": model.ridge_pp,
            "ridge_ff": model.ridge_ff,
        })
    np.savez_compressed(MODEL_PATH, **arrays)
    meta = {
        "record_type": "TEP_NON_QUALIFICATION_U4_DYNAMIC_CVA_MODEL_METADATA",
        "record_version": "1.0",
        "created_at_utc": utc_now(),
        "pca_q": int(pca["q"]),
        "pca_explained_fraction": float(pca["explained_fraction"]),
        "models": meta_models,
        "validation": validation,
        "model_npz_sha256": sha256_file(MODEL_PATH),
    }
    atomic_json(MODEL_META_PATH, meta)


def load_models() -> tuple[dict[str, Any], list[CVAModel], dict[str, Any]]:
    require_file(MODEL_PATH, "U4_MODEL_NPZ")
    require_file(MODEL_META_PATH, "U4_MODEL_METADATA")
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if meta.get("model_npz_sha256") != sha256_file(MODEL_PATH):
        raise RuntimeError("U4_MODEL_NPZ_SHA256_DRIFT")
    with np.load(MODEL_PATH, allow_pickle=False) as data:
        pca = {
            "mean": np.array(data["pca_mean"], copy=True),
            "std": np.array(data["pca_std"], copy=True),
            "loadings": np.array(data["pca_loadings"], copy=True),
            "eigenvalues": np.array(data["pca_eigenvalues"], copy=True),
            "q": int(meta["pca_q"]),
            "explained_fraction": float(meta["pca_explained_fraction"]),
        }
        models = []
        for item in meta["models"]:
            prefix = item["prefix"]
            models.append(CVAModel(
                config_key=item["config_key"],
                lag=int(item["lag"]),
                rank=int(item["rank"]),
                energy_fraction=float(item["energy_fraction"]),
                p_mean=np.array(data[f"{prefix}_p_mean"], copy=True),
                f_mean=np.array(data[f"{prefix}_f_mean"], copy=True),
                a=np.array(data[f"{prefix}_a"], copy=True),
                c=np.array(data[f"{prefix}_c"], copy=True),
                residual_mean=np.array(data[f"{prefix}_residual_mean"], copy=True),
                residual_std=np.array(data[f"{prefix}_residual_std"], copy=True),
                canonical_correlations=np.array(data[f"{prefix}_canonical_correlations"], copy=True),
                ridge_pp=float(item["ridge_pp"]),
                ridge_ff=float(item["ridge_ff"]),
            ))
    return pca, models, meta["validation"]


def validation_contract(normal_validation_scores: dict[str, pd.DataFrame], models: list[CVAModel]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for model in models:
        frame = normal_validation_scores[model.config_key]
        contract[model.config_key] = {}
        for method in METHODS:
            column = "q_raw" if method["score_arm"] == "q" else "l1_raw"
            scaler = fit_scalar_standardizer(frame, column)
            inp = scalar_input(frame, column, scaler)
            limits = {}
            for k in CUSUM_K_GRID:
                cusum = compute_cusum(inp, k)
                limit, run_max = determine_limit(cusum)
                metrics, _ = evaluate_normal(cusum, limit)
                limits[str(k)] = {
                    "control_limit": limit,
                    "normal_validation_run_far": metrics["normal_run_level_far"],
                    "run_max_min": float(run_max["run_max_statistic"].min()),
                    "run_max_max": float(run_max["run_max_statistic"].max()),
                }
            contract[model.config_key][method["method_key"]] = {"scalar_standardizer": scaler, "limits": limits}
    return contract


def development_candidate_rows(
    fault_id: int,
    fault_scores: dict[str, pd.DataFrame],
    models: list[CVAModel],
    validation: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for model in models:
        frame = fault_scores[model.config_key]
        for method in METHODS:
            mkey = method["method_key"]
            arm = method["score_arm"]
            column = "q_raw" if arm == "q" else "l1_raw"
            v = validation[model.config_key][mkey]
            inp = scalar_input(frame, column, v["scalar_standardizer"])
            for k in CUSUM_K_GRID:
                cusum = compute_cusum(inp, k)
                limit = float(v["limits"][str(k)]["control_limit"])
                metrics, _ = evaluate_fault(cusum, limit)
                rows.append({
                    "fault_id": fault_id,
                    "method_key": mkey,
                    "method_short": method["method_short"],
                    "score_arm": arm,
                    "config_key": model.config_key,
                    "lag": model.lag,
                    "canonical_rank": model.rank,
                    "canonical_energy_fraction": model.energy_fraction,
                    "cusum_k": float(k),
                    "control_limit": limit,
                    "normal_validation_run_far": float(v["limits"][str(k)]["normal_validation_run_far"]),
                    **metrics,
                })
    return pd.DataFrame(rows)


def choose_candidate(table: pd.DataFrame, method_key: str) -> pd.Series:
    sub = table[table["method_key"] == method_key].copy()
    if sub.empty:
        raise RuntimeError(f"U4_NO_DEVELOPMENT_CANDIDATES:{method_key}")
    sub["distance_from_k_005"] = np.abs(sub["cusum_k"] - 0.05)
    chosen = sub.sort_values(
        [
            "fault_run_level_detection_rate",
            "fault_detection_within_120_min",
            "fault_post_fault_threshold_occupancy",
            "all_run_penalized_median_delay_min",
            "normal_validation_run_far",
            "lag",
            "canonical_rank",
            "canonical_energy_fraction",
            "distance_from_k_005",
            "cusum_k",
        ],
        ascending=[False, False, False, True, True, True, True, True, True, True],
    ).iloc[0]
    return chosen


def subset_fault(
    full: pd.DataFrame,
    full_fault_values: np.ndarray,
    full_run_values: np.ndarray,
    fault_id: int,
    runs: np.ndarray,
    label: str,
) -> pd.DataFrame:
    mask = (full_fault_values == int(fault_id)) & np.isin(full_run_values, runs)
    subset = full.loc[mask, [*IDENTIFIERS, *VARIABLES]].copy()
    return validate_frame(subset, label, fault_id, runs, 960)


def freeze() -> None:
    verify_specification()
    if NORMAL_EVALUATION_PATH.exists():
        raise RuntimeError("U4_FREEZE_REQUIRES_NORMAL_EVALUATION_PHYSICALLY_ABSENT")
    normal_train = load_normal(NORMAL_TRAIN_PATH, "NORMAL_TRAIN", np.arange(1, 501, dtype=np.int64), 500)
    normal_validation = load_normal(NORMAL_VALIDATION_PATH, "NORMAL_VALIDATION", EXPECTED_NORMAL_VALIDATION_RUNS, 960)
    pca = fit_normal_pca(normal_train)
    print(f"U4_NORMAL_PCA=PASS:q={pca['q']}:explained={pca['explained_fraction']:.10f}")
    train_scores = project_runs(normal_train, pca)
    models, audit_rows = fit_cva_models(train_scores)
    normal_validation_scores_by_run = project_runs(normal_validation, pca)
    validation_scores = {model.config_key: score_runs(normal_validation_scores_by_run, model) for model in models}
    validation = validation_contract(validation_scores, models)
    print("U4_NORMAL_VALIDATION_CONTRACT=PASS")
    full, full_fault_values, full_run_values = load_full_testing()
    tuning_frames = []
    frozen_rows = []
    for fault_id in RESERVED_FAULTS:
        print(f"U4 freezing dynamic baseline states for reserved Fault {fault_id}")
        development = subset_fault(full, full_fault_values, full_run_values, fault_id, EXPECTED_DEVELOPMENT_RUNS, f"FAULT{fault_id}_DEVELOPMENT")
        dev_scores_by_run = project_runs(development, pca)
        fault_scores = {model.config_key: score_runs(dev_scores_by_run, model) for model in models}
        candidates = development_candidate_rows(fault_id, fault_scores, models, validation)
        tuning_frames.append(candidates)
        for method in METHODS:
            chosen = choose_candidate(candidates, method["method_key"])
            frozen_rows.append({
                "fault_id": fault_id,
                "method_key": method["method_key"],
                "method_short": method["method_short"],
                "score_arm": method["score_arm"],
                "config_key": chosen["config_key"],
                "lag": int(chosen["lag"]),
                "canonical_rank": int(chosen["canonical_rank"]),
                "canonical_energy_fraction": float(chosen["canonical_energy_fraction"]),
                "cusum_k": float(chosen["cusum_k"]),
                "control_limit": float(chosen["control_limit"]),
                "normal_validation_run_far": float(chosen["normal_validation_run_far"]),
                "development_run_detection_rate": float(chosen["fault_run_level_detection_rate"]),
                "development_detection_within_120_min": float(chosen["fault_detection_within_120_min"]),
                "development_post_fault_threshold_occupancy": float(chosen["fault_post_fault_threshold_occupancy"]),
                "development_all_run_penalized_median_delay_min": float(chosen["all_run_penalized_median_delay_min"]),
                "frozen_before_any_final_subset": True,
            })
        del development, dev_scores_by_run, fault_scores, candidates
        gc.collect()
    tuning = pd.concat(tuning_frames, ignore_index=True)
    frozen = pd.DataFrame(frozen_rows).sort_values(["fault_id", "method_key"]).reset_index(drop=True)
    if len(frozen) != len(RESERVED_FAULTS) * len(METHODS):
        raise RuntimeError("U4_FROZEN_PARAMETER_ROW_COUNT_DRIFT")
    tuning.to_csv(TUNING_PATH, index=False)
    frozen.to_csv(FROZEN_PARAMETERS_PATH, index=False)
    pd.DataFrame(audit_rows).to_csv(MODEL_AUDIT_PATH, index=False)
    save_models(pca, models, validation)
    freeze_config = {
        "record_type": "TEP_NON_QUALIFICATION_U4_DYNAMIC_CVA_FREEZE",
        "record_version": "1.0",
        "completed_at_utc": utc_now(),
        "result": "PASS",
        "specification_sha256": sha256_file(SPEC_PATH),
        "model_npz_sha256": sha256_file(MODEL_PATH),
        "model_metadata_sha256": sha256_file(MODEL_META_PATH),
        "frozen_parameters_sha256": sha256_file(FROZEN_PARAMETERS_PATH),
        "tuning_sha256": sha256_file(TUNING_PATH),
        "model_audit_sha256": sha256_file(MODEL_AUDIT_PATH),
        "pca_q": int(pca["q"]),
        "pca_explained_fraction": float(pca["explained_fraction"]),
        "model_config_count": len(models),
        "frozen_state_count": len(frozen),
        "reserved_faults": RESERVED_FAULTS,
        "all_states_frozen_before_any_final_subset": True,
        "normal_evaluation_physically_absent": True,
        "final_rows_subsetted_or_analyzed": False,
        "final_used_for_tuning": False,
        "normal_evaluation_used_for_tuning": False,
        "proposed_detector_retuned": False,
    }
    atomic_json(FREEZE_CONFIG_PATH, freeze_config)
    print("U4 all 34 dynamic baseline states frozen before any final subset.")
    print(f"U4 freeze config: {FREEZE_CONFIG_PATH}")


def verify_freeze() -> tuple[pd.DataFrame, dict[str, Any]]:
    require_file(FREEZE_CONFIG_PATH, "U4_FREEZE_CONFIG")
    config = json.loads(FREEZE_CONFIG_PATH.read_text(encoding="utf-8"))
    checks = {
        "result": config.get("result") == "PASS",
        "spec": config.get("specification_sha256") == sha256_file(SPEC_PATH),
        "model": config.get("model_npz_sha256") == sha256_file(MODEL_PATH),
        "meta": config.get("model_metadata_sha256") == sha256_file(MODEL_META_PATH),
        "parameters": config.get("frozen_parameters_sha256") == sha256_file(FROZEN_PARAMETERS_PATH),
        "tuning": config.get("tuning_sha256") == sha256_file(TUNING_PATH),
        "audit": config.get("model_audit_sha256") == sha256_file(MODEL_AUDIT_PATH),
        "frozen": config.get("all_states_frozen_before_any_final_subset") is True,
        "normal_eval_absent": config.get("normal_evaluation_physically_absent") is True,
        "final_not_subset": config.get("final_rows_subsetted_or_analyzed") is False,
        "no_final_tuning": config.get("final_used_for_tuning") is False,
        "no_eval_tuning": config.get("normal_evaluation_used_for_tuning") is False,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"U4_FREEZE_VERIFICATION_FAILED:{failed}")
    frozen = pd.read_csv(FROZEN_PARAMETERS_PATH)
    if len(frozen) != 34 or frozen.duplicated(["fault_id", "method_key"]).any():
        raise RuntimeError("U4_FROZEN_PARAMETERS_STRUCTURE_DRIFT")
    return frozen, config


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def safe_wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0.0):
        return 0.0, 1.0
    result = wilcoxon(values, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(result.statistic), float(result.pvalue)


def evaluate() -> None:
    verify_specification()
    frozen, freeze_config = verify_freeze()
    normal_evaluation = load_normal(NORMAL_EVALUATION_PATH, "NORMAL_EVALUATION", EXPECTED_NORMAL_EVALUATION_RUNS, 960)
    pca, models, validation = load_models()
    model_map = rank_configs(models)
    normal_eval_scores_by_run = project_runs(normal_evaluation, pca)
    normal_cache: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    full, full_fault_values, full_run_values = load_full_testing()
    result_rows = []
    run_frames = []
    for fault_id in RESERVED_FAULTS:
        print(f"U4 evaluating frozen dynamic baselines for reserved Fault {fault_id}")
        final = subset_fault(full, full_fault_values, full_run_values, fault_id, EXPECTED_FINAL_RUNS, f"FAULT{fault_id}_FINAL")
        final_scores_by_run = project_runs(final, pca)
        for method in METHODS:
            row = frozen[(frozen["fault_id"] == fault_id) & (frozen["method_key"] == method["method_key"])].iloc[0]
            model = model_map[str(row["config_key"])]
            mkey = method["method_key"]
            column = "q_raw" if method["score_arm"] == "q" else "l1_raw"
            scaler = validation[model.config_key][mkey]["scalar_standardizer"]
            k = float(row["cusum_k"])
            limit = float(row["control_limit"])
            expected_limit = float(validation[model.config_key][mkey]["limits"][str(k)]["control_limit"])
            if not np.isclose(limit, expected_limit, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"U4_FROZEN_CONTROL_LIMIT_DRIFT:{fault_id}:{mkey}:{limit}:{expected_limit}")
            cache_key = f"{mkey}|{model.config_key}|{k}"
            if cache_key not in normal_cache:
                nscores = score_runs(normal_eval_scores_by_run, model)
                ninp = scalar_input(nscores, column, scaler)
                ncusum = compute_cusum(ninp, k)
                nmetrics, nruns = evaluate_normal(ncusum, limit)
                normal_cache[cache_key] = (nmetrics, nruns)
            normal_metrics, normal_runs = normal_cache[cache_key]
            fscores = score_runs(final_scores_by_run, model)
            finp = scalar_input(fscores, column, scaler)
            fcusum = compute_cusum(finp, k)
            fault_metrics, fault_runs = evaluate_fault(fcusum, limit)
            result_rows.append({
                "fault_id": fault_id,
                "dataset": "frozen_final_runs_251_500",
                "method_key": mkey,
                "method_short": method["method_short"],
                "score_arm": method["score_arm"],
                "config_key": model.config_key,
                "lag": model.lag,
                "canonical_rank": model.rank,
                "canonical_energy_fraction": model.energy_fraction,
                "cusum_k": k,
                "control_limit": limit,
                **normal_metrics,
                **fault_metrics,
            })
            nr = normal_runs.copy()
            nr.insert(0, "fault_id", fault_id)
            nr.insert(1, "method_key", mkey)
            nr.insert(2, "dataset", "normal_evaluation_runs_251_500")
            nr["first_alarm_sample"] = np.nan
            nr["alarm_delay_min"] = np.nan
            nr["all_run_penalized_delay_min"] = np.nan
            nr["post_fault_threshold_occupancy"] = np.nan
            for minutes in EARLY_WINDOWS_MIN:
                nr[f"detected_within_{minutes}_min"] = np.nan
            fr = fault_runs.copy()
            fr.insert(0, "fault_id", fault_id)
            fr.insert(1, "method_key", mkey)
            fr.insert(2, "dataset", "frozen_final_runs_251_500")
            run_frames.extend([nr, fr])
        del final, final_scores_by_run
        gc.collect()
    results = pd.DataFrame(result_rows).sort_values(["fault_id", "method_key"]).reset_index(drop=True)
    run_results = pd.concat(run_frames, ignore_index=True)
    if len(results) != 34:
        raise RuntimeError("U4_RESULT_ROW_COUNT_DRIFT")
    results.to_csv(RESULTS_PATH, index=False)
    run_results.to_csv(RUN_RESULTS_PATH, index=False)

    proposed = pd.read_csv(PROPOSED_RESULTS_PATH)
    proposed = proposed[(proposed["dataset"] == "faulty_testing_final_full_960") & (proposed["method_key"] == PROPOSED_KEY)].copy()
    if sorted(proposed["fault_id"].astype(int).tolist()) != RESERVED_FAULTS:
        raise RuntimeError("U4_PROPOSED_RESULT_BINDING_DRIFT")
    proposed_runs = pd.read_csv(PROPOSED_RUN_RESULTS_PATH)
    proposed_runs = proposed_runs[(proposed_runs["dataset"] == "faulty_testing_final_full_960") & (proposed_runs["method_key"] == PROPOSED_KEY)].copy()
    if len(proposed_runs) != 4250 or proposed_runs.duplicated(["fault_id", "simulationRun"]).any():
        raise RuntimeError("U4_PROPOSED_RUN_RESULT_BINDING_DRIFT")
    proposed_runs["detected"] = proposed_runs["detected"].astype(str).str.lower().map({"true": True, "false": False})
    if proposed_runs["detected"].isna().any():
        raise RuntimeError("U4_PROPOSED_RUN_DETECTED_PARSE_FAIL")
    proposed_runs["all_run_penalized_delay_min"] = np.where(
        proposed_runs["detected"],
        pd.to_numeric(proposed_runs["alarm_delay_min"], errors="coerce"),
        MISS_PENALTY_MIN,
    )
    for minutes in EARLY_WINDOWS_MIN:
        proposed_runs[f"detected_within_{minutes}_min"] = proposed_runs["detected"] & (pd.to_numeric(proposed_runs["alarm_delay_min"], errors="coerce") <= minutes)
    proposed_early_by_fault = proposed_runs.groupby("fault_id", sort=True).agg(
        **{f"fault_detection_within_{minutes}_min": (f"detected_within_{minutes}_min", "mean") for minutes in EARLY_WINDOWS_MIN},
        all_run_penalized_mean_delay_min=("all_run_penalized_delay_min", "mean"),
    ).reset_index()
    proposed = proposed.merge(proposed_early_by_fault, on="fault_id", how="left", validate="one_to_one")
    summary_rows = []
    for label, frame, method_key in [
        ("Proposed Top-5", proposed, PROPOSED_KEY),
        *[(m["method_short"], results[results["method_key"] == m["method_key"]], m["method_key"]) for m in METHODS],
    ]:
        summary_rows.append({
            "method_key": method_key,
            "method_short": label,
            "n_faults": len(frame),
            "macro_mean_normal_run_far": float(frame["normal_run_level_far"].mean()),
            "macro_mean_run_detection": float(frame["fault_run_level_detection_rate"].mean()),
            "macro_mean_post_fault_threshold_occupancy": float(frame["fault_post_fault_point_detection_rate" if method_key == PROPOSED_KEY else "fault_post_fault_threshold_occupancy"].mean()),
            "macro_median_delay_min": float(frame["median_alarm_delay_min"].median()),
            "macro_mean_all_run_penalized_delay_min": float(frame["all_run_penalized_mean_delay_min"].mean()),
            **{f"macro_mean_detection_within_{m}_min": float(frame[f"fault_detection_within_{m}_min"].mean()) for m in EARLY_WINDOWS_MIN},
        })
    crossfault = pd.DataFrame(summary_rows)
    crossfault.to_csv(CROSSFAULT_PATH, index=False)

    comparison_rows = []
    endpoints = [
        ("normal_run_level_far", "lower"),
        ("fault_run_level_detection_rate", "higher"),
        ("fault_detection_within_15_min", "higher"),
        ("fault_detection_within_30_min", "higher"),
        ("fault_detection_within_60_min", "higher"),
        ("fault_detection_within_120_min", "higher"),
        ("fault_post_fault_threshold_occupancy", "higher"),
        ("all_run_penalized_mean_delay_min", "lower"),
    ]
    pvals_by_method: dict[str, list[float]] = {}
    temp_rows = []
    proposed_index = proposed.set_index("fault_id")
    for method in METHODS:
        mframe = results[results["method_key"] == method["method_key"]].set_index("fault_id")
        pvals = []
        for endpoint, direction in endpoints:
            pcol = "fault_post_fault_point_detection_rate" if endpoint == "fault_post_fault_threshold_occupancy" else endpoint
            dvals = mframe[endpoint].to_numpy(dtype=float) - proposed_index.loc[mframe.index, pcol].to_numpy(dtype=float)
            stat, p = safe_wilcoxon(dvals)
            pvals.append(p)
            temp_rows.append({
                "method_key": method["method_key"], "method_short": method["method_short"], "endpoint": endpoint,
                "direction": direction, "mean_dynamic_minus_proposed": float(np.mean(dvals)), "wilcoxon_statistic": stat,
                "p_value": p, "inferential_status": "paired across the 17 fixed reserved faults; post-freeze external-baseline comparison",
            })
        pvals_by_method[method["method_key"]] = pvals
    for method in METHODS:
        indices = [i for i, r in enumerate(temp_rows) if r["method_key"] == method["method_key"] and np.isfinite(r["p_value"])]
        adjusted = holm_adjust([temp_rows[i]["p_value"] for i in indices])
        for i, adj in zip(indices, adjusted):
            temp_rows[i]["holm_adjusted_p"] = float(adj)
        for i, r in enumerate(temp_rows):
            if r["method_key"] == method["method_key"] and "holm_adjusted_p" not in r:
                r["holm_adjusted_p"] = math.nan
    pd.DataFrame(temp_rows).to_csv(PAIRED_PATH, index=False)

    guardrails = pd.DataFrame([
        {"guardrail": "U4 is an independently implemented protocol-matched dynamic baseline, not an exact reproduction of Li et al. 2024 or Cao and Samuel 2016.", "required": True},
        {"guardrail": "Lag, canonical rank, and CUSUM k were selected using fault-development runs only.", "required": True},
        {"guardrail": "Normal evaluation and final fault runs were not used for tuning.", "required": True},
        {"guardrail": "The frozen Step-22-R Proposed Top-5 remains the primary analysis and was not retuned or replaced.", "required": True},
        {"guardrail": "Competitive claims must be limited to methods evaluated under the common protocol.", "required": True},
        {"guardrail": "No cross-paper numerical ranking or exact literature-reproduction claim is permitted.", "required": True},
        {"guardrail": "U4 remains finite-horizon known-fault episode monitoring in one extended TEP construction.", "required": True},
    ])
    guardrails.to_csv(GUARDRAILS_PATH, index=False)

    # Compact figures.
    plt.figure(figsize=(7.2, 5.0))
    for _, row in crossfault.iterrows():
        if row["method_key"] == PROPOSED_KEY:
            plt.scatter(100 * row["macro_mean_normal_run_far"], 100 * row["macro_mean_post_fault_threshold_occupancy"], s=90, marker="*", label=row["method_short"])
        else:
            plt.scatter(100 * row["macro_mean_normal_run_far"], 100 * row["macro_mean_post_fault_threshold_occupancy"], s=70, label=row["method_short"])
    plt.xlabel("Macro mean normal run FAR (%)")
    plt.ylabel("Macro mean post-fault threshold occupancy (%)")
    plt.title("U4 protocol-matched dynamic baseline trade-offs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_A_PNG, dpi=300)
    plt.savefig(FIG_A_PDF)
    plt.close()

    dynamic_plot = results.pivot(index="fault_id", columns="method_short", values="fault_detection_within_120_min").sort_index()
    proposed_plot = proposed.set_index("fault_id")[["fault_detection_within_120_min"]].rename(columns={"fault_detection_within_120_min": "Proposed Top-5"})
    plot = proposed_plot.join(dynamic_plot, how="inner")
    ax = plot.plot(kind="bar", figsize=(10.5, 5.2))
    ax.set_xlabel("Reserved fault")
    ax.set_ylabel("Detection within 120 min")
    ax.set_ylim(0, 1.02)
    ax.set_title("U4 dynamic-baseline early detection by reserved fault")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_B_PNG, dpi=300)
    plt.savefig(FIG_B_PDF)
    plt.close()

    final_config = {
        "record_type": "TEP_NON_QUALIFICATION_U4_DYNAMIC_CVA_EVALUATION",
        "record_version": "1.0",
        "completed_at_utc": utc_now(),
        "result": "PASS",
        "freeze_config_sha256": sha256_file(FREEZE_CONFIG_PATH),
        "frozen_parameters_sha256": sha256_file(FROZEN_PARAMETERS_PATH),
        "results_sha256": sha256_file(RESULTS_PATH),
        "run_results_sha256": sha256_file(RUN_RESULTS_PATH),
        "crossfault_summary_sha256": sha256_file(CROSSFAULT_PATH),
        "paired_comparisons_sha256": sha256_file(PAIRED_PATH),
        "guardrails_sha256": sha256_file(GUARDRAILS_PATH),
        "all_states_frozen_before_any_final_subset": freeze_config["all_states_frozen_before_any_final_subset"],
        "final_used_for_tuning": False,
        "normal_evaluation_used_for_tuning": False,
        "proposed_detector_retuned": False,
        "scientific_protocol_changed": False,
        "process_data_loaded_in_evaluate_phase": True,
    }
    atomic_json(FINAL_CONFIG_PATH, final_config)
    print("U4 protocol-matched dynamic CVA baseline evaluation finished successfully.")
    print(crossfault.to_string(index=False))


def synthetic_selftest() -> None:
    rng = np.random.default_rng(20260821)
    rows = []
    for run in range(1, 13):
        state = np.zeros(6)
        for sample in range(1, 81):
            state = 0.75 * state + rng.normal(scale=0.7, size=6)
            rows.append([0, run, sample, *state])
    cols = [*IDENTIFIERS, *[f"x{i}" for i in range(6)]]
    frame = pd.DataFrame(rows, columns=cols)
    # Local synthetic versions of variable names.
    x = frame[[f"x{i}" for i in range(6)]].to_numpy()
    mean = x.mean(0); std = x.std(0, ddof=1); z = (x-mean)/std
    cov = z.T@z/(len(z)-1); vals, vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]
    load=vecs[:,order[:4]]
    run_scores={r: ((g[[f"x{i}" for i in range(6)]].to_numpy()-mean)/std)@load for r,g in frame.groupby("simulationRun")}
    models, audit = fit_cva_models(run_scores)
    if not models or not audit:
        raise RuntimeError("U4_SYNTHETIC_MODEL_EMPTY")
    scores=score_runs(run_scores,models[0])
    if scores["q_raw"].notna().sum() <= 0 or scores["l1_raw"].notna().sum() <= 0:
        raise RuntimeError("U4_SYNTHETIC_SCORE_EMPTY")
    sc=fit_scalar_standardizer(scores,"q_raw")
    inp=scalar_input(scores,"q_raw",sc)
    cus=compute_cusum(inp,0.05); limit,_=determine_limit(cus); metrics,_=evaluate_normal(cus,limit)
    if metrics["normal_n_runs"] != 12:
        raise RuntimeError("U4_SYNTHETIC_EVALUATION_DRIFT")
    print("U4_SYNTHETIC_SELFTEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--synthetic-selftest", action="store_true")
    args = parser.parse_args()
    selected = sum([args.register, args.freeze, args.evaluate, args.synthetic_selftest])
    if selected != 1:
        raise SystemExit("Choose exactly one U4 action")
    if args.register:
        register()
    elif args.freeze:
        freeze()
    elif args.evaluate:
        evaluate()
    else:
        synthetic_selftest()


if __name__ == "__main__":
    main()
