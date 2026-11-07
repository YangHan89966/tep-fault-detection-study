#!/usr/bin/env python3
"""U2: repeated-split selection-stability and simplification audit.

Scientific role
---------------
U1 showed that the five-family adaptive protocol was not materially superior at
panel level to either fixed absolute deviation (AD) or a restricted AD-or-SL
protocol. U2 asks a different, prespecified question: does the extra family
flexibility create additional selection variance without a corresponding held-
out benefit when the complete development/evaluation pipeline is repeated on
independent run-level splits of the same 500 normal and 500 fault runs?

This is a post-U1 secondary analysis. It does not replace the original Step-22-R
primary frozen validation, and it does not promote any U1 comparator to a new
primary analysis. Ten deterministic, independently generated 250/250 splits are
used for all 17 reserved faults. Within every repeat:

* one normal split is shared across all faults;
* each fault receives an independent 250/250 development/evaluation split;
* family, Top-5 coordinates, orientations, scalers, and control limits are
  recomputed only from the repeat-specific development/normal-validation roles;
* the repeat-specific held-out normal and fault roles are used only for
  evaluation;
* the unchanged dual multiplicity gate (bootstrap CI + within-fault BH and
  permutation max-|t| FWER) is applied;
* Top-K=5, CUSUM k=0.05, and the normal-validation 95th-percentile run-maximum
  limit rule remain fixed.

The three compared protocols are Adaptive-5, Fixed-AD, and Restricted AD-or-SL.
Fixed-SL is not repeated because U1 already established three honest
abstentions and materially inferior panel performance; its U1 result remains
reported as the applicability-control branch.

All repeats reuse benchmark runs and are an internal resampling stability audit,
not external validation and not independent new process experiments.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
RECEIPT_DIR = PROJECT_ROOT / "receipts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
for directory in (TABLE_DIR, FIGURE_DIR, CONFIG_DIR, RECEIPT_DIR, CHECKPOINT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SPEC_PATH = CONFIG_DIR / "U2_selection_stability_simplification_specification.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "U2_selection_stability_simplification.json"

RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
FOCUS_FAULTS = [5, 10, 13, 17, 18, 20]
FAMILY_KEYS = [
    "signed_level",
    "absolute_deviation",
    "squared_deviation",
    "absolute_increment",
    "lag1_product",
]
FAMILY_LABELS = {
    "signed_level": "Signed level",
    "absolute_deviation": "Absolute deviation",
    "squared_deviation": "Squared deviation",
    "absolute_increment": "Absolute increment",
    "lag1_product": "Lag-1 product",
}
FAMILY_PRIORITY = {key: index + 1 for index, key in enumerate(FAMILY_KEYS)}
BRANCHES = [
    {
        "branch_key": "adaptive_5family",
        "branch_label": "Adaptive five-family",
        "allowed_families": FAMILY_KEYS,
        "rule": "choose the eligible family with largest median Top-5 absolute Hedges g; break exact ties by frozen family priority",
    },
    {
        "branch_key": "fixed_absolute_deviation",
        "branch_label": "Fixed absolute deviation",
        "allowed_families": ["absolute_deviation"],
        "rule": "use absolute deviation if at least five coordinates pass the unchanged gate; otherwise abstain",
    },
    {
        "branch_key": "restricted_ad_or_sl",
        "branch_label": "Restricted AD-or-SL",
        "allowed_families": ["signed_level", "absolute_deviation"],
        "rule": "choose between eligible signed level and absolute deviation using the unchanged family score and tie rule",
    },
]
BRANCH_BY_KEY = {row["branch_key"]: row for row in BRANCHES}
N_REPEATS = 10
MASTER_SPLIT_SEED = 2026082101
SELECTION_SEED_BASE = 2026083100
TOP_K = 5
CUSUM_K = 0.05
CONTROL_LIMIT_PERCENTILE = 95
FAULT_START_SAMPLE = 161
FINAL_SAMPLE_MAX = 960
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (FINAL_SAMPLE_MAX - FAULT_START_SAMPLE + 1) * SAMPLING_INTERVAL_MIN
EARLY_WINDOWS_MIN = [15, 30, 60, 120]
TOL = 1e-10

EXPECTED_INPUT_SHA256 = {
    "scripts/11_unpaired_runlevel_screening.py": "261f64589d6dfc2bb0098e8913358a5b5c9785fea84b67556b972d51cd3b0b86",
    "scripts/15_fair_sequential_baselines.py": "27e255613365cc025d497532aea0ef487831704b7baf6ab27c1d2d18cf642042",
    "scripts/18_faults9_15_prespecified_generalization.py": "d0f7b9b3318c93bfd4f64016d994f2f0aa2806f083df7487af44e62a04e65053",
    "scripts/19_multievidence_development_audit.py": "0a19a1c0210669c201d7cbe42c5824aed2a38cb8a08405c68879cc415a30ee6b",
    "scripts/20_frozen_multievidence_validation.py": "499f1a51d11fbe563b833722da67d002c464fb4acf6e4056ca064b42b9f79825",
    "results/tables/02_variable_list.csv": "7d64ddbab285831773cc6f49b1968e0b2bed3f5a856cea256d5e29ec52b58494",
    "results/tables/21_reserved_multifault_frozen_protocol.csv": "764ab3ed6bd364a92ccfe562746e6bf3025bac965e3572c58ee1c2ac0891acc6",
    "results/tables/U1_family_adaptation_frozen_protocol.csv": "9e5e20a2b88a0b15963eb95fa7edc23ef8bb1f53b43dc22302f92d144c6aca48",
    "results/tables/U1_family_adaptation_selected_coordinates.csv": "378e76f4d500339f1f5af9a627969fbebdca89e19a58f29d3a4487c209e3a4e2",
    "results/tables/U1_family_adaptation_crossfault_summary.csv": "b33f17375010390a029474807fa59a4e72eea7988494eff82fe4910b39eba90c",
    "configs/U1_family_adaptation_ablation.json": "286ad3009552a0fc7771199e0e1b118ac30aa540e0dbd43cc1dd1c787de72e00",
}

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_INVALID:{path}")
    return path


def load_module(relative: str, module_name: str):
    path = require_regular(PROJECT_ROOT / relative, f"U2_MODULE_{module_name}")
    expected = EXPECTED_INPUT_SHA256[relative]
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"U2_MODULE_SHA256_MISMATCH:{relative}:{observed}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"U2_MODULE_SPEC_FAIL:{relative}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_exact_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = require_regular(PROJECT_ROOT / relative, f"U2_INPUT_{relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"U2_INPUT_SHA256_MISMATCH:{relative}:{digest}")
        observed[relative] = digest
    return observed


def normal_split_seed(repeat_index: int) -> int:
    return MASTER_SPLIT_SEED + repeat_index * 10_000 + 17


def fault_split_seed(repeat_index: int, fault_id: int) -> int:
    return MASTER_SPLIT_SEED + repeat_index * 100_000 + fault_id * 101 + 29


def selection_seed(repeat_index: int) -> int:
    return SELECTION_SEED_BASE + repeat_index * 1_000_000


def deterministic_split(run_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(run_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) != 500 or len(np.unique(ids)) != 500:
        raise RuntimeError("U2_SPLIT_INPUT_GEOMETRY_INVALID")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    development = np.sort(shuffled[:250])
    evaluation = np.sort(shuffled[250:])
    if len(set(development) & set(evaluation)) != 0:
        raise RuntimeError("U2_SPLIT_OVERLAP")
    if sorted(np.concatenate([development, evaluation]).tolist()) != sorted(ids.tolist()):
        raise RuntimeError("U2_SPLIT_UNION_DRIFT")
    return development, evaluation


def specification_payload(script_hash: str) -> dict[str, Any]:
    return {
        "analysis_name": "U2 repeated-split selection-stability and simplification audit",
        "analysis_status": "post-U1 secondary analysis; original Step-22-R primary results unchanged",
        "script": "scripts/U2_selection_stability_simplification.py",
        "script_sha256": script_hash,
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "primary_results_replaced": False,
        "u1_negative_ablation_disclosed": True,
        "reserved_faults": RESERVED_FAULTS,
        "focus_faults": FOCUS_FAULTS,
        "n_repeats": N_REPEATS,
        "normal_split_shared_across_faults_within_repeat": True,
        "fault_split_independent_by_fault_within_repeat": True,
        "development_runs_per_repeat": 250,
        "evaluation_runs_per_repeat": 250,
        "master_split_seed": MASTER_SPLIT_SEED,
        "normal_split_seeds": [normal_split_seed(index) for index in range(1, N_REPEATS + 1)],
        "fault_split_seed_rule": "MASTER_SPLIT_SEED + repeat*100000 + fault_id*101 + 29",
        "selection_seed_base": SELECTION_SEED_BASE,
        "selection_seed_rule": "SELECTION_SEED_BASE + repeat*1000000; Step-19 adds fault-specific offsets",
        "branches": BRANCHES,
        "fixed_signed_level_repeated": False,
        "fixed_signed_level_reason": "U1 already established 3/17 abstentions and materially inferior performance; retained as U1 applicability control",
        "candidate_gate": (
            "bootstrap 95% CI excludes zero AND within-fault BH q<=0.05 across 260 candidates "
            "AND permutation max-|t| FWER-adjusted p<=0.05"
        ),
        "multiplicity_scope": "within each fault across the fixed 5 x 52 candidate universe",
        "family_eligibility": "at least five coordinates pass the unchanged dual gate",
        "family_choice_rule": "largest median selected absolute Hedges g with frozen priority tie-break",
        "top_k": TOP_K,
        "cusum_k": CUSUM_K,
        "control_limit_rule": "95th percentile of repeat-specific normal-development run maxima",
        "normal_training_fixed": True,
        "normal_development_used_for_selection_scaling_and_limit": True,
        "normal_evaluation_used_only_for_repeat-specific FAR": True,
        "fault_development_used_only_for_selection": True,
        "fault_evaluation_used_only_for held-out evaluation": True,
        "early_detection_windows_min": EARLY_WINDOWS_MIN,
        "miss_penalty_min": MISS_PENALTY_MIN,
        "selection_stability_endpoints": [
            "family recovery versus original frozen branch",
            "exact Top-5 set recovery",
            "ordered Top-5 recovery",
            "Top-5 Jaccard overlap",
            "orientation agreement on overlapping coordinates",
            "branch eligibility/abstention",
        ],
        "performance_endpoints": [
            "normal run-level FAR",
            "15/30/60/120-min run detection",
            "full-horizon run detection",
            "post-fault threshold occupancy",
            "all-run penalized median and mean delay",
        ],
        "inference_boundary": (
            "internal post-freeze repeated-split stability on reused benchmark runs; original final/evaluation runs "
            "can enter development roles in secondary repeats, so this is not a new holdout, external validation, "
            "independent process replication, or prospective comparison"
        ),
        "interpretation_rule": (
            "U1 must be reported; U2 tests whether added family flexibility increases selection variance "
            "without held-out benefit. No branch may be promoted to a new primary analysis post hoc."
        ),
        "network_access": False,
        "package_install_or_upgrade": False,
        "formal_p71_p72_code_invoked": False,
        "input_sha256": EXPECTED_INPUT_SHA256,
    }


def register_specification() -> None:
    if (PROJECT_ROOT / "data").exists():
        raise RuntimeError("U2_PROCESS_DATA_PATH_PRESENT_DURING_REGISTRATION")
    script_hash = sha256_file(Path(__file__))
    payload = specification_payload(script_hash)
    payload["registered_at_utc"] = utc_now()
    if SPEC_PATH.exists():
        existing = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        comparable_existing = {k: v for k, v in existing.items() if k != "registered_at_utc"}
        comparable_new = {k: v for k, v in payload.items() if k != "registered_at_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError("U2_DIFFERENT_SPECIFICATION_ALREADY_EXISTS")
        print(f"U2_IDENTICAL_SPECIFICATION_ALREADY_EXISTS={SPEC_PATH}")
        return
    atomic_json(SPEC_PATH, payload)
    print("U2_SPECIFICATION_REGISTERED_WITHOUT_PROCESS_DATA=PASS")
    print(f"U2_SPECIFICATION={SPEC_PATH}")


def verify_specification() -> dict[str, Any]:
    require_regular(SPEC_PATH, "U2_SPECIFICATION")
    registration = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = specification_payload(sha256_file(Path(__file__)))
    failures = [key for key, value in expected.items() if registration.get(key) != value]
    if failures:
        raise RuntimeError(f"U2_SPECIFICATION_VERIFICATION_FAILED:{failures}")
    return registration


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        raise RuntimeError("U2_BOOLEAN_PARSE_FAILED")
    return converted.astype(bool)


def top5_for_family(metrics: pd.DataFrame, fault_id: int, family_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = metrics[
        (metrics["fault_id"].astype(int) == int(fault_id))
        & (metrics["family_key"].astype(str) == family_key)
    ].copy()
    if len(current) != 52:
        raise RuntimeError(f"U2_FAMILY_GEOMETRY_DRIFT:{fault_id}:{family_key}:{len(current)}")
    primary = parse_bool(current["primary_stable_global_fdr"])
    strict = parse_bool(current["strict_stable_max_t_fwer"])
    joint = current[primary & strict].copy()
    sorting = [
        "abs_hedges_g",
        "bootstrap_direction_confidence",
        "bootstrap_top5_selection_frequency",
        "variable",
    ]
    selected = joint.sort_values(sorting, ascending=[False, False, False, True]).head(TOP_K).copy()
    eligible = len(joint) >= TOP_K and len(selected) == TOP_K
    if eligible:
        selected = selected.reset_index(drop=True)
        selected.insert(1, "selection_rank", np.arange(1, TOP_K + 1))
    else:
        selected = selected.iloc[0:0].copy()
        selected.insert(1, "selection_rank", pd.Series(dtype=int))
    audit = {
        "fault_id": int(fault_id),
        "family_key": family_key,
        "n_primary_gate": int(primary.sum()),
        "n_joint_gate": int((primary & strict).sum()),
        "eligible": bool(eligible),
        "median_selected_abs_hedges_g": float(selected["abs_hedges_g"].median()) if eligible else math.nan,
        "family_priority": int(FAMILY_PRIORITY[family_key]),
        "minimum_selected_bootstrap_frequency": float(selected["bootstrap_top5_selection_frequency"].min()) if eligible else math.nan,
    }
    return selected, audit


def choose_repeat_branches(metrics: pd.DataFrame, fault_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_cache: dict[str, pd.DataFrame] = {}
    audit_cache: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for family_key in FAMILY_KEYS:
        selected, audit = top5_for_family(metrics, fault_id, family_key)
        family_cache[family_key] = selected
        audit_cache[family_key] = audit
        audits.append(audit)

    protocol_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.DataFrame] = []
    for branch in BRANCHES:
        options: list[tuple[float, int, str]] = []
        for family_key in branch["allowed_families"]:
            audit = audit_cache[family_key]
            if audit["eligible"]:
                options.append((float(audit["median_selected_abs_hedges_g"]), -int(audit["family_priority"]), family_key))
        chosen = max(options)[2] if options else "none"
        if chosen == "none":
            protocol_rows.append(
                {
                    "branch_key": branch["branch_key"],
                    "branch_label": branch["branch_label"],
                    "fault_id": fault_id,
                    "branch_status": "abstain_no_eligible_allowed_family",
                    "family_key": "none",
                    "family_label": "None",
                    "selected_count": 0,
                    "selected_variables": "",
                    "orientation_signs": "",
                    "family_score_median_abs_hedges_g": math.nan,
                }
            )
            continue
        selected = family_cache[chosen].copy()
        selected.insert(0, "branch_label", branch["branch_label"])
        selected.insert(0, "branch_key", branch["branch_key"])
        selected_rows.append(selected)
        protocol_rows.append(
            {
                "branch_key": branch["branch_key"],
                "branch_label": branch["branch_label"],
                "fault_id": fault_id,
                "branch_status": "eligible_repeat_development_only",
                "family_key": chosen,
                "family_label": FAMILY_LABELS[chosen],
                "selected_count": TOP_K,
                "selected_variables": ";".join(selected["variable"].astype(str)),
                "orientation_signs": ";".join(selected["orientation_sign"].astype(int).astype(str)),
                "family_score_median_abs_hedges_g": float(audit_cache[chosen]["median_selected_abs_hedges_g"]),
            }
        )
    protocol = pd.DataFrame(protocol_rows)
    selected_long = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return protocol, selected_long, pd.DataFrame(audits)


def validate_500_run_pool(frame: pd.DataFrame, variable_cols: list[str], dataset_name: str, expected_fault: int | None) -> pd.DataFrame:
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{dataset_name}_MISSING_COLUMNS:{missing}")
    output = frame[required].copy()
    for column in ["faultNumber", "simulationRun", "sample"]:
        values = pd.to_numeric(output[column], errors="raise").to_numpy(dtype=float)
        rounded = np.rint(values)
        if np.any(~np.isfinite(values)) or not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"{dataset_name}_{column}_NOT_INTEGER")
        output[column] = rounded.astype(np.int64)
    output = output.sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    actual_runs = np.sort(output["simulationRun"].unique())
    if not np.array_equal(actual_runs, np.arange(1, 501, dtype=np.int64)):
        raise RuntimeError(f"{dataset_name}_RUN_SET_DRIFT")
    counts = output.groupby("simulationRun")["sample"].nunique()
    if len(counts) != 500 or not (counts == 960).all():
        raise RuntimeError(f"{dataset_name}_INCOMPLETE_RUNS")
    if output.duplicated(["simulationRun", "sample"]).any():
        raise RuntimeError(f"{dataset_name}_DUPLICATE_ROWS")
    if expected_fault is not None and set(output["faultNumber"].unique()) != {int(expected_fault)}:
        raise RuntimeError(f"{dataset_name}_FAULT_ID_DRIFT")
    if np.any(~np.isfinite(output[variable_cols].to_numpy(dtype=float, copy=False))):
        raise RuntimeError(f"{dataset_name}_NONFINITE_PROCESS_VALUES")
    return output


def subset_residual_runs(frame: pd.DataFrame, run_ids: np.ndarray, dataset_name: str) -> pd.DataFrame:
    ids = set(np.asarray(run_ids, dtype=int).tolist())
    subset = frame[frame["simulationRun"].isin(ids)].copy()
    actual = np.sort(subset["simulationRun"].unique())
    if not np.array_equal(actual, np.sort(np.asarray(run_ids, dtype=np.int64))):
        raise RuntimeError(f"{dataset_name}_RUN_SUBSET_DRIFT")
    counts = subset.groupby("simulationRun")["sample"].nunique()
    if len(counts) != 250 or not (counts == 960).all():
        raise RuntimeError(f"{dataset_name}_INCOMPLETE_SUBSET")
    return subset.sort_values(["simulationRun", "sample"]).reset_index(drop=True)


def add_early_metrics(metrics: dict[str, Any], runs: pd.DataFrame) -> dict[str, Any]:
    output = dict(metrics)
    detected = runs["detected"].astype(bool)
    delays = pd.to_numeric(runs["alarm_delay_min"], errors="coerce")
    for window in EARLY_WINDOWS_MIN:
        flag = detected & (delays <= window)
        output[f"detection_within_{window}min_rate"] = float(flag.mean())
        output[f"detected_within_{window}min_runs"] = int(flag.sum())
    penalized = np.where(detected.to_numpy(), delays.fillna(MISS_PENALTY_MIN).to_numpy(dtype=float), MISS_PENALTY_MIN)
    output["allrun_penalized_delay_median_min"] = float(np.median(penalized))
    output["allrun_penalized_delay_mean_min"] = float(np.mean(penalized))
    output["post_fault_threshold_occupancy"] = float(metrics["fault_post_fault_point_detection_rate"])
    return output


def empty_abstention_result(branch: pd.Series, repeat_index: int, fault_id: int, evaluation_run_ids: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    result: dict[str, Any] = {
        "repeat_index": repeat_index,
        "branch_key": str(branch["branch_key"]),
        "branch_label": str(branch["branch_label"]),
        "fault_id": fault_id,
        "branch_status": str(branch["branch_status"]),
        "family_key": "none",
        "family_label": "None",
        "selected_variables": "",
        "orientation_signs": "",
        "selected_count": 0,
        "cusum_k": CUSUM_K,
        "control_limit": math.nan,
        "normal_run_level_far": math.nan,
        "fault_run_level_detection_rate": 0.0,
        "fault_post_fault_point_detection_rate": 0.0,
        "post_fault_threshold_occupancy": 0.0,
        "allrun_penalized_delay_median_min": float(MISS_PENALTY_MIN),
        "allrun_penalized_delay_mean_min": float(MISS_PENALTY_MIN),
    }
    for window in EARLY_WINDOWS_MIN:
        result[f"detection_within_{window}min_rate"] = 0.0
        result[f"detected_within_{window}min_runs"] = 0
    runs = pd.DataFrame(
        {
            "repeat_index": repeat_index,
            "branch_key": str(branch["branch_key"]),
            "branch_label": str(branch["branch_label"]),
            "fault_id": fault_id,
            "simulationRun": np.asarray(evaluation_run_ids, dtype=int),
            "detected": False,
            "alarm_delay_min": np.nan,
            "right_censored": True,
            "allrun_penalized_delay_min": float(MISS_PENALTY_MIN),
            "branch_status": str(branch["branch_status"]),
        }
    )
    return result, runs


def evaluate_branch(step11, step15, step18, step20, repeat_index: int, fault_id: int, branch_row: pd.Series, definition: pd.DataFrame, normal_dev_res: pd.DataFrame, normal_eval_res: pd.DataFrame, fault_eval_res: pd.DataFrame, centers: dict[str, float]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scaler = step20.fit_frozen_scaler(normal_dev_res, centers, definition)
    normal_dev_input = step20.construct_scalar_input(normal_dev_res, scaler)
    normal_eval_input = step20.construct_scalar_input(normal_eval_res, scaler)
    fault_eval_input = step20.construct_scalar_input(fault_eval_res, scaler)

    validation_stat = step15.compute_cusum(normal_dev_input, CUSUM_K)
    limit, validation_run_max = step11.determine_limit(validation_stat)
    validation_metrics, _ = step11.evaluate_normal(validation_stat, limit)

    normal_stat = step15.compute_cusum(normal_eval_input, CUSUM_K)
    normal_metrics, _ = step11.evaluate_normal(normal_stat, limit)

    fault_stat = step15.compute_cusum(fault_eval_input, CUSUM_K)
    fault_metrics, fault_runs = step18.evaluate_generic_fault(step11, fault_stat, float(limit), FAULT_START_SAMPLE)
    metrics = add_early_metrics(fault_metrics, fault_runs)
    result = {
        "repeat_index": repeat_index,
        "branch_key": str(branch_row["branch_key"]),
        "branch_label": str(branch_row["branch_label"]),
        "fault_id": fault_id,
        "branch_status": str(branch_row["branch_status"]),
        "family_key": str(branch_row["family_key"]),
        "family_label": str(branch_row["family_label"]),
        "selected_variables": str(branch_row["selected_variables"]),
        "orientation_signs": str(branch_row["orientation_signs"]),
        "selected_count": int(branch_row["selected_count"]),
        "cusum_k": CUSUM_K,
        "control_limit": float(limit),
        "normal_development_run_level_far": float(validation_metrics["normal_run_level_far"]),
        "normal_run_level_far": float(normal_metrics["normal_run_level_far"]),
    }
    result.update(metrics)
    fault_runs = fault_runs.copy()
    fault_runs.insert(0, "repeat_index", repeat_index)
    fault_runs.insert(1, "branch_key", str(branch_row["branch_key"]))
    fault_runs.insert(2, "branch_label", str(branch_row["branch_label"]))
    fault_runs.insert(3, "fault_id", fault_id)
    fault_runs["allrun_penalized_delay_min"] = np.where(
        fault_runs["detected"].astype(bool),
        pd.to_numeric(fault_runs["alarm_delay_min"], errors="coerce").fillna(MISS_PENALTY_MIN),
        MISS_PENALTY_MIN,
    )
    validation_run_max = validation_run_max.copy()
    validation_run_max.insert(0, "repeat_index", repeat_index)
    validation_run_max.insert(1, "branch_key", str(branch_row["branch_key"]))
    validation_run_max.insert(2, "fault_id", fault_id)
    validation_run_max.insert(3, "control_limit", float(limit))
    return result, fault_runs, validation_run_max


def stability_metrics(current_row: pd.Series, current_definition: pd.DataFrame, original_protocol: pd.DataFrame, original_selected: pd.DataFrame) -> dict[str, Any]:
    branch_key = str(current_row["branch_key"])
    fault_id = int(current_row["fault_id"])
    original_row = original_protocol[
        (original_protocol["branch_key"].astype(str) == branch_key)
        & (original_protocol["fault_id"].astype(int) == fault_id)
    ]
    if len(original_row) != 1:
        raise RuntimeError(f"U2_ORIGINAL_PROTOCOL_LOOKUP_FAILED:{branch_key}:{fault_id}")
    original_row = original_row.iloc[0]
    current_eligible = str(current_row["branch_status"]) == "eligible_repeat_development_only"
    original_eligible = str(original_row["branch_status"]) == "eligible_frozen_before_process_data"
    current_vars = [value for value in str(current_row["selected_variables"]).split(";") if value]
    original_vars = [value for value in str(original_row["selected_variables"]).split(";") if value]
    intersection = set(current_vars) & set(original_vars)
    union = set(current_vars) | set(original_vars)
    jaccard = float(len(intersection) / len(union)) if union else math.nan
    current_signs = [int(value) for value in str(current_row["orientation_signs"]).split(";") if value]
    original_signs = [int(value) for value in str(original_row["orientation_signs"]).split(";") if value]
    current_sign_map = dict(zip(current_vars, current_signs))
    original_sign_map = dict(zip(original_vars, original_signs))
    sign_agreement = float(np.mean([current_sign_map[v] == original_sign_map[v] for v in sorted(intersection)])) if intersection else math.nan
    ordered_exact = current_vars == original_vars and len(current_vars) == TOP_K
    set_exact = set(current_vars) == set(original_vars) and len(current_vars) == TOP_K
    return {
        "original_branch_status": str(original_row["branch_status"]),
        "original_family_key": str(original_row["family_key"]),
        "original_selected_variables": str(original_row["selected_variables"]),
        "original_orientation_signs": str(original_row["orientation_signs"]),
        "eligibility_recovered": bool(current_eligible == original_eligible),
        "family_recovered": bool(current_eligible and original_eligible and str(current_row["family_key"]) == str(original_row["family_key"])),
        "top5_exact_set_recovered": bool(set_exact),
        "top5_exact_order_recovered": bool(ordered_exact),
        "top5_jaccard": jaccard,
        "orientation_agreement_on_overlap": sign_agreement,
        "overlap_count": int(len(intersection)),
    }


def build_repeat_macro(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, N_REPEATS + 1):
        for branch in BRANCHES:
            current = results[
                (results["repeat_index"].astype(int) == repeat_index)
                & (results["branch_key"].astype(str) == branch["branch_key"])
            ].copy()
            if len(current) != len(RESERVED_FAULTS):
                raise RuntimeError(f"U2_REPEAT_RESULT_GEOMETRY_DRIFT:{repeat_index}:{branch['branch_key']}")
            eligible = current[current["branch_status"] == "eligible_repeat_development_only"]
            row: dict[str, Any] = {
                "repeat_index": repeat_index,
                "branch_key": branch["branch_key"],
                "branch_label": branch["branch_label"],
                "eligible_faults": int(len(eligible)),
                "abstained_faults": int(len(current) - len(eligible)),
                "fault_coverage_rate": float(len(eligible) / len(current)),
                "conditional_macro_mean_normal_far": float(eligible["normal_run_level_far"].mean()) if len(eligible) else math.nan,
                "all_panel_run_detection_rate": float(current["fault_run_level_detection_rate"].fillna(0.0).mean()),
                "all_panel_post_fault_threshold_occupancy": float(current["post_fault_threshold_occupancy"].fillna(0.0).mean()),
                "all_panel_penalized_delay_median_min": float(current["allrun_penalized_delay_median_min"].fillna(MISS_PENALTY_MIN).median()),
                "all_panel_penalized_delay_mean_min": float(current["allrun_penalized_delay_mean_min"].fillna(MISS_PENALTY_MIN).mean()),
                "mean_top5_jaccard": float(current["top5_jaccard"].mean()),
                "exact_top5_set_recovery_rate": float(current["top5_exact_set_recovered"].mean()),
                "family_recovery_rate": float(current["family_recovered"].mean()),
            }
            for window in EARLY_WINDOWS_MIN:
                row[f"all_panel_detection_within_{window}min_rate"] = float(current[f"detection_within_{window}min_rate"].fillna(0.0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def build_crossrepeat_summary(repeat_macro: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fault_coverage_rate",
        "conditional_macro_mean_normal_far",
        "all_panel_detection_within_15min_rate",
        "all_panel_detection_within_30min_rate",
        "all_panel_detection_within_60min_rate",
        "all_panel_detection_within_120min_rate",
        "all_panel_run_detection_rate",
        "all_panel_post_fault_threshold_occupancy",
        "all_panel_penalized_delay_median_min",
        "all_panel_penalized_delay_mean_min",
        "mean_top5_jaccard",
        "exact_top5_set_recovery_rate",
        "family_recovery_rate",
    ]
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        current = repeat_macro[repeat_macro["branch_key"] == branch["branch_key"]]
        row: dict[str, Any] = {
            "branch_key": branch["branch_key"],
            "branch_label": branch["branch_label"],
            "n_repeats": len(current),
        }
        for metric in metrics:
            values = pd.to_numeric(current[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_fault_stability(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        for fault_id in RESERVED_FAULTS:
            current = results[
                (results["branch_key"] == branch["branch_key"])
                & (results["fault_id"].astype(int) == fault_id)
            ]
            row: dict[str, Any] = {
                "branch_key": branch["branch_key"],
                "branch_label": branch["branch_label"],
                "fault_id": fault_id,
                "focus_fault": fault_id in FOCUS_FAULTS,
                "n_repeats": len(current),
                "eligibility_rate": float((current["branch_status"] == "eligible_repeat_development_only").mean()),
                "family_recovery_rate": float(current["family_recovered"].mean()),
                "exact_top5_set_recovery_rate": float(current["top5_exact_set_recovered"].mean()),
                "exact_top5_order_recovery_rate": float(current["top5_exact_order_recovered"].mean()),
                "mean_top5_jaccard": float(current["top5_jaccard"].mean()),
                "mean_orientation_agreement": float(current["orientation_agreement_on_overlap"].mean()),
            }
            for metric in [
                "normal_run_level_far",
                "detection_within_15min_rate",
                "detection_within_60min_rate",
                "detection_within_120min_rate",
                "fault_run_level_detection_rate",
                "post_fault_threshold_occupancy",
                "allrun_penalized_delay_median_min",
            ]:
                values = pd.to_numeric(current[metric], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full_like(values, np.nan)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if len(finite_positions) == 0:
        return output
    finite = values[finite_positions]
    order = np.argsort(finite)
    adjusted_sorted = np.maximum.accumulate((len(finite) - np.arange(len(finite))) * finite[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    output[finite_positions] = adjusted_sorted[inverse]
    return output


def safe_wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    result = wilcoxon(values, zero_method="pratt", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def build_repeat_paired_comparisons(repeat_macro: pd.DataFrame) -> pd.DataFrame:
    endpoint_specs = [
        ("conditional_macro_mean_normal_far", "normal FAR", "lower"),
        ("all_panel_detection_within_15min_rate", "15-min detection", "higher"),
        ("all_panel_detection_within_60min_rate", "60-min detection", "higher"),
        ("all_panel_detection_within_120min_rate", "120-min detection", "higher"),
        ("all_panel_post_fault_threshold_occupancy", "post-fault threshold occupancy", "higher"),
        ("all_panel_penalized_delay_mean_min", "all-run penalized mean delay", "lower"),
        ("mean_top5_jaccard", "mean Top-5 Jaccard", "higher"),
        ("exact_top5_set_recovery_rate", "exact Top-5 recovery", "higher"),
    ]
    reference = repeat_macro[repeat_macro["branch_key"] == "adaptive_5family"].set_index("repeat_index")
    rows: list[dict[str, Any]] = []
    for comparator_key in ["fixed_absolute_deviation", "restricted_ad_or_sl"]:
        comparator = repeat_macro[repeat_macro["branch_key"] == comparator_key].set_index("repeat_index")
        common = sorted(set(reference.index) & set(comparator.index))
        for endpoint, label, direction in endpoint_specs:
            a = pd.to_numeric(reference.loc[common, endpoint], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(comparator.loc[common, endpoint], errors="coerce").to_numpy(dtype=float)
            delta = c - a if direction == "lower" else a - c
            statistic, p_value = safe_wilcoxon(delta)
            rows.append(
                {
                    "reference_branch": "adaptive_5family",
                    "comparator_branch": comparator_key,
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "adaptive_better_direction": direction,
                    "n_repeats": len(common),
                    "mean_adaptive_advantage": float(np.nanmean(delta)),
                    "median_adaptive_advantage": float(np.nanmedian(delta)),
                    "adaptive_wins": int(np.sum(delta > TOL)),
                    "ties": int(np.sum(np.abs(delta) <= TOL)),
                    "comparator_wins": int(np.sum(delta < -TOL)),
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_holm_within_endpoint"] = math.nan
    for endpoint, _, _ in endpoint_specs:
        mask = output["endpoint"] == endpoint
        output.loc[mask, "p_value_holm_within_endpoint"] = holm_adjust(pd.to_numeric(output.loc[mask, "p_value_raw"], errors="coerce").to_numpy(dtype=float))
    return output


def plot_family_recovery(fault_stability: pd.DataFrame) -> list[Path]:
    adaptive = fault_stability[fault_stability["branch_key"] == "adaptive_5family"].sort_values("fault_id")
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(adaptive))
    axis.bar(x, 100 * adaptive["family_recovery_rate"].to_numpy(dtype=float))
    axis.set_xticks(x)
    axis.set_xticklabels(adaptive["fault_id"].astype(int).astype(str))
    axis.set_ylim(0, 105)
    axis.set_xlabel("Reserved fault")
    axis.set_ylabel("Adaptive family recovery across repeats (%)")
    axis.set_title("U2 repeated-split recovery of the original adaptive family")
    axis.grid(axis="y", alpha=0.25)
    for index, fault_id in enumerate(adaptive["fault_id"].astype(int)):
        if fault_id in FOCUS_FAULTS:
            axis.text(index, 3, "*", ha="center", va="bottom", fontsize=13)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"figU2a_adaptive_family_recovery.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_jaccard(fault_stability: pd.DataFrame) -> list[Path]:
    figure, axis = plt.subplots(figsize=(12.0, 6.2))
    faults = RESERVED_FAULTS
    width = 0.25
    x = np.arange(len(faults))
    for index, branch in enumerate(BRANCHES):
        current = fault_stability[fault_stability["branch_key"] == branch["branch_key"]].set_index("fault_id").loc[faults]
        axis.bar(x + (index - 1) * width, current["mean_top5_jaccard"].to_numpy(dtype=float), width=width, label=branch["branch_label"])
    axis.set_xticks(x)
    axis.set_xticklabels([str(fault) for fault in faults])
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Reserved fault")
    axis.set_ylabel("Mean Top-5 Jaccard overlap")
    axis.set_title("U2 coordinate-set stability across ten repeated splits")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"figU2b_top5_jaccard_stability.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_repeat_tradeoffs(repeat_macro: pd.DataFrame) -> list[Path]:
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for branch in BRANCHES:
        current = repeat_macro[repeat_macro["branch_key"] == branch["branch_key"]]
        axis.scatter(100 * current["conditional_macro_mean_normal_far"], 100 * current["all_panel_post_fault_threshold_occupancy"], label=branch["branch_label"], alpha=0.8)
    axis.set_xlabel("Repeat-specific macro normal FAR (%)")
    axis.set_ylabel("Repeat-specific macro post-fault occupancy (%)")
    axis.set_title("U2 held-out operating profiles across repeated splits")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"figU2c_repeat_operating_profiles.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths




def write_fault_checkpoint(
    fault_id: int,
    split_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    run_tables: list[pd.DataFrame],
    protocol_tables: list[pd.DataFrame],
    selected_tables: list[pd.DataFrame],
    family_audits: list[pd.DataFrame],
    candidate_metrics_tables: list[pd.DataFrame],
    validation_max_tables: list[pd.DataFrame],
) -> dict[str, Any]:
    root = CHECKPOINT_DIR / f"fault_{fault_id:02d}"
    if root.exists():
        raise RuntimeError(f"U2_CHECKPOINT_ALREADY_EXISTS:{root}")
    root.mkdir(parents=True)
    frames = {
        "split_manifest.csv": pd.DataFrame([row for row in split_rows if int(row.get("fault_id", -1)) == fault_id]),
        "results.csv": pd.DataFrame([row for row in result_rows if int(row.get("fault_id", -1)) == fault_id]),
        "run_results.csv": pd.concat([frame for frame in run_tables if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
        "protocol.csv": pd.concat([frame for frame in protocol_tables if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
        "selected_coordinates.csv": pd.concat([frame for frame in selected_tables if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
        "family_gate_audit.csv": pd.concat([frame for frame in family_audits if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
        "candidate_metrics.csv": pd.concat([frame for frame in candidate_metrics_tables if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
        "validation_run_max.csv": pd.concat([frame for frame in validation_max_tables if int(frame["fault_id"].iloc[0]) == fault_id], ignore_index=True),
    }
    manifest_rows = []
    for name, frame in frames.items():
        path = root / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append({"file": name, "rows": int(len(frame)), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    record = {
        "record_type": "TEP_NON_QUALIFICATION_U2_FAULT_CHECKPOINT",
        "record_version": "1.0",
        "fault_id": fault_id,
        "completed_repeats": N_REPEATS,
        "completed_at_utc": utc_now(),
        "result": "PASS",
        "files": manifest_rows,
        "process_data_modified": False,
        "retuning_performed": False,
    }
    atomic_json(root / "CHECKPOINT_MANIFEST.json", record)
    print(f"U2_FAULT_CHECKPOINT=PASS:fault={fault_id}:repeats={N_REPEATS}")
    return record


def reporting_guardrails() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("U1 disclosure", "U1 is directly responsive to the central family-adaptation claim and must not be omitted; main-text reporting may be concise, with full tables in the supplement.", True),
            ("analysis status", "U2 is a post-U1 internal repeated-split stability audit and does not replace Step-22-R primary results.", True),
            ("resampling scope", "The same benchmark runs are repartitioned after primary freezing; original final/evaluation runs may enter development roles in secondary repeats, so repeats are not new holdouts, external validations, or independent process replications.", True),
            ("normal split", "One repeat-specific normal development/evaluation split is shared across all faults within that repeat.", True),
            ("fault split", "Fault development/evaluation splits are independent by fault and repeat.", True),
            ("multiplicity", "Every repeat uses the unchanged within-fault BH plus permutation max-|t| dual gate across 260 candidates.", True),
            ("no reverse tuning", "No U1 or U2 held-out outcome may promote Fixed-AD, AD-or-SL, or any repeat-specific state to a new primary analysis.", True),
            ("point metric", "Use post-fault threshold occupancy, not independent pointwise detection probability.", True),
            ("fixed SL", "Fixed-SL remains reported from U1 as an applicability control; its omission from repeated U2 computation is not omission from the manuscript.", True),
            ("claim", "If simpler branches are at least as stable and performant, the manuscript must explicitly prefer simplification for future prospective designs.", True),
        ],
        columns=["topic", "statement", "required"],
    )


def run_analysis() -> None:
    registration = verify_specification()
    exact_inputs = verify_exact_inputs()
    step11 = load_module("scripts/11_unpaired_runlevel_screening.py", "u2_step11")
    step15 = load_module("scripts/15_fair_sequential_baselines.py", "u2_step15")
    step18 = load_module("scripts/18_faults9_15_prespecified_generalization.py", "u2_step18")
    step19 = load_module("scripts/19_multievidence_development_audit.py", "u2_step19")
    step20 = load_module("scripts/20_frozen_multievidence_validation.py", "u2_step20")

    variable_cols = step11.load_variable_columns()
    expected_variables = [*[f"xmeas_{i}" for i in range(1, 42)], *[f"xmv_{i}" for i in range(1, 12)]]
    if variable_cols != expected_variables:
        raise RuntimeError("U2_VARIABLE_ORDER_DRIFT")
    if int(step19.BOOTSTRAP_REPLICATES) != 2000 or int(step19.PERMUTATION_REPLICATES) != 2000:
        raise RuntimeError("U2_STEP19_REPLICATE_COUNT_DRIFT")

    original_protocol = pd.read_csv(PROJECT_ROOT / "results/tables/U1_family_adaptation_frozen_protocol.csv")
    original_selected = pd.read_csv(PROJECT_ROOT / "results/tables/U1_family_adaptation_selected_coordinates.csv")
    original_protocol = original_protocol[original_protocol["branch_key"].isin(BRANCH_BY_KEY)].copy()
    original_selected = original_selected[original_selected["branch_key"].isin(BRANCH_BY_KEY)].copy()
    if len(original_protocol) != len(RESERVED_FAULTS) * len(BRANCHES):
        raise RuntimeError("U2_ORIGINAL_PROTOCOL_GEOMETRY_DRIFT")

    normal_train = pd.read_pickle(require_regular(NORMAL_TRAIN_PATH, "U2_NORMAL_TRAIN"))
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_validation = pd.read_pickle(require_regular(NORMAL_VALIDATION_PATH, "U2_NORMAL_VALIDATION"))
    normal_evaluation = pd.read_pickle(require_regular(NORMAL_EVALUATION_PATH, "U2_NORMAL_EVALUATION"))
    normal_pool = pd.concat([normal_validation, normal_evaluation], ignore_index=True)
    del normal_validation, normal_evaluation
    gc.collect()
    normal_pool = validate_500_run_pool(normal_pool, variable_cols, "U2_NORMAL_POOL", expected_fault=None)
    normal_residuals, residual_cols = step11.compute_residual_frame(normal_pool, variable_cols, model)
    del normal_pool
    gc.collect()

    testing, testing_object, testing_names = step18.read_rdata_frame(require_regular(RAW_TESTING_PATH, "U2_RAW_TESTING"), "faulty_testing")
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in testing.columns]
    if missing:
        raise RuntimeError(f"U2_TESTING_COLUMNS_MISSING:{missing}")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise").astype(np.int64)

    split_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    run_tables: list[pd.DataFrame] = []
    selected_tables: list[pd.DataFrame] = []
    protocol_tables: list[pd.DataFrame] = []
    family_audits: list[pd.DataFrame] = []
    candidate_metrics_tables: list[pd.DataFrame] = []
    validation_max_tables: list[pd.DataFrame] = []

    normal_splits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for repeat_index in range(1, N_REPEATS + 1):
        dev_ids, eval_ids = deterministic_split(np.arange(1, 501, dtype=np.int64), normal_split_seed(repeat_index))
        normal_splits[repeat_index] = (dev_ids, eval_ids)
        split_rows.append(
            {
                "repeat_index": repeat_index,
                "role": "normal",
                "fault_id": 0,
                "split_seed": normal_split_seed(repeat_index),
                "development_run_ids": ";".join(map(str, dev_ids)),
                "evaluation_run_ids": ";".join(map(str, eval_ids)),
            }
        )

    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 100)
        print(f"U2 preparing all 500 runs for reserved Fault {fault_id}")
        print("=" * 100)
        fault_full = testing.loc[fault_numbers == fault_id, required].copy()
        fault_full = validate_500_run_pool(fault_full, variable_cols, f"U2_FAULT_{fault_id}_POOL", expected_fault=fault_id)
        fault_residuals, current_cols = step11.compute_residual_frame(fault_full, variable_cols, model)
        if current_cols != residual_cols:
            raise RuntimeError(f"U2_FAULT_RESIDUAL_ORDER_DRIFT:{fault_id}")
        del fault_full
        gc.collect()

        for repeat_index in range(1, N_REPEATS + 1):
            print("\n" + "-" * 100)
            print(f"U2 Fault {fault_id}, repeat {repeat_index}/{N_REPEATS}")
            print("-" * 100)
            normal_dev_ids, normal_eval_ids = normal_splits[repeat_index]
            fault_dev_ids, fault_eval_ids = deterministic_split(np.arange(1, 501, dtype=np.int64), fault_split_seed(repeat_index, fault_id))
            split_rows.append(
                {
                    "repeat_index": repeat_index,
                    "role": "fault",
                    "fault_id": fault_id,
                    "split_seed": fault_split_seed(repeat_index, fault_id),
                    "development_run_ids": ";".join(map(str, fault_dev_ids)),
                    "evaluation_run_ids": ";".join(map(str, fault_eval_ids)),
                }
            )

            normal_dev_res = subset_residual_runs(normal_residuals, normal_dev_ids, f"U2_NORMAL_DEV_R{repeat_index}")
            normal_eval_res = subset_residual_runs(normal_residuals, normal_eval_ids, f"U2_NORMAL_EVAL_R{repeat_index}")
            fault_dev_res = subset_residual_runs(fault_residuals, fault_dev_ids, f"U2_FAULT_{fault_id}_DEV_R{repeat_index}")
            fault_eval_res = subset_residual_runs(fault_residuals, fault_eval_ids, f"U2_FAULT_{fault_id}_EVAL_R{repeat_index}")

            centers_array = normal_dev_res[residual_cols].mean(axis=0).to_numpy(dtype=float)
            centers = {column: float(value) for column, value in zip(residual_cols, centers_array)}
            normal_tensor = step19.extract_base_tensor(normal_dev_res, residual_cols, f"U2_normal_dev_repeat_{repeat_index}")
            fault_tensor = step19.extract_base_tensor(fault_dev_res, residual_cols, f"U2_fault_{fault_id}_dev_repeat_{repeat_index}")
            normal_features = step19.build_feature_run_means(normal_tensor, centers_array)
            fault_features = step19.build_feature_run_means(fault_tensor, centers_array)
            del normal_tensor, fault_tensor
            gc.collect()

            original_seed = int(step19.RANDOM_SEED)
            step19.RANDOM_SEED = selection_seed(repeat_index)
            try:
                metrics, _, _, _ = step19.analyze_fault(fault_id, normal_features, fault_features, variable_cols)
            finally:
                step19.RANDOM_SEED = original_seed
            metrics.insert(0, "repeat_index", repeat_index)
            candidate_metrics_tables.append(metrics)

            repeat_protocol, repeat_selected, repeat_family_audit = choose_repeat_branches(metrics, fault_id)
            repeat_protocol.insert(0, "repeat_index", repeat_index)
            protocol_tables.append(repeat_protocol.copy())
            if not repeat_selected.empty:
                if repeat_selected.columns.duplicated().any():
                    raise RuntimeError(
                        f"U2_REPEAT_SELECTED_DUPLICATE_COLUMNS:{repeat_index}:{fault_id}"
                    )
                if "repeat_index" in repeat_selected.columns:
                    observed_repeat = pd.to_numeric(
                        repeat_selected["repeat_index"], errors="raise"
                    ).to_numpy(dtype=np.int64)
                    if not np.all(observed_repeat == int(repeat_index)):
                        raise RuntimeError(
                            f"U2_REPEAT_SELECTED_REPEAT_INDEX_DRIFT:"
                            f"{repeat_index}:{fault_id}:{sorted(set(observed_repeat.tolist()))}"
                        )
                    repeat_selected = repeat_selected.loc[
                        :, ["repeat_index", *[
                            column for column in repeat_selected.columns
                            if column != "repeat_index"
                        ]]
                    ].copy()
                else:
                    repeat_selected.insert(0, "repeat_index", repeat_index)
                selected_tables.append(repeat_selected)
            repeat_family_audit.insert(0, "repeat_index", repeat_index)
            family_audits.append(repeat_family_audit)

            for _, branch_row in repeat_protocol.iterrows():
                branch_key = str(branch_row["branch_key"])
                if branch_row["branch_status"] != "eligible_repeat_development_only":
                    result, runs = empty_abstention_result(branch_row, repeat_index, fault_id, fault_eval_ids)
                    stability = stability_metrics(branch_row, pd.DataFrame(), original_protocol, original_selected)
                    result.update(stability)
                    result.update({"normal_split_seed": normal_split_seed(repeat_index), "fault_split_seed": fault_split_seed(repeat_index, fault_id), "selection_seed": selection_seed(repeat_index)})
                    runs["normal_split_seed"] = normal_split_seed(repeat_index)
                    runs["fault_split_seed"] = fault_split_seed(repeat_index, fault_id)
                    runs["selection_seed"] = selection_seed(repeat_index)
                    result_rows.append(result)
                    run_tables.append(runs)
                    print(f"  {branch_row['branch_label']}: ABSTAIN")
                    continue
                definition = repeat_selected[repeat_selected["branch_key"] == branch_key].copy()
                if len(definition) != TOP_K:
                    raise RuntimeError(f"U2_REPEAT_DEFINITION_COUNT_DRIFT:{repeat_index}:{fault_id}:{branch_key}")
                result, runs, validation_max = evaluate_branch(
                    step11,
                    step15,
                    step18,
                    step20,
                    repeat_index,
                    fault_id,
                    branch_row,
                    definition,
                    normal_dev_res,
                    normal_eval_res,
                    fault_eval_res,
                    centers,
                )
                stability = stability_metrics(branch_row, definition, original_protocol, original_selected)
                result.update(stability)
                result.update({"normal_split_seed": normal_split_seed(repeat_index), "fault_split_seed": fault_split_seed(repeat_index, fault_id), "selection_seed": selection_seed(repeat_index)})
                runs["normal_split_seed"] = normal_split_seed(repeat_index)
                runs["fault_split_seed"] = fault_split_seed(repeat_index, fault_id)
                runs["selection_seed"] = selection_seed(repeat_index)
                result_rows.append(result)
                run_tables.append(runs)
                validation_max_tables.append(validation_max)
                print(
                    f"  {branch_row['branch_label']}: family={branch_row['family_key']}, "
                    f"Jaccard={stability['top5_jaccard']:.3f}, FAR={100*result['normal_run_level_far']:.1f}%, "
                    f"15/60/120={100*result['detection_within_15min_rate']:.1f}/"
                    f"{100*result['detection_within_60min_rate']:.1f}/"
                    f"{100*result['detection_within_120min_rate']:.1f}%, "
                    f"occupancy={100*result['post_fault_threshold_occupancy']:.1f}%, "
                    f"delay={result['allrun_penalized_delay_median_min']:.1f} min"
                )

            del normal_dev_res, normal_eval_res, fault_dev_res, fault_eval_res, normal_features, fault_features, metrics
            gc.collect()

        write_fault_checkpoint(
            fault_id,
            split_rows,
            result_rows,
            run_tables,
            protocol_tables,
            selected_tables,
            family_audits,
            candidate_metrics_tables,
            validation_max_tables,
        )
        del fault_residuals
        gc.collect()

    del testing, fault_numbers, normal_residuals
    gc.collect()

    results = pd.DataFrame(result_rows).sort_values(["repeat_index", "fault_id", "branch_key"]).reset_index(drop=True)
    runs = pd.concat(run_tables, ignore_index=True)
    protocol = pd.concat(protocol_tables, ignore_index=True) if protocol_tables else pd.DataFrame()
    selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    family_audit = pd.concat(family_audits, ignore_index=True)
    candidates = pd.concat(candidate_metrics_tables, ignore_index=True)
    validation_max = pd.concat(validation_max_tables, ignore_index=True) if validation_max_tables else pd.DataFrame()
    split_manifest = pd.DataFrame(split_rows).sort_values(["repeat_index", "role", "fault_id"]).reset_index(drop=True)

    expected_results = N_REPEATS * len(RESERVED_FAULTS) * len(BRANCHES)
    expected_runs = expected_results * 250
    if len(results) != expected_results:
        raise RuntimeError(f"U2_RESULT_GEOMETRY_DRIFT:{len(results)}:{expected_results}")
    if len(runs) != expected_runs:
        raise RuntimeError(f"U2_RUN_GEOMETRY_DRIFT:{len(runs)}:{expected_runs}")
    if runs.duplicated(["repeat_index", "branch_key", "fault_id", "simulationRun"]).any():
        raise RuntimeError("U2_DUPLICATE_RUN_RESULTS")

    repeat_macro = build_repeat_macro(results)
    crossrepeat = build_crossrepeat_summary(repeat_macro)
    fault_stability = build_fault_stability(results)
    paired = build_repeat_paired_comparisons(repeat_macro)
    guardrails = reporting_guardrails()

    outputs = {
        "split_manifest": TABLE_DIR / "U2_repeat_split_manifest.csv",
        "candidate_metrics": TABLE_DIR / "U2_repeat_candidate_metrics.csv",
        "repeat_protocol": TABLE_DIR / "U2_repeat_frozen_protocol.csv",
        "family_gate_audit": TABLE_DIR / "U2_repeat_family_gate_audit.csv",
        "selected_coordinates": TABLE_DIR / "U2_repeat_selected_coordinates.csv",
        "results_by_fault_repeat": TABLE_DIR / "U2_results_by_fault_repeat.csv",
        "run_results": TABLE_DIR / "U2_run_results.csv",
        "validation_run_max": TABLE_DIR / "U2_validation_run_max.csv",
        "repeat_macro": TABLE_DIR / "U2_repeat_macro_summary.csv",
        "crossrepeat_summary": TABLE_DIR / "U2_crossrepeat_summary.csv",
        "fault_stability": TABLE_DIR / "U2_fault_stability_summary.csv",
        "paired_comparisons": TABLE_DIR / "U2_repeat_level_paired_comparisons.csv",
        "guardrails": TABLE_DIR / "U2_reporting_guardrails.csv",
    }
    frames = {
        "split_manifest": split_manifest,
        "candidate_metrics": candidates,
        "repeat_protocol": protocol,
        "family_gate_audit": family_audit,
        "selected_coordinates": selected,
        "results_by_fault_repeat": results,
        "run_results": runs,
        "validation_run_max": validation_max,
        "repeat_macro": repeat_macro,
        "crossrepeat_summary": crossrepeat,
        "fault_stability": fault_stability,
        "paired_comparisons": paired,
        "guardrails": guardrails,
    }
    for key, path in outputs.items():
        frames[key].to_csv(path, index=False, encoding="utf-8-sig")

    figure_paths = [*plot_family_recovery(fault_stability), *plot_jaccard(fault_stability), *plot_repeat_tradeoffs(repeat_macro)]
    config = {
        "script": "scripts/U2_selection_stability_simplification.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "specification": str(SPEC_PATH.relative_to(PROJECT_ROOT)),
        "specification_registered_at_utc": registration["registered_at_utc"],
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "result": "PASS",
        "u1_result_disclosed": True,
        "primary_step22r_replaced": False,
        "n_repeats": N_REPEATS,
        "reserved_faults": RESERVED_FAULTS,
        "focus_faults": FOCUS_FAULTS,
        "branches": BRANCHES,
        "fixed_signed_level_repeated": False,
        "master_split_seed": MASTER_SPLIT_SEED,
        "selection_seed_base": SELECTION_SEED_BASE,
        "bootstrap_replicates_per_fault_repeat": int(step19.BOOTSTRAP_REPLICATES),
        "permutation_replicates_per_fault_repeat": int(step19.PERMUTATION_REPLICATES),
        "top_k": TOP_K,
        "cusum_k": CUSUM_K,
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "raw_testing_object": testing_object,
        "raw_testing_object_names": testing_names,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "exact_input_sha256": exact_inputs,
        "output_tables": [str(path.relative_to(PROJECT_ROOT)) for path in outputs.values()],
        "output_figures": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
        "process_data_modified": False,
        "detector_retuning": False,
        "network_access": False,
        "formal_p71_p72_code_invoked": False,
        "interpretation": registration["interpretation_rule"],
    }
    atomic_json(FINAL_CONFIG_PATH, config)

    print("\n" + "=" * 100)
    print("U2 repeated-split stability and simplification summary")
    print("=" * 100)
    print(crossrepeat.to_string(index=False))
    print("\nFocus-fault stability:")
    print(fault_stability[fault_stability["fault_id"].isin(FOCUS_FAULTS)].to_string(index=False))
    print("U2_SELECTION_STABILITY_SIMPLIFICATION=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
