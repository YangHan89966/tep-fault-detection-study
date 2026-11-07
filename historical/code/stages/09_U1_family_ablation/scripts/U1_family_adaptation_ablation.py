#!/usr/bin/env python3
"""U1: Post-freeze family-adaptation ablation on the 17 reserved TEP faults.

This secondary analysis tests whether the five-family fault-adaptive protocol adds
value beyond simpler family rules.  It does not replace the frozen Step-22-R
primary results.  Branch definitions are reconstructed only from exact Step-21
development-only tables and are frozen before any normal-evaluation or final-fault
process data are loaded.

Branches
--------
1. adaptive_5family: choose among all five prespecified evidence families.
2. fixed_absolute_deviation: always use absolute deviation, with honest abstention
   if fewer than five coordinates pass the unchanged gate.
3. fixed_signed_level: always use signed level, with honest abstention if fewer
   than five coordinates pass the unchanged gate.
4. restricted_ad_or_sl: choose between absolute deviation and signed level only.

For every eligible branch, Top-K=5, CUSUM k=0.05, and the 95th-percentile normal-
validation run-maximum control-limit rule are held fixed.  Candidate coordinates
must pass both the original bootstrap-CI+BH gate and the already-computed
permutation max-|t| FWER gate.  The adaptive branch must exactly reconstruct the
Step-21 family, coordinates, and orientations and reproduce Step-22-R final
results before any new ablation result is accepted.  The script is executed in
three stages: ``--register`` records the immutable analysis specification,
``--freeze`` reconstructs and freezes all branch definitions while the project
contains no process-data directory, and ``--run`` evaluates only those frozen
branches after the outer runner links the locked data inputs.
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
for directory in (TABLE_DIR, FIGURE_DIR, CONFIG_DIR, RECEIPT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SPEC_PATH = CONFIG_DIR / "U1_family_adaptation_ablation_specification.json"
FREEZE_RECORD_PATH = CONFIG_DIR / "U1_family_adaptation_branch_freeze.json"
FINAL_CONFIG_PATH = CONFIG_DIR / "U1_family_adaptation_ablation.json"

RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
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
        "branch_key": "fixed_signed_level",
        "branch_label": "Fixed signed level",
        "allowed_families": ["signed_level"],
        "rule": "use signed level if at least five coordinates pass the unchanged gate; otherwise abstain",
    },
    {
        "branch_key": "restricted_ad_or_sl",
        "branch_label": "Restricted AD-or-SL",
        "allowed_families": ["signed_level", "absolute_deviation"],
        "rule": "choose between eligible signed level and absolute deviation by the unchanged family score and tie rule",
    },
]
TOP_K = 5
CUSUM_K = 0.05
CONTROL_LIMIT_PERCENTILE = 95
FAULT_START_SAMPLE = 161
FINAL_SAMPLE_MAX = 960
SAMPLING_INTERVAL_MIN = 3
MISS_PENALTY_MIN = (FINAL_SAMPLE_MAX - FAULT_START_SAMPLE + 1) * SAMPLING_INTERVAL_MIN
EARLY_WINDOWS_MIN = [15, 30, 60, 120]
PRIMARY_DATASET = "faulty_testing_final_full_960"
NORMAL_DATASET = "normal_evaluation"
ADAPTIVE_STEP22_KEY = "frozen_multievidence_top5_cusum"
TOL = 1e-10

EXPECTED_INPUT_SHA256 = {
    "scripts/11_unpaired_runlevel_screening.py": "261f64589d6dfc2bb0098e8913358a5b5c9785fea84b67556b972d51cd3b0b86",
    "scripts/15_fair_sequential_baselines.py": "27e255613365cc025d497532aea0ef487831704b7baf6ab27c1d2d18cf642042",
    "scripts/18_faults9_15_prespecified_generalization.py": "d0f7b9b3318c93bfd4f64016d994f2f0aa2806f083df7487af44e62a04e65053",
    "scripts/19_multievidence_development_audit.py": "0a19a1c0210669c201d7cbe42c5824aed2a38cb8a08405c68879cc415a30ee6b",
    "scripts/20_frozen_multievidence_validation.py": "499f1a51d11fbe563b833722da67d002c464fb4acf6e4056ca064b42b9f79825",
    "results/tables/02_variable_list.csv": "7d64ddbab285831773cc6f49b1968e0b2bed3f5a856cea256d5e29ec52b58494",
    "results/tables/21_reserved_multifault_all_features.csv": "d0dbfa68f017c9bbb8029ac75fb61e1d2c2750226c0c5fb0554254ac21921466",
    "results/tables/21_reserved_multifault_family_summary.csv": "260842bf55ec19458354d1f751bd26f848e974b8140b83a7d89fc391920879af",
    "results/tables/21_reserved_multifault_frozen_protocol.csv": "764ab3ed6bd364a92ccfe562746e6bf3025bac965e3572c58ee1c2ac0891acc6",
    "results/tables/22_reserved_multifault_results.csv": "dba76d9fa2d52785f60307ed83ddd0161e92510ce97059c9932a6b5e0817715f",
    "results/tables/22_reserved_multifault_run_results.csv": "cb141426568b07e7ce8008f6595968dd340034e862ea7bc24b9b7ae6215008a9",
    "configs/22_reserved_multifault_frozen_validation.json": "02301f6292c471db81fb38d66b85f0ec54873fc5d05953876bdf815fbd56e914",
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
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


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
    path = require_regular(PROJECT_ROOT / relative, f"U1_MODULE_{module_name}")
    expected = EXPECTED_INPUT_SHA256[relative]
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"U1_MODULE_SHA256_MISMATCH:{relative}:{observed}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"U1_MODULE_SPEC_FAIL:{relative}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def specification_payload(script_hash: str) -> dict[str, Any]:
    return {
        "analysis_name": "U1 family-adaptation ablation on 17 reserved TEP faults",
        "analysis_status": "post-freeze secondary ablation specification; Step-22-R primary results already exist",
        "script": "scripts/U1_family_adaptation_ablation.py",
        "script_sha256": script_hash,
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "primary_results_replaced": False,
        "reserved_faults": RESERVED_FAULTS,
        "branches": BRANCHES,
        "candidate_gate": (
            "bootstrap 95% CI excludes zero AND within-fault BH q<=0.05 across 260 candidates "
            "AND permutation max-|t| FWER-adjusted p<=0.05"
        ),
        "multiplicity_scope": "within each fault across the fixed 5 x 52 candidate universe",
        "family_eligibility": "at least five coordinates pass the unchanged dual gate",
        "within_family_ranking": [
            "descending absolute Hedges g",
            "descending bootstrap direction confidence",
            "descending bootstrap Top-5 selection frequency",
            "ascending variable identifier",
        ],
        "top_k": TOP_K,
        "cusum_k": CUSUM_K,
        "cusum_k_retuning_allowed": False,
        "control_limit_rule": "95th percentile of normal-validation run maxima, recalibrated separately for each frozen branch",
        "normal_evaluation_used_for_selection_or_calibration": False,
        "final_fault_data_used_for_selection_or_calibration": False,
        "early_detection_windows_min": EARLY_WINDOWS_MIN,
        "full_horizon_min": MISS_PENALTY_MIN,
        "point_metric_name": "post-fault threshold occupancy of the accumulated CUSUM statistic",
        "abstention_rule": "an ineligible fixed-family branch is reported as abstention and is not force-fit",
        "all_panel_abstention_accounting": {
            "run_detection": "zero for the abstained fault",
            "early_detection": "zero for the abstained fault",
            "point_occupancy": "zero for the abstained fault",
            "all_run_delay": f"{MISS_PENALTY_MIN} min penalty for every run",
            "normal_far": "reported conditionally on eligible monitors; eligibility is reported separately",
        },
        "paired_inference_unit": "TEP fault",
        "paired_reference": "adaptive_5family",
        "paired_test": "two-sided Wilcoxon signed-rank with Pratt zeros on common eligible faults",
        "multiplicity_for_paired_tests": "Holm adjustment across the three simpler branches separately within each endpoint",
        "interpretation_rule": (
            "if adaptive five-family does not materially improve on restricted AD-or-SL or fixed AD, "
            "the manuscript must reduce the five-family adaptation claim"
        ),
        "process_data_loaded_during_registration": False,
        "detector_fitting_during_registration": False,
        "network_access": False,
        "package_install_or_upgrade": False,
        "formal_p71_p72_code_invoked": False,
        "input_sha256": EXPECTED_INPUT_SHA256,
    }


def register_specification() -> None:
    script_hash = sha256_file(Path(__file__))
    payload = specification_payload(script_hash)
    payload["registered_at_utc"] = utc_now()
    if SPEC_PATH.exists():
        existing = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        existing_comparable = {k: v for k, v in existing.items() if k != "registered_at_utc"}
        new_comparable = {k: v for k, v in payload.items() if k != "registered_at_utc"}
        if existing_comparable != new_comparable:
            raise RuntimeError("U1_DIFFERENT_SPECIFICATION_ALREADY_EXISTS")
        print(f"U1_IDENTICAL_SPECIFICATION_REUSED={SPEC_PATH}")
        return
    atomic_json(SPEC_PATH, payload)
    print("U1_POSTFREEZE_ABLATION_SPECIFICATION=REGISTERED")
    print(f"U1_SPECIFICATION={SPEC_PATH}")


def verify_specification() -> dict[str, Any]:
    require_regular(SPEC_PATH, "U1_SPECIFICATION")
    registration = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    expected = specification_payload(sha256_file(Path(__file__)))
    failures = [key for key, value in expected.items() if registration.get(key) != value]
    if failures:
        raise RuntimeError(f"U1_SPECIFICATION_DRIFT:{failures}")
    return registration


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise RuntimeError(f"U1_BOOLEAN_PARSE_FAIL:{series.name}")
    return normalized.eq("true")


def verify_exact_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = require_regular(PROJECT_ROOT / relative, "U1_EXACT_INPUT")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"U1_INPUT_SHA256_MISMATCH:{relative}:{actual}")
        observed[relative] = actual
    return observed


def top5_for_family(all_features: pd.DataFrame, fault_id: int, family_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = all_features[
        (all_features["fault_id"].astype(int) == int(fault_id))
        & (all_features["family_key"].astype(str) == family_key)
    ].copy()
    if len(current) != 52:
        raise RuntimeError(f"U1_FAMILY_GEOMETRY_DRIFT:{fault_id}:{family_key}:{len(current)}")
    primary = parse_bool(current["primary_stable_global_fdr"])
    strict = parse_bool(current["strict_stable_max_t_fwer"])
    joint = current[primary & strict].copy()
    primary_only = current[primary].copy()
    sorting = [
        "abs_hedges_g",
        "bootstrap_direction_confidence",
        "bootstrap_top5_selection_frequency",
        "variable",
    ]
    ascending = [False, False, False, True]
    selected = joint.sort_values(sorting, ascending=ascending).head(TOP_K).copy()
    selected_primary_only = primary_only.sort_values(sorting, ascending=ascending).head(TOP_K).copy()
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
        "n_joint_dependence_robust_gate": int((primary & strict).sum()),
        "eligible_joint_gate": bool(eligible),
        "primary_only_top5": ";".join(selected_primary_only["variable"].astype(str)) if len(selected_primary_only) == TOP_K else "",
        "joint_gate_top5": ";".join(selected["variable"].astype(str)) if len(selected) == TOP_K else "",
        "top5_invariant_to_max_t_requirement": bool(
            len(selected_primary_only) == TOP_K
            and len(selected) == TOP_K
            and selected_primary_only["variable"].astype(str).tolist() == selected["variable"].astype(str).tolist()
        ),
        "all_selected_primary_gate": bool(len(selected) == TOP_K and parse_bool(selected["primary_stable_global_fdr"]).all()),
        "all_selected_max_t_fwer": bool(len(selected) == TOP_K and parse_bool(selected["strict_stable_max_t_fwer"]).all()),
        "median_selected_abs_hedges_g": float(selected["abs_hedges_g"].median()) if len(selected) == TOP_K else math.nan,
        "family_priority": int(FAMILY_PRIORITY[family_key]),
    }
    return selected, audit


def build_frozen_branches(
    all_features: pd.DataFrame,
    original_protocol: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.DataFrame] = []
    gate_audits: list[dict[str, Any]] = []
    family_cache: dict[tuple[int, str], pd.DataFrame] = {}
    family_audit_cache: dict[tuple[int, str], dict[str, Any]] = {}

    for fault_id in RESERVED_FAULTS:
        for family_key in FAMILY_KEYS:
            selected, audit = top5_for_family(all_features, fault_id, family_key)
            family_cache[(fault_id, family_key)] = selected
            family_audit_cache[(fault_id, family_key)] = audit
            gate_audits.append(audit)

    for branch in BRANCHES:
        branch_key = str(branch["branch_key"])
        for fault_id in RESERVED_FAULTS:
            options: list[tuple[float, int, str]] = []
            for family_key in branch["allowed_families"]:
                audit = family_audit_cache[(fault_id, family_key)]
                if audit["eligible_joint_gate"]:
                    options.append(
                        (
                            float(audit["median_selected_abs_hedges_g"]),
                            -int(audit["family_priority"]),
                            family_key,
                        )
                    )
            chosen_family = max(options)[2] if options else "none"
            if chosen_family == "none":
                protocol_rows.append(
                    {
                        "branch_key": branch_key,
                        "branch_label": branch["branch_label"],
                        "fault_id": fault_id,
                        "branch_status": "abstain_no_eligible_allowed_family",
                        "family_key": "none",
                        "family_label": "None",
                        "selected_count": 0,
                        "selected_variables": "",
                        "orientation_signs": "",
                        "family_score_median_abs_hedges_g": math.nan,
                        "n_joint_gate_stable_in_family": 0,
                        "minimum_selected_bootstrap_frequency": math.nan,
                        "all_selected_primary_gate": False,
                        "all_selected_max_t_fwer": False,
                    }
                )
                continue
            selected = family_cache[(fault_id, chosen_family)].copy()
            if len(selected) != TOP_K:
                raise RuntimeError(f"U1_SELECTED_COUNT_DRIFT:{branch_key}:{fault_id}")
            selected.insert(0, "branch_label", branch["branch_label"])
            selected.insert(0, "branch_key", branch_key)
            selected_rows.append(selected)
            audit = family_audit_cache[(fault_id, chosen_family)]
            protocol_rows.append(
                {
                    "branch_key": branch_key,
                    "branch_label": branch["branch_label"],
                    "fault_id": fault_id,
                    "branch_status": "eligible_frozen_before_process_data",
                    "family_key": chosen_family,
                    "family_label": FAMILY_LABELS[chosen_family],
                    "selected_count": TOP_K,
                    "selected_variables": ";".join(selected["variable"].astype(str)),
                    "orientation_signs": ";".join(selected["orientation_sign"].astype(int).astype(str)),
                    "family_score_median_abs_hedges_g": float(audit["median_selected_abs_hedges_g"]),
                    "n_joint_gate_stable_in_family": int(audit["n_joint_dependence_robust_gate"]),
                    "minimum_selected_bootstrap_frequency": float(selected["bootstrap_top5_selection_frequency"].min()),
                    "all_selected_primary_gate": bool(parse_bool(selected["primary_stable_global_fdr"]).all()),
                    "all_selected_max_t_fwer": bool(parse_bool(selected["strict_stable_max_t_fwer"]).all()),
                }
            )

    protocol = pd.DataFrame(protocol_rows).sort_values(["branch_key", "fault_id"]).reset_index(drop=True)
    selected_long = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    gate_audit = pd.DataFrame(gate_audits).sort_values(["fault_id", "family_priority"]).reset_index(drop=True)

    original = original_protocol.copy()
    original["fault_id"] = original["fault_id"].astype(int)
    parity_rows = []
    adaptive = protocol[protocol["branch_key"] == "adaptive_5family"].set_index("fault_id")
    original = original.set_index("fault_id")
    for fault_id in RESERVED_FAULTS:
        a = adaptive.loc[fault_id]
        o = original.loc[fault_id]
        checks = {
            "branch_status": str(a["branch_status"]) == "eligible_frozen_before_process_data",
            "family": str(a["family_key"]) == str(o["family_key"]),
            "selected_variables": str(a["selected_variables"]) == str(o["selected_variables"]),
            "orientation_signs": str(a["orientation_signs"]) == str(o["orientation_signs"]),
            "selected_count": int(a["selected_count"]) == int(o["selected_count"]) == TOP_K,
            "all_selected_strict_fwer": bool(a["all_selected_max_t_fwer"]) and bool(o["all_selected_strict_fwer"]),
        }
        parity_rows.append(
            {
                "fault_id": fault_id,
                **{f"check_{key}": value for key, value in checks.items()},
                "all_pass": bool(all(checks.values())),
                "u1_family": str(a["family_key"]),
                "step21_family": str(o["family_key"]),
                "u1_selected_variables": str(a["selected_variables"]),
                "step21_selected_variables": str(o["selected_variables"]),
            }
        )
    parity = pd.DataFrame(parity_rows)
    if not parity["all_pass"].all():
        raise RuntimeError("U1_ADAPTIVE_STEP21_PROTOCOL_PARITY_FAILED")
    return protocol, selected_long, gate_audit, parity


def write_branch_freeze(
    registration: dict[str, Any],
    exact_inputs: dict[str, str],
    protocol: pd.DataFrame,
    selected_long: pd.DataFrame,
) -> dict[str, Any]:
    protocol_path = TABLE_DIR / "U1_family_adaptation_frozen_protocol.csv"
    selected_path = TABLE_DIR / "U1_family_adaptation_selected_coordinates.csv"
    protocol.to_csv(protocol_path, index=False, encoding="utf-8-sig")
    selected_long.to_csv(selected_path, index=False, encoding="utf-8-sig")
    if (PROJECT_ROOT / "data").exists():
        raise RuntimeError("U1_PROCESS_DATA_PATH_PRESENT_DURING_BRANCH_FREEZE")
    record = {
        "record_type": "TEP_NON_QUALIFICATION_U1_FAMILY_ADAPTATION_BRANCH_FREEZE",
        "record_version": "1.0",
        "frozen_at_utc": utc_now(),
        "result": "PASS_BRANCHES_FROZEN_BEFORE_PROCESS_DATA",
        "registration_sha256": sha256_file(SPEC_PATH),
        "registration_registered_at_utc": registration["registered_at_utc"],
        "exact_input_sha256": exact_inputs,
        "protocol_csv": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_csv_sha256": sha256_file(protocol_path),
        "selected_coordinates_csv": str(selected_path.relative_to(PROJECT_ROOT)),
        "selected_coordinates_csv_sha256": sha256_file(selected_path),
        "process_data_paths_present": False,
        "process_data_loaded": False,
        "normal_evaluation_loaded": False,
        "final_fault_data_loaded": False,
        "detector_fitting_performed": False,
        "retuning_performed": False,
        "branch_count": len(BRANCHES),
        "fault_count": len(RESERVED_FAULTS),
        "eligible_by_branch": {
            branch["branch_key"]: int(
                (
                    protocol.loc[protocol["branch_key"] == branch["branch_key"], "branch_status"]
                    == "eligible_frozen_before_process_data"
                ).sum()
            )
            for branch in BRANCHES
        },
    }
    atomic_json(FREEZE_RECORD_PATH, record)
    return record


def compute_branch_definitions() -> tuple[dict[str, Any], dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registration = verify_specification()
    exact_inputs = verify_exact_inputs()
    all_features = pd.read_csv(PROJECT_ROOT / "results/tables/21_reserved_multifault_all_features.csv")
    family_summary = pd.read_csv(PROJECT_ROOT / "results/tables/21_reserved_multifault_family_summary.csv")
    original_protocol = pd.read_csv(PROJECT_ROOT / "results/tables/21_reserved_multifault_frozen_protocol.csv")
    if len(all_features) != len(RESERVED_FAULTS) * len(FAMILY_KEYS) * 52:
        raise RuntimeError("U1_ALL_FEATURES_GEOMETRY_DRIFT")
    if len(family_summary) != len(RESERVED_FAULTS) * len(FAMILY_KEYS):
        raise RuntimeError("U1_FAMILY_SUMMARY_GEOMETRY_DRIFT")
    if sorted(original_protocol["fault_id"].astype(int).tolist()) != RESERVED_FAULTS:
        raise RuntimeError("U1_ORIGINAL_PROTOCOL_FAULT_SET_DRIFT")
    protocol, selected_long, gate_audit, protocol_parity = build_frozen_branches(all_features, original_protocol)
    return registration, exact_inputs, all_features, family_summary, protocol, selected_long, gate_audit, protocol_parity


def freeze_branches_only() -> None:
    if (PROJECT_ROOT / "data").exists():
        raise RuntimeError("U1_PROCESS_DATA_PATH_PRESENT_BEFORE_BRANCH_FREEZE")
    registration, exact_inputs, _, _, protocol, selected_long, gate_audit, protocol_parity = compute_branch_definitions()
    freeze_record = write_branch_freeze(registration, exact_inputs, protocol, selected_long)
    gate_audit.to_csv(TABLE_DIR / "U1_multiplicity_gate_invariance.csv", index=False, encoding="utf-8-sig")
    protocol_parity.to_csv(TABLE_DIR / "U1_adaptive_step21_protocol_parity.csv", index=False, encoding="utf-8-sig")
    print("U1_PROCESS_DATA_PHYSICALLY_ABSENT_DURING_BRANCH_FREEZE=PASS")
    print("U1_BRANCHES_FROZEN_BEFORE_PROCESS_DATA=PASS")
    print(f"U1_ELIGIBLE_BY_BRANCH={freeze_record['eligible_by_branch']}")


def verify_existing_branch_freeze(
    registration: dict[str, Any],
    exact_inputs: dict[str, str],
    protocol: pd.DataFrame,
    selected_long: pd.DataFrame,
    gate_audit: pd.DataFrame,
    protocol_parity: pd.DataFrame,
) -> dict[str, Any]:
    require_regular(FREEZE_RECORD_PATH, "U1_BRANCH_FREEZE_RECORD")
    record = json.loads(FREEZE_RECORD_PATH.read_text(encoding="utf-8"))
    checks = {
        "result": record.get("result") == "PASS_BRANCHES_FROZEN_BEFORE_PROCESS_DATA",
        "registration_sha256": record.get("registration_sha256") == sha256_file(SPEC_PATH),
        "registration_time": record.get("registration_registered_at_utc") == registration.get("registered_at_utc"),
        "exact_inputs": record.get("exact_input_sha256") == exact_inputs,
        "process_data_paths_absent": record.get("process_data_paths_present") is False,
        "process_data_not_loaded": record.get("process_data_loaded") is False,
        "normal_evaluation_not_loaded": record.get("normal_evaluation_loaded") is False,
        "final_fault_not_loaded": record.get("final_fault_data_loaded") is False,
        "detector_not_fitted": record.get("detector_fitting_performed") is False,
        "retuning_not_performed": record.get("retuning_performed") is False,
    }
    protocol_path = require_regular(TABLE_DIR / "U1_family_adaptation_frozen_protocol.csv", "U1_FROZEN_PROTOCOL")
    selected_path = require_regular(TABLE_DIR / "U1_family_adaptation_selected_coordinates.csv", "U1_FROZEN_SELECTED")
    gate_path = require_regular(TABLE_DIR / "U1_multiplicity_gate_invariance.csv", "U1_GATE_AUDIT")
    parity_path = require_regular(TABLE_DIR / "U1_adaptive_step21_protocol_parity.csv", "U1_PROTOCOL_PARITY")
    checks["protocol_hash"] = record.get("protocol_csv_sha256") == sha256_file(protocol_path)
    checks["selected_hash"] = record.get("selected_coordinates_csv_sha256") == sha256_file(selected_path)
    checks["protocol_content"] = protocol_path.read_bytes() == dataframe_csv_bytes(protocol)
    checks["selected_content"] = selected_path.read_bytes() == dataframe_csv_bytes(selected_long)
    checks["gate_content"] = gate_path.read_bytes() == dataframe_csv_bytes(gate_audit)
    checks["parity_content"] = parity_path.read_bytes() == dataframe_csv_bytes(protocol_parity)
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"U1_BRANCH_FREEZE_VERIFICATION_FAILED:{failed}")
    return record


def add_early_metrics(metrics: dict[str, Any], runs: pd.DataFrame) -> dict[str, Any]:
    output = dict(metrics)
    detected = runs["detected"].astype(bool)
    delays = pd.to_numeric(runs["alarm_delay_min"], errors="coerce")
    for window in EARLY_WINDOWS_MIN:
        output[f"detection_within_{window}min_rate"] = float((detected & (delays <= window)).mean())
        output[f"detected_within_{window}min_runs"] = int((detected & (delays <= window)).sum())
    penalized = np.where(detected.to_numpy(), delays.fillna(MISS_PENALTY_MIN).to_numpy(dtype=float), MISS_PENALTY_MIN)
    output["allrun_penalized_delay_median_min"] = float(np.median(penalized))
    output["allrun_penalized_delay_mean_min"] = float(np.mean(penalized))
    output["post_fault_threshold_occupancy"] = float(metrics["fault_post_fault_point_detection_rate"])
    return output


def empty_abstention_result(branch: pd.Series, fault_id: int) -> tuple[dict[str, Any], pd.DataFrame]:
    result = {
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
        "normal_validation_run_level_far": math.nan,
        "normal_run_level_far": math.nan,
        "normal_run_level_far_ci_low": math.nan,
        "normal_run_level_far_ci_high": math.nan,
        "fault_n_runs": 250,
        "fault_pre_fault_alarm_runs": 0,
        "fault_pre_fault_run_alarm_rate": 0.0,
        "fault_pre_fault_point_alarm_rate": 0.0,
        "detected_runs": 0,
        "right_censored_runs": 250,
        "fault_run_level_detection_rate": 0.0,
        "fault_run_level_detection_ci_low": 0.0,
        "fault_run_level_detection_ci_high": math.nan,
        "fault_post_fault_point_detection_rate": 0.0,
        "post_fault_threshold_occupancy": 0.0,
        "median_alarm_delay_min": math.nan,
        "mean_alarm_delay_min": math.nan,
        "alarm_delay_q1_min": math.nan,
        "alarm_delay_q3_min": math.nan,
        "allrun_penalized_delay_median_min": float(MISS_PENALTY_MIN),
        "allrun_penalized_delay_mean_min": float(MISS_PENALTY_MIN),
    }
    for window in EARLY_WINDOWS_MIN:
        result[f"detection_within_{window}min_rate"] = 0.0
        result[f"detected_within_{window}min_runs"] = 0
    runs = pd.DataFrame(
        {
            "branch_key": str(branch["branch_key"]),
            "branch_label": str(branch["branch_label"]),
            "fault_id": fault_id,
            "family_key": "none",
            "simulationRun": np.arange(251, 501, dtype=int),
            "pre_fault_run_alarm": False,
            "post_fault_run_alarm": False,
            "first_alarm_sample": np.nan,
            "detected": False,
            "alarm_delay_min": np.nan,
            "right_censored": True,
            "allrun_penalized_delay_min": float(MISS_PENALTY_MIN),
            "branch_status": str(branch["branch_status"]),
        }
    )
    return result, runs


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


def build_summary(results: pd.DataFrame, run_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        key = branch["branch_key"]
        current = results[results["branch_key"] == key].copy()
        eligible = current[current["branch_status"] == "eligible_frozen_before_process_data"]
        branch_runs = run_results[run_results["branch_key"] == key]
        if len(branch_runs) != len(RESERVED_FAULTS) * 250:
            raise RuntimeError(f"U1_RUN_TABLE_GEOMETRY_DRIFT:{key}:{len(branch_runs)}")
        detected = branch_runs["detected"].astype(bool)
        delays = pd.to_numeric(branch_runs["allrun_penalized_delay_min"], errors="raise")
        row = {
            "branch_key": key,
            "branch_label": branch["branch_label"],
            "n_faults": len(RESERVED_FAULTS),
            "eligible_faults": int(len(eligible)),
            "abstained_faults": int(len(RESERVED_FAULTS) - len(eligible)),
            "fault_coverage_rate": float(len(eligible) / len(RESERVED_FAULTS)),
            "conditional_macro_mean_normal_far": float(eligible["normal_run_level_far"].mean()) if len(eligible) else math.nan,
            "all_panel_run_detection_rate": float(detected.mean()),
            "all_panel_post_fault_threshold_occupancy": float(current["post_fault_threshold_occupancy"].fillna(0.0).mean()),
            "all_panel_penalized_delay_median_min": float(np.median(delays)),
            "all_panel_penalized_delay_mean_min": float(np.mean(delays)),
        }
        for window in EARLY_WINDOWS_MIN:
            row[f"all_panel_detection_within_{window}min_rate"] = float(
                current[f"detection_within_{window}min_rate"].fillna(0.0).mean()
            )
        family_counts = current.loc[current["branch_status"] == "eligible_frozen_before_process_data", "family_key"].value_counts()
        for family_key in FAMILY_KEYS:
            row[f"faults_using_{family_key}"] = int(family_counts.get(family_key, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    endpoint_specs = [
        ("normal_run_level_far", "normal FAR", "lower"),
        ("detection_within_15min_rate", "detection within 15 min", "higher"),
        ("detection_within_30min_rate", "detection within 30 min", "higher"),
        ("detection_within_60min_rate", "detection within 60 min", "higher"),
        ("detection_within_120min_rate", "detection within 120 min", "higher"),
        ("fault_run_level_detection_rate", "full-horizon run detection", "higher"),
        ("post_fault_threshold_occupancy", "post-fault threshold occupancy", "higher"),
        ("allrun_penalized_delay_median_min", "all-run penalized median delay", "lower"),
    ]
    adaptive = results[
        (results["branch_key"] == "adaptive_5family")
        & (results["branch_status"] == "eligible_frozen_before_process_data")
    ].set_index("fault_id")
    rows = []
    for comparator in [b for b in BRANCHES if b["branch_key"] != "adaptive_5family"]:
        comp = results[
            (results["branch_key"] == comparator["branch_key"])
            & (results["branch_status"] == "eligible_frozen_before_process_data")
        ].set_index("fault_id")
        common = sorted(set(adaptive.index.astype(int)) & set(comp.index.astype(int)))
        for endpoint, label, direction in endpoint_specs:
            a = pd.to_numeric(adaptive.loc[common, endpoint], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(comp.loc[common, endpoint], errors="coerce").to_numpy(dtype=float)
            delta = c - a if direction == "lower" else a - c
            statistic, p_value = safe_wilcoxon(delta)
            rows.append(
                {
                    "reference_branch": "adaptive_5family",
                    "comparator_branch": comparator["branch_key"],
                    "comparator_label": comparator["branch_label"],
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "adaptive_better_direction": direction,
                    "common_eligible_faults": len(common),
                    "common_fault_ids": ";".join(map(str, common)),
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
        output.loc[mask, "p_value_holm_within_endpoint"] = holm_adjust(
            pd.to_numeric(output.loc[mask, "p_value_raw"], errors="coerce").to_numpy(dtype=float)
        )
    return output.sort_values(["endpoint", "comparator_branch"]).reset_index(drop=True)


def verify_adaptive_performance_parity(
    results: pd.DataFrame,
    run_results: pd.DataFrame,
    step22_results: pd.DataFrame,
    step22_runs: pd.DataFrame,
) -> pd.DataFrame:
    reference = step22_results[
        (step22_results["method_key"].astype(str) == ADAPTIVE_STEP22_KEY)
        & (step22_results["dataset"].astype(str) == PRIMARY_DATASET)
    ].copy()
    reference["fault_id"] = reference["fault_id"].astype(int)
    current = results[results["branch_key"] == "adaptive_5family"].copy()
    current["fault_id"] = current["fault_id"].astype(int)
    fields = [
        "control_limit",
        "normal_run_level_far",
        "fault_pre_fault_run_alarm_rate",
        "fault_run_level_detection_rate",
        "fault_post_fault_point_detection_rate",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
        "alarm_delay_q1_min",
        "alarm_delay_q3_min",
    ]
    rows = []
    merged = current.merge(reference[["fault_id", *fields]], on="fault_id", suffixes=("_u1", "_step22"), validate="one_to_one")
    for _, row in merged.iterrows():
        for field in fields:
            left = float(row[f"{field}_u1"])
            right = float(row[f"{field}_step22"])
            difference = left - right
            rows.append(
                {
                    "scope": "fault_summary",
                    "fault_id": int(row["fault_id"]),
                    "field": field,
                    "u1_value": left,
                    "step22_value": right,
                    "difference": difference,
                    "passed": bool(abs(difference) <= TOL),
                }
            )

    ref_runs = step22_runs[
        (step22_runs["method_key"].astype(str) == ADAPTIVE_STEP22_KEY)
        & (step22_runs["dataset"].astype(str) == PRIMARY_DATASET)
    ].copy()
    cur_runs = run_results[run_results["branch_key"] == "adaptive_5family"].copy()
    for frame in (ref_runs, cur_runs):
        frame["fault_id"] = frame["fault_id"].astype(int)
        frame["simulationRun"] = pd.to_numeric(frame["simulationRun"], errors="raise").astype(int)
    run_fields = ["pre_fault_run_alarm", "post_fault_run_alarm", "detected", "first_alarm_sample", "alarm_delay_min"]
    joined = cur_runs.merge(
        ref_runs[["fault_id", "simulationRun", *run_fields]],
        on=["fault_id", "simulationRun"],
        suffixes=("_u1", "_step22"),
        validate="one_to_one",
    )
    if len(joined) != len(RESERVED_FAULTS) * 250:
        raise RuntimeError("U1_ADAPTIVE_RUN_PARITY_GEOMETRY_DRIFT")
    for field in run_fields:
        if field in {"pre_fault_run_alarm", "post_fault_run_alarm", "detected"}:
            left = parse_bool(joined[f"{field}_u1"])
            right = parse_bool(joined[f"{field}_step22"])
            passed = left.eq(right)
            differences = np.where(passed, 0.0, 1.0)
        else:
            left = pd.to_numeric(joined[f"{field}_u1"], errors="coerce")
            right = pd.to_numeric(joined[f"{field}_step22"], errors="coerce")
            passed = (left.isna() & right.isna()) | ((left - right).abs() <= TOL)
            differences = (left - right).to_numpy(dtype=float)
        rows.append(
            {
                "scope": "all_4250_run_rows",
                "fault_id": -1,
                "field": field,
                "u1_value": math.nan,
                "step22_value": math.nan,
                "difference": float(np.nanmax(np.abs(differences))) if len(differences) else math.nan,
                "passed": bool(passed.all()),
            }
        )
    parity = pd.DataFrame(rows)
    if not parity["passed"].all():
        raise RuntimeError("U1_ADAPTIVE_STEP22_PERFORMANCE_PARITY_FAILED")
    return parity


def plot_protocol_composition(protocol: pd.DataFrame) -> list[Path]:
    branch_order = [b["branch_key"] for b in BRANCHES]
    labels = [b["branch_label"] for b in BRANCHES]
    counts = []
    for branch_key in branch_order:
        current = protocol[protocol["branch_key"] == branch_key]
        row = [int((current["family_key"] == family).sum()) for family in FAMILY_KEYS]
        row.append(int((current["branch_status"] != "eligible_frozen_before_process_data").sum()))
        counts.append(row)
    matrix = np.asarray(counts, dtype=float)
    figure, axis = plt.subplots(figsize=(11.5, 6.0))
    bottom = np.zeros(len(branch_order))
    category_labels = [FAMILY_LABELS[key] for key in FAMILY_KEYS] + ["Abstention"]
    for index, category in enumerate(category_labels):
        bars = axis.bar(np.arange(len(branch_order)), matrix[:, index], bottom=bottom, label=category)
        axis.bar_label(bars, labels=[str(int(value)) if value else "" for value in matrix[:, index]], label_type="center", fontsize=8)
        bottom += matrix[:, index]
    axis.set_xticks(np.arange(len(branch_order)))
    axis.set_xticklabels(labels, rotation=12, ha="right")
    axis.set_ylabel("Number of reserved faults")
    axis.set_ylim(0, 18)
    axis.set_title("U1 family-adaptation ablation: frozen family composition and abstention")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"figU1a_family_protocol_composition.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_early_detection(summary: pd.DataFrame) -> list[Path]:
    x = np.asarray([15, 30, 60, 120, MISS_PENALTY_MIN], dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for branch in BRANCHES:
        row = summary[summary["branch_key"] == branch["branch_key"]].iloc[0]
        y = [
            float(row["all_panel_detection_within_15min_rate"]),
            float(row["all_panel_detection_within_30min_rate"]),
            float(row["all_panel_detection_within_60min_rate"]),
            float(row["all_panel_detection_within_120min_rate"]),
            float(row["all_panel_run_detection_rate"]),
        ]
        axis.plot(np.arange(len(x)), 100 * np.asarray(y), marker="o", label=branch["branch_label"])
    axis.set_xticks(np.arange(len(x)))
    axis.set_xticklabels(["15", "30", "60", "120", "full horizon"])
    axis.set_xlabel("Time after fault onset (min)")
    axis.set_ylabel("All-panel run detection (%)")
    axis.set_ylim(-2, 103)
    axis.set_title("U1 early-detection profiles with abstentions counted as missed faults")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = FIGURE_DIR / f"figU1b_family_adaptation_early_detection.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def reporting_guardrails() -> pd.DataFrame:
    rows = [
        ("analysis status", "U1 is a post-freeze secondary ablation and does not replace the Step-22-R primary validation.", True),
        ("family adaptation", "Claim five-family value only to the extent supported against fixed AD, fixed SL, and restricted AD-or-SL.", True),
        ("abstention", "Report ineligible fixed-family faults as abstentions; do not silently force a Top-5 branch.", True),
        ("multiplicity", "The U1 selected coordinates satisfy both the original within-fault BH gate and permutation max-|t| FWER control.", True),
        ("scope", "The candidate multiplicity universe is 260 tests within each fault, not a global guarantee across all faults.", True),
        ("early detection", "Report 15/30/60/120 min detection and full-horizon detection separately.", True),
        ("point metric", "Use post-fault threshold occupancy, not independent pointwise detection probability.", True),
        ("normal FAR", "For branches with abstention, report normal FAR conditional on eligible monitors and report coverage separately.", True),
        ("external validity", "Do not describe U1 as zero-shot, unknown-fault, external-process, or real-plant validation.", True),
        ("retuning", "No final or normal-evaluation outcome may change family, coordinates, orientation, k, or the threshold rule.", True),
    ]
    return pd.DataFrame(rows, columns=["topic", "statement", "required"])


def run_analysis() -> None:
    registration, exact_inputs, _, _, protocol, selected_long, gate_audit, protocol_parity = compute_branch_definitions()
    freeze_record = verify_existing_branch_freeze(
        registration, exact_inputs, protocol, selected_long, gate_audit, protocol_parity
    )
    step22_results = pd.read_csv(PROJECT_ROOT / "results/tables/22_reserved_multifault_results.csv")
    step22_runs = pd.read_csv(PROJECT_ROOT / "results/tables/22_reserved_multifault_run_results.csv")
    print("U1_EXISTING_PRE_DATA_BRANCH_FREEZE_VERIFIED=PASS")
    print(f"U1_ELIGIBLE_BY_BRANCH={freeze_record['eligible_by_branch']}")

    step11 = load_module("scripts/11_unpaired_runlevel_screening.py", "u1_step11")
    step15 = load_module("scripts/15_fair_sequential_baselines.py", "u1_step15")
    step18 = load_module("scripts/18_faults9_15_prespecified_generalization.py", "u1_step18")
    step20 = load_module("scripts/20_frozen_multievidence_validation.py", "u1_step20")

    variable_cols = step11.load_variable_columns()
    expected_variables = [*[f"xmeas_{i}" for i in range(1, 42)], *[f"xmv_{i}" for i in range(1, 12)]]
    if variable_cols != expected_variables:
        raise RuntimeError("U1_VARIABLE_ORDER_DRIFT")

    normal_train = pd.read_pickle(require_regular(NORMAL_TRAIN_PATH, "U1_NORMAL_TRAIN"))
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    step19 = load_module("scripts/19_multievidence_development_audit.py", "u1_step19")
    normal_validation = pd.read_pickle(require_regular(NORMAL_VALIDATION_PATH, "U1_NORMAL_VALIDATION"))
    normal_validation = step19.validate_run_frame(
        normal_validation, variable_cols, "normal_validation_runs_1_250", expected_fault=None
    )
    normal_validation_res, residual_cols = step11.compute_residual_frame(normal_validation, variable_cols, model)
    del normal_validation
    gc.collect()
    centers = step20.residual_centers(normal_validation_res, residual_cols)

    normal_evaluation = pd.read_pickle(require_regular(NORMAL_EVALUATION_PATH, "U1_NORMAL_EVALUATION"))
    normal_evaluation = step20.validate_normal_frame(
        normal_evaluation,
        variable_cols,
        "normal_evaluation_runs_251_500",
        np.arange(251, 501, dtype=np.int64),
    )
    normal_evaluation_res, current_cols = step11.compute_residual_frame(normal_evaluation, variable_cols, model)
    if current_cols != residual_cols:
        raise RuntimeError("U1_NORMAL_EVALUATION_RESIDUAL_ORDER_DRIFT")
    del normal_evaluation
    gc.collect()

    testing, testing_object, testing_names = step18.read_rdata_frame(
        require_regular(RAW_TESTING_PATH, "U1_RAW_TESTING"), "faulty_testing"
    )
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in testing.columns]
    if missing:
        raise RuntimeError(f"U1_TESTING_COLUMNS_MISSING:{missing}")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    result_rows: list[dict[str, Any]] = []
    run_tables: list[pd.DataFrame] = []
    scaler_tables: list[pd.DataFrame] = []
    validation_max_tables: list[pd.DataFrame] = []

    protocol_index = protocol.set_index(["branch_key", "fault_id"])
    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 92)
        print(f"U1 evaluating frozen family branches for Fault {fault_id}")
        print("=" * 92)
        final_mask = (fault_numbers == fault_id) & run_numbers.between(251, 500)
        final = testing.loc[final_mask, required].copy()
        final, _ = step18.validate_fault_frame(
            final,
            variable_cols,
            fault_id,
            np.arange(251, 501, dtype=np.int64),
            1,
            960,
            f"fault{fault_id}_u1_final_runs_251_500",
        )
        final_res, current_cols = step11.compute_residual_frame(final, variable_cols, model)
        if current_cols != residual_cols:
            raise RuntimeError(f"U1_FINAL_RESIDUAL_ORDER_DRIFT:{fault_id}")
        del final
        gc.collect()

        for branch in BRANCHES:
            branch_key = branch["branch_key"]
            branch_row = protocol_index.loc[(branch_key, fault_id)]
            if branch_row["branch_status"] != "eligible_frozen_before_process_data":
                abstention_branch = branch_row.copy()
                abstention_branch["branch_key"] = branch_key
                result, runs = empty_abstention_result(abstention_branch, fault_id)
                result_rows.append(result)
                run_tables.append(runs)
                print(f"  {branch['branch_label']}: ABSTAIN")
                continue

            definition = selected_long[
                (selected_long["branch_key"] == branch_key)
                & (selected_long["fault_id"].astype(int) == fault_id)
            ].copy()
            if len(definition) != TOP_K:
                raise RuntimeError(f"U1_DEFINITION_COUNT_DRIFT:{branch_key}:{fault_id}")
            scaler = step20.fit_frozen_scaler(normal_validation_res, centers, definition)
            scaler.insert(0, "branch_label", branch["branch_label"])
            scaler.insert(0, "branch_key", branch_key)
            scaler_tables.append(scaler)

            validation_input = step20.construct_scalar_input(normal_validation_res, scaler)
            normal_input = step20.construct_scalar_input(normal_evaluation_res, scaler)
            final_input = step20.construct_scalar_input(final_res, scaler)
            validation_stat = step15.compute_cusum(validation_input, CUSUM_K)
            control_limit, validation_max = step11.determine_limit(validation_stat)
            validation_metrics, _ = step11.evaluate_normal(validation_stat, control_limit)
            normal_stat = step15.compute_cusum(normal_input, CUSUM_K)
            normal_metrics, normal_runs = step11.evaluate_normal(normal_stat, control_limit)
            final_stat = step15.compute_cusum(final_input, CUSUM_K)
            fault_metrics, fault_runs = step18.evaluate_generic_fault(
                step11, final_stat, control_limit, FAULT_START_SAMPLE
            )
            fault_metrics = add_early_metrics(fault_metrics, fault_runs)
            result = {
                "branch_key": branch_key,
                "branch_label": branch["branch_label"],
                "fault_id": fault_id,
                "branch_status": "eligible_frozen_before_process_data",
                "family_key": str(branch_row["family_key"]),
                "family_label": str(branch_row["family_label"]),
                "selected_variables": str(branch_row["selected_variables"]),
                "orientation_signs": str(branch_row["orientation_signs"]),
                "selected_count": TOP_K,
                "cusum_k": CUSUM_K,
                "control_limit": float(control_limit),
                "normal_validation_run_level_far": float(validation_metrics["normal_run_level_far"]),
                **normal_metrics,
                **fault_metrics,
            }
            result_rows.append(result)
            annotated = fault_runs.copy()
            annotated.insert(0, "family_key", str(branch_row["family_key"]))
            annotated.insert(0, "fault_id", fault_id)
            annotated.insert(0, "branch_label", branch["branch_label"])
            annotated.insert(0, "branch_key", branch_key)
            annotated["allrun_penalized_delay_min"] = np.where(
                annotated["detected"].astype(bool),
                pd.to_numeric(annotated["alarm_delay_min"], errors="coerce").fillna(MISS_PENALTY_MIN),
                MISS_PENALTY_MIN,
            )
            annotated["branch_status"] = "eligible_frozen_before_process_data"
            run_tables.append(annotated)
            validation_max = validation_max.copy()
            validation_max.insert(0, "control_limit", float(control_limit))
            validation_max.insert(0, "fault_id", fault_id)
            validation_max.insert(0, "branch_key", branch_key)
            validation_max_tables.append(validation_max)
            print(
                f"  {branch['branch_label']}: family={branch_row['family_key']}, "
                f"FAR={100*normal_metrics['normal_run_level_far']:.1f}%, "
                f"15/60/120={100*fault_metrics['detection_within_15min_rate']:.1f}/"
                f"{100*fault_metrics['detection_within_60min_rate']:.1f}/"
                f"{100*fault_metrics['detection_within_120min_rate']:.1f}%, "
                f"full={100*fault_metrics['fault_run_level_detection_rate']:.1f}%, "
                f"occupancy={100*fault_metrics['post_fault_threshold_occupancy']:.1f}%, "
                f"penalized median={fault_metrics['allrun_penalized_delay_median_min']:.1f} min"
            )
            del definition, scaler, validation_input, normal_input, final_input
            del validation_stat, normal_stat, final_stat, fault_runs, normal_runs
            gc.collect()
        del final_res
        gc.collect()

    del testing, fault_numbers, run_numbers, normal_validation_res, normal_evaluation_res
    gc.collect()

    results = pd.DataFrame(result_rows).sort_values(["branch_key", "fault_id"]).reset_index(drop=True)
    run_results = pd.concat(run_tables, ignore_index=True).sort_values(["branch_key", "fault_id", "simulationRun"]).reset_index(drop=True)
    scalers = pd.concat(scaler_tables, ignore_index=True) if scaler_tables else pd.DataFrame()
    validation_max = pd.concat(validation_max_tables, ignore_index=True) if validation_max_tables else pd.DataFrame()
    parity = verify_adaptive_performance_parity(results, run_results, step22_results, step22_runs)
    summary = build_summary(results, run_results)
    paired = build_paired_comparisons(results)
    guardrails = reporting_guardrails()

    output_frames = {
        "U1_family_adaptation_results_by_fault.csv": results,
        "U1_family_adaptation_run_results.csv": run_results,
        "U1_family_adaptation_crossfault_summary.csv": summary,
        "U1_family_adaptation_paired_comparisons.csv": paired,
        "U1_family_adaptation_adaptive_step22_performance_parity.csv": parity,
        "U1_family_adaptation_local_scalers.csv": scalers,
        "U1_family_adaptation_validation_run_max.csv": validation_max,
        "U1_family_adaptation_reporting_guardrails.csv": guardrails,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")

    figure_paths = [*plot_protocol_composition(protocol), *plot_early_detection(summary)]
    config = {
        "record_type": "TEP_NON_QUALIFICATION_U1_FAMILY_ADAPTATION_ABLATION",
        "record_version": "1.0",
        "completed_at_utc": utc_now(),
        "result": "PASS",
        "scope": "NON_QUALIFICATION_EXPLORATORY",
        "postfreeze_secondary_analysis": True,
        "primary_step22_results_replaced": False,
        "specification": str(SPEC_PATH.relative_to(PROJECT_ROOT)),
        "specification_sha256": sha256_file(SPEC_PATH),
        "branch_freeze_record": str(FREEZE_RECORD_PATH.relative_to(PROJECT_ROOT)),
        "branch_freeze_record_sha256": sha256_file(FREEZE_RECORD_PATH),
        "branches": BRANCHES,
        "reserved_faults": RESERVED_FAULTS,
        "top_k": TOP_K,
        "cusum_k": CUSUM_K,
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "early_detection_windows_min": EARLY_WINDOWS_MIN,
        "miss_penalty_min": MISS_PENALTY_MIN,
        "adaptive_step21_protocol_parity": True,
        "adaptive_step22_performance_parity": True,
        "all_selected_coordinates_pass_original_bh_and_max_t_fwer": True,
        "normal_evaluation_used_for_selection_or_calibration": False,
        "final_fault_data_used_for_selection_or_calibration": False,
        "detector_retuning": False,
        "testing_object": testing_object,
        "testing_object_names": testing_names,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "output_tables": [str((TABLE_DIR / filename).relative_to(PROJECT_ROOT)) for filename in output_frames],
        "output_figures": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
        "claim_decision_rule": registration["interpretation_rule"],
        "formal_p71_p72_code_invoked": False,
        "simulator_executed": False,
        "qualification_executed": False,
    }
    atomic_json(FINAL_CONFIG_PATH, config)

    print("\n" + "=" * 96)
    print("U1 family-adaptation ablation summary")
    print("=" * 96)
    print(summary.to_string(index=False))
    print("\nPaired adaptive-versus-simpler comparisons:")
    print(paired.to_string(index=False))
    print("U1_ADAPTIVE_STEP21_PROTOCOL_PARITY=PASS")
    print("U1_ADAPTIVE_STEP22_PERFORMANCE_PARITY=PASS")
    print("U1_MULTIPLICITY_DUAL_GATE_SELECTED_COORDINATES=PASS")
    print("U1_FAMILY_ADAPTATION_ABLATION=PASS")
    print(f"U1_CONFIG={FINAL_CONFIG_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true")
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_specification()
    elif args.freeze:
        freeze_branches_only()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
