"""Step 22: Preregistered frozen validation on all 17 reserved TEP faults.

Scientific role
---------------
Step 21 preregistered and selected one evidence family plus five oriented PCA
residual variables for every TEP fault not used for method development.  This
script verifies and freezes those exact definitions before it subsets any
testing-final run.  It then evaluates runs 251--500 and an independent normal
evaluation set without changing the selected family, variables, orientations,
Top-K, proposed CUSUM k, or control-limit rule.

Fair sequential comparators
---------------------------
All methods use a 95th-percentile normal-validation run-maximum limit.
Comparator CUSUM k values are selected only from testing-development runs
1--250 using the frozen Step-15 grid.  The proposed Top-5 keeps k=0.05.

* PCA Q/SPE-CUSUM
* PCA Hotelling T2-CUSUM
* PCA joint max-score CUSUM
* shrinkage-LDA residual CUSUM
* chosen-family All-52 residual CUSUM
* chosen-family Top-1 residual CUSUM
* proposed chosen-family Top-5 residual CUSUM

Container caveat
----------------
``TEP_Faulty_Testing.RData`` stores development and final runs in one object,
so pyreadr materializes the complete container.  Code execution is staged:
all 17 definitions, scalers, comparator k values, and control limits are frozen
using development and normal-validation data before any final row is subsetted.
This is auditable code separation, not a physically access-controlled holdout.

Two-stage use
-------------
Place in:
    TEP_Fault3_Project/scripts/22_reserved_multifault_frozen_validation.py

Preregister first (reads no process data):
    python scripts/22_reserved_multifault_frozen_validation.py --register

Then run:
    python scripts/22_reserved_multifault_frozen_validation.py --run

The run stops if the script bytes differ from the registered SHA-256 hash.
Only files beginning with ``22_`` and ``fig22`` are written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path

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

for directory in [TABLE_DIR, FIGURE_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
EXPECTED_DEVELOPMENT_RUNS = np.arange(1, 251, dtype=np.int64)
EXPECTED_FINAL_RUNS = np.arange(251, 501, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SAMPLING_INTERVAL_MIN = 3
TOP_K = 5
PROPOSED_CUSUM_K = 0.05
CONTROL_LIMIT_PERCENTILE = 95
PARITY_TOLERANCE = 1e-10

EXPECTED_HASHES = {
    "11_unpaired_runlevel_screening.py": (
        "261f64589d6dfc2bb0098e8913358a5b5c9785fea84b67556b972d51cd3b0b86"
    ),
    "15_fair_sequential_baselines.py": (
        "27e255613365cc025d497532aea0ef487831704b7baf6ab27c1d2d18cf642042"
    ),
    "18_faults9_15_prespecified_generalization.py": (
        "d0f7b9b3318c93bfd4f64016d994f2f0aa2806f083df7487af44e62a04e65053"
    ),
    "19_multievidence_development_audit.py": (
        "0a19a1c0210669c201d7cbe42c5824aed2a38cb8a08405c68879cc415a30ee6b"
    ),
    "20_frozen_multievidence_validation.py": (
        "499f1a51d11fbe563b833722da67d002c464fb4acf6e4056ca064b42b9f79825"
    ),
    "21_reserved_multifault_development_audit.py": (
        "3e4bc3df679fe5e829b831b6108123dab7d5eb77edb1ea82b2824c30fa97428d"
    ),
}

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
NORMAL_EVALUATION_PATH = PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"

STEP21_PREREGISTRATION_PATH = (
    CONFIG_DIR / "21_reserved_multifault_preregistration.json"
)
STEP21_CONFIG_PATH = CONFIG_DIR / "21_reserved_multifault_development_audit.json"
STEP21_PROTOCOL_PATH = TABLE_DIR / "21_reserved_multifault_frozen_protocol.csv"
STEP21_SELECTED_PATH = TABLE_DIR / "21_reserved_multifault_selected_features.csv"
STEP21_ALL_FEATURES_PATH = TABLE_DIR / "21_reserved_multifault_all_features.csv"
STEP22_PREREGISTRATION_PATH = (
    CONFIG_DIR / "22_reserved_multifault_frozen_validation_preregistration.json"
)

CUSUM_K_GRID = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]

METHOD_SPECS = [
    {
        "method_key": "pca_q_cusum",
        "method": "PCA Q/SPE-CUSUM",
        "method_short": "Q-CUSUM",
        "input_key": "pca_q",
        "tune_k": True,
        "n_variables": None,
        "method_class": "global_pca",
    },
    {
        "method_key": "pca_t2_cusum",
        "method": "PCA Hotelling T2-CUSUM",
        "method_short": "T2-CUSUM",
        "input_key": "pca_t2",
        "tune_k": True,
        "n_variables": None,
        "method_class": "global_pca",
    },
    {
        "method_key": "pca_joint_cusum",
        "method": "PCA joint max-score CUSUM",
        "method_short": "Joint CUSUM",
        "input_key": "pca_joint",
        "tune_k": True,
        "n_variables": None,
        "method_class": "global_pca",
    },
    {
        "method_key": "lda_residual_cusum",
        "method": "Shrinkage-LDA residual CUSUM",
        "method_short": "LDA residual",
        "input_key": "lda_residual",
        "tune_k": True,
        "n_variables": 52,
        "method_class": "dense_supervised",
    },
    {
        "method_key": "chosen_family_all52_cusum",
        "method": "Chosen-family All-52 residual CUSUM",
        "method_short": "Family All-52",
        "input_key": "family_all52",
        "tune_k": True,
        "n_variables": 52,
        "method_class": "same_family_sparsity_control",
    },
    {
        "method_key": "chosen_family_top1_cusum",
        "method": "Chosen-family Top-1 residual CUSUM",
        "method_short": "Family Top-1",
        "input_key": "family_top1",
        "tune_k": True,
        "n_variables": 1,
        "method_class": "same_family_sparsity_control",
    },
    {
        "method_key": "frozen_multievidence_top5_cusum",
        "method": "Frozen multi-evidence Top-5 residual CUSUM",
        "method_short": "Proposed Top-5",
        "input_key": "proposed_top5",
        "tune_k": False,
        "fixed_k": PROPOSED_CUSUM_K,
        "n_variables": TOP_K,
        "method_class": "proposed",
    },
]
METHOD_ORDER = {spec["method_key"]: index for index, spec in enumerate(METHOD_SPECS)}


def require_file(path: Path, instruction: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, module_name: str):
    require_file(path, f"place {path.name} in the scripts directory")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not construct an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preregistration_payload(script_hash: str) -> dict:
    return {
        "registration_name": "Step 22 reserved multi-fault frozen validation",
        "script": "scripts/22_reserved_multifault_frozen_validation.py",
        "script_sha256": script_hash,
        "reserved_faults": RESERVED_FAULTS,
        "development_runs": [1, 250],
        "frozen_final_runs": [251, 500],
        "fault_start_sample": FAULT_START_SAMPLE,
        "normal_evaluation_runs": [251, 500],
        "step21_protocol_frozen": True,
        "step21_family_variable_orientation_retuning_allowed": False,
        "proposed_top_k": TOP_K,
        "proposed_cusum_k": PROPOSED_CUSUM_K,
        "proposed_cusum_k_retuning_allowed": False,
        "control_limit_rule": (
            "95th percentile of normal-validation run maxima for every method"
        ),
        "comparator_k_grid": CUSUM_K_GRID,
        "comparator_k_selection_data": "target fault development runs 1--250 only",
        "comparator_k_selection_rule": (
            "maximize development run detection, then post-fault point detection, "
            "then minimize conditional median delay, then distance to 0.05, then k"
        ),
        "methods": METHOD_SPECS,
        "primary_report": (
            "report every fault and every method; macro mean/median plus per-fault "
            "run detection, post-fault point detection, delay, and normal FAR"
        ),
        "normal_evaluation_used_for_tuning": False,
        "final_used_for_tuning": False,
        "companion_loaded": False,
        "freeze_order": (
            "verify Step 21 and freeze all 17 method states before any final row "
            "is subsetted"
        ),
        "raw_container_caveat": (
            "pyreadr materializes all testing runs in one RData object; final rows "
            "are not subsetted until every fault-specific state is frozen"
        ),
        "dependency_hashes": EXPECTED_HASHES,
    }


def register_protocol() -> None:
    script_hash = sha256_file(Path(__file__))
    payload = preregistration_payload(script_hash)
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    if STEP22_PREREGISTRATION_PATH.exists():
        existing = json.loads(
            STEP22_PREREGISTRATION_PATH.read_text(encoding="utf-8")
        )
        comparable_existing = {
            key: value for key, value in existing.items() if key != "registered_at_utc"
        }
        comparable_new = {
            key: value for key, value in payload.items() if key != "registered_at_utc"
        }
        if comparable_existing != comparable_new:
            raise ValueError(
                "A different Step-22 preregistration already exists. Preserve it; "
                "do not overwrite it after seeing any final result."
            )
        print(f"Identical preregistration already exists: {STEP22_PREREGISTRATION_PATH}")
        print(f"Registered script SHA-256: {script_hash}")
        return
    STEP22_PREREGISTRATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 22 protocol registered before any process data were read.")
    print(f"Reserved faults: {RESERVED_FAULTS}")
    print(f"Registered script SHA-256: {script_hash}")
    print(f"Saved preregistration: {STEP22_PREREGISTRATION_PATH}")


def verify_preregistration() -> dict:
    require_file(
        STEP22_PREREGISTRATION_PATH,
        "run this same script once with --register",
    )
    registration = json.loads(
        STEP22_PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    current_hash = sha256_file(Path(__file__))
    expected = preregistration_payload(current_hash)
    failures = []
    if registration.get("script_sha256") != current_hash:
        failures.append("script SHA-256 differs from preregistered bytes")
    for key, value in expected.items():
        if registration.get(key) != value:
            failures.append(f"registered field changed: {key}")
    if failures:
        raise ValueError(f"Step-22 preregistration verification failed: {failures}")
    return registration


def load_dependencies():
    modules = {}
    for filename, expected_hash in EXPECTED_HASHES.items():
        path = require_file(SCRIPT_DIR / filename, f"place frozen {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Dependency hash differs for {filename}: {actual_hash}"
            )
    step11 = load_module(
        SCRIPT_DIR / "11_unpaired_runlevel_screening.py",
        "step11_for_reserved_validation",
    )
    step15 = load_module(
        SCRIPT_DIR / "15_fair_sequential_baselines.py",
        "step15_for_reserved_validation",
    )
    step18 = load_module(
        SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py",
        "step18_for_reserved_validation",
    )
    step19 = load_module(
        SCRIPT_DIR / "19_multievidence_development_audit.py",
        "step19_for_reserved_validation",
    )
    step20 = load_module(
        SCRIPT_DIR / "20_frozen_multievidence_validation.py",
        "step20_for_reserved_validation",
    )
    if list(map(float, step15.CUSUM_K_GRID)) != CUSUM_K_GRID:
        raise ValueError("Step-15 comparator k grid changed")
    if int(step11.RUN_LEVEL_PERCENTILE) != CONTROL_LIMIT_PERCENTILE:
        raise ValueError("Step-11 control-limit percentile changed")
    if int(step19.TOP_K) != TOP_K:
        raise ValueError("Step-19 Top-K changed")
    if float(step20.CUSUM_K) != PROPOSED_CUSUM_K:
        raise ValueError("Step-20 proposed k changed")
    modules.update(
        step11=step11, step15=step15, step18=step18, step19=step19, step20=step20
    )
    return modules


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError("Could not parse a required boolean column")
    return mapped.astype(bool)


def verify_step21_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    for path in [
        STEP21_PREREGISTRATION_PATH,
        STEP21_CONFIG_PATH,
        STEP21_PROTOCOL_PATH,
        STEP21_SELECTED_PATH,
        STEP21_ALL_FEATURES_PATH,
    ]:
        require_file(path, "complete Step 21 and keep all 21_* outputs")

    prereg = json.loads(STEP21_PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    config = json.loads(STEP21_CONFIG_PATH.read_text(encoding="utf-8"))
    expected_hash = EXPECTED_HASHES["21_reserved_multifault_development_audit.py"]
    checks = {
        "Step21 prereg hash": prereg.get("script_sha256") == expected_hash,
        "Step21 execution hash": config.get("script_sha256_at_execution")
        == expected_hash,
        "Step21 prereg verified": config.get("preregistration_verified") is True,
        "Step21 development only": config.get("development_only") is True,
        "Step21 no performance": config.get("performance_evaluated") is False,
        "Step21 no normal evaluation": config.get("normal_evaluation_loaded") is False,
        "Step21 no companion": config.get("faulty_training_companion_loaded") is False,
        "Step21 no final subset": config.get("testing_final_rows_subset_or_analyzed")
        is False,
        "Step21 faults": config.get("reserved_faults") == RESERVED_FAULTS,
        "Step21 all estimable": config.get("estimable_faults") == RESERVED_FAULTS,
        "Step21 no abstention": config.get("abstained_faults") == [],
        "Step21 bootstraps": int(config.get("bootstrap_replicates", -1)) == 2000,
        "Step21 permutations": int(config.get("permutation_replicates", -1))
        == 2000,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Step-21 configuration verification failed: {failed}")

    protocol = pd.read_csv(STEP21_PROTOCOL_PATH, encoding="utf-8-sig")
    selected = pd.read_csv(STEP21_SELECTED_PATH, encoding="utf-8-sig")
    all_features = pd.read_csv(STEP21_ALL_FEATURES_PATH, encoding="utf-8-sig")
    protocol["all_selected_strict_fwer"] = parse_bool(
        protocol["all_selected_strict_fwer"]
    )
    selected["recommended_by_fixed_development_rule"] = parse_bool(
        selected["recommended_by_fixed_development_rule"]
    )
    selected["primary_stable_global_fdr"] = parse_bool(
        selected["primary_stable_global_fdr"]
    )
    selected["strict_stable_max_t_fwer"] = parse_bool(
        selected["strict_stable_max_t_fwer"]
    )

    if protocol.shape[0] != len(RESERVED_FAULTS):
        raise ValueError("Step-21 protocol does not contain 17 rows")
    if protocol["fault_id"].astype(int).tolist() != RESERVED_FAULTS:
        raise ValueError("Step-21 protocol fault order changed")
    if not protocol["branch_status"].eq("estimable_frozen_after_step21").all():
        raise ValueError("Step-21 contains an unexpected abstention")
    if not protocol["selected_count"].astype(int).eq(TOP_K).all():
        raise ValueError("Step-21 contains a non-Top-5 protocol")
    if not protocol["all_selected_strict_fwer"].all():
        raise ValueError("A frozen Step-21 branch failed strict FWER")
    if all_features.shape[0] != len(RESERVED_FAULTS) * 5 * 52:
        raise ValueError("Step-21 all-feature table is incomplete")
    if all_features.duplicated(["fault_id", "family_key", "variable"]).any():
        raise ValueError("Step-21 all-feature table has duplicate candidates")

    for _, row in protocol.iterrows():
        fault_id = int(row["fault_id"])
        family_key = str(row["family_key"])
        current = selected[
            (selected["fault_id"].astype(int) == fault_id)
            & (selected["family_key"].astype(str) == family_key)
            & selected["recommended_by_fixed_development_rule"]
        ].sort_values("selection_rank")
        if len(current) != TOP_K:
            raise ValueError(f"Fault {fault_id}: frozen selected rows are incomplete")
        if ";".join(current["variable"].astype(str)) != str(
            row["selected_variables"]
        ):
            raise ValueError(f"Fault {fault_id}: selected variables changed")
        signs = ";".join(current["orientation_sign"].astype(int).astype(str))
        if signs != str(row["orientation_signs"]):
            raise ValueError(f"Fault {fault_id}: orientation signs changed")
        if not current["primary_stable_global_fdr"].all():
            raise ValueError(f"Fault {fault_id}: selected primary gate failed")
        if not current["strict_stable_max_t_fwer"].all():
            raise ValueError(f"Fault {fault_id}: selected strict FWER failed")
    return protocol, selected, all_features, config


def validate_variable_order(variable_cols: list[str]) -> None:
    expected = [
        *[f"xmeas_{index}" for index in range(1, 42)],
        *[f"xmv_{index}" for index in range(1, 12)],
    ]
    if variable_cols != expected:
        raise ValueError("The expected 52-variable order changed")


def frozen_top5_definition(
    fault_id: int,
    family_key: str,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    definition = selected[
        (selected["fault_id"].astype(int) == fault_id)
        & (selected["family_key"].astype(str) == family_key)
        & selected["recommended_by_fixed_development_rule"]
    ].sort_values("selection_rank").copy()
    if len(definition) != TOP_K:
        raise ValueError(f"Fault {fault_id}: Top-5 definition missing")
    return definition


def all52_definition(
    fault_id: int,
    family_key: str,
    all_features: pd.DataFrame,
    variable_cols: list[str],
) -> pd.DataFrame:
    definition = all_features[
        (all_features["fault_id"].astype(int) == fault_id)
        & (all_features["family_key"].astype(str) == family_key)
    ].copy()
    order = {variable: index + 1 for index, variable in enumerate(variable_cols)}
    definition["selection_rank"] = definition["variable"].map(order)
    definition = definition.sort_values("selection_rank")
    if len(definition) != 52 or definition["selection_rank"].isna().any():
        raise ValueError(f"Fault {fault_id}: All-52 family definition is incomplete")
    return definition


def tune_k_on_development(
    step11,
    step15,
    step18,
    fault_id: int,
    spec: dict,
    normal_validation_input: pd.DataFrame,
    development_input: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for k in CUSUM_K_GRID:
        validation_stat = step15.compute_cusum(normal_validation_input, k)
        limit, _ = step11.determine_limit(validation_stat)
        validation_metrics, _ = step11.evaluate_normal(validation_stat, limit)
        development_stat = step15.compute_cusum(development_input, k)
        development_metrics, _ = step18.evaluate_generic_fault(
            step11, development_stat, float(limit), FAULT_START_SAMPLE
        )
        rows.append(
            {
                "fault_id": fault_id,
                "method_key": spec["method_key"],
                "method": spec["method"],
                "cusum_k": float(k),
                "control_limit_on_normal_validation": float(limit),
                "normal_validation_run_far": float(
                    validation_metrics["normal_run_level_far"]
                ),
                "development_run_detection_rate": float(
                    development_metrics["fault_run_level_detection_rate"]
                ),
                "development_post_fault_point_detection_rate": float(
                    development_metrics["fault_post_fault_point_detection_rate"]
                ),
                "development_median_alarm_delay_min": float(
                    development_metrics["median_alarm_delay_min"]
                ),
            }
        )
    table = pd.DataFrame(rows)
    sortable = table.copy()
    sortable["delay_for_sort"] = sortable[
        "development_median_alarm_delay_min"
    ].fillna(np.inf)
    sortable["distance_from_primary_k"] = np.abs(
        sortable["cusum_k"] - PROPOSED_CUSUM_K
    )
    chosen = sortable.sort_values(
        [
            "development_run_detection_rate",
            "development_post_fault_point_detection_rate",
            "delay_for_sort",
            "distance_from_primary_k",
            "cusum_k",
        ],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    chosen_k = float(chosen["cusum_k"])
    table["selected_on_development"] = np.isclose(table["cusum_k"], chosen_k)
    return chosen_k, table


def freeze_method(
    step11,
    step15,
    step18,
    fault_id: int,
    spec: dict,
    normal_validation_input: pd.DataFrame,
    development_input: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    if spec["tune_k"]:
        k, tuning = tune_k_on_development(
            step11,
            step15,
            step18,
            fault_id,
            spec,
            normal_validation_input,
            development_input,
        )
        k_source = f"Fault-{fault_id} development only"
    else:
        k = float(spec["fixed_k"])
        k_source = "frozen before Step-22 final evaluation"
        tuning = pd.DataFrame(
            [
                {
                    "fault_id": fault_id,
                    "method_key": spec["method_key"],
                    "method": spec["method"],
                    "cusum_k": k,
                    "selected_on_development": True,
                    "selection_status": "fixed_not_tuned",
                }
            ]
        )

    validation_stat = step15.compute_cusum(normal_validation_input, k)
    limit, validation_run_max = step11.determine_limit(validation_stat)
    validation_metrics, _ = step11.evaluate_normal(validation_stat, limit)
    development_stat = step15.compute_cusum(development_input, k)
    development_metrics, development_runs = step18.evaluate_generic_fault(
        step11,
        development_stat,
        float(limit),
        FAULT_START_SAMPLE,
    )
    parameters = {
        "fault_id": fault_id,
        "method_key": spec["method_key"],
        "method": spec["method"],
        "method_short": spec["method_short"],
        "method_class": spec["method_class"],
        "n_variables": spec["n_variables"],
        "cusum_k": float(k),
        "cusum_k_source": k_source,
        "control_limit": float(limit),
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "normal_validation_run_level_far": float(
            validation_metrics["normal_run_level_far"]
        ),
        "final_retuning_allowed": False,
    }
    validation_run_max = validation_run_max.copy()
    validation_run_max.insert(0, "fault_id", fault_id)
    validation_run_max.insert(1, "method_key", spec["method_key"])
    validation_run_max.insert(2, "cusum_k", float(k))
    validation_run_max.insert(3, "control_limit", float(limit))
    return parameters, tuning, development_metrics, development_runs, validation_run_max


def annotate_runs(
    frame: pd.DataFrame,
    fault_id: int,
    spec: dict,
    dataset: str,
    k: float,
) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "dataset", dataset)
    output.insert(0, "cusum_k", float(k))
    output.insert(0, "method", spec["method"])
    output.insert(0, "method_key", spec["method_key"])
    output.insert(0, "fault_id", fault_id)
    return output


def result_row(
    fault_id: int,
    spec: dict,
    family_key: str,
    dataset: str,
    parameters: dict,
    normal_metrics: dict,
    fault_metrics: dict,
) -> dict:
    row = {
        "fault_id": fault_id,
        "method_key": spec["method_key"],
        "method": spec["method"],
        "method_short": spec["method_short"],
        "method_class": spec["method_class"],
        "method_order": METHOD_ORDER[spec["method_key"]],
        "evidence_family": family_key,
        "n_variables": spec["n_variables"],
        "dataset": dataset,
        "cusum_k": parameters["cusum_k"],
        "cusum_k_source": parameters["cusum_k_source"],
        "control_limit": parameters["control_limit"],
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "normal_validation_run_level_far": parameters[
            "normal_validation_run_level_far"
        ],
        "normal_run_level_far": float(normal_metrics["normal_run_level_far"]),
        "normal_run_level_far_ci_low": float(
            normal_metrics["normal_run_level_far_ci_low"]
        ),
        "normal_run_level_far_ci_high": float(
            normal_metrics["normal_run_level_far_ci_high"]
        ),
        "protocol_status": "frozen before final subset; no final retuning",
    }
    row.update(fault_metrics)
    return row


def plot_heatmap(
    final_results: pd.DataFrame,
    metric: str,
    title: str,
    colorbar_label: str,
    filename: str,
) -> Path:
    pivot = final_results.pivot(
        index="fault_id", columns="method_key", values=metric
    ).reindex(index=RESERVED_FAULTS, columns=list(METHOD_ORDER))
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete heatmap matrix for {metric}")
    matrix = 100.0 * pivot.to_numpy(dtype=float)
    labels = [
        next(spec["method_short"] for spec in METHOD_SPECS if spec["method_key"] == key)
        for key in pivot.columns
    ]
    figure, axis = plt.subplots(figsize=(12.5, 8.8))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(RESERVED_FAULTS)))
    axis.set_yticklabels([f"Fault {fault}" for fault in RESERVED_FAULTS])
    axis.set_title(title)
    axis.set_xlabel("Frozen sequential method")
    axis.set_ylabel("Reserved TEP fault")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value < 35 or value > 75 else "black",
                fontsize=8,
                fontweight="bold" if column == matrix.shape[1] - 1 else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    path = FIGURE_DIR / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_development_final(proposed: pd.DataFrame) -> Path:
    development = proposed[
        proposed["dataset"] == "development_fit_runs_1_250"
    ].set_index("fault_id")
    final = proposed[
        proposed["dataset"] == "faulty_testing_final_full_960"
    ].set_index("fault_id")
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    metrics = [
        ("fault_run_level_detection_rate", "Run detection (%)"),
        ("fault_post_fault_point_detection_rate", "Post-fault point detection (%)"),
    ]
    x = np.arange(len(RESERVED_FAULTS))
    for axis, (metric, label) in zip(axes, metrics):
        dev = 100 * development.loc[RESERVED_FAULTS, metric].to_numpy(dtype=float)
        fin = 100 * final.loc[RESERVED_FAULTS, metric].to_numpy(dtype=float)
        for index in range(len(x)):
            axis.plot([x[index], x[index]], [dev[index], fin[index]], color="#9A9A9A")
        axis.scatter(x, dev, label="Development 1-250", color="#4C78A8", s=34)
        axis.scatter(x, fin, label="Frozen final 251-500", color="#D95F59", s=34)
        axis.set_ylabel(label)
        axis.set_ylim(-2, 103)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(fault) for fault in RESERVED_FAULTS])
    axes[-1].set_xlabel("Reserved TEP fault")
    figure.suptitle("Frozen proposed Top-5: development-to-final transfer")
    figure.tight_layout()
    path = FIGURE_DIR / "fig22c_proposed_development_final_transfer.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def build_crossfault_summary(final_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METHOD_SPECS:
        current = final_results[final_results["method_key"] == spec["method_key"]]
        if len(current) != len(RESERVED_FAULTS):
            raise ValueError(f"Incomplete final rows for {spec['method_key']}")
        rows.append(
            {
                "method_key": spec["method_key"],
                "method_short": spec["method_short"],
                "method_order": METHOD_ORDER[spec["method_key"]],
                "n_faults": int(len(current)),
                "macro_mean_normal_far": float(current["normal_run_level_far"].mean()),
                "macro_mean_run_detection": float(
                    current["fault_run_level_detection_rate"].mean()
                ),
                "macro_median_run_detection": float(
                    current["fault_run_level_detection_rate"].median()
                ),
                "macro_mean_point_detection": float(
                    current["fault_post_fault_point_detection_rate"].mean()
                ),
                "macro_median_point_detection": float(
                    current["fault_post_fault_point_detection_rate"].median()
                ),
                "macro_median_alarm_delay_min": float(
                    current["median_alarm_delay_min"].median()
                ),
                "faults_run_detection_ge_90pct": int(
                    (current["fault_run_level_detection_rate"] >= 0.90).sum()
                ),
                "faults_run_detection_ge_50pct": int(
                    (current["fault_run_level_detection_rate"] >= 0.50).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("method_order").reset_index(drop=True)


def run_validation() -> None:
    registration = verify_preregistration()
    modules = load_dependencies()
    step11 = modules["step11"]
    step15 = modules["step15"]
    step18 = modules["step18"]
    step20 = modules["step20"]
    protocol, selected, all_features, step21_config = verify_step21_outputs()

    variable_cols = step11.load_variable_columns()
    validate_variable_order(variable_cols)

    print("=" * 96)
    print("Step 22: Preregistered frozen validation on 17 reserved TEP faults")
    print("=" * 96)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print("All Step-21 protocol and dependency checks passed.")
    print("No final row will be subsetted until every fault-specific state is frozen.")

    normal_train = pd.read_pickle(require_file(NORMAL_TRAIN_PATH, "run Step 02"))
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    eigenvalues = step15.derive_retained_eigenvalues(
        normal_train, variable_cols, model
    )
    del normal_train
    gc.collect()

    normal_validation = pd.read_pickle(
        require_file(NORMAL_VALIDATION_PATH, "run Step 02")
    )
    normal_validation_res, normal_validation_stats, residual_cols = (
        step15.compute_pca_outputs(
            normal_validation, variable_cols, model, eigenvalues
        )
    )
    del normal_validation
    gc.collect()
    centers = step20.residual_centers(normal_validation_res, residual_cols)
    global_parameters = step15.fit_global_transforms(normal_validation_stats)
    global_validation_inputs = step15.transform_global_inputs(
        normal_validation_stats, global_parameters
    )

    testing, testing_object, testing_names = step18.read_rdata_frame(
        require_file(RAW_TESTING_PATH, "place TEP_Faulty_Testing.RData in data/raw"),
        "faulty_testing",
    )
    testing_shape = tuple(int(value) for value in testing.shape)
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in testing.columns]
    if missing:
        raise ValueError(f"Faulty-Testing object is missing columns: {missing}")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")

    fault_states = {}
    all_parity = []
    all_tuning = []
    all_parameters = []
    all_scalers = []
    all_lda_weights = []
    all_validation_max = []
    all_data_audits = []
    all_development_runs = []

    # Stage A: development-only reconstruction, tuning, and freezing for all faults.
    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"Freezing Fault {fault_id} from Step 21 and development runs only")
        print("=" * 96)
        protocol_row = protocol[protocol["fault_id"].astype(int) == fault_id].iloc[0]
        family_key = str(protocol_row["family_key"])
        development_mask = (fault_numbers == fault_id) & run_numbers.between(1, 250)
        development = testing.loc[development_mask, required].copy()
        development, audit = step18.validate_fault_frame(
            development,
            variable_cols,
            fault_id,
            EXPECTED_DEVELOPMENT_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{fault_id}_development_freeze",
        )
        audit["stage"] = "development_freeze_before_any_final_subset"
        audit["final_rows_subsetted_at_this_stage"] = False
        all_data_audits.append(audit)

        development_res, development_stats, current_cols = step15.compute_pca_outputs(
            development, variable_cols, model, eigenvalues
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: residual order changed")
        del development
        gc.collect()

        top5_definition = frozen_top5_definition(
            fault_id, family_key, selected
        )
        proposed_scaler = step20.fit_frozen_scaler(
            normal_validation_res, centers, top5_definition
        )
        parity = step20.verify_step19_reconstruction(
            fault_id,
            normal_validation_res,
            development_res,
            proposed_scaler,
        )
        parity["source_step"] = 21
        all_parity.append(parity)
        print(
            "Step-21 reconstruction parity passed; maximum absolute difference="
            f"{max(parity['normal_difference'].abs().max(), parity['fault_difference'].abs().max()):.3e}"
        )

        full_definition = all52_definition(
            fault_id, family_key, all_features, variable_cols
        )
        all52_scaler = step20.fit_frozen_scaler(
            normal_validation_res, centers, full_definition
        )
        top1_definition = top5_definition.head(1).copy()
        top1_definition["selection_rank"] = 1
        top1_scaler = step20.fit_frozen_scaler(
            normal_validation_res, centers, top1_definition
        )

        for method_key, scaler in [
            ("chosen_family_all52_cusum", all52_scaler),
            ("chosen_family_top1_cusum", top1_scaler),
            ("frozen_multievidence_top5_cusum", proposed_scaler),
        ]:
            current = scaler.copy()
            current.insert(1, "method_key", method_key)
            all_scalers.append(current)

        weights, weight_table, lda_metadata = step15.fit_lda_score(
            step11,
            normal_validation_res,
            development_res,
            residual_cols,
            variable_cols,
        )
        weight_table.insert(0, "fault_id", fault_id)
        all_lda_weights.append(weight_table)
        lda_validation_raw = step15.apply_lda_score(
            normal_validation_res, residual_cols, weights
        )
        lda_standardizer = step15.fit_scalar_standardizer(lda_validation_raw)

        global_development_inputs = step15.transform_global_inputs(
            development_stats, global_parameters
        )
        lda_development_raw = step15.apply_lda_score(
            development_res, residual_cols, weights
        )
        development_inputs = {
            "pca_q": global_development_inputs["q"],
            "pca_t2": global_development_inputs["t2"],
            "pca_joint": global_development_inputs["joint"],
            "lda_residual": step15.apply_scalar_standardizer(
                lda_development_raw, lda_standardizer
            ),
            "family_all52": step20.construct_scalar_input(
                development_res, all52_scaler
            ),
            "family_top1": step20.construct_scalar_input(
                development_res, top1_scaler
            ),
            "proposed_top5": step20.construct_scalar_input(
                development_res, proposed_scaler
            ),
        }
        validation_inputs = {
            "pca_q": global_validation_inputs["q"],
            "pca_t2": global_validation_inputs["t2"],
            "pca_joint": global_validation_inputs["joint"],
            "lda_residual": step15.apply_scalar_standardizer(
                lda_validation_raw, lda_standardizer
            ),
            "family_all52": step20.construct_scalar_input(
                normal_validation_res, all52_scaler
            ),
            "family_top1": step20.construct_scalar_input(
                normal_validation_res, top1_scaler
            ),
            "proposed_top5": step20.construct_scalar_input(
                normal_validation_res, proposed_scaler
            ),
        }

        method_states = {}
        for spec in METHOD_SPECS:
            parameters, tuning, dev_metrics, dev_runs, validation_max = freeze_method(
                step11,
                step15,
                step18,
                fault_id,
                spec,
                validation_inputs[spec["input_key"]],
                development_inputs[spec["input_key"]],
            )
            parameters["evidence_family"] = (
                family_key if spec["method_class"] in {"same_family_sparsity_control", "proposed"}
                else spec["method_class"]
            )
            all_parameters.append(parameters)
            all_tuning.append(tuning)
            all_validation_max.append(validation_max)
            all_development_runs.append(
                annotate_runs(
                    dev_runs,
                    fault_id,
                    spec,
                    "development_fit_runs_1_250",
                    parameters["cusum_k"],
                )
            )
            method_states[spec["method_key"]] = {
                "parameters": parameters,
                "development_metrics": dev_metrics,
            }
            print(
                f"  {spec['method_short']}: k={parameters['cusum_k']:.2f}, "
                f"development detection={100*dev_metrics['fault_run_level_detection_rate']:.1f}%"
            )

        fault_states[fault_id] = {
            "family_key": family_key,
            "proposed_scaler": proposed_scaler,
            "all52_scaler": all52_scaler,
            "top1_scaler": top1_scaler,
            "lda_weights": weights,
            "lda_standardizer": lda_standardizer,
            "lda_metadata": lda_metadata,
            "methods": method_states,
        }
        del development_res, development_stats, development_inputs, validation_inputs
        del global_development_inputs, lda_development_raw, lda_validation_raw
        gc.collect()

    if sorted(fault_states) != RESERVED_FAULTS:
        raise ValueError("Not every reserved fault was frozen")
    print("\nAll 17 fault-specific states are frozen. Final subsetting now begins.")

    # Stage B: independent normal evaluation and final runs.
    normal_evaluation = pd.read_pickle(
        require_file(NORMAL_EVALUATION_PATH, "run Step 02")
    )
    normal_evaluation_res, normal_evaluation_stats, current_cols = (
        step15.compute_pca_outputs(
            normal_evaluation, variable_cols, model, eigenvalues
        )
    )
    if current_cols != residual_cols:
        raise ValueError("Normal-evaluation residual order changed")
    del normal_evaluation
    gc.collect()
    global_evaluation_inputs = step15.transform_global_inputs(
        normal_evaluation_stats, global_parameters
    )

    result_rows = []
    all_run_tables = list(all_development_runs)
    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 96)
        print(f"Frozen final evaluation: Fault {fault_id}")
        print("=" * 96)
        state = fault_states[fault_id]
        final_mask = (fault_numbers == fault_id) & run_numbers.between(251, 500)
        final = testing.loc[final_mask, required].copy()
        final, audit = step18.validate_fault_frame(
            final,
            variable_cols,
            fault_id,
            EXPECTED_FINAL_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{fault_id}_frozen_final_runs_251_500",
        )
        audit["stage"] = "frozen_final_evaluation"
        audit["all_states_frozen_before_subset"] = True
        all_data_audits.append(audit)
        final_res, final_stats, current_cols = step15.compute_pca_outputs(
            final, variable_cols, model, eigenvalues
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: final residual order changed")
        del final
        gc.collect()

        global_final_inputs = step15.transform_global_inputs(
            final_stats, global_parameters
        )
        lda_normal_raw = step15.apply_lda_score(
            normal_evaluation_res, residual_cols, state["lda_weights"]
        )
        lda_final_raw = step15.apply_lda_score(
            final_res, residual_cols, state["lda_weights"]
        )
        normal_inputs = {
            "pca_q": global_evaluation_inputs["q"],
            "pca_t2": global_evaluation_inputs["t2"],
            "pca_joint": global_evaluation_inputs["joint"],
            "lda_residual": step15.apply_scalar_standardizer(
                lda_normal_raw, state["lda_standardizer"]
            ),
            "family_all52": step20.construct_scalar_input(
                normal_evaluation_res, state["all52_scaler"]
            ),
            "family_top1": step20.construct_scalar_input(
                normal_evaluation_res, state["top1_scaler"]
            ),
            "proposed_top5": step20.construct_scalar_input(
                normal_evaluation_res, state["proposed_scaler"]
            ),
        }
        final_inputs = {
            "pca_q": global_final_inputs["q"],
            "pca_t2": global_final_inputs["t2"],
            "pca_joint": global_final_inputs["joint"],
            "lda_residual": step15.apply_scalar_standardizer(
                lda_final_raw, state["lda_standardizer"]
            ),
            "family_all52": step20.construct_scalar_input(
                final_res, state["all52_scaler"]
            ),
            "family_top1": step20.construct_scalar_input(
                final_res, state["top1_scaler"]
            ),
            "proposed_top5": step20.construct_scalar_input(
                final_res, state["proposed_scaler"]
            ),
        }

        for spec in METHOD_SPECS:
            method_state = state["methods"][spec["method_key"]]
            parameters = method_state["parameters"]
            k = float(parameters["cusum_k"])
            limit = float(parameters["control_limit"])
            normal_stat = step15.compute_cusum(normal_inputs[spec["input_key"]], k)
            normal_metrics, normal_runs = step11.evaluate_normal(normal_stat, limit)
            final_stat = step15.compute_cusum(final_inputs[spec["input_key"]], k)
            final_metrics, final_runs = step18.evaluate_generic_fault(
                step11, final_stat, limit, FAULT_START_SAMPLE
            )
            development_metrics = method_state["development_metrics"]
            result_rows.append(
                result_row(
                    fault_id,
                    spec,
                    state["family_key"],
                    "development_fit_runs_1_250",
                    parameters,
                    normal_metrics,
                    development_metrics,
                )
            )
            result_rows.append(
                result_row(
                    fault_id,
                    spec,
                    state["family_key"],
                    "faulty_testing_final_full_960",
                    parameters,
                    normal_metrics,
                    final_metrics,
                )
            )
            all_run_tables.append(
                annotate_runs(
                    normal_runs,
                    fault_id,
                    spec,
                    "normal_evaluation",
                    k,
                )
            )
            all_run_tables.append(
                annotate_runs(
                    final_runs,
                    fault_id,
                    spec,
                    "faulty_testing_final_full_960",
                    k,
                )
            )
            print(
                f"  {spec['method_short']}: FAR={100*normal_metrics['normal_run_level_far']:.1f}%, "
                f"detection={100*final_metrics['fault_run_level_detection_rate']:.1f}%, "
                f"point={100*final_metrics['fault_post_fault_point_detection_rate']:.1f}%, "
                f"delay={final_metrics['median_alarm_delay_min']:.1f} min"
            )
        del final_res, final_stats, final_inputs, normal_inputs
        del global_final_inputs, lda_normal_raw, lda_final_raw
        gc.collect()

    results = pd.DataFrame(result_rows).sort_values(
        ["fault_id", "dataset", "method_order"]
    ).reset_index(drop=True)
    final_results = results[
        results["dataset"] == "faulty_testing_final_full_960"
    ].copy()
    if len(final_results) != len(RESERVED_FAULTS) * len(METHOD_SPECS):
        raise ValueError("Final result matrix is incomplete")
    crossfault = build_crossfault_summary(final_results)
    run_results = pd.concat(all_run_tables, ignore_index=True)
    parity = pd.concat(all_parity, ignore_index=True)
    if not parity["exact_with_tolerance_1e_10"].all():
        raise ValueError("A Step-21 reconstruction parity row failed")
    tuning = pd.concat(all_tuning, ignore_index=True)
    parameters = pd.DataFrame(all_parameters).sort_values(
        ["fault_id", "method_key"]
    )
    scalers = pd.concat(all_scalers, ignore_index=True)
    lda_weights = pd.concat(all_lda_weights, ignore_index=True)
    validation_max = pd.concat(all_validation_max, ignore_index=True)
    data_audit = pd.DataFrame(all_data_audits)

    output_paths = {
        "results": TABLE_DIR / "22_reserved_multifault_results.csv",
        "crossfault": TABLE_DIR / "22_reserved_multifault_crossfault_summary.csv",
        "runs": TABLE_DIR / "22_reserved_multifault_run_results.csv",
        "k_tuning": TABLE_DIR / "22_reserved_multifault_k_tuning.csv",
        "parameters": TABLE_DIR / "22_reserved_multifault_method_parameters.csv",
        "scalers": TABLE_DIR / "22_reserved_multifault_local_scalers.csv",
        "lda_weights": TABLE_DIR / "22_reserved_multifault_lda_weights.csv",
        "validation_max": TABLE_DIR / "22_reserved_multifault_validation_run_max.csv",
        "parity": TABLE_DIR / "22_step21_reconstruction_parity.csv",
        "data_audit": TABLE_DIR / "22_reserved_multifault_data_audit.csv",
    }
    for key, frame in [
        ("results", results),
        ("crossfault", crossfault),
        ("runs", run_results),
        ("k_tuning", tuning),
        ("parameters", parameters),
        ("scalers", scalers),
        ("lda_weights", lda_weights),
        ("validation_max", validation_max),
        ("parity", parity),
        ("data_audit", data_audit),
    ]:
        frame.to_csv(output_paths[key], index=False, encoding="utf-8-sig")

    figure_paths = [
        plot_heatmap(
            final_results,
            "fault_run_level_detection_rate",
            "Frozen final run-level detection across 17 reserved TEP faults",
            "Run detection (%)",
            "fig22a_reserved_final_run_detection.png",
        ),
        plot_heatmap(
            final_results,
            "fault_post_fault_point_detection_rate",
            "Frozen final post-fault point detection across 17 reserved TEP faults",
            "Post-fault point detection (%)",
            "fig22b_reserved_final_point_detection.png",
        ),
        plot_development_final(
            results[results["method_key"] == "frozen_multievidence_top5_cusum"]
        ),
    ]

    config = {
        "script": "scripts/22_reserved_multifault_frozen_validation.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "preregistration": str(
            STEP22_PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)
        ),
        "preregistration_registered_at_utc": registration["registered_at_utc"],
        "preregistration_verified": True,
        "step21_script_sha256_verified": EXPECTED_HASHES[
            "21_reserved_multifault_development_audit.py"
        ],
        "step21_preregistration_verified": True,
        "step21_reconstruction_all_passed": True,
        "maximum_absolute_reconstruction_difference": float(
            max(parity["normal_difference"].abs().max(), parity["fault_difference"].abs().max())
        ),
        "reserved_faults": RESERVED_FAULTS,
        "development_runs": [1, 250],
        "final_runs": [251, 500],
        "normal_evaluation_runs": [251, 500],
        "all_states_frozen_before_any_final_subset": True,
        "final_used_for_tuning": False,
        "normal_evaluation_used_for_tuning": False,
        "companion_loaded": False,
        "raw_testing_container_materialized": True,
        "raw_testing_container_caveat": registration["raw_container_caveat"],
        "raw_testing_file": str(RAW_TESTING_PATH.relative_to(PROJECT_ROOT)),
        "raw_testing_size_bytes": int(RAW_TESTING_PATH.stat().st_size),
        "raw_testing_object": testing_object,
        "raw_testing_object_names": testing_names,
        "raw_testing_complete_shape": testing_shape,
        "methods": METHOD_SPECS,
        "comparator_k_grid": CUSUM_K_GRID,
        "proposed_top_k": TOP_K,
        "proposed_cusum_k": PROPOSED_CUSUM_K,
        "control_limit_percentile": CONTROL_LIMIT_PERCENTILE,
        "dependency_hashes": EXPECTED_HASHES,
        "step21_configuration": step21_config,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
        "next_step_constraint": (
            "report every final result; any repeated-split or family-ambiguity "
            "analysis must be labelled secondary and may not replace these rows"
        ),
    }
    config_path = CONFIG_DIR / "22_reserved_multifault_frozen_validation.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 96)
    print("Step 22 cross-fault frozen-final summary")
    print("=" * 96)
    print(crossfault.to_string(index=False))
    print(f"Saved configuration: {config_path}")
    print("Step 22 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="write the immutable Step-22 preregistration without reading process data",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify preregistration and execute frozen final validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_protocol()
    else:
        run_validation()


if __name__ == "__main__":
    main()
