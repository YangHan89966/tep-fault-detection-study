"""Step 21: Preregistered development-only audit on the reserved TEP faults.

Scientific role
---------------
Steps 19--20 established a five-family residual-evidence pipeline on Faults 9
and 15 after the signed-level branch abstained.  Step 21 freezes that complete
pipeline *before* inspecting any additional fault outcome and applies it to
every remaining TEP fault except Fault 3 (original method-development target)
and Faults 9/15 (multi-evidence method-development targets).

Reserved fault set:
    1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20

Only testing runs 1--250 are used in this script.  No normal-evaluation data,
testing runs 251--500, or Faulty-Training companion data enter screening,
family choice, variable choice, orientation choice, or performance analysis.

Important container caveat
--------------------------
``TEP_Faulty_Testing.RData`` stores all runs in one R object, so pyreadr must
materialize the complete container in memory.  This script immediately forms
development masks and never subsets, transforms, summarizes, or evaluates
runs 251--500.  The source code and preregistration make that separation
auditable, but the physical RData container is not an access-controlled blind
holdout.

Two-stage use
-------------
Place in:
    TEP_Fault3_Project/scripts/21_reserved_multifault_development_audit.py

From the project root, preregister first (fast; reads no process data):
    python scripts/21_reserved_multifault_development_audit.py --register

Then run the development-only audit (long-running):
    python scripts/21_reserved_multifault_development_audit.py --run

The ``--run`` action stops if the script bytes differ from the registered
SHA-256 hash.  Only files beginning with ``21_`` and ``fig21`` are written.
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
EXCLUDED_DEVELOPMENT_FAULTS = {
    3: "original signed-residual method-development target",
    9: "multi-evidence family-development target",
    15: "multi-evidence family-development target",
}
EXPECTED_ALL_FAULTS = list(range(1, 21))
EXPECTED_DEVELOPMENT_RUNS = np.arange(1, 251, dtype=np.int64)
EXPECTED_SAMPLE_MIN = 1
EXPECTED_SAMPLE_MAX = 960
FAULT_START_SAMPLE = 161
SCREEN_END_SAMPLE = 320
TOP_K = 5

EXPECTED_STEP18_SHA256 = (
    "d0f7b9b3318c93bfd4f64016d994f2f0aa2806f083df7487af44e62a04e65053"
)
EXPECTED_STEP19_SHA256 = (
    "0a19a1c0210669c201d7cbe42c5824aed2a38cb8a08405c68879cc415a30ee6b"
)

NORMAL_TRAIN_PATH = PROCESSED_DIR / "normal_train.pkl"
NORMAL_VALIDATION_PATH = PROCESSED_DIR / "normal_validation_runs_1_250.pkl"
RAW_TESTING_PATH = RAW_DIR / "TEP_Faulty_Testing.RData"
PREREGISTRATION_PATH = CONFIG_DIR / "21_reserved_multifault_preregistration.json"


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


def protocol_payload(script_hash: str) -> dict:
    return {
        "registration_name": "Step 21 reserved multi-fault development audit",
        "script": "scripts/21_reserved_multifault_development_audit.py",
        "script_sha256": script_hash,
        "reserved_fault_rule": (
            "all TEP fault IDs 1--20 except method-development faults 3, 9, and 15"
        ),
        "reserved_faults": RESERVED_FAULTS,
        "excluded_development_faults": {
            str(key): value for key, value in EXCLUDED_DEVELOPMENT_FAULTS.items()
        },
        "fault_selection_uses_outcomes": False,
        "development_runs": [1, 250],
        "reserved_final_runs": [251, 500],
        "screening_samples": [FAULT_START_SAMPLE, SCREEN_END_SAMPLE],
        "candidate_families": [
            "signed_level",
            "absolute_deviation",
            "squared_deviation",
            "absolute_increment",
            "lag1_product",
        ],
        "candidate_tests_per_fault": 260,
        "bootstrap_replicates": 2000,
        "permutation_replicates": 2000,
        "primary_feature_gate": (
            "two-sided 95% independent-run bootstrap CI excludes zero AND "
            "global BH q<=0.05 across all 260 candidates for that fault"
        ),
        "strict_diagnostic": "studentized max-|t| permutation FWER p<=0.05",
        "family_eligibility": "at least five primary-gate-stable residuals",
        "family_choice_rule": (
            "among eligible families maximize the median absolute Hedges g of "
            "the selected Top-5; break exact ties by the frozen family priority"
        ),
        "top_k": TOP_K,
        "orientation_rule": "sign of the development-minus-normal run-mean shift",
        "normal_reference": "normal training for PCA; normal validation for screening",
        "performance_evaluated_in_step21": False,
        "normal_evaluation_loaded": False,
        "faulty_training_companion_loaded": False,
        "testing_final_rows_subset_or_analyzed": False,
        "raw_container_caveat": (
            "pyreadr materializes the complete TEP_Faulty_Testing.RData object, "
            "but Step 21 only subsets and analyzes runs 1--250"
        ),
        "planned_frozen_evaluation": (
            "a separate script must verify this Step-21 script/config and then "
            "evaluate the frozen per-fault branch on runs 251--500"
        ),
        "dependency_hashes": {
            "18_faults9_15_prespecified_generalization.py": EXPECTED_STEP18_SHA256,
            "19_multievidence_development_audit.py": EXPECTED_STEP19_SHA256,
        },
    }


def register_protocol() -> None:
    script_hash = sha256_file(Path(__file__))
    payload = protocol_payload(script_hash)
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()

    if PREREGISTRATION_PATH.exists():
        existing = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
        comparable_existing = {
            key: value for key, value in existing.items() if key != "registered_at_utc"
        }
        comparable_new = {
            key: value for key, value in payload.items() if key != "registered_at_utc"
        }
        if comparable_existing != comparable_new:
            raise ValueError(
                "A different Step-21 preregistration already exists. Preserve it; "
                "do not overwrite it after seeing any result."
            )
        print(f"Identical preregistration already exists: {PREREGISTRATION_PATH}")
        print(f"Registered script SHA-256: {script_hash}")
        return

    PREREGISTRATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 21 protocol registered before any process data were read.")
    print(f"Reserved faults: {RESERVED_FAULTS}")
    print(f"Registered script SHA-256: {script_hash}")
    print(f"Saved preregistration: {PREREGISTRATION_PATH}")


def verify_preregistration() -> dict:
    require_file(
        PREREGISTRATION_PATH,
        "run this same script once with --register",
    )
    registration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    current_hash = sha256_file(Path(__file__))
    expected = protocol_payload(current_hash)
    failures = []
    if registration.get("script_sha256") != current_hash:
        failures.append("script SHA-256 differs from the preregistered bytes")
    for key, value in expected.items():
        if registration.get(key) != value:
            failures.append(f"registered field changed: {key}")
    if failures:
        raise ValueError(f"Step-21 preregistration verification failed: {failures}")
    return registration


def load_dependencies():
    step18_path = SCRIPT_DIR / "18_faults9_15_prespecified_generalization.py"
    step19_path = SCRIPT_DIR / "19_multievidence_development_audit.py"
    if sha256_file(require_file(step18_path, "place the frozen Step-18 script")) != EXPECTED_STEP18_SHA256:
        raise ValueError("Step-18 dependency hash differs from the frozen version")
    if sha256_file(require_file(step19_path, "place the frozen Step-19 script")) != EXPECTED_STEP19_SHA256:
        raise ValueError("Step-19 dependency hash differs from the frozen version")
    step18 = load_module(step18_path, "step18_for_reserved_multifault")
    step19 = load_module(step19_path, "step19_for_reserved_multifault")

    fixed_checks = {
        "Step19 families": step19.FAMILY_KEYS
        == [
            "signed_level",
            "absolute_deviation",
            "squared_deviation",
            "absolute_increment",
            "lag1_product",
        ],
        "Step19 candidate count": len(step19.FAMILY_KEYS) * 52 == 260,
        "Step19 bootstraps": int(step19.BOOTSTRAP_REPLICATES) == 2000,
        "Step19 permutations": int(step19.PERMUTATION_REPLICATES) == 2000,
        "Step19 global FDR": float(step19.GLOBAL_FDR_ALPHA) == 0.05,
        "Step19 FWER": float(step19.FWER_ALPHA) == 0.05,
        "Step19 Top-K": int(step19.TOP_K) == TOP_K,
        "Step19 screen": int(step19.FAULT_START_SAMPLE) == FAULT_START_SAMPLE
        and int(step19.SCREEN_END_SAMPLE) == SCREEN_END_SAMPLE,
    }
    failed = [name for name, passed in fixed_checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen dependency checks failed: {failed}")
    return step18, step19


def validate_variable_order(variable_cols: list[str]) -> None:
    expected = [
        *[f"xmeas_{index}" for index in range(1, 42)],
        *[f"xmv_{index}" for index in range(1, 12)],
    ]
    if variable_cols != expected:
        raise ValueError("The expected 52-variable order has changed")


def build_fault_protocol(
    family_summary: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fault_id in RESERVED_FAULTS:
        summary = family_summary[family_summary["fault_id"] == fault_id]
        recommended = summary[summary["recommended_by_fixed_development_rule"]]
        if len(recommended) == 0:
            rows.append(
                {
                    "fault_id": fault_id,
                    "branch_status": "abstain_no_eligible_family",
                    "family_key": "none",
                    "family_label": "None",
                    "n_primary_stable_in_family": 0,
                    "n_strict_fwer_stable_in_family": 0,
                    "selected_count": 0,
                    "selected_variables": "",
                    "orientation_signs": "",
                    "minimum_selected_bootstrap_frequency": np.nan,
                    "all_selected_strict_fwer": False,
                }
            )
            continue
        if len(recommended) != 1:
            raise ValueError(f"Fault {fault_id}: recommendation is not unique")
        recommendation = recommended.iloc[0]
        family_key = str(recommendation["family_key"])
        current = selected[
            (selected["fault_id"] == fault_id)
            & (selected["family_key"].astype(str) == family_key)
            & selected["recommended_by_fixed_development_rule"]
        ].sort_values("selection_rank")
        if len(current) != TOP_K:
            raise ValueError(f"Fault {fault_id}: recommended branch is not Top-{TOP_K}")
        if not current["primary_stable_global_fdr"].all():
            raise ValueError(f"Fault {fault_id}: selected feature failed the primary gate")
        rows.append(
            {
                "fault_id": fault_id,
                "branch_status": "estimable_frozen_after_step21",
                "family_key": family_key,
                "family_label": str(recommendation["family_label"]),
                "n_primary_stable_in_family": int(
                    recommendation["n_primary_stable_global_fdr"]
                ),
                "n_strict_fwer_stable_in_family": int(
                    recommendation["n_strict_stable_max_t_fwer"]
                ),
                "selected_count": int(len(current)),
                "selected_variables": ";".join(current["variable"].astype(str)),
                "orientation_signs": ";".join(
                    current["orientation_sign"].astype(int).astype(str)
                ),
                "minimum_selected_bootstrap_frequency": float(
                    current["bootstrap_top5_selection_frequency"].min()
                ),
                "all_selected_strict_fwer": bool(
                    current["strict_stable_max_t_fwer"].all()
                ),
            }
        )
    protocol = pd.DataFrame(rows)
    if protocol["fault_id"].tolist() != RESERVED_FAULTS:
        raise ValueError("Reserved-fault protocol order changed")
    return protocol


def plot_family_applicability(
    family_summary: pd.DataFrame,
    protocol: pd.DataFrame,
) -> list[Path]:
    labels = [
        "Signed\nlevel",
        "Absolute\ndeviation",
        "Squared\ndeviation",
        "Absolute\nincrement",
        "Lag-1\nproduct",
    ]
    family_keys = [
        "signed_level",
        "absolute_deviation",
        "squared_deviation",
        "absolute_increment",
        "lag1_product",
    ]
    pivot = family_summary.pivot(
        index="fault_id",
        columns="family_key",
        values="n_primary_stable_global_fdr",
    ).reindex(index=RESERVED_FAULTS, columns=family_keys)
    if pivot.isna().any().any():
        raise ValueError("Incomplete fault/family applicability matrix")

    matrix = pivot.to_numpy(dtype=float)
    figure_height = max(7.5, 0.43 * len(RESERVED_FAULTS) + 2.5)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticks(np.arange(len(RESERVED_FAULTS)))
    axis.set_yticklabels([f"Fault {fault}" for fault in RESERVED_FAULTS])
    axis.set_xlabel("Frozen residual-evidence family")
    axis.set_ylabel("Reserved TEP fault")
    axis.set_title(
        "Reserved-fault development audit: primary-gate stable residual counts"
    )

    protocol_by_fault = protocol.set_index("fault_id")
    threshold = max(float(np.nanmax(matrix)) * 0.55, 1.0)
    for row, fault_id in enumerate(RESERVED_FAULTS):
        chosen = str(protocol_by_fault.loc[fault_id, "family_key"])
        for column, family_key in enumerate(family_keys):
            count = int(matrix[row, column])
            marker = "*" if family_key == chosen else ""
            color = "white" if count >= threshold else "black"
            axis.text(
                column,
                row,
                f"{count}{marker}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold" if marker else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Number of stable residual variables")
    figure.text(
        0.5,
        0.012,
        "* family chosen by the preregistered development rule; no star denotes abstention. "
        "No final-set performance is shown.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    png = FIGURE_DIR / "fig21a_reserved_multifault_development_audit.png"
    pdf = FIGURE_DIR / "fig21a_reserved_multifault_development_audit.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def run_development_audit() -> None:
    registration = verify_preregistration()
    step18, step19 = load_dependencies()
    step11 = step19.load_step11_module()
    variable_cols = step11.load_variable_columns()
    validate_variable_order(variable_cols)

    print("=" * 92)
    print("Step 21: Preregistered reserved-fault development-only audit")
    print("=" * 92)
    print(f"Registered script SHA-256: {registration['script_sha256']}")
    print(f"Reserved faults: {RESERVED_FAULTS}")
    print("No performance metric and no final/companion branch is evaluated.")

    normal_train = pd.read_pickle(
        require_file(NORMAL_TRAIN_PATH, "run Step 02")
    )
    print(f"Loading PCA training data: {normal_train.shape}")
    model = step11.fit_pca_model(normal_train, variable_cols)
    del normal_train
    gc.collect()

    normal_validation = pd.read_pickle(
        require_file(NORMAL_VALIDATION_PATH, "run Step 02")
    )
    normal_validation = step19.validate_run_frame(
        normal_validation,
        variable_cols,
        "normal_validation_runs_1_250",
        expected_fault=None,
    )
    print(f"Computing normal-validation PCA residuals: {normal_validation.shape}")
    normal_residuals, residual_cols = step11.compute_residual_frame(
        normal_validation, variable_cols, model
    )
    normal_center = normal_residuals[residual_cols].mean(axis=0).to_numpy(
        dtype=float
    )
    normal_tensor = step19.extract_base_tensor(
        normal_residuals, residual_cols, "normal_validation"
    )
    normal_features = step19.build_feature_run_means(
        normal_tensor, normal_center
    )
    del normal_validation, normal_residuals, normal_tensor
    gc.collect()

    require_file(RAW_TESTING_PATH, "place TEP_Faulty_Testing.RData in data/raw")
    testing, testing_object, testing_names = step18.read_rdata_frame(
        RAW_TESTING_PATH, "faulty_testing"
    )
    testing_shape = tuple(int(value) for value in testing.shape)
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in testing.columns]
    if missing:
        raise ValueError(f"Faulty-Testing object is missing columns: {missing}")
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    run_numbers = pd.to_numeric(testing["simulationRun"], errors="raise")
    available_faults = sorted(
        int(value) for value in pd.unique(fault_numbers) if int(value) != 0
    )
    missing_faults = sorted(set(EXPECTED_ALL_FAULTS) - set(available_faults))
    if missing_faults:
        raise ValueError(f"Testing RData is missing expected faults: {missing_faults}")

    all_metrics = []
    all_selected = []
    all_family_summaries = []
    audit_rows = []
    recommendations = {}

    keep = required
    for fault_id in RESERVED_FAULTS:
        print("\n" + "=" * 92)
        print(f"Fault {fault_id}: reserved development runs 1--250 only")
        print("=" * 92)
        development_mask = (fault_numbers == fault_id) & run_numbers.between(1, 250)
        development = testing.loc[development_mask, keep].copy()
        development, audit = step18.validate_fault_frame(
            development,
            variable_cols,
            fault_id,
            EXPECTED_DEVELOPMENT_RUNS,
            EXPECTED_SAMPLE_MIN,
            EXPECTED_SAMPLE_MAX,
            f"fault{fault_id}_reserved_development_runs_1_250",
        )
        audit["source_container"] = str(RAW_TESTING_PATH.relative_to(PROJECT_ROOT))
        audit["final_rows_subset_or_analyzed"] = False
        audit_rows.append(audit)

        development_residuals, current_cols = step11.compute_residual_frame(
            development, variable_cols, model
        )
        if current_cols != residual_cols:
            raise ValueError(f"Fault {fault_id}: residual column order changed")
        development_tensor = step19.extract_base_tensor(
            development_residuals,
            residual_cols,
            f"fault{fault_id}_reserved_development",
        )
        fault_features = step19.build_feature_run_means(
            development_tensor, normal_center
        )
        del development, development_residuals, development_tensor
        gc.collect()

        metrics, selected, summary, recommended_family = step19.analyze_fault(
            fault_id,
            normal_features,
            fault_features,
            variable_cols,
        )
        all_metrics.append(metrics)
        all_selected.append(selected)
        all_family_summaries.append(summary)
        recommendations[str(fault_id)] = recommended_family
        print("\n  Development family summary:")
        print(
            summary[
                [
                    "family_label",
                    "n_primary_stable_global_fdr",
                    "n_strict_stable_max_t_fwer",
                    "family_eligible_primary_gate",
                    "selected_variables",
                ]
            ].to_string(index=False)
        )
        print(f"  Frozen-rule recommendation: {recommended_family}")
        del fault_features, metrics, selected, summary
        gc.collect()

    del fault_numbers, run_numbers, testing
    gc.collect()

    metrics = pd.concat(all_metrics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    family_summary = pd.concat(all_family_summaries, ignore_index=True)
    audit_table = pd.DataFrame(audit_rows)
    protocol = build_fault_protocol(family_summary, selected)

    output_paths = {
        "all_features": TABLE_DIR / "21_reserved_multifault_all_features.csv",
        "selected_features": TABLE_DIR / "21_reserved_multifault_selected_features.csv",
        "family_summary": TABLE_DIR / "21_reserved_multifault_family_summary.csv",
        "fault_protocol": TABLE_DIR / "21_reserved_multifault_frozen_protocol.csv",
        "data_audit": TABLE_DIR / "21_reserved_multifault_data_audit.csv",
    }
    metrics.to_csv(output_paths["all_features"], index=False, encoding="utf-8-sig")
    selected.to_csv(
        output_paths["selected_features"], index=False, encoding="utf-8-sig"
    )
    family_summary.to_csv(
        output_paths["family_summary"], index=False, encoding="utf-8-sig"
    )
    protocol.to_csv(
        output_paths["fault_protocol"], index=False, encoding="utf-8-sig"
    )
    audit_table.to_csv(
        output_paths["data_audit"], index=False, encoding="utf-8-sig"
    )
    figure_paths = plot_family_applicability(family_summary, protocol)

    estimable_faults = protocol.loc[
        protocol["branch_status"] == "estimable_frozen_after_step21", "fault_id"
    ].astype(int).tolist()
    abstained_faults = protocol.loc[
        protocol["branch_status"] != "estimable_frozen_after_step21", "fault_id"
    ].astype(int).tolist()
    config = {
        "script": "scripts/21_reserved_multifault_development_audit.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "preregistration": str(PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
        "preregistration_registered_at_utc": registration["registered_at_utc"],
        "preregistration_verified": True,
        "development_only": True,
        "performance_evaluated": False,
        "normal_evaluation_loaded": False,
        "faulty_training_companion_loaded": False,
        "testing_final_rows_subset_or_analyzed": False,
        "raw_testing_container_materialized": True,
        "raw_testing_container_caveat": registration["raw_container_caveat"],
        "raw_testing_file": str(RAW_TESTING_PATH.relative_to(PROJECT_ROOT)),
        "raw_testing_size_bytes": int(RAW_TESTING_PATH.stat().st_size),
        "raw_testing_object": testing_object,
        "raw_testing_object_names": testing_names,
        "raw_testing_complete_shape": testing_shape,
        "reserved_faults": RESERVED_FAULTS,
        "excluded_development_faults": registration["excluded_development_faults"],
        "development_runs": [1, 250],
        "reserved_final_runs": [251, 500],
        "screening_samples": [FAULT_START_SAMPLE, SCREEN_END_SAMPLE],
        "candidate_families": step19.FAMILY_SPECS,
        "candidate_tests_per_fault": len(step19.FAMILY_SPECS) * len(variable_cols),
        "bootstrap_replicates": int(step19.BOOTSTRAP_REPLICATES),
        "permutation_replicates": int(step19.PERMUTATION_REPLICATES),
        "primary_feature_gate": registration["primary_feature_gate"],
        "strict_diagnostic": registration["strict_diagnostic"],
        "family_eligibility": registration["family_eligibility"],
        "family_choice_rule": registration["family_choice_rule"],
        "top_k": TOP_K,
        "random_seed": int(step19.RANDOM_SEED),
        "recommended_family_by_fault": recommendations,
        "estimable_faults": estimable_faults,
        "abstained_faults": abstained_faults,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "dependency_hashes": registration["dependency_hashes"],
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
        "next_step_constraint": (
            "Step 22 must verify this script/config and evaluate each frozen "
            "estimable branch on runs 251--500 without changing family, Top-5, "
            "orientations, K, CUSUM k, or the normal-validation control-limit rule"
        ),
    }
    config_path = CONFIG_DIR / "21_reserved_multifault_development_audit.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 92)
    print("Step 21 summary: frozen per-fault development protocol")
    print("=" * 92)
    print(protocol.to_string(index=False))
    print(f"\nEstimable faults: {estimable_faults}")
    print(f"Abstained faults: {abstained_faults}")
    print("No final-set, normal-evaluation, or companion performance was computed.")
    print(f"Saved configuration: {config_path}")
    print("Step 21 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--register",
        action="store_true",
        help="write the immutable preregistration without reading process data",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="verify the preregistration and execute the development-only audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.register:
        register_protocol()
    else:
        run_development_audit()


if __name__ == "__main__":
    main()
