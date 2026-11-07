"""
Step 18: Prespecified generalization to difficult TEP Faults 9 and 15.

Scientific question
-------------------
Does the pairing-free local signed-residual framework developed for Fault 3
generalize, under the same rules, to the other two conventionally difficult
TEP faults (9 and 15)?

Protocol fixed before seeing Fault-9/15 results
-----------------------------------------------
* The PCA model is fitted only on FaultFree Training, retaining >=95% variance.
* Faulty Testing runs 1--250 are development runs; 251--500 are final runs.
* Screening uses independent run means over samples 161--320, a 2,000-repeat
  run-level bootstrap, a two-sided 95% CI excluding zero, absolute Hedges' g,
  and at most five residuals.
* The proposed method keeps K=5 (when five stable residuals exist), k=0.05,
  and a 95th-percentile normal-validation run-maximum control limit.
* Comparator CUSUM k values use the same prespecified Step-15 grid and are
  selected only on the target-fault development runs.
* Faulty Training is never used for model selection or tuning.  It is applied
  only after each target-fault method is frozen.
* All results for both faults are saved, including failures.

Companion-set interpretation
----------------------------
Faulty Training uses 500-sample runs with fault onset at sample 21; Faulty
Testing uses 960-sample runs with onset at 161.  The script therefore also
evaluates original final-test samples 141--640 after relabelling them 1--500
and resetting each chart.  This gives the same 20 pre-fault and 480 post-fault
samples as the companion set.  The script audits whether companion pre-fault
rows are identical to same-numbered FaultFree Training rows; these results are
not described as independent external-plant validation.

Place in:
    TEP_Fault3_Project/scripts/18_faults9_15_prespecified_generalization.py

Run from project root:
    python scripts/18_faults9_15_prespecified_generalization.py

Prerequisites
-------------
* Steps 02, 11, and 15.
* ``TEP_Faulty_Testing.RData`` and ``TEP_Faulty_Training.RData`` in data/raw.

The first run extracts and caches only Faults 9 and 15 as pickle files under
data/processed.  This avoids rereading the two large RData files on a rerun.
Do not attach those large cache files when returning the Step-18 results.
"""

from pathlib import Path
import gc
import importlib.util
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET_FAULTS = [9, 15]
FAULT_TESTING_START = 161
FAULT_TRAINING_START = 21
SAMPLING_INTERVAL_MIN = 3
SCREEN_NORMAL_START = 161
SCREEN_NORMAL_END = 320
SCREEN_COMPANION_START = 21
SCREEN_COMPANION_END = 180
PRIMARY_TOP_K = 5
PRIMARY_CUSUM_K = 0.05
RUN_LEVEL_PERCENTILE = 95
BOOTSTRAP_REPLICATES_EXPECTED = 2000
COMPETITOR_K_GRID_EXPECTED = [0.01, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]

MATCHED_ORIGINAL_FIRST_SAMPLE = 141
MATCHED_ORIGINAL_LAST_SAMPLE = 640
MATCHED_SAMPLE_OFFSET = 140

RAW_TESTING = RAW_DIR / "TEP_Faulty_Testing.RData"
RAW_TRAINING = RAW_DIR / "TEP_Faulty_Training.RData"

CACHE_PATHS = {
    fault_id: {
        "development": PROCESSED_DIR
        / f"fault{fault_id}_development_runs_1_250.pkl",
        "final": PROCESSED_DIR
        / f"fault{fault_id}_final_test_runs_251_500.pkl",
        "companion": PROCESSED_DIR
        / f"fault{fault_id}_companion_training_runs_1_500.pkl",
    }
    for fault_id in TARGET_FAULTS
}

METHOD_SPECS = [
    {
        "method_key": "pca_q_raw",
        "method": "PCA Q/SPE (raw)",
        "method_short": "Q raw",
        "input_key": "pca_q",
        "sequential": False,
        "tune_k": False,
    },
    {
        "method_key": "pca_t2_raw",
        "method": "PCA Hotelling T2 (raw)",
        "method_short": "T2 raw",
        "input_key": "pca_t2",
        "sequential": False,
        "tune_k": False,
    },
    {
        "method_key": "pca_joint_raw",
        "method": "PCA joint max-score (raw)",
        "method_short": "Joint raw",
        "input_key": "pca_joint",
        "sequential": False,
        "tune_k": False,
    },
    {
        "method_key": "pca_q_cusum",
        "method": "PCA Q/SPE-CUSUM",
        "method_short": "Q-CUSUM",
        "input_key": "pca_q",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "pca_t2_cusum",
        "method": "PCA T2-CUSUM",
        "method_short": "T2-CUSUM",
        "input_key": "pca_t2",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "pca_joint_cusum",
        "method": "PCA joint max-score CUSUM",
        "method_short": "Joint CUSUM",
        "input_key": "pca_joint",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "all52_signed_cusum",
        "method": "All-52 unpaired signed residual CUSUM",
        "method_short": "All-52 signed",
        "input_key": "all52_signed",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "lda_residual_cusum",
        "method": "Shrinkage-LDA residual CUSUM",
        "method_short": "LDA residual",
        "input_key": "lda_residual",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "top1_signed_cusum",
        "method": "Unpaired signed Top-1 residual CUSUM",
        "method_short": "Top-1 signed",
        "input_key": "top1_signed",
        "sequential": True,
        "tune_k": True,
    },
    {
        "method_key": "proposed_top5_cusum",
        "method": "Proposed unpaired signed Top-5 residual CUSUM",
        "method_short": "Proposed Top-5",
        "input_key": "proposed_top5",
        "sequential": True,
        "tune_k": False,
        "fixed_k": PRIMARY_CUSUM_K,
    },
]


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_module(path, name):
    require_file(path, f"place {path.name} in the scripts directory")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    step15 = load_module(
        SCRIPT_DIR / "15_fair_sequential_baselines.py", "step15_generalization"
    )
    required15 = [
        "load_step11_module",
        "compute_pca_outputs",
        "derive_retained_eigenvalues",
        "fit_global_transforms",
        "transform_global_inputs",
        "fit_lda_score",
        "apply_lda_score",
        "fit_scalar_standardizer",
        "apply_scalar_standardizer",
        "signed_local_input",
        "compute_cusum",
        "raw_monitoring_statistic",
        "evaluate_raw_method",
        "evaluate_cusum_method",
        "tune_cusum_k_on_development",
    ]
    missing = [name for name in required15 if not hasattr(step15, name)]
    if missing:
        raise AttributeError(f"Step 15 is missing functions: {missing}")
    step11 = step15.load_step11_module()
    if int(step11.BOOTSTRAP_REPLICATES) != BOOTSTRAP_REPLICATES_EXPECTED:
        raise ValueError(
            "Step 11 bootstrap replicate count has changed from the fixed 2000"
        )
    if list(step15.CUSUM_K_GRID) != COMPETITOR_K_GRID_EXPECTED:
        raise ValueError("Step 15 competitor k grid has changed")
    return step11, step15


def normalize_integer_column(frame, column):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    rounded = np.rint(values)
    if np.any(~np.isfinite(values)) or not np.allclose(
        values, rounded, rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"Column {column} is not finite integer-valued")
    frame[column] = rounded.astype(np.int64)


def variable_columns_from_table():
    path = require_file(TABLE_DIR / "02_variable_list.csv", "run Step 02")
    table = pd.read_csv(path)
    columns = table["variable"].astype(str).tolist()
    if columns != [
        *[f"xmeas_{i}" for i in range(1, 42)],
        *[f"xmv_{i}" for i in range(1, 12)],
    ]:
        raise ValueError("Step-02 variable order is not the expected 52 variables")
    return columns


def read_rdata_frame(path, preferred_object):
    require_file(path, f"place {path.name} in data/raw")
    print(f"Reading {path.name}; this can take several minutes...")
    objects = pyreadr.read_r(str(path))
    if not objects:
        raise ValueError(f"No object found in {path.name}")
    names = list(objects.keys())
    if preferred_object in objects:
        name = preferred_object
    elif len(objects) == 1:
        name = names[0]
    else:
        raise ValueError(f"Cannot choose object from {path.name}: {names}")
    frame = objects[name]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Object {name} in {path.name} is not a DataFrame")
    return frame, name, names


def validate_fault_frame(
    frame,
    variable_cols,
    fault_id,
    expected_runs,
    expected_sample_min,
    expected_sample_max,
    dataset_name,
):
    required = ["faultNumber", "simulationRun", "sample", *variable_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {missing}")
    frame = frame[required].copy()
    for column in ["faultNumber", "simulationRun", "sample"]:
        normalize_integer_column(frame, column)
    frame = frame.sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    if set(frame["faultNumber"].unique()) != {int(fault_id)}:
        raise ValueError(f"{dataset_name} contains an unexpected fault number")
    actual_runs = np.sort(frame["simulationRun"].unique())
    if not np.array_equal(actual_runs, np.asarray(expected_runs, dtype=np.int64)):
        raise ValueError(
            f"{dataset_name} run IDs differ from the expected set: "
            f"count={len(actual_runs)}, min={actual_runs.min()}, max={actual_runs.max()}"
        )
    if int(frame["sample"].min()) != int(expected_sample_min) or int(
        frame["sample"].max()
    ) != int(expected_sample_max):
        raise ValueError(f"{dataset_name} has an unexpected sample range")
    expected_length = expected_sample_max - expected_sample_min + 1
    counts = frame.groupby("simulationRun")["sample"].nunique()
    if not (counts == expected_length).all():
        raise ValueError(f"{dataset_name} contains incomplete runs")
    duplicates = int(
        frame.duplicated(["faultNumber", "simulationRun", "sample"]).sum()
    )
    if duplicates:
        raise ValueError(f"{dataset_name} contains duplicate identifier rows")
    values = frame[variable_cols].to_numpy(dtype=float, copy=False)
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise ValueError(f"{dataset_name} has {nonfinite} non-finite values")
    audit = {
        "dataset": dataset_name,
        "fault_id": int(fault_id),
        "rows": int(len(frame)),
        "runs": int(len(actual_runs)),
        "run_min": int(actual_runs.min()),
        "run_max": int(actual_runs.max()),
        "sample_min": int(frame["sample"].min()),
        "sample_max": int(frame["sample"].max()),
        "rows_per_run_min": int(counts.min()),
        "rows_per_run_max": int(counts.max()),
        "duplicate_identifier_rows": duplicates,
        "nonfinite_process_values": nonfinite,
    }
    return frame, audit


def all_cache_files_exist():
    return all(
        path.exists()
        for fault_paths in CACHE_PATHS.values()
        for path in fault_paths.values()
    )


def prepare_fault_caches(variable_cols):
    if all_cache_files_exist():
        print("All Fault-9/15 processed caches already exist; raw extraction skipped.")
        return

    keep = ["faultNumber", "simulationRun", "sample", *variable_cols]
    testing, testing_object, testing_names = read_rdata_frame(
        RAW_TESTING, "faulty_testing"
    )
    testing_shape = tuple(int(value) for value in testing.shape)
    fault_numbers = pd.to_numeric(testing["faultNumber"], errors="raise")
    for fault_id in TARGET_FAULTS:
        subset = testing.loc[fault_numbers == fault_id, keep].copy()
        subset, _ = validate_fault_frame(
            subset,
            variable_cols,
            fault_id,
            np.arange(1, 501),
            1,
            960,
            f"fault{fault_id}_testing_all_runs",
        )
        development = subset[subset["simulationRun"].between(1, 250)].copy()
        final = subset[subset["simulationRun"].between(251, 500)].copy()
        development.to_pickle(CACHE_PATHS[fault_id]["development"])
        final.to_pickle(CACHE_PATHS[fault_id]["final"])
        print(
            f"Saved Fault {fault_id} testing caches: "
            f"development={development.shape}, final={final.shape}"
        )
        del subset, development, final
        gc.collect()
    del fault_numbers, testing
    gc.collect()

    training, training_object, training_names = read_rdata_frame(
        RAW_TRAINING, "faulty_training"
    )
    training_shape = tuple(int(value) for value in training.shape)
    fault_numbers = pd.to_numeric(training["faultNumber"], errors="raise")
    for fault_id in TARGET_FAULTS:
        subset = training.loc[fault_numbers == fault_id, keep].copy()
        subset, _ = validate_fault_frame(
            subset,
            variable_cols,
            fault_id,
            np.arange(1, 501),
            1,
            500,
            f"fault{fault_id}_companion_all_runs",
        )
        subset.to_pickle(CACHE_PATHS[fault_id]["companion"])
        print(f"Saved Fault {fault_id} companion cache: {subset.shape}")
        del subset
        gc.collect()
    del fault_numbers, training
    gc.collect()

    extraction_manifest = {
        "faulty_testing_file": str(RAW_TESTING.relative_to(PROJECT_ROOT)),
        "faulty_testing_size_bytes": int(RAW_TESTING.stat().st_size),
        "faulty_testing_object": testing_object,
        "faulty_testing_object_names": testing_names,
        "faulty_testing_complete_shape": testing_shape,
        "faulty_training_file": str(RAW_TRAINING.relative_to(PROJECT_ROOT)),
        "faulty_training_size_bytes": int(RAW_TRAINING.stat().st_size),
        "faulty_training_object": training_object,
        "faulty_training_object_names": training_names,
        "faulty_training_complete_shape": training_shape,
        "target_faults": TARGET_FAULTS,
        "cache_files": {
            str(fault_id): {
                role: str(path.relative_to(PROJECT_ROOT))
                for role, path in paths.items()
            }
            for fault_id, paths in CACHE_PATHS.items()
        },
    }
    manifest_path = CONFIG_DIR / "faults9_15_raw_extraction_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(extraction_manifest, handle, indent=2, ensure_ascii=False)


def load_fault_cache(fault_id, role, variable_cols):
    path = require_file(
        CACHE_PATHS[fault_id][role], "rerun the Step-18 cache extraction"
    )
    frame = pd.read_pickle(path)
    if role == "development":
        runs, sample_max = np.arange(1, 251), 960
    elif role == "final":
        runs, sample_max = np.arange(251, 501), 960
    else:
        runs, sample_max = np.arange(1, 501), 500
    frame, audit = validate_fault_frame(
        frame,
        variable_cols,
        fault_id,
        runs,
        1,
        sample_max,
        f"fault{fault_id}_{role}",
    )
    audit["cache_path"] = str(path.relative_to(PROJECT_ROOT))
    return frame, audit


def extract_normal_training_pre_reference(normal_train, variable_cols):
    reference = normal_train[
        normal_train["sample"].between(1, FAULT_TRAINING_START - 1)
    ][["simulationRun", "sample", *variable_cols]].copy()
    for column in ["simulationRun", "sample"]:
        normalize_integer_column(reference, column)
    return reference.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )


def prefault_identity_audit(reference, companion, variable_cols):
    current = companion[companion["sample"] < FAULT_TRAINING_START][
        ["simulationRun", "sample", *variable_cols]
    ].copy()
    current = current.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )
    if not np.array_equal(
        reference[["simulationRun", "sample"]].to_numpy(),
        current[["simulationRun", "sample"]].to_numpy(),
    ):
        raise ValueError("Companion and PCA-training pre-fault IDs do not align")
    normal_values = reference[variable_cols].to_numpy(dtype=float)
    current_values = current[variable_cols].to_numpy(dtype=float)
    absolute = np.abs(normal_values - current_values)
    return {
        "prefault_rows_compared": int(len(current)),
        "prefault_exact_cell_fraction": float(
            np.mean(normal_values == current_values)
        ),
        "prefault_exact_row_fraction": float(
            np.mean(np.all(normal_values == current_values, axis=1))
        ),
        "prefault_max_absolute_difference": float(absolute.max()),
        "prefault_identical_to_pca_training": bool(
            np.array_equal(normal_values, current_values)
        ),
    }


def make_matched_original_segment(final_frame):
    matched = final_frame[
        final_frame["sample"].between(
            MATCHED_ORIGINAL_FIRST_SAMPLE, MATCHED_ORIGINAL_LAST_SAMPLE
        )
    ].copy()
    matched["sample"] = matched["sample"].astype(int) - MATCHED_SAMPLE_OFFSET
    counts = matched.groupby("simulationRun")["sample"].nunique()
    if len(counts) != 250 or not (counts == 500).all():
        raise ValueError("Original onset/horizon matching produced incomplete runs")
    return matched.sort_values(["simulationRun", "sample"]).reset_index(drop=True)


def stable_rank(metrics):
    return metrics[metrics["direction_stable_95ci"]].sort_values(
        [
            "abs_hedges_g",
            "bootstrap_direction_confidence",
            "bootstrap_top5_selection_frequency",
            "variable",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def run_primary_screens(
    step11,
    fault_id,
    variable_cols,
    residual_cols,
    normal_validation_res,
    development_res,
    companion_res,
):
    normal_run_means = step11.calculate_run_means(
        normal_validation_res,
        residual_cols,
        SCREEN_NORMAL_START,
        SCREEN_NORMAL_END,
    )
    development_means = step11.calculate_run_means(
        development_res,
        residual_cols,
        SCREEN_NORMAL_START,
        SCREEN_NORMAL_END,
    )
    dev_metrics, _, dev_selected = step11.screen_unpaired_runlevel(
        normal_run_means,
        development_means,
        variable_cols,
        f"fault{fault_id}_testing_development_primary",
        SCREEN_NORMAL_START,
        SCREEN_NORMAL_END,
        step11.RANDOM_SEED + 1000 * fault_id,
    )
    dev_metrics.insert(0, "fault_id", fault_id)
    dev_metrics.insert(1, "screen_source", "faulty_testing_development")
    dev_selected.insert(0, "fault_id", fault_id)
    dev_selected.insert(1, "screen_source", "faulty_testing_development")

    companion_means = step11.calculate_run_means(
        companion_res,
        residual_cols,
        SCREEN_COMPANION_START,
        SCREEN_COMPANION_END,
    )
    companion_metrics, _, companion_selected = step11.screen_unpaired_runlevel(
        normal_run_means,
        companion_means,
        variable_cols,
        f"fault{fault_id}_companion_postfreeze_diagnostic",
        SCREEN_COMPANION_START,
        SCREEN_COMPANION_END,
        step11.RANDOM_SEED + 1000 * fault_id + 1,
    )
    companion_metrics.insert(0, "fault_id", fault_id)
    companion_metrics.insert(1, "screen_source", "faulty_training_postfreeze")
    companion_selected.insert(0, "fault_id", fault_id)
    companion_selected.insert(1, "screen_source", "faulty_training_postfreeze")

    dev_lookup = dev_metrics.set_index("variable")
    companion_lookup = companion_metrics.set_index("variable")
    companion_top5 = set(companion_selected["variable"].astype(str))
    transfer_rows = []
    for rank, variable in enumerate(dev_selected["variable"].astype(str), 1):
        dev_row = dev_lookup.loc[variable]
        companion_row = companion_lookup.loc[variable]
        transfer_rows.append(
            {
                "fault_id": fault_id,
                "development_selection_rank": rank,
                "variable": variable,
                "development_sign": int(
                    dev_row["sign_for_positive_evidence"]
                ),
                "development_hedges_g": float(dev_row["hedges_g"]),
                "development_direction_stable": bool(
                    dev_row["direction_stable_95ci"]
                ),
                "companion_sign": int(
                    companion_row["sign_for_positive_evidence"]
                ),
                "companion_hedges_g": float(companion_row["hedges_g"]),
                "companion_direction_stable": bool(
                    companion_row["direction_stable_95ci"]
                ),
                "sign_agreement": bool(
                    int(dev_row["sign_for_positive_evidence"])
                    == int(companion_row["sign_for_positive_evidence"])
                ),
                "selected_in_companion_top5": variable in companion_top5,
            }
        )
    transfer = pd.DataFrame(transfer_rows)
    screen_summary = {
        "fault_id": fault_id,
        "development_stable_variable_count": int(
            dev_metrics["direction_stable_95ci"].sum()
        ),
        "development_selected_count": int(len(dev_selected)),
        "development_selected_variables": ";".join(
            dev_selected["variable"].astype(str)
        ),
        "development_selected_signs": ";".join(
            dev_selected["sign_for_positive_evidence"].astype(int).astype(str)
        ),
        "protocol_full_top5_available": bool(len(dev_selected) == PRIMARY_TOP_K),
        "companion_stable_variable_count": int(
            companion_metrics["direction_stable_95ci"].sum()
        ),
        "companion_postfreeze_top5": ";".join(
            companion_selected["variable"].astype(str)
        ),
        "development_companion_top5_overlap": int(
            len(
                set(dev_selected["variable"].astype(str))
                & set(companion_selected["variable"].astype(str))
            )
        ),
        "development_selected_sign_agreement_count": int(
            transfer["sign_agreement"].sum()
        )
        if len(transfer)
        else 0,
        "development_selected_companion_stable_count": int(
            transfer["companion_direction_stable"].sum()
        )
        if len(transfer)
        else 0,
    }
    all_metrics = pd.concat([dev_metrics, companion_metrics], ignore_index=True)
    all_selected = pd.concat(
        [dev_selected, companion_selected], ignore_index=True
    )
    return (
        dev_metrics,
        dev_selected,
        all_metrics,
        all_selected,
        transfer,
        screen_summary,
    )


def build_input_sets(
    step11,
    step15,
    variable_cols,
    residual_cols,
    residuals,
    statistics,
    development_screen_metrics,
    development_selected,
):
    global_parameters = step15.fit_global_transforms(
        statistics["normal_validation"]
    )
    global_inputs = {
        name: step15.transform_global_inputs(frame, global_parameters)
        for name, frame in statistics.items()
    }
    inputs = {
        "pca_q": {name: value["q"] for name, value in global_inputs.items()},
        "pca_t2": {name: value["t2"] for name, value in global_inputs.items()},
        "pca_joint": {
            name: value["joint"] for name, value in global_inputs.items()
        },
    }

    stable = stable_rank(development_screen_metrics)
    top1 = stable.head(1).copy()
    all52 = development_screen_metrics[
        ["variable", "residual_variable", "sign_for_positive_evidence"]
    ].copy()
    local_tables = {"all52_signed": all52}
    if len(top1):
        local_tables["top1_signed"] = top1
    if len(development_selected):
        local_tables["proposed_top5"] = development_selected
    for key, selected in local_tables.items():
        scaler = step11.fit_signed_scaler(
            residuals["normal_validation"], selected
        )
        inputs[key] = {
            name: step15.signed_local_input(step11, frame, scaler)
            for name, frame in residuals.items()
        }

    weights, weight_table, lda_metadata = step15.fit_lda_score(
        step11,
        residuals["normal_validation"],
        residuals["fault3_development"],
        residual_cols,
        variable_cols,
    )
    lda_raw = {
        name: step15.apply_lda_score(frame, residual_cols, weights)
        for name, frame in residuals.items()
    }
    lda_scaler = step15.fit_scalar_standardizer(
        lda_raw["normal_validation"]
    )
    inputs["lda_residual"] = {
        name: step15.apply_scalar_standardizer(frame, lda_scaler)
        for name, frame in lda_raw.items()
    }
    return inputs, weight_table, lda_metadata, global_parameters


def evaluate_generic_fault(step11, statistic, limit, fault_start_sample):
    df = statistic.sort_values(["simulationRun", "sample"]).copy()
    df["alarm"] = df["monitoring_statistic"] > float(limit)
    pre = df[df["sample"] < fault_start_sample]
    post = df[df["sample"] >= fault_start_sample]
    if pre.empty or post.empty:
        raise ValueError("Generic fault evaluation requires pre/post samples")
    runs = pd.DataFrame(
        {"simulationRun": sorted(df["simulationRun"].unique())}
    )
    pre_alarm = (
        pre.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="pre_fault_run_alarm")
    )
    post_alarm = (
        post.groupby("simulationRun")["alarm"]
        .any()
        .reset_index(name="post_fault_run_alarm")
    )
    first_alarm = (
        post[post["alarm"]]
        .groupby("simulationRun")["sample"]
        .min()
        .reset_index(name="first_alarm_sample")
    )
    runs = runs.merge(pre_alarm, on="simulationRun", how="left")
    runs = runs.merge(post_alarm, on="simulationRun", how="left")
    runs = runs.merge(first_alarm, on="simulationRun", how="left")
    for column in ["pre_fault_run_alarm", "post_fault_run_alarm"]:
        runs[column] = runs[column].fillna(False).astype(bool)
    runs["detected"] = runs["post_fault_run_alarm"]
    runs["alarm_delay_min"] = (
        runs["first_alarm_sample"] - fault_start_sample
    ) * SAMPLING_INTERVAL_MIN
    runs["right_censored"] = ~runs["detected"]

    n_runs = int(len(runs))
    pre_count = int(runs["pre_fault_run_alarm"].sum())
    detected = int(runs["detected"].sum())
    pre_low, pre_high = step11.clopper_pearson(pre_count, n_runs)
    det_low, det_high = step11.clopper_pearson(detected, n_runs)
    delays = runs.loc[runs["detected"], "alarm_delay_min"].dropna()
    metrics = {
        "fault_n_runs": n_runs,
        "fault_pre_fault_alarm_runs": pre_count,
        "fault_pre_fault_run_alarm_rate": pre_count / n_runs,
        "fault_pre_fault_run_alarm_ci_low": pre_low,
        "fault_pre_fault_run_alarm_ci_high": pre_high,
        "fault_pre_fault_point_alarm_rate": float(pre["alarm"].mean()),
        "detected_runs": detected,
        "right_censored_runs": n_runs - detected,
        "fault_run_level_detection_rate": detected / n_runs,
        "fault_run_level_detection_ci_low": det_low,
        "fault_run_level_detection_ci_high": det_high,
        "fault_post_fault_point_detection_rate": float(post["alarm"].mean()),
        "median_alarm_delay_min": float(delays.median())
        if len(delays)
        else np.nan,
        "mean_alarm_delay_min": float(delays.mean())
        if len(delays)
        else np.nan,
        "alarm_delay_q1_min": float(delays.quantile(0.25))
        if len(delays)
        else np.nan,
        "alarm_delay_q3_min": float(delays.quantile(0.75))
        if len(delays)
        else np.nan,
    }
    return metrics, runs


def normalize_step15_fault_metrics(metrics):
    mapping = {
        "fault_n_runs": "fault3_n_runs",
        "fault_pre_fault_alarm_runs": "fault3_pre_fault_alarm_runs",
        "fault_pre_fault_run_alarm_rate": "fault3_pre_fault_run_far",
        "fault_pre_fault_run_alarm_ci_low": "fault3_pre_fault_run_far_ci_low",
        "fault_pre_fault_run_alarm_ci_high": "fault3_pre_fault_run_far_ci_high",
        "fault_pre_fault_point_alarm_rate": "fault3_pre_fault_point_far",
        "detected_runs": "detected_runs",
        "fault_run_level_detection_rate": "fault3_run_level_detection_rate",
        "fault_run_level_detection_ci_low": (
            "fault3_run_level_detection_ci_low"
        ),
        "fault_run_level_detection_ci_high": (
            "fault3_run_level_detection_ci_high"
        ),
        "fault_post_fault_point_detection_rate": (
            "fault3_post_fault_point_detection_rate"
        ),
        "median_alarm_delay_min": "median_alarm_delay_min",
        "mean_alarm_delay_min": "mean_alarm_delay_min",
        "alarm_delay_q1_min": "alarm_delay_q1_min",
        "alarm_delay_q3_min": "alarm_delay_q3_min",
    }
    result = {output: metrics.get(source, np.nan) for output, source in mapping.items()}
    result["right_censored_runs"] = int(
        int(result["fault_n_runs"]) - int(result["detected_runs"])
    )
    return result


def monitoring_statistic(step15, input_frame, sequential, k):
    if sequential:
        return step15.compute_cusum(input_frame, float(k))
    return step15.raw_monitoring_statistic(input_frame)


def annotate_runs(frame, fault_id, method_key, method, dataset):
    output = frame.copy()
    output.insert(0, "dataset", dataset)
    output.insert(0, "method", method)
    output.insert(0, "method_key", method_key)
    output.insert(0, "fault_id", fault_id)
    return output


def evaluate_fault_methods(
    step11,
    step15,
    fault_id,
    inputs,
    selected_count,
):
    rows = []
    tuning_tables = []
    run_tables = []
    for spec in METHOD_SPECS:
        if spec["input_key"] not in inputs:
            rows.append(
                {
                    "fault_id": fault_id,
                    "method_key": spec["method_key"],
                    "method": spec["method"],
                    "method_short": spec["method_short"],
                    "dataset": "protocol_not_estimable",
                    "protocol_status": (
                        "no direction-stable residual available for local method"
                    ),
                    "n_selected_variables": selected_count
                    if spec["method_key"] == "proposed_top5_cusum"
                    else np.nan,
                }
            )
            continue
        method_inputs = inputs[spec["input_key"]]
        if not spec["sequential"]:
            k = np.nan
            metrics, _, normal_runs, final_runs = step15.evaluate_raw_method(
                step11, method_inputs
            )
            k_source = "not applicable"
        else:
            if spec["tune_k"]:
                k, tuning = step15.tune_cusum_k_on_development(
                    step11, spec["method_key"], method_inputs
                )
                tuning.insert(0, "fault_id", fault_id)
                tuning.insert(2, "method", spec["method"])
                tuning_tables.append(tuning)
                k_source = f"Fault-{fault_id} development only"
            else:
                k = float(spec["fixed_k"])
                k_source = "frozen proposed value"
            metrics, _, normal_runs, final_runs = step15.evaluate_cusum_method(
                step11, method_inputs, k
            )

        control_limit = float(metrics["control_limit"])
        common = {
            "fault_id": fault_id,
            "method_key": spec["method_key"],
            "method": spec["method"],
            "method_short": spec["method_short"],
            "sequential": bool(spec["sequential"]),
            "cusum_k": k,
            "cusum_k_source": k_source,
            "control_limit": control_limit,
            "normal_run_level_far": float(metrics["normal_run_level_far"]),
            "normal_run_level_far_ci_low": float(
                metrics["normal_run_level_far_ci_low"]
            ),
            "normal_run_level_far_ci_high": float(
                metrics["normal_run_level_far_ci_high"]
            ),
            "n_selected_variables": selected_count
            if spec["method_key"] == "proposed_top5_cusum"
            else (
                1 if spec["method_key"] == "top1_signed_cusum" else np.nan
            ),
            "protocol_status": "evaluated without final/companion retuning",
        }
        full = {
            "dataset": "faulty_testing_final_full_960",
            "fault_start_sample": FAULT_TESTING_START,
            "pre_fault_samples_per_run": FAULT_TESTING_START - 1,
            "post_fault_samples_per_run": 960 - FAULT_TESTING_START + 1,
        }
        full.update(common)
        full.update(normalize_step15_fault_metrics(metrics))
        rows.append(full)
        run_tables.append(
            annotate_runs(
                final_runs,
                fault_id,
                spec["method_key"],
                spec["method"],
                "faulty_testing_final_full_960",
            )
        )
        run_tables.append(
            annotate_runs(
                normal_runs,
                fault_id,
                spec["method_key"],
                spec["method"],
                "normal_evaluation",
            )
        )

        for dataset, start in [
            ("original_onset_horizon_matched", FAULT_TRAINING_START),
            ("faulty_training_companion", FAULT_TRAINING_START),
        ]:
            statistic = monitoring_statistic(
                step15,
                method_inputs[dataset],
                spec["sequential"],
                k,
            )
            current_metrics, current_runs = evaluate_generic_fault(
                step11, statistic, control_limit, start
            )
            row = {
                "dataset": dataset,
                "fault_start_sample": start,
                "pre_fault_samples_per_run": start - 1,
                "post_fault_samples_per_run": 500 - start + 1,
            }
            row.update(common)
            row.update(current_metrics)
            rows.append(row)
            run_tables.append(
                annotate_runs(
                    current_runs,
                    fault_id,
                    spec["method_key"],
                    spec["method"],
                    dataset,
                )
            )

    results = pd.DataFrame(rows)
    tuning = (
        pd.concat(tuning_tables, ignore_index=True)
        if tuning_tables
        else pd.DataFrame()
    )
    runs = pd.concat(run_tables, ignore_index=True) if run_tables else pd.DataFrame()
    return results, tuning, runs


def plot_proposed_summary(results):
    subset = results[
        (results["method_key"] == "proposed_top5_cusum")
        & results["dataset"].isin(
            [
                "faulty_testing_final_full_960",
                "original_onset_horizon_matched",
                "faulty_training_companion",
            ]
        )
    ].copy()
    if subset.empty:
        return []
    datasets = [
        ("faulty_testing_final_full_960", "Final full", "#4C78A8"),
        ("original_onset_horizon_matched", "Original matched", "#7A9E9F"),
        ("faulty_training_companion", "Companion", "#D95F59"),
    ]
    faults = TARGET_FAULTS
    x = np.arange(len(faults), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.2), sharex=True)
    for position, (dataset, label, color) in enumerate(datasets):
        current = (
            subset[subset["dataset"] == dataset]
            .set_index("fault_id")
            .reindex(faults)
        )
        offset = (position - 1) * width
        axes[0].bar(
            x + offset,
            100 * current["fault_run_level_detection_rate"].to_numpy(dtype=float),
            width,
            color=color,
            label=label,
        )
        axes[1].bar(
            x + offset,
            100
            * current["fault_post_fault_point_detection_rate"].to_numpy(
                dtype=float
            ),
            width,
            color=color,
        )
        axes[2].bar(
            x + offset,
            current["median_alarm_delay_min"].to_numpy(dtype=float),
            width,
            color=color,
        )
    axes[0].set_ylabel("Run detection (%)")
    axes[1].set_ylabel("Post-fault point detection (%)")
    axes[2].set_ylabel("Median delay (min)")
    axes[0].legend(loc="best")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"Fault {fault_id}" for fault_id in faults])
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Prespecified Top-5 protocol on difficult TEP faults")
    fig.tight_layout()
    png = FIGURE_DIR / "fig18a_faults9_15_prespecified_generalization.png"
    pdf = FIGURE_DIR / "fig18a_faults9_15_prespecified_generalization.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def print_proposed_summary(results, screen_summary):
    print("\n" + "=" * 88)
    print("Step 18 proposed-method summary")
    print("=" * 88)
    rows = results[
        (results["method_key"] == "proposed_top5_cusum")
        & (results["dataset"] != "protocol_not_estimable")
    ].copy()
    columns = [
        "fault_id",
        "dataset",
        "n_selected_variables",
        "normal_run_level_far",
        "fault_run_level_detection_rate",
        "fault_post_fault_point_detection_rate",
        "median_alarm_delay_min",
    ]
    if len(rows):
        print(rows[columns].to_string(index=False))
    else:
        print("The proposed signed-residual method was not estimable.")
    print("\nScreening transfer summary:")
    print(pd.DataFrame(screen_summary).to_string(index=False))


def main():
    print("=" * 88)
    print("Step 18: Prespecified generalization to TEP Faults 9 and 15")
    print("=" * 88)
    print(
        "Both faults will be reported under the same fixed protocol; companion "
        "data will not be used for tuning."
    )

    step11, step15 = load_dependencies()
    variable_cols = variable_columns_from_table()
    prepare_fault_caches(variable_cols)

    normal_paths = {
        "train": PROCESSED_DIR / "normal_train.pkl",
        "validation": PROCESSED_DIR / "normal_validation_runs_1_250.pkl",
        "evaluation": PROCESSED_DIR / "normal_evaluation_runs_251_500.pkl",
    }
    for path in normal_paths.values():
        require_file(path, "run Step 02")

    print("Loading normal datasets and fitting the unchanged PCA model...")
    normal_train = pd.read_pickle(normal_paths["train"])
    normal_pre_reference = extract_normal_training_pre_reference(
        normal_train, variable_cols
    )
    model = step11.fit_pca_model(normal_train, variable_cols)
    eigenvalues = step15.derive_retained_eigenvalues(
        normal_train, variable_cols, model
    )
    del normal_train
    gc.collect()

    normal_validation = pd.read_pickle(normal_paths["validation"])
    normal_evaluation = pd.read_pickle(normal_paths["evaluation"])
    normal_validation_res, normal_validation_stats, residual_cols = (
        step15.compute_pca_outputs(
            normal_validation, variable_cols, model, eigenvalues
        )
    )
    normal_evaluation_res, normal_evaluation_stats, current_cols = (
        step15.compute_pca_outputs(
            normal_evaluation, variable_cols, model, eigenvalues
        )
    )
    if current_cols != residual_cols:
        raise ValueError("Normal residual column order changed")
    del normal_validation, normal_evaluation
    gc.collect()

    all_audits = []
    all_screen_metrics = []
    all_selected = []
    all_transfer = []
    all_screen_summary = []
    all_results = []
    all_tuning = []
    all_runs = []
    all_lda_weights = []
    global_parameter_rows = []

    for fault_id in TARGET_FAULTS:
        print("\n" + "=" * 88)
        print(f"Fault {fault_id}")
        print("=" * 88)
        development, audit = load_fault_cache(
            fault_id, "development", variable_cols
        )
        all_audits.append(audit)
        final, audit = load_fault_cache(fault_id, "final", variable_cols)
        all_audits.append(audit)
        companion, audit = load_fault_cache(
            fault_id, "companion", variable_cols
        )
        audit.update(
            prefault_identity_audit(
                normal_pre_reference, companion, variable_cols
            )
        )
        all_audits.append(audit)
        print(
            "Companion pre-fault identical to PCA-training normal: "
            f"{audit['prefault_identical_to_pca_training']}"
        )
        matched = make_matched_original_segment(final)

        development_res, development_stats, current_cols = (
            step15.compute_pca_outputs(
                development, variable_cols, model, eigenvalues
            )
        )
        final_res, final_stats, final_cols = step15.compute_pca_outputs(
            final, variable_cols, model, eigenvalues
        )
        matched_res, matched_stats, matched_cols = step15.compute_pca_outputs(
            matched, variable_cols, model, eigenvalues
        )
        companion_res, companion_stats, companion_cols = (
            step15.compute_pca_outputs(
                companion, variable_cols, model, eigenvalues
            )
        )
        if not all(
            columns == residual_cols
            for columns in [current_cols, final_cols, matched_cols, companion_cols]
        ):
            raise ValueError(f"Residual column order changed for Fault {fault_id}")
        del development, final, matched, companion
        gc.collect()

        (
            dev_screen,
            dev_selected,
            screen_metrics,
            selected_tables,
            transfer,
            screen_summary,
        ) = run_primary_screens(
            step11,
            fault_id,
            variable_cols,
            residual_cols,
            normal_validation_res,
            development_res,
            companion_res,
        )
        all_screen_metrics.append(screen_metrics)
        all_selected.append(selected_tables)
        all_transfer.append(transfer)
        all_screen_summary.append(screen_summary)
        print(
            f"Development stable variables: "
            f"{screen_summary['development_stable_variable_count']}; "
            f"selected={dev_selected['variable'].tolist()}"
        )
        print(
            f"Post-freeze companion Top-5: "
            f"{screen_summary['companion_postfreeze_top5']}; overlap="
            f"{screen_summary['development_companion_top5_overlap']}/5"
        )

        residuals = {
            "normal_validation": normal_validation_res,
            "normal_evaluation": normal_evaluation_res,
            "fault3_development": development_res,
            "fault3_final_test": final_res,
            "original_onset_horizon_matched": matched_res,
            "faulty_training_companion": companion_res,
        }
        statistics = {
            "normal_validation": normal_validation_stats,
            "normal_evaluation": normal_evaluation_stats,
            "fault3_development": development_stats,
            "fault3_final_test": final_stats,
            "original_onset_horizon_matched": matched_stats,
            "faulty_training_companion": companion_stats,
        }
        inputs, weights, lda_metadata, global_parameters = build_input_sets(
            step11,
            step15,
            variable_cols,
            residual_cols,
            residuals,
            statistics,
            dev_screen,
            dev_selected,
        )
        weights.insert(0, "fault_id", fault_id)
        all_lda_weights.append(weights)
        for parameter, value in global_parameters.items():
            global_parameter_rows.append(
                {
                    "fault_id": fault_id,
                    "parameter": parameter,
                    "value": value,
                }
            )
        global_parameter_rows.append(
            {
                "fault_id": fault_id,
                "parameter": "ledoit_wolf_shrinkage",
                "value": lda_metadata["ledoit_wolf_shrinkage"],
            }
        )
        results, tuning, runs = evaluate_fault_methods(
            step11,
            step15,
            fault_id,
            inputs,
            int(len(dev_selected)),
        )
        all_results.append(results)
        all_tuning.append(tuning)
        all_runs.append(runs)
        proposed = results[
            (results["method_key"] == "proposed_top5_cusum")
            & (results["dataset"] == "faulty_testing_final_full_960")
        ]
        if len(proposed):
            row = proposed.iloc[0]
            print(
                f"Proposed final: normal FAR="
                f"{100*row['normal_run_level_far']:.2f}%, run detection="
                f"{100*row['fault_run_level_detection_rate']:.2f}%, point="
                f"{100*row['fault_post_fault_point_detection_rate']:.2f}%, "
                f"delay={row['median_alarm_delay_min']} min"
            )

        del (
            development_res,
            development_stats,
            final_res,
            final_stats,
            matched_res,
            matched_stats,
            companion_res,
            companion_stats,
            residuals,
            statistics,
            inputs,
        )
        gc.collect()

    audits = pd.DataFrame(all_audits)
    screen_metrics = pd.concat(all_screen_metrics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    transfer = pd.concat(all_transfer, ignore_index=True)
    screen_summary = pd.DataFrame(all_screen_summary)
    results = pd.concat(all_results, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    runs = pd.concat(all_runs, ignore_index=True)
    lda_weights = pd.concat(all_lda_weights, ignore_index=True)
    global_parameters = pd.DataFrame(global_parameter_rows)

    paths = {
        "data_audit": TABLE_DIR / "18_faults9_15_data_audit.csv",
        "screen_metrics": TABLE_DIR
        / "18_faults9_15_screening_all_variables.csv",
        "selected": TABLE_DIR / "18_faults9_15_selected_variables.csv",
        "direction_transfer": TABLE_DIR
        / "18_faults9_15_companion_direction_transfer.csv",
        "screen_summary": TABLE_DIR
        / "18_faults9_15_screening_transfer_summary.csv",
        "results": TABLE_DIR / "18_faults9_15_all_method_results.csv",
        "tuning": TABLE_DIR / "18_faults9_15_competitor_k_tuning.csv",
        "run_results": TABLE_DIR / "18_faults9_15_run_results.csv",
        "lda_weights": TABLE_DIR / "18_faults9_15_lda_weights.csv",
        "global_parameters": TABLE_DIR
        / "18_faults9_15_global_transform_parameters.csv",
    }
    for key, frame in [
        ("data_audit", audits),
        ("screen_metrics", screen_metrics),
        ("selected", selected),
        ("direction_transfer", transfer),
        ("screen_summary", screen_summary),
        ("results", results),
        ("tuning", tuning),
        ("run_results", runs),
        ("lda_weights", lda_weights),
        ("global_parameters", global_parameters),
    ]:
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    figure_paths = plot_proposed_summary(results)
    print_proposed_summary(results, all_screen_summary)

    config = {
        "script": "scripts/18_faults9_15_prespecified_generalization.py",
        "target_faults": TARGET_FAULTS,
        "protocol_fixed_before_results": True,
        "development_source": (
            "TEP_Faulty_Testing.RData, target-fault runs 1-250"
        ),
        "final_source": "TEP_Faulty_Testing.RData, target-fault runs 251-500",
        "companion_source": (
            "TEP_Faulty_Training.RData, all 500 target-fault runs; post-freeze"
        ),
        "faulty_training_used_for_tuning": False,
        "screening": {
            "development_window": [SCREEN_NORMAL_START, SCREEN_NORMAL_END],
            "companion_diagnostic_window": [
                SCREEN_COMPANION_START,
                SCREEN_COMPANION_END,
            ],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES_EXPECTED,
            "bootstrap_unit": "independent simulation run",
            "confidence": 0.95,
            "ranking": "absolute Hedges' g among CI-stable residuals",
            "top_k": PRIMARY_TOP_K,
        },
        "proposed_cusum_k": PRIMARY_CUSUM_K,
        "competitor_k_grid": COMPETITOR_K_GRID_EXPECTED,
        "run_level_control_limit_percentile": RUN_LEVEL_PERCENTILE,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "screening_summaries": all_screen_summary,
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
        "large_cache_files_not_for_result_archive": [
            str(path.relative_to(PROJECT_ROOT))
            for fault_paths in CACHE_PATHS.values()
            for path in fault_paths.values()
        ],
    }
    config_path = CONFIG_DIR / "faults9_15_prespecified_generalization.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved configuration: {config_path}")
    print("Step 18 finished successfully.")


if __name__ == "__main__":
    main()
