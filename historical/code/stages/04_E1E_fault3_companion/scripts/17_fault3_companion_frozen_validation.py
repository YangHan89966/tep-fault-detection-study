"""
Step 17: Frozen companion-set validation for TEP Fault 3.

Purpose
-------
The Rieth TEP release contains a second set of faulty simulations in
``TEP_Faulty_Training.RData``.  None of those faulty runs was used by Steps
02--16.  This script evaluates the Step-15 monitors on Fault 3 from that file
without changing the PCA model, variable definitions, residual signs, CUSUM
reference values, scalar transformations, or control limits.

Important interpretation
------------------------
This is an untouched *companion-set* fault validation, not validation on a
different plant or simulator.  Faulty Training contains 500 samples per run
and the benchmark fault begins at sample 21, whereas Faulty Testing contains
960 samples per run and the fault begins at sample 161.  To separate a data-
source effect from the different onset/horizon, the script also evaluates an
onset/horizon-matched 500-sample segment of the original final-test runs
(original samples 141--640 are relabelled 1--500, with fault onset at 21).

The script additionally tests whether the 20 pre-fault samples in Faulty
Training are identical to the same run/sample rows in FaultFree Training,
which was used to fit PCA.  That provenance check must be reported when the
companion-set result is described.

No retuning rule
----------------
* Method definitions and selected k values come from Step 15.
* Control limits come directly from Step 15.
* The script reconstructs every normal-validation chart and requires its
  control limit to reproduce the stored Step-15 value before evaluating the
  new data.
* Faulty Training is never used to choose a variable, sign, k, transform, or
  threshold.

Place in:
    TEP_Fault3_Project/scripts/17_fault3_companion_frozen_validation.py

Run from project root:
    python scripts/17_fault3_companion_frozen_validation.py

Prerequisites
-------------
* Steps 02, 11, and 15 have completed successfully.
* Keep ``11_unpaired_runlevel_screening.py`` and
  ``15_fair_sequential_baselines.py`` in the scripts directory.
* Place ``TEP_Faulty_Training.RData`` in ``data/raw``.
"""

from pathlib import Path
import gc
import hashlib
import importlib.util
import json

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

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

FAULT_ID = 3
SAMPLING_INTERVAL_MIN = 3

# Fixed benchmark definitions, specified before reading the companion result.
ORIGINAL_FAULT_START = 161
COMPANION_FAULT_START = 21
COMPANION_LAST_SAMPLE = 500

# Original samples 141--640 give 20 pre-fault and 480 post-fault samples,
# exactly matching the companion run layout after relabelling to 1--500.
MATCHED_ORIGINAL_FIRST_SAMPLE = 141
MATCHED_ORIGINAL_LAST_SAMPLE = 640
MATCHED_SAMPLE_OFFSET = MATCHED_ORIGINAL_FIRST_SAMPLE - 1

PRIMARY_WINDOW = "early_160_primary"
PRIMARY_TOP_K = 5
PRIMARY_CUSUM_K = 0.05
CONTROL_LIMIT_ATOL = 1e-9

RAW_COMPANION_PATH = RAW_DIR / "TEP_Faulty_Training.RData"
BRIDGE_MATRIX_PATH = PROJECT_ROOT / "data" / "bridge" / "fault3_companion_matrix.float64.npy"
BRIDGE_EXTRACTION_AUDIT_PATH = PROJECT_ROOT / "data" / "bridge" / "fault3_companion_extraction_audit.json"
EXPECTED_RAW_COMPANION_SHA256 = "058df8bd90c5f575f44d2f6085bdd3bc3662e3e9cc25a50e34af42be3748952b"
EXPECTED_RAW_COMPANION_SIZE_BYTES = 494063194
EXPECTED_BRIDGE_MATRIX_DTYPE = "float64"
STEP15_RESULTS_PATH = TABLE_DIR / "15_fair_sequential_baseline_results.csv"
STEP15_WEIGHTS_PATH = TABLE_DIR / "15_shrinkage_lda_residual_weights.csv"
STEP15_TRANSFORMS_PATH = TABLE_DIR / "15_scalar_transform_parameters.csv"

METHOD_SPECS = [
    ("pca_q_raw", "pca_q", False),
    ("pca_t2_raw", "pca_t2", False),
    ("pca_joint_raw", "pca_joint", False),
    ("pca_q_cusum", "pca_q", True),
    ("pca_t2_cusum", "pca_t2", True),
    ("pca_joint_cusum", "pca_joint", True),
    ("all52_signed_cusum", "all52_signed", True),
    ("lda_residual_cusum", "lda_residual", True),
    ("top1_signed_cusum", "top1_signed", True),
    ("proposed_top5_cusum", "proposed_top5", True),
]

FULL_RESULT_FIELD_MAP = {
    "n_runs": "fault3_n_runs",
    "pre_fault_alarm_runs": "fault3_pre_fault_alarm_runs",
    "pre_fault_run_alarm_rate": "fault3_pre_fault_run_far",
    "pre_fault_point_alarm_rate": "fault3_pre_fault_point_far",
    "detected_runs": "detected_runs",
    "run_level_detection_rate": "fault3_run_level_detection_rate",
    "run_level_detection_ci_low": "fault3_run_level_detection_ci_low",
    "run_level_detection_ci_high": "fault3_run_level_detection_ci_high",
    "post_fault_point_detection_rate": (
        "fault3_post_fault_point_detection_rate"
    ),
    "median_alarm_delay_min": "median_alarm_delay_min",
    "mean_alarm_delay_min": "mean_alarm_delay_min",
    "alarm_delay_q1_min": "alarm_delay_q1_min",
    "alarm_delay_q3_min": "alarm_delay_q3_min",
}


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_python_module(path, module_name):
    require_file(path, f"place {path.name} in the scripts directory")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    step15 = load_python_module(
        SCRIPT_DIR / "15_fair_sequential_baselines.py", "step15_frozen"
    )
    required15 = [
        "load_step11_module",
        "load_unpaired_definitions",
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
    ]
    missing = [name for name in required15 if not hasattr(step15, name)]
    if missing:
        raise AttributeError(f"Step 15 is missing required functions: {missing}")
    step11 = step15.load_step11_module()
    return step11, step15


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_integer_column(frame, column):
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{column} contains non-finite values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"{column} contains non-integer values")
    frame[column] = rounded.astype(np.int64)


def read_fault3_companion(variable_cols):
    matrix_path = require_file(
        BRIDGE_MATRIX_PATH,
        "run the exact-hash WSL pyreadr extraction bridge",
    )
    audit_path = require_file(
        BRIDGE_EXTRACTION_AUDIT_PATH,
        "run the exact-hash WSL pyreadr extraction bridge",
    )
    extraction = json.loads(audit_path.read_text(encoding="utf-8"))
    if extraction.get("result") != "PASS":
        raise RuntimeError("COMPANION_EXTRACTION_AUDIT_NOT_PASS")
    if extraction.get("raw_file_sha256") != EXPECTED_RAW_COMPANION_SHA256:
        raise RuntimeError(
            "COMPANION_RAW_SHA256_MISMATCH:"
            f"{extraction.get('raw_file_sha256')}"
        )
    if int(extraction.get("raw_file_size_bytes", -1)) != EXPECTED_RAW_COMPANION_SIZE_BYTES:
        raise RuntimeError("COMPANION_RAW_SIZE_MISMATCH")
    observed_matrix_sha = sha256_file(matrix_path)
    if observed_matrix_sha != extraction.get("matrix_npy_sha256"):
        raise RuntimeError(
            "COMPANION_BRIDGE_MATRIX_SHA256_MISMATCH:"
            f"{observed_matrix_sha}"
        )
    if matrix_path.stat().st_size != int(extraction.get("matrix_npy_size_bytes", -1)):
        raise RuntimeError("COMPANION_BRIDGE_MATRIX_SIZE_MISMATCH")

    columns = extraction.get("columns")
    if not isinstance(columns, list) or len(columns) != 55:
        raise ValueError("Companion bridge column metadata is invalid")
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.ndim != 2 or tuple(int(v) for v in matrix.shape) != (250000, 55):
        raise ValueError(f"Companion bridge matrix shape is invalid: {matrix.shape}")
    if str(matrix.dtype) != EXPECTED_BRIDGE_MATRIX_DTYPE:
        raise ValueError(f"Companion bridge matrix dtype is invalid: {matrix.dtype}")
    if extraction.get("matrix_shape") != [250000, 55]:
        raise ValueError("Companion extraction audit matrix shape is invalid")
    if extraction.get("matrix_dtype") != EXPECTED_BRIDGE_MATRIX_DTYPE:
        raise ValueError("Companion extraction audit matrix dtype is invalid")

    full = pd.DataFrame(matrix, columns=columns)
    required_columns = [
        "faultNumber",
        "simulationRun",
        "sample",
        *variable_cols,
    ]
    missing = [column for column in required_columns if column not in full.columns]
    if missing:
        raise ValueError(f"Companion bridge data is missing columns: {missing}")
    fault3 = full.loc[:, required_columns].copy()
    del full, matrix
    gc.collect()

    for column in ["faultNumber", "simulationRun", "sample"]:
        normalize_integer_column(fault3, column)
    fault3 = fault3.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )
    if not (fault3["faultNumber"] == FAULT_ID).all():
        raise ValueError("Companion bridge contains a non-Fault-3 row")

    expected_runs = np.arange(1, 501, dtype=np.int64)
    actual_runs = np.sort(fault3["simulationRun"].unique())
    if not np.array_equal(actual_runs, expected_runs):
        raise ValueError(
            "Fault 3 companion runs are not exactly 1--500: "
            f"min={actual_runs.min()}, max={actual_runs.max()}, "
            f"count={len(actual_runs)}"
        )
    if int(fault3["sample"].min()) != 1 or int(fault3["sample"].max()) != 500:
        raise ValueError(
            "Fault 3 companion samples are not exactly within 1--500"
        )
    counts = fault3.groupby("simulationRun")["sample"].nunique()
    if not (counts == COMPANION_LAST_SAMPLE).all():
        raise ValueError("At least one Fault 3 companion run is incomplete")
    duplicated = int(
        fault3.duplicated(["faultNumber", "simulationRun", "sample"]).sum()
    )
    if duplicated:
        raise ValueError(f"Companion data contains {duplicated} duplicate rows")
    values = fault3[variable_cols].to_numpy(dtype=float, copy=False)
    nonfinite = int((~np.isfinite(values)).sum())
    if nonfinite:
        raise ValueError(
            f"Fault 3 companion variables contain {nonfinite} non-finite values"
        )

    audit = {
        "file_name": extraction["raw_file_name"],
        "file_size_bytes": int(extraction["raw_file_size_bytes"]),
        "file_sha256": extraction["raw_file_sha256"],
        "r_object_names": ";".join(extraction["r_object_names"]),
        "selected_r_object": extraction["selected_r_object"],
        "complete_object_rows": int(extraction["complete_object_shape"][0]),
        "complete_object_columns": int(extraction["complete_object_shape"][1]),
        "fault_id": FAULT_ID,
        "fault3_rows": int(len(fault3)),
        "fault3_runs": int(fault3["simulationRun"].nunique()),
        "sample_min": int(fault3["sample"].min()),
        "sample_max": int(fault3["sample"].max()),
        "rows_per_run_min": int(counts.min()),
        "rows_per_run_max": int(counts.max()),
        "duplicate_identifier_rows": duplicated,
        "nonfinite_process_values": nonfinite,
        "prespecified_fault_start_sample": COMPANION_FAULT_START,
        "pre_fault_samples_per_run": COMPANION_FAULT_START - 1,
        "post_fault_samples_per_run": (
            COMPANION_LAST_SAMPLE - COMPANION_FAULT_START + 1
        ),
        "ingestion_bridge": "WSL_PYREADR_TO_EXACT_FLOAT64_NPY_THEN_WINDOWS_TEP_RUNTIME",
        "ingestion_bridge_matrix_sha256": observed_matrix_sha,
        "ingestion_bridge_extraction_audit_sha256": sha256_file(audit_path),
        "scientific_source_reference_sha256": "275f088af8ea56896108863b1fcad6fec81db6f7ad5b434343e8ea74d14dd841",
        "faulty_training_used_for_any_tuning": False,
    }
    return fault3, audit


def extract_normal_training_pre_reference(normal_train, variable_cols):
    reference = normal_train[
        normal_train["sample"].between(1, COMPANION_FAULT_START - 1)
    ][["simulationRun", "sample", *variable_cols]].copy()
    for column in ["simulationRun", "sample"]:
        normalize_integer_column(reference, column)
    reference = reference.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )
    return reference


def compare_prefault_with_pca_training(reference, companion, variable_cols):
    companion_pre = companion[
        companion["sample"] < COMPANION_FAULT_START
    ][["simulationRun", "sample", *variable_cols]].copy()
    companion_pre = companion_pre.sort_values(
        ["simulationRun", "sample"]
    ).reset_index(drop=True)
    if len(reference) != len(companion_pre):
        raise ValueError(
            "Pre-fault provenance comparison has unequal row counts: "
            f"normal={len(reference)}, faulty={len(companion_pre)}"
        )
    reference_ids = reference[["simulationRun", "sample"]].to_numpy()
    companion_ids = companion_pre[["simulationRun", "sample"]].to_numpy()
    if not np.array_equal(reference_ids, companion_ids):
        raise ValueError("Pre-fault provenance identifiers do not align")
    normal_values = reference[variable_cols].to_numpy(dtype=float)
    faulty_values = companion_pre[variable_cols].to_numpy(dtype=float)
    absolute = np.abs(normal_values - faulty_values)
    exact_cells = int(np.count_nonzero(normal_values == faulty_values))
    total_cells = int(normal_values.size)
    row_exact = np.all(normal_values == faulty_values, axis=1)
    return {
        "prefault_rows_compared": int(len(reference)),
        "prefault_variable_cells_compared": total_cells,
        "prefault_exact_cell_fraction": exact_cells / total_cells,
        "prefault_exact_row_fraction": float(row_exact.mean()),
        "prefault_max_absolute_difference": float(absolute.max()),
        "prefault_identical_to_pca_training": bool(
            np.array_equal(normal_values, faulty_values)
        ),
    }


def make_matched_original_segment(frame):
    matched = frame[
        frame["sample"].between(
            MATCHED_ORIGINAL_FIRST_SAMPLE, MATCHED_ORIGINAL_LAST_SAMPLE
        )
    ].copy()
    matched["sample"] = (
        pd.to_numeric(matched["sample"], errors="raise").astype(int)
        - MATCHED_SAMPLE_OFFSET
    )
    counts = matched.groupby("simulationRun")["sample"].nunique()
    if len(counts) != 250 or not (counts == COMPANION_LAST_SAMPLE).all():
        raise ValueError("Could not create complete matched original segments")
    if int(matched["sample"].min()) != 1 or int(matched["sample"].max()) != 500:
        raise ValueError("Matched original sample relabelling failed")
    return matched.sort_values(["simulationRun", "sample"]).reset_index(drop=True)


def load_step15_results():
    require_file(STEP15_RESULTS_PATH, "run Step 15")
    table = pd.read_csv(STEP15_RESULTS_PATH)
    if table["method_key"].duplicated().any():
        raise ValueError("Step 15 results contain duplicate method keys")
    table = table.set_index("method_key", drop=False)
    expected = [item[0] for item in METHOD_SPECS]
    missing = [key for key in expected if key not in table.index]
    if missing:
        raise ValueError(f"Step 15 results are missing methods: {missing}")
    proposed = table.loc["proposed_top5_cusum"]
    if not np.isclose(
        float(proposed["cusum_k"]), PRIMARY_CUSUM_K, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Stored proposed CUSUM k is not the frozen value 0.05")
    return table


def build_input_sets(
    step11,
    step15,
    variable_cols,
    normal_validation_res,
    fault_development_res,
    residual_cols,
    residuals,
    statistics,
):
    all_primary, stable_ranked, top1, frozen_top5 = (
        step15.load_unpaired_definitions(variable_cols)
    )
    if frozen_top5["variable"].tolist() != [
        "xmeas_2",
        "xmeas_21",
        "xmv_1",
        "xmv_8",
        "xmeas_15",
    ]:
        raise ValueError(
            "Frozen Top-5 differs from the preregistered Step-11 primary set: "
            f"{frozen_top5['variable'].tolist()}"
        )

    global_parameters = step15.fit_global_transforms(
        statistics["normal_validation"]
    )
    global_inputs = {
        dataset: step15.transform_global_inputs(frame, global_parameters)
        for dataset, frame in statistics.items()
    }
    input_sets = {
        "pca_q": {
            dataset: values["q"] for dataset, values in global_inputs.items()
        },
        "pca_t2": {
            dataset: values["t2"] for dataset, values in global_inputs.items()
        },
        "pca_joint": {
            dataset: values["joint"] for dataset, values in global_inputs.items()
        },
    }

    all52_table = all_primary[
        ["variable", "residual_variable", "sign_for_positive_evidence"]
    ].copy()
    local_tables = {
        "all52_signed": all52_table,
        "top1_signed": top1,
        "proposed_top5": frozen_top5,
    }
    local_scalers = {}
    for key, selected in local_tables.items():
        scaler = step11.fit_signed_scaler(normal_validation_res, selected)
        local_scalers[key] = scaler
        input_sets[key] = {
            dataset: step15.signed_local_input(step11, frame, scaler)
            for dataset, frame in residuals.items()
        }

    lda_weights, lda_weight_table, lda_metadata = step15.fit_lda_score(
        step11,
        normal_validation_res,
        fault_development_res,
        residual_cols,
        variable_cols,
    )
    lda_raw = {
        dataset: step15.apply_lda_score(frame, residual_cols, lda_weights)
        for dataset, frame in residuals.items()
    }
    lda_scaler = step15.fit_scalar_standardizer(
        lda_raw["normal_validation"]
    )
    input_sets["lda_residual"] = {
        dataset: step15.apply_scalar_standardizer(frame, lda_scaler)
        for dataset, frame in lda_raw.items()
    }

    definitions = {
        "frozen_top5_variables": frozen_top5["variable"].tolist(),
        "frozen_top5_signs": frozen_top5[
            "sign_for_positive_evidence"
        ].astype(int).tolist(),
        "frozen_top1_variable": top1["variable"].tolist(),
        "frozen_top1_sign": top1[
            "sign_for_positive_evidence"
        ].astype(int).tolist(),
        "n_direction_stable_primary_variables": int(len(stable_ranked)),
        "global_parameters": global_parameters,
        "lda_scaler": lda_scaler,
        "lda_metadata": lda_metadata,
    }
    return input_sets, definitions, lda_weight_table, local_scalers


def check_transform_and_weight_parity(definitions, lda_weight_table):
    checks = []
    if STEP15_TRANSFORMS_PATH.exists():
        stored = pd.read_csv(STEP15_TRANSFORMS_PATH).set_index("parameter")
        reconstructed = dict(definitions["global_parameters"])
        reconstructed["lda_score_normal_mean"] = definitions["lda_scaler"][
            "mean"
        ]
        reconstructed["lda_score_normal_std"] = definitions["lda_scaler"][
            "std"
        ]
        reconstructed["ledoit_wolf_shrinkage"] = definitions["lda_metadata"][
            "ledoit_wolf_shrinkage"
        ]
        for name, value in reconstructed.items():
            if name not in stored.index:
                raise ValueError(f"Stored Step-15 transforms omit {name}")
            reference = float(stored.loc[name, "value"])
            difference = float(value) - reference
            checks.append(
                {
                    "check_type": "scalar_transform",
                    "item": name,
                    "reconstructed_value": float(value),
                    "stored_value": reference,
                    "absolute_difference": abs(difference),
                    "passed": bool(
                        np.isclose(
                            difference, 0.0, rtol=0.0, atol=1e-12
                        )
                    ),
                }
            )
    else:
        raise FileNotFoundError(
            f"Required Step-15 transform table not found: {STEP15_TRANSFORMS_PATH}"
        )

    if STEP15_WEIGHTS_PATH.exists():
        stored = pd.read_csv(STEP15_WEIGHTS_PATH).set_index("variable")
        current = lda_weight_table.set_index("variable")
        if set(stored.index) != set(current.index):
            raise ValueError("Stored and reconstructed LDA variables differ")
        for variable in current.index:
            value = float(current.loc[variable, "lda_weight"])
            reference = float(stored.loc[variable, "lda_weight"])
            difference = value - reference
            checks.append(
                {
                    "check_type": "lda_weight",
                    "item": variable,
                    "reconstructed_value": value,
                    "stored_value": reference,
                    "absolute_difference": abs(difference),
                    "passed": bool(
                        np.isclose(
                            difference, 0.0, rtol=0.0, atol=1e-12
                        )
                    ),
                }
            )
    else:
        raise FileNotFoundError(
            f"Required Step-15 LDA weight table not found: {STEP15_WEIGHTS_PATH}"
        )
    return checks


def evaluate_fault_statistic(step11, statistic, limit, fault_start_sample):
    df = statistic.sort_values(["simulationRun", "sample"]).copy()
    df["alarm"] = df["monitoring_statistic"] > float(limit)
    pre = df[df["sample"] < int(fault_start_sample)].copy()
    post = df[df["sample"] >= int(fault_start_sample)].copy()
    if pre.empty or post.empty:
        raise ValueError("Fault evaluation requires nonempty pre/post segments")

    all_runs = pd.DataFrame(
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
    runs = all_runs.merge(pre_alarm, on="simulationRun", how="left")
    runs = runs.merge(post_alarm, on="simulationRun", how="left")
    runs = runs.merge(first_alarm, on="simulationRun", how="left")
    for column in ["pre_fault_run_alarm", "post_fault_run_alarm"]:
        runs[column] = runs[column].fillna(False).astype(bool)
    runs["detected"] = runs["post_fault_run_alarm"]
    runs["alarm_delay_min"] = (
        runs["first_alarm_sample"] - int(fault_start_sample)
    ) * SAMPLING_INTERVAL_MIN
    runs["right_censored"] = ~runs["detected"]

    n_runs = int(len(runs))
    pre_alarm_runs = int(runs["pre_fault_run_alarm"].sum())
    detected_runs = int(runs["detected"].sum())
    pre_low, pre_high = step11.clopper_pearson(pre_alarm_runs, n_runs)
    det_low, det_high = step11.clopper_pearson(detected_runs, n_runs)
    delays = runs.loc[runs["detected"], "alarm_delay_min"].dropna()
    metrics = {
        "n_runs": n_runs,
        "n_samples_per_run": int(df.groupby("simulationRun").size().iloc[0]),
        "fault_start_sample": int(fault_start_sample),
        "pre_fault_samples_per_run": int(fault_start_sample - 1),
        "post_fault_samples_per_run": int(
            df["sample"].max() - fault_start_sample + 1
        ),
        "pre_fault_alarm_runs": pre_alarm_runs,
        "pre_fault_run_alarm_rate": pre_alarm_runs / n_runs,
        "pre_fault_run_alarm_ci_low": pre_low,
        "pre_fault_run_alarm_ci_high": pre_high,
        "pre_fault_point_alarm_rate": float(pre["alarm"].mean()),
        "detected_runs": detected_runs,
        "right_censored_runs": n_runs - detected_runs,
        "run_level_detection_rate": detected_runs / n_runs,
        "run_level_detection_ci_low": det_low,
        "run_level_detection_ci_high": det_high,
        "post_fault_point_detection_rate": float(post["alarm"].mean()),
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


def statistic_from_input(step15, input_frame, sequential, k):
    if sequential:
        return step15.compute_cusum(input_frame, float(k))
    return step15.raw_monitoring_statistic(input_frame)


def full_result_row(step15_row):
    row = {
        "dataset": "original_final_full_960",
        "dataset_role": "stored Step-15 untouched final test",
        "n_samples_per_run": 960,
        "fault_start_sample": ORIGINAL_FAULT_START,
        "pre_fault_samples_per_run": ORIGINAL_FAULT_START - 1,
        "post_fault_samples_per_run": 960 - ORIGINAL_FAULT_START + 1,
        "right_censored_runs": int(
            int(step15_row["fault3_n_runs"])
            - int(step15_row["detected_runs"])
        ),
    }
    for output_name, stored_name in FULL_RESULT_FIELD_MAP.items():
        row[output_name] = step15_row.get(stored_name, np.nan)
    return row


def evaluate_all_methods(step11, step15, input_sets, step15_results):
    result_rows = []
    run_tables = []
    parity_rows = []
    for method_key, input_key, sequential in METHOD_SPECS:
        stored = step15_results.loc[method_key]
        k = float(stored["cusum_k"]) if sequential else np.nan
        stored_limit = float(stored["control_limit"])

        validation_stat = statistic_from_input(
            step15,
            input_sets[input_key]["normal_validation"],
            sequential,
            k,
        )
        reconstructed_limit, _ = step11.determine_limit(validation_stat)
        difference = float(reconstructed_limit) - stored_limit
        passed = bool(
            np.isclose(
                reconstructed_limit,
                stored_limit,
                rtol=0.0,
                atol=CONTROL_LIMIT_ATOL,
            )
        )
        parity_rows.append(
            {
                "check_type": "control_limit",
                "item": method_key,
                "reconstructed_value": float(reconstructed_limit),
                "stored_value": stored_limit,
                "absolute_difference": abs(difference),
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(
                f"Frozen control-limit parity failed for {method_key}: "
                f"reconstructed={reconstructed_limit}, stored={stored_limit}"
            )

        common = {
            "method_key": method_key,
            "method": stored["method"],
            "method_short": stored["method_short"],
            "sequential": bool(sequential),
            "frozen_cusum_k": k,
            "frozen_control_limit": stored_limit,
            "frozen_normal_evaluation_run_far": float(
                stored["normal_run_level_far"]
            ),
        }

        original_full = full_result_row(stored)
        original_full.update(common)
        result_rows.append(original_full)

        for dataset, role in [
            (
                "original_onset_horizon_matched",
                "original final-test samples 141-640, reset/relabelled",
            ),
            (
                "faulty_training_companion",
                "untouched companion fault realizations; no retuning",
            ),
        ]:
            statistic = statistic_from_input(
                step15,
                input_sets[input_key][dataset],
                sequential,
                k,
            )
            metrics, run_results = evaluate_fault_statistic(
                step11,
                statistic,
                stored_limit,
                COMPANION_FAULT_START,
            )
            row = {"dataset": dataset, "dataset_role": role}
            row.update(common)
            row.update(metrics)
            result_rows.append(row)
            run_results.insert(0, "dataset", dataset)
            run_results.insert(0, "method", stored["method"])
            run_results.insert(0, "method_key", method_key)
            run_tables.append(run_results)

        companion = result_rows[-1]
        matched = result_rows[-2]
        print(
            f"{stored['method']}: companion detection="
            f"{100*companion['run_level_detection_rate']:.2f}%, point="
            f"{100*companion['post_fault_point_detection_rate']:.2f}%, "
            f"delay={companion['median_alarm_delay_min']} min | "
            f"matched-original detection="
            f"{100*matched['run_level_detection_rate']:.2f}%, point="
            f"{100*matched['post_fault_point_detection_rate']:.2f}%, "
            f"delay={matched['median_alarm_delay_min']} min"
        )

    results = pd.DataFrame(result_rows)
    run_results = pd.concat(run_tables, ignore_index=True)
    parity = pd.DataFrame(parity_rows)
    return results, run_results, parity


def make_comparison_table(results):
    index_columns = [
        "method_key",
        "method",
        "method_short",
        "frozen_cusum_k",
        "frozen_control_limit",
        "frozen_normal_evaluation_run_far",
    ]
    value_columns = [
        "pre_fault_run_alarm_rate",
        "run_level_detection_rate",
        "post_fault_point_detection_rate",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
    ]
    # Pivot on the non-null method key only.  Raw charts legitimately have
    # frozen_cusum_k=NaN; using that field directly in the pivot index would
    # silently drop those methods.
    metadata = results[index_columns].drop_duplicates("method_key")
    wide = results.pivot(
        index="method_key", columns="dataset", values=value_columns
    )
    wide.columns = [f"{metric}__{dataset}" for metric, dataset in wide.columns]
    wide = metadata.merge(wide.reset_index(), on="method_key", how="left")
    for metric in [
        "run_level_detection_rate",
        "post_fault_point_detection_rate",
        "median_alarm_delay_min",
    ]:
        companion = f"{metric}__faulty_training_companion"
        matched = f"{metric}__original_onset_horizon_matched"
        full = f"{metric}__original_final_full_960"
        wide[f"companion_minus_matched__{metric}"] = (
            wide[companion] - wide[matched]
        )
        wide[f"companion_minus_full__{metric}"] = wide[companion] - wide[full]
    return wide


def plot_companion_comparison(results):
    main_keys = [
        "pca_q_cusum",
        "all52_signed_cusum",
        "lda_residual_cusum",
        "top1_signed_cusum",
        "proposed_top5_cusum",
    ]
    subset = results[
        results["method_key"].isin(main_keys)
        & results["dataset"].isin(
            [
                "original_onset_horizon_matched",
                "faulty_training_companion",
            ]
        )
    ].copy()
    label_map = (
        subset.drop_duplicates("method_key")
        .set_index("method_key")["method_short"]
        .to_dict()
    )
    labels = [label_map[key] for key in main_keys]
    x = np.arange(len(main_keys), dtype=float)
    width = 0.36
    datasets = [
        ("original_onset_horizon_matched", "Original matched", "#7A9E9F"),
        ("faulty_training_companion", "Companion set", "#D95F59"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 10.0), sharex=True)
    for offset, (dataset, label, color) in zip([-width / 2, width / 2], datasets):
        current = subset[subset["dataset"] == dataset].set_index("method_key")
        axes[0].bar(
            x + offset,
            [100 * current.loc[key, "run_level_detection_rate"] for key in main_keys],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset,
            [
                100 * current.loc[key, "post_fault_point_detection_rate"]
                for key in main_keys
            ],
            width,
            color=color,
        )
        axes[2].bar(
            x + offset,
            [current.loc[key, "median_alarm_delay_min"] for key in main_keys],
            width,
            color=color,
        )
    axes[0].set_ylabel("Run detection (%)")
    axes[1].set_ylabel("Post-fault point detection (%)")
    axes[2].set_ylabel("Median delay (min)")
    axes[0].legend(loc="best")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    fig.suptitle(
        "Frozen Fault 3 monitors: companion set vs onset/horizon-matched original"
    )
    fig.tight_layout()
    png = FIGURE_DIR / "fig17a_fault3_companion_frozen_validation.png"
    pdf = FIGURE_DIR / "fig17a_fault3_companion_frozen_validation.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    print("=" * 80)
    print("Step 17: Frozen Fault-3 companion-set validation")
    print("=" * 80)
    print(
        "No Faulty-Training result will be used to retune variables, signs, "
        "k, transforms, or control limits."
    )

    step11, step15 = load_dependencies()
    step15_results = load_step15_results()
    variable_cols = step11.load_variable_columns()

    # Load only the original datasets required to reconstruct Step 15 and the
    # matched final-test diagnostic; avoid Step 11's all-dataset loader.
    original_paths = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR
        / "normal_validation_runs_1_250.pkl",
        "fault3_development": PROCESSED_DIR
        / "fault3_development_runs_1_250.pkl",
        "fault3_final_test": PROCESSED_DIR
        / "fault3_final_test_runs_251_500.pkl",
    }
    for path in original_paths.values():
        require_file(path, "run Step 02")

    print("Loading PCA training data...")
    normal_train = pd.read_pickle(original_paths["normal_train"])
    normal_pre_reference = extract_normal_training_pre_reference(
        normal_train, variable_cols
    )
    model = step11.fit_pca_model(normal_train, variable_cols)
    eigenvalues = step15.derive_retained_eigenvalues(
        normal_train, variable_cols, model
    )
    del normal_train
    gc.collect()

    print("Computing frozen-development PCA outputs...")
    normal_validation = pd.read_pickle(original_paths["normal_validation"])
    normal_validation_res, normal_validation_stats, residual_cols = (
        step15.compute_pca_outputs(
            normal_validation, variable_cols, model, eigenvalues
        )
    )
    del normal_validation
    gc.collect()

    # Recompute the full development residual frame exactly as Step 15 did.
    # The LDA fitter itself extracts the prespecified early-160 window.  Using
    # the identical computation path makes the stored-weight parity check a
    # stronger guard against accidental pipeline drift.
    fault_development = pd.read_pickle(original_paths["fault3_development"])
    fault_development_res, _, current_cols = step15.compute_pca_outputs(
        fault_development, variable_cols, model, eigenvalues
    )
    if current_cols != residual_cols:
        raise ValueError("Residual column order changed for development data")
    del fault_development
    gc.collect()

    original_final = pd.read_pickle(original_paths["fault3_final_test"])
    original_matched = make_matched_original_segment(original_final)
    del original_final
    gc.collect()
    original_matched_res, original_matched_stats, current_cols = (
        step15.compute_pca_outputs(
            original_matched, variable_cols, model, eigenvalues
        )
    )
    if current_cols != residual_cols:
        raise ValueError("Residual column order changed for original matched data")
    del original_matched
    gc.collect()

    companion, data_audit = read_fault3_companion(variable_cols)
    provenance = compare_prefault_with_pca_training(
        normal_pre_reference, companion, variable_cols
    )
    del normal_pre_reference
    gc.collect()
    data_audit.update(provenance)
    print(
        "Pre-fault identity with PCA-training normal runs: "
        f"{data_audit['prefault_identical_to_pca_training']} "
        f"(exact-cell fraction="
        f"{data_audit['prefault_exact_cell_fraction']:.6f})"
    )

    companion_res, companion_stats, current_cols = step15.compute_pca_outputs(
        companion, variable_cols, model, eigenvalues
    )
    if current_cols != residual_cols:
        raise ValueError("Residual column order changed for companion data")
    del companion
    gc.collect()

    residuals = {
        "normal_validation": normal_validation_res,
        "original_onset_horizon_matched": original_matched_res,
        "faulty_training_companion": companion_res,
    }
    statistics = {
        "normal_validation": normal_validation_stats,
        "original_onset_horizon_matched": original_matched_stats,
        "faulty_training_companion": companion_stats,
    }
    input_sets, definitions, lda_weight_table, local_scalers = build_input_sets(
        step11,
        step15,
        variable_cols,
        normal_validation_res,
        fault_development_res,
        residual_cols,
        residuals,
        statistics,
    )

    transform_weight_checks = check_transform_and_weight_parity(
        definitions, lda_weight_table
    )
    failed = [row for row in transform_weight_checks if not row["passed"]]
    if failed:
        raise RuntimeError(
            "Frozen transform/weight parity failed before companion evaluation: "
            f"{failed[:5]}"
        )

    results, run_results, control_limit_checks = evaluate_all_methods(
        step11, step15, input_sets, step15_results
    )
    parity = pd.concat(
        [pd.DataFrame(transform_weight_checks), control_limit_checks],
        ignore_index=True,
    )
    if not parity["passed"].all():
        raise RuntimeError("At least one frozen reconstruction parity check failed")

    comparison = make_comparison_table(results)
    audit_table = pd.DataFrame(
        [{"audit_item": key, "value": value} for key, value in data_audit.items()]
    )

    paths = {
        "data_audit": TABLE_DIR / "17_fault3_companion_data_audit.csv",
        "results_long": TABLE_DIR
        / "17_fault3_companion_frozen_results_long.csv",
        "comparison": TABLE_DIR
        / "17_fault3_companion_frozen_comparison.csv",
        "run_results": TABLE_DIR
        / "17_fault3_companion_frozen_run_results.csv",
        "parity": TABLE_DIR / "17_fault3_companion_parity_checks.csv",
        "lda_weights": TABLE_DIR
        / "17_reconstructed_shrinkage_lda_weights.csv",
    }
    audit_table.to_csv(paths["data_audit"], index=False, encoding="utf-8-sig")
    results.to_csv(paths["results_long"], index=False, encoding="utf-8-sig")
    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")
    run_results.to_csv(paths["run_results"], index=False, encoding="utf-8-sig")
    parity.to_csv(paths["parity"], index=False, encoding="utf-8-sig")
    lda_weight_table.to_csv(
        paths["lda_weights"], index=False, encoding="utf-8-sig"
    )
    figure_paths = plot_companion_comparison(results)

    summary_columns = [
        "method_short",
        "frozen_normal_evaluation_run_far",
        "run_level_detection_rate__original_final_full_960",
        "post_fault_point_detection_rate__original_onset_horizon_matched",
        "median_alarm_delay_min__original_onset_horizon_matched",
        "run_level_detection_rate__faulty_training_companion",
        "post_fault_point_detection_rate__faulty_training_companion",
        "median_alarm_delay_min__faulty_training_companion",
    ]
    print("\n" + "=" * 80)
    print("Step 17 summary")
    print("=" * 80)
    print(comparison[summary_columns].to_string(index=False))
    print(
        f"\nAll {len(parity)} frozen reconstruction parity checks passed."
    )
    if data_audit["prefault_identical_to_pca_training"]:
        print(
            "CAUTION: Companion pre-fault rows are identical to same-numbered "
            "PCA-training normal runs. Describe this as an untouched companion "
            "fault validation, not a fully independent external dataset."
        )

    config = {
        "script": "scripts/17_fault3_companion_frozen_validation.py",
        "purpose": (
            "No-retuning validation on Fault 3 from TEP_Faulty_Training.RData"
        ),
        "interpretation": (
            "untouched companion-set fault validation; same TEP simulator, "
            "with pre-fault provenance explicitly audited"
        ),
        "fault_id": FAULT_ID,
        "companion_fault_start_sample": COMPANION_FAULT_START,
        "companion_samples_per_run": COMPANION_LAST_SAMPLE,
        "matched_original_segment": {
            "source_samples": [
                MATCHED_ORIGINAL_FIRST_SAMPLE,
                MATCHED_ORIGINAL_LAST_SAMPLE,
            ],
            "relabeled_samples": [1, COMPANION_LAST_SAMPLE],
            "fault_start_after_relabeling": COMPANION_FAULT_START,
            "cusum_reset_at_segment_start": True,
        },
        "frozen_from_step15": {
            "method_definitions": True,
            "selected_variables": definitions["frozen_top5_variables"],
            "selected_signs": definitions["frozen_top5_signs"],
            "top_k": PRIMARY_TOP_K,
            "proposed_cusum_k": PRIMARY_CUSUM_K,
            "competitor_k_values": {
                key: (
                    None
                    if not sequential
                    else float(step15_results.loc[key, "cusum_k"])
                )
                for key, _, sequential in METHOD_SPECS
            },
            "control_limit_source": str(
                STEP15_RESULTS_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "faulty_training_used_for_any_tuning": False,
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "data_audit": data_audit,
        "all_parity_checks_passed": bool(parity["passed"].all()),
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "fault3_companion_frozen_validation.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved configuration: {config_path}")
    print("Step 17 finished successfully.")


if __name__ == "__main__":
    main()
