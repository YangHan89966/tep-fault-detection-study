from pathlib import Path
import json
import gc

import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


RAW_FILES = {
    "faultfree_training": RAW_DIR / "TEP_FaultFree_Training.RData",
    "faultfree_testing": RAW_DIR / "TEP_FaultFree_Testing.RData",
    "faulty_testing": RAW_DIR / "TEP_Faulty_Testing.RData",
}


def read_rdata_first_object(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nReading: {path.name}")
    result = pyreadr.read_r(str(path))

    if len(result) == 0:
        raise ValueError(f"No object found in {path.name}")

    first_key = list(result.keys())[0]
    print(f"Object name: {first_key}")

    df = result[first_key]

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Object in {path.name} is not a pandas DataFrame.")

    return df


def get_variable_columns(df: pd.DataFrame):
    xmeas_cols = [f"xmeas_{i}" for i in range(1, 42)]
    xmv_cols = [f"xmv_{i}" for i in range(1, 12)]
    variable_cols = xmeas_cols + xmv_cols

    missing = [c for c in variable_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing variable columns: {missing}")

    return xmeas_cols, xmv_cols, variable_cols


def save_pickle(df: pd.DataFrame, filename: str):
    path = PROCESSED_DIR / filename
    df.to_pickle(path)
    print(f"Saved: {path} | shape={df.shape}")
    return path


def make_summary(name: str, df: pd.DataFrame):
    return {
        "dataset": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "fault_numbers": ",".join(map(str, sorted(df["faultNumber"].unique()))),
        "simulation_runs": df["simulationRun"].nunique(),
        "run_min": int(df["simulationRun"].min()),
        "run_max": int(df["simulationRun"].max()),
        "sample_min": int(df["sample"].min()),
        "sample_max": int(df["sample"].max()),
    }


def main():
    print("=" * 80)
    print("Step 02: Prepare leakage-free Fault 3 datasets")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Read fault-free training data.
    # ------------------------------------------------------------------
    df_train = read_rdata_first_object(RAW_FILES["faultfree_training"])
    xmeas_cols, xmv_cols, variable_cols = get_variable_columns(df_train)

    id_cols = ["faultNumber", "simulationRun", "sample"]
    keep_cols = id_cols + variable_cols

    df_train = df_train[keep_cols].copy()

    # ------------------------------------------------------------------
    # 2. Read fault-free testing data and split into validation/evaluation.
    # ------------------------------------------------------------------
    df_normal_test = read_rdata_first_object(RAW_FILES["faultfree_testing"])
    df_normal_test = df_normal_test[keep_cols].copy()

    normal_validation = df_normal_test[
        df_normal_test["simulationRun"].between(1, 250)
    ].copy()

    normal_evaluation = df_normal_test[
        df_normal_test["simulationRun"].between(251, 500)
    ].copy()

    del df_normal_test
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Read faulty testing data and extract Fault 3.
    # ------------------------------------------------------------------
    df_faulty_test = read_rdata_first_object(RAW_FILES["faulty_testing"])
    df_fault3 = df_faulty_test[df_faulty_test["faultNumber"] == 3][keep_cols].copy()

    del df_faulty_test
    gc.collect()

    fault3_development = df_fault3[
        df_fault3["simulationRun"].between(1, 250)
    ].copy()

    fault3_final_test = df_fault3[
        df_fault3["simulationRun"].between(251, 500)
    ].copy()

    del df_fault3
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Save processed datasets.
    # ------------------------------------------------------------------
    print("\nSaving processed datasets...")

    saved_files = {
        "normal_train": str(save_pickle(df_train, "normal_train.pkl")),
        "normal_validation_runs_1_250": str(
            save_pickle(normal_validation, "normal_validation_runs_1_250.pkl")
        ),
        "normal_evaluation_runs_251_500": str(
            save_pickle(normal_evaluation, "normal_evaluation_runs_251_500.pkl")
        ),
        "fault3_development_runs_1_250": str(
            save_pickle(fault3_development, "fault3_development_runs_1_250.pkl")
        ),
        "fault3_final_test_runs_251_500": str(
            save_pickle(fault3_final_test, "fault3_final_test_runs_251_500.pkl")
        ),
    }

    # ------------------------------------------------------------------
    # 5. Save variable list.
    # ------------------------------------------------------------------
    variable_info = pd.DataFrame(
        {
            "variable": variable_cols,
            "type": ["xmeas"] * len(xmeas_cols) + ["xmv"] * len(xmv_cols),
            "index": list(range(1, 42)) + list(range(1, 12)),
        }
    )

    variable_info_path = TABLE_DIR / "02_variable_list.csv"
    variable_info.to_csv(variable_info_path, index=False, encoding="utf-8-sig")
    print(f"Saved variable list: {variable_info_path}")

    # ------------------------------------------------------------------
    # 6. Save split summary.
    # ------------------------------------------------------------------
    summaries = [
        make_summary("normal_train", df_train),
        make_summary("normal_validation_runs_1_250", normal_validation),
        make_summary("normal_evaluation_runs_251_500", normal_evaluation),
        make_summary("fault3_development_runs_1_250", fault3_development),
        make_summary("fault3_final_test_runs_251_500", fault3_final_test),
    ]

    summary_df = pd.DataFrame(summaries)
    summary_path = LOG_DIR / "02_processed_data_split_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved split summary: {summary_path}")

    # ------------------------------------------------------------------
    # 7. Save experiment config.
    # ------------------------------------------------------------------
    config = {
        "project": "TEP Fault 3 local signed residual CUSUM study",
        "fault_id": 3,
        "fault_start_sample": 161,
        "sampling_interval_min": 3,
        "n_id_columns": 3,
        "id_columns": id_cols,
        "n_process_variables": 52,
        "xmeas_columns": xmeas_cols,
        "xmv_columns": xmv_cols,
        "variable_columns": variable_cols,
        "normal_train": {
            "source": "TEP_FaultFree_Training.RData",
            "runs": "1-500",
            "samples": "1-500",
            "purpose": "PCA modeling only",
        },
        "normal_validation": {
            "source": "TEP_FaultFree_Testing.RData",
            "runs": "1-250",
            "samples": "1-960",
            "purpose": "threshold selection and validation",
        },
        "normal_evaluation": {
            "source": "TEP_FaultFree_Testing.RData",
            "runs": "251-500",
            "samples": "1-960",
            "purpose": "final normal false alarm evaluation",
        },
        "fault3_development": {
            "source": "TEP_Faulty_Testing.RData",
            "faultNumber": 3,
            "runs": "1-250",
            "samples": "1-960",
            "purpose": "residual variable screening and method development",
        },
        "fault3_final_test": {
            "source": "TEP_Faulty_Testing.RData",
            "faultNumber": 3,
            "runs": "251-500",
            "samples": "1-960",
            "purpose": "final leakage-free Fault 3 testing",
        },
        "processed_files": saved_files,
    }

    config_path = CONFIG_DIR / "fault3_data_split_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Saved config: {config_path}")

    print("\n" + "=" * 80)
    print("Step 02 finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()