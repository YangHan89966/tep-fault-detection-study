from pathlib import Path
import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULT_DIR = PROJECT_ROOT / "results" / "logs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "faultfree_training": RAW_DIR / "TEP_FaultFree_Training.RData",
    "faultfree_testing": RAW_DIR / "TEP_FaultFree_Testing.RData",
    "faulty_testing": RAW_DIR / "TEP_Faulty_Testing.RData",
}


def read_rdata_first_object(path: Path) -> pd.DataFrame:
    """Read the first object from an RData file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nReading: {path.name}")
    result = pyreadr.read_r(str(path))

    print(f"Objects inside {path.name}: {list(result.keys())}")

    if len(result) == 0:
        raise ValueError(f"No object found in {path}")

    first_key = list(result.keys())[0]
    df = result[first_key]

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The first object in {path.name} is not a pandas DataFrame.")

    return df


def basic_check(df: pd.DataFrame, name: str):
    """Print and return basic information of a dataframe."""
    print("\n" + "=" * 80)
    print(f"Dataset: {name}")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print(f"First 10 columns: {list(df.columns[:10])}")
    print(f"Last 10 columns: {list(df.columns[-10:])}")

    expected_id_cols = ["faultNumber", "simulationRun", "sample"]
    xmeas_cols = [f"xmeas_{i}" for i in range(1, 42)]
    xmv_cols = [f"xmv_{i}" for i in range(1, 12)]
    expected_var_cols = xmeas_cols + xmv_cols

    missing_id_cols = [c for c in expected_id_cols if c not in df.columns]
    missing_var_cols = [c for c in expected_var_cols if c not in df.columns]

    print(f"Missing ID columns: {missing_id_cols}")
    print(f"Missing variable columns: {missing_var_cols}")

    if all(c in df.columns for c in expected_id_cols):
        print(f"faultNumber unique values: {sorted(df['faultNumber'].unique())[:30]}")
        print(f"Number of simulationRun: {df['simulationRun'].nunique()}")
        print(f"sample min-max: {df['sample'].min()} - {df['sample'].max()}")

        run_lengths = df.groupby(["faultNumber", "simulationRun"])["sample"].nunique()
        print("Run length summary:")
        print(run_lengths.describe())

    return {
        "dataset": name,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_fault_numbers": df["faultNumber"].nunique() if "faultNumber" in df.columns else None,
        "n_runs": df["simulationRun"].nunique() if "simulationRun" in df.columns else None,
        "sample_min": df["sample"].min() if "sample" in df.columns else None,
        "sample_max": df["sample"].max() if "sample" in df.columns else None,
        "missing_id_cols": ", ".join(missing_id_cols),
        "missing_var_cols": ", ".join(missing_var_cols),
    }


def fault3_check(df_faulty: pd.DataFrame):
    """Check Fault 3 subset in faulty testing data."""
    print("\n" + "=" * 80)
    print("Fault 3 / IDV(3) check")
    print("=" * 80)

    if "faultNumber" not in df_faulty.columns:
        raise ValueError("faultNumber column not found.")

    fault3 = df_faulty[df_faulty["faultNumber"] == 3].copy()

    print(f"Fault 3 shape: {fault3.shape}")
    print(f"Fault 3 simulation runs: {fault3['simulationRun'].nunique()}")
    print(f"Fault 3 sample min-max: {fault3['sample'].min()} - {fault3['sample'].max()}")

    run_lengths = fault3.groupby("simulationRun")["sample"].nunique()
    print("Fault 3 run length summary:")
    print(run_lengths.describe())

    dev = fault3[fault3["simulationRun"].between(1, 250)]
    test = fault3[fault3["simulationRun"].between(251, 500)]

    print("\nLeakage-free split check:")
    print(f"Fault 3 development runs 1-250 shape: {dev.shape}")
    print(f"Fault 3 final test runs 251-500 shape: {test.shape}")
    print(f"Development run count: {dev['simulationRun'].nunique()}")
    print(f"Final test run count: {test['simulationRun'].nunique()}")

    fault_start_sample = 161
    pre_fault = fault3[fault3["sample"] < fault_start_sample]
    post_fault = fault3[fault3["sample"] >= fault_start_sample]

    print("\nFault injection setting:")
    print(f"Assumed fault start sample: {fault_start_sample}")
    print(f"Pre-fault rows: {pre_fault.shape[0]}")
    print(f"Post-fault rows: {post_fault.shape[0]}")

    summary = {
        "fault3_rows": fault3.shape[0],
        "fault3_cols": fault3.shape[1],
        "fault3_runs": fault3["simulationRun"].nunique(),
        "fault3_sample_min": fault3["sample"].min(),
        "fault3_sample_max": fault3["sample"].max(),
        "fault3_dev_rows": dev.shape[0],
        "fault3_test_rows": test.shape[0],
        "fault3_dev_runs": dev["simulationRun"].nunique(),
        "fault3_test_runs": test["simulationRun"].nunique(),
        "fault_start_sample": fault_start_sample,
        "pre_fault_rows": pre_fault.shape[0],
        "post_fault_rows": post_fault.shape[0],
    }

    return summary


def main():
    summaries = []

    # Read and check fault-free training data.
    df_train = read_rdata_first_object(FILES["faultfree_training"])
    summaries.append(basic_check(df_train, "faultfree_training"))

    # Read and check fault-free testing data.
    df_normal_test = read_rdata_first_object(FILES["faultfree_testing"])
    summaries.append(basic_check(df_normal_test, "faultfree_testing"))

    # Read and check faulty testing data.
    df_faulty_test = read_rdata_first_object(FILES["faulty_testing"])
    summaries.append(basic_check(df_faulty_test, "faulty_testing"))

    # Fault 3 specific check.
    fault3_summary = fault3_check(df_faulty_test)

    # Save summary files.
    summary_df = pd.DataFrame(summaries)
    summary_path = RESULT_DIR / "01_raw_data_basic_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fault3_summary_path = RESULT_DIR / "01_fault3_data_summary.csv"
    pd.DataFrame([fault3_summary]).to_csv(
        fault3_summary_path, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("Saved reports")
    print("=" * 80)
    print(summary_path)
    print(fault3_summary_path)

    print("\nRaw data check finished successfully.")


if __name__ == "__main__":
    main()