"""Step 27: Frozen-results consistency audit and manuscript claim registry.

Scientific role
---------------
This step is a publication-packaging audit, not another experiment.  It reads
only stored Step-22--Step-26 result tables.  It does not load TEP process data,
fit a model, select a variable, tune a CUSUM parameter, calibrate a threshold,
or recompute any performance endpoint.

The audit has three purposes:

1. verify that the frozen per-fault results, post-freeze robustness analyses,
   detector-bank diagnostic, and exploratory joint gate remain numerically
   and semantically distinct;
2. fail loudly if a manuscript-facing number no longer agrees with its source;
3. create a compact claim registry, reporting guardrails, and placement plan
   for assembling manuscript version 21.

Place this file in the project ``scripts`` directory and run:

    python scripts/27_frozen_results_consistency_audit.py --run

The script intentionally has no registration stage because it performs no
prospective performance analysis.  Its input hashes and all checks are saved
in the final audit configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"
CONFIG_DIR = PROJECT_ROOT / "configs"

for directory in (TABLE_DIR, FIGURE_DIR, REPORT_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


RESERVED_FAULTS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
PROPOSED_KEY = "frozen_multievidence_top5_cusum"
TOP1_KEY = "chosen_family_top1_cusum"
TOLERANCE = 1e-10


INPUT_PATHS = {
    "step22_results": TABLE_DIR / "22_reserved_multifault_results.csv",
    "step22_config": CONFIG_DIR / "22_reserved_multifault_frozen_validation.json",
    "step23_primary": TABLE_DIR / "23_reserved_primary_summary_with_ci.csv",
    "step23_guardrails": TABLE_DIR / "23_reporting_guardrails.csv",
    "step23b_by_fault": TABLE_DIR / "23b_run_detection_and_allrun_delay_by_fault.csv",
    "step23b_delay_macro": TABLE_DIR / "23b_allrun_delay_macro_summary.csv",
    "step23b_far": TABLE_DIR / "23b_crossed_normal_far_bootstrap.csv",
    "step23b_integrity": TABLE_DIR / "23b_frozen_run_result_integrity_audit.csv",
    "step24_parity": TABLE_DIR / "24_clean_reconstruction_parity.csv",
    "step24_loss": TABLE_DIR / "24_channel_loss_crossfault_summary.csv",
    "step24_contamination": TABLE_DIR / "24_channel_contamination_crossfault_summary.csv",
    "step24_contrasts": TABLE_DIR / "24_top5_vs_top1_robustness_contrasts.csv",
    "step24_guardrails": TABLE_DIR / "24_reporting_guardrails.csv",
    "step25_parity": TABLE_DIR / "25_monitor_clean_reconstruction_parity.csv",
    "step25_bank_by_fault": TABLE_DIR / "25_bank_detection_by_fault.csv",
    "step25_summary": TABLE_DIR / "25_detector_bank_summary.csv",
    "step25_guardrails": TABLE_DIR / "25_reporting_guardrails.csv",
    "step26_by_fault": TABLE_DIR / "26_joint_gate_by_fault.csv",
    "step26_comparison": TABLE_DIR / "26_joint_gate_comparison_with_step25.csv",
    "step26_summary": TABLE_DIR / "26_joint_gate_summary.csv",
    "step26_guardrails": TABLE_DIR / "26_reporting_guardrails.csv",
}


OUTPUT_PATHS = {
    "checks": TABLE_DIR / "27_cross_step_consistency_checks.csv",
    "claims": TABLE_DIR / "27_manuscript_claim_registry.csv",
    "numbers": TABLE_DIR / "27_manuscript_numbers.csv",
    "placement": TABLE_DIR / "27_figure_table_placement_plan.csv",
    "guardrails": TABLE_DIR / "27_reporting_guardrails.csv",
    "manifest": TABLE_DIR / "27_input_output_manifest.csv",
    "report": REPORT_DIR / "27_manuscript_claim_sheet.md",
    "config": CONFIG_DIR / "27_frozen_results_consistency_audit.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def read_csv(key: str) -> pd.DataFrame:
    path = require_file(INPUT_PATHS[key], key)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{key} is empty: {path}")
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def as_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        bad = sorted(normalized[parsed.isna()].unique().tolist())
        raise ValueError(f"{label} contains non-boolean values: {bad}")
    return parsed.astype(bool)


def numeric(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains missing or non-numeric values")
    return parsed.astype(float)


def close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance))


def all_close(left: Any, right: Any, tolerance: float = TOLERANCE) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        )
    )


def format_value(value: Any) -> str:
    if isinstance(value, (np.bool_, bool)):
        return "True" if bool(value) else "False"
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "NaN"
        return f"{float(value):.12g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return ";".join(format_value(item) for item in list(value))
    return str(value)


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        scope: str,
        statement: str,
        passed: bool,
        observed: Any,
        expected: Any,
        source_keys: str,
        severity: str = "error",
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "scope": scope,
                "statement": statement,
                "passed": bool(passed),
                "severity": severity,
                "observed": format_value(observed),
                "expected": format_value(expected),
                "source_keys": source_keys,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def select_one(frame: pd.DataFrame, label: str, **conditions: Any) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        if column not in frame.columns:
            raise ValueError(f"{label}: missing selection column {column}")
        if isinstance(value, float):
            mask &= np.isclose(
                pd.to_numeric(frame[column], errors="coerce"), value, rtol=0, atol=TOLERANCE
            )
        else:
            mask &= frame[column] == value
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(
            f"{label}: expected one row for {conditions}, found {len(selected)}"
        )
    return selected.iloc[0]


def add_number(
    rows: list[dict[str, Any]],
    number_id: str,
    analysis_layer: str,
    endpoint: str,
    estimate: float,
    unit: str,
    source: str,
    ci_low: float | None = None,
    ci_high: float | None = None,
    manuscript_role: str = "main text",
) -> None:
    rows.append(
        {
            "number_id": number_id,
            "analysis_layer": analysis_layer,
            "endpoint": endpoint,
            "estimate": estimate,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "unit": unit,
            "source": source,
            "manuscript_role": manuscript_role,
        }
    )


def audit_step22_23(
    audit: Audit, frames: dict[str, pd.DataFrame], number_rows: list[dict[str, Any]]
) -> dict[str, float]:
    step22 = frames["step22_results"]
    require_columns(
        step22,
        [
            "fault_id",
            "method_key",
            "dataset",
            "normal_run_level_far",
            "fault_run_level_detection_rate",
            "fault_post_fault_point_detection_rate",
            "median_alarm_delay_min",
        ],
        "Step 22 results",
    )
    final = step22[
        (step22["dataset"] == "faulty_testing_final_full_960")
        & (step22["method_key"] == PROPOSED_KEY)
    ].copy()
    faults = sorted(pd.to_numeric(final["fault_id"]).astype(int).tolist())
    audit.add(
        "S22-01",
        "primary frozen validation",
        "The Proposed Top-5 final table contains exactly the 17 reserved faults.",
        faults == RESERVED_FAULTS,
        faults,
        RESERVED_FAULTS,
        "step22_results",
    )
    run_detection = numeric(
        final["fault_run_level_detection_rate"], "Step 22 run detection"
    )
    audit.add(
        "S22-02",
        "primary frozen validation",
        "Every reserved fault has 100% run-level detection.",
        len(final) == 17 and all_close(run_detection, 1.0),
        run_detection.min() if len(run_detection) else np.nan,
        1.0,
        "step22_results",
    )

    mean_far = float(numeric(final["normal_run_level_far"], "Step 22 FAR").mean())
    mean_point = float(
        numeric(
            final["fault_post_fault_point_detection_rate"], "Step 22 point detection"
        ).mean()
    )
    median_delay = float(
        numeric(final["median_alarm_delay_min"], "Step 22 delay").median()
    )

    step23 = frames["step23_primary"]
    require_columns(
        step23,
        ["method_key", "endpoint_key", "estimate", "ci_95_low", "ci_95_high"],
        "Step 23 primary summary",
    )
    s23_far = select_one(
        step23, "Step 23 proposed FAR", method_key=PROPOSED_KEY, endpoint_key="normal_run_far"
    )
    s23_run = select_one(
        step23,
        "Step 23 proposed run detection",
        method_key=PROPOSED_KEY,
        endpoint_key="run_detection_mean",
    )
    s23_point = select_one(
        step23,
        "Step 23 proposed point detection",
        method_key=PROPOSED_KEY,
        endpoint_key="point_detection_mean",
    )
    s23_delay = select_one(
        step23,
        "Step 23 proposed delay",
        method_key=PROPOSED_KEY,
        endpoint_key="alarm_delay_median",
    )
    comparisons = [
        ("S23-01", float(s23_far["estimate"]) / 100.0, mean_far, "macro FAR"),
        ("S23-02", float(s23_run["estimate"]) / 100.0, float(run_detection.mean()), "run detection"),
        ("S23-03", float(s23_point["estimate"]) / 100.0, mean_point, "point detection"),
        ("S23-04", float(s23_delay["estimate"]), median_delay, "median delay"),
    ]
    for check_id, observed, expected, endpoint in comparisons:
        audit.add(
            check_id,
            "post-freeze statistical synthesis",
            f"Step 23 {endpoint} reproduces the Step 22 frozen final table.",
            close(observed, expected),
            observed,
            expected,
            "step22_results;step23_primary",
        )

    step23b_fault = frames["step23b_by_fault"]
    require_columns(
        step23b_fault,
        [
            "fault_id",
            "method_key",
            "n_runs",
            "detected_runs",
            "run_detection_exact_ci_95_low",
        ],
        "Step 23B by-fault sensitivity",
    )
    s23b_proposed = step23b_fault[step23b_fault["method_key"] == PROPOSED_KEY]
    audit.add(
        "S23B-01",
        "primary sensitivity analysis",
        "All 17 proposed-method cells contain 250 final runs and 250 detections.",
        len(s23b_proposed) == 17
        and all_close(numeric(s23b_proposed["n_runs"], "Step 23B n_runs"), 250)
        and all_close(
            numeric(s23b_proposed["detected_runs"], "Step 23B detected_runs"), 250
        ),
        len(s23b_proposed),
        17,
        "step23b_by_fault",
    )
    integrity = frames["step23b_integrity"]
    require_columns(integrity, ["check", "passed"], "Step 23B integrity audit")
    integrity_pass = as_bool(integrity["passed"], "Step 23B integrity checks")
    audit.add(
        "S23B-03",
        "primary sensitivity analysis",
        "Every frozen Step-22 run-result integrity check passed before Step 23B synthesis.",
        bool(integrity_pass.all()),
        int(integrity_pass.sum()),
        len(integrity),
        "step23b_integrity",
    )
    exact_low = float(
        numeric(
            s23b_proposed["run_detection_exact_ci_95_low"], "Step 23B exact CI"
        ).min()
    )

    step23b_far = frames["step23b_far"]
    require_columns(
        step23b_far,
        ["method_key", "macro_normal_run_far", "crossed_ci_95_low", "crossed_ci_95_high"],
        "Step 23B FAR bootstrap",
    )
    far_row = select_one(step23b_far, "Step 23B proposed FAR", method_key=PROPOSED_KEY)
    audit.add(
        "S23B-02",
        "primary sensitivity analysis",
        "The crossed-bootstrap macro FAR estimate equals the Step 22 macro FAR.",
        close(float(far_row["macro_normal_run_far"]), mean_far),
        far_row["macro_normal_run_far"],
        mean_far,
        "step22_results;step23b_far",
    )

    delay_macro = frames["step23b_delay_macro"]
    require_columns(
        delay_macro,
        ["method_key", "endpoint_key", "estimate", "ci_95_low", "ci_95_high"],
        "Step 23B delay macro summary",
    )
    penalized_delay = select_one(
        delay_macro,
        "Step 23B proposed all-run delay",
        method_key=PROPOSED_KEY,
        endpoint_key="allrun_penalized_delay_median",
    )

    add_number(
        number_rows,
        "primary_macro_normal_far",
        "Step 22/23B primary frozen per-fault validation",
        "Macro normal run-level FAR",
        100.0 * float(far_row["macro_normal_run_far"]),
        "percent",
        "23b_crossed_normal_far_bootstrap.csv",
        100.0 * float(far_row["crossed_ci_95_low"]),
        100.0 * float(far_row["crossed_ci_95_high"]),
    )
    add_number(
        number_rows,
        "primary_macro_run_detection",
        "Step 22/23B primary frozen per-fault validation",
        "Macro mean run-level detection",
        100.0 * float(run_detection.mean()),
        "percent",
        "22_reserved_multifault_results.csv",
    )
    add_number(
        number_rows,
        "primary_macro_point_detection",
        "Step 22/23 primary frozen per-fault validation",
        "Macro mean post-fault point detection",
        100.0 * mean_point,
        "percent",
        "23_reserved_primary_summary_with_ci.csv",
        float(s23_point["ci_95_low"]),
        float(s23_point["ci_95_high"]),
    )
    add_number(
        number_rows,
        "primary_macro_median_allrun_delay",
        "Step 23B primary sensitivity analysis",
        "Macro median all-run penalized delay",
        float(penalized_delay["estimate"]),
        "minutes",
        "23b_allrun_delay_macro_summary.csv",
        float(penalized_delay["ci_95_low"]),
        float(penalized_delay["ci_95_high"]),
    )
    add_number(
        number_rows,
        "primary_fault_specific_detection_lower_bound",
        "Step 23B primary sensitivity analysis",
        "Minimum exact 95% lower bound for a 250/250 fault-specific result",
        100.0 * exact_low,
        "percent",
        "23b_run_detection_and_allrun_delay_by_fault.csv",
    )
    return {
        "primary_far": mean_far,
        "primary_point": mean_point,
        "primary_delay": float(penalized_delay["estimate"]),
    }


def audit_step24(
    audit: Audit, frames: dict[str, pd.DataFrame], number_rows: list[dict[str, Any]]
) -> None:
    parity = frames["step24_parity"]
    require_columns(
        parity,
        ["fault_id", "method_key", "all_step22_metrics_exact_with_tolerance_1e_12"],
        "Step 24 clean parity",
    )
    parity_pass = as_bool(
        parity["all_step22_metrics_exact_with_tolerance_1e_12"], "Step 24 parity"
    )
    audit.add(
        "S24-01",
        "post-freeze robustness",
        "All 34 clean Top-1/Top-5 reconstructions reproduce Step 22 exactly.",
        len(parity) == 34 and bool(parity_pass.all()),
        int(parity_pass.sum()),
        34,
        "step24_parity",
    )

    loss = frames["step24_loss"]
    require_columns(
        loss,
        [
            "method_key",
            "channel_aggregation",
            "macro_mean_fault_run_level_detection_rate",
            "macro_mean_allrun_penalized_mean_delay_min",
            "faults_with_250_of_250_detected",
        ],
        "Step 24 loss summary",
    )
    top1_loss = select_one(
        loss, "Step 24 Top-1 loss worst", method_key=TOP1_KEY, channel_aggregation="worst"
    )
    top5_loss = select_one(
        loss,
        "Step 24 Top-5 loss worst",
        method_key=PROPOSED_KEY,
        channel_aggregation="worst",
    )
    audit.add(
        "S24-02",
        "post-freeze robustness",
        "Single-channel loss disables every Top-1 detector but no Top-5 detector.",
        close(float(top1_loss["macro_mean_fault_run_level_detection_rate"]), 0.0)
        and int(top1_loss["faults_with_250_of_250_detected"]) == 0
        and close(float(top5_loss["macro_mean_fault_run_level_detection_rate"]), 1.0)
        and int(top5_loss["faults_with_250_of_250_detected"]) == 17,
        [
            top1_loss["faults_with_250_of_250_detected"],
            top5_loss["faults_with_250_of_250_detected"],
        ],
        [0, 17],
        "step24_loss",
    )

    contamination = frames["step24_contamination"]
    require_columns(
        contamination,
        [
            "contamination_sigma",
            "method_key",
            "channel_aggregation",
            "macro_mean_normal_run_level_far",
            "macro_mean_fault_run_level_detection_rate",
            "macro_mean_allrun_penalized_mean_delay_min",
            "faults_with_250_of_250_detected",
        ],
        "Step 24 contamination summary",
    )
    top1_contam = select_one(
        contamination,
        "Step 24 Top-1 sigma=1 mean",
        method_key=TOP1_KEY,
        channel_aggregation="mean",
        contamination_sigma=1.0,
    )
    top5_contam_mean = select_one(
        contamination,
        "Step 24 Top-5 sigma=1 mean",
        method_key=PROPOSED_KEY,
        channel_aggregation="mean",
        contamination_sigma=1.0,
    )
    top5_contam_worst = select_one(
        contamination,
        "Step 24 Top-5 sigma=1 worst",
        method_key=PROPOSED_KEY,
        channel_aggregation="worst",
        contamination_sigma=1.0,
    )
    audit.add(
        "S24-03",
        "post-freeze robustness",
        "At sigma=1, Top-5 retains all 17 fault detections under mean and worst channel aggregation.",
        int(top5_contam_mean["faults_with_250_of_250_detected"]) == 17
        and int(top5_contam_worst["faults_with_250_of_250_detected"]) == 17
        and close(
            float(top5_contam_mean["macro_mean_fault_run_level_detection_rate"]), 1.0
        )
        and close(
            float(top5_contam_worst["macro_mean_fault_run_level_detection_rate"]), 1.0
        ),
        [
            top5_contam_mean["faults_with_250_of_250_detected"],
            top5_contam_worst["faults_with_250_of_250_detected"],
        ],
        [17, 17],
        "step24_contamination",
    )

    add_number(
        number_rows,
        "robustness_top1_loss_faults_detected",
        "Step 24 post-freeze evidence-channel loss",
        "Faults with 250/250 runs detected after sole Top-1 channel loss",
        float(top1_loss["faults_with_250_of_250_detected"]),
        "faults of 17",
        "24_channel_loss_crossfault_summary.csv",
        manuscript_role="main text or supplement",
    )
    add_number(
        number_rows,
        "robustness_top5_loss_faults_detected",
        "Step 24 post-freeze evidence-channel loss",
        "Faults with 250/250 runs detected under worst Top-5 channel loss",
        float(top5_loss["faults_with_250_of_250_detected"]),
        "faults of 17",
        "24_channel_loss_crossfault_summary.csv",
        manuscript_role="main text or supplement",
    )
    add_number(
        number_rows,
        "robustness_top1_sigma1_far",
        "Step 24 post-freeze evidence-channel contamination",
        "Top-1 macro mean normal run-level FAR at sigma=1",
        100.0 * float(top1_contam["macro_mean_normal_run_level_far"]),
        "percent",
        "24_channel_contamination_crossfault_summary.csv",
        manuscript_role="main text or supplement",
    )
    add_number(
        number_rows,
        "robustness_top5_sigma1_far_mean",
        "Step 24 post-freeze evidence-channel contamination",
        "Top-5 macro mean normal run-level FAR at sigma=1",
        100.0 * float(top5_contam_mean["macro_mean_normal_run_level_far"]),
        "percent",
        "24_channel_contamination_crossfault_summary.csv",
        manuscript_role="main text or supplement",
    )
    add_number(
        number_rows,
        "robustness_top5_sigma1_far_worst",
        "Step 24 post-freeze evidence-channel contamination",
        "Top-5 worst-channel macro mean normal run-level FAR at sigma=1",
        100.0 * float(top5_contam_worst["macro_mean_normal_run_level_far"]),
        "percent",
        "24_channel_contamination_crossfault_summary.csv",
        manuscript_role="main text or supplement",
    )


def audit_step25_26(
    audit: Audit, frames: dict[str, pd.DataFrame], number_rows: list[dict[str, Any]]
) -> None:
    parity = frames["step25_parity"]
    require_columns(
        parity,
        [
            "normal_parity_exact_with_tolerance_1e_12",
            "target_final_parity_exact_with_tolerance_1e_12",
        ],
        "Step 25 parity",
    )
    normal_pass = as_bool(
        parity["normal_parity_exact_with_tolerance_1e_12"], "Step 25 normal parity"
    )
    final_pass = as_bool(
        parity["target_final_parity_exact_with_tolerance_1e_12"],
        "Step 25 final parity",
    )
    audit.add(
        "S25-01",
        "post-freeze detector-bank diagnostic",
        "All 17 bank monitors reproduce their frozen normal and target-fault results.",
        len(parity) == 17 and bool(normal_pass.all()) and bool(final_pass.all()),
        int((normal_pass & final_pass).sum()),
        17,
        "step25_parity",
    )

    s25 = frames["step25_summary"].iloc[0]
    required_s25 = [
        "normal_bank_run_level_far",
        "macro_mean_bank_run_detection",
        "macro_mean_target_among_earliest_rate",
        "macro_median_target_earliest_rank",
        "n_monitors",
        "n_actual_faults",
    ]
    require_columns(frames["step25_summary"], required_s25, "Step 25 summary")
    bank_fault = frames["step25_bank_by_fault"]
    require_columns(
        bank_fault,
        ["actual_fault_id", "bank_run_detection_rate", "bank_allrun_penalized_mean_delay_min"],
        "Step 25 bank by fault",
    )
    faults = sorted(pd.to_numeric(bank_fault["actual_fault_id"]).astype(int).tolist())
    audit.add(
        "S25-02",
        "post-freeze detector-bank diagnostic",
        "The OR bank covers all 17 reserved faults but has 38% normal run FAR.",
        faults == RESERVED_FAULTS
        and all_close(numeric(bank_fault["bank_run_detection_rate"], "Step 25 bank DR"), 1.0)
        and close(float(s25["normal_bank_run_level_far"]), 0.38),
        [s25["normal_bank_run_level_far"], s25["macro_mean_bank_run_detection"]],
        [0.38, 1.0],
        "step25_summary;step25_bank_by_fault",
    )

    s26 = frames["step26_summary"].iloc[0]
    required_s26 = [
        "analysis_status",
        "n_monitors",
        "n_reserved_faults",
        "normal_validation_run_far",
        "normal_evaluation_run_far",
        "normal_evaluation_run_far_ci_low",
        "normal_evaluation_run_far_ci_high",
        "step25_or_bank_normal_run_far",
        "faults_with_250_of_250_detected",
        "macro_mean_run_detection",
        "macro_mean_post_fault_point_detection",
        "macro_median_allrun_penalized_mean_delay_min",
        "macro_mean_target_top3_at_gate_rate",
        "normal_step25_parity_max_abs_difference",
        "normal_validation_local_far_parity_max_abs_difference",
        "crossfault_step25_parity_max_abs_difference",
        "normal_evaluation_used_for_calibration",
        "fault_final_used_for_calibration",
        "confirmatory_inference",
    ]
    require_columns(frames["step26_summary"], required_s26, "Step 26 summary")
    s26_by_fault = frames["step26_by_fault"]
    require_columns(
        s26_by_fault,
        [
            "actual_fault_id",
            "detected_runs",
            "run_detection_rate",
            "post_fault_point_detection_rate",
            "allrun_penalized_mean_delay_min",
            "target_top3_at_gate_rate_all_runs",
            "step25_or_bank_run_detection_rate",
        ],
        "Step 26 by-fault table",
    )
    s26_faults = sorted(
        pd.to_numeric(s26_by_fault["actual_fault_id"]).astype(int).tolist()
    )
    audit.add(
        "S26-01",
        "exploratory post-freeze joint gate",
        "The joint gate evaluates all 17 faults and detects all 250 runs for every fault.",
        s26_faults == RESERVED_FAULTS
        and all_close(numeric(s26_by_fault["detected_runs"], "Step 26 detected runs"), 250)
        and all_close(numeric(s26_by_fault["run_detection_rate"], "Step 26 DR"), 1.0),
        int((pd.to_numeric(s26_by_fault["detected_runs"]) == 250).sum()),
        17,
        "step26_by_fault",
    )
    audit.add(
        "S26-02",
        "exploratory post-freeze joint gate",
        "The joint gate reduces normal run FAR relative to the hypothetical Step-25 OR bank.",
        close(float(s26["step25_or_bank_normal_run_far"]), float(s25["normal_bank_run_level_far"]))
        and float(s26["normal_evaluation_run_far"]) < float(s26["step25_or_bank_normal_run_far"]),
        [s26["step25_or_bank_normal_run_far"], s26["normal_evaluation_run_far"]],
        [s25["normal_bank_run_level_far"], "joint gate lower"],
        "step25_summary;step26_summary",
    )
    zero_parity = [
        float(s26["normal_step25_parity_max_abs_difference"]),
        float(s26["normal_validation_local_far_parity_max_abs_difference"]),
        float(s26["crossfault_step25_parity_max_abs_difference"]),
    ]
    audit.add(
        "S26-03",
        "exploratory post-freeze joint gate",
        "All Step-25 reconstruction parity checks remain exact in Step 26.",
        all_close(zero_parity, 0.0),
        zero_parity,
        [0.0, 0.0, 0.0],
        "step26_summary",
    )
    calibration_flags = [
        as_bool(pd.Series([s26["normal_evaluation_used_for_calibration"]]), "Step 26 normal calibration").iloc[0],
        as_bool(pd.Series([s26["fault_final_used_for_calibration"]]), "Step 26 fault calibration").iloc[0],
        as_bool(pd.Series([s26["confirmatory_inference"]]), "Step 26 confirmation").iloc[0],
    ]
    audit.add(
        "S26-04",
        "exploratory post-freeze joint gate",
        "Neither normal evaluation nor final fault data calibrated the gate, and the analysis is not confirmatory.",
        calibration_flags == [False, False, False]
        and str(s26["analysis_status"]) == "post-freeze exploratory deployment layer",
        calibration_flags,
        [False, False, False],
        "step26_summary;step26_guardrails",
    )

    comparison = frames["step26_comparison"]
    require_columns(
        comparison,
        [
            "actual_fault_id",
            "run_detection_difference_step26_minus_step25",
            "delay_difference_step26_minus_step25_min",
        ],
        "Step 26 comparison",
    )
    audit.add(
        "S26-05",
        "exploratory post-freeze joint gate",
        "Step 26 preserves Step-25 bank run detection on all 17 faults.",
        len(comparison) == 17
        and all_close(
            numeric(
                comparison["run_detection_difference_step26_minus_step25"],
                "Step 26 detection difference",
            ),
            0.0,
        ),
        numeric(
            comparison["run_detection_difference_step26_minus_step25"],
            "Step 26 detection difference",
        ).abs().max(),
        0.0,
        "step26_comparison",
    )

    add_number(
        number_rows,
        "or_bank_normal_far",
        "Step 25 post-freeze detector-bank diagnostic",
        "Hypothetical 17-monitor OR-bank normal run-level FAR",
        100.0 * float(s25["normal_bank_run_level_far"]),
        "percent",
        "25_detector_bank_summary.csv",
        manuscript_role="limitation / supplementary diagnostic",
    )
    add_number(
        number_rows,
        "or_bank_target_among_earliest",
        "Step 25 post-freeze detector-bank diagnostic",
        "Macro mean target-monitor-among-earliest rate",
        100.0 * float(s25["macro_mean_target_among_earliest_rate"]),
        "percent",
        "25_detector_bank_summary.csv",
        manuscript_role="limitation / supplementary diagnostic",
    )
    add_number(
        number_rows,
        "or_bank_target_rank",
        "Step 25 post-freeze detector-bank diagnostic",
        "Macro median target-monitor earliest rank",
        float(s25["macro_median_target_earliest_rank"]),
        "rank of 17",
        "25_detector_bank_summary.csv",
        manuscript_role="limitation / supplementary diagnostic",
    )
    add_number(
        number_rows,
        "joint_gate_normal_far",
        "Step 26 exploratory post-freeze deployment layer",
        "Normal-evaluation joint-gate run-level FAR",
        100.0 * float(s26["normal_evaluation_run_far"]),
        "percent",
        "26_joint_gate_summary.csv",
        100.0 * float(s26["normal_evaluation_run_far_ci_low"]),
        100.0 * float(s26["normal_evaluation_run_far_ci_high"]),
        manuscript_role="secondary exploratory result",
    )
    add_number(
        number_rows,
        "joint_gate_macro_run_detection",
        "Step 26 exploratory post-freeze deployment layer",
        "Macro mean joint-gate run-level detection",
        100.0 * float(s26["macro_mean_run_detection"]),
        "percent",
        "26_joint_gate_summary.csv",
        manuscript_role="secondary exploratory result",
    )
    add_number(
        number_rows,
        "joint_gate_macro_point_detection",
        "Step 26 exploratory post-freeze deployment layer",
        "Macro mean joint-gate post-fault point detection",
        100.0 * float(s26["macro_mean_post_fault_point_detection"]),
        "percent",
        "26_joint_gate_summary.csv",
        manuscript_role="secondary exploratory result",
    )
    add_number(
        number_rows,
        "joint_gate_macro_delay",
        "Step 26 exploratory post-freeze deployment layer",
        "Macro median all-run penalized mean delay",
        float(s26["macro_median_allrun_penalized_mean_delay_min"]),
        "minutes",
        "26_joint_gate_summary.csv",
        manuscript_role="secondary exploratory result",
    )
    add_number(
        number_rows,
        "joint_gate_target_top3",
        "Step 26 exploratory post-freeze deployment layer",
        "Macro mean target-monitor Top-3 rate at gate time",
        100.0 * float(s26["macro_mean_target_top3_at_gate_rate"]),
        "percent",
        "26_joint_gate_summary.csv",
        manuscript_role="descriptive attribution diagnostic",
    )


def audit_source_guardrails(audit: Audit, frames: dict[str, pd.DataFrame]) -> None:
    step23 = frames["step23_guardrails"]
    require_columns(
        step23,
        ["guardrail_id", "manuscript_instruction"],
        "Step 23 reporting guardrails",
    )
    s23_text = " ".join(step23["manuscript_instruction"].astype(str)).lower()
    audit.add(
        "G-S23",
        "source reporting guardrails",
        "Step 23 explicitly blocks universal-superiority and external-universality claims.",
        "not universal superiority" in s23_text
        and "not external-process universality" in s23_text,
        "claim limits present" if (
            "not universal superiority" in s23_text
            and "not external-process universality" in s23_text
        ) else "claim limit missing",
        "both claim limits present",
        "step23_guardrails",
    )

    step24 = frames["step24_guardrails"]
    require_columns(
        step24,
        ["topic", "statement", "allowed_or_required"],
        "Step 24 reporting guardrails",
    )
    sensor_rows = step24[step24["topic"] == "sensor_claim"]
    sensor_blocked = (
        len(sensor_rows) == 1
        and not as_bool(
            sensor_rows["allowed_or_required"], "Step 24 sensor guardrail"
        ).iloc[0]
    )
    audit.add(
        "G-S24",
        "source reporting guardrails",
        "Step 24 explicitly blocks physical sensor-failure robustness claims.",
        sensor_blocked,
        sensor_blocked,
        True,
        "step24_guardrails",
    )

    step25 = frames["step25_guardrails"]
    require_columns(
        step25, ["topic", "statement", "required"], "Step 25 reporting guardrails"
    )
    s25_text = " ".join(step25["statement"].astype(str)).lower()
    s25_required = as_bool(step25["required"], "Step 25 required guardrails")
    bank_limits = (
        bool(s25_required.all())
        and "hypothetical" in s25_text
        and "not the step-22 primary" in s25_text
        and "not constitute a trained or validated multiclass diagnosis rule" in s25_text
    )
    audit.add(
        "G-S25",
        "source reporting guardrails",
        "Step 25 retains the hypothetical-OR and no-validated-diagnosis limitations.",
        bank_limits,
        bank_limits,
        True,
        "step25_guardrails",
    )

    step26 = frames["step26_guardrails"]
    require_columns(
        step26,
        ["topic", "required_language", "prohibited_language"],
        "Step 26 reporting guardrails",
    )
    s26_required = " ".join(step26["required_language"].astype(str)).lower()
    s26_prohibited = " ".join(step26["prohibited_language"].astype(str)).lower()
    gate_limits = (
        "exploratory post-freeze deployment layer" in s26_required
        and "prospectively confirmed final method" in s26_prohibited
        and "validated fault diagnosis accuracy" in s26_prohibited
    )
    audit.add(
        "G-S26",
        "source reporting guardrails",
        "Step 26 retains exploratory-status and descriptive-attribution limitations.",
        gate_limits,
        gate_limits,
        True,
        "step26_guardrails",
    )


def guardrail_table() -> pd.DataFrame:
    rows = [
        (
            "G01",
            "Primary population",
            "Primary cross-fault inference uses only the 17 Step-22 reserved faults.",
            "Pooling Faults 3, 9, or 15 into the reserved-fault primary macro analysis.",
        ),
        (
            "G02",
            "Primary method",
            "The primary result is the frozen per-fault Proposed Top-5 local monitor from Step 22/23B.",
            "Replacing the primary result with the Step-25 OR bank or Step-26 joint gate.",
        ),
        (
            "G03",
            "Robustness scope",
            "Step 24 perturbs standardized residual-evidence channels with frozen denominator and threshold.",
            "Claiming physical raw-sensor failure or sensor-noise robustness.",
        ),
        (
            "G04",
            "Detector bank",
            "Step 25 is a post-freeze cross-trigger and deployment/selectivity audit; its OR rule is hypothetical.",
            "Calling the OR bank the deployed primary detector or a validated diagnosis rule.",
        ),
        (
            "G05",
            "Joint gate status",
            "Step 26 is an exploratory post-freeze deployment layer calibrated only on normal validation.",
            "Calling Step 26 prospectively confirmed, independently validated, or the unchanged Step-22 method.",
        ),
        (
            "G06",
            "False-alarm terminology",
            "Use false-alarm rate for normal-operation data and cross-trigger rate for real-fault data.",
            "Calling a non-target monitor alarm on a real fault a false alarm.",
        ),
        (
            "G07",
            "Attribution",
            "Target-monitor rank is a descriptive localization diagnostic.",
            "Calling target rank fault-classification accuracy.",
        ),
        (
            "G08",
            "Claim strength",
            "Claim competitive sparse, interpretable, and fault-adaptive monitoring within TEP.",
            "Claiming universal superiority or external-process universality.",
        ),
        (
            "G09",
            "Step-22 repair",
            "Retain the Step-22-R numerical-repair disclosure.",
            "Presenting the repair as a scientific protocol change or omitting it.",
        ),
        (
            "G10",
            "Number separation",
            "Keep 4.376% primary macro FAR, 38.0% OR-bank FAR, and 3.2% exploratory joint-gate FAR separately labelled.",
            "Reporting any of the three as a generic FAR for the same operating rule.",
        ),
    ]
    return pd.DataFrame(rows, columns=["guardrail_id", "topic", "required_language", "prohibited_language"])


def claim_registry(numbers: pd.DataFrame) -> pd.DataFrame:
    def number(number_id: str) -> pd.Series:
        selected = numbers[numbers["number_id"] == number_id]
        if len(selected) != 1:
            raise ValueError(f"Claim registry could not resolve number {number_id}")
        return selected.iloc[0]

    primary_far = number("primary_macro_normal_far")
    primary_point = number("primary_macro_point_detection")
    primary_delay = number("primary_macro_median_allrun_delay")
    joint_far = number("joint_gate_normal_far")
    claims = [
        {
            "claim_id": "C01",
            "analysis_layer": "primary confirmatory frozen local monitoring",
            "claim": (
                "Across the 17 reserved TEP faults, the frozen Proposed Top-5 monitor detected "
                "all 250 final runs for every fault."
            ),
            "status": "primary",
            "source_numbers": "primary_macro_run_detection;primary_fault_specific_detection_lower_bound",
            "allowed_strength": "direct result",
        },
        {
            "claim_id": "C02",
            "analysis_layer": "primary confirmatory frozen local monitoring",
            "claim": (
                f"The primary macro normal run FAR was {primary_far['estimate']:.3f}% "
                f"(crossed 95% CI {primary_far['ci_95_low']:.3f}%--{primary_far['ci_95_high']:.3f}%)."
            ),
            "status": "primary",
            "source_numbers": "primary_macro_normal_far",
            "allowed_strength": "direct result",
        },
        {
            "claim_id": "C03",
            "analysis_layer": "primary confirmatory frozen local monitoring",
            "claim": (
                f"Macro mean post-fault point detection was {primary_point['estimate']:.3f}% and the "
                f"macro median all-run penalized delay was {primary_delay['estimate']:.1f} min."
            ),
            "status": "primary",
            "source_numbers": "primary_macro_point_detection;primary_macro_median_allrun_delay",
            "allowed_strength": "direct result",
        },
        {
            "claim_id": "C04",
            "analysis_layer": "post-freeze residual-evidence robustness",
            "claim": (
                "Removing the sole Top-1 residual-evidence channel eliminated detection on all 17 faults, "
                "whereas the frozen Top-5 retained 250/250 detections for every fault under any single-channel loss."
            ),
            "status": "secondary robustness",
            "source_numbers": "robustness_top1_loss_faults_detected;robustness_top5_loss_faults_detected",
            "allowed_strength": "residual-evidence-channel robustness only",
        },
        {
            "claim_id": "C05",
            "analysis_layer": "post-freeze detector-bank diagnostic",
            "claim": (
                "A hypothetical OR of all 17 frozen monitors retained complete bank detection but produced "
                "a 38.0% normal run FAR, demonstrating that coverage alone is not a deployable arbitration rule."
            ),
            "status": "secondary diagnostic",
            "source_numbers": "or_bank_normal_far",
            "allowed_strength": "deployment limitation",
        },
        {
            "claim_id": "C06",
            "analysis_layer": "exploratory post-freeze joint gate",
            "claim": (
                f"A single normal-validation-calibrated joint gate reduced normal-evaluation run FAR to "
                f"{joint_far['estimate']:.1f}% (95% CI {joint_far['ci_95_low']:.2f}%--"
                f"{joint_far['ci_95_high']:.2f}%) while retaining 250/250 detections for every reserved fault."
            ),
            "status": "exploratory",
            "source_numbers": "joint_gate_normal_far;joint_gate_macro_run_detection",
            "allowed_strength": "exploratory deployment evidence, not confirmation",
        },
        {
            "claim_id": "C07",
            "analysis_layer": "overall interpretation",
            "claim": (
                "The evidence supports a fault-adaptive sparse residual-monitoring framework with robust "
                "local evidence fusion and a promising, but still exploratory, detector-bank deployment layer."
            ),
            "status": "synthesis",
            "source_numbers": "C01--C06",
            "allowed_strength": "TEP-scoped; no universal-superiority claim",
        },
    ]
    return pd.DataFrame(claims)


def placement_plan() -> pd.DataFrame:
    rows = [
        (
            "primary comparison table",
            "23_reserved_primary_summary_with_ci.csv;23b_allrun_delay_macro_summary.csv",
            "Results: reserved-fault primary validation",
            "main text",
            "Use the frozen per-fault operating rule; report confidence intervals.",
        ),
        (
            "Fig. 22 primary performance",
            "fig22a_reserved_final_run_detection.png;fig22b_reserved_final_point_detection.png",
            "Results: reserved-fault primary validation",
            "main text or supplement",
            "Do not mix development faults 3, 9, and 15 into the primary panel.",
        ),
        (
            "Fig. 23 performance trade-off",
            "fig23a_reserved_performance_tradeoffs.png",
            "Results: comparative performance",
            "main text",
            "Supports competitive trade-off, not universal superiority.",
        ),
        (
            "Fig. 24 evidence-channel robustness",
            "fig24a_single_channel_loss_robustness.png;fig24b_single_channel_contamination_robustness.png",
            "Results/Discussion: redundancy and robustness",
            "one main-text panel plus full supplement",
            "Use residual-evidence-channel language; never raw-sensor robustness.",
        ),
        (
            "Fig. 25 detector-bank audit",
            "fig25a_normal_bank_false_alarm_audit.png;fig25b_crossfault_alarm_matrix.png;fig25c_target_monitor_rank_diagnostics.png",
            "Discussion: deployment and attribution limits",
            "supplement",
            "Retain the unfavorable 38% OR-bank FAR and weak selectivity.",
        ),
        (
            "Fig. 26 exploratory joint gate",
            "fig26a_joint_gate_operating_point.png;fig26b_joint_gate_detection_by_fault.png;fig26c_joint_gate_attribution_rank.png",
            "Discussion: exploratory system-level deployment",
            "main text for 26a; supplement for 26b--c",
            "Label every panel exploratory and post-freeze.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["artifact", "source_files", "recommended_section", "placement", "reporting_note"],
    )


def render_report(
    checks: pd.DataFrame,
    numbers: pd.DataFrame,
    claims: pd.DataFrame,
    guardrails: pd.DataFrame,
) -> str:
    failed = checks[~checks["passed"]]
    lines = [
        "# Step 27 frozen-results audit and manuscript claim sheet",
        "",
        "This report was generated from stored Step-22--Step-26 outputs only. No process data, fitting, selection, tuning, calibration, or performance recomputation occurred.",
        "",
        "## Audit status",
        "",
        f"- Checks passed: **{int(checks['passed'].sum())}/{len(checks)}**",
        f"- Failed error-level checks: **{int(((~checks['passed']) & (checks['severity'] == 'error')).sum())}**",
        "",
    ]
    if not failed.empty:
        lines.extend(["### Failed checks", ""])
        for row in failed.itertuples(index=False):
            lines.append(f"- `{row.check_id}`: {row.statement} (observed: {row.observed}; expected: {row.expected})")
        lines.append("")

    lines.extend(["## Manuscript-ready numeric registry", ""])
    for row in numbers.itertuples(index=False):
        interval = ""
        if pd.notna(row.ci_95_low) and pd.notna(row.ci_95_high):
            interval = f"; 95% CI {row.ci_95_low:.3f}--{row.ci_95_high:.3f}"
        lines.append(
            f"- **{row.number_id}**: {row.endpoint} = {row.estimate:.3f} {row.unit}{interval}. "
            f"Source: `{row.source}`."
        )

    lines.extend(["", "## Allowed claims", ""])
    for row in claims.itertuples(index=False):
        lines.append(f"- **{row.claim_id} ({row.status})** — {row.claim}")

    lines.extend(["", "## Non-negotiable reporting guardrails", ""])
    for row in guardrails.itertuples(index=False):
        lines.append(
            f"- **{row.guardrail_id} — {row.topic}:** {row.required_language} "
            f"Avoid: {row.prohibited_language}"
        )
    lines.append("")
    return "\n".join(lines)


def build_manifest() -> pd.DataFrame:
    rows = []
    for key, path in INPUT_PATHS.items():
        rows.append(
            {
                "artifact_role": "input",
                "artifact_key": key,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    for key, path in OUTPUT_PATHS.items():
        if key == "manifest":
            continue
        rows.append(
            {
                "artifact_role": "output",
                "artifact_key": key,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def run_audit() -> None:
    for key, path in INPUT_PATHS.items():
        require_file(path, key)

    frames = {
        key: read_csv(key)
        for key in INPUT_PATHS
        if INPUT_PATHS[key].suffix.lower() == ".csv"
    }
    audit = Audit()
    number_rows: list[dict[str, Any]] = []

    audit_step22_23(audit, frames, number_rows)
    audit_step24(audit, frames, number_rows)
    audit_step25_26(audit, frames, number_rows)
    audit_source_guardrails(audit, frames)

    checks = audit.frame()
    numbers = pd.DataFrame(number_rows)
    guardrails = guardrail_table()
    claims = claim_registry(numbers)
    placement = placement_plan()

    checks.to_csv(OUTPUT_PATHS["checks"], index=False, encoding="utf-8-sig")
    claims.to_csv(OUTPUT_PATHS["claims"], index=False, encoding="utf-8-sig")
    numbers.to_csv(OUTPUT_PATHS["numbers"], index=False, encoding="utf-8-sig")
    placement.to_csv(OUTPUT_PATHS["placement"], index=False, encoding="utf-8-sig")
    guardrails.to_csv(OUTPUT_PATHS["guardrails"], index=False, encoding="utf-8-sig")

    report_text = render_report(checks, numbers, claims, guardrails)
    OUTPUT_PATHS["report"].write_text(report_text, encoding="utf-8")

    manifest = build_manifest()
    manifest.to_csv(OUTPUT_PATHS["manifest"], index=False, encoding="utf-8-sig")

    failed_errors = checks[(~checks["passed"]) & (checks["severity"] == "error")]
    config = {
        "analysis": "Step 27 frozen-results consistency audit and manuscript claim registry",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/27_frozen_results_consistency_audit.py",
        "script_sha256_at_execution": sha256_file(Path(__file__)),
        "reserved_faults": RESERVED_FAULTS,
        "no_process_data_loaded": True,
        "no_performance_recomputed": True,
        "no_selection_fitting_tuning_or_calibration": True,
        "input_sha256": {
            key: sha256_file(path) for key, path in INPUT_PATHS.items()
        },
        "checks_total": int(len(checks)),
        "checks_passed": int(checks["passed"].sum()),
        "failed_error_checks": failed_errors["check_id"].tolist(),
        "output_paths": {
            key: str(path.relative_to(PROJECT_ROOT))
            for key, path in OUTPUT_PATHS.items()
        },
        "next_step_constraint": (
            "If all error-level checks pass, stop expanding the experiment tree and use the "
            "claim registry to assemble manuscript v21. Any later analysis must remain clearly "
            "labelled secondary and may not replace the frozen Step-22/23B primary results."
        ),
    }
    OUTPUT_PATHS["config"].write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Refresh the manifest after the configuration has been written.
    manifest = build_manifest()
    manifest.to_csv(OUTPUT_PATHS["manifest"], index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("Step 27 summary")
    print("=" * 96)
    print(f"Consistency checks passed: {int(checks['passed'].sum())}/{len(checks)}")
    if failed_errors.empty:
        print("No error-level inconsistency was found.")
        print("The experiment tree is now closed for manuscript v21 assembly.")
    else:
        print("ERROR-LEVEL CHECKS FAILED:")
        print(failed_errors[["check_id", "statement", "observed", "expected"]].to_string(index=False))
    print("Saved outputs:")
    for key, path in OUTPUT_PATHS.items():
        print(f"  {key}: {path}")

    if not failed_errors.empty:
        raise RuntimeError(
            "Step 27 found error-level inconsistencies. Do not edit the manuscript numbers "
            "until the failed checks are resolved."
        )
    print("Step 27 finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        required=True,
        help="run the stored-results audit and generate the manuscript claim registry",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run:
        run_audit()


if __name__ == "__main__":
    main()
