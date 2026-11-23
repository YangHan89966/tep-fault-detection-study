"""Audit temporal pairing in a public physical multiphase-plant dataset.

No controller coefficients are estimated. The specified Kp = 1.7 is checked
against recorded PID2 fields. A negative lag pairs a target in row t with an
error/input from an earlier row. This is a descriptive, exploratory audit of
released files, not a prospective process predictor or fault-detection test.

All 46 Working_conditions JSON files are inventoried. The main aggregate uses
one copy of each distinct recorded trajectory and only rows whose complete
[-3,+3] neighborhood has PID2 enabled. This common support is used for all
seven lags. The interior-output analysis additionally requires the ORIGINAL
recorded P+I+D to lie strictly between 6501 and 16699 microamperes. It is not
reselected for each lag. Anomaly records are outside the analysis scope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import zipfile

import numpy as np

ZIP_SHA256 = "e5fa6872449d4b538065b498ff547ebfddbe751831c8cceb852f30935a185f4f"
KP, LOWER, UPPER = 1.7, 6500.0, 16700.0
SLOPE, INTERCEPT = 0.04406197, -171.040768
LAGS = tuple(range(-3, 4))
P_TOL, OUT_TOL, S_TOL, PV_OUT_TOL = 0.02, 1.02, 0.00051, 1.04
HEADERS = ["I2", "D2", "P2", "E2", "IN2", "OUT2", "ONoff2"]


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values, tolerance):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "rmse": None, "mae": None, "max_abs": None,
                "min": None, "max": None, "mismatch_n": 0,
                "mismatch_fraction": None, "tolerance": tolerance}
    return {"n": len(values), "rmse": float(np.sqrt(np.mean(values**2))),
            "mae": float(np.mean(np.abs(values))),
            "max_abs": float(np.max(np.abs(values))),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "mismatch_n": int(np.sum(np.abs(values) > tolerance)),
            "mismatch_fraction": float(np.mean(np.abs(values) > tolerance)),
            "tolerance": tolerance}


def prefix(record, prefix_string, values):
    record.update({prefix_string + key: value for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=home / "data" / "Data.zip")
    parser.add_argument("--output", type=Path, default=home / "results")
    args = parser.parse_args()
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise ValueError(f"Unexpected source archive SHA256: {digest}")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, per_run, lag_rows, audit_examples, interval_rows = [], [], [], [], []
    arrays = []
    hashes = {}
    dates_all = Counter()
    with zipfile.ZipFile(args.data) as archive:
        all_json = [n for n in archive.namelist()
                    if n.endswith(".json") and
                    ("/Working_conditions/" in n or "/Anomalies/" in n)]
        # Count actual experiment dates independently of the repository date span.
        for name in archive.namelist():
            if name.endswith(".json") and "/General_description/" not in name:
                obj = json.loads(archive.read(name))
                if "Simulation" in obj:
                    dates_all[obj["Simulation"]["date"]] += 1
        names = [n for n in archive.namelist()
                 if "/Working_conditions/" in n and n.endswith(".json")]
        names.sort(key=lambda n: int(re.search(r"working_condition(\d+)\.json$", n)[1]))
        assert len(names) == 46
        for name in names:
            raw = archive.read(name)
            obj = json.loads(raw)
            meta, pid = obj["Simulation"], obj["PID_Controllers"]["PID2"]
            assert pid["Header"] == HEADERS
            a = np.asarray(pid["Values"], dtype=float)
            time = np.asarray(meta["sim_time"], dtype=float)
            level = np.asarray(obj["Sensors"]["S6"], dtype=float)
            assert a.ndim == 2 and a.shape[1] == 7
            assert len(a) == len(time) == len(level)
            assert np.isfinite(a).all() and np.isfinite(time).all()
            assert np.all(np.diff(time) > 0)
            I, D, P, E, IN, OUT, ON = a.T
            assert set(np.unique(ON)).issubset({0.0, 1.0})
            run = int(re.search(r"working_condition(\d+)\.json$", name)[1])
            # Deduplicate by all released measurement/control arrays and times,
            # not only PID2. Keep the lowest numeric run identifier.
            numeric_payload = {k: obj[k] for k in
                               ["Sensors", "Pneumatic_Valves", "PID_Controllers"]}
            numeric_payload["sim_time"] = meta["sim_time"]
            trajectory_hash = hashlib.sha256(json.dumps(
                numeric_payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            duplicate_of = hashes.get(trajectory_hash)
            hashes.setdefault(trajectory_hash, run)
            unique = duplicate_of is None
            active = ON == 1
            ids = np.arange(3, len(a) - 3)
            ids = ids[np.all(np.stack([active[ids + lag] for lag in LAGS]), axis=0)]
            component_sum = P + I + D
            interior = (component_sum[ids] > LOWER + 1) & (component_sum[ids] < UPPER - 1)
            un_ids = ids[interior]
            active_sp = np.unique((IN + E)[active])
            rec = {
                "run": run, "run_name": f"working_condition{run}",
                "source_member": name, "source_member_sha256": hashlib.sha256(raw).hexdigest(),
                "date": meta["date"], "start_time": meta["time"],
                "declared_samples": meta["samples number"], "actual_rows": len(a),
                "nominal_sample_s": meta["sample time"],
                "duration_first_to_last_s": float(time[-1] - time[0]),
                "active_rows": int(active.sum()), "common_rows": len(ids),
                "interior_rows": len(un_ids),
                "trajectory_sha256": trajectory_hash,
                "duplicate_of": duplicate_of if duplicate_of is not None else "",
                "include_primary": bool(unique and len(ids)),
                "setpoint_level_mm": meta["setpoint"][1],
                "active_setpoint_current_values": ";".join(f"{x:g}" for x in active_sp),
                "active_setpoint_current_count": len(active_sp),
                "output_integer_all_rows": bool(np.all(OUT == np.round(OUT))),
                "dt_min_s": float(np.min(np.diff(time))),
                "dt_median_s": float(np.median(np.diff(time))),
                "dt_max_s": float(np.max(np.diff(time))),
            }
            prefix(rec, "sensor_", stats(level - (SLOPE * IN + INTERCEPT), S_TOL))
            inventory.append(rec)
            if not len(ids):
                continue
            assert len(active_sp) == 1, "Setpoint changes require segmented interpretation."
            assert np.all((OUT[ids] >= LOWER) & (OUT[ids] <= UPPER))
            output_identity = OUT[ids] - np.clip(component_sum[ids], LOWER, UPPER)
            row = {"run": run, "date": meta["date"], "duplicate_of": rec["duplicate_of"],
                   "include_primary": unique, "common_rows": len(ids),
                   "interior_rows": len(un_ids),
                   "output_at_limit_rows": int(np.sum((OUT[ids] == LOWER) | (OUT[ids] == UPPER))),
                   "setpoint_current_uA": active_sp[0],
                   "dt_median_s": rec["dt_median_s"]}
            prefix(row, "component_output_", stats(output_identity, OUT_TOL))
            by_lag = {}
            for lag in LAGS:
                p_residual = P[ids] - KP * E[ids + lag]
                output_residual = OUT[ids] - np.clip(KP * E[ids + lag] + I[ids] + D[ids], LOWER, UPPER)
                q_residual = OUT[un_ids] - I[un_ids] - D[un_ids] - KP * E[un_ids + lag]
                # Independently use the released physical level column rather
                # than the error column. Reconstruct current via the published
                # sensor calibration; retain its 0.001 mm rounding error.
                pv_current = (level[un_ids + lag] - INTERCEPT) / SLOPE
                q_pv_residual = OUT[un_ids] - I[un_ids] - D[un_ids] + KP * pv_current - KP * active_sp[0]
                by_lag[lag] = {"P": p_residual, "output": output_residual,
                               "q": q_residual, "q_physical_PV": q_pv_residual}
                for metric, residual, tol in [("P", p_residual, P_TOL),
                                               ("output", output_residual, OUT_TOL),
                                               ("q", q_residual, OUT_TOL),
                                               ("q_physical_PV", q_pv_residual, PV_OUT_TOL)]:
                    lag_rows.append({"run": run, "include_primary": unique,
                                     "duplicate_of": rec["duplicate_of"], "lag_rows": lag,
                                     "metric": metric, **stats(residual, tol)})
                    if lag in (-1, 0):
                        prefix(row, f"{metric}_{'previous' if lag == -1 else 'same'}_", stats(residual, tol))
            per_run.append(row)
            intervals = time[ids] - time[ids - 1]
            for q in (0, 0.05, 0.5, 0.95, 1):
                interval_rows.append({"run": run, "include_primary": unique,
                                      "quantile": q, "previous_row_interval_s": np.quantile(intervals, q)})
            arrays.append({"run": run, "unique": unique, "ids": ids, "un_ids": un_ids,
                           "by_lag": by_lag, "output_identity": output_identity,
                           "previous_row_intervals": intervals, "date": meta["date"],
                           "same_p_mismatch": np.abs(by_lag[0]["P"]) > P_TOL,
                           "input_changed": IN[ids] != IN[ids - 1]})
            # Complete fixed illustrative run, without selecting its best interval.
            if run == 14:
                common_set, interior_set = set(ids), set(un_ids)
                for k in range(len(a)):
                    previous_error = E[k - 1] if k else None
                    audit_examples.append({
                        "row_zero_based": k, "time_s": time[k], "S6_mm": level[k],
                        "I2_uA": I[k], "D2_uA": D[k], "P2_uA": P[k],
                        "E2_uA": E[k], "IN2_uA": IN[k], "OUT2_uA": OUT[k],
                        "ONoff2": ON[k], "primary_common_row": k in common_set,
                        "interior_row": k in interior_set,
                        "KP_times_E_same_uA": KP * E[k],
                        "KP_times_E_previous_uA": KP * previous_error if k else None,
                        "component_output_pred_uA": np.clip(component_sum[k], LOWER, UPPER),
                        "output_pred_same_uA": np.clip(KP * E[k] + I[k] + D[k], LOWER, UPPER),
                        "output_pred_previous_uA": np.clip(KP * previous_error + I[k] + D[k], LOWER, UPPER) if k else None,
                        "adjusted_output_q_uA": OUT[k] - I[k] - D[k],
                        "q_residual_same_uA": OUT[k] - I[k] - D[k] - KP * E[k],
                        "q_residual_previous_uA": OUT[k] - I[k] - D[k] - KP * previous_error if k else None,
                    })
    aggregate_rows = []
    group_arrays = {}
    for group, included in {
        "unique_runs_primary": [a for a in arrays if a["unique"]],
        "all_enabled_files": arrays,
        "unique_except_illustration14": [a for a in arrays if a["unique"] and a["run"] != 14],
    }.items():
        group_arrays[group] = included
        for lag in LAGS:
            for metric, tol in [("P", P_TOL), ("output", OUT_TOL), ("q", OUT_TOL),
                                ("q_physical_PV", PV_OUT_TOL)]:
                residual = np.concatenate([a["by_lag"][lag][metric] for a in included])
                aggregate_rows.append({"group": group, "runs_with_rows": sum(len(a["by_lag"][lag][metric]) > 0 for a in included),
                                       "lag_rows": lag, "metric": metric, **stats(residual, tol)})
    primary = group_arrays["unique_runs_primary"]
    identity = np.concatenate([a["output_identity"] for a in primary])
    intervals = np.concatenate([a["previous_row_intervals"] for a in primary])
    primary_rows = [row for row in per_run if row["include_primary"]]
    q_rows = [row for row in primary_rows if row["interior_rows"]]
    p_previous = np.concatenate([a["by_lag"][-1]["P"] for a in primary])
    p_current = np.concatenate([a["by_lag"][0]["P"] for a in primary])
    event_matches = sum(np.sum(a["same_p_mismatch"] == a["input_changed"]) for a in primary)
    summary = {
        "source_doi": "10.5281/zenodo.16978349", "source_archive_sha256": digest,
        "study": "Exploratory descriptive audit of released PID2 field pairing; no fitted parameters",
        "constants": {"Kp": KP, "output_lower_uA": LOWER, "output_upper_uA": UPPER,
                      "S6_slope_mm_per_uA": SLOPE, "S6_intercept_mm": INTERCEPT},
        "lag_definition": "Target at row t is paired with E2 or IN2 at row t + lag; lag -1 is the preceding recorded row.",
        "common_support": "Rows t for which all seven rows t-3,...,t+3 exist and ONoff2 equals 1 throughout.",
        "interior_support": "Common rows additionally satisfying 6501 < recorded P2+I2+D2 < 16699; fixed for every lag.",
        "tolerances_uA": {"P": P_TOL, "output_and_adjusted_output": OUT_TOL,
                          "adjusted_output_using_physical_PV": PV_OUT_TOL},
        "tolerance_reason": "P/I/D are stored to 0.01 uA; OUT is integer uA. 1.02 uA conservatively allows integer truncation plus component rounding. For the physical S6 column, 1.04 uA additionally allows Kp*0.0005/0.04406197 = 0.0192910 uA from 0.001 mm sensor rounding. Tolerances are audit resolution bounds, not statistical confidence limits.",
        "all_experiment_dates_from_files": dict(sorted(dates_all.items())),
        "normal_experiment_dates_from_files": dict(sorted(Counter(row["date"] for row in inventory).items())),
        "normal_files": len(inventory), "enabled_files": len(per_run),
        "disabled_files": [row["run"] for row in inventory if not row["active_rows"]],
        "duplicates": [{"run": row["run"], "duplicate_of": row["duplicate_of"],
                        "trajectory_sha256": row["trajectory_sha256"]} for row in inventory if row["duplicate_of"] != ""],
        "duplicate_definition": "Equality of all Sensors, Pneumatic_Valves, PID_Controllers objects and sim_time after canonical JSON serialization. Metadata descriptions are not part of the key.",
        "unique_enabled_runs": len(primary),
        "common_rows_unique": sum(len(a["ids"]) for a in primary),
        "interior_rows_unique": sum(len(a["un_ids"]) for a in primary),
        "unique_runs_with_interior_rows": len(q_rows),
        "run_without_interior_rows": [row["run"] for row in primary_rows if not row["interior_rows"]],
        "declared_samples_discrepancy": dict(Counter(str(row["actual_rows"] - row["declared_samples"]) for row in inventory)),
        "P_same": stats(p_current, P_TOL), "P_previous": stats(p_previous, P_TOL),
        "component_output_identity": stats(identity, OUT_TOL),
        "same_P_mismatch_equals_input_change": {"matching_rows": int(event_matches),
                                               "total_rows": sum(len(a["ids"]) for a in primary)},
        "previous_row_interval_s": {str(q): float(np.quantile(intervals, q)) for q in (0, .05, .5, .95, 1)},
        "per_run_P_same_mismatch_fraction_range": [min(row["P_same_mismatch_fraction"] for row in primary_rows), max(row["P_same_mismatch_fraction"] for row in primary_rows)],
        "per_run_q_same_rmse_range_uA": [min(row["q_same_rmse"] for row in q_rows), max(row["q_same_rmse"] for row in q_rows)],
        "per_run_q_previous_rmse_range_uA": [min(row["q_previous_rmse"] for row in q_rows), max(row["q_previous_rmse"] for row in q_rows)],
        "per_run_q_RMSE_ratio_same_over_previous": {
            "min": min(row["q_same_rmse"] / row["q_previous_rmse"] for row in q_rows),
            "median": float(np.median([row["q_same_rmse"] / row["q_previous_rmse"] for row in q_rows])),
            "max": max(row["q_same_rmse"] / row["q_previous_rmse"] for row in q_rows)},
        "limitations": [
            "Physical experimental plant, not a commercial production-plant historian.",
            "The file dates are June 2024; the repository's broader 2024-03/2025-04 acquisition span is not the date range of these files.",
            "Known controller algebra and existing components are checked, not an independently predicted plant output.",
            "Exact acquisition/publication ordering and application source code are unavailable; no software defect or causal scheduling mechanism is established.",
            "A row lag is not a process dead time; actual adjacent timestamps are retained and summarized.",
            "Repeated held values and shared acquisition conditions preclude treating all rows or runs as independent replications; no inferential p-values or iid intervals are reported.",
            "The earlier row convention was discovered during inspection including working_condition14; other runs provide descriptive replication, not preregistered confirmation.",
            "Neither anomalies nor fault detection performance are evaluated.",
        ],
    }
    for metric, tol in [("output", OUT_TOL), ("q", OUT_TOL),
                        ("q_physical_PV", PV_OUT_TOL)]:
        same = np.concatenate([a["by_lag"][0][metric] for a in primary])
        previous = np.concatenate([a["by_lag"][-1][metric] for a in primary])
        summary[f"{metric}_same"] = stats(same, tol)
        summary[f"{metric}_previous"] = stats(previous, tol)
        summary[f"{metric}_RMSE_ratio_same_over_previous"] = float(
            np.sqrt(np.mean(same**2) / np.mean(previous**2)))
    checks = {
        "source_sha256_verified": True,
        "all_P_previous_within_tolerance": bool(np.all(np.abs(p_previous) <= P_TOL)),
        "all_component_output_within_quantization_tolerance": bool(np.all(np.abs(identity) <= OUT_TOL)),
        "all_q_previous_within_quantization_tolerance": summary["q_previous"]["mismatch_n"] == 0,
        "all_q_physical_PV_previous_within_quantization_tolerance": summary["q_physical_PV_previous"]["mismatch_n"] == 0,
        "all_sensor_calibrations_within_rounding_tolerance": all(row["sensor_mismatch_n"] == 0 for row in inventory),
        "same_P_mismatch_exactly_tracks_input_changes": event_matches == summary["common_rows_unique"],
        "all_common_rows_identical_across_lags": True,
        "all_interior_rows_identical_across_lags": True,
        "one_current_setpoint_per_enabled_run": all(row["active_setpoint_current_count"] == 1 for row in inventory if row["active_rows"]),
    }
    assert all(checks.values()), checks
    write_csv(args.output / "run_inventory.csv", inventory)
    write_csv(args.output / "run_summary.csv", per_run)
    write_csv(args.output / "lag_metrics_by_run.csv", lag_rows)
    write_csv(args.output / "lag_metrics_aggregate.csv", aggregate_rows)
    write_csv(args.output / "previous_row_intervals_by_run.csv", interval_rows)
    write_csv(args.output / "working_condition14_trace.csv", audit_examples)
    # Store individual residuals for a reproducible distribution plot. No raw
    # source data are needed in the report; these are derived audit quantities.
    residual_rows = []
    for a in primary:
        for position, row_id in enumerate(a["un_ids"]):
            residual_rows.append({"run": a["run"], "row_zero_based": row_id,
                                  "q_same_residual_uA": a["by_lag"][0]["q"][position],
                                  "q_previous_residual_uA": a["by_lag"][-1]["q"][position],
                                  "q_physical_PV_same_residual_uA": a["by_lag"][0]["q_physical_PV"][position],
                                  "q_physical_PV_previous_residual_uA": a["by_lag"][-1]["q_physical_PV"][position]})
    write_csv(args.output / "adjusted_output_residuals.csv", residual_rows)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "validation.json", checks)
    print(json.dumps(serializable({"unique_enabled_runs": summary["unique_enabled_runs"],
                      "common_rows": summary["common_rows_unique"],
                      "interior_rows": summary["interior_rows_unique"],
                      "P_same": summary["P_same"], "P_previous": summary["P_previous"],
                      "checks": checks}), indent=2))


if __name__ == "__main__":
    main()
