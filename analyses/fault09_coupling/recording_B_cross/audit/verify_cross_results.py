#!/usr/bin/env python3
"""Independent output audit. Does not import the experiment's numerical code.

Recomputes signed two-sided CUSUM and event definitions from saved score arrays.
The 745 source trajectories are not included in the return and are not re-read.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_integrity import verify_public_bundle, check_historical_digest
PUBLIC_ROOT = None
PUBLIC_EDITS = {}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def check_lock(folder, records):
    for rel, expected in records.items():
        p = folder / rel
        assert p.is_file() and not p.is_symlink(), str(p)
        assert p.resolve().is_relative_to(folder.resolve()), str(p)
        check_historical_digest(PUBLIC_ROOT, p, expected, PUBLIC_EDITS)
    return len(records)


def close(a, b, label, atol=1e-8, rtol=1e-9):
    assert np.shape(a) == np.shape(b), label + " shape"
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=label)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def scalar_cusum(scores):
    """Scalar loop implementation of C+ = max(0,C+ + x-k), C- analog."""
    out = np.empty_like(scores, dtype=np.float64)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[2]):
            plus = minus = 0.0
            for t in range(scores.shape[1]):
                x = float(scores[i, t, j])
                plus = max(0.0, (plus + x) - 0.05)
                minus = max(0.0, (minus - x) - 0.05)
                out[i, t, j] = max(plus, minus)
    return out


def independent_event(alarm):
    indices = np.flatnonzero(alarm)
    pre = bool(np.any(indices < 160))
    post = indices[indices >= 160]
    delay = int(post[0] - 160) if len(post) else 800
    clean = bool(len(post)) and not pre
    loss = 3 * delay if clean else 2400
    return {
        "normal_alarm": int(len(indices) > 0), "pre_alarm": int(pre),
        "delay_samples": delay, "loss_minutes": loss,
        **{"clean_" + str(h): int(clean and 3 * delay <= h) for h in (120, 480, 2400)},
    }


def main():
    global PUBLIC_ROOT, PUBLIC_EDITS
    parser = argparse.ArgumentParser()
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, pkg, out = args.returned.resolve(), args.reference_package.resolve(), args.output.resolve()
    PUBLIC_ROOT = root
    PUBLIC_EDITS, public_file_count = verify_public_bundle(root)
    r, ref = root / "results", pkg / "reference"
    out.mkdir(parents=True, exist_ok=True)
    counts = {"public_manifest_files": public_file_count}
    final = read_json(r / "FINAL_LOCK.json")
    assert final["status"] == "COMPLETE" and final["new_simulations"] == 0
    counts["final_lock_files"] = check_lock(r, final["files"])
    bind = read_json(r / "EXECUTION_BINDING.json")
    assert sha(root / "code/run_cross.py") == bind["runner_sha256"]
    assert sha(root / "PROTOCOL.json") == bind["protocol_sha256"]
    assert (root / "PROTOCOL.json").read_bytes() == (pkg / "PROTOCOL.json").read_bytes()
    counts["bound_vendor_files"] = check_lock(pkg / "vendor", bind["vendor_files"])
    reference_paths = {}
    for name in ("MODEL_LOCK.json", "THRESHOLDS_LOCK.json", "FINAL_LOCK.json", "NORMAL_SUPPORT_LOCK.json"):
        lock = read_json(ref / name)
        folder = ref / lock["relative_path"]
        reference_paths[name] = folder
        counts["reference_" + name] = check_lock(folder, lock["files"])
    protocol = read_json(root / "PROTOCOL.json")
    check_lock(pkg / "assets", protocol["asset_sha256"])
    assert protocol["k"] == 0.05 and protocol["outer_calibration_rank"] == 190
    assert sha(root / "EXPERIMENT_SPECIFICATION.txt") == protocol["specification_sha256"]

    cases = read_json(r / "EVALUATION_CASES.json")
    manifest = read_json(pkg / "assets/INPUT_MANIFEST.json")["panels"]["D439_supplement"]["cases"]
    expected_cases = [c for role in ("normal_evaluation", "fault9_evaluation") for c in manifest if c["role"] == role]
    assert cases == expected_cases
    assert len(cases) == 240 and len({c["case_id"] for c in cases}) == 240
    assert (r / "EVALUATION_CASES.json").read_bytes() == (reference_paths["FINAL_LOCK.json"] / "D439_supplement_CASES.json").read_bytes()
    table = read_csv(r / "PER_RUN_EVENTS.csv")
    by_key = {(v["estimator"], v["monitor_id"], v["case_id"]): v for v in table}
    assert len(table) == len(by_key) == 1920
    main_rows = {(v["estimator"], v["monitor_id"]): v for v in read_csv(r / "MAIN_RESULTS.csv")}
    cross_rows = {(v["estimator"], v["recording"], v["pool"]): v for v in read_csv(r / "CROSS_SUMMARY.csv")}
    conditions = [("M1", "snapshot", "moment"), ("M2", "native", "moment"), ("M3", "snapshot", "mean"), ("M4", "native", "mean")]
    arrays, summaries, errors = {}, [], {}
    counts["independent_evaluation_alarm_bits"] = 0
    counts["independent_per_run_rows"] = 0
    for estimator in ("original", "batch"):
        model = read_json(r / (estimator + "_MODEL_LOCK.json"))
        thresholds = read_json(r / (estimator + "_THRESHOLDS_LOCK.json"))
        counts[estimator + "_model_files"] = check_lock(r / ("models_" + estimator), model["files"])
        assert model["protocol_sha256"] == bind["protocol_sha256"]
        assert model["pca_sha256"] == sha(pkg / "assets/PCA_BY_VIEW.npz")
        assert thresholds["model_lock_sha256"] == sha(r / (estimator + "_MODEL_LOCK.json"))
        assert thresholds["rank"] == 190 and thresholds["normal_n"] == 199
        assert model["time_utc"] < thresholds["time_utc"] < final["time_utc"]
        cal = npz(r / (estimator + "_CALIBRATION.npz"))
        ev = npz(r / (estimator + "_EVALUATION.npz"))
        arrays[estimator] = {"cal": cal, "ev": ev}
        assert cal["scores"].shape == (199, 960, 4) and ev["scores"].shape == (240, 960, 4)
        peaks = scalar_cusum(cal["scores"]).max(axis=1)
        errors[estimator + "_calibration_peaks"] = close(peaks, cal["peaks"], estimator + " peaks", atol=1e-12, rtol=1e-12)
        assert np.all(cal["anchors"] > 0)
        h = np.array([cal["anchors"][j] * sorted(peaks[:, j] / cal["anchors"][j])[189] for j in range(4)])
        errors[estimator + "_thresholds"] = close(h, cal["thresholds"], estimator + " thresholds", atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(ev["thresholds"], cal["thresholds"])
        np.testing.assert_array_equal(np.array(thresholds["thresholds"]), cal["thresholds"])
        alarm = scalar_cusum(ev["scores"]) > h
        saved = np.unpackbits(ev["alarm_packed"], axis=1, count=960, bitorder="little").astype(bool)
        np.testing.assert_array_equal(alarm, saved)
        counts["independent_evaluation_alarm_bits"] += int(alarm.size)
        arrays[estimator]["alarm"] = alarm
        for j, (mid, recording, pool) in enumerate(conditions):
            monitor = estimator + "_" + mid
            local_events = []
            for i, case in enumerate(cases):
                event = independent_event(alarm[i, :, j])
                row = by_key[(estimator, monitor, case["case_id"])]
                assert row["role"] == case["role"] and int(row["seed_u32"]) == case["seed_u32"]
                assert row["recording"] == recording and row["pool"] == pool
                for key, value in event.items():
                    assert int(row[key]) == value, (estimator, monitor, case["case_id"], key)
                local_events.append(event)
                counts["independent_per_run_rows"] += 1
            ns, fs = local_events[:120], local_events[120:]
            summary = dict(estimator=estimator, recording=recording, pool=pool, monitor_id=monitor,
                           threshold=float(h[j]), normal_n=120, fault_n=120,
                           normal_alarm_n=sum(v["normal_alarm"] for v in ns),
                           prefault_alarm_n=sum(v["pre_alarm"] for v in fs),
                           clean_2h_n=sum(v["clean_120"] for v in fs),
                           clean_8h_n=sum(v["clean_480"] for v in fs),
                           clean_40h_n=sum(v["clean_2400"] for v in fs),
                           mean_loss_minutes=sum(v["loss_minutes"] for v in fs)/120)
            for key, value in summary.items():
                actual = main_rows[(estimator, monitor)][key]
                if isinstance(value, str): assert actual == value
                else: assert np.isclose(float(actual), value, rtol=0, atol=1e-10), (monitor, key)
            cross = cross_rows[(estimator, recording, pool)]
            mappings = {"normal_alarm_n": "normal_alarm_n", "prefault_alarm_n": "pre_alarm_n", "clean_2h_n": "clean_120_n", "clean_8h_n": "clean_480_n", "clean_40h_n": "clean_2400_n", "mean_loss_minutes": "mean_penalized_loss_minutes"}
            for src, dst in mappings.items(): assert np.isclose(float(cross[dst]), summary[src], rtol=0, atol=1e-10)
            for name, field in (("normal_alarm", "normal_alarm_n"), ("pre_alarm", "prefault_alarm_n"), ("clean_120", "clean_2h_n"), ("clean_480", "clean_8h_n"), ("clean_2400", "clean_40h_n")):
                assert np.isclose(float(cross[name+"_percent"]), 100*summary[field]/120, rtol=0, atol=1e-10)
            summaries.append(summary)
    original_cal = npz(reference_paths["THRESHOLDS_LOCK.json"] / "D439_supplement.npz")
    original_ev = npz(reference_paths["FINAL_LOCK.json"] / "D439_supplement_EVALUATION.npz")
    for k in ("scores", "anchors", "peaks", "thresholds"):
        errors["baseline_calibration_"+k] = close(arrays["original"]["cal"][k], original_cal[k], "original cal "+k)
    errors["baseline_evaluation_scores"] = close(arrays["original"]["ev"]["scores"], original_ev["scores"], "original eval scores")
    np.testing.assert_array_equal(arrays["original"]["ev"]["alarm_packed"], original_ev["alarm_packed"])
    counts["original_reference_alarm_bits_equal"] = 240*960*4
    geometry = npz(pkg / "assets/NORMAL_DESIGN_GEOMETRY.npz")
    for view in ("native", "snapshot"):
        originals = r / "models_original" / view
        changed = r / "models_batch" / view
        old_reference = reference_paths["MODEL_LOCK.json"] / view
        for filename in ("DESIGN.npz", "FIT_MOMENTS.npz", "MODEL.npz"):
            a, b = npz(originals / filename), npz(old_reference / filename)
            assert set(a) == set(b)
            for key in a:
                if a[key].dtype.kind == 'b': np.testing.assert_array_equal(a[key], b[key])
                else: errors[view+"_baseline_"+filename+"_"+key] = close(a[key],b[key],view+filename+key)
        a, b = npz(originals / "DESIGN.npz"), npz(changed / "DESIGN.npz")
        for key in a:
            if key != "B": np.testing.assert_array_equal(a[key], b[key])
        for filename in ("FIT_MOMENTS.npz", "NORMAL_SUPPORT.npz"):
            aa, bb = npz(originals / filename), npz(changed / filename)
            assert set(aa) == set(bb)
            for key in aa: np.testing.assert_array_equal(aa[key], bb[key])
        x = geometry[view+"__residual_z"]
        if not bool(b["support_identity"]): x = x @ b["support_projection"]
        y = (x - b["centre"]) @ b["whitener"]
        blocks = np.stack([y[i,t:t+120].sum(axis=0)/np.sqrt(120.0) for i in range(40) for t in range(0,960,120)])
        centered = blocks - blocks.mean(axis=0)
        covariance = centered.T @ centered / 319
        errors[view+"_batch_B_from_archived_40N"] = close(covariance,b["B"],view+" batch covariance")
        sel, selref = read_json(originals/"SELECTION.json"), read_json(old_reference/"SELECTION.json")
        for key in ("choices", "mean_count", "moment_count", "candidate_labels"): assert sel[key] == selref[key]
        inner, innerref = npz(originals/"INNER.npz"), npz(old_reference/"INNER.npz")
        for key in ("selection_alarm_packed", "loss_samples_total"): np.testing.assert_array_equal(inner[key],innerref[key])
    events = [json.loads(s) for s in (r/"EVENTS.jsonl").read_text().splitlines()]
    role_counts = {v["role"]: v["count"] for v in events if v["event"] == "ROLE_READ"}
    assert role_counts == {"normal_design":40,"fault9_fit":100,"normal_anchor":32,"normal_inner":59,"fault9_selection":75,"normal_calibration":199,"normal_evaluation":120,"fault9_evaluation":120}
    assert all(v["raw_hashes_verified"] for v in events if v["event"] == "ROLE_READ")
    report = dict(status="PASS", independent_code=True, imports_experiment_numerical_code=False,
                  checks=counts, calibration_and_evaluation_independently_recomputed=True,
                  per_run_and_summary_exact=True, batch_B_recomputed_from_archived_40N=True,
                  geometry_except_B_and_fit_moments_unchanged=True,
                  reported_source_read_counts=role_counts, max_absolute_errors=errors,
                  publication_metadata_transformations_verified=sorted(PUBLIC_EDITS),
                  limitations=["745 original raw trajectories are external dependencies and are not re-read by this output audit.",
                               "The verified output chain and source-read events support completeness but do not replace a second raw-data execution.",
                               "Bootstrap contrasts are outside this script's scope.",
                               "Inference remains conditional on one training/calibration realization and previously analyzed recording panel 2; stream dependence is not resolved."])
    (out/"INDEPENDENT_OUTPUT_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"INDEPENDENT_MAIN_RESULTS.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps({"status":"PASS","checks":counts,"max_baseline_abs_error":max(errors.values())},indent=2))


if __name__ == "__main__":
    main()
