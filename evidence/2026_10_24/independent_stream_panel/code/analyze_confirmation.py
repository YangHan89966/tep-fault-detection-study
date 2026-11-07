"""Reanalyze the complete published 1199-trajectory panel; reject interim inference.

Intervals condition on the frozen representations and realized normal
calibration. Paired resampling units are physical trajectories. The five R5
states are averaged within each trajectory, never counted as five test runs.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.stats import beta
from score_core import load_scorers, events, primary_loss_contrast

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026102207
EXPECTED = {("normal_calibration", 0): 199, ("normal_evaluation", 0): 400,
            ("fault_evaluation", 3): 200, ("fault_evaluation", 9): 200,
            ("fault_evaluation", 15): 200}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            raise ValueError("Empty CSV requires an explicit schema")
        fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def cp95(k, n):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Invalid binomial counts")
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional run values required")
    draw = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 1000):
        count = min(1000, BOOTSTRAP_DRAWS-start)
        draw[start:start+count] = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    lo95, hi95, lo_family, hi_family = np.quantile(draw, [.025, .975, .05/6., 1.-.05/6.], method="linear")
    return {"physical_n": len(values), "estimate": float(values.mean()),
            "ci95_low": float(lo95), "ci95_high": float(hi95),
            "ci98_333_low": float(lo_family), "ci98_333_high": float(hi_family),
            "bootstrap_replicates": BOOTSTRAP_DRAWS, "bootstrap_degenerate": int(lo95 == hi95)}


def paired_difference(outcome, metadata, fault, endpoint):
    # Reuse the verified physical-run contrast map for binary endpoints too.
    return primary_loss_contrast({"loss_minutes": np.asarray(outcome[endpoint], dtype=np.float64)}, metadata, fault)


def require_complete(root, rows):
    required = ("RAW.npz", "CHARTS.npz", "RUN_INFO.json")
    missing = [r["run_id"] for r in rows
               if any(not (root / "runs" / r["run_id"] / name).is_file() for name in required)]
    if missing:
        raise RuntimeError(f"No interim analysis: {len(missing)} planned physical runs lack required files; first missing={missing[0]}")


def verify_release_manifest(root):
    """Verify the publication copy, not a newly asserted experimental freeze."""
    root = root.resolve()
    manifest_path = root / "FILE_MANIFEST.csv"
    if manifest_path.is_symlink():
        raise ValueError("Release manifest must not be a symbolic link")
    with manifest_path.open(encoding="utf-8", newline="") as f:
        entries = list(csv.DictReader(f))
    if not entries or not {"path", "size_bytes", "sha256"}.issubset(entries[0]):
        raise ValueError("Malformed release manifest")
    names = set()
    for row in entries:
        name = row["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in names or name == "FILE_MANIFEST.csv":
            raise ValueError("Unsafe or duplicate release manifest path: " + name)
        names.add(name)
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError("Missing or unsafe release file: " + name)
        if path.stat().st_size != int(row["size_bytes"]) or sha(path) != row["sha256"]:
            raise ValueError("Release file identity differs: " + name)
    expected_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                      and p != manifest_path and "__pycache__" not in p.parts}
    if names != expected_files:
        raise ValueError("Release file inventory differs from FILE_MANIFEST.csv")
    return sha(manifest_path)


def read_manifest(root):
    with (root / "MANIFEST.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = {}
    for row in rows:
        row["fault_id"] = int(row["fault_id"])
        group = (row["role"], row["fault_id"])
        counts[group] = counts.get(group, 0) + 1
        if Path(row["run_id"]).name != row["run_id"]:
            raise ValueError("Unsafe run identifier")
    if counts != EXPECTED or len(rows) != 1199 or len({r["run_id"] for r in rows}) != 1199 or len({r["key_hex"] for r in rows}) != 1199:
        raise ValueError("Manifest differs from the fixed 1199-trajectory panel")
    return rows


def verify_inputs(root):
    release_manifest_sha = verify_release_manifest(root)
    rows = read_manifest(root)
    require_complete(root, rows)  # Reject incomplete panels before reading performance.
    # This identity refers to the original scientific run list, which is unchanged.
    expected_run_manifest = "0388636ba6a83458d78b0c8c58f69d092591905641967d6147769ebc3f5f641a"
    if sha(root / "MANIFEST.csv") != expected_run_manifest:
        raise ValueError("Original scientific run manifest has changed")
    for ordinal, row in enumerate(rows, 1):
        if int(row["ordinal"]) != ordinal or int(row["key_hex"], 16) != (0x983f6de2d135521de3a0b50a << 32) | ordinal:
            raise ValueError("Run ordinal or independent-stream key differs")
    hp = root / "calibration/THRESHOLDS_LOCK.json"
    hlock = read_json(hp)
    if hlock["n_calibration"] != 199 or hlock["order_1based"] != 190:
        raise ValueError("Wrong calibration order/size")
    metadata, _ = load_scorers(root)
    if hlock["arm_metadata"] != metadata or read_json(root / "ARM_METADATA.json") != metadata:
        raise ValueError("Fixed arm metadata differs")
    thresholds = np.asarray(hlock["thresholds"], dtype=np.float64)
    if thresholds.shape != (22,) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid threshold vector")
    cal_rows = [r for r in rows if r["role"] == "normal_calibration"]
    if hlock["calibration_run_ids"] != [r["run_id"] for r in cal_rows]:
        raise ValueError("Calibration role/order differs")
    run_metadata_hashes, shutdowns = {}, []
    for row in rows:
        folder = root / "runs" / row["run_id"]
        info = read_json(folder / "RUN_INFO.json")
        for key in ("run_id", "role", "fault_id", "key_hex"):
            if info[key] != row[key]:
                raise ValueError("Run identity differs: " + row["run_id"])
        if info["rng_kind"] != "philox" or info["rng_key_hex"] != row["key_hex"]:
            raise ValueError("RNG/source role identity differs")
        if info["rows"] != 960 or info["duration_hours"] != 48 or info["integration_step_seconds"] != 1 or info["record_interval_seconds"] != 180:
            raise ValueError("Run horizon or time grid differs")
        # Preserved original hashes describe the historical experimental record,
        # not the sanitized publication metadata or this adapter's code.
        if info["protocol_sha256"] != "ba89eaa7262486f1419452760bb6357643dbeb61235cad542854cb83474a5cf5" or info["protocol_lock_sha256"] != "21a1ec76c68f97b08fe108e61d07e22cc735f4ec5af0b01b439768378e515217":
            raise ValueError("Historical protocol identity differs")
        expected_threshold_identity = None if row["role"] == "normal_calibration" else "3ce41ad2dae91aad4085aca9eb50b8cf7248d8b76569c75d9058090671be9be1"
        if info["thresholds_lock_sha256"] != expected_threshold_identity:
            raise ValueError("Historical calibration identity differs")
        if not isinstance(info["shutdown_occurred"], bool) or info["rng_draw_count"] <= 0:
            raise ValueError("Invalid run diagnostic metadata")
        if info["shutdown_occurred"]:
            shutdowns.append({"run_id": row["run_id"], "role": row["role"], "fault_id": row["fault_id"],
                              "shutdown_step": info.get("shutdown_step"), "retained_in_panel": 1})
        run_metadata_hashes[row["run_id"]] = sha(folder / "RUN_INFO.json")
    maxima = np.stack([read_chart(root, row).max(axis=0) for row in cal_rows])
    if not np.array_equal(np.sort(maxima, axis=0)[189], thresholds):
        raise ValueError("Thresholds do not equal the 190th smallest calibration maxima")
    maxima_file = hp.parent / "CALIBRATION_MAXIMA.npz"
    if sha(maxima_file) != hlock["calibration_maxima_sha256"]:
        raise ValueError("Saved calibration maxima changed")
    return rows, metadata, thresholds, shutdowns, {
        "verification_scope": "Publication-copy file identities and scientific inputs; not a new experimental protocol freeze",
        "release_file_manifest_sha256": release_manifest_sha,
        "run_manifest_sha256": sha(root / "MANIFEST.csv"),
        "public_threshold_metadata_sha256": sha(hp),
        "public_run_metadata_sha256": run_metadata_hashes}


def read_chart(root, row):
    with np.load(root / "runs" / row["run_id"] / "CHARTS.npz", allow_pickle=False) as z:
        chart = z["charts"].copy()
    if chart.shape != (960, 22) or chart.dtype != np.float64 or not np.isfinite(chart).all():
        raise ValueError("Malformed chart: " + row["run_id"])
    return chart


def analyze(root, out):
    rows, metadata, thresholds, shutdowns, identities = verify_inputs(root)
    if out.exists():
        raise FileExistsError("Use a new results directory; existing analysis is not overwritten")
    grouped = {}
    per_run = []
    for role, fault in EXPECTED:
        if role == "normal_calibration":
            continue
        selected = [r for r in rows if r["role"] == role and r["fault_id"] == fault]
        event_list = [events(read_chart(root, row), thresholds) for row in selected]
        ev = {k: np.stack([e[k] for e in event_list]) for k in event_list[0]}
        grouped[(role, fault)] = ev
        relevant = [j for j, m in enumerate(metadata) if role == "normal_evaluation" or m["fault"] == fault]
        for i, row in enumerate(selected):
            for j in relevant:
                per_run.append({"run_id": row["run_id"], "role": role, "fault_id": fault,
                                "arm_id": metadata[j]["arm_id"], "state": metadata[j]["state"],
                                "view": metadata[j]["view"], "normal_alarm": int(ev["normal_alarm"][i,j]),
                                "prefault_alarm": int(ev["pre_alarm"][i,j]), "clean40": int(ev["clean"][i,j]),
                                "clean2": int(ev["clean_120"][i,j]), "clean8": int(ev["clean_480"][i,j]),
                                "delay_minutes": int(ev["delay_minutes"][i,j]), "loss_minutes": int(ev["loss_minutes"][i,j])})
    normal = grouped[("normal_evaluation", 0)]
    summaries = []
    for j, m in enumerate(metadata):
        fault = grouped[("fault_evaluation", m["fault"])]
        row = {"arm_id": m["arm_id"], "fault_id": m["fault"], "state": m["state"], "view": m["view"],
               "threshold": float(thresholds[j]), "normal_physical_n": 400, "fault_physical_n": 200}
        for name, values in [("normal_alarm", normal["normal_alarm"][:,j]), ("normal_post_floor", normal["clean"][:,j]),
                             ("fault_prefault_alarm", fault["pre_alarm"][:,j]), ("clean40", fault["clean"][:,j]),
                             ("clean2", fault["clean_120"][:,j]), ("clean8", fault["clean_480"][:,j])]:
            k, n = int(values.sum()), len(values); lo, hi = cp95(k,n)
            row.update({name+"_count": k, name+"_pct": 100*k/n, name+"_cp95_low_pct": 100*lo, name+"_cp95_high_pct": 100*hi})
        clean = fault["clean"][:,j]
        row.update(mean_loss_minutes=float(fault["loss_minutes"][:,j].mean()),
                   mean_delay_own_clean_minutes=float(fault["delay_minutes"][clean,j].mean()) if clean.any() else None)
        summaries.append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary, secondary = [], []
    for fault_id in (3,15,9):
        ev = grouped[("fault_evaluation",fault_id)]
        name, diff = primary_loss_contrast(ev,metadata,fault_id)
        summary = bootstrap_mean(diff,rng)
        favorable = 1 if fault_id == 3 else -1
        supported = summary["ci98_333_low"] > 0 if favorable == 1 else summary["ci98_333_high"] < 0
        primary.append({"contrast":name,"fault_id":fault_id,"endpoint":"mean_penalized_loss_40h",
                        "unit":"minutes","favorable_sign":favorable,**summary,"supported_by_family_interval":int(supported)})
    for fault_id in (3,15,9):
        ev = grouped[("fault_evaluation",fault_id)]
        for endpoint in ("clean_120","clean_480","clean","pre_alarm"):
            name, diff = paired_difference(ev,metadata,fault_id,endpoint)
            s=bootstrap_mean(100*diff,rng)
            secondary.append({"contrast":name,"fault_id":fault_id,"endpoint":endpoint,"unit":"percentage_points",
                              **{k:v for k,v in s.items() if not k.startswith("ci98_")}})
        for endpoint in ("normal_alarm","clean"):
            name,diff=paired_difference(normal,metadata,fault_id,endpoint)
            s=bootstrap_mean(100*diff,rng)
            secondary.append({"contrast":name,"fault_id":fault_id,"endpoint":"normal_"+endpoint,"unit":"percentage_points",
                              **{k:v for k,v in s.items() if not k.startswith("ci98_")}})
    aggregate=[]
    for fault_id, names in [(3,("mean_pool_signed","same_mean_signed_axis_squared")),(15,("moment_pool_squared","mean_pool_squared"))]:
        ev=grouped[("fault_evaluation",fault_id)]
        for name in names:
            cols=[j for j,m in enumerate(metadata) if m["fault"]==fault_id and m["choice_key"]==name]
            if len(cols)!=5:raise ValueError("Expected all five fixed R5 states")
            for endpoint in ("loss_minutes","clean","clean_120","clean_480"):
                value=ev[endpoint][:,cols].mean(axis=1);scale=1 if endpoint=="loss_minutes" else 100
                s=bootstrap_mean(value*scale,rng)
                aggregate.append({"fault_id":fault_id,"method":name,"endpoint":endpoint,"fixed_states":5,
                                  "unit":"minutes" if scale==1 else "percent",
                                  **{k:v for k,v in s.items() if not k.startswith("ci98_")}})
            s=bootstrap_mean(100*normal["normal_alarm"][:,cols].mean(axis=1),rng)
            aggregate.append({"fault_id":fault_id,"method":name,"endpoint":"normal_alarm","fixed_states":5,"unit":"percent",
                              **{k:v for k,v in s.items() if not k.startswith("ci98_")}})
    out.mkdir(parents=True)
    write_csv(out/"per_run_events.csv",per_run)
    write_csv(out/"arm_summary.csv",summaries)
    write_csv(out/"primary_contrasts.csv",primary)
    write_csv(out/"secondary_contrasts.csv",secondary)
    write_csv(out/"five_state_averages.csv",aggregate)
    write_csv(out/"shutdown_runs.csv",shutdowns,["run_id","role","fault_id","shutdown_step","retained_in_panel"])
    result={"status":"COMPLETE_PUBLIC_PANEL_REANALYZED","written_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "physical_trajectories":1199,"calibration_n":199,"normal_evaluation_n":400,"per_fault_evaluation_n":200,
            "paired_recording_views_count_as_one_trajectory":True,"five_fixed_states_do_not_multiply_n":True,
            "primary_family_size":3,"primary_confidence_level":1-.05/3,"bootstrap_replicates":BOOTSTRAP_DRAWS,
            "bootstrap_seed":BOOTSTRAP_SEED,"primary":primary,"shutdown_run_count":len(shutdowns),
            "interval_scope":"Conditional on frozen models and current independent calibration thresholds; no training/calibration-population uncertainty. Approximate percentile bootstrap.",
            "limitations":["Original failed F9 2h prediction remains failed","Does not retroactively repair original intervals","Does not confirm independent relearning or universal recording effects","Interpret loss contrasts alongside each arm's realized normal FAR"],
            "identities":identities,"analysis_code_sha256":sha(Path(__file__)),
            "csv_sha256":{p.name:sha(p) for p in sorted(out.glob("*.csv"))}}
    write_json(out/"RESULTS.json",result)
    print(json.dumps({"status":result["status"],"physical_trajectories":1199,"primary":primary},indent=2))


def self_test(root):
    metadata,_=load_scorers(root)
    losses=np.zeros((2,22));losses[:,[1,5,9,13,17]]=[[3,6,9,12,15],[30,30,30,30,30]]
    label,d=primary_loss_contrast({"loss_minutes":losses},metadata,3)
    assert np.array_equal(d,[9,30]) and d.shape==(2,)
    losses[:,20]=10;losses[:,21]=2
    _,d9=primary_loss_contrast({"loss_minutes":losses},metadata,9)
    assert np.array_equal(d9,[-8,-8])
    b=bootstrap_mean(np.array([7.,7.]),np.random.default_rng(1))
    assert b["physical_n"]==2 and b["estimate"]==b["ci95_low"]==b["ci95_high"]==7 and b["bootstrap_degenerate"]==1
    assert cp95(0,400)[0]==0 and cp95(400,400)[1]==1
    with tempfile.TemporaryDirectory() as t:
        try:require_complete(Path(t),[{"run_id":"missing"}])
        except RuntimeError:pass
        else:raise AssertionError("Incomplete-panel analysis should fail")
    qa={"all_passed":True,"analysis_code_sha256":sha(Path(__file__)),"scope":"Deterministic in-memory run-unit/direction/interval/before-performance gate fixtures; no simulations",
        "checks":["five_state_within_run_average","F9_CI_minus_native","constant_bootstrap_marked_degenerate","CP_boundary_counts","incomplete_panel_refused"]}
    print(json.dumps(qa,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args();root=args.root.resolve()
    if args.self_test:self_test(root)
    else:
        out = args.out.resolve() if args.out else root.parent / (root.name + "_reanalysis")
        if out == root or out.is_relative_to(root):
            ap.error("Write reanalysis outside the immutable publication package")
        analyze(root, out)


if __name__=="__main__":main()
