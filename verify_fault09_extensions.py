"""Check frozen Fault 9 files and optionally replay saved evaluation results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
F9 = ROOT / "analyses/fault09_relearning"
RUST = ROOT / "analyses/rust_fixed_monitor"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_files():
    complete = F9 / "study_complete"
    code = read_json(F9 / "locks/CODE_LOCK.json")
    for name, expected in code["files"].items():
        require(digest(F9 / name) == expected, f"Changed Fault 9 source: {name}")
    final = read_json(complete / "FINAL_COMPLETE.json")
    for name, expected in final["all_outputs"].items():
        require(digest(complete / "outcomes" / name) == expected,
                f"Changed Fault 9 result: {name}")
    for field, name in (("models_lock_sha", "MODELS_LOCK.json"),
                        ("thresholds_lock_sha", "THRESHOLDS_LOCK.json"),
                        ("receipts_sha", "RAW_RECEIPTS.csv")):
        require(digest(complete / name) == final[field], f"Changed binding: {name}")
    with (complete / "RAW_RECEIPTS.csv").open(encoding="utf-8-sig", newline="") as handle:
        receipts = list(csv.DictReader(handle))
    require(len(receipts) == len({r["case_id"] for r in receipts}) == 2719,
            "Expected 2,719 unique recorded-run identities")
    archive = F9 / "F9_PAIRED_RELEARNING_COMPLETE.zip"
    require(digest(archive) == "7839254b073d226faeb7c7dcd7ac9e2f2b37ea19ed0e30b7981722fd5ff20e2b",
            "Changed completed-results archive")
    limits = read_json(RUST / "limits.json")
    require(digest(RUST / "FIXED_AFFINE_MAPS.npz") == limits["model_sha256"],
            "Changed Rust fixed score")
    require(digest(RUST / "protocol.json") == limits["protocol_sha256"],
            "Changed Rust protocol")
    return {"status": "PASS", "locked_code_files": len(code["files"]),
            "locked_outcome_files": len(final["all_outputs"]),
            "recorded_run_identities": len(receipts)}


def replay_saved_results():
    with tempfile.TemporaryDirectory(prefix="tep-fault09-replay-") as temporary:
        output = Path(temporary) / "relearning.json"
        subprocess.run([sys.executable, "-B", str(F9 / "replay_relearning.py"),
                        str(F9 / "F9_PAIRED_RELEARNING_COMPLETE.zip"),
                        "--out", str(output)], check=True)
        relearning = read_json(output)
    require(relearning["status"] == "PASS" and relearning["physical_keys"] == 2719
            and relearning["evaluation_rows"] == 12000
            and len(relearning["all_14_contrasts"]) == 14,
            "Incomplete relearning replay")
    rust = json.loads(subprocess.run(
        [sys.executable, "-B", str(RUST / "check_rust_results.py")],
        check=True, capture_output=True, text=True).stdout)
    require(rust["status"] == "PASS"
            and rust["roles"] == {"cal": 59, "normal": 60, "fault": 75}
            and rust["fault40"] == {"both": 21, "controller_input_only": 49,
                                     "native_only": 0, "neither": 5},
            "Incomplete Rust endpoint replay")
    return {"status": "PASS", "scope": "Saved models, calibration and run-level outcomes",
            "relearning_evaluation_rows": relearning["evaluation_rows"],
            "relearning_contrasts": len(relearning["all_14_contrasts"]),
            "paired_bootstrap_draws": relearning["paired_bootstrap_draws"],
            "rust_runs": sum(rust["roles"].values()), "rust_fault40": rust["fault40"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true",
                        help="Recount saved outcomes and repeat the paired bootstrap (requires NumPy).")
    args = parser.parse_args()
    result = {"files": check_files()}
    if args.replay:
        result["saved_result_replay"] = replay_saved_results()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
