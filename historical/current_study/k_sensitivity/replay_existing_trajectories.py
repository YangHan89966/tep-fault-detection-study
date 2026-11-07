"""Portable entry point for the extracted fixed-state K sensitivity functions.

Requires original, hash-matching process trajectories supplied separately.
This entry point has been checked for import/inspection, not raw-data execution.
It never simulates trajectories or fits models. All outputs use a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))
import scientific_runner as core


def verify_source():
    manifest = json.loads((SOURCE / "EXTRACTION_MANIFEST.json").read_text())
    for record in manifest["exact_files"]:
        actual = hashlib.sha256((SOURCE / record["path"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError("Source/asset hash mismatch: " + record["path"])
    actual = hashlib.sha256((SOURCE / "scientific_runner.py").read_bytes()).hexdigest()
    if actual != manifest["scientific_runner_sha256"]:
        raise ValueError("Extracted scientific_runner.py hash mismatch")
    return len(manifest["exact_files"]) + 1


def explicit_n_root(stage, explicit):
    if explicit is None:
        raise ValueError("N requires explicit development and final roots")
    root = Path(explicit)
    role = "normal_validation" if stage == "development" else "normal_evaluation"
    if not (root / role / "fault_00/run_0001/values_52.npy").is_file():
        raise FileNotFoundError("Missing expected N role: " + role)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Verify packaged source/assets; read no trajectories")
    parser.add_argument("--only", choices=("N", "DV", "both"), default="both")
    parser.add_argument("--auto-k", action="store_true", help="Include the original five-candidate development selection")
    parser.add_argument("--n-development-root", type=Path)
    parser.add_argument("--n-final-root", type=Path)
    parser.add_argument("--dv-root", type=Path, help="Root containing the original DV data directory")
    parser.add_argument("--output", type=Path, help="New output directory; must not exist")
    args = parser.parse_args()
    count = verify_source()
    if args.inspect:
        print(json.dumps({"source_and_asset_files_verified": count,
            "raw_trajectories_included": False,
            "runtime_raw_replay_verified": False,
            "frozen_models_and_input_hash_indexes_included": True,
            "required_trajectory_counts": {"N_fixed_k": 4750, "N_with_auto_k": 9000,
                "DV_fixed_k": 1291, "DV_with_auto_k": 1575}}, indent=2))
        return 0
    targets = ("N", "DV") if args.only == "both" else (args.only,)
    if args.output is None:
        parser.error("--output is required for a raw scoring replay")
    if "N" in targets:
        if args.n_development_root is None or args.n_final_root is None:
            parser.error("N requires --n-development-root and --n-final-root")
        explicit_n_root("development", args.n_development_root)
        explicit_n_root("final", args.n_final_root)
    if "DV" in targets:
        if args.dv_root is None or not (args.dv_root / "data").is_dir():
            parser.error("DV requires --dv-root pointing to a root containing data/")
    args.output.mkdir(parents=True, exist_ok=False)
    # The original N procedure invokes this name. Only path resolution is supplied here.
    core.find_n_root = explicit_n_root
    assets = core.Assets(SOURCE)
    run = core.Run(args.output)
    results = {}
    try:
        for study in targets:
            results[study] = (core.do_n if study == "N" else core.do_dv)(assets, args, run)
        core.write_json(args.output / "SUMMARY.json", results)
        core.write_json(args.output / "BASELINE_CHECKS.json", run.checks)
        core.write_csv(args.output / "DELTA_007_MINUS_005.csv", run.deltas)
        if args.auto_k:
            core.write_csv(args.output / "AUTO_MINUS_005.csv", core.delta_rows(run.rows, "auto_dev"))
        core.write_json(args.output / "PORTABLE_REPLAY_STATUS.json", {
            "complete_for_requested_scope": True, "requested_studies": list(targets),
            "auto_k": args.auto_k, "trajectory_reads": run.read_counts,
            "baseline_checks_passed": len(run.checks), "numpy": core.np.__version__,
            "scope": "Existing-trajectory fixed-state scoring only; no simulator or model fitting",
            "entry_point": "Portable wrapper; extracted scientific functions unchanged"})
    finally:
        run.log_file.close()
    print("Existing-trajectory scoring completed for " + ", ".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
