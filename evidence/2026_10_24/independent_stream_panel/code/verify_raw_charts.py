"""Verify all stored raw arrays against fixed-model chart calculations.

This is deterministic scoring of stored data, not new process simulation,
model fitting, threshold selection, or independent validation. Publication
file hashes are checked before opening the arrays. Exact equality is reported;
no numerical tolerance is used to hide a mismatch.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import datetime
import json
import platform
from pathlib import Path
import numpy as np
from analyze_confirmation import verify_inputs, read_chart, sha
from score_core import load_scorers


def verify(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("Choose a new verification output path")
    if out == root or out.is_relative_to(root):
        raise ValueError("Write verification output outside the immutable package")
    rows, _, _, _, identities = verify_inputs(root)
    _, score = load_scorers(root)
    mismatches = []
    global_maximum_absolute_difference = 0.0
    for i, row in enumerate(rows, 1):
        raw_path = root / "runs" / row["run_id"] / "RAW.npz"
        with np.load(raw_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        actual = score(arrays)
        expected = read_chart(root, row)
        difference = float(np.max(np.abs(actual - expected)))
        global_maximum_absolute_difference = max(global_maximum_absolute_difference, difference)
        if not np.array_equal(actual, expected):
            mismatches.append({"run_id": row["run_id"], "max_absolute_difference": difference,
                               "differing_elements": int(np.count_nonzero(actual != expected))})
        if i % 100 == 0 or i == len(rows):
            print(f"Checked {i}/{len(rows)} stored trajectories; exact mismatches: {len(mismatches)}", flush=True)
    result = {
        "scope": "Stored RAW arrays through unchanged fixed scoring to stored CHARTS; no new simulations or fitting",
        "checked_trajectories": len(rows),
        "exact_equal_trajectories": len(rows) - len(mismatches),
        "all_exact_equal": not mismatches,
        "maximum_absolute_difference": global_maximum_absolute_difference,
        "mismatches": mismatches,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "publication_file_manifest_sha256": identities["release_file_manifest_sha256"],
        "scoring_code_sha256": sha(root / "code" / "score_core.py"),
        "verification_code_sha256": sha(Path(__file__)),
        "interpretation": "Exact equality is environment-dependent; a mismatch must be investigated, not counted as success."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("checked_trajectories", "exact_equal_trajectories", "all_exact_equal", "maximum_absolute_difference")}, indent=2))
    return not mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.root, args.out) else 1)


if __name__ == "__main__":
    main()
