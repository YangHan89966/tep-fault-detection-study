"""Verify packaged source identities and original k=.05 baseline alarm arrays.

This uses saved outcomes and frozen assets only. It never reads raw trajectories.
Run: python -B verify_saved_sources.py
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
SAVED = ROOT / "saved_results"
FAULTS = (1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20)


def main():
    manifest = json.loads((SOURCE / "EXTRACTION_MANIFEST.json").read_text())
    source_count = 0
    for row in manifest["exact_files"]:
        if hashlib.sha256((SOURCE / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source/asset hash mismatch: " + row["path"])
        source_count += 1
    if hashlib.sha256((SOURCE / "scientific_runner.py").read_bytes()).hexdigest() != manifest["scientific_runner_sha256"]:
        raise ValueError("Extracted scientific source hash mismatch")
    original_s18_members = 0
    for line in (SAVED / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(None, 1)
        name = name.lstrip(" *")
        if hashlib.sha256((SAVED / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Original S18 member hash mismatch: " + name)
        original_s18_members += 1
    bits = {"N": 0, "DV": 0}
    thresholds = {"N": 0, "DV": 0}
    for study, faults in (("N", FAULTS), ("DV", (3, 9, 15))):
        for fault in faults:
            with np.load(SAVED / f"masks/{study}_F{fault:02d}_PAIRED.npz", allow_pickle=False) as current, \
                 np.load(SOURCE / f"assets/{study}/F{fault:02d}/BASELINE.npz", allow_pickle=False) as old:
                if current["policies"].tolist()[0] != "k005":
                    raise ValueError("Unexpected original-policy position")
                for key in ("fault", "normal"):
                    observed = current[key + "_masks"][0]
                    expected = old[key + "_alarm_packbits"][:2] if study == "N" else old[key + "_masks"]
                    if not np.array_equal(observed, expected):
                        raise ValueError(f"Original alarm mismatch: {study}/{fault}/{key}")
                    bits[study] += observed.size * 8
                if study == "DV":
                    if not np.array_equal(current["methods"], old["methods"]):
                        raise ValueError("DV method order mismatch")
                    with np.load(SOURCE / f"assets/DV/F{fault:02d}/OUTER.npz", allow_pickle=False) as outer:
                        expected_h = outer["thresholds"][current["candidate_indices"]]
                        if not np.array_equal(current["thresholds"][0], expected_h):
                            raise ValueError("DV original threshold mismatch")
                        thresholds[study] += len(expected_h)
                else:
                    expected_h = [json.loads((SOURCE / f"assets/N/F{fault:02d}/{method}.json").read_text())["control_limit"]
                                  for method in current["methods"].tolist()]
                    if not np.array_equal(current["thresholds"][0], expected_h):
                        raise ValueError("N original threshold mismatch")
                    thresholds[study] += len(expected_h)
    result = {"passed": True, "exact_source_asset_files_checked": source_count,
        "extracted_scientific_module_checked": True, "unchanged_S18_members_checked": original_s18_members,
        "original_k005_alarm_bits_checked": bits, "original_k005_threshold_values_checked": thresholds,
        "scope": "Saved masks and frozen assets; no raw-to-score replay, simulation, fitting or threshold changes"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
