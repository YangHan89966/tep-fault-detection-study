#!/usr/bin/env python3
"""Reaggregate the released, fixed F15 scores; never fit or simulate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# A fixed thread count also avoids oversubscribing small bootstrap products.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
import numpy as np
import frozen_core as core
import risk_time

RTOL = 1e-12
ATOL = 1e-10
METHODS = ("mean_pool_squared", "moment_pool_squared")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def compare_csv(actual, reference):
    fields_a, rows_a = read_csv(actual)
    fields_b, rows_b = read_csv(reference)
    require(fields_a == fields_b, f"CSV schema differs: {reference.name}")
    require(len(rows_a) == len(rows_b), f"CSV row count differs: {reference.name}")
    maximum_error = 0.0
    for index, (a, b) in enumerate(zip(rows_a, rows_b), 1):
        for field in fields_a:
            if a[field] == b[field]:
                continue
            try:
                x, y = float(a[field]), float(b[field])
            except ValueError:
                raise ValueError(f"CSV text differs: {reference.name}, row {index}, {field}")
            require(math.isfinite(x) and math.isfinite(y) and
                    math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL),
                    f"CSV value differs: {reference.name}, row {index}, {field}")
            maximum_error = max(maximum_error, abs(x - y))
    return {"rows": len(rows_a), "fields": len(fields_a),
            "max_abs_numeric_error": maximum_error, "passed": True}


def docx_tables(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml")).find("w:body", ns)
    found = {}
    pending = None
    for child in body:
        text = "".join(node.text or "" for node in child.findall(".//w:t", ns))
        if child.tag == "{" + ns["w"] + "}p":
            if text.startswith("Table S48."):
                pending = "S48"
            elif text.startswith("Table S49."):
                pending = "S49"
        elif child.tag == "{" + ns["w"] + "}tbl" and pending:
            require(pending not in found, "Duplicate SI table caption")
            found[pending] = [
                ["".join(t.text or "" for t in cell.findall(".//w:t", ns))
                 for cell in row.findall("w:tc", ns)]
                for row in child.findall("w:tr", ns)]
            pending = None
    require(set(found) == {"S48", "S49"}, "S48/S49 not found in SI DOCX")
    return found


def exported_tables():
    result = {}
    for name in ("S48", "S49"):
        with (ROOT / "manuscript_tables" / f"Table_{name}.csv").open(
                encoding="utf-8", newline="") as stream:
            result[name] = list(csv.reader(stream))
    return result


def numbers(text):
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace("−", "-").replace(",", ""))


def check_si(tables, endpoints, contrasts):
    endpoint = {(int(row["rank"]), row["method"]): row for row in endpoints}
    contrast = {(int(row["rank"]), row["metric"]): row for row in contrasts}
    table = tables["S48"]
    require(len(table) == 27 and all(len(row) == 6 for row in table), "S48 shape")
    ranks = []
    for row in table[1:]:
        rank = int(numbers(row[0])[0])
        ranks.append(rank)
        require(("*" in row[0]) == (rank == 247), "S48 baseline marker")
        for cell, metric in zip(row[1:5], ("normal_FAR_pct_count",
                "clean_by_120min_pct_count", "clean_by_480min_pct_count",
                "clean_by_2400min_pct_count")):
            observed = [int(value) for value in numbers(cell)]
            expected = [int(endpoint[rank, method][metric]) for method in METHODS]
            require(observed == expected, f"S48 count differs at rank {rank}: {metric}")
        saved = contrast[rank, "loss_2400min"]
        expected = [f"{float(saved[key]):.3f}" for key in
                    ("estimate", "pointwise95_low", "pointwise95_high")]
        require(numbers(row[5]) == expected, f"S48 loss/CI differs at rank {rank}")
    require(ranks == list(range(234, 260)), "S48 does not preserve all 26 ranks")
    table = tables["S49"]
    require(len(table) == 7 and all(len(row) == 5 for row in table), "S49 shape")
    observed_pairs = []
    for row in table[1:]:
        rank = int(numbers(row[0])[0])
        require(("*" in row[0]) == (rank == 247), "S49 baseline marker")
        require(row[1] in ("Mean / Squared", "Moment / Squared"), "S49 method label")
        method = METHODS[0 if row[1] == "Mean / Squared" else 1]
        observed_pairs.append((rank, method))
        expected = [f"{float(endpoint[rank, method][f'loss_{h}min']):.3f}"
                    for h in (120, 480, 2400)]
        require(row[2:] == expected, f"S49 mean loss differs: rank {rank}, {method}")
    require(observed_pairs == [(r, m) for r in (234, 247, 259) for m in METHODS],
            "S49 rank/method order differs")
    return {"S48_rows": 26, "S48_displayed_numeric_values_checked": 286,
            "S49_rows": 6, "S49_loss_values_checked": 18,
            "comparison": "Exact displayed counts and three-decimal losses/intervals",
            "passed": True}


class QuietLog:
    def msg(self, _message):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--si-docx", type=Path,
                        help="Also compare S48/S49 directly with the supplied SI DOCX")
    parser.add_argument("--report", type=Path,
                        help="Write a JSON report; parent directory must already exist")
    args = parser.parse_args()
    report = {"status": "STARTED", "python": platform.python_version(),
              "numpy": np.__version__, "simulation": False, "refit": False,
              "threshold_selection": False, "raw_trajectory_replay": False,
              "numeric_tolerances": {"rtol": RTOL, "atol": ATOL}}
    manifest = json.loads((ROOT / "PACKAGE_FILES_SHA256.json").read_text())
    for name, expected in manifest.items():
        require(digest(ROOT / name) == expected, f"Package integrity failure: {name}")
    report["package_files_hashed"] = len(manifest)
    original_hashes = json.loads((ROOT / "provenance" /
                                 "ORIGINAL_RETURN_FILES_SHA256.json").read_text())
    results = ROOT / "results"
    for path in results.iterdir():
        require(digest(path) == original_hashes[path.name],
                f"Original returned material changed: {path.name}")
    report["returned_numerical_files_byte_identical"] = len(list(results.iterdir()))
    scores = load_npz(results / "SCORES_FIXED_F15.npz")
    require(set(scores) == {"fault", "normal", "methods", "run_ids", "thresholds", "ranks"},
            "Unexpected saved score keys")
    require(tuple(scores["methods"]) == METHODS, "Method identities/order differ")
    require(np.array_equal(scores["run_ids"], np.arange(1, 251)), "Run identities differ")
    require(np.array_equal(scores["ranks"], np.arange(234, 260)), "Rank grid differs")
    for role in ("normal", "fault"):
        require(scores[role].shape == (250, 960, 2) and
                scores[role].dtype == np.dtype("float64") and
                np.isfinite(scores[role]).all(), f"Invalid {role} score paths")
    protocol = json.loads((results / "PROTOCOL_FIXED.json").read_text())
    assets = ROOT / "frozen_assets"
    inner, outer, baseline = (load_npz(assets / f"{name}.npz")
                               for name in ("INNER", "OUTER", "BASELINE"))
    choices = json.loads((assets / "SELECTION.json").read_text())["choices"]
    pick = [choices[method] for method in METHODS]
    require(np.array_equal(np.quantile(inner["anchor_peaks"], .9, axis=0,
                                      method="linear"), inner["anchors"]),
            "Saved anchor quantile differs")
    require(np.array_equal(core.thresholds(outer["peaks"], inner["anchors"], 247)[0],
                           outer["thresholds"]), "Original 49 thresholds differ")
    recalculated_limits = np.stack([core.thresholds(outer["peaks"][:, pick],
        inner["anchors"][pick], int(rank))[0] for rank in scores["ranks"]])
    require(np.array_equal(recalculated_limits, scores["thresholds"]),
            "Saved 52 thresholds differ from frozen outer-calibration order statistics")
    _, threshold_rows = read_csv(results / "THRESHOLDS_FIXED_BEFORE_TEST.csv")
    require(len(threshold_rows) == 52, "Threshold CSV count")
    for row, (rank, method) in zip(threshold_rows,
            [(int(r), m) for r in scores["ranks"] for m in METHODS]):
        require(int(row["rank"]) == rank and row["method"] == method and
                float(row["threshold"]) == recalculated_limits[rank - 234, METHODS.index(method)]
                and float(row["k"]) == .05 and int(row["outer_n"]) == 259
                and float(row["rank_reference_probability"]) == (260 - rank) / 260
                and row["original_DV_point"] == str(rank == 247), "Threshold CSV differs")
    report["frozen_calibration"] = {"saved_anchor_values_exact": 49,
        "original_thresholds_exact": 49, "risk_time_thresholds_exact": 52}
    base_ix = [list(baseline["methods"]).index(method) for method in METHODS]
    for role in ("normal", "fault"):
        mask = scores[role] > scores["thresholds"][247 - 234]
        require(np.array_equal(np.packbits(mask.transpose(2, 0, 1), axis=2),
                               baseline[f"{role}_masks"][base_ix]),
                f"Original rank-247 {role} bitmap differs")
        if role == "fault":
            for key, value in core.events(mask).items():
                require(np.array_equal(value, baseline[key][:, base_ix]),
                        f"Original rank-247 event differs: {key}")
    report["original_baseline"] = {"methods_rechecked": 2,
        "alarm_bits_exact": 960000, "fault_event_fields_exact": 7,
        "scope": "Two released score branches only; original upstream nine-branch checks retained separately"}
    print("Integrity, all 52 fixed thresholds, and two-branch baseline bitmaps: PASS", flush=True)
    with tempfile.TemporaryDirectory(prefix="tep_f15_saved_recheck_") as directory:
        generated = Path(directory)
        rows, contrasts = risk_time.sensitivity(scores["fault"], scores["normal"],
            scores["thresholds"], scores["ranks"], protocol, generated, QuietLog())
        require((rows, contrasts) == (52, 260), "Reaggregation dimensions differ")
        report["reaggregated_csvs"] = {name: compare_csv(generated / name, results / name)
            for name in ("RISK_TIME_ENDPOINTS.csv", "FAULT_PER_RUN.csv", "NORMAL_PER_RUN.csv",
                         "CLEAN_DETECTION_CURVES.csv", "PAIRED_RANK_CONTRASTS.csv")}
        require(json.loads((generated / "BOOTSTRAP_SPEC.json").read_text()) ==
                json.loads((results / "BOOTSTRAP_SPEC.json").read_text()), "Bootstrap specification differs")
        regenerated_scores = load_npz(generated / "SCORES_FIXED_F15.npz")
        require(set(scores) == set(regenerated_scores) and all(np.array_equal(
            value, regenerated_scores[key]) for key, value in scores.items()), "Saved scores changed")
    _, endpoint_rows = read_csv(results / "RISK_TIME_ENDPOINTS.csv")
    _, contrast_rows = read_csv(results / "PAIRED_RANK_CONTRASTS.csv")
    report["si_export_check"] = check_si(exported_tables(), endpoint_rows, contrast_rows)
    if args.si_docx:
        report["si_docx_check"] = check_si(docx_tables(args.si_docx), endpoint_rows, contrast_rows)
        report["si_docx_sha256"] = digest(args.si_docx)
    report["negative_early_results"] = {
        "all_52_rows_zero_prefault_alarms": all(int(r["prefault_alarm_pct_count"]) == 0 for r in endpoint_rows),
        "all_52_rows_zero_60min_clean_detections": all(int(r["clean_by_60min_pct_count"]) == 0 for r in endpoint_rows),
        "mean_max_120min_clean_count": max(int(r["clean_by_120min_pct_count"]) for r in endpoint_rows if r["method"] == METHODS[0]),
        "moment_max_120min_clean_count": max(int(r["clean_by_120min_pct_count"]) for r in endpoint_rows if r["method"] == METHODS[1])}
    report["status"] = "PASS_SAVED_SCORES_ALL_52_ROWS_26_RANKS_AND_SI_S48_S49"
    if args.report:
        require(not args.report.exists(), "Report already exists; choose a new path")
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
