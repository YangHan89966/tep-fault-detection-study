"""Fixed six-arm graded measurement-noise coupling analysis.

This script analyzes already generated scores. It never fits a representation,
selects a direction, simulates a process, or changes a source score file.
Use `calibrate` first, then `evaluate`; the second stage requires the first-stage
threshold lock and verifies its input and implementation identities.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import numpy as np

ARMS = ("rho_0", "rho_025", "rho_05", "rho_075", "rho_1", "actual_post_update")
RHOS = np.array([0., .25, .5, .75, 1.])
N_TIME = 960
PRE_TIME = 160
K = .05
RANK = 190
COUNTS = {"normal_calibration": 199, "normal_evaluation": 400, "fault_evaluation": 200}
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2026103009
ENDPOINTS = ("clean40", "clean8", "clean2", "loss_minutes", "normal_alarm", "normal_floor")
CONTRASTS = ((0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (0, 5))
PARAMETERS = {
    "arms": list(ARMS), "n_time": N_TIME, "pre_time": PRE_TIME,
    "k": K, "calibration_ascending_rank_1based": RANK,
    "role_counts": COUNTS, "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_seed": BOOTSTRAP_SEED, "alarm_rule": "strict_C_greater_than_h",
    "recurrence": "cp=max(0,(cp+score)-k); cm=max(0,(cm-score)-k)",
    "reset": "beginning_of_trajectory_only", "failure_penalty_minutes": 2400,
    "bootstrap_percentiles": [2.5, 97.5], "quantile_method": "linear",
    "bootstrap_index_order": "normal_indices_then_fault_indices",
    "bootstrap_unit": "whole_trajectory_paired_across_all_six_arms",
    "contrast_direction": "first_arm_minus_second_arm",
    "rates_in_contrasts": "percentage_points", "score_variance": "pooled_normal_evaluation_ddof1",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def manifest(root):
    path = root / "MANIFEST.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"run_id", "role", "fault_id", "index", "ordinal", "key_hex"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Incomplete manifest columns")
    for row in rows:
        for field in ("fault_id", "index", "ordinal"):
            row[field] = int(row[field])
        role, run_id = row["role"], row["run_id"]
        if role not in COUNTS or row["fault_id"] != (9 if role == "fault_evaluation" else 0):
            raise ValueError("Unexpected role/fault in coupling manifest: " + run_id)
        if not run_id or Path(run_id).name != run_id or any(c in run_id for c in ("/", "\\", ":")):
            raise ValueError("Unsafe run identifier")
    for field in ("run_id", "ordinal", "key_hex"):
        if len({r[field] for r in rows}) != len(rows):
            raise ValueError("Duplicate manifest " + field)
    counts = {role: sum(r["role"] == role for r in rows) for role in COUNTS}
    if counts != COUNTS:
        raise ValueError("Fixed panel allocation differs: " + repr(counts))
    return sorted(rows, key=lambda row: row["ordinal"])


def identities(root):
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    return {"study_id": protocol.get("study_id"), "protocol_sha256": sha(root / "PROTOCOL.json"),
            "code_lock_sha256": sha(root / "CODE_LOCK.json"),
            "manifest_sha256": sha(root / "MANIFEST.csv"),
            "analysis_code_sha256": sha(Path(__file__)), "parameters": PARAMETERS,
            "numpy_version": np.__version__, "python_version": platform.python_version()}


def load_scores(root, rows):
    scores, files = [], []
    for row in rows:
        path = root / "scores" / (row["run_id"] + ".npz")
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Missing or unsafe score path: " + str(path))
        file_hash = sha(path)
        with np.load(path, allow_pickle=False) as z:
            a = z["scores"].copy()
            names = tuple(str(x) for x in z["arm_names"])
        if a.shape != (N_TIME, len(ARMS)) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("Malformed score array: " + row["run_id"])
        if names != ARMS:
            raise ValueError("Score arm order differs: " + row["run_id"])
        scores.append(a)
        files.append({"run_id": row["run_id"], "path": path.relative_to(root).as_posix(), "sha256": file_hash})
    return np.stack(scores), files


def charts(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1:] != (N_TIME, len(ARMS)) or not np.isfinite(scores).all():
        raise ValueError("Expected finite (runs,960,6) scores")
    cp = np.zeros((scores.shape[0], scores.shape[2]))
    cm = cp.copy()
    out = np.empty_like(scores)
    for t in range(N_TIME):
        cp = np.maximum(0., (cp + scores[:, t]) - K)
        cm = np.maximum(0., (cm - scores[:, t]) - K)
        out[:, t] = np.maximum(cp, cm)
    return out


def events(values, thresholds):
    values = np.asarray(values, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (N_TIME, len(ARMS)) or not np.isfinite(values).all():
        raise ValueError("Expected finite (runs,960,6) chart values")
    if thresholds.shape != (len(ARMS),) or not np.isfinite(thresholds).all():
        raise ValueError("Expected six finite thresholds")
    hit = values > thresholds[None, None, :]
    pre = hit[:, :PRE_TIME].any(axis=1)
    post = hit[:, PRE_TIME:]
    detected = post.any(axis=1)
    delay = np.where(detected, post.argmax(axis=1) * 3, 2400)
    clean = detected & ~pre
    return {"any_alarm": hit.any(axis=1), "pre_alarm": pre,
            "post_alarm": detected, "clean40": clean,
            "clean8": clean & (delay <= 480), "clean2": clean & (delay <= 120),
            "first_post_delay_minutes": delay,
            "loss_minutes": np.where(clean, delay, 2400)}


def wilson(count, n):
    if n <= 0 or not 0 <= count <= n:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (0. if count == 0 else max(0., centre - half),
            1. if count == n else min(1., centre + half))


def original_calibration_replay(root, rows, peaks, thresholds):
    """Check the two unchanged arms against the archived 22-arm calibration."""
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol.get("synthetic_fixture") is True:
        return {"status": "SKIPPED_SYNTHETIC_FIXTURE_ONLY", "scientific_result": False}
    old_lock_path = root / "reference" / "THRESHOLDS_LOCK.json"
    old_maxima_path = root / "reference" / "CALIBRATION_MAXIMA.npz"
    old = json.loads(old_lock_path.read_text(encoding="utf-8"))
    if old["n_calibration"] != 199 or old["order_1based"] != 190:
        raise ValueError("Archived calibration design differs")
    if old["calibration_run_ids"] != [r["run_id"] for r in rows]:
        raise ValueError("Archived calibration case order differs")
    if sha(old_maxima_path) != old["calibration_maxima_sha256"]:
        raise ValueError("Archived calibration maxima hash differs")
    with np.load(old_maxima_path, allow_pickle=False) as z:
        old_maxima = z["maxima"]
    if old_maxima.shape != (199, 22):
        raise ValueError("Archived calibration maxima shape differs")
    original_columns, current_columns = [21, 20], [4, 5]
    old_h = np.asarray(old["thresholds"], dtype=np.float64)[original_columns]
    peak_difference = peaks[:, current_columns] - old_maxima[:, original_columns]
    h_difference = thresholds[current_columns] - old_h
    report = {"status": "PASS" if not np.any(peak_difference) and not np.any(h_difference) else "FAIL",
              "arms": [ARMS[i] for i in current_columns], "original_columns": original_columns,
              "current_columns": current_columns, "n_calibration": 199, "ascending_rank_1based": 190,
              "maximum_absolute_peak_difference": float(np.max(np.abs(peak_difference))),
              "maximum_absolute_threshold_difference": float(np.max(np.abs(h_difference))),
              "exact_peak_equality": bool(np.array_equal(peaks[:, current_columns], old_maxima[:, original_columns])),
              "exact_threshold_equality": bool(np.array_equal(thresholds[current_columns], old_h)),
              "reference_lock_sha256": sha(old_lock_path), "reference_maxima_sha256": sha(old_maxima_path)}
    write_json(root / "results" / "ORIGINAL_CALIBRATION_REPLAY.json", report)
    if report["status"] != "PASS":
        raise ValueError("Unchanged endpoint calibration failed exact archived replay")
    return report


def calibrate(root):
    root = Path(root).resolve(strict=True)
    rows = [r for r in manifest(root) if r["role"] == "normal_calibration"]
    scores, files = load_scores(root, rows)
    cal_charts = charts(scores)
    peaks = cal_charts.max(axis=1)
    thresholds = np.sort(peaks, axis=0)[RANK - 1]
    lock_path = root / "results" / "THRESHOLDS_LOCK.json"
    binding = identities(root)
    if lock_path.exists():
        old = json.loads(lock_path.read_text(encoding="utf-8"))
        if old["binding"] != binding or old["calibration_score_files"] != files:
            raise ValueError("Existing threshold lock has changed inputs or implementation")
        if old["thresholds"] != thresholds.tolist():
            raise ValueError("Thresholds do not reproduce the existing lock")
        if sha(root / "results" / "CALIBRATION.npz") != old["calibration_npz_sha256"]:
            raise ValueError("Existing calibration artifact hash differs")
        return old
    replay = original_calibration_replay(root, rows, peaks, thresholds)
    target = root / "results" / "CALIBRATION.npz"
    write_npz(target, arm_names=np.asarray(ARMS), run_ids=np.asarray([r["run_id"] for r in rows]),
              peaks=peaks, thresholds=thresholds, ascending_rank_1based=np.array(RANK))
    lock = {"status": "LOCKED", "created_utc": utc(), "binding": binding,
            "thresholds": thresholds.tolist(), "arm_names": list(ARMS),
            "original_calibration_replay": replay,
            "calibration_score_files": files, "calibration_npz_sha256": sha(target),
            "evaluation_results_computed_by_this_script_before_lock": False}
    write_json(lock_path, lock)
    return lock


def verify_lock(root):
    path = root / "results" / "THRESHOLDS_LOCK.json"
    if not path.is_file():
        raise ValueError("Calibrate and lock thresholds before evaluation")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED" or lock["binding"] != identities(root):
        raise ValueError("Threshold lock binding differs")
    expected_calibration_ids = [r["run_id"] for r in manifest(root) if r["role"] == "normal_calibration"]
    if [r["run_id"] for r in lock["calibration_score_files"]] != expected_calibration_ids:
        raise ValueError("Calibration lock case order or identities differ from the manifest")
    for record in lock["calibration_score_files"]:
        source = root / record["path"]
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("Unsafe calibration score path")
        if sha(source) != record["sha256"]:
            raise ValueError("A calibration score file changed after locking")
    if len(lock["calibration_score_files"]) != COUNTS["normal_calibration"]:
        raise ValueError("Incomplete calibration input lock")
    if sha(root / "results" / "CALIBRATION.npz") != lock["calibration_npz_sha256"]:
        raise ValueError("Calibration artifact changed after locking")
    thresholds = np.asarray(lock["thresholds"], dtype=np.float64)
    if thresholds.shape != (len(ARMS),) or not np.isfinite(thresholds).all():
        raise ValueError("Threshold lock requires six finite thresholds")
    with np.load(root / "results" / "CALIBRATION.npz", allow_pickle=False) as archived_calibration:
        if not np.array_equal(thresholds, archived_calibration["thresholds"]):
            raise ValueError("Threshold lock values differ from the verified calibration artifact")
    replay = lock["original_calibration_replay"]
    if replay["status"] == "PASS":
        if sha(root / "reference" / "THRESHOLDS_LOCK.json") != replay["reference_lock_sha256"]:
            raise ValueError("Archived calibration reference lock changed")
        if sha(root / "reference" / "CALIBRATION_MAXIMA.npz") != replay["reference_maxima_sha256"]:
            raise ValueError("Archived calibration reference maxima changed")
    elif replay["status"] != "SKIPPED_SYNTHETIC_FIXTURE_ONLY":
        raise ValueError("No passing archived calibration replay")
    return lock


def bootstrap(normal, fault, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Common whole-run indices for all arms; differences formed before averaging."""
    nn, nf = normal["any_alarm"].shape[0], fault["clean40"].shape[0]
    rng = np.random.default_rng(seed)
    ni = rng.integers(0, nn, size=(draws, nn), dtype=np.int32)
    fi = rng.integers(0, nf, size=(draws, nf), dtype=np.int32)
    distributions = np.empty((draws, len(CONTRASTS), len(ENDPOINTS)), dtype=np.float64)
    estimates = np.empty((len(CONTRASTS), len(ENDPOINTS)), dtype=np.float64)
    for e, endpoint in enumerate(ENDPOINTS):
        if endpoint == "normal_alarm":
            source, idx = normal["any_alarm"], ni
        elif endpoint == "normal_floor":
            source, idx = normal["clean40"], ni
        else:
            source, idx = fault[endpoint], fi
        scale = 1. if endpoint == "loss_minutes" else 100.
        # Cast before subtracting Boolean/integer arrays.
        source = source.astype(np.float64)
        difference = np.stack([source[:, a] - source[:, b] for a, b in CONTRASTS], axis=1) * scale
        estimates[:, e] = difference.mean(axis=0)
        for start in range(0, draws, 500):
            end = min(start + 500, draws)
            distributions[start:end, :, e] = difference[idx[start:end]].mean(axis=1)
    bounds = np.quantile(distributions, [.025, .975], axis=0, method="linear")
    return {"normal_indices": ni, "fault_indices": fi, "estimates": estimates,
            "bounds": bounds, "distributions": distributions}


def main_rows(normal, fault, scores, thresholds):
    n, f = normal["any_alarm"].shape[0], fault["clean40"].shape[0]
    means = scores.mean(axis=(0, 1))
    variances = scores.var(axis=(0, 1), ddof=1)
    rows = []
    for j, arm in enumerate(ARMS):
        row = {"arm": arm, "rho": float(RHOS[j]) if j < len(RHOS) else "",
               "threshold": float(thresholds[j]), "n_fault": f, "n_normal": n,
               "normal_score_mean": float(means[j]), "normal_score_variance_ddof1": float(variances[j])}
        for label, source, count_n in (("clean40", fault["clean40"], f), ("clean8", fault["clean8"], f),
                                        ("clean2", fault["clean2"], f), ("prefault_alarm", fault["pre_alarm"], f),
                                        ("normal_alarm", normal["any_alarm"], n), ("normal_floor", normal["clean40"], n)):
            count = int(source[:, j].sum())
            low, high = wilson(count, count_n)
            row.update({label + "_n": count, label + "_percent": count / count_n * 100,
                        label + "_wilson95_low_percent": low * 100, label + "_wilson95_high_percent": high * 100})
        row["mean_penalized_loss_minutes"] = float(fault["loss_minutes"][:, j].mean())
        rows.append(row)
    return rows


def paired_rows(boot):
    rows = []
    for c, (a, b) in enumerate(CONTRASTS):
        for e, endpoint in enumerate(ENDPOINTS):
            rows.append({"contrast": ARMS[a] + "_minus_" + ARMS[b], "first_arm": ARMS[a],
                         "second_arm": ARMS[b], "endpoint": endpoint,
                         "unit": "minutes" if endpoint == "loss_minutes" else "percentage_points",
                         "estimate": float(boot["estimates"][c, e]),
                         "ci95_low": float(boot["bounds"][0, c, e]),
                         "ci95_high": float(boot["bounds"][1, c, e]),
                         "interval_scope": "marginal_descriptive_conditional_paired_percentile"})
    return rows


def per_run_rows(normal, fault, normal_rows, fault_rows):
    output = []
    for source, rows in ((normal, normal_rows), (fault, fault_rows)):
        for i, run in enumerate(rows):
            for j, arm in enumerate(ARMS):
                row = {"run_id": run["run_id"], "role": run["role"], "fault_id": run["fault_id"],
                       "arm": arm}
                row.update({name: int(values[i, j]) for name, values in source.items()})
                output.append(row)
    return output


def draw_figure(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), layout="constrained")
    blue, orange, black, grey = "#176B9A", "#C0641E", "#26343D", "#65737E"
    graded, actual = rows[:5], rows[5]
    axes[0].plot(RHOS, [r["normal_score_variance_ddof1"] for r in graded], color=black, marker="o", lw=1.6)
    axes[0].axhline(actual["normal_score_variance_ddof1"], color=grey, ls="--", lw=1.2, label="Actual post-update")
    axes[0].set_ylabel("Normal score variance")
    axes[0].set_title("(a) Normal-score variability", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    for label, color, marker, title in (("clean40", blue, "o", "40 h"), ("clean8", orange, "s", "8 h")):
        y = np.array([r[label + "_percent"] for r in graded])
        lo = np.array([r[label + "_wilson95_low_percent"] for r in graded])
        hi = np.array([r[label + "_wilson95_high_percent"] for r in graded])
        axes[1].errorbar(RHOS, y, yerr=np.vstack((y-lo, hi-y)), fmt=marker+"-", color=color,
                         lw=1.6, capsize=3, ms=4.5, label=title + " graded coupling")
        axes[1].axhline(actual[label + "_percent"], color=color, ls="--", lw=1.1, alpha=.7,
                        label=title + " actual post-update")
    axes[1].set_ylim(-2, 102)
    axes[1].set_ylabel("Clean detection (%)")
    axes[1].set_title("(b) Fault 9 detection", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="best", fontsize=7.5)
    label = "normal_alarm"
    y = np.array([r[label + "_percent"] for r in graded])
    lo = np.array([r[label + "_wilson95_low_percent"] for r in graded])
    hi = np.array([r[label + "_wilson95_high_percent"] for r in graded])
    axes[2].errorbar(RHOS, y, yerr=np.vstack((y-lo, hi-y)), fmt="o-", color=black, lw=1.6, capsize=3, ms=4.5)
    axes[2].axhline(actual[label + "_percent"], color=grey, ls="--", lw=1.2, label="Actual post-update")
    axes[2].axhline(5., color="#AEB7BD", ls=":", lw=1., label="Nominal 5% calibration")
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("Normal trajectories with an alarm (%)")
    axes[2].set_title("(c) Normal 48 h alarms", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, loc="best", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel(r"Shared-noise fraction, $\rho$")
        ax.set_xticks(RHOS, ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(-.04, 1.04)
        ax.grid(axis="y", color="#E1E6EA", lw=.6, zorder=0)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(destination / ("Graded_coupling." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    caption = (
        "Graded measurement-noise coupling for the fixed historical Fault 9 monitor. "
        "Five shared-noise fractions retain 0, 3, 6, 9 or 12 of the controller measurement's "
        "twelve uniform noise components; the other components use an independent keyed stream. "
        "Each trajectory uses the same physical process and controller output across all fractions. "
        "The five fractions use the controller-input measurement time; actual post-update data "
        "are plotted separately as horizontal references, not as the rho=0 endpoint. "
        "(a) Pooled score variance over all 960 samples of 400 normal evaluation trajectories "
        "(sample variance, ddof=1). (b) Clean Fault 9 detection within 40 h and 8 h among 200 "
        "fault trajectories. (c) Any alarm over 48 h among 400 normal trajectories. Each arm "
        "has its own threshold, the 190th ascending maximum among the same 199 normal "
        "calibration trajectories. Error bars are marginal 95% Wilson intervals; connecting "
        "segments guide the eye and do not impose monotonicity. This is a post hoc sensitivity "
        "analysis of an already analyzed independent-stream panel, conditional on the fixed "
        "representation and realized calibration.\n")
    (destination / "Figure_caption.txt").write_text(caption, encoding="utf-8")


def evaluate(root, make_figure=True):
    root = Path(root).resolve(strict=True)
    rows = manifest(root)
    lock = verify_lock(root)
    thresholds = np.asarray(lock["thresholds"], dtype=np.float64)
    nr = [r for r in rows if r["role"] == "normal_evaluation"]
    fr = [r for r in rows if r["role"] == "fault_evaluation"]
    normal_scores, normal_files = load_scores(root, nr)
    fault_scores, fault_files = load_scores(root, fr)
    normal, fault = events(charts(normal_scores), thresholds), events(charts(fault_scores), thresholds)
    boot = bootstrap(normal, fault)
    main = main_rows(normal, fault, normal_scores, thresholds)
    paired = paired_rows(boot)
    destination = root / "results"
    write_csv(destination / "Main_results.csv", main)
    write_csv(destination / "Paired_contrasts.csv", paired)
    write_csv(destination / "Per_run_events.csv", per_run_rows(normal, fault, nr, fr))
    saved = {"normal_" + k: v for k, v in normal.items()}
    saved.update({"fault_" + k: v for k, v in fault.items()})
    saved.update(arm_names=np.asarray(ARMS), thresholds=thresholds,
                 normal_run_ids=np.asarray([r["run_id"] for r in nr]),
                 fault_run_ids=np.asarray([r["run_id"] for r in fr]),
                 normal_score_mean=normal_scores.mean(axis=(0, 1)),
                 normal_score_variance_ddof1=normal_scores.var(axis=(0, 1), ddof=1))
    write_npz(destination / "EVALUATION.npz", **saved)
    write_npz(destination / "BOOTSTRAP.npz", normal_indices=boot["normal_indices"], fault_indices=boot["fault_indices"],
              contrast_indices=np.asarray(CONTRASTS), endpoints=np.asarray(ENDPOINTS),
              contrast_estimates=boot["estimates"], contrast_bounds95=boot["bounds"],
              contrast_distributions=boot["distributions"], seed=np.array(BOOTSTRAP_SEED))
    if make_figure:
        draw_figure(main, destination / "figures")
    summary = {"status": "COMPLETE", "completed_utc": utc(), "binding": identities(root),
               "threshold_lock_sha256": sha(root / "results" / "THRESHOLDS_LOCK.json"),
               "panel_counts": COUNTS, "main_results": main, "paired_contrasts": paired,
               "evaluation_score_files": normal_files + fault_files,
               "inference": {"status": "descriptive_post_hoc_sensitivity",
                             "ci": "marginal_95_percent_paired_percentile_bootstrap",
                             "unit": "whole_physical_trajectory_preserving_all_arms",
                             "conditional_on": ["fixed_historical_representation", "realized_199_run_calibration"],
                             "not_covered": ["training_uncertainty", "calibration_population_uncertainty",
                                             "simultaneous_multiple_comparison_coverage", "industrial_generalization"]}}
    write_json(destination / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate", "all", "figure"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.stage in ("calibrate", "all"):
        lock = calibrate(args.root)
        print("Calibration locked: " + str(args.root / "results" / "THRESHOLDS_LOCK.json"), flush=True)
    if args.stage in ("evaluate", "all"):
        result = evaluate(args.root)
        print("Completed six-arm evaluation: " + str(args.root / "results" / "SUMMARY.json"), flush=True)
    if args.stage == "figure":
        with (args.root / "results" / "Main_results.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in row.keys() - {"arm", "rho"}:
                row[key] = float(row[key])
        draw_figure(rows, args.root / "results" / "figures")


if __name__ == "__main__":
    main()
