"""Paired analysis of the Fault-9 recording x B-estimator cross.

Usage: python analyze_cross.py PER_RUN_EVENTS.csv --output OUT_DIRECTORY

Reads existing event records only. Does not select candidates, alter thresholds,
or simulate trajectories. Labels and case identities, never CSV row order, define
the pairing. All eight arms share the fault resampling indices; normal indices
are drawn independently. Intervals condition on the fitted monitors and realized
calibration and are marginal descriptive intervals on an already analysed panel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARM_ORDER = tuple((e, v, p) for e in ("original", "batch")
                  for v in ("native", "snapshot") for p in ("mean", "moment"))
ROLES = ("normal_evaluation", "fault9_evaluation")
ENDPOINTS = ("clean_120", "clean_480", "clean_2400", "loss_minutes",
             "normal_alarm", "pre_alarm")
ROLE_FOR = {k: (ROLES[0] if k == "normal_alarm" else ROLES[1]) for k in ENDPOINTS}
REQUIRED = {"case_id", "role", "estimator", "recording", "pool", *ENDPOINTS}
SCOPE = (
    "Post-hoc comparison on the previously analysed recording panel 2. "
    "Marginal 95% percentile intervals, conditional on this learned state and "
    "realized calibration; they do not include learning or calibration variability. "
    "Whole-trajectory paired resampling preserves within-trajectory dependence "
    "and arm pairing but does not resolve dependence between trajectories caused "
    "by shared random-stream segments. No independent-confirmation claim."
)
OUTPUT_NAMES = ("CROSS_SUMMARY.csv", "CROSS_CONTRASTS.csv",
                "CROSS_CONTRAST_DEFINITIONS.json", "CROSS_ANALYSIS.json",
                "CROSS_BOOTSTRAP_INDICES.npz")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_label(arm):
    return "/".join(arm)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def numeric(value, field):
    v = str(value).strip()
    if v.lower() in ("true", "false"):
        x = float(v.lower() == "true")
    else:
        x = float(v)
    require(np.isfinite(x), f"Nonfinite {field}")
    if field != "loss_minutes":
        require(x in (0., 1.), f"Nonbinary {field}: {value}")
    else:
        require(0 <= x <= 2400 and x % 3 == 0,
                f"Invalid 3-minute-grid penalized loss: {value}")
    return x


def load_events(path, expected_n=120):
    """Build [case, arm] endpoint arrays with independently sorted role IDs."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and REQUIRED <= set(reader.fieldnames),
                f"Missing columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    require(rows, "Empty event table")
    groups = {(role, arm): {} for role in ROLES for arm in ARM_ORDER}
    seeds = {}
    for line, row in enumerate(rows, 2):
        role = row["role"]
        arm = tuple(row[k] for k in ("estimator", "recording", "pool"))
        require(role in ROLES and arm in ARM_ORDER, f"Unexpected label at line {line}")
        case = row["case_id"]
        require(bool(case.strip()), f"Empty case_id at line {line}")
        dest = groups[(role, arm)]
        require(case not in dest, f"Duplicate case/arm: {role}/{case}/{arm_label(arm)}")
        vals = {k: numeric(row[k], k) for k in ENDPOINTS}
        require(vals["clean_120"] <= vals["clean_480"] <= vals["clean_2400"],
                f"Non-nested detection horizons: {case}/{arm_label(arm)}")
        require(not (vals["pre_alarm"] and vals["clean_2400"]),
                f"Clean detection with prefault alarm: {case}/{arm_label(arm)}")
        require(vals["normal_alarm"] >= vals["clean_2400"] and
                vals["normal_alarm"] >= vals["pre_alarm"],
                f"Any-alarm flag contradicts detection: {case}/{arm_label(arm)}")
        if vals["clean_2400"] == 0:
            require(vals["loss_minutes"] == 2400,
                    f"Failure/prefault alarm must receive 2400 min: {case}/{arm_label(arm)}")
        else:
            require(vals["loss_minutes"] <= 2397, f"Clean detection outside window: {case}")
            for k, limit in (("clean_120", 120), ("clean_480", 480)):
                require(vals[k] == float(vals["loss_minutes"] <= limit),
                        f"Horizon flag and delay disagree: {case}/{arm_label(arm)}/{k}")
        if row.get("seed_u32", "") != "":
            seed = int(row["seed_u32"])
            require(0 <= seed < 2**32, f"Invalid uint32 seed: {case}")
            key = (role, case)
            require(key not in seeds or seeds[key] == seed, f"Seed differs between arms: {case}")
            seeds[key] = seed
        dest[case] = vals
    ids, arrays = {}, {}
    for role in ROLES:
        reference = set(groups[(role, ARM_ORDER[0])])
        require(bool(reference), f"Missing role: {role}")
        if expected_n is not None:
            require(len(reference) == expected_n, f"Expected {expected_n} {role}, got {len(reference)}")
        for arm in ARM_ORDER:
            require(set(groups[(role, arm)]) == reference,
                    f"Case identities differ across arms: {role}/{arm_label(arm)}")
        ids[role] = sorted(reference)
        arrays[role] = {k: np.array([[groups[(role, arm)][case][k] for arm in ARM_ORDER]
                                    for case in ids[role]], dtype=np.float64)
                        for k in ENDPOINTS}
    require(set(ids[ROLES[0]]).isdisjoint(ids[ROLES[1]]),
            "Normal/fault case IDs overlap; use globally distinct original IDs")
    return ids, arrays, len(rows)


def make_contrasts():
    defs = []
    def add(name, kind, terms, description):
        w = np.zeros(len(ARM_ORDER), dtype=np.float64)
        for arm, coef in terms:
            w[ARM_ORDER.index(arm)] += coef
        require(np.count_nonzero(w) in (2, 4) and w.sum() == 0, "Invalid contrast")
        defs.append({"name": name, "kind": kind, "description": description,
                     "terms": [{"arm": arm_label(ARM_ORDER[j]), "weight": int(x)}
                               for j, x in enumerate(w) if x], "weights": w})
    for estimator in ("original", "batch"):
        for pool in ("mean", "moment"):
            add(f"recording_{estimator}_{pool}", "recording",
                [((estimator, "snapshot", pool), 1), ((estimator, "native", pool), -1)],
                "controller-input minus post-update, at the stated estimator and pool")
    for view in ("native", "snapshot"):
        for pool in ("mean", "moment"):
            add(f"estimator_{view}_{pool}", "estimator",
                [(("batch", view, pool), 1), (("original", view, pool), -1)],
                "batch means minus original B, at the stated recording and pool")
    for estimator in ("original", "batch"):
        for view in ("native", "snapshot"):
            add(f"C2_{estimator}_{view}", "C2",
                [((estimator, view, "moment"), 1), ((estimator, view, "mean"), -1)],
                "second-moment pool minus mean-shift pool, both signed")
    for pool in ("mean", "moment"):
        add(f"interaction_recording_by_estimator_{pool}", "interaction",
            [(("batch", "snapshot", pool), 1), (("batch", "native", pool), -1),
             (("original", "snapshot", pool), -1), (("original", "native", pool), 1)],
            "(controller-input minus post-update) under batch means minus the same difference under original B")
    return defs


def paired_bootstrap_means(values, indices, chunk=1000):
    require(values.ndim == 2 and indices.ndim == 2 and indices.shape[1] == len(values),
            "Bootstrap dimensions disagree")
    means = np.empty((len(indices), values.shape[1]), dtype=np.float64)
    for lo in range(0, len(indices), chunk):
        hi = min(lo + chunk, len(indices))
        means[lo:hi] = values[indices[lo:hi]].mean(axis=1)
    return means


def contrast_rows(arrays, indices, contrasts=None):
    defs = make_contrasts() if contrasts is None else contrasts
    rows = []
    for endpoint in ENDPOINTS:
        role = ROLE_FOR[endpoint]
        raw = arrays[role][endpoint]
        scale = 1. if endpoint == "loss_minutes" else 100.
        boot = paired_bootstrap_means(raw, indices[role]) * scale
        for d in defs:
            w = d["weights"]
            per_case = raw @ w * scale
            sampled = boot @ w
            low, high = np.quantile(sampled, [.025, .975], method="linear")
            positive = np.flatnonzero(w > 0); negative = np.flatnonzero(w < 0)
            paired_binary = endpoint != "loss_minutes" and len(positive) == len(negative) == 1
            a = raw[:, positive[0]] if paired_binary else None
            b = raw[:, negative[0]] if paired_binary else None
            rows.append({
                "contrast": d["name"], "contrast_type": d["kind"],
                "definition": d["description"], "endpoint": endpoint, "role": role,
                "n": len(raw), "estimate": float(per_case.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "unit": "minutes" if endpoint == "loss_minutes" else "percentage_points",
                "positive_per_run_n": int((per_case > 0).sum()),
                "negative_per_run_n": int((per_case < 0).sum()),
                "zero_per_run_n": int((per_case == 0).sum()),
                "both_yes_n": int(((a == 1) & (b == 1)).sum()) if paired_binary else "",
                "positive_arm_only_n": int(((a == 1) & (b == 0)).sum()) if paired_binary else "",
                "negative_arm_only_n": int(((a == 0) & (b == 1)).sum()) if paired_binary else "",
                "both_no_n": int(((a == 0) & (b == 0)).sum()) if paired_binary else "",
                "interval": "marginal paired percentile 95%",
            })
    return rows


def summary_rows(ids, arrays):
    normal = arrays[ROLES[0]]; fault = arrays[ROLES[1]]
    nn = len(ids[ROLES[0]]); nf = len(ids[ROLES[1]])
    rows = []
    for j, (estimator, view, pool) in enumerate(ARM_ORDER):
        r = {"estimator": estimator, "recording": view, "pool": pool,
             "normal_n": nn, "fault_n": nf}
        for endpoint in ENDPOINTS:
            if endpoint == "loss_minutes":
                r["mean_penalized_loss_minutes"] = float(fault[endpoint][:, j].mean())
            else:
                values = normal[endpoint][:, j] if endpoint == "normal_alarm" else fault[endpoint][:, j]
                r[endpoint + "_n"] = int(values.sum())
                r[endpoint + "_percent"] = float(values.mean() * 100.)
        rows.append(r)
    return rows


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(input_csv, output, draws=20000, seed=20261030, expected_n=120):
    require(draws >= 1000, "Use at least 1000 bootstrap draws")
    source = Path(input_csv).resolve(); output = Path(output).resolve()
    original_hash = sha256(source)
    ids, arrays, nrows = load_events(source, expected_n=expected_n)
    output.mkdir(parents=True, exist_ok=True)
    require(not any((output / n).exists() for n in OUTPUT_NAMES),
            "Analysis output already exists; choose a new directory")
    # Explicitly independent normal/fault draws; eight arms share each role's draws.
    rng = np.random.default_rng(seed)
    indices = {role: rng.integers(0, len(ids[role]), size=(draws, len(ids[role])), dtype=np.int32)
               for role in ROLES}
    definitions = make_contrasts()
    summary = summary_rows(ids, arrays)
    comparisons = contrast_rows(arrays, indices, definitions)
    write_csv(output / "CROSS_SUMMARY.csv", summary)
    write_csv(output / "CROSS_CONTRASTS.csv", comparisons)
    public_defs = [{k: v for k, v in d.items() if k != "weights"} for d in definitions]
    (output / "CROSS_CONTRAST_DEFINITIONS.json").write_text(
        json.dumps({"recording_labels": {"native": "post-update", "snapshot": "controller-input"},
                    "arm_order": [arm_label(a) for a in ARM_ORDER], "contrasts": public_defs,
                    "interpretation": {
                        "loss_minutes": "Negative two-arm contrasts favor the positive-weight arm.",
                        "clean_detection": "Positive two-arm contrasts favor the positive-weight arm.",
                        "normal_alarm_and_pre_alarm": "Negative two-arm contrasts mean fewer alarms in the positive-weight arm.",
                        "interaction": "A difference of recording contrasts; its sign is endpoint-dependent. It is not a causal fraction.",
                        "binary_pair_counts": "Positive arm is the +1 arm; negative arm is the -1 arm. Empty for four-arm interactions.",
                    }}, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(output / "CROSS_BOOTSTRAP_INDICES.npz",
                        normal_indices=indices[ROLES[0]], fault_indices=indices[ROLES[1]],
                        normal_case_ids=np.array(ids[ROLES[0]]), fault_case_ids=np.array(ids[ROLES[1]]))
    require(sha256(source) == original_hash, "Input changed during analysis")
    report = {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
              "input_file": source.name, "input_sha256": original_hash, "input_rows": nrows,
              "arm_count": 8, "role_counts": {role: len(ids[role]) for role in ROLES},
              "bootstrap_draws": draws, "bootstrap_seed": seed,
              "resampling": "Normal and fault case indices drawn separately; same within-role indices for every arm and endpoint",
              "quantiles": "numpy.quantile([0.025,0.975], method='linear')",
              "contrasts": len(definitions), "contrast_endpoint_rows": len(comparisons),
              "multiplicity": "Marginal intervals; no familywise adjustment; no hypothesis acceptance rule",
              "scope": SCOPE, "input_unchanged": True,
              "versions": {"python": platform.python_version(), "numpy": np.__version__},
              "output_sha256": {name: sha256(output / name) for name in OUTPUT_NAMES if name != "CROSS_ANALYSIS.json"}}
    (output / "CROSS_ANALYSIS.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261030)
    parser.add_argument("--expected-n", type=int, default=120,
                        help="Expected number of normal and fault cases (120 in recording panel 2)")
    args = parser.parse_args()
    report = run(args.input_csv, args.output, args.bootstrap_draws, args.bootstrap_seed, args.expected_n)
    print(json.dumps(report, indent=2, ensure_ascii=False))
