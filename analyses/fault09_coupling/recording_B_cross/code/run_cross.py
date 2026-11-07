#!/usr/bin/env python3
"""Recompute the fixed Fault-9 recording x temporal-covariance experiment.

Reads the 306 existing development and 439 panel-2 trajectories. No simulator,
network, sampling of process trajectories, or changes to source data.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
import uuid
import zipfile

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import inherited_dim as old
import relearning_core as core
from normal_support import apply_support, validate_support
from existing_data import check_views
from common import sha, read_json, write_json, write_csv, load_npz, npz, seal, verify_files, environment

PANEL = "D439_supplement"
VIEWS = ("native", "snapshot")
ROLES = ("normal_design", "fault9_fit", "normal_anchor", "normal_inner", "fault9_selection")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(out, name, **details):
    with (out / "EVENTS.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time_utc=utc(), event=name, **details), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def local_path(value):
    s = str(value).strip().strip('"')
    if os.name != "nt" and len(s) > 2 and s[1] == ":":
        s = "/mnt/" + s[0].lower() + "/" + s[2:].lstrip("\\/").replace("\\", "/")
    if os.name == "nt" and s.startswith("/mnt/") and len(s) > 7 and s[6] == "/":
        s = s[5].upper() + ":/" + s[7:]
    return Path(s).expanduser()


def record_discovery_error(inspection_errors, path, operation, exc):
    if inspection_errors is not None:
        item = dict(path=str(path), operation=operation,
                    error_type=type(exc).__name__, error=str(exc))
        if item not in inspection_errors:
            inspection_errors.append(item)


def discovery_children(path, inspection_errors=None):
    """Optional search locations may be absent or inaccessible."""
    try:
        if not path.is_dir():
            return []
        return list(path.iterdir())
    except OSError as exc:
        record_discovery_error(inspection_errors, path, "list_directory", exc)
        return []


def workspace_candidates(bases, inspection_errors=None):
    found, seen = [], set()
    queue = [(local_path(p), 0) for p in bases if str(p).strip()]
    while queue:
        p, depth = queue.pop(0)
        try:
            if p.is_symlink() or not p.is_dir():
                continue
            resolved = p.resolve()
        except OSError as exc:
            record_discovery_error(inspection_errors, p, "inspect_directory", exc)
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            if (p / "cases").is_dir():
                found.append(resolved)
        except OSError as exc:
            record_discovery_error(inspection_errors, p / "cases", "inspect_cases", exc)
        if depth >= 5:
            continue
        for q in discovery_children(p, inspection_errors):
            # Inspect eligible children when dequeued, inside the guarded block.
            if q.name.startswith("TEP") or q.name in ("workspace", "Desktop", "Downloads", "桌面", "下载"):
                queue.append((q, depth + 1))
    return found


def rows_for(expected, panel):
    dev = [dict(r["case"], raw_file="RECORDING.npz", raw_sha256=r["raw_sha256"],
                complete_sha256=r["complete_sha256"], metadata_sha256=r["metadata_sha256"]) for r in expected["cases"]]
    return dev, panel["cases"]


def missing_rows(ws, rows, inspection_errors=None):
    if ws is None:
        return [dict(case_id=r["case_id"], raw_file=r["raw_file"]) for r in rows]
    missing = []
    for r in rows:
        folder = ws / "cases" / r["case_id"]
        available = True
        for path in (folder / r["raw_file"], folder / "COMPLETE.json"):
            try:
                if not path.is_file():
                    available = False
            except OSError as exc:
                record_discovery_error(inspection_errors, path, "inspect_input_file", exc)
                available = False
        if not available:
            missing.append(dict(case_id=r["case_id"], raw_file=r["raw_file"]))
    return missing


def default_source_bases(inspection_errors=None):
    home = Path.home()
    defaults = [ROOT.parent, home / "Desktop", home / "Downloads", home / "OneDrive" / "Desktop"]
    for drive in discovery_children(Path("/mnt"), inspection_errors):
        if len(drive.name) == 1 and "a" <= drive.name <= "z":
            for profile in discovery_children(drive / "Users", inspection_errors):
                defaults += [profile / "Desktop", profile / "Downloads", profile / "OneDrive" / "Desktop"]
    return defaults


def locate_sources(config, expected, panel, out):
    dev, pan = rows_for(expected, panel)
    inspection_errors = []
    # Explicit paths do not require enumerating unrelated profiles or mounts.
    defaults = default_source_bases(inspection_errors) if any(
        not config.get(name, "") for name in ("relearning_workspace", PANEL)) else []
    locations, reports = {}, {}
    for name, rows in (("relearning_workspace", dev), (PANEL, pan)):
        explicit = config.get(name, "")
        bases = [explicit] if explicit else defaults
        candidates = workspace_candidates(bases, inspection_errors)
        assessed = [(p, missing_rows(p, rows, inspection_errors)) for p in candidates]
        assessed.sort(key=lambda pair: (len(pair[1]), len(str(pair[0])), str(pair[0])))
        selected, missing = assessed[0] if assessed else (None, missing_rows(None, rows))
        locations[name] = selected if not missing else None
        reports[name] = dict(required=len(rows), found=len(rows) - len(missing),
                             candidate=str(selected) if selected else None, missing=missing)
    report = dict(status="READY" if all(locations.values()) else "MISSING_INPUTS", time_utc=utc(),
                  required_development=306, required_panel2=439, total_required=745,
                  new_simulations=0, sources=reports, discovery_version="1.3",
                  inspection_errors=inspection_errors)
    write_json(out / "INPUT_PREFLIGHT.json", report)
    if inspection_errors:
        print(f"SOURCE_SCAN: skipped {len(inspection_errors)} inaccessible path checks; see INPUT_PREFLIGHT.json.", flush=True)
    if not all(locations.values()):
        write_json(out / "missing_input_report.json", report)
        print("MISSING_INPUTS: need the original 306-development and 439-panel2 workspaces.", flush=True)
        for name, d in reports.items():
            print(f"  {name}: {d['found']}/{d['required']} raw files found; candidate={d['candidate']}", flush=True)
    return locations, report


def checked_raw(ws, r):
    folder = ws / "cases" / r["case_id"]
    path = folder / r["raw_file"]
    if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Unsafe raw source path: " + str(path))
    checks = {"COMPLETE.json": r["complete_sha256"], r["raw_file"]: r["raw_sha256"]}
    verify_files(folder, checks)
    if "metadata_sha256" in r:
        # The original panel stores META.json, while its RETURN archive renames
        # it METADATA.json. Resolve the live filename from the hash-verified
        # completion record instead of inferring it from the archive layout.
        complete = read_json(folder / "COMPLETE.json")
        if "metadata_sha256" in complete and complete["metadata_sha256"] != r["metadata_sha256"]:
            raise ValueError("Completion metadata hash differs from frozen manifest: " + r["case_id"])
        if "metadata_file" in complete:
            metadata_file = complete["metadata_file"]
        elif r["raw_file"] == "RECORDING.npz":
            # Original supervised-development cases use this flat layout.
            metadata_file = "METADATA.json"
        else:
            raise ValueError("Verified completion record lacks metadata_file: " + r["case_id"])
        if not isinstance(metadata_file, str) or not metadata_file or "\\" in metadata_file:
            raise ValueError("Invalid metadata path in completion record: " + r["case_id"])
        relative = PurePosixPath(metadata_file)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
            raise ValueError("Unsafe metadata path in completion record: " + r["case_id"])
        verify_files(folder, {metadata_file: r["metadata_sha256"]})
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in ("native_values", "fast_control_input_measurements", "phase", "native_time_seconds", "diagnostic_IDV")}
    return check_views(raw, int(r["fault_id"]))


def read_role(ws, rows, role, pca, out):
    wanted = [r for r in rows if r["role"] == role]
    vals = {v: [] for v in VIEWS}
    for i, row in enumerate(wanted, 1):
        raw = checked_raw(ws, row)
        for view in VIEWS:
            vals[view].append(raw[view])
        if i % 25 == 0 or i == len(wanted):
            print(f"Read existing {role}: {i}/{len(wanted)}", flush=True)
    if not wanted:
        raise ValueError("Empty input role: " + role)
    projected = {v: core.project_raw(np.stack(vals[v]), core.pca_view(pca, v)) for v in VIEWS}
    event(out, "ROLE_READ", role=role, count=len(wanted), raw_hashes_verified=True)
    return projected, wanted


def reference_paths():
    ref = ROOT / "reference"
    if (ref / "workspace").is_dir():
        ref = ref / "workspace"
    locks = {n: read_json(ref / n) for n in ("NORMAL_SUPPORT_LOCK.json", "MODEL_LOCK.json", "THRESHOLDS_LOCK.json", "FINAL_LOCK.json")}
    folders = {}
    for n, lock in locks.items():
        folder = ref / lock["relative_path"]
        verify_files(folder, lock["files"])
        folders[n] = folder
    if sha(ROOT / "assets" / "PCA_BY_VIEW.npz") != locks["MODEL_LOCK.json"]["PCA_sha256"]:
        raise ValueError("PCA differs from frozen original baseline")
    return folders, locks


def batch_covariance(y):
    n, length, dim = y.shape
    if (n, length) != (40, 960):
        raise ValueError("Batch covariance requires exactly 40 x 960 design observations")
    sums = y.reshape(40, 8, 120, dim).sum(axis=2).reshape(320, dim) / np.sqrt(120.)
    B = np.cov(sums, rowvar=False, ddof=1)
    eig = np.linalg.eigvalsh((B + B.T) / 2)
    if not np.isfinite(B).all() or eig[0] <= np.trace(B) / dim * 1e-12:
        raise ValueError("Batch covariance violates inherited positive-definite check; no added regularization")
    return B


def fit_changed(nd, ff, folder, support):
    # Freeze all geometry and fitting statistics through the exact inherited fit.
    des, _, _ = core.fit_state(nd, ff, folder, support)
    effective = apply_support(nd, support)
    original_B = des["B"].copy()
    des["B"] = batch_covariance(old.features(effective, des))
    mat = load_npz(folder / "FIT_MOMENTS.npz")
    axes, labels, small, _, _, diag = old.build_axes_regularized(mat["mean_early"], mat["mean_late"], des["B"], des, core.VARIANT)
    mean = axes[small].copy()
    variances = np.einsum("ad,de,ae->a", mean, des["G"], mean)
    if not np.all(variances > 1e-24):
        raise ValueError("Degenerate mean candidate")
    mean /= np.sqrt(variances)[:, None]
    moment, mlabels, mdiag = old.learned_axes(mat["contrast"], des, "ridge_1e4")
    if not 1 <= len(mean) <= 8 or not 1 <= len(moment) <= 8:
        raise ValueError("Candidate budget changed")
    npz(folder / "DESIGN.npz", **des)
    npz(folder / "MODEL.npz", mean=mean, moment=moment)
    info = read_json(folder / "FIT_INFO.json")
    info.update(estimator="batch", mean_count=len(mean), moment_count=len(moment), mean_labels=[labels[j] for j in small],
                moment_labels=mlabels, mean_diagnostics=diag, moment_diagnostics=mdiag,
                block_length=120, blocks_per_run=8, independent_run_count=40, block_count=320,
                covariance_ddof=1, geometry_and_fit_moments_unchanged=True,
                raw_D_retained_for_original_geometry_only=True,
                effective_temporal_metric="B; raw_D is retained as original geometry metadata and is not used to construct the changed candidates")
    write_json(folder / "FIT_INFO.json", info)
    npz(folder / "B_ESTIMATOR_COMPARISON.npz", original_B=original_B, batch_B=des["B"])


def require_close(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.allclose(a, b, atol=1e-8, rtol=1e-9):
        raise ValueError("Baseline reproduction failed: " + label)
    return float(np.max(np.abs(a - b))) if a.size else 0.


def compare_original_models(models, reference):
    differences = {}
    for view in VIEWS:
        here, there = models / view, reference / view
        for name in ("DESIGN.npz", "FIT_MOMENTS.npz", "MODEL.npz"):
            a, b = load_npz(here / name), load_npz(there / name)
            if a.keys() != b.keys():
                raise ValueError("Baseline model keys changed: " + name)
            for key in a:
                if np.asarray(a[key]).dtype.kind == "b":
                    if not np.array_equal(a[key], b[key]):
                        raise ValueError("Baseline boolean geometry changed")
                else:
                    differences[f"{view}/{name}/{key}"] = require_close(a[key], b[key], f"{view}/{name}/{key}")
        for key in ("choices", "mean_count", "moment_count", "candidate_labels"):
            if read_json(here / "SELECTION.json")[key] != read_json(there / "SELECTION.json")[key]:
                raise ValueError("Baseline candidate selection changed: " + view + "/" + key)
        a, b = load_npz(here / "INNER.npz"), load_npz(there / "INNER.npz")
        for key in ("selection_alarm_packed", "loss_samples_total"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("Baseline selection alarm/loss changed: " + view)
    return differences


def evaluate(estimator, objs, cases, models, h, out):
    u = core.four_inputs(objs, models)
    stats = core.chart(u)
    mask = stats > h
    ev = core.events(mask)
    npz(out / f"{estimator}_EVALUATION.npz", scores=u, thresholds=h, alarm_packed=np.packbits(mask, axis=1, bitorder="little"), alarm_shape=np.array(mask.shape))
    rows, summaries = [], []
    normal = np.array([c["role"] == "normal_evaluation" for c in cases])
    fault = ~normal
    if (int(normal.sum()), int(fault.sum())) != (120, 120):
        raise ValueError("Panel2 evaluation counts changed")
    for j, (mid, view, pool) in enumerate(core.CONDITIONS):
        monitor = estimator + "_" + mid
        for i, c in enumerate(cases):
            rows.append(dict(case_id=c["case_id"], role=c["role"], estimator=estimator, recording=view, pool=pool,
                             monitor_id=monitor, seed_u32=int(c["seed_u32"]), normal_alarm=int(ev["normal_alarm"][i,j]),
                             pre_alarm=int(ev["pre_alarm"][i,j]), clean_120=int(ev["clean_120"][i,j]),
                             clean_480=int(ev["clean_480"][i,j]), clean_2400=int(ev["clean_2400"][i,j]),
                             loss_minutes=int(ev["loss_minutes"][i,j]), delay_samples=int(ev["delay_samples"][i,j])))
        summaries.append(dict(estimator=estimator, recording=view, pool=pool, monitor_id=monitor, threshold=float(h[j]),
                              normal_n=120, fault_n=120, normal_alarm_n=int(ev["normal_alarm"][normal,j].sum()),
                              prefault_alarm_n=int(ev["pre_alarm"][fault,j].sum()), clean_2h_n=int(ev["clean_120"][fault,j].sum()),
                              clean_8h_n=int(ev["clean_480"][fault,j].sum()), clean_40h_n=int(ev["clean_2400"][fault,j].sum()),
                              mean_loss_minutes=float(ev["loss_minutes"][fault,j].mean())))
    return u, mask, rows, summaries


def combine_roles(a, b):
    return {v: {k: np.concatenate((a[v][k], b[v][k]), axis=0) for k in a[v]} for v in VIEWS}


def lock_models(out, estimator, model_root):
    path = out / f"{estimator}_MODEL_LOCK.json"
    protocol_hash=sha(ROOT / "PROTOCOL.json")
    if protocol_hash != read_json(out / "EXECUTION_BINDING.json")["protocol_sha256"]:
        raise ValueError("Protocol changed during execution")
    write_json(path, dict(estimator=estimator, time_utc=utc(), files=seal(model_root),
                         pca_sha256=sha(ROOT / "assets" / "PCA_BY_VIEW.npz"),
                         protocol_sha256=protocol_hash,
                         already_analyzed_panel=True, evaluation_used_for_fitting=False))
    event(out, "MODELS_LOCKED", estimator=estimator, lock_sha256=sha(path))


def run(args, out):
    if not (ROOT / "PROTOCOL.json").is_file():
        raise FileNotFoundError("Required frozen specification is missing: PROTOCOL.json")
    write_json(out / "EXECUTION_BINDING.json",dict(time_utc=utc(),protocol_sha256=sha(ROOT/"PROTOCOL.json"),
               runner_sha256=sha(Path(__file__)),vendor_files=seal(ROOT/"vendor"),environment=environment()))
    config = read_json(args.config) if Path(args.config).is_file() else {}
    expected = read_json(ROOT / "assets" / "EXPECTED_PARENT.json")
    manifest = read_json(ROOT / "assets" / "INPUT_MANIFEST.json")
    panel = manifest["panels"][PANEL]
    if panel["calibration_rank"] != 190 or panel["case_n"] != 439:
        raise ValueError("Panel2 fixed manifest changed")
    dev_rows, panel_rows = rows_for(expected, panel)
    if dict(Counter(r["role"] for r in dev_rows)) != {"normal_design":40,"normal_anchor":32,"normal_inner":59,"fault9_fit":100,"fault9_selection":75}:
        raise ValueError("Development role counts changed")
    locations, report = locate_sources(config, expected, panel, out)
    if args.preflight or report["status"] != "READY":
        return 0 if report["status"] == "READY" else 2
    folders, locks = reference_paths()
    pca = load_npz(ROOT / "assets" / "PCA_BY_VIEW.npz")
    support = {v: load_npz(folders["NORMAL_SUPPORT_LOCK.json"] / (v + "_SUPPORT.npz")) for v in VIEWS}
    for v in VIEWS:
        validate_support(support[v])
    write_json(out / "NORMAL_SUPPORT_LOCK.json", dict(time_utc=utc(), reference_lock=locks["NORMAL_SUPPORT_LOCK.json"],
               reuse_exact_original_support=True, pca_sha256=sha(ROOT / "assets" / "PCA_BY_VIEW.npz")))
    dev = {}
    for role in ROLES:
        dev[role], _ = read_role(locations["relearning_workspace"], dev_rows, role, pca, out)
    original = out / "models_original"
    original.mkdir()
    for v in VIEWS:
        core.fit_state(dev["normal_design"][v], dev["fault9_fit"][v], original / v, support[v])
        core.select_state(dev["normal_anchor"][v], dev["normal_inner"][v], dev["fault9_selection"][v], original / v)
    geometry_errors = compare_original_models(original, folders["MODEL_LOCK.json"])
    lock_models(out, "original", original)
    calibration, cal_cases = read_role(locations[PANEL], panel_rows, "normal_calibration", pca, out)
    if len(cal_cases) != 199:
        raise ValueError("Expected 199 normal calibration runs")
    uc = core.four_inputs(calibration, original)
    anchors = core.four_anchors(original)
    peaks = core.chart(uc).max(axis=1)
    h = core.thresholds(peaks, anchors, 190)
    refcal = load_npz(folders["THRESHOLDS_LOCK.json"] / (PANEL + ".npz"))
    require_close(uc, refcal["scores"], "original calibration scores")
    require_close(h, refcal["thresholds"], "original thresholds")
    npz(out / "original_CALIBRATION.npz", scores=uc, peaks=peaks, anchors=anchors, thresholds=h)
    write_json(out / "original_THRESHOLDS_LOCK.json", dict(time_utc=utc(), thresholds=h.tolist(), rank=190,
               normal_n=199, model_lock_sha256=sha(out / "original_MODEL_LOCK.json")))
    normal, normal_cases = read_role(locations[PANEL], panel_rows, "normal_evaluation", pca, out)
    fault, fault_cases = read_role(locations[PANEL], panel_rows, "fault9_evaluation", pca, out)
    objs, cases = combine_roles(normal, fault), normal_cases + fault_cases
    del normal, fault
    u, mask, rows, summaries = evaluate("original", objs, cases, original, h, out)
    refev = load_npz(folders["FINAL_LOCK.json"] / (PANEL + "_EVALUATION.npz"))
    score_error = require_close(u, refev["scores"], "original evaluation scores")
    refmask = np.unpackbits(refev["alarm_packed"], axis=1, count=960, bitorder="little").astype(bool)
    if not np.array_equal(mask, refmask):
        raise ValueError("Baseline evaluation alarm bits differ; batch recalculation stopped")
    audit = dict(status="PASS", time_utc=utc(), evaluation_alarm_bits=int(mask.size),
                 all_evaluation_alarm_bits_equal=True, evaluation_score_max_abs_error=score_error,
                 calibration_thresholds_exactly_equal=bool(np.array_equal(h,refcal["thresholds"])),
                 calibration_scores_exactly_equal=bool(np.array_equal(uc,refcal["scores"])),
                 evaluation_scores_exactly_equal=bool(np.array_equal(u,refev["scores"])),
                 numeric_comparison_atol=1e-8,numeric_comparison_rtol=1e-9,
                 geometry_max_abs_errors=geometry_errors, original_results=summaries)
    write_json(out / "BASELINE_REPRODUCTION.json", audit)
    event(out, "BASELINE_REPRODUCTION_PASS", evaluation_alarm_bits=int(mask.size))
    print("Original four monitors reproduced. Starting fixed batch-means comparison.", flush=True)
    batch = out / "models_batch"
    batch.mkdir()
    for v in VIEWS:
        fit_changed(dev["normal_design"][v], dev["fault9_fit"][v], batch / v, support[v])
        core.select_state(dev["normal_anchor"][v], dev["normal_inner"][v], dev["fault9_selection"][v], batch / v)
    del dev
    lock_models(out, "batch", batch)
    buc = core.four_inputs(calibration, batch)
    ba = core.four_anchors(batch)
    bp = core.chart(buc).max(axis=1)
    bh = core.thresholds(bp, ba, 190)
    npz(out / "batch_CALIBRATION.npz", scores=buc, peaks=bp, anchors=ba, thresholds=bh)
    write_json(out / "batch_THRESHOLDS_LOCK.json", dict(time_utc=utc(), thresholds=bh.tolist(), rank=190,
               normal_n=199, model_lock_sha256=sha(out / "batch_MODEL_LOCK.json"),
               frozen_before_batch_evaluation=True, panel_previously_analyzed=True))
    event(out, "BATCH_THRESHOLDS_LOCKED", lock_sha256=sha(out / "batch_THRESHOLDS_LOCK.json"))
    _, _, br, bs = evaluate("batch", objs, cases, batch, bh, out)
    write_csv(out / "PER_RUN_EVENTS.csv", rows + br)
    write_csv(out / "MAIN_RESULTS.csv", summaries + bs)
    write_json(out / "EVALUATION_CASES.json", cases)
    analyzer = ROOT / "code" / "analyze_cross.py"
    if not analyzer.is_file():
        raise FileNotFoundError("Missing required paired analysis: " + str(analyzer))
    subprocess.run([sys.executable, str(analyzer), str(out / "PER_RUN_EVENTS.csv"), "--output", str(out)], check=True)
    event(out, "EXPERIMENT_COMPLETE", new_simulations=0, monitors=8)
    write_json(out / "FINAL_LOCK.json", dict(status="COMPLETE", time_utc=utc(), files=seal(out), environment=environment(),
               new_simulations=0, development_runs=306, calibration_runs=199, normal_evaluation_runs=120,
               fault_evaluation_runs=120, interpretation="Post-hoc sensitivity analysis on the previously analyzed panel; conditional on this one training and calibration realization."))
    return 0


def self_test(out):
    rng = np.random.default_rng(20261030)
    y = rng.normal(size=(40,960,3))
    B = batch_covariance(y)
    blocks = np.array([sum(run[start:start+120]) / np.sqrt(120.) for run in y for start in range(0,960,120)])
    centered = blocks - blocks.mean(axis=0)
    independent_B = centered.T @ centered / 319
    np.testing.assert_allclose(B, independent_B, rtol=1e-13, atol=1e-13)
    # Changed later blocks must affect batch B while first-block B stays fixed.
    changed = y.copy(); changed[:,120:,:] *= 2
    assert not np.allclose(B, batch_covariance(changed))
    assert np.array_equal(np.cov(y[:,:120].sum(1),rowvar=False), np.cov(changed[:,:120].sum(1),rowvar=False))
    def obj(n, shift=0.):
        z=rng.normal(size=(n,960,3)); z[:,160:,0] += shift
        return dict(z=z,Q=np.sum(z*z,axis=2),run_id=np.arange(1,n+1),record_suffix_start=np.full(n,961),record_suffix_samples=np.zeros(n,dtype=int))
    nd,ff=obj(40),obj(100,.15)
    from normal_support import build_support
    s=build_support(nd)
    before=out/"synthetic_original"; after=out/"synthetic_batch"
    core.fit_state(nd,ff,before,s);fit_changed(nd,ff,after,s)
    a,b=load_npz(before/"DESIGN.npz"),load_npz(after/"DESIGN.npz")
    assert not np.allclose(a["B"],b["B"])
    for key in a:
        if key != "B": np.testing.assert_array_equal(a[key],b[key])
    na,ni,fs=obj(32),obj(59),obj(75,.15)
    core.select_state(na,ni,fs,after)
    u=core.selected_inputs(obj(4,.15),after)
    np.testing.assert_allclose(core.chart(u),core.scalar_chart(u),rtol=0,atol=0)
    # Corrupt raw input is rejected before any numerical load.
    hash_fixture=out/"hash_guard_fixture"/"cases"/"test"
    hash_fixture.mkdir(parents=True)
    (hash_fixture/"COMPLETE.json").write_text("{}",encoding="utf-8")
    (hash_fixture/"RECORDING.npz").write_bytes(b"corrupt fixture, not scientific data")
    row=dict(case_id="test",raw_file="RECORDING.npz",raw_sha256="0"*64,
             complete_sha256=sha(hash_fixture/"COMPLETE.json"),fault_id=0)
    try:
        checked_raw(out/"hash_guard_fixture",row)
    except ValueError as exc:
        if "RECORDING.npz" not in str(exc):raise
    else:
        raise AssertionError("Raw hash mismatch was not rejected")
    # Exercise both actual storage layouts (development root and panel attempt).
    for name,relative in (("development","RECORDING.npz"),("panel","attempt_fixture/RECORDING.npz")):
        folder=out/"loader_fixture"/"cases"/name
        rawpath=folder/relative;rawpath.parent.mkdir(parents=True)
        phase=np.zeros((960,4),dtype=int);phase[:,0]=np.arange(1,961)*180;phase[:,3]=3
        native=np.zeros((960,52));snapshot=np.zeros((960,41))
        npz(rawpath,native_values=native,fast_control_input_measurements=snapshot,phase=phase,
            native_time_seconds=np.arange(961)*180.,diagnostic_IDV=np.zeros((960,20),dtype=int))
        metadata=rawpath.with_name("METADATA.json" if name == "development" else "META.json")
        metadata.write_text("{}",encoding="utf-8")
        complete=folder/"COMPLETE.json"
        completion={} if name == "development" else dict(
            metadata_file=metadata.relative_to(folder).as_posix(),metadata_sha256=sha(metadata))
        write_json(complete,completion)
        fixture=dict(case_id=name,raw_file=relative,raw_sha256=sha(rawpath),metadata_sha256=sha(metadata),
                     complete_sha256=sha(complete),fault_id=0)
        loaded=checked_raw(out/"loader_fixture",fixture)
        np.testing.assert_array_equal(loaded["native"],native)
        np.testing.assert_array_equal(loaded["snapshot"],native)
    # Genuine archived data checks are separately identified from fixtures.
    folders,_=reference_paths()
    archived_cal=load_npz(folders["THRESHOLDS_LOCK.json"]/(PANEL+".npz"))
    actual_h=core.thresholds(core.chart(archived_cal["scores"]).max(1),archived_cal["anchors"],190)
    require_close(actual_h,archived_cal["thresholds"],"archived calibration replay")
    archived_ev=load_npz(folders["FINAL_LOCK.json"]/(PANEL+"_EVALUATION.npz"))
    actual_alarm=core.chart(archived_ev["scores"])>actual_h
    expected_alarm=np.unpackbits(archived_ev["alarm_packed"],axis=1,count=960,bitorder="little").astype(bool)
    np.testing.assert_array_equal(actual_alarm,expected_alarm)
    outcomes=core.events(actual_alarm)
    np.testing.assert_array_equal(outcomes["normal_alarm"][:120].sum(0),[7,7,10,11])
    np.testing.assert_array_equal(outcomes["clean_480"][120:].sum(0),[85,4,44,0])
    np.testing.assert_array_equal(outcomes["clean_2400"][120:].sum(0),[119,38,112,16])
    genuine_geometry={}
    geometry_path=ROOT/"assets"/"NORMAL_DESIGN_GEOMETRY.npz"
    if geometry_path.is_file():
        geometry=load_npz(geometry_path)
        for view in VIEWS:
            support=load_npz(folders["NORMAL_SUPPORT_LOCK.json"]/(view+"_SUPPORT.npz"))
            normal=dict(z=geometry[view+"__residual_z"],Q=geometry[view+"__Q"])
            effective=apply_support(normal,support)
            design=old.normal_design(effective,list(range(1,41)),core.VARIANT)
            saved=load_npz(folders["MODEL_LOCK.json"]/view/"DESIGN.npz")
            errors={k:require_close(value,saved[k],"real40N/"+view+"/"+k)
                    for k,value in design.items() if np.asarray(value).dtype.kind!="b"}
            mat=load_npz(folders["MODEL_LOCK.json"]/view/"FIT_MOMENTS.npz")
            axes,_,small,_,_,_=old.build_axes_regularized(mat["mean_early"],mat["mean_late"],design["B"],design,core.VARIANT)
            mean=axes[small].copy()
            mean/=np.sqrt(np.einsum("ad,de,ae->a",mean,design["G"],mean))[:,None]
            moment,_,_=old.learned_axes(mat["contrast"],design,"ridge_1e4")
            saved_model=load_npz(folders["MODEL_LOCK.json"]/view/"MODEL.npz")
            errors["mean_candidates"]=require_close(mean,saved_model["mean"],"real40N/"+view+"/mean candidates")
            errors["moment_candidates"]=require_close(moment,saved_model["moment"],"real40N/"+view+"/moment candidates")
            genuine_geometry[view]=errors
    report=dict(status="PASS",scope="Software verification, with synthetic fixtures and genuine archived baseline checks identified separately; no new batch evaluation result", batch_formula_independently_verified=True,
                later_blocks_used=True, original_geometry_exactly_unchanged=True, fit_select_score_completed=True,
                scalar_vector_recurrences_equal=True,raw_hash_mismatch_rejected=True,
                development_and_panel_nested_metadata_layouts_verified=True,
                genuine_archived_panel2_alarm_bits_equal=int(actual_alarm.size),
                genuine_archived_panel2_counts_equal=True,genuine_40N_geometry_and_candidate_errors=genuine_geometry,
                environment=environment())
    write_json(out/"SELF_TEST.json",report)
    print(json.dumps(report,indent=2),flush=True)


def package_return(out):
    dest=ROOT/"returns";dest.mkdir(exist_ok=True)
    archive=dest/("TEP_P1_F9_Recording_B_Cross_RETURN_"+out.name+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():z.write(p,"results/"+p.relative_to(out).as_posix())
        for p in sorted((ROOT/"code").glob("*.py")):z.write(p,"code/"+p.name)
        if (ROOT/"PROTOCOL.json").is_file():z.write(ROOT/"PROTOCOL.json","PROTOCOL.json")
        if (ROOT/"EXPERIMENT_SPECIFICATION.txt").is_file():z.write(ROOT/"EXPERIMENT_SPECIFICATION.txt","EXPERIMENT_SPECIFICATION.txt")
    print("RETURN FILE:",archive,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"SOURCE_PATHS.json"))
    parser.add_argument("--output",default=None)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    out=Path(args.output).resolve() if args.output else ROOT/"outputs"/stamp
    out.mkdir(parents=True,exist_ok=False)
    result=1
    try:
        if args.self_test:
            self_test(out); result=0
        else:
            result=run(args,out)
    except Exception as exc:
        write_json(out/"ERROR.json",dict(status="STOPPED",time_utc=utc(),error=str(exc),traceback=traceback.format_exc(),new_simulations=0))
        print(traceback.format_exc(),file=sys.stderr,flush=True)
    finally:
        if not args.self_test:package_return(out)
    return result


if __name__=="__main__":
    raise SystemExit(main())
