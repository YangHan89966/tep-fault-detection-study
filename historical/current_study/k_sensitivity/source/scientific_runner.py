"""Exploratory 0.05 -> 0.07 sensitivity, conditional on frozen v1.7.1 states."""

from __future__ import annotations

import argparse

import csv

import hashlib

import io

import json

import os

import platform

import sys

import time

import traceback

import uuid

import zipfile

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import numpy as np

import reference_n as rn

import reference_dv as rd

FAULTS_N = (1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20)

METHODS_N = ("adaptive_5family_top5", "fixed_ad_top5", "independent_top1", "q_cusum")

KS = (.05, .07)

AUTO_GRID = (.01, .03, .05, .07, .10)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def write_json(path, obj):
    write_text_new(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def write_text_new(path, text):
    with path.open("x", encoding="utf-8") as out:
        out.write(text)

def write_bytes_new(path, data):
    with path.open("xb") as out:
        out.write(data)

def save_npz_new(path, **arrays):
    with path.open("xb") as out:
        np.savez_compressed(out, **arrays)

def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class Assets:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def npz(self, name):
        with np.load(io.BytesIO(self.read(name)), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def verify(self):
        for name, expected in self.json("BUNDLE_HASHES.json").items():
            if digest(self.read(name)) != expected:
                raise ValueError("补测包文件损坏: " + name)

class Run:
    def __init__(self, out):
        self.out = out
        self.started = time.perf_counter()
        self.checks = []
        self.rows = []
        self.deltas = []
        self.contrasts = []
        self.status = {}
        self.read_counts = {"N": 0, "DV": 0}
        self.log_file = (out / "RUN_LOG.txt").open("x", encoding="utf-8")

    def log(self, message):
        line = f"[{time.perf_counter() - self.started:8.1f}s] {message}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def close_array(self, name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        ok = a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-8)
        error = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None
        self.checks.append({"check": name, "passed": bool(ok), "max_abs_error": error,
                            "type": "numeric_replay", "rtol": 1e-9, "atol": 1e-8})
        if not ok:
            raise ValueError("0.05 数值复现不一致: " + name)

    def exact(self, name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        ok = a.shape == b.shape and np.array_equal(a, b)
        self.checks.append({"check": name, "passed": bool(ok), "type": "exact",
                            "different_entries": int(np.count_nonzero(a != b)) if a.shape == b.shape else None})
        if not ok:
            raise ValueError("0.05 报警/端点复现不一致: " + name)

def subset(index, role, fault, count):
    rows = sorted((r for r in index if r["role"] == role and r["fault"] == fault), key=lambda r: r["run"])
    if [r["run"] for r in rows] != list(range(1, count + 1)):
        raise ValueError("输入清单编号/数量不符: " + role)
    return rows

def check_values(a, label):
    if a.dtype != np.dtype("float64") or a.shape != (960, 52) or not np.isfinite(a).all():
        raise ValueError("轨迹格式不一致: " + label)
    return a

def read_group(rows, reader, study, run):
    start = time.perf_counter()
    label = f"{study} {rows[0]['role']} F{rows[0]['fault']:02d}"
    run.log(f"读取 {label}: {len(rows)} 条已有轨迹")
    values = np.empty((len(rows), 960, 52), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, a in enumerate(pool.map(reader, rows)):
            values[i] = a
    run.read_counts[study] += len(rows)
    run.log(f"{label} 读取并校验完成，{time.perf_counter() - start:.1f} 秒")
    return values

def load_n(rows, roots, run):
    def reader(row):
        raw = (roots[row["stage"]] / row["relative"]).read_bytes()
        if digest(raw) != row["sha256"]:
            raise ValueError("N 输入内容变化: " + row["relative"])
        return check_values(np.load(io.BytesIO(raw), allow_pickle=False), row["relative"])
    return read_group(rows, reader, "N", run)

def load_dv(rows, root, run):
    def reader(row):
        folder = root / row["relative"]
        manifest = (folder / "FILES.json").read_bytes()
        if digest(manifest) != row["manifest_sha256"]:
            raise ValueError("DV 输入清单变化: " + row["relative"])
        entries = [e for e in json.loads(manifest)["entries"] if e["path"] == "OUTPUT.npz"]
        if len(entries) != 1:
            raise ValueError("DV OUTPUT 清单不唯一")
        raw = (folder / "OUTPUT.npz").read_bytes()
        if len(raw) != entries[0]["bytes"] or digest(raw) != entries[0]["sha256"]:
            raise ValueError("DV 轨迹内容变化: " + row["relative"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            return check_values(data["values"], row["relative"])
    return read_group(rows, reader, "DV", run)

def unpack(a, order):
    return np.unpackbits(a, count=960, axis=-1, bitorder=order).astype(bool)

def event_arrays(mask):
    """Input (run,time); first affected released sample is 161, delay zero."""
    pre = mask[:, :160].any(1)
    post = mask[:, 160:]
    detected = post.any(1)
    first = np.where(detected, post.argmax(1), 800)
    clean = detected & ~pre
    return {"prefault": pre, "detected": detected, "clean": clean,
            "delay_min": first * 3, "event_loss_min": np.where(clean, first, 800) * 3,
            "occupancy_points": post.sum(1)}

def metrics(study, fault, method, k, h, fault_mask, normal_mask, shutdown, policy=None):
    ev = event_arrays(fault_mask)
    early = ev["detected"] if study == "N" else ev["clean"]
    primary = ev["delay_min"] if study == "N" else ev["event_loss_min"]
    cut = np.where(shutdown > 0, np.clip(shutdown - 1, 0, 960), 960)
    first_full = 160 + ev["delay_min"] // 3
    before = ev["detected"] & (first_full < cut)
    observed = (np.arange(960)[None, :] >= 160) & (np.arange(960)[None, :] < cut[:, None])
    row = {"study": study, "fault": fault, "method": method,
           "policy": policy or ("k005" if k == .05 else "k007"), "k": float(k), "h": float(h),
           "normal_far_pct": float(normal_mask.any(1).mean() * 100),
           "prefault_alarm_pct": float(ev["prefault"].mean() * 100),
           "detection_pct": float(ev["detected"].mean() * 100),
           "clean_detection_pct": float(ev["clean"].mean() * 100),
           "early_60_pct": float((early & (ev["delay_min"] <= 60)).mean() * 100),
           "early_120_pct": float((early & (ev["delay_min"] <= 120)).mean() * 100),
           "early_480_pct": float((early & (ev["delay_min"] <= 480)).mean() * 100),
           "primary_loss_min": float(primary.mean()),
           "miss_penalized_mean_delay_min": float(ev["delay_min"].mean()),
           "complete_episode_mean_loss_min": float(ev["event_loss_min"].mean()),
           "occupancy_pct": float(ev["occupancy_points"].mean() / 8),
           "before_shutdown_detection_pct": float(before.mean() * 100),
           "mean_above_before_shutdown_points": float((fault_mask & observed).sum(1).mean())}
    return row

def delta_rows(rows, new_policy="k007"):
    by = {}
    for r in rows:
        by.setdefault((r["study"], r["fault"], r["method"]), {})[r["policy"]] = r
    result = []
    for (study, fault, method), versions in by.items():
        if "k005" not in versions or new_policy not in versions:
            continue
        old, new = versions["k005"], versions[new_policy]
        d = {"study": study, "fault": fault, "method": method, "comparison": new_policy + "_minus_k005"}
        for key in old:
            if key not in {"study", "fault", "method", "policy", "k", "h"}:
                d["delta_" + key] = new[key] - old[key]
        result.append(d)
    return result

def choose_k(candidates):
    """Only development/selection outcomes may enter this function."""
    chosen = min(candidates, key=lambda row: (row["total_event_loss_samples"],
                 abs(round(100 * row["k"]) - 5), row["k"]))
    return float(chosen["k"])

def selection_row(study, fault, method, k, h, mask):
    ev = event_arrays(mask)
    return {"study": study, "fault": fault, "method": method, "k": k, "selection_h": float(h),
            "selection_n": len(mask), "total_event_loss_samples": int(ev["event_loss_min"].sum() // 3),
            "mean_event_loss_min": float(ev["event_loss_min"].mean()),
            "prefault_alarm_pct": float(ev["prefault"].mean() * 100),
            "clean_detection_pct": float(ev["clean"].mean() * 100)}

def policy_spec(auto, n_methods, selected=None):
    policies = ["k005", "k007"] + (["auto_dev"] if auto else [])
    values = np.repeat(np.array(KS)[:, None], n_methods, axis=1)
    if auto:
        values = np.concatenate([values, np.asarray(selected, dtype=float)[None, :]], axis=0)
    return policies, values

def do_n(assets, args, run):
    roots = {"development": find_n_root("development", args.n_development_root),
             "final": find_n_root("final", args.n_final_root)}
    run.log("N 数据路径已找到: " + str(roots))
    index = assets.json("assets/N_INDEX.json")
    pca = assets.npz("assets/PCA.npz")
    states = {f: [assets.json(f"assets/N/F{f:02d}/{m}.json") for m in METHODS_N] for f in FAULTS_N}
    rcal = rn.residual(load_n(subset(index, "normal_validation", 0, 250), roots, run), pca)
    limits, grid_limits, policies_by_fault, k_by_fault = {}, {}, {}, {}
    grid = AUTO_GRID if args.auto_k else KS
    for f in FAULTS_N:
        grid_limits[f] = {}
        for j, state in enumerate(states[f][:2]):
            if state["status"] != "ESTIMABLE" or state["cusum_k"] != .05:
                raise ValueError("本补测包要求被比较的 Top-5 状态为现有可估计 k=0.05 状态")
            u = rn.scalar_input(rcal, state)
            for k in grid:
                recal = float(np.percentile(rn.cusum(u, k).max(1), 95, method="linear"))
                if k == .05:
                    run.close_array(f"N/F{f:02d}/{j}/normal_h", recal, state["control_limit"])
                grid_limits[f][j, k] = state["control_limit"] if k == .05 else recal
    del rcal
    run.log("N: 34 个分支的正常门槛校准完成")
    selection = []
    if args.auto_k:
        auto_index = assets.json("assets/N_AUTO_INDEX.json")
        for f in FAULTS_N:
            rdev = rn.residual(load_n(subset(auto_index, "fault_development", f, 250), roots, run), pca)
            chosen = []
            for j, state in enumerate(states[f][:2]):
                u = rn.scalar_input(rdev, state)
                candidates = [selection_row("N", f, METHODS_N[j], k, grid_limits[f][j, k],
                              rn.cusum(u, k) > grid_limits[f][j, k]) for k in AUTO_GRID]
                kstar = choose_k(candidates)
                for row in candidates:
                    row["selected"] = row["k"] == kstar
                selection.extend(candidates)
                chosen.append(kstar)
            del rdev
            policies_by_fault[f], k_by_fault[f] = policy_spec(True, 2, chosen)
            run.log(f"N F{f:02d}: 开发集选 k={chosen}，尚未读取本组最终测试数据")
        write_csv(run.out / "N_AUTO_SELECTION.csv", selection)
        write_json(run.out / "N_AUTO_SELECTION_FIXED.json", {
            "grid": list(AUTO_GRID), "criterion": "min total complete-episode loss; ties closest to 0.05 then smaller k",
            "selection_role": "fault_development", "normal_h_role": "normal_validation", "evaluation_used": False,
            "selected_k": {str(f): k_by_fault[f][-1].tolist() for f in FAULTS_N}})
    else:
        for f in FAULTS_N:
            policies_by_fault[f], k_by_fault[f] = policy_spec(False, 2)
    for f in FAULTS_N:
        limits[f] = np.array([[grid_limits[f][j, float(krow[j])] for j in range(2)] for krow in k_by_fault[f]])
    rnormal = rn.residual(load_n(subset(index, "normal_evaluation", 0, 250), roots, run), pca)
    normal_masks = {}
    for f in FAULTS_N:
        masks = np.empty((len(policies_by_fault[f]), 2, 250, 960), dtype=bool)
        for j, state in enumerate(states[f][:2]):
            u = rn.scalar_input(rnormal, state)
            for ki, k in enumerate(k_by_fault[f][:, j]):
                masks[ki, j] = rn.cusum(u, k) > limits[f][ki, j]
        normal_masks[f] = masks
    del rnormal
    result_rows = []
    for f in FAULTS_N:
        baseline = assets.npz(f"assets/N/F{f:02d}/BASELINE.npz")
        old_normal = unpack(baseline["normal_alarm_packbits"], "little")
        old_fault = unpack(baseline["fault_alarm_packbits"], "little")
        run.exact(f"N/F{f:02d}/normal_alarm_bitmap", normal_masks[f][0], old_normal[:2])
        r = rn.residual(load_n(subset(index, "fault_final", f, 250), roots, run), pca)
        masks = np.empty((len(policies_by_fault[f]), 2, 250, 960), dtype=bool)
        for j, state in enumerate(states[f][:2]):
            u = rn.scalar_input(r, state)
            for ki, k in enumerate(k_by_fault[f][:, j]):
                masks[ki, j] = rn.cusum(u, k) > limits[f][ki, j]
        del r
        run.exact(f"N/F{f:02d}/fault_alarm_bitmap", masks[0], old_fault[:2])
        for j, method in enumerate(METHODS_N[:2]):
            for ki, k in enumerate(k_by_fault[f][:, j]):
                result_rows.append(metrics("N", f, method, k, limits[f][ki, j], masks[ki, j],
                                           normal_masks[f][ki, j], baseline["shutdown_first_sample"], policies_by_fault[f][ki]))
        for j, method in enumerate(METHODS_N[2:], start=2):
            state = states[f][j]
            result_rows.append(metrics("N", f, method, state["cusum_k"], state["control_limit"],
                                       old_fault[j], old_normal[j], baseline["shutdown_first_sample"], "frozen_comparator"))
        save_npz_new(run.out / f"N_F{f:02d}_PAIRED.npz", policies=np.array(policies_by_fault[f]),
            k_by_policy=k_by_fault[f], methods=np.array(METHODS_N[:2]),
            run_ids=np.arange(1, 251), thresholds=limits[f],
            fault_masks=np.packbits(masks, axis=-1, bitorder="little"),
            normal_masks=np.packbits(normal_masks[f], axis=-1, bitorder="little"),
            shutdown_sample=baseline["shutdown_first_sample"], bitorder=np.array("little"))
        run.log(f"N F{f:02d}: 0.05 全部报警位复现通过；0.07 对照完成")
    write_csv(run.out / "N_PER_FAULT.csv", result_rows)
    macro = []
    for method in METHODS_N:
        groups = [(policy, [r for r in result_rows if r["method"] == method and r["policy"] == policy]) for policy in policies_by_fault[FAULTS_N[0]]] if method in METHODS_N[:2] else [
            ("frozen_per_fault", [r for r in result_rows if r["method"] == method])]
        for policy, rr in groups:
            if len(rr) != 17:
                raise ValueError("N 宏平均必须包含全部 17 类故障")
            row = {"method": method, "policy": policy,
                   "k": .05 if policy == "k005" else .07 if policy == "k007" else "selected_per_fault" if policy == "auto_dev" else "frozen_per_fault"}
            for key in rr[0]:
                if key not in {"study", "fault", "method", "policy", "k", "h"}:
                    row[key] = float(np.mean([r[key] for r in rr]))
            macro.append(row)
    write_csv(run.out / "N_MACRO.csv", macro)
    run.rows.extend(result_rows)
    run.deltas.extend(delta_rows([r for r in result_rows if r["method"] in METHODS_N[:2]]))
    return {"macro": macro, "branch_comparisons": 34, "auto_k": args.auto_k,
            "raw_trajectories": 9000 if args.auto_k else 4750}

def project_dv(values, pca, basis):
    z = (values - pca["mean"]) / pca["scale"]
    r = z - (z @ pca["loadings"].T) @ pca["loadings"]
    sub = r @ basis
    q = np.sum(r * r, axis=2)
    if not np.allclose(np.sum(sub * sub, axis=2), q, rtol=1e-9, atol=1e-9):
        raise ValueError("DV 残差基不匹配")
    return {"z": sub, "Q": q}

def dv_inputs(obj, state):
    des, model, pars, methods, idx = state
    mean = rd.project_axes(obj, des, model["mean"])
    moment = rd.project_axes(obj, des, model["moment"])
    keys = ("abs_mean", "abs_sd", "sq_mean", "sq_sd")
    xm = rd.transform_inputs(mean, {k: pars["mean_" + k] for k in keys})
    xn = rd.transform_inputs(moment, {k: pars["moment_" + k] for k in keys})
    # Full eight-axis projections above preserve the original reduction order.
    # Select only after transforming; recurrence has no cross-column reduction.
    inputs = np.empty((len(obj["z"]), 960, len(idx)), dtype=np.float64)
    for column, candidate in enumerate(idx):
        if candidate == 48:
            inputs[:, :, column] = (np.log1p(obj["Q"]) - float(des["q_mean"])) / float(des["q_sd"])
        else:
            family, axis = divmod(candidate, 16)
            inputs[:, :, column] = xm[family][:, :, axis] if axis < 8 else xn[family][:, :, axis - 8]
    return inputs

def dv_scores(inputs, k, q_column):
    out = rd.recurrence(inputs, k=k)
    # The Q branch uses the original cumsum/minimum accumulation, upper-only.
    qi = inputs[:, :, q_column]
    qk = float(k) if np.ndim(k) == 0 else float(k[q_column])
    p = np.concatenate([np.zeros((len(qi), 1)), np.cumsum(qi - qk, axis=1)], axis=1)
    out[:, :, q_column] = (p - np.minimum.accumulate(p, axis=1))[:, 1:]
    if not np.isfinite(out).all():
        raise ValueError("DV 非有限统计量")
    return out

def dv_state(assets, f):
    prefix = f"assets/DV/F{f:02d}/"
    des = assets.npz(prefix + "DESIGN.npz")
    model = assets.npz(prefix + "MODEL.npz")
    pars = assets.npz(prefix + "PARAMETERS.npz")
    choices = assets.json(prefix + "SELECTION.json")["choices"]
    methods = list(choices)
    idx = np.array([choices[m] for m in methods])
    return des, model, pars, methods, idx

def do_dv(assets, args, run):
    root = Path(args.dv_root) if args.dv_root else DV_DEFAULT
    if not (root / "data").is_dir():
        raise FileNotFoundError("未找到 DV data 目录: " + str(root) + "；可用 --dv-root 指定")
    index = assets.json("assets/DV_INDEX.json")
    if args.auto_k:
        index += assets.json("assets/DV_AUTO_INDEX.json")
    pca = assets.npz("assets/PCA.npz")
    basis = np.load(io.BytesIO(assets.read("assets/BASIS.npy")), allow_pickle=False)
    def role(name, fault, n):
        return project_dv(load_dv(subset(index, name, fault, n), root, run), pca, basis)
    anchor = role("normal_anchor", 0, 32)
    states = {f: dv_state(assets, f) for f in (3, 9, 15)}
    selected, selection = {}, []
    if args.auto_k:
        inner = role("normal_inner_calibration", 0, 59)
        for f in (3, 9, 15):
            state = states[f]
            methods = state[3]
            qcol = methods.index("Q_CUSUM")
            ua, ui = dv_inputs(anchor, state), dv_inputs(inner, state)
            us = dv_inputs(role("fault_selection", f, 75), state)
            candidates = []
            for k in AUTO_GRID:
                anchors = np.quantile(dv_scores(ua, k, qcol).max(1), .9, axis=0, method="linear")
                if not np.all(anchors > 0):
                    raise ValueError("DV 自动选择的 anchor 非正")
                peaks = dv_scores(ui, k, qcol).max(1)
                h = anchors * np.sort(peaks / anchors, axis=0)[56]
                mask = dv_scores(us, k, qcol) > h
                candidates.extend(selection_row("DV", f, m, k, h[j], mask[:, :, j]) for j, m in enumerate(methods))
            selected[f] = [choose_k([c for c in candidates if c["method"] == m]) for m in methods]
            for c in candidates:
                c["selected"] = c["k"] == selected[f][methods.index(c["method"])]
            selection.extend(candidates)
            run.log(f"DV F{f:02d}: 内层验证选 k={selected[f]}，尚未读取本组最终测试数据")
        del inner, ua, ui, us
        write_csv(run.out / "DV_AUTO_SELECTION.csv", selection)
        write_json(run.out / "DV_AUTO_SELECTION_FIXED.json", {
            "grid": list(AUTO_GRID), "criterion": "min total complete-episode loss; ties closest to 0.05 then smaller k",
            "selection_role": "fault_selection", "normal_h_role": "normal_inner_calibration",
            "inner_rank": 57, "inner_n": 59, "evaluation_used": False, "outer_used_for_selection": False,
            "selected_k": {str(f): dict(zip(states[f][3], selected[f])) for f in (3, 9, 15)}})
    outer = role("normal_outer_calibration", 0, 259)
    normal = role("normal_evaluation", 0, 250)
    result_rows, contrasts = [], []
    for f in (3, 9, 15):
        prefix = f"assets/DV/F{f:02d}/"
        state = states[f]
        des, model, pars, methods, idx = state
        policies, k_values = policy_spec(args.auto_k, len(methods), selected.get(f))
        qcol = methods.index("Q_CUSUM")
        baseline = assets.npz(prefix + "BASELINE.npz")
        run.exact(f"DV/F{f:02d}/method_order", np.array(methods), baseline["methods"])
        stored_inner, stored_outer = assets.npz(prefix + "INNER.npz"), assets.npz(prefix + "OUTER.npz")
        phase = assets.npz(prefix + "PHASES.npz")
        ua, uo, un = (dv_inputs(obj, state) for obj in (anchor, outer, normal))
        uf = dv_inputs(role("fault_evaluation", f, 250), state)
        limits = np.empty((len(policies), len(methods)))
        masks = np.empty((len(policies), len(methods), 250, 960), dtype=bool)
        nmasks = np.empty_like(masks)
        for ki, k in enumerate(k_values):
            apeaks = dv_scores(ua, k, qcol).max(1)
            anchors = np.quantile(apeaks, .9, axis=0, method="linear")
            peaks = dv_scores(uo, k, qcol).max(1)
            if not np.all(anchors > 0):
                raise ValueError("DV 正常 anchor 非正")
            h = anchors * np.sort(peaks / anchors, axis=0)[246]
            if ki == 0:
                run.close_array(f"DV/F{f:02d}/anchor_peaks", apeaks, stored_inner["anchor_peaks"][:, idx])
                run.close_array(f"DV/F{f:02d}/outer_peaks", peaks, stored_outer["peaks"][:, idx])
                run.close_array(f"DV/F{f:02d}/thresholds", h, stored_outer["thresholds"][idx])
                h = stored_outer["thresholds"][idx]
            limits[ki] = h
            masks[ki] = (dv_scores(uf, k, qcol) > h).transpose(2, 0, 1)
            nmasks[ki] = (dv_scores(un, k, qcol) > h).transpose(2, 0, 1)
        run.exact(f"DV/F{f:02d}/fault_alarm_bitmap", np.packbits(masks[0], axis=-1), baseline["fault_masks"])
        run.exact(f"DV/F{f:02d}/normal_alarm_bitmap", np.packbits(nmasks[0], axis=-1), baseline["normal_masks"])
        for j, method in enumerate(methods):
            for ki, k in enumerate(k_values[:, j]):
                result_rows.append(metrics("DV", f, method, k, limits[ki, j], masks[ki, j], nmasks[ki, j], phase["shutdown_sample"], policies[ki]))
        comparison, reference = {
            3: ("same_mean_signed_axis_squared", "mean_pool_signed"),
            9: ("moment_pool_signed", "mean_pool_signed"),
            15: ("moment_pool_squared", "mean_pool_squared")}[f]
        for policy in policies:
            a = next(r for r in result_rows if r["fault"] == f and r["method"] == comparison and r["policy"] == policy)
            b = next(r for r in result_rows if r["fault"] == f and r["method"] == reference and r["policy"] == policy)
            contrasts.append({"fault": f, "policy": policy, "k": a["k"] if policy != "auto_dev" else "auto_per_branch",
                "comparison": comparison, "reference": reference, "comparison_k": a["k"], "reference_k": b["k"],
                "contrast_mean_loss_min": a["primary_loss_min"] - b["primary_loss_min"],
                "contrast_normal_far_pp": a["normal_far_pct"] - b["normal_far_pct"],
                "contrast_clean_120_pp": a["early_120_pct"] - b["early_120_pct"],
                "predicted_sign_of_loss_contrast": "positive" if f == 3 else "negative"})
        save_npz_new(run.out / f"DV_F{f:02d}_PAIRED.npz", policies=np.array(policies), k_by_policy=k_values, methods=np.array(methods),
            run_ids=np.arange(1, 251), candidate_indices=idx, thresholds=limits,
            fault_masks=np.packbits(masks, axis=-1), normal_masks=np.packbits(nmasks, axis=-1),
            shutdown_sample=phase["shutdown_sample"], constancy_marker=phase["common_constancy_marker_start"], bitorder=np.array("big"))
        run.log(f"DV F{f:02d}: 9 个分支的 0.05 全部报警位复现通过；0.07 对照完成")
    write_csv(run.out / "DV_PER_FAULT.csv", result_rows)
    write_csv(run.out / "DV_PRIMARY_CONTRASTS.csv", contrasts)
    run.rows.extend(result_rows)
    run.deltas.extend(delta_rows(result_rows))
    run.contrasts.extend(contrasts)
    return {"primary_contrasts": contrasts, "branch_comparisons": 27, "auto_k": args.auto_k,
            "raw_trajectories": 1575 if args.auto_k else 1291}
