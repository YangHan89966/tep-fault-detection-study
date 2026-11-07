"""Fixed first-paper 22-arm scoring; no fitting, tuning, or simulation.

Each original R5 state supplies F3 Signed, the SAME selected axis Squared,
F15 selected moment Squared, and selected mean Squared. All use the original
two-sided CUSUM, including Squared. The last two arms use the same historical
F9 affine map on native and controller-input records. Paths in metadata are
relative to the portable study root.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Mapping
import numpy as np

N_SAMPLES = 960
PRE_SAMPLES = 160
PENALTY_MINUTES = 2400
CUSUM_K = .05
HISTORICAL_CHANNEL = "H_D01_moment_A01_signed"
HISTORICAL_MAP_SHA256 = "7a20407a0fafa0028931de8d73c32b2ceae9585175e59dadd836777a97efcf66"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}
    for k, a in result.items():
        if a.dtype.kind in "fc" and not np.isfinite(a).all():
            raise ValueError(f"Nonfinite frozen parameter {path}:{k}")
    return result


def _require_shape(a, shape, label):
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected finite float64 {shape}, got {a.shape}/{a.dtype}")


def raw_views(arrays):
    if "native_values" in arrays:
        native = np.asarray(arrays["native_values"])
        cache = np.asarray(arrays["fast_control_input_measurements"])
        _require_shape(cache, (960, 41), "controller input cache")
        controller = np.concatenate((cache, native[:, 41:52]), axis=1)
    else:
        native = np.asarray(arrays["native"])
        controller = np.asarray(arrays["controller_input"])
    _require_shape(native, (960, 52), "native")
    _require_shape(controller, (960, 52), "controller input")
    if not np.array_equal(native[:, 41:], controller[:, 41:]):
        raise ValueError("Manipulated-variable values differ between paired recordings")
    return native, controller


def project_r5(raw, pca, basis):
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = raw[None]
    if raw.ndim != 3 or raw.shape[1:] != (960, 52) or raw.dtype != np.float64 or not np.isfinite(raw).all():
        raise ValueError("Projection expects finite float64 (runs,960,52)")
    standardized = (raw - pca["mean"]) / pca["scale"]
    residual = standardized - (standardized @ pca["loadings"].T) @ pca["loadings"]
    return {"z": residual @ basis, "Q": np.sum(residual * residual, axis=2)}


def recurrence(inputs):
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 3 or not np.isfinite(inputs).all():
        raise ValueError("CUSUM expects finite (runs,time,channels)")
    cp = np.zeros((inputs.shape[0], inputs.shape[2]))
    cm = cp.copy()
    out = np.empty_like(inputs)
    for t in range(inputs.shape[1]):
        cp = np.maximum(0., (cp + inputs[:, t, :]) - CUSUM_K)
        cm = np.maximum(0., (cm - inputs[:, t, :]) - CUSUM_K)
        out[:, t, :] = np.maximum(cp, cm)
    return out


def load_r5_state(model_dir, rep, fault):
    model_dir = Path(model_dir)
    design = load_npz(model_dir / "DESIGN.npz")
    axes = load_npz(model_dir / "MODEL.npz")
    params = load_npz(model_dir / "PARAMETERS.npz")
    selection = json.loads((model_dir / "SELECTION.json").read_text())
    _require_shape(design["centre"], (16,), "design centre")
    _require_shape(design["whitener"], (16, 16), "design whitener")
    for pool in ("mean", "moment"):
        _require_shape(axes[pool], (8, 16), pool + " axes")
        for name in ("sq_mean", "sq_sd"):
            _require_shape(params[pool + "_" + name], (8,), pool + "_" + name)
        if not np.all(params[pool + "_sq_sd"] > 0):
            raise ValueError("Nonpositive frozen square-score scale")
    if fault not in (3, 15) or int(selection["fault"]) != fault or not 1 <= rep <= 5:
        raise ValueError("Invalid fixed R5 state identity")
    roles = (("mean_pool_signed", "same_mean_signed_axis_squared") if fault == 3
             else ("moment_pool_squared", "mean_pool_squared"))
    chosen = [int(selection["choices"][name]) for name in roles]
    km, ka = len(axes["mean"]), len(axes["moment"])
    if fault == 3 and chosen[1] != 2 * (km + ka) + chosen[0]:
        raise ValueError("C1 does not use the same Signed-selected axis")
    metadata = []
    for name, ci in zip(roles, chosen):
        family_index, within_family = divmod(ci, km + ka)
        pool = "mean" if within_family < km else "moment"
        axis = within_family if pool == "mean" else within_family - km
        family = ("signed", "absolute", "squared")[family_index]
        metadata.append({"arm_id": f"R5_rep{rep:02}_F{fault:02}_{name}",
                         "study": "R5_fixed", "state": rep, "rep": rep, "fault": fault,
                         "runview": "native", "view": "native", "choice_key": name,
                         "source_candidate_index": ci, "source_candidate_name": selection["candidate_names"][ci],
                         "pool": pool, "axis_index_0based": axis, "scoretransform": family,
                         "side": "two", "k": CUSUM_K,
                         "score_location": None if family == "signed" else float(params[pool + "_sq_mean"][axis]),
                         "score_scale": None if family == "signed" else float(params[pool + "_sq_sd"][axis]),
                         "signed_score_additional_normalization": False,
                         "model_directory": f"assets/R5/rep{rep:02}/F{fault:02}"})
    return {"rep": rep, "fault": fault, "design": design, "axes": axes,
            "params": params, "selection": selection, "roles": roles,
            "indices": chosen, "metadata": metadata}


def r5_selected_scores(projected, state):
    design, axes, params = state["design"], state["axes"], state["params"]
    features = (projected["z"] - design["centre"]) @ design["whitener"]
    # Retain original full blocks and reduction order rather than one-axis dots.
    mean = np.einsum("ntd,ad->nta", features, axes["mean"], optimize=True)
    mean_squared = (mean * mean - params["mean_sq_mean"]) / params["mean_sq_sd"]
    if state["fault"] == 3:
        j = state["indices"][0]
        charts = np.stack((recurrence(mean)[:, :, j], recurrence(mean_squared)[:, :, j]), axis=2)
        inputs = np.stack((mean[:, :, j], mean_squared[:, :, j]), axis=2)
    else:
        moment = np.einsum("ntd,ad->nta", features, axes["moment"], optimize=True)
        moment_squared = (moment * moment - params["moment_sq_mean"]) / params["moment_sq_sd"]
        km, ka = len(axes["mean"]), len(axes["moment"])
        jm = state["indices"][0] - 2 * (km + ka) - km
        jn = state["indices"][1] - 2 * (km + ka)
        charts = np.stack((recurrence(moment_squared)[:, :, jm], recurrence(mean_squared)[:, :, jn]), axis=2)
        inputs = np.stack((moment_squared[:, :, jm], mean_squared[:, :, jn]), axis=2)
    if charts.shape[1:] != (960, 2) or not np.isfinite(charts).all():
        raise ValueError("Malformed fixed R5 chart")
    return charts, inputs


def load_historical_map(path):
    path = Path(path)
    if file_sha256(path) != HISTORICAL_MAP_SHA256:
        raise ValueError("Historical affine map identity differs")
    source = load_npz(path)
    j = list(source["channels"]).index(HISTORICAL_CHANNEL)
    beta = source["beta"][:, j].copy()
    _require_shape(beta, (52,), "historical beta")
    return {"channel": HISTORICAL_CHANNEL, "beta": beta,
            "intercept": float(source["intercept"][j]), "source_index": j,
            "sha256": HISTORICAL_MAP_SHA256}


def historical_inputs(native, controller, fixed_map):
    _require_shape(native, (960, 52), "native")
    _require_shape(controller, (960, 52), "controller input")
    beta, intercept = fixed_map["beta"], fixed_map["intercept"]
    return np.stack((native @ beta + intercept, controller @ beta + intercept), axis=-1)


def load_scorers(root):
    assets = Path(root) / "assets"
    pca = load_npz(assets / "pca500.npz")
    basis = np.load(assets / "basis500.npy", allow_pickle=False)
    for name in ("mean", "scale"):
        _require_shape(pca[name], (52,), "PCA " + name)
    _require_shape(pca["loadings"], (36, 52), "PCA loadings")
    _require_shape(basis, (52, 16), "residual basis")
    if not np.all(pca["scale"] > 0):
        raise ValueError("Nonpositive PCA scale")
    states, metadata = [], []
    for rep in range(1, 6):
        for fault in (3, 15):
            state = load_r5_state(assets / "R5" / f"rep{rep:02}" / f"F{fault:02}", rep, fault)
            states.append(state)
            metadata.extend(state["metadata"])
    fixed_map = load_historical_map(assets / "FIXED_AFFINE_MAPS.npz")
    for view in ("native", "controller_input"):
        metadata.append({"arm_id": "F9_historical_" + view, "study": "F9_fixed_historical",
                         "state": None, "rep": None, "fault": 9, "runview": view, "view": view,
                         "choice_key": HISTORICAL_CHANNEL, "source_candidate_index": fixed_map["source_index"],
                         "source_candidate_name": HISTORICAL_CHANNEL, "pool": "moment",
                         "axis_index_0based": None, "historical_axis_label": "moment_A01", "scoretransform": "signed",
                         "side": "two", "k": CUSUM_K, "score_location": None, "score_scale": None,
                         "signed_score_additional_normalization": False, "map_sha256": fixed_map["sha256"],
                         "affine_intercept": fixed_map["intercept"]})
    if len(metadata) != 22 or len({r["arm_id"] for r in metadata}) != 22:
        raise ValueError("Expected exactly 22 unique frozen arms")
    for j, row in enumerate(metadata):
        row.update(column=j, n_samples=960, pre_samples=160,
                   reset="episode_beginning_only", alarm_comparison="strict_greater",
                   calibration_statistic="max_over_all_960_samples", sample_minutes=3)
    def score(arrays):
        native, controller = raw_views(arrays)
        projected = project_r5(native, pca, basis)
        pieces = [r5_selected_scores(projected, state)[0][0] for state in states]
        pieces.append(recurrence(historical_inputs(native, controller, fixed_map)[None])[0])
        result = np.concatenate(pieces, axis=1)
        _require_shape(result, (960, 22), "fixed 22-arm chart")
        return result
    return metadata, score


def events(charts, thresholds):
    charts = np.asarray(charts, dtype=np.float64)
    single = charts.ndim == 2
    if single:
        charts = charts[None]
    if charts.ndim != 3 or charts.shape[1] != 960 or not np.isfinite(charts).all():
        raise ValueError("Expected finite (runs,960,arms) charts")
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (charts.shape[2],) or not np.isfinite(thresholds).all():
        raise ValueError("Expected one finite threshold per arm")
    above = charts > thresholds[None, None, :]
    pre = above[:, :160].any(axis=1)
    post = above[:, 160:]
    det = post.any(axis=1)
    delay = np.where(det, post.argmax(axis=1) * 3, 2400)
    clean = det & ~pre
    result = {"normal_alarm": above.any(axis=1), "pre_alarm": pre, "detected_post": det,
              "clean": clean, "delay_minutes": delay, "loss_minutes": np.where(clean, delay, 2400),
              "clean_120": clean & (delay <= 120), "clean_480": clean & (delay <= 480),
              "post_occupancy_points": post.sum(axis=1)}
    return {k: v[0] for k, v in result.items()} if single else result


def calibrate_charts(charts, rank=190, expected_n=199):
    charts = np.asarray(charts, dtype=np.float64)
    if charts.ndim != 3 or charts.shape[:2] != (expected_n, 960) or not np.isfinite(charts).all() or not 1 <= rank <= expected_n:
        raise ValueError("Incomplete or invalid independent calibration panel")
    peaks = charts.max(axis=1)
    return np.sort(peaks, axis=0)[rank - 1], peaks


def primary_loss_contrast(outcome, metadata, fault):
    loss = np.asarray(outcome["loss_minutes"], dtype=np.float64)
    if loss.ndim != 2 or loss.shape[1] != len(metadata):
        raise ValueError("Expected run-level loss array")
    lookup = {r["arm_id"]: j for j, r in enumerate(metadata)}
    if fault in (3, 15):
        diffs = []
        for rep in range(1, 6):
            prefix = f"R5_rep{rep:02}_F{fault:02}_"
            a, b = (("same_mean_signed_axis_squared", "mean_pool_signed") if fault == 3
                    else ("moment_pool_squared", "mean_pool_squared"))
            diffs.append(loss[:, lookup[prefix + a]] - loss[:, lookup[prefix + b]])
        return ("C1_same_axis_squared_minus_signed" if fault == 3 else "C3_moment_minus_mean_squared"), np.stack(diffs, axis=1).mean(axis=1)
    if fault == 9:
        return "F9_controller_input_minus_native", loss[:, lookup["F9_historical_controller_input"]] - loss[:, lookup["F9_historical_native"]]
    raise ValueError("Only F3, F9 and F15 are specified")
