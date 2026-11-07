"""Instrument original TESUB6/TESUB7 for one complete archived 48-hour run.

This engineering replay uses an existing key, original process/controller and
original single stream. Observers only copy original values. No coupling scores
or detector outcomes are computed. All 12 uniforms are independently captured.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import MethodType
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recover_noise import recover


def verify(source: Path, raw: Path, output: Path):
    started = time.perf_counter()
    meta = json.loads((raw / "RUN_INFO.json").read_text())
    with np.load(raw / "RAW.npz", allow_pickle=False) as f:
        arrays = {name: f[name].copy() for name in f.files}
    restored = recover(arrays, meta, verify_full_rng_trace=True)
    spec = importlib.util.spec_from_file_location("full_noise_capture_rng_engine", source / "code/rng_engine.py")
    rng_engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rng_engine
    spec.loader.exec_module(rng_engine)
    engine = rng_engine.load_engine(source / "engine")
    key = int(meta["rng_key_hex"], 16)
    sim = engine.TEPSimulator(random_seed=1, control_mode=engine.ControlMode.CLOSED_LOOP, backend="python")
    provider = rng_engine.PhiloxUniform(key)
    capturing = False
    uniform_capture = []

    def observe_uniform(_process, sign):
        value = provider(sign)
        if capturing:
            if sign != 1:
                raise AssertionError("Noise requested a non-unit uniform sign")
            uniform_capture.append(value)
        return value

    sim.process._tesub7 = MethodType(observe_uniform, sim.process)
    sim.initialize()
    if provider.draw_count != meta["initialization_rng_draw_count"]:
        raise AssertionError("Initialization consumption differs")
    original_noise = sim.process._tesub6
    original_evaluate = sim.process.evaluate
    current_noise = np.zeros(41)
    current_events = np.full(41, -1, dtype=np.int64)
    current_uniforms = np.zeros((41, 12))
    evaluation = 0
    captured_events, captured_noise, captured_uniforms = [], [], []
    pending = []

    def observe_noise(std):
        nonlocal capturing
        before = provider.draw_count
        uniform_capture.clear()
        capturing = True
        try:
            value = original_noise(std)
        finally:
            capturing = False
        after = provider.draw_count
        if after - before != 12 or len(uniform_capture) != 12:
            raise AssertionError("Original TESUB6 did not use exactly twelve uniforms")
        pending.append((before, float(std), value, np.array(uniform_capture)))
        return value

    def observe_evaluate(t, yy):
        nonlocal evaluation
        pending.clear()
        tp = sim.process._teproc
        gas, product = t >= tp.tgas, t >= tp.tprod
        result = original_evaluate(t, yy)
        evaluation += 1
        ids = (list(range(22)) if t > 0 else [])
        if gas:
            ids += list(range(22, 36))
        if product:
            ids += list(range(36, 41))
        if len(ids) != len(pending):
            raise AssertionError("Unexpected independent source noise-call count")
        for i, (event, std, value, uniforms) in zip(ids, pending):
            if std != tp.xns[i]:
                raise AssertionError("Noise-call coordinate/SD mismatch")
            current_noise[i] = value
            current_events[i] = event
            current_uniforms[i] = uniforms
        if evaluation % 180 == 177:
            captured_events.append(current_events.copy())
            captured_noise.append(current_noise.copy())
            captured_uniforms.append(current_uniforms.copy())
        if evaluation % 18000 == 0:
            print(f"CAPTURE {meta['run_id']} {evaluation}/172800", flush=True)
        return result

    sim.process._tesub6 = observe_noise
    sim.process.evaluate = observe_evaluate
    activation = []
    original_set = sim.set_disturbance

    def observe_set(index, value):
        result = original_set(index, value)
        activation.append({"index_1based": int(index), "value": int(value),
                           "time_hours": float(sim.time), "step_count": int(sim.step_count),
                           "flags_after": sim.process._idv.tolist()})
        return result

    sim.set_disturbance = observe_set
    tap = rng_engine._recorder_class(engine.dual)(sim, provider, 180, True)
    result = engine.horizon(sim, np, 48., meta["fault_id"], meta["fault_onset_hours"], 180)
    native = np.concatenate((result["measurements"][1:], result["manipulated_vars"][1:, :11]), axis=1)
    cached = np.asarray(tap.cached_measurements)
    controller_input = native.copy()
    controller_input[:, :41] = cached
    replayed = {
        "native_values": native,
        "native_xmv12": result["manipulated_vars"][1:, 11],
        "native_time_seconds": result["time_seconds"],
        "record_time_seconds": result["time_seconds"][1:],
        "diagnostic_D_feed_temperature": np.asarray(tap.input_temperature),
        "diagnostic_IDV": np.asarray(tap.disturbance_flags, dtype=np.int32),
        "diagnostic_rng_draw_count": np.asarray(tap.rng_counts_at_records, dtype=np.int64),
        "controller_input_values": controller_input,
        "fast_control_input_measurements": cached,
        "phase": np.asarray(tap.phase, dtype=np.int64),
        "ctrl11_accumulated_integral": np.asarray(tap.integral_at_records),
    }
    checks = {}
    for name, value in replayed.items():
        checks["raw_exact_" + name] = np.array_equal(value, arrays[name])
    noise_arrays = {
        "noise_event_id": np.asarray(captured_events),
        "input_noise": np.asarray(captured_noise),
        "original_uniforms": np.asarray(captured_uniforms),
    }
    for name, value in noise_arrays.items():
        checks["independent_capture_exact_" + name] = np.array_equal(value, restored[name])
    summary = tap.summary()
    for name in ("full_step_trace_sha256", "rng_consumption_trace_sha256",
                 "evaluate_calls", "control_calls"):
        checks["metadata_exact_" + name] = summary[name] == meta[name]
    checks["metadata_exact_activation_events"] = activation == meta["activation_events"]
    checks["metadata_exact_physical_state_sha256"] = rng_engine._physical_state_digest(sim, engine.dual) == meta["physical_state_sha256"]
    checks["metadata_exact_archived_state_sha256"] = engine.dual.state_digest(sim) == meta["archived_state_sha256"]
    provider_state = provider.snapshot()
    for name in ("logical_state_sha256", "storage_state_sha256"):
        checks["metadata_exact_" + name] = provider_state[name] == meta["rng_state"][name]
    checks["shutdown_flag_exact"] = result["shutdown_occurred"] == meta["shutdown_occurred"]
    if not all(checks.values()):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "FAIL", "checks": checks}, indent=2))
        raise AssertionError("Source capture differs: " + str([k for k, v in checks.items() if not v]))
    report = {
        "status": "PASS_FULL_48H_LITERAL_SOURCE_NOISE_CAPTURE",
        "software_only_replay": True, "new_scientific_trajectories": 0,
        "run_id": meta["run_id"], "fault_id": meta["fault_id"],
        "key_hex": meta["rng_key_hex"], "duration_hours": 48.,
        "source_uniform_array_cells_verified": int(noise_arrays["original_uniforms"].size),
        "source_generated_uniform_cells_verified": int(restored["has_noise"].sum()) * 12,
        "zero_noise_measurement_cells_before_first_analyzer_sample": int((~restored["has_noise"]).sum()),
        "source_noise_and_event_cells_verified": int(noise_arrays["input_noise"].size),
        "source_noise_event_id_sha256": hashlib.sha256(noise_arrays["noise_event_id"].tobytes()).hexdigest(),
        "source_original_uniforms_sha256": hashlib.sha256(noise_arrays["original_uniforms"].tobytes()).hexdigest(),
        "source_input_noise_sha256": hashlib.sha256(noise_arrays["input_noise"].tobytes()).hexdigest(),
        "checks": checks, "recovery_diagnostics": restored["diagnostics"],
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    verify(args.source, args.run, args.output)
