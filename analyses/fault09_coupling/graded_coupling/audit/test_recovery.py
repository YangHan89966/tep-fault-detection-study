"""Independent noise-event instrumentation and negative recovery checks."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import MethodType
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from recover_noise import recover, uniform_words


def main(source: Path, raw: Path, output: Path):
    started = time.perf_counter()
    meta = json.loads((raw / "RUN_INFO.json").read_text())
    with np.load(raw / "RAW.npz", allow_pickle=False) as f:
        arrays = {name: f[name].copy() for name in f.files}
    restored = recover(arrays, meta, verify_full_rng_trace=True)
    spec = importlib.util.spec_from_file_location("noise_test_rng_engine", source / "code/rng_engine.py")
    rng_engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rng_engine
    spec.loader.exec_module(rng_engine)
    engine = rng_engine.load_engine(source / "engine")
    key = int(meta["rng_key_hex"], 16)
    sim = engine.TEPSimulator(random_seed=1, control_mode=engine.ControlMode.CLOSED_LOOP, backend="python")
    provider = rng_engine.PhiloxUniform(key)
    sim.process._tesub7 = MethodType(lambda _p, sign: provider(sign), sim.process)
    sim.initialize()
    original_noise = sim.process._tesub6
    original_evaluate = sim.process.evaluate
    current_noise = np.zeros(41)
    current_events = np.full(41, -1, dtype=np.int64)
    evaluation = 0
    recovered_events, recovered_noise = [], []
    pending = []

    def observe_noise(std):
        before = provider.draw_count
        value = original_noise(std)
        after = provider.draw_count
        if after - before != 12:
            raise AssertionError("Original TESUB6 did not use exactly twelve uniforms")
        pending.append((before, float(std), value))
        return value

    def observe_evaluate(t, yy):
        nonlocal evaluation
        pending.clear()
        tp = sim.process._teproc
        gas, product = t >= tp.tgas, t >= tp.tprod
        out = original_evaluate(t, yy)
        evaluation += 1
        ids = (list(range(22)) if t > 0 else [])
        if gas:
            ids += list(range(22, 36))
        if product:
            ids += list(range(36, 41))
        if len(ids) != len(pending):
            raise AssertionError("Unexpected independent source noise-call count")
        for i, (start, std, value) in zip(ids, pending):
            if std != tp.xns[i]:
                raise AssertionError("Noise-call coordinate/SD mismatch")
            current_noise[i] = value
            current_events[i] = start
        if evaluation % 180 == 177:
            recovered_events.append(current_events.copy())
            recovered_noise.append(current_noise.copy())
        return out

    sim.process._tesub6 = observe_noise
    sim.process.evaluate = observe_evaluate
    # Software-only run: same archived normal key, no detector outcomes.
    engine.horizon(sim, np, .6, 0, None, 180)
    independent_events = np.asarray(recovered_events)
    independent_noise = np.asarray(recovered_noise)
    n = independent_events.shape[0]
    if not np.array_equal(independent_events, restored["noise_event_id"][:n]):
        raise AssertionError("Recovered event mapping differs from instrumented source")
    if not np.array_equal(independent_noise, restored["input_noise"][:n]):
        raise AssertionError("Recovered noise differs from literal source TESUB6")

    # Sequential provider vs random access, including non-block-aligned offsets.
    provider_check = rng_engine.PhiloxUniform(key, buffer_size=17)
    sequential = np.array([provider_check(1) for _ in range(123)])
    for start, count in ((0, 123), (1, 19), (3, 18), (17, 80), (121, 2)):
        if not np.array_equal(uniform_words(key, start, count), sequential[start:start+count]):
            raise AssertionError("Counter-based access differs from original buffered stream")

    checks = {}
    bad = dict(meta, shutdown_occurred=True)
    try:
        recover(arrays, bad)
    except ValueError:
        checks["shutdown_rejected"] = True
    else:
        raise AssertionError("Shutdown metadata was accepted")
    bad = dict(arrays, diagnostic_rng_draw_count=arrays["diagnostic_rng_draw_count"].copy())
    bad["diagnostic_rng_draw_count"][12] += 1
    try:
        recover(bad, meta)
    except ValueError:
        checks["changed_draw_anchor_rejected"] = True
    else:
        raise AssertionError("Changed RNG anchor was accepted")
    bad = dict(arrays, diagnostic_D_feed_temperature=arrays["diagnostic_D_feed_temperature"].copy())
    bad["diagnostic_D_feed_temperature"][3] += 1.
    try:
        recover(bad, meta)
    except ValueError:
        checks["changed_temperature_rejected"] = True
    else:
        raise AssertionError("Changed temperature was accepted")
    result = {
        "status": "PASS_INDEPENDENT_SOURCE_NOISE_INSTRUMENTATION",
        "software_only": True, "new_scientific_trajectories": 0,
        "software_simulation_hours": .6, "original_run_id": meta.get("run_id"),
        "normal_full_48h_rng_trace_exact": restored["diagnostics"]["full_rng_trace_sha256"],
        "literal_source_noise_event_cells_exact": int(independent_events.size),
        "literal_source_noise_values_exact": int(independent_noise.size),
        "all_41_measurements_and_delayed_composition_hold_checked": True,
        "random_access_matches_original_buffered_provider": True,
        "negative_checks": checks, "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    main(args.source, args.run, args.output)
