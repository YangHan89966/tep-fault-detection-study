"""Recover the original measurement-noise uniforms without simulating TEP.

This module replays only the pinned random-walk schedules and measurement-noise
draw counts. It never evaluates the process ODE, changes a saved trajectory, or
fits/scores a detector. It is restricted to non-shutdown normal/Fault 9 runs of
the independently keyed Philox panel. Original 12-uniform transforms are exact.

Source: tep 2.5.0 python_backend.py (_tesub5, _tesub6, _tefunc), original
1-second registered horizon, and rng_engine.PhiloxUniform high32 convention.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

import numpy as np

ALGORITHM = "numpy.Philox4x64_high32_over_2pow32_v1"
DENOMINATOR = 4294967296.0
NOISE_SD = np.array([
    .0012, 18., 22., .05, .2, .21, .3, .5, .01, .0017, .01, 1., .3,
    .125, 1., .3, .115, .01, 1.15, .2, .01, .01,
    .25, .1, .25, .1, .25, .025, .25, .1, .25, .1, .25, .025,
    .05, .05, .01, .01, .01, .5, .5,
], dtype=np.float64)
HSPAN = [.2, .7, .25, .7, .15, .15, 1., 1., .4, 1.5, 2., 1.5]
HZERO = [.5, 1., .5, 1., .25, .25, 2., 2., .5, 2., 3., 2.]


def uniform_words(key: int | str, start: int, count: int) -> np.ndarray:
    """Exact random access to the original logical raw64-word stream."""
    key = int(key, 0) if isinstance(key, str) else int(key)
    start, count = int(start), int(count)
    if not (0 <= key < 2**128 and start >= 0 and count >= 0):
        raise ValueError("Invalid Philox key or word interval")
    block, skip = divmod(start, 4)
    bitgen = np.random.Philox(key=key, counter=block)
    raw = bitgen.random_raw(skip + count)[skip:]
    return (raw >> np.uint64(32)).astype(np.float64) / DENOMINATOR


def _array(arrays: Mapping[str, Any], name: str, shape: tuple[int, ...]):
    value = np.asarray(arrays[name])
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError("Missing/malformed array: " + name)
    return value


def _metadata(metadata: Mapping[str, Any]) -> tuple[int, int, float | None]:
    if metadata.get("rng_kind") != "philox":
        raise ValueError("Only independently keyed Philox runs are supported")
    if metadata.get("shutdown_occurred") is not False:
        raise ValueError("Noise recovery requires an explicitly non-shutdown run")
    if metadata.get("fault_id") not in (0, 9):
        raise ValueError("Only normal/Fault 9 RNG schedules are supported")
    for field, expected in (("duration_hours", 48.), ("final_step_count", 172800),
                            ("initialization_rng_draw_count", 0),
                            ("integration_step_seconds", 1),
                            ("record_interval_seconds", 180)):
        if metadata.get(field) != expected:
            raise ValueError("Unexpected metadata setting: " + field)
    state = metadata["rng_state"]["logical"]
    if state.get("algorithm") != ALGORITHM:
        raise ValueError("Unexpected random-source algorithm")
    key = int(metadata["rng_key_hex"], 16)
    if key != int(state["key_hex"], 16):
        raise ValueError("Inconsistent key metadata")
    fault = int(metadata["fault_id"])
    onset = metadata.get("fault_onset_hours")
    if (fault == 9 and onset != 8.) or (fault == 0 and onset is not None):
        raise ValueError("Unexpected fault-onset setting")
    return key, fault, onset


def _replay_schedule(key: int, fault: int):
    """Return exact event starts/counts and independent temperature diagnostics.

    Normal/F9 never activates the three special pulse walks, so their current
    spline is zero and each update consumes exactly one interval draw. The
    first nine walks always consume three uniforms regardless of active flag.
    Their interval schedule is independent of the physical process state.
    """
    next_walk = [.1] * 12
    last_walk = [0.] * 12
    # Only walk 3 contributes to the retained D-feed-temperature diagnostic.
    a, b, c, d = 45., 0., 0., 0.
    count = 0
    t = 0.
    dt = 1. / 3600.
    next_gas, next_product = .1, .25
    held = np.full(41, -1, dtype=np.int64)
    input_events = np.empty((960, 41), dtype=np.int64)
    native_events = np.empty_like(input_events)
    record_counts = np.empty(960, dtype=np.int64)
    temperatures = np.empty(960, dtype=np.float64)
    counts_every_step = np.empty(172800, dtype=np.int64)
    record_times = np.empty(960, dtype=np.float64)
    offsets22 = 12 * np.arange(22, dtype=np.int64)
    offsets14 = 12 * np.arange(14, dtype=np.int64)
    offsets5 = 12 * np.arange(5, dtype=np.int64)
    next_event = .1
    for call in range(1, 172801):
        # Exactly the source's one 'if' per walk per model evaluation.
        if t >= next_event:
            for i in range(12):
                if t >= next_walk[i]:
                    draws = uniform_words(key, count, 3 if i < 9 else 1)
                    count += draws.size
                    h = HSPAN[i] * (2. * float(draws[0]) - 1.) + HZERO[i]
                    old_next = next_walk[i]
                    if i == 2:
                        hwlk = old_next - last_walk[i]
                        swlk = a + hwlk * (b + hwlk * (c + hwlk * d))
                        spwlk = b + hwlk * (2. * c + 3. * hwlk * d)
                        idv = int(fault == 9 and t >= 8.)
                        s1 = 10. * (2. * float(draws[1]) - 1.) * idv + 45.
                        s1p = 0. * (2. * float(draws[2]) - 1.) * idv
                        a, b = swlk, spwlk
                        c = (3. * (s1 - swlk) - h * (s1p + 2. * spwlk)) / h**2
                        d = (2. * (swlk - s1) + h * (s1p + spwlk)) / h**3
                    last_walk[i] = old_next
                    next_walk[i] = old_next + h
            next_event = min(next_walk)
        # Time zero initialization/evaluation adds no continuous noise.
        if t > 0.:
            held[:22] = count + offsets22
            count += 22 * 12
        if t >= next_gas:
            held[22:36] = count + offsets14
            count += 14 * 12
            next_gas += .1
        if t >= next_product:
            held[36:41] = count + offsets5
            count += 5 * 12
            next_product += .25
        counts_every_step[call - 1] = count
        # Cached input at fast-control step 177 came from evaluation 177.
        if call % 180 == 177:
            input_events[call // 180] = held
        if call % 180 == 0:
            row = call // 180 - 1
            native_events[row] = held
            record_counts[row] = count
            hh = t - last_walk[2]
            temperatures[row] = a + hh * (b + hh * (c + hh * d))
            record_times[row] = (t + dt) * 3600.
        # The original horizon repeatedly adds dt, rather than multiplying it.
        t += dt
    return input_events, native_events, record_counts, temperatures, counts_every_step, record_times


def _fetch_events(key: int, events: np.ndarray) -> np.ndarray:
    values = np.zeros(events.shape + (12,), dtype=np.float64)
    unique = np.unique(events[events >= 0])
    event_values = {int(start): uniform_words(key, int(start), 12) for start in unique}
    flat_values = values.reshape(-1, 12)
    for i, start in enumerate(events.reshape(-1)):
        if start >= 0:
            flat_values[i] = event_values[int(start)]
    return values


def recover(arrays: Mapping[str, Any], metadata: Mapping[str, Any], *,
            verify_full_rng_trace: bool = False) -> dict[str, Any]:
    """Recover frozen controller-input noise and verify the source schedule.

    Return arrays use shape (960,41[,12]); event id is the zero-based first
    raw64-word index of the original noise event, -1 before first analyzer
    sample. The held-event identity preserves analyzer sample-and-hold exactly.
    Set verify_full_rng_trace=True for an additional all-step source-trace check.
    """
    started = time.perf_counter()
    key, fault, _ = _metadata(metadata)
    native = _array(arrays, "native_values", (960, 52))
    cached = _array(arrays, "fast_control_input_measurements", (960, 41))
    phase = _array(arrays, "phase", (960, 4))
    expected_phase = np.column_stack((np.arange(180, 172801, 180),
                                     np.arange(177, 172798, 180),
                                     np.arange(177, 172798, 180), np.full(960, 3)))
    if not np.array_equal(phase, expected_phase):
        raise ValueError("Unexpected controller-input phase")
    counts = _array(arrays, "diagnostic_rng_draw_count", (960,))
    input_events, native_events, replay_counts, temp, all_counts, times = _replay_schedule(key, fault)
    if not np.array_equal(replay_counts, counts):
        bad = np.flatnonzero(replay_counts != counts)
        raise ValueError(f"RNG count mismatch at record {int(bad[0])+1}: "
                         f"replayed={replay_counts[bad[0]]}, saved={counts[bad[0]]}")
    if int(replay_counts[-1]) != metadata["rng_draw_count"]:
        raise ValueError("Final RNG count differs")
    saved_temp = _array(arrays, "diagnostic_D_feed_temperature", (960,))
    temp_error = float(np.max(np.abs(saved_temp - temp)))
    if not np.allclose(saved_temp, temp, rtol=0., atol=1e-12):
        raise ValueError("Reconstructed D-feed temperature differs")
    saved_times = _array(arrays, "record_time_seconds", (960,))
    if not np.array_equal(times, saved_times):
        raise ValueError("Reconstructed sampling time differs")
    flags = _array(arrays, "diagnostic_IDV", (960, 20))
    expected_flags = np.zeros((960, 20), dtype=flags.dtype)
    if fault == 9:
        expected_flags[160:, 8] = 1
    if not np.array_equal(flags, expected_flags):
        raise ValueError("Fault flags differ from the supported schedule")
    last_u32 = int(uniform_words(key, int(counts[-1]) - 1, 1)[0] * DENOMINATOR)
    if last_u32 != metadata["rng_state"]["storage"]["last_consumed_u32"]:
        raise ValueError("Final uniform differs")
    full_trace = None
    if verify_full_rng_trace:
        logical = {"algorithm": ALGORITHM, "key_hex": f"0x{key:032x}", "next_raw64_word_index": 0}
        logical_digest = hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        trace = hashlib.sha256(logical_digest.encode())
        for count in all_counts:
            count = int(count)
            last = -1 if count == 0 else int(uniform_words(key, count - 1, 1)[0] * DENOMINATOR)
            trace.update(count.to_bytes(8, "little"))
            trace.update(last.to_bytes(8, "little", signed=True))
        full_trace = trace.hexdigest()
        if full_trace != metadata["rng_consumption_trace_sha256"]:
            raise ValueError("Full-step RNG consumption trace differs")
    original = _fetch_events(key, input_events)
    mask = input_events >= 0
    noise = (original.sum(axis=-1) - 6.) * NOISE_SD
    noise[~mask] = 0.
    noiseless = cached - noise
    diagnostics = {
        "status": "PASS_EXACT_RNG_SCHEDULE_RECOVERY",
        "method": "replay_random_schedule_without_physical_simulation",
        "run_id": metadata.get("run_id"), "key_hex": f"0x{key:032x}",
        "record_count_checks": 960, "temperature_max_abs_error": temp_error,
        "sample_times_bitwise_equal": True, "final_draw_count": int(counts[-1]),
        "final_uniform_u32": last_u32, "full_rng_trace_sha256": full_trace,
        "uniform_event_count": int(np.unique(input_events[mask]).size),
        "noise_transform": "sum_12_original_uniforms_minus_6_times_xns",
        "zero_noise_cells_before_first_composition_sample": int((~mask).sum()),
        "wall_seconds": time.perf_counter() - started,
    }
    return {"original_uniforms": original, "has_noise": mask,
            "noise_event_id": input_events, "input_noise": noise,
            "input_noiseless": noiseless, "native_noise_event_id": native_events,
            "noise_sd": NOISE_SD.copy(), "diagnostics": diagnostics}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_directory", type=Path)
    p.add_argument("--full-trace", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    meta = json.loads((args.run_directory / "RUN_INFO.json").read_text())
    with np.load(args.run_directory / "RAW.npz", allow_pickle=False) as source:
        result = recover(source, meta, verify_full_rng_trace=args.full_trace)
    report = json.dumps(result["diagnostics"], ensure_ascii=False, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
