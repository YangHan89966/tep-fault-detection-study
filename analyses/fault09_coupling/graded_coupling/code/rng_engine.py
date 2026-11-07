"""Isolated random-source adapter for the pinned first-paper Python TEP engine.

The recovered physical model, controller, registered 1-second horizon and all
noise transformations are unchanged. Only an individual process TESUB7 callable
is replaced before initialize(). One physical run has one keyed uniform stream;
both recording conventions are passive views of the same trajectory.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

UPSTREAM_COMMIT = "9a6c8e5fcef4a2850778704e7793c87b0a187005"
SOURCE_MANIFEST_SHA256 = "db221bfa0f40d75724ea4b37622f130d4cc55562f65f94889efe293d0f8dad62"
ADAPTER_VERSION = "1.0.1-recovered"
U32_DENOMINATOR = 4294967296.0
DEFAULT_BUFFER_SIZE = 65536
_LOADED: dict[str, Any] = {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json_digest(value: Any) -> str:
    return _sha(json.dumps(_jsonable(value), sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode())


def parse_key(value: int | str) -> int:
    key = int(value, 16 if value.lower().startswith("0x") else 10) if isinstance(value, str) else int(value)
    if not 0 <= key < 2**128:
        raise ValueError("Philox key must be an unsigned 128-bit integer")
    return key


class PhiloxUniform:
    """Use the high 32 bits of one raw64 word per original uniform request."""
    algorithm = "numpy.Philox4x64_high32_over_2pow32_v1"

    def __init__(self, key: int | str, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self.key = parse_key(key)
        self.buffer_size = int(buffer_size)
        if self.buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.bit_generator = np.random.Philox(key=self.key, counter=0)
        self.buffer = np.empty(0, dtype=np.float64)
        self.position = 0
        self.generated_count = 0
        self.draw_count = 0
        self.last_uniform: float | None = None

    @property
    def last_u32(self):
        return None if self.last_uniform is None else int(self.last_uniform * U32_DENOMINATOR)

    def __call__(self, sign: int) -> float:
        if self.position == self.buffer.size:
            raw = self.bit_generator.random_raw(self.buffer_size)
            self.buffer = (raw >> np.uint64(32)).astype(np.float64) / U32_DENOMINATOR
            self.position = 0
            self.generated_count += self.buffer_size
        u = float(self.buffer[self.position])
        self.position += 1
        self.draw_count += 1
        self.last_uniform = u
        return u if sign >= 0 else 2.0 * u - 1.0

    def snapshot(self) -> dict[str, Any]:
        """Read-only audit state; logical consumed count differs from buffering."""
        logical = {"algorithm": self.algorithm, "key_hex": f"0x{self.key:032x}",
                   "next_raw64_word_index": self.draw_count}
        storage = {"bit_generator_state": _jsonable(self.bit_generator.state),
                   "generated_raw64_words": self.generated_count,
                   "buffer_size": self.buffer_size, "buffer_length": int(self.buffer.size),
                   "buffer_position": self.position,
                   "buffer_f64le_sha256": _sha(self.buffer.astype("<f8", copy=False).tobytes()),
                   "last_consumed_u32": self.last_u32}
        return {"logical": logical, "logical_state_sha256": _json_digest(logical),
                "storage": storage, "storage_state_sha256": _json_digest(storage)}


class LegacyUniform:
    """Engineering-only original LCG-compatible provider, never a formal stream."""
    algorithm = "legacy_LCG_9228907_mod_2pow32"

    def __init__(self, seed: int, process: Any):
        self.seed = int(seed)
        self.process = process
        self.draw_count = 0
        self.last_u32 = None

    def __call__(self, sign: int) -> float:
        g = (int(self.process._g) * 9228907) % 4294967296
        self.process._g = float(g)
        self.draw_count += 1
        self.last_u32 = g
        u = g / U32_DENOMINATOR
        return u if sign >= 0 else 2.0 * u - 1.0

    def snapshot(self):
        logical = {"algorithm": self.algorithm, "initial_seed": self.seed,
                   "draw_count": self.draw_count, "state_g": int(self.process._g)}
        digest = _json_digest(logical)
        return {"logical": logical, "logical_state_sha256": digest,
                "storage": logical, "storage_state_sha256": digest}


class OriginalLCGAudit(LegacyUniform):
    """Count and observe calls to the untouched bound TESUB7 routine."""
    def __init__(self, seed: int, process: Any):
        super().__init__(seed, process)
        self.original = process._tesub7

    def __call__(self, sign: int) -> float:
        value = self.original(sign)
        self.draw_count += 1
        self.last_u32 = int(self.process._g)
        return value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_engine(source_dir: str | Path):
    root = Path(source_dir).resolve(strict=True)
    cache_key = str(root)
    if cache_key in _LOADED:
        return _LOADED[cache_key]
    manifest_bytes = (root / "source_notes/ORIGINAL_SOURCE_IDENTITIES.json").read_bytes()
    if _sha(manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Pinned source manifest differs")
    records = json.loads(manifest_bytes)
    for record in records:
        path = root / record["destination"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("Pinned source missing or symlinked: " + str(path))
        data = path.read_bytes()
        if len(data) != record["bytes"] or _sha(data) != record["sha256"]:
            raise ValueError("Pinned source bytes differ: " + str(path))
    expected_tep = root / "vendor/src/tep/__init__.py"
    existing = sys.modules.get("tep")
    if existing is not None and Path(existing.__file__).resolve() != expected_tep:
        raise RuntimeError("A different tep package is already imported")
    sys.dont_write_bytecode = True
    for directory in ("code", "vendor", "vendor/src"):
        sys.path.insert(0, str(root / directory))
    os.environ["TEP_BACKEND"] = "python"
    tep = importlib.import_module("tep")
    simulator = importlib.import_module("tep.simulator")
    if tep.__version__ != "2.5.0" or Path(tep.__file__).resolve() != expected_tep:
        raise RuntimeError("Unexpected tep version or import origin")
    tag = _sha(str(root).encode())[:16]
    dual = _load_module("_p1_dual_" + tag, root / "code/dual_core.py")
    horizon = _load_module("_p1_horizon_" + tag, root / "code/horizon.py")
    engine = SimpleNamespace(root=root, dual=dual, horizon=horizon.simulate_registered_horizon,
                             TEPSimulator=simulator.TEPSimulator, ControlMode=simulator.ControlMode,
                             records=records)
    _LOADED[cache_key] = engine
    return engine


def _physical_state_digest(sim, dual):
    process = dual.plain_state(sim.process)
    process.pop("_g", None)
    process.pop("_random_seed", None)
    return _json_digest({"process": process, "controller": dual.plain_state(sim.controller),
                         "time_hex": float(sim.time).hex(), "step_count": int(sim.step_count)})


def _recorder_class(dual):
    class RNGPassiveRecorder(dual.PassiveRecorder):
        def __init__(self, sim, provider, interval, paired):
            self.provider = provider
            self.rng_trace = hashlib.sha256()
            self.rng_trace.update(_json_digest(provider.snapshot()["logical"]).encode())
            self.rng_counts_at_records = []
            self.input_temperature = []
            self.disturbance_flags = []
            super().__init__(sim, interval, paired)

        def evaluate(self, t, yy):
            # Parent invokes original evaluate exactly once; everything else reads.
            out = super().evaluate(t, yy)
            self.rng_trace.update(int(self.provider.draw_count).to_bytes(8, "little"))
            last = self.provider.last_u32
            self.rng_trace.update((-1 if last is None else int(last)).to_bytes(8, "little", signed=True))
            if self.evaluate_calls % self.interval == 0:
                self.rng_counts_at_records.append(int(self.provider.draw_count))
                self.input_temperature.append(float(self.sim.process._teproc.tst[0]))
                self.disturbance_flags.append(self.sim.process._idv.copy())
            return out

        def summary(self):
            result = super().summary()
            result["rng_consumption_trace_sha256"] = self.rng_trace.hexdigest()
            return result
    return RNGPassiveRecorder


def run_one(fault_id: int, key: int | str, duration_hours: float = 48.0,
            onset_hours: float | None = 8.0, mode: str = "dual", *,
            source_dir: str | Path | None = None, rng_kind: str = "philox",
            buffer_size: int = DEFAULT_BUFFER_SIZE):
    """Return recorded arrays and metadata; caller chooses and persists formal IDs."""
    if fault_id not in (0, 3, 9, 15) or mode not in ("plain", "trace", "dual"):
        raise ValueError("Invalid fault or mode")
    if rng_kind not in ("philox", "legacy", "original"):
        raise ValueError("Invalid random-source kind")
    key_int = parse_key(key)
    duration_hours = float(duration_hours)
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError("Duration must be positive and finite")
    if fault_id == 0:
        onset_hours = None
    elif onset_hours is None or not math.isfinite(float(onset_hours)) or float(onset_hours) < 0:
        raise ValueError("Fault onset must be finite and nonnegative")
    else:
        onset_hours = float(onset_hours)
    engine = load_engine(source_dir or Path(__file__).resolve().parents[1] / "engine")
    legacy_seed = key_int if rng_kind != "philox" else 1
    if rng_kind != "philox" and not 0 <= legacy_seed < 2**53:
        raise ValueError("Legacy seed must be exactly representable as float64")
    sim = engine.TEPSimulator(random_seed=legacy_seed, control_mode=engine.ControlMode.CLOSED_LOOP,
                              backend="python")
    if rng_kind == "philox":
        provider = PhiloxUniform(key_int, buffer_size)
    elif rng_kind == "legacy":
        provider = LegacyUniform(legacy_seed, sim.process)
    else:
        provider = OriginalLCGAudit(legacy_seed, sim.process)
    def tesub7(_process, sign):
        return provider(sign)
    sim.process._tesub7 = MethodType(tesub7, sim.process)
    started = time.perf_counter()
    sim.initialize()
    initialization_draw_count = int(provider.draw_count)
    activation = []
    original_set = sim.set_disturbance
    def set_once(index, value):
        result = original_set(index, value)
        activation.append({"index_1based": int(index), "value": int(value),
                           "time_hours": float(sim.time), "step_count": int(sim.step_count),
                           "flags_after": sim.process._idv.tolist()})
        return result
    sim.set_disturbance = set_once
    tap = None if mode == "plain" else _recorder_class(engine.dual)(sim, provider, 180, mode == "dual")
    horizon_started = time.perf_counter()
    try:
        result = engine.horizon(sim, np, duration_hours, fault_id, onset_hours, 180)
        wall_seconds = time.perf_counter() - horizon_started
        native = np.concatenate((result["measurements"][1:], result["manipulated_vars"][1:, :11]), axis=1)
        arrays = {"native_values": native, "native_xmv12": result["manipulated_vars"][1:, 11],
                  "native_time_seconds": result["time_seconds"], "record_time_seconds": result["time_seconds"][1:]}
        if tap is not None:
            arrays.update(diagnostic_D_feed_temperature=np.asarray(tap.input_temperature),
                          diagnostic_IDV=np.asarray(tap.disturbance_flags, dtype=np.int32).reshape(-1, 20),
                          diagnostic_rng_draw_count=np.asarray(tap.rng_counts_at_records, dtype=np.int64))
        if mode == "dual":
            cached = np.asarray(tap.cached_measurements, dtype=np.float64).reshape(-1, 41)
            cached_mvs = np.asarray(tap.cached_mvs, dtype=np.float64).reshape(-1, 12)
            observed_native = np.asarray(tap.native_measurements, dtype=np.float64).reshape(-1, 41)
            if not np.array_equal(observed_native, result["measurements"][1:]):
                raise RuntimeError("Native observer and saved record differ")
            if not np.array_equal(cached_mvs, result["manipulated_vars"][1:]):
                raise RuntimeError("Cached and saved manipulated variables differ")
            phase = np.asarray(tap.phase, dtype=np.int64).reshape(-1, 4)
            if phase.size and not np.all(phase[:, 3] == 3):
                raise RuntimeError("Controller-input recording age differs from three steps")
            controller_input = native.copy()
            controller_input[:, :41] = cached
            if not np.array_equal(controller_input[:, 41:], native[:, 41:]):
                raise RuntimeError("Paired views have different valve outputs")
            arrays.update(controller_input_values=controller_input, fast_control_input_measurements=cached,
                          phase=phase, ctrl11_accumulated_integral=np.asarray(tap.integral_at_records))
        expected = int(fault_id != 0 and onset_hours is not None and duration_hours > onset_hours)
        if len(activation) != expected:
            raise RuntimeError("Unexpected fault activation count")
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError("Nonfinite saved array")
        if mode != "plain" and fault_id == 0 and np.any(arrays["diagnostic_IDV"]):
            raise RuntimeError("Normal trajectory contains an active disturbance")
        if mode != "plain" and fault_id != 0 and duration_hours == 48.0 and onset_hours == 8.0:
            flags = np.zeros_like(arrays["diagnostic_IDV"])
            flags[160:, fault_id - 1] = 1
            if not np.array_equal(flags, arrays["diagnostic_IDV"]):
                raise RuntimeError("Fault does not first appear in record 161")
        rows = int(math.ceil(duration_hours / sim.dt)) // 180
        if native.shape != (rows, 52):
            raise RuntimeError("Unexpected output shape")
        if not np.allclose(arrays["record_time_seconds"], np.arange(1, rows + 1) * 180., rtol=0, atol=1e-5):
            raise RuntimeError("Unexpected record time grid")
        metadata = {k: v for k, v in result.items() if k not in ("time_seconds", "measurements", "manipulated_vars")}
        metadata.update(adapter_version=ADAPTER_VERSION, adapter_sha256=_sha(Path(__file__).read_bytes()),
                        upstream_commit=UPSTREAM_COMMIT, source_manifest_sha256=SOURCE_MANIFEST_SHA256,
                        source_directory=str(engine.root), tep_version="2.5.0", backend="python", rng_kind=rng_kind,
                        rng_key_hex=f"0x{key_int:032x}" if rng_kind == "philox" else None,
                        legacy_seed=legacy_seed if rng_kind != "philox" else None,
                        rng_state=provider.snapshot(), rng_draw_count=int(provider.draw_count),
                        initialization_rng_draw_count=initialization_draw_count,
                        physical_state_sha256=_physical_state_digest(sim, engine.dual),
                        archived_state_sha256=engine.dual.state_digest(sim), fault_id=int(fault_id),
                        fault_onset_hours=onset_hours, activation_events=activation,
                        duration_hours=duration_hours, mode=mode, rows=rows, wall_seconds=wall_seconds,
                        total_wall_seconds=time.perf_counter()-started, final_step_count=int(sim.step_count),
                        integration_step_seconds=1, record_interval_seconds=180,
                        python=sys.version, numpy=np.__version__)
        if tap is not None:
            metadata.update(tap.summary())
        return arrays, metadata
    finally:
        if tap is not None:
            tap.detach()
        sim.set_disturbance = original_set


def compare_runs(a, ma, b, mb):
    checks = {}
    for name in sorted(a.keys() & b.keys()):
        checks["array_" + name] = (a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
                                    and a[name].tobytes() == b[name].tobytes())
    for name in ("physical_state_sha256", "final_step_count", "shutdown_occurred", "shutdown_step",
                 "activation_events", "rng_draw_count"):
        checks[name] = ma[name] == mb[name]
    checks["rng_logical_state"] = ma["rng_state"]["logical_state_sha256"] == mb["rng_state"]["logical_state_sha256"]
    checks["rng_storage_state"] = ma["rng_state"]["storage_state_sha256"] == mb["rng_state"]["storage_state_sha256"]
    for name in ("full_step_trace_sha256", "rng_consumption_trace_sha256", "evaluate_calls", "control_calls"):
        if name in ma and name in mb:
            checks[name] = ma[name] == mb[name]
    if not all(checks.values()):
        raise AssertionError("Passivity/regression failure: " + json.dumps(checks, sort_keys=True))
    return checks
