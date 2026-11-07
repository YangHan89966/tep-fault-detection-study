"""Fault-9 extension of the archived passive recording experiment.

Only the horizon's existing fault_id / fault_onset arguments change. Observers
read/copy existing buffers and do not calculate new measurements or RNG values.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
from dual_core import (ROOT, load_backend, verify_sources, PassiveRecorder,
                       state_digest, compare_runs, policies, digest)


class FaultTap(PassiveRecorder):
    def __init__(self, sim, interval: int, dual: bool):
        self.input_temperature = []
        self.disturbance_flags = []
        self.rng_at_records = []
        super().__init__(sim, interval, dual)

    def evaluate(self, t, yy):
        out = super().evaluate(t, yy)
        if self.evaluate_calls % self.interval == 0:
            # Ground-truth buffer: retained ONLY to verify injection, never scored.
            self.input_temperature.append(float(self.sim.process._teproc.tst[0]))
            self.disturbance_flags.append(self.sim.process._idv.copy())
            self.rng_at_records.append(float(self.sim.process._g))
        return out


def run_fault(seed: int, duration_hours: float = 48., mode: str = 'dual',
              fault_id: int = 9, fault_onset_hours: float | None = 8.):
    if mode not in ('dual', 'trace', 'plain'):
        raise ValueError('mode must be dual, trace or plain')
    if fault_id not in (0, 9):
        raise ValueError('This diagnostic permits only normal or Fault 9')
    cls, ctl, horizon = load_backend()
    sim = cls(random_seed=int(seed), control_mode=ctl.CLOSED_LOOP, backend='python')
    sim.initialize()
    activation = []
    original_set = sim.set_disturbance
    def set_once(index, value):
        result = original_set(index, value)
        activation.append({'index_1based':int(index),'value':int(value),
                           'time_hours':float(sim.time),'step_count':int(sim.step_count),
                           'flags_after':sim.process._idv.tolist()})
        return result
    sim.set_disturbance = set_once
    tap = None if mode == 'plain' else FaultTap(sim, 180, mode == 'dual')
    tic = time.perf_counter()
    res = horizon(sim, np, float(duration_hours), fault_id, fault_onset_hours, 180)
    elapsed = time.perf_counter() - tic
    meta = {k:v for k,v in res.items() if k not in ('time_seconds','measurements','manipulated_vars')}
    meta.update(seed_u32=int(seed), fault_id=fault_id, fault_onset_hours=fault_onset_hours,
                activation_events=activation, duration_hours=float(duration_hours), mode=mode,
                wall_seconds=elapsed, final_state_sha256=state_digest(sim),
                final_rng_value=float(sim.process._g), final_step_count=int(sim.step_count),
                python=sys.version, numpy=np.__version__)
    arr = {'native_values':np.concatenate([res['measurements'][1:],res['manipulated_vars'][1:,:11]],axis=1),
           'native_xmv12':res['manipulated_vars'][1:,11], 'native_time_seconds':res['time_seconds']}
    if tap is not None:
        meta.update(tap.summary())
        arr.update(diagnostic_D_feed_temperature=np.asarray(tap.input_temperature),
                   diagnostic_IDV=np.asarray(tap.disturbance_flags,dtype=np.int32),
                   diagnostic_rng_at_records=np.asarray(tap.rng_at_records))
        if mode == 'dual':
            if not np.array_equal(np.asarray(tap.native_measurements),res['measurements'][1:]):
                raise ValueError('Observed native measurement differs')
            if not np.array_equal(np.asarray(tap.cached_mvs),res['manipulated_vars'][1:]):
                raise ValueError('Observed manipulated variable differs')
            arr.update(fast_control_input_measurements=np.asarray(tap.cached_measurements),
                       phase=np.asarray(tap.phase,dtype=np.int64),
                       ctrl11_accumulated_integral=np.asarray(tap.integral_at_records))
            if not np.all(arr['phase'][:,3] == 3):
                raise ValueError('Unexpected recording source age')
        tap.detach()
    sim.set_disturbance = original_set
    if not all(np.isfinite(x).all() for x in arr.values()):
        raise ValueError('Nonfinite saved output')
    expected = 0 if fault_id == 0 or fault_onset_hours is None or duration_hours <= fault_onset_hours else 1
    if len(activation) != expected:
        raise ValueError('Unexpected fault activation count')
    return arr, meta


def compare_fault_runs(a, ma, b, mb):
    checks = compare_runs(a, ma, b, mb)
    checks['activation_events_equal'] = ma['activation_events'] == mb['activation_events']
    for k in ['native_values','native_xmv12','native_time_seconds',
              'diagnostic_D_feed_temperature','diagnostic_IDV','diagnostic_rng_at_records']:
        if k in a and k in b:
            checks['byte_equal_'+k] = (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                      and a[k].tobytes() == b[k].tobytes())
    if not all(checks.values()):
        raise ValueError('Fault passivity failure: '+json.dumps(checks))
    return checks
