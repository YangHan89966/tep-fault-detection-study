"""Passive same-trajectory recording probe. Original model/horizon are never edited.

The optional observer caches input vectors already provided to fast control and
reads state buffers after the one original evaluate call. No auxiliary evaluation
or RNG call is permitted. Trace-only mode is the reference for an observed run.
"""
from __future__ import annotations
import hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> list[dict]:
    records=json.loads((ROOT/'source_notes/ORIGINAL_SOURCE_IDENTITIES.json').read_text())
    for r in records:
        p=ROOT/r['destination']
        if p.is_symlink() or not p.is_file() or p.stat().st_size != r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError('Source identity mismatch: '+r['destination'])
    return records


def load_backend():
    verify_sources()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/'vendor/src'));sys.path.insert(0,str(ROOT/'vendor'))
    os.environ['TEP_BACKEND']='python'
    from tep.simulator import TEPSimulator,ControlMode
    import tep
    from horizon import simulate_registered_horizon
    if tep.__version__!='2.5.0' or Path(tep.__file__).resolve() != ROOT/'vendor/src/tep/__init__.py':
        raise RuntimeError('Wrong model imported')
    return TEPSimulator,ControlMode,simulate_registered_horizon


def hash_array(h, a):
    a=np.ascontiguousarray(a,dtype='<f8')
    h.update(a.tobytes())


def plain_state(value: Any, seen=None, path='$'):
    """Deterministic numerical state; aliases/cycles use structural paths."""
    if seen is None:seen={}
    if isinstance(value,np.ndarray):
        return {'dtype':value.dtype.str,'shape':list(value.shape),'hash':hashlib.sha256(value.tobytes(order='C')).hexdigest()}
    if isinstance(value,(float,np.floating)):return {'float_hex':float(value).hex()}
    if isinstance(value,(int,np.integer)):return int(value)
    if value is None or isinstance(value,(str,bool)):return value
    key=id(value)
    if key in seen:return {'reference':seen[key]}
    seen[key]=path
    if isinstance(value,(list,tuple)):return [plain_state(v,seen,path+'/'+str(i)) for i,v in enumerate(value)]
    if isinstance(value,dict):return {str(k):plain_state(v,seen,path+'/'+str(k)) for k,v in sorted(value.items(),key=lambda kv:str(kv[0])) if not callable(v)}
    if hasattr(value,'__dict__'):
        return {k:plain_state(v,seen,path+'/'+k) for k,v in sorted(vars(value).items()) if not callable(v)}
    raise TypeError(type(value))


def state_digest(sim):
    v={'process':plain_state(sim.process),'controller':plain_state(sim.controller),
       'time_hex':float(sim.time).hex(),'step_count':sim.step_count}
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PassiveRecorder:
    """Wrap calls once, return originals unchanged; trace covers every process step."""
    def __init__(self,sim,record_interval:int,dual:bool):
        self.sim=sim;self.interval=record_interval;self.dual=dual
        self.evaluate_calls=0;self.control_calls=0
        self.trace=hashlib.sha256();self.last_input=None;self.last_mv=None
        self.last_input_evaluation=None;self.last_control_step=None
        self.cached_measurements=[];self.cached_mvs=[];self.native_measurements=[]
        self.phase=[];self.saturation=np.zeros(3,dtype=np.int64)
        self.fast_updates=0;self.pi_max_error=np.zeros(3);self.invariant_min=np.full(3,np.inf);self.invariant_max=np.full(3,-np.inf)
        self.integral=0.;self.integral_at_records=[]
        self.setpoint_min=np.full(3,np.inf);self.setpoint_max=np.full(3,-np.inf)
        self.orig_calculate=sim.controller.calculate;self.orig_evaluate=sim.process.evaluate
        sim.controller.calculate=self.calculate;sim.process.evaluate=self.evaluate

    def calculate(self,xmeas,xmv,time_step):
        # Trace the exact input and exact returned control without changing buffers.
        self.control_calls+=1
        self.trace.update(b'C');self.trace.update(int(time_step).to_bytes(8,'little'))
        hash_array(self.trace,xmeas);hash_array(self.trace,xmv)
        before=None
        if self.dual and time_step%3==0:
            before=[]
            for ci,mi,vi in [(7,11,6),(8,14,7),(11,16,10)]:
                c=getattr(self.sim.controller,'ctrl'+str(ci))
                before.append((c.setpoint,c.gain,c.taui,c.scale,c.err_old,float(xmv[vi]),float(xmeas[mi]),c.output_min,c.output_max))
            self.last_input=np.array(xmeas,copy=True)
            self.last_input_evaluation=self.evaluate_calls
            self.last_control_step=int(time_step)
        out=self.orig_calculate(xmeas,xmv,time_step)
        hash_array(self.trace,out)
        if before is not None:
            self.fast_updates+=1;self.last_mv=np.array(out,copy=True)
            for k,(vi,par) in enumerate(zip((6,7,10),before)):
                sp,gain,taui,scale,old,prev,mn,lo,hi=par
                err=(sp-mn)*scale
                integ=gain*err*(3./3600.)/taui if taui>0 else 0.
                expected=prev+gain*(err-old)+integ
                clipped=np.clip(expected,lo,hi)
                self.pi_max_error[k]=max(self.pi_max_error[k],abs(float(out[vi])-clipped))
                self.saturation[k]+=int(expected<=lo or expected>=hi)
                self.setpoint_min[k]=min(self.setpoint_min[k],sp);self.setpoint_max[k]=max(self.setpoint_max[k],sp)
                if k==2:self.integral+=integ
                q=float(out[vi])+gain*scale*mn-(self.integral if k==2 else 0.)
                self.invariant_min[k]=min(self.invariant_min[k],q);self.invariant_max[k]=max(self.invariant_max[k],q)
        return out

    def evaluate(self,t,yy):
        # Exactly one native call, no extra get_xmeas/evaluate/noise sampling.
        self.trace.update(b'E');hash_array(self.trace,[t]);hash_array(self.trace,yy)
        result=self.orig_evaluate(t,yy)
        self.evaluate_calls+=1
        p=self.sim.process
        hash_array(self.trace,result);hash_array(self.trace,p.state.xmeas);hash_array(self.trace,p.state.xmv)
        hash_array(self.trace,[p._g])
        if self.dual and self.evaluate_calls%self.interval==0:
            if self.last_input is None:raise RuntimeError('Missing fast-control cache')
            if not np.array_equal(p.state.xmv,self.last_mv):raise RuntimeError('Output changed since cached fast control')
            self.cached_measurements.append(self.last_input.copy());self.cached_mvs.append(p.state.xmv.copy())
            self.native_measurements.append(p.state.xmeas.copy())
            self.phase.append((self.evaluate_calls,self.last_input_evaluation,self.last_control_step,self.evaluate_calls-self.last_input_evaluation))
            self.integral_at_records.append(self.integral)
        return result

    def detach(self):
        self.sim.controller.calculate=self.orig_calculate;self.sim.process.evaluate=self.orig_evaluate

    def summary(self):
        return dict(control_calls=self.control_calls,evaluate_calls=self.evaluate_calls,
                    full_step_trace_sha256=self.trace.hexdigest(),fast_updates=self.fast_updates,
                    saturation_counts=self.saturation.tolist(),max_PI_recurrence_abs_error=self.pi_max_error.tolist(),
                    integral_adjusted_invariant_ranges=(self.invariant_max-self.invariant_min).tolist() if self.dual else [],
                    setpoint_min=self.setpoint_min.tolist() if self.dual else [],
                    setpoint_max=self.setpoint_max.tolist() if self.dual else [])


def run_one(seed:int,duration_hours:float,mode:str='dual'):
    if mode not in ('dual','trace','plain'):raise ValueError(mode)
    cls,ctl,horizon=load_backend()
    sim=cls(random_seed=int(seed),control_mode=ctl.CLOSED_LOOP,backend='python');sim.initialize()
    timer=time.perf_counter()
    tap=None if mode=='plain' else PassiveRecorder(sim,180,mode=='dual')
    result=horizon(sim,np,float(duration_hours),0,None,180)
    elapsed=time.perf_counter()-timer
    metadata={k:v for k,v in result.items() if k not in ('time_seconds','measurements','manipulated_vars')}
    metadata.update(seed_u32=seed,duration_hours=duration_hours,mode=mode,wall_seconds=elapsed,
                    final_state_sha256=state_digest(sim),final_rng_value=float(sim.process._g),
                    final_step_count=sim.step_count,python=sys.version,numpy=np.__version__)
    native=np.concatenate([result['measurements'][1:],result['manipulated_vars'][1:,:11]],axis=1)
    arrays={'native_values':native,'native_xmv12':result['manipulated_vars'][1:,11],'native_time_seconds':result['time_seconds']}
    if tap is not None:
        metadata.update(tap.summary())
        if mode=='dual':
            cached=np.asarray(tap.cached_measurements)
            if not np.array_equal(np.asarray(tap.native_measurements),result['measurements'][1:]):raise RuntimeError('Native recorder differs')
            if not np.array_equal(np.asarray(tap.cached_mvs),result['manipulated_vars'][1:]):raise RuntimeError('Cached output differs')
            arrays.update(fast_control_input_measurements=cached,phase=np.asarray(tap.phase,dtype=np.int64),ctrl11_accumulated_integral=np.asarray(tap.integral_at_records))
            if not np.all(arrays['phase'][:,3]==3):raise RuntimeError('Unexpected recording age')
        tap.detach()
    if not all(np.isfinite(a).all() for a in arrays.values()):raise RuntimeError('Non-finite values')
    return arrays,metadata


def compare_runs(a,ma,b,mb):
    checks={k:bool(np.array_equal(a[k],b[k])) for k in ('native_values','native_xmv12','native_time_seconds')}
    for k in ('final_state_sha256','final_rng_value','final_step_count','shutdown_occurred','shutdown_step'):
        checks[k]=ma[k]==mb[k]
    if 'full_step_trace_sha256' in ma and 'full_step_trace_sha256' in mb:
        checks['full_step_trace']=ma['full_step_trace_sha256']==mb['full_step_trace_sha256']
        checks['evaluate_count']=ma['evaluate_calls']==mb['evaluate_calls']
        checks['controller_count']=ma['control_calls']==mb['control_calls']
    if not all(checks.values()):raise RuntimeError('Passivity failure: '+json.dumps(checks))
    return checks


def policies(arrays):
    native=arrays['native_values'];cache=arrays['fast_control_input_measurements']
    out={'native':native}
    for name,ids in [('all_fast_input',list(range(41))),('pair7_8_only',[11,14]),('pair11_only',[16]),('three_pairs_only',[11,14,16])]:
        alt=native.copy();alt[:,ids]=cache[:,ids];out[name]=alt
    return out
