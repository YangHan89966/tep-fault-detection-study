"""One v1.8 repeated-learning DEVELOPMENT trajectory; no evaluation or learning entry point."""
from __future__ import annotations
import sys,os,json,traceback,time,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from storage import strict_json,write_json,sha,verify_tree,now


def validate_arrays(result,fault,np):
    native=np.asarray(result['time_seconds'],dtype=np.float64)
    x=np.asarray(result['measurements'],dtype=np.float64)
    mv=np.asarray(result['manipulated_vars'],dtype=np.float64)
    if native.shape!=(961,) or x.shape!=(961,41) or mv.shape!=(961,12):raise ValueError('registered output shape')
    if not (np.isfinite(native).all() and np.isfinite(x).all() and np.isfinite(mv).all()):raise ValueError('nonfinite simulator output')
    error=float(np.max(np.abs(native-np.arange(961,dtype=float)*180)))
    if error>1e-6 or not np.all(np.diff(native)>0):raise ValueError('native time mismatch')
    if fault==0 and result['shutdown_occurred']:raise ValueError('normal shutdown; retain and stop, no replacement seed')
    return np.concatenate((x[1:],mv[1:,:11]),axis=1),mv[1:,11],native,error


def main():
    taskpath=Path(sys.argv[1]);attempt=taskpath.parent
    lock=attempt.joinpath('WORKER_LOCK').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX)
    started=now();clock=time.perf_counter();task=strict_json(taskpath)
    try:
        release=Path(__file__).resolve().parents[1];verify_tree(release,'RELEASE_MANIFEST.json')
        if sha(release/'RELEASE_MANIFEST.json')!=task['release_manifest_sha256']:raise ValueError('worker release mismatch')
        row=task['row'];root=Path(task['root']);freeze=strict_json(root/'FREEZE.json')
        if freeze['seed_manifest_sha256']!=task['seed_manifest_sha256']:raise ValueError('worker seed freeze mismatch')
        seeds=strict_json(root/'SEED_MANIFEST.json')
        actual=next((r for r in seeds if r['canonical_label']==row['canonical_label']),None)
        if actual!=row:raise ValueError('worker row not in frozen manifest')
        # Stage A is deliberately incapable of generating the new shared test panel.
        if row['stage']!='development' or row.get('replication') not in (1,2,3,4,5):
            raise ValueError('STAGE_A_DEVELOPMENT_ONLY: evaluation generation is not available')
        if freeze.get('study_id')!='TEP_V18_RELEARNING_R5_v1.0':
            raise ValueError('wrong study freeze')
        if sha(root/'SEED_MANIFEST.json')!=task['seed_manifest_sha256']:
            raise ValueError('seed manifest bytes changed')
        if row['fault'] not in (0,3,9,15):raise ValueError('fault outside new study')
        import numpy as np
        if sys.version_info[:3]!=(3,12,3) or np.__version__!='2.2.6':raise ValueError('pinned runtime mismatch')
        # Only after release, seed and phase checks; exact bundled source, no pip/install.
        sys.path.insert(0,str(release/'simulator_source/src'))
        os.environ['TEP_BACKEND']='python'
        from tep.simulator import TEPSimulator,ControlMode
        import tep
        if tep.__version__!='2.5.0' or not Path(tep.__file__).resolve().is_relative_to(release/'simulator_source/src'):raise ValueError('wrong simulator import')
        from horizon import simulate_registered_horizon
        sim=TEPSimulator(random_seed=int(row['seed_u32']),control_mode=ControlMode.CLOSED_LOOP,backend='python')
        sim.initialize()
        result=simulate_registered_horizon(sim,np,48.,row['fault'],8. if row['fault'] else None,180)
        # Preserve returned arrays before normal-role/nonfinite/time validation can fail.
        values=np.concatenate((result['measurements'][1:],result['manipulated_vars'][1:,:11]),axis=1)
        np.savez_compressed(attempt/'OUTPUT.npz',values=values,xmv12=result['manipulated_vars'][1:,11],
                            native_time_seconds=result['time_seconds'])
        with (attempt/'OUTPUT.npz').open('rb') as durable_output:
            os.fsync(durable_output.fileno())
        meta={k:v for k,v in result.items() if k not in ('time_seconds','measurements','manipulated_vars')}
        meta.update({'canonical_label':row['canonical_label'],'role':row['role'],'fault':row['fault'],'run':row['run'],
          'seed_u32':row['seed_u32'],'stage':row['stage'],'replication':row['replication'],'study_id':'TEP_V18_RELEARNING_R5_v1.0',
          'started_utc':started,'ended_utc':now(),'wall_seconds':time.perf_counter()-clock,
          'python':sys.version,'numpy':np.__version__,'tep_version':tep.__version__,
          'simulator_file':str(Path(tep.__file__)),'backend':'python','four_rng_streams':False,
          'preonset_sha256':__import__('hashlib').sha256(values[:160].astype('<f8').tobytes()).hexdigest()})
        write_json(attempt/'METADATA.json',meta)
        _,_,_,err=validate_arrays(result,row['fault'],np)
        write_json(attempt/'COMPLETE.json',{'canonical_label':row['canonical_label'],
          'seed_u32':row['seed_u32'],'stage':row['stage'],'status':'COMPUTATION_COMPLETE',
          'output_sha256':sha(attempt/'OUTPUT.npz'),'metadata_sha256':sha(attempt/'METADATA.json'),
          'task_sha256':sha(taskpath),'time_max_error_seconds':err,'ended_utc':now()})
        print('CASE_COMPLETE='+row['canonical_label'],flush=True);return 0
    except BaseException as e:
        write_json(attempt/'FAILED.json',{'status':'STOP_PRESERVE_INPUTS_NO_SEED_SUBSTITUTION',
          'label':task.get('row',{}).get('canonical_label'),'error_type':type(e).__name__,
          'error':str(e),'traceback':traceback.format_exc(),'started_utc':started,'ended_utc':now()})
        traceback.print_exc();return 2
    finally:lock.close()

if __name__=='__main__':raise SystemExit(main())
