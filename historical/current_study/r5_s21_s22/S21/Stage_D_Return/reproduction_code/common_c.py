"""Stage C test-only identities and pre-test lock checks; no learning/scoring imports."""
from __future__ import annotations
import os,sys,hashlib
from collections import Counter
from pathlib import Path
from storage import strict_json,sha,regular,safe_child,verify_tree,Ledger
STUDY='TEP_V18_RELEARNING_R5_v1.0'
PAYLOAD=Path(__file__).resolve().parents[1]
class CLedger(Ledger):
    allowed={'SETUP','SESSION_START','PRETEST_LOCK_RECHECK','ATTEMPT_START','ATTEMPT_COMMIT','RECOVER_COMMIT','INTERRUPTED_SAME_SEED_RETRY','SESSION_END','SESSION_FAILURE','TEST_PANEL_COMPLETE'}
def require(ok,message):
    if not ok:raise ValueError(message)
def validate_test_row(row):
    require(row.get('stage')=='evaluation' and row.get('replication')==0,'STAGE_C_SHARED_TEST_ONLY: no development or relearning')
    f=row.get('fault');n=row.get('run');require(type(f) is int and f in (0,3,9,15) and type(n) is int and 1<=n<=250,'Invalid shared test role/run')
    role='normal_evaluation' if f==0 else 'fault_evaluation'
    require(row.get('role')==role,'Wrong test role')
    require(row.get('relative')==f'data/shared_test/{role}/F{f:02d}/R{n:04d}','Test destination outside frozen shared-test tree')
    require(row.get('canonical_label')==f'{STUDY}|rep=00|role={role}|fault={f:02d}|run={n:04d}','Test label changed')
    require(type(row.get('seed_u32')) is int and 0<row['seed_u32']<2**32 and row['effective_initial_state']==row['seed_u32'],'Invalid frozen initialization')
def test_rows(root,binding):
    p=root/'SEED_MANIFEST.json';regular(p);require(sha(p)==binding['stage_a_control_hashes']['SEED_MANIFEST.json'],'Frozen seed manifest changed')
    allrows=strict_json(p);require(len(allrows)==5575 and len({r['canonical_label'] for r in allrows})==5575 and len({r['seed_u32'] for r in allrows})==5575,'Frozen batch identities/collisions')
    rows=[r for r in allrows if r['stage']=='evaluation']
    for r in rows:validate_test_row(r)
    require(Counter((r['role'],r['fault']) for r in rows)==Counter({('normal_evaluation',0):250,('fault_evaluation',3):250,('fault_evaluation',9):250,('fault_evaluation',15):250}),'Shared panel must contain 1000 trajectories')
    return sorted(rows,key=lambda r:(r['run'],r['fault']))
def runtime(binding):
    import numpy as np
    require(sys.version_info[:3]==(3,12,3) and np.__version__=='2.2.6','Requires existing Python 3.12.3 / NumPy 2.2.6; no installation')
    require(str(Path(sys.executable).absolute())==binding['runtime'],'Wrong runtime path')
    require(sys.flags.isolated and sys.dont_write_bytecode,'Use provided -I -B launcher')
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(k)=='1','Keep numerical threads=1: '+k)
    return {'python':sys.version,'numpy':np.__version__,'numpy_file':np.__file__,'executable':sys.executable,'numerical_threads':1}
def gate(root:Path,payload:Path=PAYLOAD):
    """Checks only old control/model bytes. Never loads or scores any test array."""
    binding=strict_json(payload/'BINDING.json')
    require(str(root)==binding['root'] and root.is_dir() and not root.is_symlink(),'Existing study root missing/different')
    safe_child(Path('/'),str(root).lstrip('/'))
    for name,digest in binding['stage_a_control_hashes'].items():
        p=root/name;regular(p);require(sha(p)==digest,'Stage A frozen control changed: '+name)
    freeze=strict_json(root/'FREEZE.json');require(freeze['study_id']==STUDY,'Wrong original study')
    regular(root/'release/RELEASE_MANIFEST.json')
    require(sha(root/'release/RELEASE_MANIFEST.json')==binding['stage_a_release_manifest_sha256'],'Original generator release identity changed')
    verify_tree(root/'release','RELEASE_MANIFEST.json')
    # Bundled horizon and all simulator source bytes equal the original Stage A release.
    require(sha(payload/'code/horizon.py')==sha(root/'release/code/horizon.py')==binding['source_hashes']['horizon.py'],'Horizon implementation changed')
    for p in (payload/'simulator_source').rglob('*'):
        require(not p.is_symlink(),'Simulator source symlink')
        if p.is_file():
            old=safe_child(root/'release/simulator_source',p.relative_to(payload/'simulator_source').as_posix());regular(old)
            require(sha(p)==sha(old),'Generator scientific source changed')
    require(not (root/'BLOCKED.json').exists(),'Stage A failure requires review')
    require(strict_json(root/'ACTIVE_SESSION.json').get('active') is False,'Stage A still active')
    require((root/'attempts').is_dir() and not any((root/'attempts').iterdir()),'Unfinished Stage A attempts')
    b=safe_child(root,binding['stage_b_subdirectory']);require(b.is_dir(),'Stage B model lock missing')
    for e in strict_json(payload/'reference/STAGE_B_EXPECTED_FILES.json')['entries']:
        p=safe_child(b,e['path']);regular(p)
        require(p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'Locked Stage B bytes changed: '+e['path'])
    require(not (b/'SESSION_ERROR.json').exists(),'Stage B failure requires review')
    for area in ('attempts','archive'):
        p=b/area;require(p.is_dir() and not p.is_symlink() and not any(p.iterdir()),'Unexpected Stage B partial/archived attempt')
    require(strict_json(b/'ACTIVE_SESSION.json').get('active') is False,'Stage B still active')
    require(sha(b/'ALL_MODELS_LOCK.json')==binding['all_models_lock_sha256'],'All-model lock identity changed')
    lock=strict_json(b/'ALL_MODELS_LOCK.json')
    require(lock['configuration_count']==135 and lock['replication_count']==5 and lock['test_trajectories_generated']==lock['test_trajectories_scored']==0,'Invalid pre-test model lock')
    require(lock['binding']['seed_manifest_sha256']==freeze['seed_manifest_sha256'],'Models and tests belong to different seed freeze')
    require({p.name for p in (b/'models').iterdir()}=={f'rep{r:02d}' for r in range(1,6)},'Need exactly five model directories')
    for r in lock['replications']:
        p=safe_child(b,r['path']);require(r['configurations']==27,'Incomplete repetition');verify_tree(p)
        require(sha(p/'FILES.json')==r['manifest_sha256'] and sha(p/'REPLICATION_LOCK.json')==r['replication_lock_sha256'],'Repetition lock differs')
    require(sha(b/'ALL_CONFIGURATIONS.csv')==lock['configuration_table_sha256'],'Configuration table differs')
    return binding,freeze,lock

def validate_complete(folder,row,c_freeze,read_values=False):
    validate_test_row(row)
    for name in ('TASK.json','METADATA.json','COMPLETE.json'):regular(folder/name)
    require(not (folder/'FAILED.json').exists(),'Failed trajectory cannot be committed')
    c=strict_json(folder/'COMPLETE.json');m=strict_json(folder/'METADATA.json');task=strict_json(folder/'TASK.json')
    require(c['status']=='COMPUTATION_COMPLETE' and task['row']==row,'Test task/completion identity differs')
    for key in ('canonical_label','seed_u32','stage'):
        require(c[key]==m[key]==row[key],'Test completion identity: '+key)
    for key in ('role','fault','run','replication'):require(m[key]==row[key],'Test metadata: '+key)
    require(m['study_id']==STUDY and m['all_models_lock_sha256']==c_freeze['all_models_lock_sha256'],'Test is not bound to the five pre-test locks')
    for key in ('payload_manifest_sha256','seed_manifest_sha256','all_models_lock_sha256'):
        require(task[key]==c_freeze[key],'Test source/pre-test binding differs: '+key)
    require(task['stage_c_freeze_sha256']==c_freeze['_file_sha256'],'Test C freeze differs')
    for name,key in [('OUTPUT.npz','output_sha256'),('METADATA.json','metadata_sha256'),('TASK.json','task_sha256')]:
        regular(folder/name);require(sha(folder/name)==c[key],'Test file digest differs: '+name)
    require(row['fault']!=0 or not m['shutdown_occurred'],'Normal shutdown: retain and stop, never replace seed')
    require(m['backend']=='python' and m['numpy']=='2.2.6' and m['tep_version']=='2.5.0','Generator version differs')
    if read_values:
        import numpy as np
        with np.load(folder/'OUTPUT.npz',allow_pickle=False) as z:
            require(set(z.files)=={'values','xmv12','native_time_seconds'},'Test output keys')
            for key,shape in [('values',(960,52)),('xmv12',(960,)),('native_time_seconds',(961,))]:
                a=z[key];require(a.shape==shape and a.dtype==np.float64 and bool(np.isfinite(a).all()),'Test array shape/finiteness: '+key)
            t=z['native_time_seconds'];err=float(np.max(np.abs(t-np.arange(961)*180.)))
            require(err<=1e-6 and bool((np.diff(t)>0).all()) and abs(err-c['time_max_error_seconds'])<=1e-15,'Test native time grid')
            require(hashlib.sha256(z['values'][:160].astype('<f8').tobytes()).hexdigest()==m['preonset_sha256'],'Test preonset bytes')
    return m

def cfreeze(work):
    p=work/'FREEZE_C.json';regular(p);return {**strict_json(p),'_file_sha256':sha(p)}
