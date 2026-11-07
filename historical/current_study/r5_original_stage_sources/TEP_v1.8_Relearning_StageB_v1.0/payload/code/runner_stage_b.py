"""Stage B only: read the completed 4,575 development runs, learn and lock five states.
No simulator, new initializations, new trajectories, test inputs or test evaluation.
Operational wrapper around byte-unchanged Stage A science.py / inherited.py.
"""
from __future__ import annotations
import argparse,csv,fcntl,gc,hashlib,io,os,shutil,signal,sys,time,traceback,uuid,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import science
from storage import (now,sha,canonical,strict_json,write_json,write_csv,regular,
                     safe_child,seal_tree,verify_tree,Ledger)
PAYLOAD=Path(__file__).resolve().parents[1]
FAULTS=(3,9,15)
METHODS=tuple(['mean_pool_'+s for s in science.FAMILIES]+['moment_pool_'+s for s in science.FAMILIES]+['same_mean_signed_axis_absolute','same_mean_signed_axis_squared','Q_CUSUM'])
ROLES={('normal_design',0):40,('normal_anchor',0):32,('normal_inner_calibration',0):59,('normal_outer_calibration',0):259}
ROLES.update({(role,f):n for f in FAULTS for role,n in [('fault_fit',100),('fault_selection',75)]})

class StopRequested(Exception):
    pass

class BLedger(Ledger):
    allowed=Ledger.allowed|{'INPUTS_VERIFIED','MODEL_START','MODEL_LOCKED','INTERRUPTED_RELEARN_SAME_INPUTS','ALL_MODELS_LOCKED','SESSION_PAUSED'}


def require(ok: bool,message: str) -> None:
    if not ok:raise ValueError(message)


def environment(binding: dict) -> dict:
    require(sys.version_info[:3]==(3,12,3) and np.__version__=='2.2.6','Use the existing Python 3.12.3 / NumPy 2.2.6 environment. No automatic installation.')
    require(str(Path(sys.executable).absolute())==binding['runtime'],'Wrong runtime path')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','Numerical thread setting must remain 1: '+key)
    require(not any(m=='tep' or m.startswith('tep.') for m in sys.modules),'Stage B must not import the simulator')
    return {'executable':sys.executable,'python':sys.version,'numpy':np.__version__,'numpy_file':np.__file__,'platform':sys.platform,'numerical_threads':1}


def read_index(path: Path) -> list[dict]:
    regular(path)
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    labels=set();counts=Counter()
    for r in rows:
        for k in ('replication','fault','run','seed_u32'):r[k]=int(r[k])
        label=r['canonical_label'];require(label not in labels,'Duplicate completed trajectory');labels.add(label)
        require(r['stage']=='development' and r['replication'] in range(1,6),'Test or invalid repetition in development input')
        counts[r['replication'],r['role'],r['fault']]+=1
    require(len(rows)==4575,'Development count must be 4,575')
    for rep in range(1,6):require({(role,f):v for (p,role,f),v in counts.items() if p==rep}==ROLES,'Incorrect role counts in repetition '+str(rep))
    return rows


def root_baseline(root: Path,binding: dict) -> tuple[list[dict],dict,dict]:
    require(root.is_dir() and not root.is_symlink(),'Existing Stage A root unavailable; nothing is created or regenerated')
    safe_child(Path('/'),str(root).lstrip('/'))
    for name in ('FREEZE.json','SEED_MANIFEST.json','SEED_EXCLUSION.json','EVENTS.jsonl'):
        regular(root/name);require(sha(root/name)==binding['reference_hashes'][name],'Stage A baseline changed: '+name)
    require(not (root/'BLOCKED.json').exists(),'Stage A has a retained failure; stop for review')
    active=strict_json(root/'ACTIVE_SESSION.json');require(active.get('active') is False,'Stage A active marker is not closed')
    require((root/'attempts').is_dir() and not any((root/'attempts').iterdir()),'Unfinished Stage A attempts remain')
    require(not (root/'data/shared_test').exists(),'Shared tests must not exist before all five model locks')
    regular(root/'release/RELEASE_MANIFEST.json')
    require(sha(root/'release/RELEASE_MANIFEST.json')==binding['stage_a_release_manifest_sha256'],'Stage A scientific release mismatch')
    verify_tree(root/'release','RELEASE_MANIFEST.json')
    for name,digest in binding['unchanged_scientific_source'].items():
        require(sha(root/'release/code'/name)==digest==sha(PAYLOAD/'code'/name),'Scientific function bytes changed: '+name)
    freeze=strict_json(root/'FREEZE.json')
    for name,key in [('PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz','PCA_sha256'),('RESIDUAL_BASIS_NEWSEED_PCA.npy','residual_basis_sha256')]:
        require(sha(PAYLOAD/'assets'/name)==freeze[key],'Fixed PCA asset changed')
    require(sha(PAYLOAD/'reference/EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json')==freeze['experiment_protocol_sha256'],'Fixed scientific protocol changed')
    a_ledger=Ledger(root/'EVENTS.jsonl',established=True)
    counts=Counter(x['event'] for x in a_ledger.records)
    require(counts==Counter({'SETUP':1,'SESSION_START':8,'SESSION_END':8,'ATTEMPT_START':4575,'ATTEMPT_COMMIT':4575}),'Unexpected Stage A events')
    require(a_ledger.records[-1]['data']['status']=='DEVELOPMENT_4575_COMPLETE_NO_LEARNING_OR_TEST','Stage A not at the returned terminal state')
    index=read_index(PAYLOAD/'reference/COMPLETED_INDEX.csv')
    seeds=strict_json(root/'SEED_MANIFEST.json');by={r['canonical_label']:r for r in seeds}
    require(len(seeds)==len(by)==5575,'Seed manifest row identities')
    commits={r['data']['label']:r['data'] for r in a_ledger.records if r['event']=='ATTEMPT_COMMIT'}
    require(len(commits)==4575,'Duplicate Stage A commit')
    for r in index:
        s=by[r['canonical_label']]
        for k in ('stage','replication','role','fault','run','seed_u32','relative'):require(r[k]==s[k],'Input allocation changed: '+k)
        c=commits[r['canonical_label']]
        require(c['relative']==r['relative'] and c['manifest_sha256']==r['files_manifest_sha256'],'Input is not bound to its completion event')
    require({r['canonical_label'] for r in index}=={r['canonical_label'] for r in seeds if r['stage']=='development'},'Missing development identities')
    return index,by,freeze


def validate_leaf(root: Path,r: dict,seed: dict,freeze: dict,read_arrays: bool=True) -> None:
    folder=safe_child(root,r['relative']);require(folder.is_dir(),'Completed trajectory missing; no automatic regeneration')
    for name,key in [('OUTPUT.npz','output_sha256'),('METADATA.json','metadata_sha256'),('COMPLETE.json','complete_sha256'),('FILES.json','files_manifest_sha256')]:
        regular(folder/name);require(sha(folder/name)==r[key],'Returned input hash differs: '+r['relative']+'/'+name)
    verify_tree(folder)
    require(not (folder/'FAILED.json').exists(),'Failed trajectory retained')
    c=strict_json(folder/'COMPLETE.json');m=strict_json(folder/'METADATA.json');task=strict_json(folder/'TASK.json')
    require(c['status']=='COMPUTATION_COMPLETE' and task['row']==seed,'Task identity differs')
    for k in ('canonical_label','seed_u32','stage'):require(c[k]==r[k] and m[k]==r[k],'Completed metadata identity differs')
    for k in ('replication','role','fault','run'):require(m[k]==r[k],'Metadata role differs')
    require(m['study_id']==freeze['study_id'],'Wrong study metadata')
    require(m['backend']=='python' and m['numpy']=='2.2.6' and m['tep_version']=='2.5.0','Original generation implementation differs')
    require(r['fault']!=0 or not m['shutdown_occurred'],'Normal shutdown requires review')
    for k in ('release_manifest_sha256','seed_manifest_sha256'):require(task[k]==freeze[k],'Task is not bound to frozen source')
    require(sha(folder/'TASK.json')==c['task_sha256'] and c['output_sha256']==r['output_sha256'] and c['metadata_sha256']==r['metadata_sha256'],'Completion digest mismatch')
    if read_arrays:
        with np.load(folder/'OUTPUT.npz',allow_pickle=False) as z:
            require(set(z.files)=={'values','xmv12','native_time_seconds'},'Unexpected trajectory fields')
            for key,shape in [('values',(960,52)),('xmv12',(960,)),('native_time_seconds',(961,))]:
                a=z[key];require(a.shape==shape and a.dtype==np.float64 and bool(np.isfinite(a).all()),'Invalid raw array: '+key)
            native=z['native_time_seconds'];err=float(np.abs(native-np.arange(961)*180.).max())
            require(err<=1e-6 and abs(err-c['time_max_error_seconds'])<=1e-15 and bool((np.diff(native)>0).all()),'Native time grid changed')
            require(hashlib.sha256(z['values'][:160].astype('<f8').tobytes()).hexdigest()==m['preonset_sha256']==r['preonset_sha256'],'Preonset raw bytes changed')


def verify_all_inputs(root,index,seeds,freeze,stop) -> dict:
    leaves={safe_child(root,r['relative']) for r in index}
    for p in (root/'data').rglob('*'):
        require(not p.is_symlink(),'Symlink in development data')
        if p.is_file():require(p.parent in leaves,'Unexpected file in data; test files are not permitted')
    clock=time.monotonic()
    for i,r in enumerate(index,1):
        if stop():raise StopRequested('Paused during input verification; no new model started')
        validate_leaf(root,r,seeds[r['canonical_label']],freeze)
        if i%250==0 or i==len(index):print(f'VERIFY RAW INPUTS {i}/4575 | {(time.monotonic()-clock)/60:.1f} min',flush=True)
    return {'utc':now(),'raw_trajectories_verified':4575,'raw_shape':[960,52],'per_replication':[915]*5,'hash_shape_finite_and_time_checks':'PASS','all_inputs_development_only':True,'test_data_present':False,'seconds':time.monotonic()-clock}


def load_role(root,index,rep,role,fault,pca,basis):
    require(rep in range(1,6) and (role,fault) in ROLES,'Only declared development roles may be loaded')
    rows=sorted([r for r in index if r['replication']==rep and r['role']==role and r['fault']==fault],key=lambda r:r['run'])
    require([r['run'] for r in rows]==list(range(1,ROLES[role,fault]+1)),'Run ordering/count error')
    raw=np.empty((len(rows),960,52),dtype=np.float64)
    for i,r in enumerate(rows):
        path=safe_child(root,r['relative'])/'OUTPUT.npz';regular(path)
        require(sha(path)==r['output_sha256'],'Input changed since preflight')
        with np.load(path,allow_pickle=False) as z:raw[i]=z['values']
    return science.project(raw,pca,basis)


def model_check(out: Path,fault: int) -> list[dict]:
    """Recheck selection by integer loss totals and exact ordered calibration operations."""
    info=strict_json(out/'FIT_INFO.json');sel=strict_json(out/'SELECTION.json')
    require(info['fault']==sel['fault']==fault and not info['selection_used'] and not info['outer_used'] and not info['evaluation_used'],'Fit role flags')
    require(not sel['outer_used'] and not sel['evaluation_used'] and set(sel['choices'])==set(METHODS),'Selection role flags/configurations')
    st=science.state(out);des,mean,moment,pars=st;km=len(mean);ka=len(moment);k=km+ka;c=3*k+1
    require(km==info['mean_count'] and ka==info['moment_count'] and 2<=km<=8 and 1<=ka<=8,'Candidate pool counts')
    require(sel['candidate_names']==science.old.names(km,ka),'Candidate index/name mapping')
    for a in list(des.values())+list(pars.values())+[mean,moment]:require(bool(np.isfinite(np.asarray(a)).all()),'Nonfinite model')
    for axes in (mean,moment):require(bool(np.allclose(np.einsum('ad,de,ae->a',axes,des['G'],axes),1,rtol=1e-9,atol=1e-9)),'Axis actual variance normalization')
    inn=science.load_npz(out/'INNER.npz');outer=science.load_npz(out/'OUTER.npz')
    require(inn['anchor_peaks'].shape==(32,c) and inn['inner_peaks'].shape==(59,c),'Inner calibration count')
    anchors=np.quantile(inn['anchor_peaks'],.9,axis=0,method='linear')
    require(np.array_equal(anchors,inn['anchors']) and bool((anchors>0).all()),'Anchor scaling mismatch')
    require(inn['event_loss_samples'].shape==(75,c),'Selection sample count')
    pre=inn['prefault_alarm'];det=inn['detected'];clean=inn['clean_detection'];first=inn['first_delay_samples_penalized'];loss=inn['event_loss_samples']
    require(np.array_equal(clean,((det!=0)&(pre==0)).astype(np.int64)),'Clean-event identity')
    require(np.array_equal(loss,np.where(clean,first,800)) and loss.dtype.kind in 'iu' and bool(((loss>=0)&(loss<=800)).all()),'Integer selection loss identity')
    totals=loss.sum(0,dtype=np.int64);expected={}
    for g,fam in enumerate(science.FAMILIES):
        ids=list(range(g*k,g*k+km));expected['mean_pool_'+fam]=min(ids,key=lambda j:(int(totals[j]),j))
        ids=list(range(g*k+km,(g+1)*k));expected['moment_pool_'+fam]=min(ids,key=lambda j:(int(totals[j]),j))
    j=expected['mean_pool_signed'];expected.update(same_mean_signed_axis_absolute=k+j,same_mean_signed_axis_squared=2*k+j,Q_CUSUM=3*k)
    require(expected==sel['choices'],'Stored choice does not minimize integer selection loss')
    for peaks,a,q,h,rank in [(inn['inner_peaks'],anchors,inn['inner_q'],inn['inner_thresholds'],57),(outer['peaks'],anchors,outer['q'],outer['thresholds'],247)]:
        require(bool(np.isfinite(peaks).all()),'Nonfinite calibration peaks');expectedq=np.sort(peaks/a,axis=0)[rank-1];expectedh=a*expectedq
        require(np.array_equal(q,expectedq) and np.array_equal(h,expectedh),'Control-limit rank or floating operation order mismatch')
    require(outer['peaks'].shape==(259,c) and int(outer['n'])==259 and int(outer['rank'])==247,'Outer calibration definition')
    return [{'fault':fault,'method':m,'candidate_index':expected[m],'candidate_name':sel['candidate_names'][expected[m]],'k':0.05,'threshold':float(outer['thresholds'][expected[m]]),'chart':'upper' if m=='Q_CUSUM' else 'two-sided','selection_total_loss_samples':int(totals[expected[m]]),'selection_runs':75,'outer_runs':259,'outer_rank':247} for m in METHODS]


def check_replication(path: Path,rep: int,source_binding: dict) -> list[dict]:
    verify_tree(path);record=strict_json(path/'REPLICATION_LOCK.json')
    require(record['replication']==rep and record['binding']==source_binding and record['configuration_count']==27,'Repetition lock source binding')
    records=[]
    for f in FAULTS:records.extend(model_check(path/f'F{f:02d}',f))
    require(record['configurations']==records,'Repetition locked values mismatch')
    return records


def process_replication(root,index,rep,destination,source_binding,session,ledger,pca,basis,stop) -> None:
    dest=destination/'models'/f'rep{rep:02d}'
    if dest.exists():check_replication(dest,rep,source_binding);print(f'REUSE LOCKED REP {rep}/5 | 27 configurations unchanged',flush=True);return
    if stop():raise StopRequested('Paused before the next repetition')
    attempt=destination/'attempts'/f'rep{rep:02d}_{session}';attempt.mkdir()
    write_json(attempt/'ATTEMPT.json',{'replication':rep,'binding':source_binding,'started_utc':now(),'session':session})
    ledger.add('MODEL_START',replication=rep,attempt=attempt.name,session=session)
    print(f'LEARN REP {rep}/5 | normal design + independent fit/selection/calibration roles',flush=True)
    clock=time.monotonic();normal={}
    try:
        for role in ('normal_design','normal_anchor','normal_inner_calibration'):
            print(f'  LOAD {role}',flush=True);normal[role]=load_role(root,index,rep,role,0,pca,basis)
        for fault in FAULTS:
            folder=attempt/f'F{fault:02d}'
            print(f'  REP {rep}/5 F{fault:02d} FIT DIRECTIONS',flush=True)
            fit=load_role(root,index,rep,'fault_fit',fault,pca,basis)
            science.fit_one(normal['normal_design'],fit,fault,folder);del fit;gc.collect()
            print(f'  REP {rep}/5 F{fault:02d} SELECT using 75 development runs; no outer/test data',flush=True)
            fs=load_role(root,index,rep,'fault_selection',fault,pca,basis)
            science.select_one(normal['normal_anchor'],normal['normal_inner_calibration'],fs,fault,folder)
            check=science.serial_scalar_check(fs,folder,limit_runs=2)
            write_json(folder/'SCALAR_RECURRENCE_CHECK.json',{**check,'source_role':'fault_selection','only_dense_branches':True,'no_test_data':True})
            del fs;gc.collect()
            write_json(folder/'CHOICE_BEFORE_OUTER.json',{'utc':now(),'selection_sha256':sha(folder/'SELECTION.json'),'outer_data_read_so_far':False,'test_data_read':False})
        # Every fault's selection is already fixed before this repetition's outer values are loaded.
        print(f'  REP {rep}/5 LOAD 259 OUTER CALIBRATION RUNS (choices already written)',flush=True)
        outer=load_role(root,index,rep,'normal_outer_calibration',0,pca,basis)
        configs=[]
        for fault in FAULTS:
            folder=attempt/f'F{fault:02d}';before=strict_json(folder/'CHOICE_BEFORE_OUTER.json')
            science.calibrate_one(outer,folder,rank=247)
            require(sha(folder/'SELECTION.json')==before['selection_sha256'],'Selection changed after outer calibration')
            configs.extend(model_check(folder,fault))
            print(f'  REP {rep}/5 F{fault:02d} CALIBRATED AND CHECKED | 9 configurations',flush=True)
        del outer;normal.clear();gc.collect()
        require(len(configs)==27,'Wrong configuration count')
        write_json(attempt/'REPLICATION_LOCK.json',{'replication':rep,'configuration_count':27,'configurations':configs,'binding':source_binding,'locked_utc':now(),'elapsed_seconds':time.monotonic()-clock,'PCA_retrained':False,'test_generated':False,'test_scored':False,'scope':'Finite independent downstream learning/calibration repetition; not a performance result.'})
        for saved in attempt.rglob('*'):
            if saved.is_file():
                with saved.open('rb') as fh:os.fsync(fh.fileno())
        seal_tree(attempt);check_replication(attempt,rep,source_binding)
        require(not dest.exists(),'Repetition destination already exists; no overwrite')
        os.rename(attempt,dest)
        ledger.add('MODEL_LOCKED',replication=rep,manifest_sha256=sha(dest/'FILES.json'),configurations=27,session=session)
        print(f'LOCKED REP {rep}/5 | 27/27 configurations | {(time.monotonic()-clock)/60:.1f} min',flush=True)
    except BaseException as exc:
        if attempt.exists() and not (attempt/'FILES.json').exists():
            write_json(attempt/'FAILED.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'no_automatic_retry':True,'retain_same_inputs':True})
        raise


def recover_attempts(work: Path,binding: dict,ledger: BLedger) -> None:
    for attempt in sorted((work/'attempts').iterdir()):
        require(attempt.is_dir() and not attempt.is_symlink(),'Unexpected learning attempt path')
        require(not (attempt/'FAILED.json').exists(),'Prior numerical/software failure retained. Return report; do not delete or automatically rerun.')
        if (attempt/'FILES.json').exists():
            rec=strict_json(attempt/'REPLICATION_LOCK.json');rep=rec['replication'];check_replication(attempt,rep,binding)
            dest=work/'models'/f'rep{rep:02d}';require(not dest.exists(),'Conflicting learned-state destination')
            os.rename(attempt,dest);ledger.add('MODEL_LOCKED',replication=rep,manifest_sha256=sha(dest/'FILES.json'),configurations=27,recovered=True)
        else:
            target=work/'archive'/attempt.name;require(not target.exists(),'Interrupted archive conflict')
            os.rename(attempt,target)
            ledger.add('INTERRUPTED_RELEARN_SAME_INPUTS',archived=target.name,reason='Unsealed attempt after lost process; repeat same deterministic procedure and inputs, never replace a locked state')


def finalize(work,source_binding,ledger):
    rows=[];reps=[]
    require({p.name for p in (work/'models').iterdir()}=={f'rep{r:02d}' for r in range(1,6)},'Five repetitions are required for the global lock')
    for rep in range(1,6):
        path=work/'models'/f'rep{rep:02d}';configs=check_replication(path,rep,source_binding)
        reps.append({'replication':rep,'path':f'models/rep{rep:02d}','manifest_sha256':sha(path/'FILES.json'),'replication_lock_sha256':sha(path/'REPLICATION_LOCK.json'),'configurations':27})
        rows.extend({'replication':rep,**r} for r in configs)
    require(len(rows)==135,'Need 135 configurations')
    p=work/'ALL_MODELS_LOCK.json'
    if p.exists():
        lock=strict_json(p);require(lock['binding']==source_binding and lock['replications']==reps and lock['configuration_count']==135,'Existing all-model lock differs')
        require((work/'ALL_CONFIGURATIONS.csv').read_bytes()==__import__('storage').csv_bytes(rows),'Existing configuration table differs')
    else:
        csvpath=work/'ALL_CONFIGURATIONS.csv'
        if csvpath.exists():require(csvpath.read_bytes()==__import__('storage').csv_bytes(rows),'Unfinished configuration table differs')
        else:write_csv(csvpath,rows)
        lock={'schema':'TEP_V18_ALL_MODELS_PRETEST_LOCK_v1.0','binding':source_binding,'locked_utc':now(),'replications':reps,'replication_count':5,'configuration_count':135,'configuration_table_sha256':sha(csvpath),'test_trajectories_generated':0,'test_trajectories_scored':0,'PCA_retrained':False,'performance_conclusions_available':False,'next_phase':'A distinct later tool may generate the fixed 1,000 shared tests after validating this lock. This tool does not generate or score tests.'}
        write_json(p,lock)
    prior=[r for r in ledger.records if r['event']=='ALL_MODELS_LOCKED']
    if prior:require(len(prior)==1 and prior[0]['data']['lock_sha256']==sha(p),'Global lock ledger differs')
    else:ledger.add('ALL_MODELS_LOCKED',lock_sha256=sha(p),configurations=135,test_generated=0,test_scored=0)
    return lock


def request_stop(root,work):
    regular(work/'ACTIVE_SESSION.json');rec=strict_json(work/'ACTIVE_SESSION.json')
    require(rec.get('active') is True,'No active Stage B session')
    sid=rec['session'];require(bool(sid) and set(sid)<=set('0123456789abcdefTZ_'),'Invalid Stage B session identifier')
    p=work/'stop_requests'/(sid+'.json')
    if not p.exists():write_json(p,{'session':sid,'utc':now(),'stop_after_current_replication':True})
    print('STOP REQUESTED: finish the current repetition, preserve all locked states, then return a report.',flush=True)


def export_report(work,output,session,status,error,source_binding):
    output.mkdir(parents=True,exist_ok=True);dest=output/('TEP_V18_B_'+session+'_RETURN');dest.mkdir(exist_ok=False)
    write_json(dest/'SESSION_RESULT.json',{'session':session,'utc':now(),'status':status,'error':error,'study_root':source_binding.get('study_root'),'scope':'Stage B downstream learning and calibration only','test_generation':False,'test_evaluation':False,'new_seed_allocation':False,'simulator_import':False,'original_stage_a_data_modified':False,'raw_arrays_exported':False})
    shutil.copyfile(PAYLOAD/'BINDING.json',dest/'TOOL_BINDING.json');shutil.copyfile(PAYLOAD/'MANIFEST.json',dest/'TOOL_MANIFEST.json')
    for name in ('BINDING.json','EVENTS.jsonl','INPUT_VERIFICATION.json','ACTIVE_SESSION.json','ALL_MODELS_LOCK.json','ALL_CONFIGURATIONS.csv','SESSION_ERROR.json'):
        p=work/name
        if p.is_file() and not p.is_symlink():shutil.copyfile(p,dest/name)
    if (work/'models').is_dir():
        for p in sorted((work/'models').iterdir()):
            require(p.is_dir() and not p.is_symlink(),'Unexpected model folder during export');verify_tree(p);shutil.copytree(p,dest/'models'/p.name)
    for area in ('attempts','archive'):
        if (work/area).is_dir():
            for p in sorted((work/area).rglob('*.json')):
                require(not p.is_symlink(),'Attempt report symlink')
                q=dest/'partial_evidence'/area/p.relative_to(work/area);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    seal_tree(dest);verify_tree(dest)
    archive=output/(dest.name+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Return archive CRC failure')
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('\nUPLOAD_THIS='+str(archive),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['LearnLock','Stop'],default='LearnLock');ap.add_argument('--output-parent',type=Path,required=True);args=ap.parse_args()
    binding=strict_json(PAYLOAD/'BINDING.json');root=Path(binding['root']);work=root/binding['work_subdirectory']
    if args.mode=='Stop':request_stop(root,work);return 0
    session=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    status='FAILED';error=None;lease=None;acquired=False;ledger=None;source_binding={};want_stop=[False];handlers={};export_failed=False
    try:
        env=environment(binding);verify_tree(PAYLOAD,'MANIFEST.json')
        require(root.is_dir() and not root.is_symlink(),'Completed Stage A root not found')
        safe_child(Path('/'),str(root).lstrip('/'))
        lockpath=root.with_name(root.name+'.operation.lock');regular(lockpath);lease=lockpath.open('a+b')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another generation/learning process is active. Wait; do not run two windows.')
        acquired=True
        index,seeds,freeze=root_baseline(root,binding)
        source_binding={'study_root':str(root),'tool_id':binding['tool_id'],'tool_manifest_sha256':sha(PAYLOAD/'MANIFEST.json'),'stage_a_return_sha256':binding['stage_a_return_sha256'],'stage_a_freeze_sha256':sha(root/'FREEZE.json'),'stage_a_event_ledger_sha256':sha(root/'EVENTS.jsonl'),'completed_index_sha256':binding['reference_hashes']['COMPLETED_INDEX.csv'],'seed_manifest_sha256':freeze['seed_manifest_sha256'],'experiment_protocol_sha256':freeze['experiment_protocol_sha256'],'PCA_sha256':freeze['PCA_sha256'],'residual_basis_sha256':freeze['residual_basis_sha256'],'scientific_sources':binding['unchanged_scientific_source'],'environment':env}
        require(not work.is_symlink(),'Learning destination must not be a symlink')
        if not work.exists():
            require(shutil.disk_usage(root).free>=binding['minimum_free_disk_bytes'],'Less than 1 GiB free for learning outputs')
            work.mkdir()
        for d in ('models','attempts','archive','stop_requests'):
            require(not (work/d).is_symlink(),'Learning subdirectory symlink');(work/d).mkdir(exist_ok=True)
        if (work/'BINDING.json').exists():require(strict_json(work/'BINDING.json')==source_binding,'Existing Stage B source/input binding changed; no overwrite')
        else:write_json(work/'BINDING.json',source_binding)
        ledger=BLedger(work/'EVENTS.jsonl');ledger.add('SESSION_START',session=session,stage='downstream_learning_only',source_binding=source_binding)
        write_json(work/'ACTIVE_SESSION.json',{'session':session,'active':True,'started_utc':now(),'stage':'B'},replace=True)
        def on_signal(signum,frame):
            want_stop[0]=True;print('PAUSE REQUESTED: completing the current replication before stopping.',flush=True)
        for s in (signal.SIGINT,signal.SIGTERM):handlers[s]=signal.signal(s,on_signal)
        def stopped():return want_stop[0] or (work/'stop_requests'/(session+'.json')).exists()
        check=verify_all_inputs(root,index,seeds,freeze,stopped);write_json(work/'INPUT_VERIFICATION.json',{**check,'session':session,'binding':source_binding},replace=True);ledger.add('INPUTS_VERIFIED',count=4575,session=session)
        recover_attempts(work,source_binding,ledger)
        pca=science.load_npz(PAYLOAD/'assets/PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz');basis=np.load(PAYLOAD/'assets/RESIDUAL_BASIS_NEWSEED_PCA.npy',allow_pickle=False)
        for rep in range(1,6):
            process_replication(root,index,rep,work,source_binding,session,ledger,pca,basis,stopped)
            if stopped() and rep<5:raise StopRequested('Finished current replication; later repetitions remain')
        # Re-read all raw input identities before issuing the all-five pretest lock.
        root_baseline(root,binding)
        print('FINAL INPUT RECHECK before the all-five model lock',flush=True)
        verify_all_inputs(root,index,seeds,freeze,lambda:False)
        finalize(work,source_binding,ledger)
        require(not any(m=='tep' or m.startswith('tep.') for m in sys.modules),'Unexpected simulator import')
        status='ALL_135_CONFIGURATIONS_LOCKED_NO_TEST_GENERATED'
        ledger.add('SESSION_END',session=session,status=status,test_generated=0,test_scored=0)
    except StopRequested as exc:
        status='LEARNING_PAUSED_RESUMABLE';print(str(exc),flush=True)
        if ledger:ledger.add('SESSION_PAUSED',session=session,status=status)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
        if ledger:
            ledger.add('SESSION_FAILURE',session=session,error=error)
            write_json(work/'SESSION_ERROR.json',{'session':session,'utc':now(),'error':error},replace=True)
    finally:
        if ledger:write_json(work/'ACTIVE_SESSION.json',{'session':session,'active':False,'ended_utc':now(),'status':status},replace=True)
        print('\nTEP_V18_STAGE_B_STATUS='+status,flush=True)
        try:export_report(work if ledger else Path('/nonexistent_TEP_V18_B_no_owned_snapshot'),args.output_parent,session,status,error,source_binding)
        except BaseException:
            export_failed=True;traceback.print_exc()
            print('RETURN_EXPORT_FAILED: model files are preserved. Return the Windows log.',flush=True)
        for s,h in handlers.items():signal.signal(s,h)
        if lease:lease.close()
    return 0 if status!='FAILED' and not export_failed else 2

if __name__=='__main__':raise SystemExit(main())
