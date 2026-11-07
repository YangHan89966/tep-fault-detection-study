"""Local Stage A: freeze 5,575 allocations, generate ONLY 4,575 development rows.
No fitting, test generation, scoring, installation, or writes to a previous study.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,os,shutil,signal,subprocess,sys,time,traceback,uuid,zipfile
from collections import Counter
from pathlib import Path
from storage import now,sha,strict_json,write_json,write_csv,canonical,regular,safe_child,seal_tree,verify_tree,Ledger
from seed_allocation import STUDY,previous_states,allocate,validate_manifest,ordered_development,DEVELOPMENT_ROLES

RELEASE=Path(__file__).resolve().parents[1]


def runtime() -> dict:
    import numpy as np
    if sys.version_info[:3]!=(3,12,3) or np.__version__!='2.2.6':
        raise ValueError('Requires existing Python 3.12.3 / NumPy 2.2.6. No software is installed automatically.')
    return {'python':sys.version,'numpy':np.__version__,'numpy_file':np.__file__,'executable':sys.executable}


def init_root(root: Path,release: Path,environment: dict) -> None:
    verify_tree(release,'RELEASE_MANIFEST.json')
    protocol=release/'protocol/EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json'
    binding=strict_json(release/'LOCAL_BINDING.json')
    if sha(protocol)!=binding['locked_experiment_protocol_sha256']:raise ValueError('Locked experiment protocol changed')
    excluded,prior=previous_states(release);rows,allocation=allocate(excluded)
    exclusion={**prior,**allocation}
    if root.is_symlink():raise ValueError('Study root must not be a symbolic link')
    if root.exists():
        freeze=strict_json(root/'FREEZE.json')
        if freeze.get('study_id')!=STUDY or freeze.get('release_manifest_sha256')!=sha(release/'RELEASE_MANIFEST.json'):
            raise ValueError('Existing study is a different release. No overwrite or replacement.')
        verify_tree(root/'release','RELEASE_MANIFEST.json')
        if sha(root/'release/RELEASE_MANIFEST.json')!=freeze['release_manifest_sha256']:raise ValueError('Owned release identity differs')
        for name,key in [('SEED_MANIFEST.json','seed_manifest_sha256'),('SEED_EXCLUSION.json','seed_exclusion_sha256')]:
            if sha(root/name)!=freeze[key]:raise ValueError('Frozen '+name+' changed')
        if strict_json(root/'SEED_MANIFEST.json')!=rows or strict_json(root/'SEED_EXCLUSION.json')!=exclusion:
            raise ValueError('Frozen allocation does not reproduce from the pinned exclusions')
        Ledger(root/'EVENTS.jsonl',established=True)
        return
    if not root.parent.is_dir():raise ValueError('Confirmed Linux home is unavailable')
    if shutil.disk_usage(root.parent).free<binding['requires_free_disk_bytes']:raise ValueError('Less than 10 GiB available; no generation started')
    temp=root.with_name(root.name+'.setup_'+uuid.uuid4().hex[:10]);temp.mkdir(exist_ok=False)
    try:
        shutil.copytree(release,temp/'release');verify_tree(temp/'release','RELEASE_MANIFEST.json')
        for d in ('data','attempts','archive','sessions','stop_requests'):(temp/d).mkdir()
        write_json(temp/'SEED_MANIFEST.json',rows);write_json(temp/'SEED_EXCLUSION.json',exclusion)
        freeze={'study_id':STUDY,'created_before_first_process_utc':now(),'planned_total':5575,
          'development_total':4575,'shared_evaluation_total':1000,'replications':5,
          'release_manifest_sha256':sha(release/'RELEASE_MANIFEST.json'),
          'experiment_protocol_sha256':sha(protocol),'seed_manifest_sha256':sha(temp/'SEED_MANIFEST.json'),
          'seed_exclusion_sha256':sha(temp/'SEED_EXCLUSION.json'),
          'PCA_sha256':sha(release/'assets/PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz'),
          'residual_basis_sha256':sha(release/'assets/RESIDUAL_BASIS_NEWSEED_PCA.npy'),
          'environment':environment,'process_runs_at_freeze':0,
          'scope':'Stage A generates development only; all five learned-state locks required before later shared tests.'}
        write_json(temp/'FREEZE.json',freeze);Ledger(temp/'EVENTS.jsonl').add('SETUP',freeze=freeze)
        if root.exists():raise ValueError('Root appeared during setup')
        os.rename(temp,root)
    except BaseException:
        if temp.exists():write_json(temp/'SETUP_FAILED.json',{'utc':now(),'traceback':traceback.format_exc()})
        raise


def validate_complete(folder: Path,row: dict,freeze: dict,read_values: bool=False) -> dict:
    if (folder/'FAILED.json').exists():raise ValueError('Failed run is retained and cannot be committed')
    c=strict_json(folder/'COMPLETE.json');m=strict_json(folder/'METADATA.json');t=strict_json(folder/'TASK.json')
    if c.get('status')!='COMPUTATION_COMPLETE' or t['row']!=row:raise ValueError('Run/task identity')
    for obj in (c,m):
        for key in ('canonical_label','seed_u32','stage'):
            if obj.get(key)!=row[key]:raise ValueError('Completed identity '+key)
    if m.get('study_id')!=STUDY or m.get('replication')!=row['replication']:raise ValueError('Study/replication identity')
    if any(m[k]!=row[k] for k in ('role','fault','run')):raise ValueError('Run role identity')
    if row['stage']!='development' or row['replication'] not in range(1,6):raise ValueError('Unexpected test trajectory')
    for key in ('release_manifest_sha256','seed_manifest_sha256'):
        if t.get(key)!=freeze[key]:raise ValueError('Run source/freeze binding')
    for f,key in [('OUTPUT.npz','output_sha256'),('METADATA.json','metadata_sha256'),('TASK.json','task_sha256')]:
        regular(folder/f)
        if sha(folder/f)!=c[key]:raise ValueError('Completed file digest: '+f)
    if row['fault']==0 and m['shutdown_occurred']:raise ValueError('Normal shutdown; preserve and review')
    if m.get('backend')!='python' or m.get('numpy')!='2.2.6' or m.get('tep_version')!='2.5.0':raise ValueError('Generation backend/version')
    if read_values:
        import numpy as np
        with np.load(folder/'OUTPUT.npz',allow_pickle=False) as z:
            if set(z.files)!={'values','xmv12','native_time_seconds'}:raise ValueError('Output keys')
            for key,shape in [('values',(960,52)),('xmv12',(960,)),('native_time_seconds',(961,))]:
                a=z[key]
                if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Output shape/type/finiteness '+key)
            native=z['native_time_seconds'];err=float(np.max(np.abs(native-np.arange(961)*180.)))
            if err>1e-6 or not np.all(np.diff(native)>0) or abs(err-c['time_max_error_seconds'])>1e-15:raise ValueError('Native time grid')
            if hashlib.sha256(z['values'][:160].astype('<f8').tobytes()).hexdigest()!=m['preonset_sha256']:raise ValueError('Preonset digest')
    return m


def promote(root: Path,attempt: Path,row: dict,ledger: Ledger,recovered: bool=False) -> dict:
    m=validate_complete(attempt,row,strict_json(root/'FREEZE.json'),read_values=True)
    if not (attempt/'FILES.json').exists():seal_tree(attempt)
    verify_tree(attempt)
    dest=safe_child(root,row['relative']);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('Completed destination already exists; no replacement')
    os.rename(attempt,dest)
    ledger.add('RECOVER_COMMIT' if recovered else 'ATTEMPT_COMMIT',label=row['canonical_label'],
      relative=row['relative'],manifest_sha256=sha(dest/'FILES.json'))
    return m


def recover(root: Path,rows: list[dict],ledger: Ledger) -> None:
    by={r['canonical_label']:r for r in rows};freeze=strict_json(root/'FREEZE.json')
    starts={r['data'].get('attempt') for r in ledger.records if r['event']=='ATTEMPT_START'}
    for attempt in sorted((root/'attempts').iterdir()):
        if not attempt.is_dir() or attempt.is_symlink():raise ValueError('Unexpected attempt path')
        # A crash before dispatch cannot have run a worker. Keep even this setup evidence.
        if not (attempt/'TASK.json').exists():
            if attempt.relative_to(root).as_posix() in starts:raise ValueError('Dispatched attempt lost its task')
            os.rename(attempt,root/'archive'/attempt.name)
            ledger.add('INTERRUPTED_SAME_SEED_RETRY',label=None,archived='archive/'+attempt.name,reason='undispatched incomplete setup')
            continue
        task=strict_json(attempt/'TASK.json');row=task['row']
        if by.get(row['canonical_label'])!=row or row['stage']!='development':raise ValueError('Orphan task identity/stage')
        lease=(attempt/'WORKER_LOCK').open('a+b')
        try:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Previous worker is still active; wait before resuming')
            if (attempt/'FAILED.json').exists():raise ValueError('Prior runtime/numerical failure. Keep files and return the report; do not reroll.')
            exitfile=attempt/'WORKER_EXIT.json'
            if exitfile.exists() and strict_json(exitfile)['returncode']>0:raise ValueError('Prior failed worker exit; review required')
            if (attempt/'COMPLETE.json').exists():
                promote(root,attempt,row,ledger,recovered=True)
            else:
                dest=root/'archive'/attempt.name
                if dest.exists():raise ValueError('Interrupted archive already exists')
                os.rename(attempt,dest)
                ledger.add('INTERRUPTED_SAME_SEED_RETRY',label=row['canonical_label'],archived=dest.relative_to(root).as_posix(),seed_unchanged=True)
        finally:lease.close()
    committed={r['data']['label'] for r in ledger.records if r['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT')}
    started={r['data']['label'] for r in ledger.records if r['event']=='ATTEMPT_START'}
    for row in rows:
        p=safe_child(root,row['relative'])
        if p.exists() and row['canonical_label'] not in committed:
            if row['canonical_label'] not in started:raise ValueError('Data lacks a start record')
            verify_tree(p);validate_complete(p,row,freeze)
            ledger.add('RECOVER_COMMIT',label=row['canonical_label'],relative=row['relative'],manifest_sha256=sha(p/'FILES.json'))


def inventory(root: Path,rows: list[dict],ledger: Ledger) -> dict:
    freeze=strict_json(root/'FREEZE.json');done={};leaves=set()
    commits={}
    for rec in ledger.records:
        if rec['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT'):
            data=rec['data'];label=data['label']
            if label in commits:raise ValueError('Duplicate commit record')
            commits[label]=data
    for row in rows:
        p=safe_child(root,row['relative']);label=row['canonical_label']
        if p.exists():
            if row['stage']!='development':raise ValueError('Stage A must not contain test data')
            verify_tree(p);m=validate_complete(p,row,freeze)
            if label not in commits or sha(p/'FILES.json')!=commits[label]['manifest_sha256']:raise ValueError('Commit binding absent/different')
            done[label]=m;leaves.add(p)
        elif label in commits:raise ValueError('Committed run is missing; refusing to silently regenerate it')
    for p in (root/'data').rglob('*'):
        if p.is_symlink():raise ValueError('Symlink in generated data')
        if p.is_file() and p.parent not in leaves:raise ValueError('Unexpected file in data tree')
    if set(commits)!=set(done):raise ValueError('Unregistered commit')
    return done


def progress(root: Path,rows: list[dict],done: dict,status: str,session: str) -> dict:
    c=Counter((m['replication'],m['role'],m['fault']) for m in done.values())
    roles=[{'replication':rep,'role':role,'fault':fault,'completed':c[rep,role,fault],'planned':n}
      for rep in range(1,6) for role,fault,n in DEVELOPMENT_ROLES]
    obj={'study_id':STUDY,'utc':now(),'session':session,'status':status,
      'development_completed':len(done),'development_total':4575,
      'test_completed':0,'shared_test_planned':1000,'planned_all_stages':5575,
      'per_replication':[{'replication':rep,'completed':sum(m['replication']==rep for m in done.values()),'planned':915} for rep in range(1,6)],
      'roles':roles,'learning_started_by_this_tool':False,'test_generation_available':False,
      'original_documents_or_data_modified':False}
    write_json(root/'PROGRESS.json',obj,replace=True)
    return obj


def request_stop(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():return 'NO_STUDY_OR_ACTIVE_SESSION'
    if strict_json(root/'FREEZE.json').get('study_id')!=STUDY:raise ValueError('Wrong study root')
    p=root/'ACTIVE_SESSION.json'
    if not p.exists():return 'NO_ACTIVE_SESSION'
    active=strict_json(p)
    if not active.get('active'):return 'NO_ACTIVE_SESSION'
    session=active['session']
    if not isinstance(session,str) or any(c not in '0123456789TZ_abcdef' for c in session):raise ValueError('Invalid session identity')
    target=root/'stop_requests'/(session+'.json')
    if not target.exists():write_json(target,{'session':session,'requested_utc':now(),'request':'stop launching; finish current trajectories'})
    return 'STOP_REQUESTED: wait for the running window to finish and write its return ZIP'


def generate(root: Path,rows: list[dict],done: dict,hours: float,workers: int,ledger: Ledger,session: str,lock_fd: int) -> str:
    if (root/'BLOCKED.json').exists():raise ValueError('Prior failure is blocked; return its evidence before retrying')
    if any((root/n).exists() for n in ('MODEL_LOCK.json','FIVE_FIT_MODEL_LOCK.json','EVALUATION_LOCK.json')):
        raise ValueError('Learning/test lock exists; development generation cannot continue')
    todo=[r for r in ordered_development(rows) if r['canonical_label'] not in done]
    if not todo:return 'DEVELOPMENT_4575_COMPLETE_NO_LEARNING_OR_TEST'
    freeze=strict_json(root/'FREEZE.json');active={};index=0;stop=False;stop_notice=False;failure=None;interruption=False
    clock=time.monotonic();deadline=clock+hours*3600;last_heartbeat=clock
    def stop_signal(signum,frame):
        nonlocal stop
        stop=True;print('\nSTOP_REQUESTED: finishing active trajectories. Do not close this window.',flush=True)
    prior_handlers={s:signal.signal(s,stop_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
      VECLIB_MAXIMUM_THREADS='1',TEP_BACKEND='python',PYTHONDONTWRITEBYTECODE='1')
    ledger.add('SESSION_START',session=session,stage='development',hours=hours,workers=workers)
    write_json(root/'ACTIVE_SESSION.json',{'session':session,'active':True,'started_utc':now(),'pid':os.getpid()},replace=True)
    progress(root,rows,done,'GENERATING_DEVELOPMENT',session)
    def collect(proc,log,attempt,row):
        nonlocal stop,failure,interruption
        rc=proc.returncode;log.close()
        write_json(attempt/'WORKER_EXIT.json',{'returncode':rc,'pid':proc.pid,'observed_utc':now()})
        if rc==0:
            try:
                m=promote(root,attempt,row,ledger);done[row['canonical_label']]=m
                print(f"COMPLETE {len(done)}/4575 | rep {row['replication']}/5 | {row['role']} F{row['fault']:02d} R{row['run']:04d} | {m['wall_seconds']:.1f}s",flush=True)
            except BaseException as exc:stop=True;failure='Commit validation failed: '+repr(exc)
        elif rc<0 and not (attempt/'FAILED.json').exists():
            stop=True;interruption=True
            print('WORKER_INTERRUPTED: original task/seed retained; no replacement is chosen.',flush=True)
        else:
            stop=True;failure='WORKER_FAILED: '+row['canonical_label']+' exit='+str(rc)
            print(failure,flush=True)
        progress(root,rows,done,'STOPPING' if stop else 'GENERATING_DEVELOPMENT',session)
    try:
        while index<len(todo) or active:
            stop_path=root/'stop_requests'/(session+'.json')
            if time.monotonic()>=deadline or stop_path.exists():stop=True
            if stop and not stop_notice:
                print('SESSION_LIMIT_OR_STOP: no more launches; waiting for '+str(len(active))+' active trajectories.',flush=True);stop_notice=True
            while not stop and time.monotonic()<deadline and len(active)<workers and index<len(todo):
                if shutil.disk_usage(root).free<2*1024**3:
                    stop=True;interruption=True;print('LOW_DISK: stopping new launches without changing seeds.',flush=True);break
                row=todo[index];index+=1
                attempt=root/'attempts'/(f"rep{row['replication']:02d}_{row['role']}_F{row['fault']:02d}_R{row['run']:04d}_"+uuid.uuid4().hex[:10]);attempt.mkdir()
                task={'root':str(root),'row':row,'release_manifest_sha256':freeze['release_manifest_sha256'],
                  'seed_manifest_sha256':freeze['seed_manifest_sha256'],'session':session}
                write_json(attempt/'TASK.json',task)
                ledger.add('ATTEMPT_START',label=row['canonical_label'],attempt=attempt.relative_to(root).as_posix(),session=session)
                log=(attempt/'WORKER_LOG.txt').open('wb')
                try:
                    proc=subprocess.Popen([sys.executable,'-I','-B',str(root/'release/code/worker.py'),str(attempt/'TASK.json')],
                      stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd='/',env=env,
                      start_new_session=True,pass_fds=(lock_fd,))
                except BaseException:
                    log.close();raise
                active[proc.pid]=(proc,log,attempt,row)
                print(f"START rep {row['replication']}/5 {row['role']} F{row['fault']:02d} R{row['run']:04d}",flush=True)
            if not active:break
            for pid,items in list(active.items()):
                if items[0].poll() is not None:
                    del active[pid];collect(*items)
            if time.monotonic()-last_heartbeat>=30:
                print(f"RUNNING | completed {len(done)}/4575 | active {len(active)} | session elapsed {(time.monotonic()-clock)/60:.1f} min",flush=True)
                last_heartbeat=time.monotonic()
            time.sleep(.3)
        if failure:
            write_json(root/'BLOCKED.json',{'utc':now(),'session':session,'failure':failure,'action':'preserve all files and return the report; no automatic seed substitution'})
            raise RuntimeError(failure)
        result='DEVELOPMENT_4575_COMPLETE_NO_LEARNING_OR_TEST' if len(done)==4575 else ('SESSION_INTERRUPTED_SAME_SEEDS_RETAINED' if interruption else 'SESSION_PAUSED_RESUMABLE')
        ledger.add('SESSION_END',session=session,stage='development',status=result,completed=len(done),elapsed_seconds=time.monotonic()-clock)
        return result
    finally:
        # An exception in the coordinator must not leave a worker silently untracked.
        for items in list(active.values()):
            proc=items[0];print('WAITING_FOR_ALREADY_STARTED_RUN='+items[3]['canonical_label'],flush=True)
            proc.wait()
            try:collect(*items)
            except BaseException:traceback.print_exc()
        write_json(root/'ACTIVE_SESSION.json',{'session':session,'active':False,'ended_utc':now()},replace=True)
        for s,h in prior_handlers.items():signal.signal(s,h)


def return_package(root: Path,output: Path,session: str,status: str,error=None) -> Path:
    output.mkdir(parents=True,exist_ok=True)
    temp=output/('TEP_V18_A_'+session+'_RETURN');temp.mkdir(exist_ok=False)
    write_json(temp/'SESSION_RESULT.json',{'study_id':STUDY,'utc':now(),'status':status,'error':error,'root':str(root),
      'scope':'Generation of downstream development only. No model fitting or test generation in this tool.',
      'old_study_files_modified':False,'old_9500_seed_rows_returned':False,'raw_process_arrays_returned':False})
    for name in ('FREEZE.json','SEED_EXCLUSION.json','SEED_MANIFEST.json','EVENTS.jsonl','PROGRESS.json','BLOCKED.json','ACTIVE_SESSION.json'):
        p=root/name
        if p.is_file() and not p.is_symlink():shutil.copyfile(p,temp/name)
    for sub in ('LOCAL_BINDING.json','RELEASE_MANIFEST.json','protocol/EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json','provenance/UNCHANGED_SCIENTIFIC_SOURCE_CHECK.json'):
        p=RELEASE/sub
        if p.is_file():dest=temp/'release_evidence'/sub;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    index=[]
    if (root/'data').is_dir():
        for p in sorted((root/'data').rglob('COMPLETE.json')):
            if p.is_symlink():continue
            try:
                c=strict_json(p);m=strict_json(p.parent/'METADATA.json')
                index.append({**{k:m[k] for k in ('canonical_label','replication','role','fault','run','seed_u32','stage','wall_seconds','shutdown_occurred','started_utc','ended_utc','preonset_sha256')},
                  'relative':p.parent.relative_to(root).as_posix(),'output_sha256':c['output_sha256'],
                  'metadata_sha256':c['metadata_sha256'],'complete_sha256':sha(p),
                  'files_manifest_sha256':sha(p.parent/'FILES.json')})
            except BaseException as exc:
                write_json(temp/('INDEX_READ_ERROR_'+uuid.uuid4().hex[:8]+'.json'),{'path':str(p),'error':repr(exc)})
    write_csv(temp/'COMPLETED_INDEX.csv',index)
    # Incomplete logs are copied (bounded tail only); complete raw arrays stay on the host.
    for attempt in sorted((root/'attempts').glob('*')):
        if attempt.is_symlink() or not attempt.is_dir():continue
        for name in ('TASK.json','FAILED.json','WORKER_LOG.txt','COMPLETE.json','WORKER_EXIT.json'):
            p=attempt/name
            if p.is_file() and not p.is_symlink():
                dest=temp/'incomplete_attempts'/attempt.name/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if name=='WORKER_LOG.txt' and p.stat().st_size>131072:
                    with p.open('rb') as f:f.seek(-131072,2);dest.write_bytes(b'[Last 128 KiB; full log retained locally]\n'+f.read())
                else:shutil.copyfile(p,dest)
    seal_tree(temp);verify_tree(temp)
    archive=output/(temp.name+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(temp.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(temp).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Return ZIP CRC')
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    print('\nUPLOAD_THIS='+str(archive),flush=True)
    return archive


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['GenerateDevelopment','Stop'],required=True)
    ap.add_argument('--hours',type=float,default=.5);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--output-parent',required=True,type=Path);args=ap.parse_args()
    if not (0<args.hours<=12 and 1<=args.workers<=4):raise ValueError('Hours must be positive and <=12, workers 1..4')
    binding=strict_json(RELEASE/'LOCAL_BINDING.json');root=Path(binding['root'])
    if args.mode=='Stop':print(request_stop(root),flush=True);return 0
    session=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    status='FAILED';error=None;lock=None;ledger=None;done={};rows=[];acquired=False
    try:
        if str(Path(sys.executable).absolute())!=binding['runtime']:raise ValueError('Wrong Python path: use the confirmed virtual environment')
        environment=runtime();verify_tree(RELEASE,'RELEASE_MANIFEST.json')
        if not root.parent.is_dir():raise ValueError('Confirmed Linux home is not available')
        lockpath=root.with_name(root.name+'.operation.lock')
        if lockpath.is_symlink():raise ValueError('Operation lock must not be a symlink')
        lock=lockpath.open('a+b')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('A prior session or worker is still running. Do not start another copy.')
        acquired=True
        init_root(root,RELEASE,environment);rows=strict_json(root/'SEED_MANIFEST.json');validate_manifest(rows)
        ledger=Ledger(root/'EVENTS.jsonl',established=True)
        if (root/'BLOCKED.json').exists():raise ValueError('Prior failure is blocked; return its report before restarting')
        recover(root,rows,ledger);done=inventory(root,rows,ledger)
        print(f"DEVELOPMENT {len(done)}/4575 | new tests remain 0/1000 | {args.workers} workers | {args.hours:g} hour launch window",flush=True)
        status=generate(root,rows,done,args.hours,args.workers,ledger,session,lock.fileno())
        progress(root,rows,done,status,session)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
        if ledger is not None:
            try:ledger.add('SESSION_FAILURE',session=session,error=error)
            except BaseException:pass
        if acquired and rows:
            try:progress(root,rows,done,'FAILED',session)
            except BaseException:pass
    finally:
        print('\nTEP_V18_STAGE_A_STATUS='+status,flush=True)
        try:
            # A lock conflict must not snapshot a concurrently changing study.
            if acquired:return_package(root,args.output_parent,session,status,error)
            else:return_package(Path('/nonexistent_TEP_V18_no_lock_snapshot'),args.output_parent,session,status,error)
        except BaseException:traceback.print_exc()
        if lock is not None:lock.close()
    return 0 if status!='FAILED' else 2


if __name__=='__main__':raise SystemExit(main())
