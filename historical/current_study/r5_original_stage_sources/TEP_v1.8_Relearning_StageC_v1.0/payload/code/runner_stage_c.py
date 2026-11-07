"""Generate only the already allocated 1,000 shared tests. No fitting or scoring."""
from __future__ import annotations
import argparse,csv,fcntl,os,shutil,signal,subprocess,sys,time,traceback,uuid,zipfile
from pathlib import Path
from collections import Counter
from storage import now,sha,strict_json,write_json,write_csv,regular,safe_child,seal_tree,verify_tree
from common_c import PAYLOAD,STUDY,CLedger,require,gate,test_rows,runtime,validate_complete,validate_test_row,cfreeze


def setup(root,work,binding,environment):
    """Establish only C state. A allocations, development data and B files are read-only."""
    fixed={'study_id':STUDY,'tool_id':binding['tool_id'],'payload_manifest_sha256':sha(PAYLOAD/'MANIFEST.json'),
           'seed_manifest_sha256':binding['stage_a_control_hashes']['SEED_MANIFEST.json'],
           'all_models_lock_sha256':binding['all_models_lock_sha256'],'stage_b_return_sha256':binding['stage_b_return_sha256'],
           'shared_test_total':1000,'root':str(root),'test_data_relative':'data/shared_test','environment':environment}
    require(not work.is_symlink(),'Stage C directory symlink')
    if work.exists():
        regular(work/'FREEZE_C.json');f=strict_json(work/'FREEZE_C.json')
        require(all(f.get(k)==v for k,v in fixed.items()),'Different existing Stage C tool/batch: no overwrite')
        CLedger(work/'EVENTS.jsonl',established=True)
    else:
        require(not (root/'data/shared_test').exists(),'Unaccounted test files exist before Stage C setup; preserve and review')
        require(shutil.disk_usage(root).free>=2*1024**3,'Less than 2 GiB free: nothing started')
        tmp=work.with_name(work.name+'.setup_'+uuid.uuid4().hex[:8]);tmp.mkdir()
        for n in ('attempts','archive','stop_requests'):(tmp/n).mkdir()
        write_json(tmp/'FREEZE_C.json',{**fixed,'created_utc':now(),'scope':'Fixed shared test generation after all-five model lock; no score evaluation or training'})
        CLedger(tmp/'EVENTS.jsonl').add('SETUP',freeze_sha256=sha(tmp/'FREEZE_C.json'),all_models_lock_sha256=fixed['all_models_lock_sha256'],test_generated=0)
        require(not work.exists(),'Stage C destination appeared');os.rename(tmp,work)
    for n in ('attempts','archive','stop_requests'):
        require((work/n).is_dir() and not (work/n).is_symlink(),'Stage C state subdirectory missing/symlink')
    return cfreeze(work)


def promote(root,work,attempt,row,ledger,recovered=False):
    m=validate_complete(attempt,row,cfreeze(work),read_values=True)
    if not (attempt/'FILES.json').exists():seal_tree(attempt)
    verify_tree(attempt);dest=safe_child(root,row['relative']);dest.parent.mkdir(parents=True,exist_ok=True)
    require(not dest.exists(),'Completed test destination already exists: no replacement')
    os.rename(attempt,dest)
    ledger.add('RECOVER_COMMIT' if recovered else 'ATTEMPT_COMMIT',label=row['canonical_label'],relative=row['relative'],manifest_sha256=sha(dest/'FILES.json'))
    return m


def recover(root,work,rows,ledger):
    by={r['canonical_label']:r for r in rows}
    starts={r['data'].get('attempt') for r in ledger.records if r['event']=='ATTEMPT_START'}
    for attempt in sorted((work/'attempts').iterdir()):
        require(attempt.is_dir() and not attempt.is_symlink(),'Unknown test attempt path')
        if not (attempt/'TASK.json').exists():
            require(attempt.relative_to(work).as_posix() not in starts,'Dispatched attempt lost task: stop for review')
            os.rename(attempt,work/'archive'/attempt.name)
            ledger.add('INTERRUPTED_SAME_SEED_RETRY',label=None,archived=attempt.name,reason='undispatched setup; no trajectory ran')
            continue
        task=strict_json(attempt/'TASK.json');row=task['row'];validate_test_row(row)
        require(by.get(row['canonical_label'])==row,'Unregistered test attempt')
        lp=attempt/'WORKER_LOCK';require(not lp.is_symlink(),'Worker lock symlink');lease=lp.open('a+b')
        try:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Prior worker still active; wait, do not run another session')
            require(not (attempt/'FAILED.json').exists(),'Prior numerical/software failure preserved; no automatic retry')
            ex=attempt/'WORKER_EXIT.json'
            if ex.exists():require(strict_json(ex)['returncode']<=0,'Positive worker error exit: review required')
            if (attempt/'COMPLETE.json').exists():promote(root,work,attempt,row,ledger,recovered=True)
            else:
                dest=work/'archive'/attempt.name;require(not dest.exists(),'Interrupted archive conflict');os.rename(attempt,dest)
                ledger.add('INTERRUPTED_SAME_SEED_RETRY',label=row['canonical_label'],archived=dest.name,seed_unchanged=True,reason='process interruption; retain partial attempt and rerun identical frozen task')
        finally:lease.close()
    committed={r['data']['label'] for r in ledger.records if r['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT')}
    started={r['data']['label'] for r in ledger.records if r['event']=='ATTEMPT_START'}
    for row in rows:
        p=safe_child(root,row['relative'])
        if p.exists() and row['canonical_label'] not in committed:
            require(row['canonical_label'] in started,'Test data without launch record')
            verify_tree(p);validate_complete(p,row,cfreeze(work),read_values=True)
            ledger.add('RECOVER_COMMIT',label=row['canonical_label'],relative=row['relative'],manifest_sha256=sha(p/'FILES.json'))


def inventory(root,work,rows,ledger):
    commits={};done={};leaves=set();f=cfreeze(work)
    for rec in ledger.records:
        if rec['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT'):
            d=rec['data'];require(d['label'] not in commits,'Duplicate test commit');commits[d['label']]=d
    for row in rows:
        p=safe_child(root,row['relative']);label=row['canonical_label']
        if p.exists():
            require(p.is_dir() and not p.is_symlink(),'Invalid test directory');verify_tree(p)
            m=validate_complete(p,row,f,read_values=True)
            require(label in commits and commits[label]['relative']==row['relative'] and commits[label]['manifest_sha256']==sha(p/'FILES.json'),'Commit identity differs')
            done[label]=m;leaves.add(p)
        else:require(label not in commits,'Committed test is missing: refusing regeneration')
    shared=safe_child(root,'data/shared_test')
    if shared.exists():
        for p in shared.rglob('*'):
            require(not p.is_symlink(),'Test data symlink')
            if p.is_file():require(p.parent in leaves,'Unrecognized file under shared test data')
    require(set(commits)==set(done),'Unregistered test completion')
    return done


def progress(work,done,status,session):
    counts=Counter(m['fault'] for m in done.values())
    obj={'study_id':STUDY,'utc':now(),'session':session,'status':status,'test_completed':len(done),'test_total':1000,
         'per_condition':[{'fault':f,'completed':counts[f],'planned':250} for f in (0,3,9,15)],
         'models_locked':135,'development_completed':4575,'test_scored':0,'retraining_available':False,
         'original_A_B_files_modified':False,'shared_panel_not_five_distinct_panels':True}
    write_json(work/'PROGRESS_C.json',obj,replace=True)


def request_stop(root,work):
    require(work.is_dir() and not work.is_symlink(),'Stage C has not started')
    regular(work/'ACTIVE_SESSION.json');a=strict_json(work/'ACTIVE_SESSION.json')
    if a.get('active') is not True:return 'NO_ACTIVE_STAGE_C_SESSION'
    sid=a['session'];require(isinstance(sid,str) and bool(sid) and set(sid)<=set('0123456789abcdefTZ_'),'Invalid session identifier')
    p=work/'stop_requests'/(sid+'.json')
    if not p.exists():write_json(p,{'session':sid,'utc':now(),'request':'stop new tests and finish already-started trajectories'})
    return 'STOP REQUESTED: keep the generation window open until its return ZIP is written'


def generate(root,work,rows,done,hours,workers,ledger,session,lock_fd):
    require(not (work/'BLOCKED.json').exists(),'Retained Stage C failure: return its evidence; no automatic seed replacement')
    todo=[r for r in rows if r['canonical_label'] not in done]
    if not todo:
        require(len(done)==1000,'A complete panel needs all 1000 shared tests')
        return 'SHARED_TEST_1000_COMPLETE_NOT_EVALUATED'
    f=cfreeze(work);active={};i=0;stop=False;stop_notice=False;failure=None;interrupted=False;clock=time.monotonic();deadline=clock+hours*3600;heart=clock
    def on_signal(sig,frame):
        nonlocal stop
        stop=True;print('STOP REQUESTED: finishing current tests. Keep this window open.',flush=True)
    handlers={s:signal.signal(s,on_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',TEP_BACKEND='python',PYTHONDONTWRITEBYTECODE='1')
    def collect(proc,log,attempt,row):
        nonlocal stop,failure,interrupted
        log.close();write_json(attempt/'WORKER_EXIT.json',{'returncode':proc.returncode,'pid':proc.pid,'observed_utc':now()})
        if proc.returncode==0:
            try:
                m=promote(root,work,attempt,row,ledger);done[row['canonical_label']]=m
                print(f"COMPLETE {len(done)}/1000 | SHARED F{row['fault']:02d} R{row['run']:04d} | {m['wall_seconds']:.1f}s",flush=True)
            except BaseException as e:stop=True;failure='Test commit validation: '+repr(e)
        elif proc.returncode<0 and not (attempt/'FAILED.json').exists():
            stop=True;interrupted=True;print('WORKER INTERRUPTED: same frozen task retained',flush=True)
        else:stop=True;failure='Worker failed: '+row['canonical_label']+' exit='+str(proc.returncode);print(failure,flush=True)
        progress(work,done,'STOPPING' if stop else 'GENERATING_SHARED_TEST',session)
    try:
        while i<len(todo) or active:
            if time.monotonic()>=deadline or (work/'stop_requests'/(session+'.json')).exists():stop=True
            if stop and not stop_notice:
                print(f'SESSION LIMIT OR STOP: waiting for {len(active)} active tests; no more launches.',flush=True);stop_notice=True
            while not stop and time.monotonic()<deadline and len(active)<workers and i<len(todo):
                if shutil.disk_usage(root).free<2*1024**3:
                    stop=True;interrupted=True;print('LOW DISK: pausing safely; no seed replacement.',flush=True);break
                row=todo[i];i+=1;validate_test_row(row)
                attempt=work/'attempts'/f"F{row['fault']:02d}_R{row['run']:04d}_{uuid.uuid4().hex[:10]}";attempt.mkdir()
                task={'root':str(root),'row':row,'payload_manifest_sha256':f['payload_manifest_sha256'],
                      'seed_manifest_sha256':f['seed_manifest_sha256'],'all_models_lock_sha256':f['all_models_lock_sha256'],
                      'stage_c_freeze_sha256':f['_file_sha256'],'session':session,'operation_lock_fd':lock_fd}
                write_json(attempt/'TASK.json',task)
                ledger.add('ATTEMPT_START',label=row['canonical_label'],attempt=attempt.relative_to(work).as_posix(),session=session,all_models_lock_sha256=f['all_models_lock_sha256'])
                log=(attempt/'WORKER_LOG.txt').open('wb')
                try:
                    proc=subprocess.Popen([sys.executable,'-I','-B',str(PAYLOAD/'code/worker_c.py'),str(attempt/'TASK.json')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd='/',env=env,start_new_session=True,pass_fds=(lock_fd,))
                except BaseException:log.close();raise
                active[proc.pid]=(proc,log,attempt,row);print(f"START SHARED F{row['fault']:02d} R{row['run']:04d}",flush=True)
            if not active:break
            for pid,items in list(active.items()):
                if items[0].poll() is not None:
                    del active[pid];collect(*items)
            if time.monotonic()-heart>=30:
                print(f'RUNNING | shared tests {len(done)}/1000 | active {len(active)} | session elapsed {(time.monotonic()-clock)/60:.1f} min',flush=True);heart=time.monotonic()
            time.sleep(.3)
        if failure:
            write_json(work/'BLOCKED.json',{'session':session,'utc':now(),'failure':failure,'action':'preserve all files and return evidence; no seed substitution'})
            raise RuntimeError(failure)
        return 'SHARED_TEST_1000_COMPLETE_NOT_EVALUATED' if len(done)==1000 else ('SESSION_INTERRUPTED_SAME_SEEDS_RETAINED' if interrupted else 'SESSION_PAUSED_RESUMABLE')
    finally:
        for proc,log,attempt,row in list(active.values()):
            print('WAITING FOR ALREADY STARTED TEST: '+row['canonical_label'],flush=True);proc.wait()
            try:collect(proc,log,attempt,row)
            except BaseException:traceback.print_exc()
        for s,h in handlers.items():signal.signal(s,h)


def write_index(root,work,rows,done):
    result=[]
    for row in sorted(rows,key=lambda r:r['relative']):
        if row['canonical_label'] not in done:continue
        p=safe_child(root,row['relative']);c=strict_json(p/'COMPLETE.json');m=done[row['canonical_label']]
        result.append({**{k:row[k] for k in ('canonical_label','replication','role','fault','run','seed_u32','stage','relative')},
                       **{k:m[k] for k in ('wall_seconds','shutdown_occurred','started_utc','ended_utc','preonset_sha256','all_models_lock_sha256')},
                       'output_sha256':c['output_sha256'],'metadata_sha256':c['metadata_sha256'],
                       'complete_sha256':sha(p/'COMPLETE.json'),'files_manifest_sha256':sha(p/'FILES.json')})
    write_csv(work/'TEST_COMPLETED_INDEX.csv',result,replace=True)
    return result


def finalize(work,ledger):
    f=cfreeze(work);index=work/'TEST_COMPLETED_INDEX.csv';regular(index)
    with index.open(newline='',encoding='utf-8') as fh: entries=list(csv.DictReader(fh))
    require(len(entries)==1000 and len({r['canonical_label'] for r in entries})==1000,'Only a complete 1000-row unique index can be finalized')
    require(Counter((r['stage'],int(r['replication']),int(r['fault'])) for r in entries)==Counter({('evaluation',0,x):250 for x in (0,3,9,15)}),'Final panel roles differ')
    require(all(r['all_models_lock_sha256']==f['all_models_lock_sha256'] for r in entries),'Final panel model binding differs')
    data={'study_id':STUDY,'test_count':1000,'per_condition_count':250,'configuration_count':135,'all_models_lock_sha256':f['all_models_lock_sha256'],
          'seed_manifest_sha256':f['seed_manifest_sha256'],'test_completed_index_sha256':sha(index),'test_scored':0}
    p=work/'PANEL_COMPLETION.json'
    if p.exists():require(all(strict_json(p).get(k)==v for k,v in data.items()),'Existing completed-panel identity changed')
    else:write_json(p,{**data,'completed_utc':now(),'next_step':'Return this generation report. A separate frozen-score evaluation tool is required; no automatic evaluation.'})
    prior=[r for r in ledger.records if r['event']=='TEST_PANEL_COMPLETE']
    if prior:require(len(prior)==1 and prior[0]['data']['completion_sha256']==sha(p),'Completed-panel ledger mismatch')
    else:ledger.add('TEST_PANEL_COMPLETE',completion_sha256=sha(p),count=1000,test_scored=0)


def export(work,output,session,status,error):
    output.mkdir(parents=True,exist_ok=True);dest=output/('TEP_V18_C_'+session+'_RETURN');dest.mkdir()
    write_json(dest/'SESSION_RESULT.json',{'session':session,'status':status,'error':error,'utc':now(),'scope':'Shared test generation only after five model locks',
               'model_fitting':False,'scoring_or_evaluation':False,'new_seed_allocation':False,'original_A_B_modified':False,'raw_arrays_exported':False})
    shutil.copyfile(PAYLOAD/'BINDING.json',dest/'TOOL_BINDING.json');shutil.copyfile(PAYLOAD/'MANIFEST.json',dest/'TOOL_MANIFEST.json')
    for n in ('FREEZE_C.json','EVENTS.jsonl','PROGRESS_C.json','ACTIVE_SESSION.json','TEST_COMPLETED_INDEX.csv','PANEL_COMPLETION.json','BLOCKED.json','SESSION_ERROR.json','PRETEST_CHECK.json'):
        p=work/n
        if p.is_file() and not p.is_symlink():shutil.copyfile(p,dest/n)
    for n in ('STAGE_B_ALL_MODELS_LOCK.json','STAGE_B_ALL_CONFIGURATIONS.csv','STAGE_B_EXPECTED_FILES.json','SEED_MANIFEST.json','EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json'):
        q=dest/'reference'/n;q.parent.mkdir(exist_ok=True);shutil.copyfile(PAYLOAD/'reference'/n,q)
    for area in ('attempts','archive'):
        for folder in sorted((work/area).glob('*')):
            if not folder.is_dir() or folder.is_symlink():continue
            for n in ('TASK.json','FAILED.json','COMPLETE.json','WORKER_EXIT.json','WORKER_LOG.txt'):
                p=folder/n
                if p.is_file() and not p.is_symlink():
                    q=dest/'partial_evidence'/area/folder.name/n;q.parent.mkdir(parents=True,exist_ok=True)
                    if p.stat().st_size>131072 and n=='WORKER_LOG.txt':
                        with p.open('rb') as f:f.seek(-131072,2);q.write_bytes(b'[Last 128 KiB; full original preserved]\n'+f.read())
                    else:shutil.copyfile(p,q)
    seal_tree(dest);verify_tree(dest);zp=output/(dest.name+'.zip')
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest).as_posix())
    with zipfile.ZipFile(zp) as z:require(z.testzip() is None,'Return ZIP CRC')
    Path(str(zp)+'.sha256').write_text(sha(zp)+'  '+zp.name+'\n',encoding='ascii')
    print('\nUPLOAD_THIS='+str(zp),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['GenerateTest','Stop'],default='GenerateTest');ap.add_argument('--hours',type=float,default=2);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--output-parent',type=Path,required=True);args=ap.parse_args()
    require(0<args.hours<=12 and 1<=args.workers<=8,'Hours must be >0 and <=12; workers 1..8')
    b=strict_json(PAYLOAD/'BINDING.json');root=Path(b['root']);work=safe_child(root,b['work_subdirectory'])
    if args.mode=='Stop':print(request_stop(root,work),flush=True);return 0
    session=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
    lease=None;ledger=None;owned=False;status='FAILED';error=None;done={};rows=[];export_failed=False;clock=time.monotonic()
    try:
        env=runtime(b);verify_tree(PAYLOAD,'MANIFEST.json')
        require(root.is_dir() and not root.is_symlink(),'Original study directory missing')
        lp=root.with_name(root.name+'.operation.lock');regular(lp);lease=lp.open('a+b')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another A/B/C session or surviving worker is active; wait before starting')
        b,old_freeze,model_lock=gate(root);rows=test_rows(root,b)
        setup(root,work,b,env);owned=True;ledger=CLedger(work/'EVENTS.jsonl',established=True)
        require(not (work/'BLOCKED.json').exists(),'Previous Stage C failure retained: no automatic retry')
        ledger.add('SESSION_START',session=session,stage='shared_test_generation',hours=args.hours,workers=args.workers)
        write_json(work/'ACTIVE_SESSION.json',{'session':session,'stage':'C','active':True,'started_utc':now(),'pid':os.getpid()},replace=True)
        ck={'utc':now(),'all_models_lock_sha256':b['all_models_lock_sha256'],'model_configurations_checked':135,'stage_b_byte_bindings':'PASS','A_seed_freeze_unchanged':True,'test_panel_rows':1000,'scoring_or_retraining':False}
        write_json(work/'PRETEST_CHECK.json',ck,replace=True);ledger.add('PRETEST_LOCK_RECHECK',session=session,**ck)
        recover(root,work,rows,ledger);done=inventory(root,work,rows,ledger)
        print(f'SHARED TEST {len(done)}/1000 | ALL 135 MODELS LOCKED | {args.workers} workers | {args.hours:g} hour launch window | NO EVALUATION',flush=True)
        progress(work,done,'GENERATING_SHARED_TEST',session)
        status=generate(root,work,rows,done,args.hours,args.workers,ledger,session,lease.fileno())
        gate(root)  # Frozen old bytes must also be unchanged at the session boundary.
        done=inventory(root,work,rows,ledger);write_index(root,work,rows,done)
        if len(done)==1000:finalize(work,ledger)
        ledger.add('SESSION_END',session=session,status=status,completed=len(done),test_scored=0,elapsed_seconds=time.monotonic()-clock)
        progress(work,done,status,session)
    except BaseException as e:
        status='FAILED';error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};traceback.print_exc()
        if ledger:
            ledger.add('SESSION_FAILURE',session=session,error=error)
            write_json(work/'SESSION_ERROR.json',{'session':session,'utc':now(),'error':error},replace=True)
            progress(work,done,status,session)
    finally:
        if ledger:write_json(work/'ACTIVE_SESSION.json',{'session':session,'active':False,'ended_utc':now(),'status':status},replace=True)
        print('\nTEP_V18_STAGE_C_STATUS='+status,flush=True)
        try:export(work if owned else Path('/nonexistent_TEP_V18_C_no_owned_snapshot'),args.output_parent,session,status,error)
        except BaseException:export_failed=True;traceback.print_exc();print('RETURN EXPORT FAILED: files preserved; return Windows log.',flush=True)
        if lease:lease.close()
    return 0 if status!='FAILED' and not export_failed else 2
if __name__=='__main__':raise SystemExit(main())
