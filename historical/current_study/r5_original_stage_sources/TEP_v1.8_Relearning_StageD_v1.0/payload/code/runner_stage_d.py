"""Evaluate the same 135 locked configurations on the complete shared panel. No fitting/generation."""
from __future__ import annotations
import argparse,csv,fcntl,os,shutil,signal,sys,time,traceback,uuid,zipfile
from pathlib import Path
import numpy as np
from storage import strict_json,sha,regular,safe_child,write_json,write_csv,seal_tree,verify_tree,Ledger,now
from inputs_d import PAYLOAD,STUDY,FAULTS,require,gate,runtime,verify_raw,read_condition
import analysis_d as analysis
import science

class DLedger(Ledger):
    allowed={'SETUP','SESSION_START','INPUTS_VERIFIED','EVALUATE_START','REP_COMPLETE','RECOVER_REP','INTERRUPTED_SAME_TASK','SUMMARY_START','SUMMARY_COMPLETE','RECOVER_SUMMARY','EVALUATE_COMPLETE','SESSION_END','SESSION_FAILURE'}

def frozen_bindings(b,payload=PAYLOAD):
    return {'study_id':STUDY,'tool_id':b['tool_id'],'all_models_lock_sha256':b['all_models_lock_sha256'],
            'test_completed_index_sha256':b['test_completed_index_sha256'],'panel_completion_sha256':b['panel_completion_sha256'],
            'evaluation_spec_sha256':sha(payload/'EVALUATION_SPEC.json'),'payload_manifest_sha256':sha(payload/'MANIFEST.json')}

def setup(work,b,environment,payload=PAYLOAD):
    frozen={**frozen_bindings(b,payload),'environment':environment,'root':b['root']}
    require(not work.is_symlink(),'Evaluation directory symlink')
    if work.exists():
        regular(work/'FREEZE_D.json');old=strict_json(work/'FREEZE_D.json')
        require(all(old.get(k)==v for k,v in frozen.items()),'Existing evaluation belongs to a different release/binding; preserve it')
        DLedger(work/'EVENTS.jsonl',established=True)
    else:
        require(shutil.disk_usage(work.parent).free>=1024**3,'Less than 1 GiB free: evaluation not started')
        tmp=work.with_name(work.name+'.setup_'+uuid.uuid4().hex[:8]);tmp.mkdir()
        for n in ('attempts','archive','evaluated','stop_requests'):(tmp/n).mkdir()
        write_json(tmp/'FREEZE_D.json',{**frozen,'created_utc':now(),'scope':'Score all frozen models; no test-dependent fitting, selection, thresholding or regeneration'})
        shutil.copyfile(payload/'EVALUATION_SPEC.json',tmp/'EVALUATION_SPEC.json')
        DLedger(tmp/'EVENTS.jsonl').add('SETUP',freeze_sha256=sha(tmp/'FREEZE_D.json'),evaluation_started=False)
        require(not work.exists(),'Evaluation destination appeared');os.rename(tmp,work)
    for n in ('attempts','archive','evaluated','stop_requests'):
        require((work/n).is_dir() and not (work/n).is_symlink(),'Missing evaluation subdirectory: '+n)
    return frozen_bindings(b,payload)

def check_rep(path,rep,bindings,full=True):
    verify_tree(path);m=strict_json(path/'REP_COMPLETION.json')
    require(m['replication']==rep and m['configurations']==27 and m['bindings']==bindings,'Scored repetition binding differs')
    require({p.name for p in path.iterdir() if p.is_dir()}=={f'F{f:02d}' for f in FAULTS},'Incomplete scored fault set')
    for f in FAULTS:
        info,*_=analysis.validate_pair(path/f'F{f:02d}',rep,f,full)
        require(info['frozen_bindings']==bindings,'Pair result binding changed')
    return m

def recover(work,ledger,bindings):
    starts={r['data']['attempt']:r['data'] for r in ledger.records if r['event'] in ('EVALUATE_START','SUMMARY_START')}
    for p in sorted((work/'attempts').iterdir()):
        require(p.is_dir() and not p.is_symlink(),'Unknown evaluation attempt')
        require(not (p/'FAILED.json').exists(),'Previous numerical/software failure retained; return evidence')
        rel=p.relative_to(work).as_posix();require(rel in starts,'Unrecorded evaluation attempt; preserve and review')
        task=starts[rel]
        if task.get('kind')=='summary':
            if (p/'FILES.json').exists():
                verify_tree(p);require(not (work/'summary').exists(),'Summary destination already exists');os.rename(p,work/'summary')
                ledger.add('RECOVER_SUMMARY',manifest_sha256=sha(work/'summary/FILES.json'),same_inputs=True)
            else:
                os.rename(p,work/'archive'/p.name);ledger.add('INTERRUPTED_SAME_TASK',attempt=rel,kind='summary',same_inputs=True)
        else:
            rep=task['replication'];dest=work/'evaluated'/f'rep{rep:02d}'
            if (p/'FILES.json').exists():
                check_rep(p,rep,bindings);require(not dest.exists(),'Scored repetition already exists');os.rename(p,dest)
                ledger.add('RECOVER_REP',replication=rep,manifest_sha256=sha(dest/'FILES.json'),same_inputs=True)
            else:
                os.rename(p,work/'archive'/p.name);ledger.add('INTERRUPTED_SAME_TASK',attempt=rel,replication=rep,same_inputs=True)
    commits={}
    for rec in ledger.records:
        if rec['event'] in ('REP_COMPLETE','RECOVER_REP'):
            d=rec['data'];require(d['replication'] not in commits,'Duplicate scored-repetition commit');commits[d['replication']]=d['manifest_sha256']
    for p in (work/'evaluated').iterdir():
        require(p.name in {f'rep{i:02d}' for i in range(1,6)} and p.is_dir() and not p.is_symlink(),'Unregistered evaluation directory')
        rep=int(p.name[-2:]);check_rep(p,rep,bindings)
        if rep not in commits:
            require(any(x.get('replication')==rep for x in starts.values()),'Scored result without start event')
            ledger.add('RECOVER_REP',replication=rep,manifest_sha256=sha(p/'FILES.json'),same_inputs=True);commits[rep]=sha(p/'FILES.json')
        require(commits[rep]==sha(p/'FILES.json'),'Committed result bytes changed')
    require(set(commits)=={int(p.name[-2:]) for p in (work/'evaluated').iterdir()},'Committed evaluation missing: do not overwrite/regenerate')
    if (work/'summary').exists():
        verify_tree(work/'summary');entries=[r for r in ledger.records if r['event'] in ('SUMMARY_COMPLETE','RECOVER_SUMMARY')]
        if not entries:
            require(any(r['event']=='SUMMARY_START' for r in ledger.records),'Summary without aggregation start')
            ledger.add('RECOVER_SUMMARY',manifest_sha256=sha(work/'summary/FILES.json'),same_inputs=True)
        else:require(len(entries)==1 and entries[0]['data']['manifest_sha256']==sha(work/'summary/FILES.json'),'Summary commit changed')
    return set(commits)

def request_stop(work):
    require(work.is_dir() and not work.is_symlink(),'Stage D has not started')
    regular(work/'ACTIVE_SESSION.json');a=strict_json(work/'ACTIVE_SESSION.json')
    if a.get('active') is not True:return 'NO_ACTIVE_STAGE_D_SESSION'
    sid=a['session'];require(isinstance(sid,str) and sid and set(sid)<=set('0123456789abcdefTZ_'),'Invalid session identifier')
    p=work/'stop_requests'/(sid+'.json')
    if not p.exists():write_json(p,{'session':sid,'utc':now(),'request':'Finish current repetition, then pause; during summary finish current summary'})
    return 'STOP_REQUESTED_WAIT_FOR_CURRENT_REPETITION'

def progress(work,done,status,session):
    write_json(work/'PROGRESS_D.json',{'utc':now(),'session':session,'status':status,'repetitions_scored':len(done),'configurations_scored':len(done)*27,
       'configurations_total':135,'unique_test_trajectories':1000,'fault_n_per_fit':250,'normal_n':250,'new_trajectories_generated':0,'models_retrained':0},replace=True)

def export(work,payload,output,session,status,error,broot=None):
    output.mkdir(parents=True,exist_ok=True);dest=output/('TEP_V18_D_'+session+'_RETURN');dest.mkdir()
    write_json(dest/'SESSION_RESULT.json',{'utc':now(),'session':session,'status':status,'error':error,'fitting_or_recalibration':False,'test_generation':False,
       'scope':'Author-operated raw-to-frozen-score evaluation and conditional shared-panel inference; not independent simulator certification','raw_52_arrays_exported':False})
    shutil.copyfile(payload/'BINDING.json',dest/'TOOL_BINDING.json');shutil.copyfile(payload/'MANIFEST.json',dest/'TOOL_MANIFEST.json')
    for n in ('FREEZE_D.json','EVALUATION_SPEC.json','EVENTS.jsonl','INPUT_VERIFICATION.json','FINAL_INPUT_VERIFICATION.json','PROGRESS_D.json','ACTIVE_SESSION.json','COMPLETION.json','SESSION_ERROR.json'):
        p=work/n
        if p.is_file() and not p.is_symlink():shutil.copyfile(p,dest/n)
    for n in ('evaluated','summary'):
        p=work/n
        if p.is_dir() and not p.is_symlink():shutil.copytree(p,dest/n)
    # Include fixed models needed for score interpretation; copied, never modified.
    if broot and (broot/'models').is_dir():
        for p in (broot/'models').rglob('*'):
            if p.is_file() and p.name in ('DESIGN.npz','MODEL.npz','PARAMETERS.npz','SELECTION.json','OUTER.npz','REPLICATION_LOCK.json'):
                q=dest/'frozen_models'/p.relative_to(broot/'models');q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for n in ('code','reference','assets'):shutil.copytree(payload/n,dest/('reproduction_'+n))
    shutil.copyfile(payload/'EVALUATION_SPEC.json',dest/'reproduction_spec.json')
    for area in ('attempts','archive'):
        for p in (work/area).glob('**/*'):
            if p.is_file() and not p.is_symlink() and p.name in ('FAILED.json','TASK.json','REP_COMPLETION.json'):
                q=dest/'partial_evidence'/area/p.relative_to(work/area);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    seal_tree(dest);verify_tree(dest);zp=output/(dest.name+'.zip')
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest).as_posix())
    with zipfile.ZipFile(zp) as z:require(z.testzip() is None,'Return archive CRC failure')
    Path(str(zp)+'.sha256').write_text(sha(zp)+'  '+zp.name+'\n',encoding='ascii')
    print('\nUPLOAD_THIS='+str(zp),flush=True)
    return zp

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['Evaluate','Stop'],default='Evaluate');ap.add_argument('--output-parent',type=Path,required=True);args=ap.parse_args()
    b=strict_json(PAYLOAD/'BINDING.json');root=Path(b['root']);work=safe_child(root,b['work_subdirectory'])
    if args.mode=='Stop':print(request_stop(work),flush=True);return 0
    sid=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8];ledger=None;lease=None;owned=False;status='FAILED';error=None;done=set();broot=None;attempt=None;handlers={};stop=[False];export_failed=False
    try:
        env=runtime(b);verify_tree(PAYLOAD,'MANIFEST.json');regular(root.with_name(root.name+'.operation.lock'))
        lease=root.with_name(root.name+'.operation.lock').open('a+b')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another A/B/C/D operation or surviving worker is active; wait')
        b,broot,croot,rows,index=gate(root,PAYLOAD);bindings=setup(work,b,env,PAYLOAD);owned=True;ledger=DLedger(work/'EVENTS.jsonl',established=True)
        require(not (work/'SESSION_ERROR.json').exists(),'Previous Stage D failure preserved; return evidence before retry')
        ledger.add('SESSION_START',session=sid,operation='fixed_model_scoring')
        write_json(work/'ACTIVE_SESSION.json',{'active':True,'session':sid,'pid':os.getpid(),'started_utc':now()},replace=True)
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        stopped=lambda:stop[0] or (work/'stop_requests'/(sid+'.json')).exists()
        done=recover(work,ledger,bindings)
        print(f'STAGE D | {len(done)}/5 repetitions reused | 135 FIXED configurations | NO GENERATION OR FITTING',flush=True)
        evid=verify_raw(root,croot,rows,index)
        write_json(work/'INPUT_VERIFICATION.json',{'utc':now(),'test_count':len(evid),'raw_values_verified':True,'all_models_lock_sha256':b['all_models_lock_sha256'],
           'test_index_sha256':b['test_completed_index_sha256'],'inputs':evid,'read_only_prior_A_B_C':True},replace=True)
        ledger.add('INPUTS_VERIFIED',session=sid,raw_count=1000,verification_sha256=sha(work/'INPUT_VERIFICATION.json'))
        spec=strict_json(PAYLOAD/'EVALUATION_SPEC.json');protocol=strict_json(PAYLOAD/'reference/EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json')
        pca=science.load_npz(PAYLOAD/'assets/PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz');basis=np.load(PAYLOAD/'assets/RESIDUAL_BASIS_NEWSEED_PCA.npy',allow_pickle=False)
        objects={}
        for rep in range(1,6):
            if rep in done:continue
            if stopped():break
            if not objects:
                for f in (0,3,9,15):
                    print(f'LOAD FIXED TEST INPUTS F{f:02d} | 250 runs',flush=True);objects[f]=read_condition(root,croot,rows,index,f,pca,basis)
            attempt=work/'attempts'/f'rep{rep:02d}_{sid}'
            require(not attempt.exists(),'Attempt destination exists')
            ledger.add('EVALUATE_START',session=sid,replication=rep,attempt=attempt.relative_to(work).as_posix(),bindings=bindings)
            attempt.mkdir();write_json(attempt/'TASK.json',{'replication':rep,'bindings':bindings,'session':sid})
            print(f'EVALUATE REP {rep}/5 | 27 locked configurations',flush=True)
            for f in FAULTS:
                print(f'  SCORE F{f:02d}: all 9 configurations, 250 normal + 250 fault',flush=True)
                analysis.score_pair(rep,f,objects[0],objects[f],broot/'models'/f'rep{rep:02d}'/f'F{f:02d}',attempt/f'F{f:02d}',bindings)
            gate(root,PAYLOAD)
            write_json(attempt/'REP_COMPLETION.json',{'replication':rep,'configurations':27,'bindings':bindings,'completed_utc':now()})
            seal_tree(attempt);check_rep(attempt,rep,bindings,False);dest=work/'evaluated'/f'rep{rep:02d}';require(not dest.exists(),'Committed repetition would be overwritten');os.rename(attempt,dest);attempt=None
            ledger.add('REP_COMPLETE',session=sid,replication=rep,manifest_sha256=sha(dest/'FILES.json'));done.add(rep);progress(work,done,'SCORING',sid)
            print(f'SCORED REP {rep}/5 | total {len(done)*27}/135',flush=True)
        objects.clear()
        if len(done)<5 or (stopped() and not (work/'summary').exists()):status='SESSION_PAUSED_RESUMABLE'
        else:
            if not (work/'summary').exists():
                print('AGGREGATE | all 135 configurations | 20000 paired shared-ID draws | no refitting',flush=True)
                attempt=work/'attempts'/('summary_'+sid)
                ledger.add('SUMMARY_START',session=sid,kind='summary',attempt=attempt.relative_to(work).as_posix(),bindings=bindings)
                analysis.aggregate(work/'evaluated',attempt,spec,protocol['primary_contrasts'])
                require(not (work/'summary').exists(),'Existing summary would be overwritten');os.rename(attempt,work/'summary');attempt=None
                ledger.add('SUMMARY_COMPLETE',session=sid,manifest_sha256=sha(work/'summary/FILES.json'))
            gate(root,PAYLOAD)
            for row in rows:require(sha(root/row['relative']/'OUTPUT.npz')==index[row['canonical_label']]['output_sha256'],'Original raw test changed during scoring')
            write_json(work/'FINAL_INPUT_VERIFICATION.json',{'utc':now(),'original_model_and_A_B_C_bytes':'PASS','raw_test_output_hashes_unchanged':1000,'retraining':False,'generation':False},replace=True)
            completed={'bindings':bindings,'configurations_scored':135,'unique_test_trajectories':1000,'repetitions':5,'primary_contrasts':3,'per_fit_contrasts':15,'summary_manifest_sha256':sha(work/'summary/FILES.json')}
            cp=work/'COMPLETION.json'
            if cp.exists():require(all(strict_json(cp).get(k)==v for k,v in completed.items()),'Existing completed evaluation changed')
            else:write_json(cp,{**completed,'completed_utc':now()})
            events=[r for r in ledger.records if r['event']=='EVALUATE_COMPLETE']
            if not events:ledger.add('EVALUATE_COMPLETE',session=sid,completion_sha256=sha(cp),configuration_count=135)
            else:require(len(events)==1 and events[0]['data']['completion_sha256']==sha(cp),'Evaluation completion ledger differs')
            status='ALL_135_CONFIGURATIONS_EVALUATED_NO_REFIT'
        ledger.add('SESSION_END',session=sid,status=status,scored=len(done)*27);progress(work,done,status,sid)
    except BaseException as e:
        error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};traceback.print_exc();status='FAILED'
        if attempt and attempt.is_dir() and not (attempt/'FILES.json').exists():write_json(attempt/'FAILED.json',error,replace=True)
        if ledger:
            ledger.add('SESSION_FAILURE',session=sid,error=error);write_json(work/'SESSION_ERROR.json',{'session':sid,'utc':now(),'error':error},replace=True);progress(work,done,status,sid)
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)
        if ledger:write_json(work/'ACTIVE_SESSION.json',{'active':False,'session':sid,'ended_utc':now(),'status':status},replace=True)
        print('\nTEP_V18_STAGE_D_STATUS='+status,flush=True)
        try:export(work if owned else Path('/nonexistent_TEP_V18_D_no_snapshot'),PAYLOAD,args.output_parent,sid,status,error,broot)
        except BaseException:export_failed=True;traceback.print_exc();print('Return export failed: local originals preserved; send Windows log.',flush=True)
        if lease:lease.close()
    return 0 if status!='FAILED' and not export_failed else 2
if __name__=='__main__':raise SystemExit(main())
