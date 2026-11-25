"""Resumable prospective Fault 9 paired-relearning experiment. Never refit on calibration/evaluation."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse,concurrent.futures,csv,datetime,hashlib,json,platform,shutil,sys,time,uuid,zipfile
from pathlib import Path
import numpy as np
from simulation import scientific_runner as sim
from learning import pipeline as learn
HERE=Path(__file__).resolve().parent
LOCK='be482fddc09941ed45b306da3f3d4a266812111c923cc564e0973b55ce3a7c99'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return sim.sha(Path(p))
def need(x,msg):
    if not x:raise ValueError(msg)
def get(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.new')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def tables(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def design():
    l=HERE/'locks';need(sha(l/'DESIGN_LOCK.json')==LOCK,'Design lock changed.')
    for f,h in get(l/'DESIGN_LOCK.json')['files'].items():need(sha(l/f)==h,'Design input changed: '+f)
    with (l/'CASE_MANIFEST.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    need(len(rows)==len(set(x['key_hex'] for x in rows))==len(set(x['case_id'] for x in rows))==2719,'Wrong manifest size or keys')
    return rows
def verify_code():
    f=HERE/'locks/CODE_LOCK.json';b=get(f);need(b['design_lock']==LOCK,'Wrong numerical release design')
    for n,h in b['files'].items():need(sha(HERE/n)==h,'Numerical code or QA reference changed: '+n)
    return sha(f)
def stage_guard(work,stage,rows):
    future=('calibration','evaluation') if stage=='development' else ('evaluation',)
    bad=[r['case_id'] for r in rows if r['stage'] in future and (work/'raw'/r['case_id']).exists()]
    need(not bad,'Later-stage raw data are present ahead of the required model/threshold lock. Do not delete source data; preserve for investigation: '+','.join(bad[:2]))
def record_ok(row,work):
    f=work/'raw'/row['case_id']
    receipt=sim.completed(f,sim.identity(row,LOCK))
    if receipt is None:return None
    need(receipt['shutdown_occurred'] is False,'Shutdown in registered case '+row['case_id'])
    info=get(f/'RUN_INFO.json')
    ident=sim.identity(row,LOCK)
    need(all(info.get(k)==v for k,v in ident.items()),'RUN_INFO does not identify this registered case: '+row['case_id'])
    need(info['final_step_count']==172800 and 0<int(info['rng_draw_count'])==int(receipt['rng_draw_count'])<2**64 and info['shutdown_occurred'] is False and info['physical_state_sha256']==receipt['physical_state_sha256'],'Invalid duration or RNG lineage')
    return receipt
def work_one(args):
    row,root=args;rec=sim.run_row(row,Path(root)/'raw',LOCK)
    if rec.get('shutdown_occurred') is not False:raise RuntimeError('Shutdown in '+row['case_id']+'; no replacement permitted.')
    return row['case_id']
def count_done(work,rows):return sum((work/'raw'/r['case_id']/'COMPLETE.json').is_file() for r in rows)
def status(work,rows=None):
    rows=design() if rows is None else rows
    return {'design_lock_sha256':LOCK,'at_utc':now(),'completed_physical_files':{s:count_done(work,[r for r in rows if r['stage']==s]) for s in ('development','calibration','evaluation')},'n_total':2719,'learned_recording_specific_models_locked':(work/'MODELS_LOCK.json').exists(),'thresholds_locked':(work/'THRESHOLDS_LOCK.json').exists(),'final':(work/'FINAL_COMPLETE.json').exists()}
def fill(rows,work,all_rows,jobs,cap):
    miss=[r for r in rows if record_ok(r,work) is None]
    if not miss:return True
    todo=miss[:cap[0]] if cap[0] is not None else miss
    if not todo:return False
    print('Generating',len(todo),'of',len(miss),'remaining cases in this stage. Workers:',jobs,flush=True)
    # Keep at most jobs futures in flight so STOP_AFTER_BATCH never queues a whole day of work.
    it=iter(todo);done=0;pending={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        def push():
            if (work/'STOP_AFTER_BATCH').exists():return
            try:r=next(it)
            except StopIteration:return
            pending[pool.submit(work_one,(r,str(work)))]=r['case_id']
        for _ in range(jobs):push()
        while pending:
            finished,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in finished:
                case=pending.pop(fut);fut.result();done+=1
                if done%max(1,jobs)==0 or done==len(todo):
                    put(work/'PROGRESS.json',status(work,all_rows));print(now(),case,done,'/',len(todo),flush=True)
                push()
    if cap[0] is not None:cap[0]-=done
    return done==len(miss)
def model_paths(work):return [work/'models'/f'R{j:02}' for j in range(1,6)]
def read_table(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def known_file_hashes(base,entries):
    base=Path(base)
    for n,h in entries.items():
        p=base/n
        need(not Path(n).is_absolute() and '..' not in Path(n).parts and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(base.resolve()) and sha(p)==h,'Missing/altered artifact: '+n)

def verify_model_inputs(work,p,rr):
    lock=get(p/'MODEL_LOCK.json')
    need('INPUT_RECEIPTS.csv' in lock['files'] and 'MODEL_SUMMARY.json' in lock['files'],'Model does not bind its training inputs')
    # Explicitly verify disk bytes, even if the scoring module has a cached model.
    known_file_hashes(p,lock['files'])
    learn.anchors(p)
    receipts=read_table(p/'INPUT_RECEIPTS.csv');lookup={r['case_id']:r for r in receipts}
    need(len(receipts)==len(rr)==len(lookup)==384 and set(lookup)=={r['case_id'] for r in rr},'Wrong training cohort: '+p.name)
    need(str(get(p/'MODEL_SUMMARY.json')['rep'])==str(rr[0]['rep']),'Wrong training replicate: '+p.name)
    for r in rr:
        ref=lookup[r['case_id']]; rec=record_ok(r,work)
        need(rec is not None and all(str(ref[k])==str(r[k]) for k in ('case_id','key_hex','index','rep','role','fault_id')) and ref['raw_sha256']==rec['outputs']['RAW.npz'],'Model input not the registered original bytes: '+r['case_id'])

def verify_models(work,code,rows):
    obj=get(work/'MODELS_LOCK.json');need(obj['design']==LOCK and obj['code']==code and obj['new_calibration_or_evaluation_seen'] is False,'Model lock binding changed')
    models=model_paths(work)
    need(set(obj['models'])=={p.name for p in models},'Incorrect model lock coverage')
    for j,p in enumerate(models,1):
        need(sha(p/'MODEL_LOCK.json')==obj['models'][p.name],'Model data lock changed')
        verify_model_inputs(work,p,[r for r in rows if r['stage']=='development' and int(r['rep'])==j])

def verify_thresholds(work,code,cal):
    t=get(work/'THRESHOLDS_LOCK.json');fp=work/'calibration/CALIBRATION.npz'
    need(t['design']==LOCK and t['code']==code and t['models_lock_sha']==sha(work/'MODELS_LOCK.json') and t['calibration_sha']==sha(fp) and t['rank']==190 and t['n']==199 and t['new_evaluation_seen'] is False,'Threshold lock binding invalid')
    with np.load(fp,allow_pickle=False) as d:
        ids=[r['case_id'] for r in cal]
        need(len(set(ids))==len(ids)==199 and list(d['case_ids'])==ids,'Wrong calibration case identity/order')
        raw_sha=d['raw_sha256'].tolist();pk=d['maxima'];h=d['thresholds']
        need(len(raw_sha)==199 and pk.shape==(199,5,4) and np.isfinite(pk).all() and h.shape==(5,4) and np.all(h>0) and np.array_equal(h,np.asarray(t['h'])),'Threshold/peak shape/values do not match')
        for i,r in enumerate(cal):
            rec=record_ok(r,work);need(rec is not None and raw_sha[i]==rec['outputs']['RAW.npz'],'Calibration inputs changed: '+r['case_id'])
        independent=np.stack([learn.anchored_limits(pk[:,j],m) for j,m in enumerate(model_paths(work))])
        need(np.array_equal(h,independent) and np.allclose(h,np.sort(pk,axis=0)[189],rtol=2e-14,atol=1e-12),'Stored thresholds no longer equal anchored rank190')
    return h
def score_chunks(rows,work,models):
    for start in range(0,len(rows),6):
        block=rows[start:start+6];a=[];b=[]
        for r in block:
            need(record_ok(r,work) is not None,'No valid raw input '+r['case_id'])
            with np.load(work/'raw'/r['case_id']/'RAW.npz',allow_pickle=False) as data:v=learn.raw_pair(data,int(r['fault_id']))
            a.append(v['native']);b.append(v['snapshot'])
        pa=np.stack(a);ci=np.stack(b)
        charts=np.stack([learn.score_batch(pa,ci,m) for m in models],axis=2)
        need(charts.shape==(len(block),960,5,4) and np.isfinite(charts).all(),'Nonfinite or malformed chart')
        yield start,charts
def same_runtime_qa(work):
    p=work/'ENGINE_QA_LOCAL.json'
    triple=[sys.version,np.__version__,platform.platform()]
    if p.exists():
        d=get(p)
        arrays={'native_values','native_xmv12','native_time_seconds','record_time_seconds','diagnostic_D_feed_temperature','diagnostic_IDV','diagnostic_rng_draw_count','controller_input_values','fast_control_input_measurements','phase','ctrl11_accumulated_integral'}
        digests={'rng_draw_count','rng_consumption_trace_sha256','full_step_trace_sha256','physical_state_sha256'}
        if (d.get('runtime')==triple and d.get('engineering_only') is True and d.get('pass') is True and d.get('duration_hours')==48 and d.get('key_hex')=='0x51414f4e4c595f5445505f524e475f33' and d.get('engine_sha')==sim.INDEX_SHA and set(d.get('exact_array_bytes',{}))==arrays and all(d['exact_array_bytes'].values()) and set(d.get('exact_full_run_digests',{}))==digests and all(d['exact_full_run_digests'].values())):
            return
    print('Verifying the complete 48h engineering reference on this runtime (not a scientific case)...',flush=True)
    if str(HERE/'simulation') not in sys.path:sys.path.insert(0,str(HERE/'simulation'))
    from engineering_check import main as test
    d=test(work/'ENGINE_QA_REPLAY.json');d.update(runtime=triple,engine_sha=sim.INDEX_SHA);put(p,d)
def verify_final(work,rows,code):
    d=get(work/'FINAL_COMPLETE.json');need(d['code']==code and d['design']==LOCK and d['n_physical']==len(rows)==2719,'Final design/software/size mismatch')
    need(d['models_lock_sha']==sha(work/'MODELS_LOCK.json') and d['thresholds_lock_sha']==sha(work/'THRESHOLDS_LOCK.json') and d['receipts_sha']==sha(work/'RAW_RECEIPTS.csv'),'Final bundle binding mismatch')
    verify_models(work,code,rows);verify_thresholds(work,code,[r for r in rows if r['stage']=='calibration'])
    records=read_table(work/'RAW_RECEIPTS.csv');need([r['case_id'] for r in records]==[r['case_id'] for r in rows],'Incomplete or reordered raw receipt manifest')
    for r,seen in zip(rows,records):
        rec=record_ok(r,work)
        need(rec is not None and all(str(seen[k])==str(r[k]) for k in ('case_id','rep','role','key_hex')) and seen['raw_sha256']==rec['outputs']['RAW.npz'] and seen['run_info_sha256']==rec['outputs']['RUN_INFO.json'] and seen['raw_receipt_sha256']==sha(work/'raw'/r['case_id']/'COMPLETE.json'),'Changed final receipt: '+r['case_id'])
    out=work/'outcomes';expected={'PER_RUN.csv','OPERATING_POINTS.csv','PAIRED_CONTRASTS.csv','LEAVE_ONE_OUT.csv','MATCHED_5PCT.csv','MATCHED_5PCT_CONTRASTS.csv','SUMMARY.json'}
    need(set(d['all_outputs'])==expected,'Incomplete or unexpected inference outputs');known_file_hashes(out,d['all_outputs'])
    summary=get(out/'SUMMARY.json');need(summary['status']=='COMPLETE' and not summary['no_scientific_claim'] and summary['protocol_sha256']==sha(HERE/'locks/PROTOCOL.json') and summary['file_sha256']=={k:v for k,v in d['all_outputs'].items() if k!='SUMMARY.json'},'Output summary not bound to numerical outcomes')
    return d

def archive_final(work,rows):
    dest=work/'F9_PAIRED_RELEARNING_COMPLETE.zip';tmp=dest.with_suffix('.part')
    docs=['DESIGN_LOCK.json','PROTOCOL.json','PREDECESSOR_IDENTITIES.json','CASE_MANIFEST.csv','CODE_LOCK.json']
    wf=[work/n for n in ['MODELS_LOCK.json','THRESHOLDS_LOCK.json','FINAL_COMPLETE.json','RAW_RECEIPTS.csv','ENGINE_QA_LOCAL.json','calibration/CALIBRATION.npz']]
    wf+=[work/'outcomes'/n for n in get(work/'FINAL_COMPLETE.json')['all_outputs']]
    for p in model_paths(work):
        wf += [p/'MODEL_LOCK.json'] + [p/n for n in get(p/'MODEL_LOCK.json')['files']]
    # Exact sealed file list only: never bundle scratch/incomplete model attempts.
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in docs:z.write(HERE/'locks'/n,'locks/'+n)
        for path in sorted(wf):z.write(path,path.relative_to(work).as_posix())
    os.replace(tmp,dest);print('Complete return ZIP:',dest,flush=True)
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--work',type=Path,default=HERE/'study');a.add_argument('--workers',type=int,default=min(8,max(1,(os.cpu_count() or 2)-1)));a.add_argument('--max-new',type=int);a.add_argument('--status',action='store_true');a.add_argument('--check',action='store_true');args=a.parse_args()
    rows=design();w=args.work.resolve();w.mkdir(exist_ok=True,parents=True)
    if args.status:print(json.dumps(status(w,rows),indent=2));return
    need(1<=args.workers<=64 and (args.max_new is None or args.max_new>=0),'Invalid resource limit')
    cap=[args.max_new]
    with sim.exclusive(w):
        sim.pinned_engine();code=verify_code();same_runtime_qa(w)
        if args.check:print('Design, implementation and full-horizon reference verified.');return
        if (w/'STOP_AFTER_BATCH').exists():print('Paused. Remove STOP_AFTER_BATCH only when resumption is intended.');return
        if (w/'FINAL_COMPLETE.json').exists():
            verify_final(w,rows,code);archive_final(w,rows);return
        models=model_paths(w)
        if not (w/'MODELS_LOCK.json').exists():
            stage_guard(w,'development',rows)
            for j,target in enumerate(models,1):
                rr=[r for r in rows if int(r['rep'])==j and r['stage']=='development']
                if not fill(rr,w,rows,args.workers,cap):print('Partial development checkpoint.');return
                if not target.exists():
                    target.parent.mkdir(exist_ok=True,parents=True);tmp=target.with_name(target.name+'.work_'+uuid.uuid4().hex[:8])
                    ins=[dict(r,raw_sha256=record_ok(r,w)['outputs']['RAW.npz']) for r in rr]
                    learn.fit_replicate(ins,w/'raw',tmp);os.replace(tmp,target)
                verify_model_inputs(w,target,rr)
                print('Training replicate locked:',j,flush=True)
            put(w/'MODELS_LOCK.json',dict(design=LOCK,code=code,utc=now(),models={p.name:sha(p/'MODEL_LOCK.json') for p in models},new_calibration_or_evaluation_seen=False))
        verify_models(w,code,rows)
        cal=[r for r in rows if r['stage']=='calibration'];cd=w/'calibration';cd.mkdir(exist_ok=True)
        if not (w/'THRESHOLDS_LOCK.json').exists():
            stage_guard(w,'calibration',rows)
            if not fill(cal,w,rows,args.workers,cap):print('Partial calibration checkpoint.');return
            pk=np.empty((199,5,4))
            for start,v in score_chunks(cal,w,models):pk[start:start+len(v)]=v.max(axis=1)
            h=np.stack([learn.anchored_limits(pk[:,j],m) for j,m in enumerate(models)])
            need(np.allclose(h,np.sort(pk,axis=0)[189],rtol=2e-14,atol=1e-12),'Rank190 check failed')
            np.savez_compressed(cd/'CALIBRATION.npz',maxima=pk,thresholds=h,case_ids=[r['case_id'] for r in cal],raw_sha256=[record_ok(r,w)['outputs']['RAW.npz'] for r in cal])
            put(w/'THRESHOLDS_LOCK.json',dict(design=LOCK,code=code,utc=now(),models_lock_sha=sha(w/'MODELS_LOCK.json'),calibration_sha=sha(cd/'CALIBRATION.npz'),h=h.tolist(),rank=190,n=199,new_evaluation_seen=False))
        h=verify_thresholds(w,code,cal)
        nr=[r for r in rows if r['role']=='normal_evaluation'];fr=[r for r in rows if r['role']=='fault_evaluation']
        if not fill(nr+fr,w,rows,args.workers,cap):print('Partial evaluation generation; no partial scientific estimates.');return
        cache=w/'chart_cache';cache.mkdir(exist_ok=True);out=w/'outcomes'
        # Preserve uncommitted outputs after interrupted inference; do not overwrite or consume them.
        if out.exists():os.replace(out,w/('outcomes_uncommitted_'+uuid.uuid4().hex[:12]))
        out.mkdir()
        for rr,role in [(nr,'normal'),(fr,'fault')]:
            target=cache/(role+'.npy');arr=np.lib.format.open_memmap(target,mode='w+',dtype='float64',shape=(len(rr),960,5,4))
            for start,v in score_chunks(rr,w,models):arr[start:start+len(v)]=v
            arr.flush();del arr
        from inference import summarize
        summarize(np.load(cache/'normal.npy',mmap_mode='r'),np.load(cache/'fault.npy',mmap_mode='r'),h,out,[r['case_id'] for r in nr],[r['case_id'] for r in fr])
        receipts=[]
        for r in rows:
            rec=record_ok(r,w);receipts.append(dict(case_id=r['case_id'],rep=r['rep'],role=r['role'],key_hex=r['key_hex'],raw_sha256=rec['outputs']['RAW.npz'],run_info_sha256=rec['outputs']['RUN_INFO.json'],raw_receipt_sha256=sha(w/'raw'/r['case_id']/'COMPLETE.json')))
        tables(w/'RAW_RECEIPTS.csv',receipts)
        put(w/'FINAL_COMPLETE.json',dict(utc=now(),design=LOCK,code=code,n_physical=len(receipts),models_lock_sha=sha(w/'MODELS_LOCK.json'),thresholds_lock_sha=sha(w/'THRESHOLDS_LOCK.json'),all_outputs={p.name:sha(p) for p in out.glob('*') if p.is_file()},receipts_sha=sha(w/'RAW_RECEIPTS.csv')))
        verify_final(w,rows,code);archive_final(w,rows)
if __name__=='__main__':main()
