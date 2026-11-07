"""Offline, hash-checked graded noise coupling on the independent-stream panel."""
from __future__ import annotations
import argparse, concurrent.futures, csv, datetime, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from recover_noise import recover, uniform_words
from score_core import load_historical_map, recurrence, events

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['rho_0','rho_025','rho_05','rho_075','rho_1','actual_post_update']
STUDY = 'TEP_P1_F9_GRADED_NOISE_COUPLING_20261030_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def verify_protocol(root=ROOT):
    p=json.loads((root/'PROTOCOL.json').read_text())
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    if lock['protocol_sha256']!=sha(root/'PROTOCOL.json'):raise ValueError('Protocol changed')
    if p['panel']['manifest_sha256']!=sha(root/'MANIFEST.csv'):raise ValueError('Manifest changed')
    if p['study_id']!=STUDY or p['coupling']['arms']!=ARMS:raise ValueError('Wrong study')
    cases=rows(root/'MANIFEST.csv')
    counts={r:sum(c['role']==r for c in cases) for r in ['normal_calibration','normal_evaluation','fault_evaluation']}
    if counts!={'normal_calibration':199,'normal_evaluation':400,'fault_evaluation':200}:raise ValueError('Panel mismatch')
    if any(int(c['fault_id'])!=9 for c in cases if c['role']=='fault_evaluation'):raise ValueError('Wrong fault')
    keys=[replacement_key(c['run_id']) for c in cases]
    if len(set(keys))!=799 or set(keys)&{int(c['key_hex'],16) for c in rows(root/'reference/MANIFEST.csv')}:raise ValueError('Replacement key collision')
    return p,cases

def replacement_key(run_id):
    return int.from_bytes(hashlib.sha256((STUDY+'|replacement|'+run_id).encode()).digest()[:16],'big')

def replacement_uniforms(key,event_ids):
    flat=event_ids.ravel();unique,inverse=np.unique(flat,return_inverse=True)
    table=np.zeros((len(unique),12),dtype=np.float64)
    for i,event in enumerate(unique):
        if event>=0:table[i]=uniform_words(key,int(event),12)
    return table[inverse].reshape(event_ids.shape+(12,))

def sequential_noise(uniforms,sd,mask):
    total=np.zeros(uniforms.shape[:-1],dtype=np.float64)
    for j in range(12):total+=uniforms[...,j]
    out=(total-6.)*sd
    out[~mask]=0.
    return out

def make_scores(raw,recovered,fixed_map,key):
    original=recovered['original_uniforms'];mask=recovered['has_noise'];sd=recovered['noise_sd']
    replacement=replacement_uniforms(key,recovered['noise_event_id'])
    old_noise=sequential_noise(original,sd,mask)
    if not np.array_equal(old_noise,recovered['input_noise']):raise ValueError('Original noise transform differs')
    snapshot=raw['native_values'].copy();snapshot[:,:41]=raw['fast_control_input_measurements']
    scores=np.empty((960,6),dtype=np.float64)
    moment_sums=np.zeros((5,41,6),dtype=np.float64)
    beta,intercept=fixed_map['beta'],fixed_map['intercept']
    for ai,m in enumerate([0,3,6,9,12]):
        mixed=np.concatenate((original[...,:m],replacement[...,m:]),axis=-1)
        new_noise=sequential_noise(mixed,sd,mask)
        view=snapshot if m==12 else snapshot.copy()
        if m!=12:view[:,:41]+=new_noise-old_noise
        scores[:,ai]=view@beta+intercept
        # Unique measurement events, not repetitions of held analyzer values.
        for j in range(41):
            ids=recovered['noise_event_id'][:,j]
            _,idx=np.unique(ids,return_index=True);idx=idx[ids[idx]>=0]
            a=old_noise[idx,j]/sd[j];b=new_noise[idx,j]/sd[j]
            moment_sums[ai,j]=[len(idx),a.sum(),b.sum(),a@a,b@b,a@b]
    scores[:,5]=raw['native_values']@beta+intercept
    if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
    return scores,moment_sums

def load_source(folder,case,file_index):
    hashes={}
    for name in ['RAW.npz','RUN_INFO.json','CHARTS.npz']:
        path=folder/name; rel='runs/'+case['run_id']+'/'+name
        rec=file_index[rel]
        if path.is_symlink() or path.stat().st_size!=int(rec['size_bytes']) or sha(path)!=rec['sha256']:raise ValueError('Source integrity failure: '+rel)
        hashes[name]=rec['sha256']
    meta=json.loads((folder/'RUN_INFO.json').read_text())
    if int(meta['rng_key_hex'],16)!=int(case['key_hex'],16) or int(meta['fault_id'])!=int(case['fault_id']):raise ValueError('Run identity differs')
    with np.load(folder/'RAW.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    with np.load(folder/'CHARTS.npz',allow_pickle=False) as z:old_chart=z['charts'][:,[21,20]]
    return raw,meta,old_chart,hashes

def job(arguments):
    root,input_root,case,threshold_hash=arguments;root=Path(root);input_root=Path(input_root)
    started=time.perf_counter();run_id=case['run_id'];dest=root/'scores'/f'{run_id}.npz'
    report_path=root/'scores'/f'{run_id}.json'
    code_lock=sha(root/'CODE_LOCK.json');protocol_hash=sha(root/'PROTOCOL.json')
    if case['role']!='normal_calibration':
        lock_file=root/'results/THRESHOLDS_LOCK.json'
        if threshold_hash is None or not lock_file.is_file() or sha(lock_file)!=threshold_hash:raise ValueError('Verified calibration lock changed or missing')
    if dest.exists() and report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous['score_sha256']==sha(dest) and previous['code_lock_sha256']==code_lock and previous['protocol_sha256']==protocol_hash and previous['threshold_lock_sha256_at_generation']==threshold_hash:return previous
        raise ValueError('Completed output identity changed: '+run_id)
    index={x['path']:x for x in rows(root/'reference/FILE_MANIFEST.csv')}
    raw,meta,old_chart,hashes=load_source(input_root/'runs'/run_id,case,index)
    recovered=recover(raw,meta,verify_full_rng_trace=True)
    if recovered['diagnostics']['temperature_max_abs_error']>1e-12:raise ValueError('Temperature recovery outside frozen tolerance')
    scores,moments=make_scores(raw,recovered,load_historical_map(root/'assets/FIXED_AFFINE_MAPS.npz'),replacement_key(run_id))
    endpoints=recurrence(scores[:,[4,5]][None])[0]
    # Exact alarm equivalence is mandatory; no widening threshold tolerance.
    old_lock=json.loads((root/'reference/THRESHOLDS_LOCK.json').read_text())
    old_h=np.asarray(old_lock['thresholds'])[[21,20]]
    delta=float(np.max(np.abs(endpoints-old_chart)))
    if delta>1e-9 or not np.array_equal(endpoints>old_h,old_chart>old_h):raise ValueError('Original endpoint chart replay failed')
    ee=events(endpoints,old_h)
    lookup={(r['run_id'],r['arm_id']):r for r in rows(root/'reference/per_run_events.csv') if r['run_id']==run_id}
    if case['role']!='normal_calibration':
        for j,view in enumerate(['controller_input','native']):
            ref=lookup[(run_id,'F9_historical_'+view)]
            for new,old in [('normal_alarm','normal_alarm'),('pre_alarm','prefault_alarm'),('clean','clean40'),('clean_120','clean2'),('clean_480','clean8'),('delay_minutes','delay_minutes'),('loss_minutes','loss_minutes')]:
                if int(ee[new][j])!=int(ref[old]):raise ValueError('Original endpoint event differs: '+run_id+'/'+old)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,scores=scores,arm_names=np.array(ARMS),noise_moment_sums=moments,original_endpoint_peaks=endpoints.max(axis=0))
    os.replace(tmp,dest)
    report={'run_id':run_id,'role':case['role'],'fault_id':int(case['fault_id']),'status':'COMPLETE',
      'protocol_sha256':protocol_hash,'code_lock_sha256':code_lock,'source_hashes':hashes,
      'threshold_lock_sha256_at_generation':threshold_hash,
      'original_key_hex':case['key_hex'],'replacement_key_hex':f'0x{replacement_key(run_id):032x}',
      'score_sha256':sha(dest),'endpoint_chart_max_abs_error':delta,'original_alarm_bits_checked':1920,
      'original_events_match':case['role']!='normal_calibration','noise_recovery':recovered['diagnostics'],
      'wall_seconds':time.perf_counter()-started}
    save_json(report_path,report);return report

def verify_code_lock(root):
    lock=json.loads((root/'CODE_LOCK.json').read_text())
    for name,h in lock['files'].items():
        if sha(root/name)!=h:raise ValueError('Code changed since lock: '+name)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input-root',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--stage',choices=['calibration','evaluation'],required=True);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();root=args.root.resolve();p,cases=verify_protocol(root);verify_code_lock(root)
    if not (root/'checks/FULL48_NOISE_CAPTURE_VALIDATION.json').is_file():raise ValueError('Independent48h noise capture validation required')
    if json.loads((root/'checks/FULL48_NOISE_CAPTURE_VALIDATION.json').read_text())['status']!='PASS':raise ValueError('Noise capture validation not passed')
    threshold_hash=None
    if args.stage=='evaluation':
        from analyze_coupling import verify_lock
        verify_lock(root)
        threshold_hash=sha(root/'results/THRESHOLDS_LOCK.json')
    selected=[c for c in cases if (c['role']=='normal_calibration')==(args.stage=='calibration')]
    start=time.perf_counter();reports=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(job,(str(root),str(args.input_root.resolve()),c,threshold_hash)) for c in selected]
        for future in concurrent.futures.as_completed(futures):
            report=future.result();reports.append(report)
            if len(reports)%10==0 or len(reports)==len(selected):print(f'{args.stage}: {len(reports)}/{len(selected)} completed; {time.perf_counter()-start:.1f}s',flush=True)
    save_json(root/'checks'/f'{args.stage.upper()}_REPLAY.json',{'status':'PASS','runs':len(reports),'records_checked':sum(x['noise_recovery']['record_count_checks'] for x in reports),'alarm_bits_checked':sum(x['original_alarm_bits_checked'] for x in reports),'max_endpoint_chart_error':max(x['endpoint_chart_max_abs_error'] for x in reports),'protocol_sha256':sha(root/'PROTOCOL.json'),'wall_seconds':time.perf_counter()-start})
    print('STAGE COMPLETE',args.stage,flush=True)

if __name__=='__main__':main()
