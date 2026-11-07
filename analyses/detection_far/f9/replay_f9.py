"""Replay the same fixed historical Fault 9 score, under the two observed recordings.

Usage: python replay_f9.py CONFIRMATION_ARCHIVE_OR_EXTRACTED_DIR NUMERICAL_EVIDENCE_ZIP OUTPUT_DIR
Needs NumPy; never fits a model or runs a simulator. Strict > h, no restart at fault onset.
"""
import csv, hashlib, io, json, sys, zipfile
from pathlib import Path
from collections import Counter
import numpy as np

def ensure(ok,message):
    if not ok: raise ValueError(message)

def csvrows(b): return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(b): return hashlib.sha256(b).hexdigest()

class Archive:
    def __init__(self,path,marker,panel):
        if path.is_dir():
            self.base=next(path.rglob(marker)).parent; self.raw=lambda s:(self.base/s).read_bytes()
        else:
            self.z=zipfile.ZipFile(path);self.prefix=next(s[:-len(marker)] for s in self.z.namelist() if s.endswith('/'+marker)); self.raw=lambda s:self.z.read(self.prefix+s)
        self.panel=panel; self.ledger={};raw=self.raw(marker);self.hashes={r['path']:r['sha256'] for r in json.loads(raw)['files']}; self.ledger[marker]=sha(raw)
    def read(self,s):
        raw=self.raw(s);digest=sha(raw);ensure(digest==self.hashes[s], 'Input digest mismatch: '+s);self.ledger[s]=digest;return raw
    def rows(self,s):return csvrows(self.read(s))

def peaks(s,panel):
    # Two-sided per-run historical recursion in its original operation order.
    n,T,d=s.shape;ensure(T==960 and d==2 and s.dtype==np.float64 and np.isfinite(s).all(),'Invalid scores')
    cp=np.zeros((n,d));cm=cp.copy();chart=np.empty_like(s)
    for t in range(T):
        cp=np.maximum(0.,(cp+s[:,t])-0.05);cm=np.maximum(0.,(cm-s[:,t])-0.05)
        chart[:,t]=np.maximum(cp,cm)
    if panel=='recording_panel_2':
        # Equivalent partial-sum formulation gives bit-for-bit panel-2 rank 190.
        def reflected(scores):
            total=np.cumsum(scores-.05,axis=1)
            return total-np.minimum(0.,np.minimum.accumulate(total,axis=1))
        ref=np.maximum(reflected(s),reflected(-s))
        np.testing.assert_allclose(ref,chart,rtol=1e-11,atol=1e-11);chart=ref
    return dict(all48_max=chart.max(1),pre_max=chart[:,:160].max(1),post2_max=chart[:,160:201].max(1),post8_max=chart[:,160:321].max(1),post40_max=chart[:,160:].max(1))

def replay(a,rows,arm_names,indices,hs,truth,targets,calref=None):
    ss=[];keys=[]
    for role,rid,path in rows:
        with np.load(io.BytesIO(a.read(path)),allow_pickle=False) as z:
            cols=z['arm_names' if a.panel=='confirmation' else 'views'].tolist()
            ensure([cols[j] for j in indices]==arm_names,'View identity differs: '+rid);ss.append(z['scores'][:,indices]);keys.append((role,rid))
    ensure(len(keys)==len(set(keys)),'Duplicate runs')
    counts=Counter(x for x,_ in keys);ensure(counts==dict(normal_calibration=199,normal_evaluation=400 if a.panel=='confirmation' else 120,fault_evaluation=200 if a.panel=='confirmation' else 120),'Panel mismatch')
    v=peaks(np.stack(ss),a.panel); out=[];summ=[]
    role=np.array([k[0] for k in keys]);cal=role=='normal_calibration';norm=role=='normal_evaluation';fault=role=='fault_evaluation'
    for j,(record,h) in enumerate(zip(['post_update','controller_input'],hs)):
        raw_cal=np.sort(v['all48_max'][cal,j]);ensure(raw_cal[189]==h,'Rank 190 is not the archived threshold')
        if calref is not None:np.testing.assert_array_equal(v['all48_max'][cal,j],calref[:,j])
        pre=v['pre_max'][:,j]>h;alarm=v['all48_max'][:,j]>h
        clean={t:(~pre)&(v[f'post{t}_max'][:,j]>h) for t in [2,8,40]}
        sums={'normal_alarm_n':int(alarm[norm].sum()),'pre_alarm_n':int(pre[fault].sum()),**{f'clean{t}_n':int(x[fault].sum()) for t,x in clean.items()}}
        ensure(sums==targets[j],f'Summary mismatch {a.panel} {record} {sums}')
        checked=0
        for i,(r,id_) in enumerate(keys):
            if r!='normal_calibration':
                true=truth[(id_,record)]
                vals=[int(alarm[i])] if r=='normal_evaluation' else [int(pre[i]),int(clean[2][i]),int(clean[8][i]),int(clean[40][i])]
                ensure(true==vals,f'Event mismatch {a.panel} {record} {id_}: {true}/{vals}'); checked+=1
            row=dict(panel=a.panel,model='fixed_historical_signed_CUSUM',recording=record,source_arm=arm_names[j],role=r,run_id=id_,original_h=h)
            row.update({k:x[i,j] for k,x in v.items()});row['normal48_max']=v['all48_max'][i,j] if not fault[i] else '';out.append(row)
        summ.append(dict(panel=a.panel,recording=record,original_h=h,replayed_calibration_rank190=raw_cal[189],n_cal=199,n_normal=int(norm.sum()),n_fault=int(fault.sum()),**sums,n_original_evaluation_runs_checked=checked))
    return out,summ

def main(conf,evid,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    c=Archive(conf,'RESULT_PACKAGE_MANIFEST.json','confirmation')
    e=Archive(evid,'FILE_MANIFEST.json','recording_panel_2');base='data/fault9_validation/'
    outcomes=[];summary=[]
    rows=[(r['role'],r['run_id'],'scores/'+r['run_id']+'.npz') for r in c.rows('MANIFEST.csv')]
    arms=['F9_historical_native','F9_historical_controller_input'];rec=['post_update','controller_input']
    data={r['arm_id']:r for r in c.rows('reference/arm_summary.csv')};hs=[float(data[a]['threshold']) for a in arms]
    targets=[{k:int(data[a][t]) for k,t in [('normal_alarm_n','normal_alarm_count'),('pre_alarm_n','fault_prefault_alarm_count'),('clean2_n','clean2_count'),('clean8_n','clean8_count'),('clean40_n','clean40_count')]} for a in arms]
    truth={}
    for r in c.rows('reference/per_run_events.csv'):
        if r['arm_id'] in arms:
            rs=['normal_alarm'] if r['role']=='normal_evaluation' else ['prefault_alarm','clean2','clean8','clean40']
            truth[r['run_id'],rec[arms.index(r['arm_id'])]]=[int(r[k]) for k in rs]
    with np.load(io.BytesIO(c.read('reference/CALIBRATION_MAXIMA.npz')),allow_pickle=False) as z:calref=z['maxima'][:,[20,21]]
    am=json.loads(c.read('reference/ARM_METADATA.json'));am={r['arm_id']:r for r in am if r['arm_id'] in arms}
    for j,a in enumerate(arms): ensure(am[a]['column']==20+j and am[a]['k']==.05 and am[a]['choice_key']=='H_D01_moment_A01_signed' and am[a]['map_sha256']==e.hashes[base+'assets/FIXED_AFFINE_MAPS.npz'],'Original model mismatch')
    e.read(base+'assets/FIXED_AFFINE_MAPS.npz')
    r,s=replay(c,rows,['actual_post_update','rho_1'],[5,4],hs,truth,targets,calref);outcomes.extend(r);summary.extend(s)
    rows=[]
    for p in sorted(e.hashes):
        if p.startswith(base+'cases/') and p.endswith('/SCORES.npz'):
            rid=p.split('/')[-2];rr=rid.rsplit('_',1)[0].replace('fault9_evaluation','fault_evaluation');rows.append((rr,rid,p))
    names=['native','all_fast_input'];data={r['view']:r for r in e.rows(base+'results/MAIN_RESULTS.csv') if r['method']=='Signed_CUSUM_reference'}
    hs=[float(data[a]['threshold']) for a in names]
    targets=[{k:int(data[a][t]) for k,t in [('normal_alarm_n','normal_alarm_n'),('pre_alarm_n','pre_alarm_n'),('clean2_n','clean_120_n'),('clean8_n','clean_480_n'),('clean40_n','clean_2400_n')]} for a in names]
    truth={}
    for r in e.rows(base+'results/ALL_RUN_OUTCOMES.csv'):
        if r['method']=='Signed_CUSUM_reference':
            rs=['normal_alarm'] if r['role']=='normal_evaluation' else ['pre_alarm','clean_120','clean_480','clean_2400']
            truth[r['case_id'],rec[names.index(r['view'])]]=[int(r[k]) for k in rs]
    r,s=replay(e,rows,names,[0,1],hs,truth,targets);outcomes.extend(r);summary.extend(s)
    writecsv(out_dir/'F9_RUN_MAXIMA.csv',outcomes);writecsv(out_dir/'F9_ORIGINAL_POINTS.csv',summary)
    writecsv(out_dir/'F9_SOURCES.csv',[dict(panel=a.panel,member=p,sha256=d) for a in [c,e] for p,d in sorted(a.ledger.items())])
    print('Verified original thresholds, all run-level endpoints and aggregate counts for both panels and views.')
if __name__=='__main__':main(*(Path(x) for x in sys.argv[1:]))
