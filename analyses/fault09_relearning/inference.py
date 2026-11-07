"""Locked run-level inference for the new five-replicate Fault 9 experiment.

summarize(normal_charts, fault_charts, h, out_dir, normal_ids, fault_ids)
expects chart[n,time,replicate,monitor]; monitors M1,M2,M3,M4 are in the
frozen order. Chart arguments are float arrays/memmaps or .npy file paths.
No scientific evaluation data are read on import.
"""
from pathlib import Path
import csv,hashlib,json,os,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
PROTO='a8966836eb00a3e21779c3f57ec935e17f1de6ec224a535f77faf578f1f2eb1c'
MON=('M1','M2','M3','M4'); POOLS=(('moment',0,1),('mean',2,3))
NAMES={'normal':('any_alarm','clean40'), 'fault':('clean40','clean8','clean2','prefault','loss_min')}
DISPLAY={('normal','any_alarm'):'normal_48h_FAR',('normal','clean40'):'normal_clean_floor',('fault','clean40'):'fault_clean40',('fault','clean8'):'fault_clean8',('fault','clean2'):'fault_clean2',('fault','prefault'):'fault_prefault_alarm',('fault','loss_min'):'fault_mean_penalized_loss'}
SEED=2026110406;DRAWS=20000

def _require(test,text):
    if not test:raise ValueError(text)
def _hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def _csv(path,rows):
    it=iter(rows); first=next(it)
    with path.open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(it)
def _arr(v):
    return np.load(v,mmap_mode='r',allow_pickle=False) if isinstance(v,(str,os.PathLike)) else v
def _ids(seq,n):
    seq=[str(x) for x in seq]; _require(len(seq)==len(set(seq))==n and all(seq),'Incomplete or duplicate evaluation IDs'); return seq
def _expected():
    lock=json.loads((ROOT/'locks/DESIGN_LOCK.json').read_text())
    _require(_hash(ROOT/'locks/PROTOCOL.json')==PROTO==lock['files']['PROTOCOL.json'],'Wrong frozen protocol')
    p=ROOT/'locks/CASE_MANIFEST.csv';_require(_hash(p)==lock['files']['CASE_MANIFEST.csv'],'Case manifest changed')
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return [[r['case_id'] for r in rows if r['stage']=='evaluation' and r['role']==role] for role in ('normal_evaluation','fault_evaluation')]

def _measure(source,n,h,index):
    x=_arr(source)
    _require(getattr(x,'shape',None)==(n,960,5,4) and x.dtype.kind=='f' and x.dtype.itemsize==8,'Need float64 chart array of shape (n,960,5,4), containing finite chart statistics')
    out={k:np.empty((n,5,4),dtype=np.float64 if k.endswith('max') else np.int16) for k in ('pre_max','post2_max','post8_max','post40_max','full48_max','first','first_post','prefault','any_alarm','clean40','clean8','clean2','loss_min')}
    for i in range(0,n,8):
        sl=slice(i,min(i+8,n)); t=np.asarray(x[index[sl]])
        _require(np.isfinite(t).all() and (t>=0).all(),'Bad CUSUM value: nonfinite or negative')
        pre=t[:,:160];post=t[:,160:];p=pre.max(1);f=post.max(1)
        for k,v in [('pre_max',p),('post2_max',post[:,:41].max(1)),('post8_max',post[:,:161].max(1)),('post40_max',f),('full48_max',np.maximum(p,f))]:out[k][sl]=v
        mask=t>h;post_mask=mask[:,160:];a=mask.any(1);d=post_mask.any(1);pre_alarm=p>h
        first=np.where(a,1+mask.argmax(1),0);fp=np.where(d,161+post_mask.argmax(1),0)
        clean=d&~pre_alarm;loss=np.where(clean,3*(fp-161),2400)
        for k,v in [('first',first),('first_post',fp),('prefault',pre_alarm),('any_alarm',a),('clean40',clean),('clean8',clean&(loss<=480)),('clean2',clean&(loss<=120)),('loss_min',loss)]:out[k][sl]=v
    return out

def _sensitivity(u,h):
    ok=u['pre_max']<=h
    return {'any_alarm':(u['full48_max']>h),'prefault':~ok,'clean40':ok&(u['post40_max']>h),'clean8':ok&(u['post8_max']>h),'clean2':ok&(u['post2_max']>h)}

def _unit_k(k):return ('min',1.) if k=='loss_min' else ('pp',100.)
def _direction(v):return 'positive' if v>0 else 'negative' if v<0 else 'zero'

def _impl(normal_charts,fault_charts,h,out_dir,normal_ids,fault_ids,expected,fixture):
    h=np.array(_arr(h),dtype=float)
    _require(h.shape==(5,4) and np.isfinite(h).all() and (h>0).all(),'Need 20 finite positive prelocked external limits')
    ids0,ids9=_ids(normal_ids,400),_ids(fault_ids,200)
    _require(not(set(ids0)&set(ids9)),'Overlapping role identities')
    u={};ids={};limits=np.copy(h)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    files=['PER_RUN.csv','OPERATING_POINTS.csv','PAIRED_CONTRASTS.csv','LEAVE_ONE_OUT.csv','MATCHED_5PCT.csv','MATCHED_5PCT_CONTRASTS.csv','SUMMARY.json']
    _require(not any((out/k).exists() for k in files),'Output already exists; use a fresh directory')
    for role,supplied,truth,source in zip(('normal','fault'),(ids0,ids9),expected,(normal_charts,fault_charts)):
        ids[role]=list(truth)
        _require(set(supplied)==set(truth),'Evaluation identities do not match the frozen manifest')
        pos={name:i for i,name in enumerate(supplied)};index=np.array([pos[name] for name in truth])
        u[role]=_measure(source,len(supplied),h,index)
    # These thresholds are descriptive sensitivity only; they never change original outcomes.
    nmax=u['normal']['full48_max'];target_h=np.partition(nmax,379,axis=0)[379]
    ms={role:_sensitivity(s,target_h) for role,s in u.items()}
    tie=(nmax==target_h).sum(0);nex=(nmax>target_h).sum(0)
    op=[];cs=[];ls=[];matched=[];mc=[];primary={}
    def op_rows(role,values,keys,kind,hh,collection):
        n=len(ids[role]);prefix='empirical_5pct' if kind=='sensitivity' else 'external_rank190'
        for k in keys:
            unit,scale=_unit_k(k);raw=values[k].sum(0,dtype=np.int64);av=raw*scale/n
            for rep in range(5):
                for a,mon in enumerate(MON):
                    row={'threshold_source':prefix,'replicate':rep+1,'monitor':mon,'recording':'CI' if a%2==0 else 'post','pool':'moment' if a<2 else 'mean','role':role,'endpoint':DISPLAY[role,k],'h':float(hh[rep,a]),'n_runs':n,'numerator_sum':int(raw[rep,a]),'estimate':float(av[rep,a]),'unit':'min' if unit=='min' else 'percent'}
                    if kind=='sensitivity':row.update(target_normal_crossings=20,actual_normal_crossings=int(nex[rep,a]),actual_FAR_pct=float(nex[rep,a]/4),normal_peaks_equal_to_h=int(tie[rep,a]))
                    collection.append(row)
    for role in ('normal','fault'):
        op_rows(role,u[role],NAMES[role],'primary',h,op)
        op_rows(role,ms[role],[k for k in NAMES[role] if k!='loss_min'],'sensitivity',target_h,matched)
    rng=np.random.default_rng(SEED)
    # Per protocol: independent normal indices first, fault indices second, same
    # row indices across all views and repeats. Repeats are NOT resampling units.
    draws={role:rng.integers(0,len(ids[role]),size=(DRAWS,len(ids[role])),dtype=np.int32) for role in ('normal','fault')}
    for role,keys in NAMES.items():
        n=len(ids[role]);ix=draws[role]
        for k in keys:
            name=DISPLAY[role,k]; unit,scale=_unit_k(k)
            for pool,ci,post in POOLS:
                y=u[role][k];d=y[:,:,ci].astype(np.int32)-y[:,:,post].astype(np.int32)
                sums=d.sum(0,dtype=np.int64);total=int(sums.sum());est=total*scale/(5*n);individual=sums*scale/n
                physical=d.sum(1,dtype=np.int32)
                bs=np.empty(DRAWS)
                for start in range(0,DRAWS,500):
                    end=min(start+500,DRAWS);bs[start:end]=np.sum(physical[ix[start:end]],axis=1,dtype=np.int64)*scale/(n*5)
                lo,hi=np.quantile(bs,[.025,.975],method='linear');lbl='M1-M2' if ci==0 else 'M3-M4'
                cols=dict(pool=pool,arms=lbl,endpoint=name,role=role,n_physical=n)
                dr=dict(**cols,ci_estimate=float(y[:,:,ci].sum(dtype=np.int64)*scale/(n*5)),post_estimate=float(y[:,:,post].sum(dtype=np.int64)*scale/(n*5)),ci_minus_post=float(est),contrast_unit=unit,lower95=float(lo),upper95=float(hi),n_positive=int((sums>0).sum()),n_zero=int((sums==0).sum()),n_negative=int((sums<0).sum()))
                cs.append(dr)
                if pool=='moment' and name in ('fault_clean40','normal_48h_FAR'):primary[name]=dr.copy()
                for i in range(5):
                    ls.append(dict(**cols,replicate=i+1,individual_ci_minus_post=float(individual[i]),individual_sign=_direction(sums[i]),omit_this_replicate_ci_minus_post=float((total-int(sums[i]))*scale/(n*4)),contrast_unit=unit))
                if k!='loss_min':
                    yy=ms[role][k].astype(np.int16)
                    for i in range(6):
                        rr=i if i<5 else slice(None); m=1 if i<5 else 5
                        a,b=yy[:,rr,ci].sum(dtype=np.int64),yy[:,rr,post].sum(dtype=np.int64)
                        mc.append(dict(**cols,replicate=i+1 if i<5 else 'mean_5',ci_pct=float(a*100/(n*m)),post_pct=float(b*100/(n*m)),ci_minus_post_pp=float((a-b)*100/(n*m)),ci_actual_FAR_pct=float(np.mean(nex[rr,ci])/4),post_actual_FAR_pct=float(np.mean(nex[rr,post])/4)))
    det=primary['fault_clean40']['lower95']>0;guard=primary['normal_48h_FAR']['upper95']<=5
    summary=dict(study_id='TEP_P1_F9_PAIRED_RELEARNING_INDEPENDENT_STREAMS_20261104_v1',status='SOFTWARE_QA_ONLY' if fixture else 'COMPLETE',protocol_sha256=PROTO,no_scientific_claim=bool(fixture),arms=list(MON),sample={'normal':400,'fault':200,'training_replicates':5},bootstrap=dict(draws=DRAWS,seed=SEED,unit='physical_run',order='normal then fault, separate roles, common draws across monitors and repeats',type='pointwise percentile, linear quantiles, conditional on five trained replicates and common 199 calibration runs',other_endpoints='supportive; no multiplicity adjustment'),primary_detection=primary['fault_clean40'],normal_risk=primary['normal_48h_FAR'],decision={'detection_lower95_gt_0':bool(det),'normal_risk_upper95_le_5pp':bool(guard),'supports_gain_without_material_excess_of_normal_alarms':bool(det and guard)},matched_FAR_note='380th ascending of 400 normal evaluation maxima separately per replicate/monitor; actual counts/ties reported. Post-training descriptive comparison; not an independent limit or alternative primary decision.',case_order_note='All arrays aligned to frozen manifest order before resampling; every physical case kept.')
    def events():
        for role in ('normal','fault'):
            for i,name in enumerate(ids[role]):
                for j in range(5):
                    for k,mon in enumerate(MON):
                        row=dict(case_id=name,role=role,replicate=j+1,monitor=mon,h=float(limits[j,k]))
                        row.update({v:float(a[i,j,k]) if v.endswith('max') else int(a[i,j,k]) for v,a in u[role].items()});yield row
    with tempfile.TemporaryDirectory(dir=out,prefix='.inference-') as wd:
        p=Path(wd)
        for filename,rr in zip(files[:6],[events(),op,cs,ls,matched,mc]):_csv(p/filename,rr)
        summary['file_sha256']={k:_hash(p/k) for k in files[:6]};(p/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        for k in files:os.replace(p/k,out/k)
    return summary

def summarize(normal_charts,fault_charts,h,out_dir,normal_ids,fault_ids):
    """Score fixed full evaluation panel; descriptive 5% sensitivity is kept separate.
    Each ID sequence specifies the order of its chart input. On disk .npy arrays can
    be mmap'ed. Returns the same dict written in SUMMARY.json. The orchestrator
    verifies all threshold and physical-run locks/checksums before calling.
    """
    return _impl(normal_charts,fault_charts,h,out_dir,normal_ids,fault_ids,_expected(),False)
