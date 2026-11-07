"""Published risk/time aggregation functions extracted unchanged from the supplied runner.
See provenance/SOURCE_EXTRACTION.json for source identity and line ranges.
"""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import frozen_core as core

METHODS = ('mean_pool_squared', 'moment_pool_squared')

def putj(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def putcsv(p,rows):
    rows=iter(rows)
    try:first=next(rows)
    except StopIteration:return
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)


def putnpz(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)


@lru_cache(maxsize=256)
def cp(k,n=250):return core.clopper_pearson(int(k),n)


def event_metrics(mask,normal):
    # mask and normal: run x time x method. No latching or onset reset.
    ev=core.events(mask);clean=ev['clean_detection'].astype(bool);delay=3*ev['first_delay_samples_penalized']
    vals={'normal_FAR_pct':normal.any(1).astype(float)*100,
          'prefault_alarm_pct':ev['prefault_alarm'].astype(float)*100,
          'occupancy_pct':ev['occupancy_points'].astype(float)/8}
    for H in (60,120,480,2400):vals[f'clean_by_{H}min_pct']=(clean&(delay<=H)).astype(float)*100
    for H in (120,480,2400):vals[f'loss_{H}min']=np.where(clean&(delay<=H),delay,H).astype(float)
    return ev,vals


def bootstrap_differences(d,protocol,role):
    # d rows retain paired method differences and all rank/endpoint columns.
    seed=int.from_bytes(hashlib.sha256((protocol['id']+'|'+role).encode('ascii')).digest()[:8],'big')
    rng=np.random.Generator(np.random.PCG64(seed));n=len(d);out=np.empty((20000,d.shape[1]))
    for lo in range(0,20000,100):
        ix=rng.integers(0,n,size=(100,n),dtype=np.int64)
        weights=np.bincount((ix+n*np.arange(100)[:,None]).ravel(),minlength=100*n).reshape(100,n)
        out[lo:lo+100]=(weights@d)/n
    return np.quantile(out,[.025,.975],axis=0,method='linear'),seed


def sensitivity(scores,nscores,limits,ranks,protocol,out,log):
    rows=[];event_rows=[];normal_rows=[];curve_rows=[];diffs={};vals_by_rank=[]
    for rank,h in zip(ranks,limits):
        mask=scores>h;normal=nscores>h;ev,vals=event_metrics(mask,normal)
        vals_by_rank.append(vals)
        for key,v in vals.items():diffs.setdefault(key,[]).append(v[:,1]-v[:,0])
        for j,m in enumerate(METHODS):
            row={'rank':int(rank),'method':m,'k':.05,'threshold':float(h[j]),'rank_reference_probability':(260-int(rank))/260,'normal_n':250,'fault_n':250}
            for key,v in vals.items():
                row[key]=float(v[:,j].mean())
                if key.endswith('_pct') and key!='occupancy_pct':
                    k=int(np.count_nonzero(v[:,j]));a,b=cp(k)
                    row[key+'_count']=k;row[key+'_CP95_low']=100*a;row[key+'_CP95_high']=100*b
            rows.append(row)
            clean=ev['clean_detection'][:,j].astype(bool);delay=3*ev['first_delay_samples_penalized'][:,j]
            for i in range(250):
                e={'rank':int(rank),'method':m,'run_id':i+1}
                e.update({k:int(v[i,j]) for k,v in ev.items()})
                e.update({f'loss_{H}min':float(vals[f'loss_{H}min'][i,j]) for H in (120,480,2400)})
                event_rows.append(e)
                nm=normal[i,:,j]
                normal_rows.append({'rank':int(rank),'method':m,'run_id':i+1,'alarm':int(nm.any()),'above_points':int(nm.sum())})
            for t in range(0,2401,3):
                k=int((clean&(delay<=t)).sum())
                curve_rows.append({'rank':int(rank),'method':m,'minutes_after_onset':t,'clean_count':k,'n':250,'clean_pct':100*k/250})
    putcsv(out/'RISK_TIME_ENDPOINTS.csv',rows)
    putcsv(out/'FAULT_PER_RUN.csv',event_rows);putcsv(out/'NORMAL_PER_RUN.csv',normal_rows)
    putcsv(out/'CLEAN_DETECTION_CURVES.csv',curve_rows)
    log.msg('全部 26 个校准秩已保留；计算运行级配对区间，不选择“最佳点”。')
    # All ranks share each role's bootstrap schedule. Fault and normal schedules are distinct.
    contrast_rows=[];seed_report={}
    for role,keys in [('normal',['normal_FAR_pct']),('fault',[k for k in diffs if k!='normal_FAR_pct'])]:
        matrix=np.concatenate([np.stack(diffs[k],axis=1) for k in keys],axis=1)
        qs,seed=bootstrap_differences(matrix,protocol,role);seed_report[role]=seed
        for ki,key in enumerate(keys):
            for ri,rank in enumerate(ranks):
                d=matrix[:,ki*len(ranks)+ri];c=ki*len(ranks)+ri
                contrast_rows.append({'rank':int(rank),'contrast':'moment_pool_squared_minus_mean_pool_squared',
                  'metric':key,'estimate':float(d.mean()),'pointwise95_low':float(qs[0,c]),'pointwise95_high':float(qs[1,c]),
                  'bootstrap_draws':20000,'empirically_degenerate':bool(np.all(d==d[0])),
                  'same_calibration_rank_not_equal_true_FAR':True,'exploratory_not_simultaneous':True})
    putcsv(out/'PAIRED_RANK_CONTRASTS.csv',contrast_rows);putj(out/'BOOTSTRAP_SPEC.json',{'seeds':seed_report,'protocol':protocol['intervals']})
    # Save continuous score paths, not raw process variables; permits exact threshold reaggregation.
    putnpz(out/'SCORES_FIXED_F15.npz',fault=scores,normal=nscores,methods=np.array(METHODS),run_ids=np.arange(1,251),thresholds=limits,ranks=np.array(ranks))
    return len(rows),len(contrast_rows)


