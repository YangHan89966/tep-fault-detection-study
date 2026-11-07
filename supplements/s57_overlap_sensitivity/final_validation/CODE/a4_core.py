"""Shared saved-statistics endpoints, exclusion and physical-run paired bootstrap."""
import hashlib,json,csv
from pathlib import Path
import numpy as np
from historical_core import write_csv,loadz
SCENARIOS=('original','exclude_test','exclude_test_and_calibration')
CONTRASTS={3:('same_mean_signed_axis_squared','mean_pool_signed'),9:('moment_pool_signed','mean_pool_signed'),15:('moment_pool_squared','mean_pool_squared')}

def keep_from_seeds(seeds):return np.asarray(seeds,dtype=np.uint64)%32!=0

def rank95(n):
 r=(95*(int(n)+1)+99)//100
 if not 1<=r<=n:raise ValueError('Insufficient calibration runs for requested empirical quantile')
 return r

def events(mask):
 a=np.asarray(mask,bool)
 pre=a[:,:160].any(1);post=a[:,160:];det=post.any(1);delay=np.where(det,post.argmax(1)*3,2400);clean=~pre&det
 return dict(pre=pre,detected=det,clean40=clean,clean8=clean&(delay<=480),clean2=clean&(delay<=120),loss=np.where(clean,delay,2400),any_alarm=a.any(1))

def paired_ci(v,key,level=1-.05/3):
 v=np.asarray(v,float)
 if v.ndim!=1 or len(v)==0:raise ValueError('one difference per physical run required')
 seed=int.from_bytes(hashlib.sha256(('A459-20261009|'+key).encode()).digest()[:16],'big');rng=np.random.default_rng(seed);means=np.empty(20000)
 for j in range(0,20000,500):means[j:j+500]=v[rng.integers(0,len(v),(500,len(v)))].mean(1)
 lo,hi=np.quantile(means,[(1-level)/2,1-(1-level)/2])
 return dict(estimate=float(v.mean()),ci_low=float(lo),ci_high=float(hi),confidence_level=level,n_physical_runs=len(v),draws=20000,bootstrap_seed=str(seed))

def summarize_one(panel,state,fault,names,sf,sn,oldh,newh,fkeep,nkeep,calkeep=None,anchors=None,peaks=None,out=None):
 rows=[];events_rows=[];mats={};oldh=np.asarray(oldh);newh=np.asarray(newh)
 assert sf.shape[1]==sn.shape[1]==960 and sf.shape[2]==sn.shape[2]==len(names)
 for scenario in SCENARIOS:
  h=newh if scenario==SCENARIOS[2] else oldh;fk=fkeep if scenario!=SCENARIOS[0] else np.ones(len(sf),bool);nk=nkeep if scenario!=SCENARIOS[0] else np.ones(len(sn),bool)
  fe=events(sf>h);ne=events(sn>h);mats[scenario]={k:fe[k][fk].copy() for k in fe}
  for j,name in enumerate(names):
   row=dict(panel=panel,state=state,fault=fault,scenario=scenario,method=str(name),fault_n=int(fk.sum()),normal_n=int(nk.sum()),threshold=float(h[j]),clean40_n=int(fe['clean40'][fk,j].sum()),clean40_pct=float(fe['clean40'][fk,j].mean()*100),clean8_n=int(fe['clean8'][fk,j].sum()),clean8_pct=float(fe['clean8'][fk,j].mean()*100),clean2_pct=float(fe['clean2'][fk,j].mean()*100),loss_min=float(fe['loss'][fk,j].mean()),normal_alarm_n=int(ne['any_alarm'][nk,j].sum()),normal_FAR_pct=float(ne['any_alarm'][nk,j].mean()*100),normal_floor_pct=float(ne['clean40'][nk,j].mean()*100),prefault_alarm_pct=float(fe['pre'][fk,j].mean()*100))
   rows.append(row)
   for role,ee,kk in [('fault',fe,fk),('normal',ne,nk)]:
    for i in np.flatnonzero(kk):events_rows.append(dict(panel=panel,state=state,fault=fault,scenario=scenario,method=str(name),role=role,run_id=int(i+1),**{k:int(v[i,j]) for k,v in ee.items()}))
 if out:
  out=Path(out);out.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(out/'SUFFICIENT_PATHS.npz',methods=np.array(names),fault_pre_max=sf[:,:160].max(1),normal_pre_max=sn[:,:160].max(1),fault_post_running_max=np.maximum.accumulate(sf[:,160:],axis=1),normal_post_running_max=np.maximum.accumulate(sn[:,160:],axis=1),old_thresholds=oldh,new_thresholds=newh,fault_keep=fkeep,normal_keep=nkeep,calibration_keep=np.asarray(calkeep) if calkeep is not None else np.empty(0,bool),anchors=np.asarray(anchors) if anchors is not None else np.ones(len(names)),calibration_peaks=np.asarray(peaks) if peaks is not None else np.empty((0,len(names))))
  write_csv(out/'ENDPOINTS.csv',rows);write_csv(out/'EVENTS.csv',events_rows)
 return rows,events_rows,mats

def aggregate(rows,eventrows,out):
 out=Path(out);summary=[];groups={}
 for r in rows:groups.setdefault((r['panel'],r['fault'],r['method'],r['scenario']),[]).append(r)
 for key,g in sorted(groups.items()):
  row=dict(zip(['panel','fault','method','scenario'],key),states=len(g),fault_n_shared=g[0]['fault_n'],normal_n_shared=g[0]['normal_n'])
  for metric in ['clean40_pct','clean8_pct','clean2_pct','loss_min','normal_FAR_pct','normal_floor_pct','prefault_alarm_pct']:
   a=np.array([r[metric] for r in g]);row[metric+'_mean']=float(a.mean());row[metric+'_min']=float(a.min());row[metric+'_max']=float(a.max())
  summary.append(row)
 write_csv(out/'STATE_ENDPOINTS.csv',rows);write_csv(out/'MEAN_ENDPOINTS.csv',summary)
 contrasts=[]
 panels=sorted(set(r['panel'] for r in eventrows));lookup={(r['panel'],r['state'],r['fault'],r['scenario'],r['method'],r['role'],r['run_id']):r for r in eventrows}
 for panel in panels:
  states=sorted(set(r['state'] for r in eventrows if r['panel']==panel))
  for fault,(a,b) in CONTRASTS.items():
   for scen in SCENARIOS:
    ids=sorted(r['run_id'] for r in eventrows if r['panel']==panel and r['state']==states[0] and r['fault']==fault and r['scenario']==scen and r['method']==a and r['role']=='fault')
    if not ids:continue
    dif=np.array([[lookup[(panel,st,fault,scen,a,'fault',i)]['loss']-lookup[(panel,st,fault,scen,b,'fault',i)]['loss'] for i in ids] for st in states]);v=dif.mean(0)
    contrasts.append(dict(panel=panel,fault=fault,contrast='C'+str((3,9,15).index(fault)+1),scenario=scen,method_a=a,method_b=b,states=len(states),states_predicted_sign=int(((dif.mean(1)>0) if fault==3 else (dif.mean(1)<0)).sum()),**paired_ci(v,panel+str(fault)+scen)))
 write_csv(out/'C1_C2_C3_PAIRED_983333.csv',contrasts)
 # Differences between scenario means; retain all adverse changes.
 maps={(r['panel'],r['fault'],r['method'],r['scenario']):r for r in summary};changes=[]
 for r in summary:
  if r['scenario']=='original':continue
  old=maps[(r['panel'],r['fault'],r['method'],'original')]
  changes.append({k:r[k] for k in ['panel','fault','method','scenario']}|{metric:r[metric]-old[metric] for metric in r if metric.endswith('_mean')})
 write_csv(out/'CHANGES_FROM_ORIGINAL.csv',changes)
 maxima={}
 for scen in SCENARIOS[1:]:
  rr=[r for r in changes if r['scenario']==scen]
  maxima[scen]={metric:max((abs(r[metric]) for r in rr),default=None) for metric in ('loss_min_mean','clean40_pct_mean','clean8_pct_mean','normal_FAR_pct_mean')}
 (out/'MAX_ABSOLUTE_METHOD_CHANGES.json').write_text(json.dumps(dict(scope='only methods and panels in this output; not every manuscript analysis',maxima=maxima),indent=2))
 return summary,contrasts
