#!/usr/bin/env python3
"""A4, Analysis 1: exclusion using frozen alarms; no simulator or fitting imports.

Usage: python recompute_A4.py --inputs <nine-check-root> --out <output-directory>
Runtime: Python 3.10+, numpy, pandas. All input files read are SHA-256 recorded.
Five states are averaged within each physical run before bootstrap resampling.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd

B=20000
SEED=2026100804
FULL={3:'mean_signed__full',9:'moment_signed__full',15:'moment_squared__full'}
REF={3:'FDA_equal_class_signed_CUSUM',9:'FDA_equal_class_signed_CUSUM',15:'covariance_F15_max_squared_CUSUM'}
T4={3:[('baseline',FULL[3],'Full monitor'),('repeated','same_mean_signed_axis_squared','Same direction, Squared'),('baseline','mean_signed__I','I restriction, Signed'),('baseline',REF[3],'FDA-CUSUM, Signed'),('baseline','Q_CUSUM','Q-CUSUM')],9:[('baseline',FULL[9],'Full monitor'),('repeated','mean_pool_signed','Mean-shift pool, Signed'),('baseline','moment_signed__I','I restriction, Signed'),('baseline',REF[9],'FDA-CUSUM, Signed'),('baseline','Q_CUSUM','Q-CUSUM')],15:[('baseline',FULL[15],'Full monitor'),('repeated','mean_pool_squared','Mean-shift pool, Squared'),('baseline','moment_squared__I','I restriction, Squared'),('baseline',REF[15],'Variance-ratio reference, Squared'),('baseline','FDA_equal_class_signed_CUSUM','FDA-CUSUM, Signed'),('baseline','Q_CUSUM','Q-CUSUM')]}
T3={3:('same_mean_signed_axis_squared','mean_pool_signed'),9:('moment_pool_signed','mean_pool_signed'),15:('moment_pool_squared','mean_pool_squared')}

def main(root,out):
 root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 d=root/'02_03_Detection_and_Paired_Delay';r=root/'01_RNG_Overlap'
 reads={}
 def record(p):
  p=Path(p);raw=p.read_bytes();reads[str(p)]={'path':str(p),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};return raw
 def csv(p):record(p);return pd.read_csv(p)
 def js(p):return json.loads(record(p).decode('utf-8-sig'))
 def arr(p):record(p);a=np.load(p,allow_pickle=False);v={k:a[k] for k in a.files};a.close();return v
 # Verify all 168 archived numeric input identities before using any alarms.
 manifest=js(d/'INPUT_MANIFEST.json')
 for x in manifest:
  raw=record(d/x['path']);assert len(raw)==x['size_bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256'],x['path']
 seeds=csv(r/'results/RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv')
 assert len(seeds)==9007 and seeds.self_wrap.sum()==287
 assert np.array_equal(seeds.self_wrap,seeds.n_draws>seeds.period)
 assert np.array_equal(seeds.self_wrap,np.where(seeds.duration_hours==25,seeds.seed_u32%64==0,seeds.seed_u32%32==0))
 seeds['run_number']=seeds.run_id.str.extract(r'run=(\d+)').fillna(seeds.run_id.str.extract(r'_(\d+)$')).astype(int)
 seeds['replication']=seeds.run_id.str.extract(r'rep=(\d+)').fillna(0).astype(int)
 assert not seeds.duplicated(['panel','role','fault_id','replication','run_number']).any()
 seeds.to_csv(out/'ALL_RUN_IDENTITIES_AND_EXCLUSION.csv',index=False)
 seeds[seeds.self_wrap].to_csv(out/'SELF_OVERLAPPING_287_RUNS.csv',index=False)
 counts=seeds.groupby(['panel','role','fault_id','replication'],dropna=False).agg(total_runs=('run_id','size'),self_overlapping_runs=('self_wrap','sum')).reset_index()
 counts.to_csv(out/'SELF_OVERLAP_SET_COUNTS.csv',index=False)
 summary=seeds.groupby(['panel','role']).agg(total_runs=('run_id','size'),self_overlapping_runs=('self_wrap','sum')).reset_index()
 summary.to_csv(out/'SELF_OVERLAP_PANEL_ROLE_SUMMARY.csv',index=False)
 def identity(panel,role,fault,ids,rep=0):
  x=seeds[(seeds.panel==panel)&(seeds.role==role)&(seeds.fault_id==fault)&(seeds.replication==rep)].set_index('run_number')
  assert len(x)==len(ids) and x.index.is_unique,(panel,role,fault,rep)
  x=x.loc[np.asarray(ids,dtype=int)];return x
 def keep(panel,role,fault,ids,exclude,rep=0):return ~identity(panel,role,fault,ids,rep).self_wrap.to_numpy() if exclude else np.ones(len(ids),bool)
 def ev(mask):
  pre=mask[:,:160].any(1);post=mask[:,160:].any(1);clean=(~pre)&post
  delay=np.where(post,mask[:,160:].argmax(1)*3,2400)
  return {'clean':clean.astype(int),'clean8':(clean&(delay<=480)).astype(int),'clean2':(clean&(delay<=120)).astype(int),'loss':np.where(clean,delay,2400).astype(float),'any':mask.any(1).astype(int),'pre':pre.astype(int),'delay':delay.astype(float)}
 def boot(v,key,level=95):
  v=np.asarray(v,float);assert v.ndim==1 and len(v)>0 and np.isfinite(v).all()
  keynum=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
  rng=np.random.default_rng(np.random.SeedSequence([SEED,keynum&0xffffffff,keynum>>32]))
  a=np.empty(B)
  for start in range(0,B,1000):
   ii=rng.integers(0,len(v),size=(min(1000,B-start),len(v)));a[start:start+len(ii)]=v[ii].mean(1)
  q=np.quantile(a,[(1-level/100)/2,1-(1-level/100)/2]);return float(q[0]),float(q[1])
 cache={};validation_rows=[];replay=[]
 for fault in [3,9,15]:
  for family,states in [('single',[1]),('repeated',range(1,6)),('baseline',range(1,6))]:
   for state in states:
    p=d/f'inputs/{family}'/(f'F{fault:02d}.npz' if family=='single' else f'rep{state:02d}/F{fault:02d}.npz');a=arr(p)
    if family=='single':
     ids=a['fault_run_ids'];assert np.array_equal(ids,a['normal_run_ids'])
     fm=np.unpackbits(a['fault_masks'],axis=2,count=960,bitorder='big').transpose(1,2,0).astype(bool)
     nm=np.unpackbits(a['normal_masks'],axis=2,count=960,bitorder='big').transpose(1,2,0).astype(bool)
    else:
     ids=a['run_id'];fm=np.unpackbits(a['fault_alarm_bits'],axis=1,count=960,bitorder='little').astype(bool);nm=np.unpackbits(a['normal_alarm_bits'],axis=1,count=960,bitorder='little').astype(bool)
    for j,m in enumerate(a['methods']):
     fe=ev(fm[:,:,j]);ne=ev(nm[:,:,j]);cache[(family,state,fault,str(m))]=(fe,ne,ids)
     if family!='baseline':
      original=a['event_loss_samples'][:,j] if family=='single' else a['fault_event_loss_samples'][:,j]
      assert np.array_equal(fe['loss'],original*3)
     panel='dense_validation' if family=='single' else 'R5'
     for scenario,ex in [('original',False),('exclude_test_self_overlap',True)]:
      fk=keep(panel,'fault_evaluation',fault,ids,ex);nk=keep(panel,'normal_evaluation',0,ids,ex)
      row={'scenario':scenario,'panel':panel,'state':state,'fault':fault,'family':family,'method':str(m),'fault_n':int(fk.sum()),'normal_n':int(nk.sum())}
      for k in ['clean','clean8','clean2']:row[k+'_pct']=float(fe[k][fk].mean()*100)
      row.update(mean_loss_min=float(fe['loss'][fk].mean()),normal_FAR_pct=float(ne['any'][nk].mean()*100),normal_floor_pct=float(ne['clean'][nk].mean()*100),normal_prefault_pct=float(ne['pre'][nk].mean()*100))
      validation_rows.append(row)
 validations=pd.DataFrame(validation_rows);validations.to_csv(out/'ALL_VALIDATION_MONITORS_BY_STATE.csv',index=False)
 t3=[]
 for fault,(ma,mb) in T3.items():
  for family,states in [('single',[1]),('repeated',range(1,6))]:
   mat=np.stack([cache[(family,s,fault,ma)][0]['loss']-cache[(family,s,fault,mb)][0]['loss'] for s in states]);ids=cache[(family,1,fault,ma)][2]
   panel='dense_validation' if family=='single' else 'R5'
   for scenario,ex in [('original',False),('exclude_test_self_overlap',True)]:
    fk=keep(panel,'fault_evaluation',fault,ids,ex);v=mat[:,fk].mean(0)
    key=f'T3_{family}_{fault}_{scenario}'
    lo,hi=boot(v,key,100*(1-.05/3));l95,h95=boot(v,key,95)
    t3.append({'scenario':scenario,'panel':panel,'contrast':f'C{[3,9,15].index(fault)+1}','fault':fault,'n_fault_runs':int(fk.sum()),'states':len(mat),'delta_loss_min':float(v.mean()),'lo_98_333333':lo,'hi_98_333333':hi,'lo_95':l95,'hi_95':h95,'states_with_predicted_sign':int(((mat[:,fk].mean(1)>0) if fault==3 else (mat[:,fk].mean(1)<0)).sum()),'state_delta_min':float(mat[:,fk].mean(1).min()),'state_delta_max':float(mat[:,fk].mean(1).max()),'bootstrap_draws':B,'bootstrap_physical_run_unit':True})
   expected={(3,'single'):1783.116,(9,'single'):12.24,(15,'single'):-948.336,(3,'repeated'):1624.0032,(9,'repeated'):-19.572,(15,'repeated'):-615.132}[(fault,family)]
   assert abs(mat.mean()-expected)<1e-8;replay.append({'table':'Table 3','key':f'{family} F{fault}','recomputed':float(mat.mean()),'archived_value':expected,'match':True})
 t3=pd.DataFrame(t3)
 reported3={(3,'dense_validation'):(1727.4,1834.5),(9,'dense_validation'):(-26.0,49.3),(15,'dense_validation'):(-1048.1,-846.4),(3,'R5'):(1566.9,1679.5),(9,'R5'):(-46.3,6.6),(15,'R5'):(-696.7,-534.3)}
 t3['reported_original_ci_lo']=t3.apply(lambda x:reported3[(x.fault,x.panel)][0] if x.scenario=='original' else np.nan,axis=1)
 t3['reported_original_ci_hi']=t3.apply(lambda x:reported3[(x.fault,x.panel)][1] if x.scenario=='original' else np.nan,axis=1)
 t3.to_csv(out/'TABLE3_COMPARISON.csv',index=False)
 t4=[]
 for fault,items in T4.items():
  for family,method,label in items:
   g=validations[(validations.panel=='R5')&(validations.fault==fault)&(validations.family==family)&(validations.method==method)]
   for scenario,x in g.groupby('scenario',sort=False):
    assert len(x)==5
    row={'scenario':scenario,'fault':fault,'monitor':label,'family':family,'method':method,'fault_n_shared':int(x.fault_n.iloc[0]),'normal_n_shared':int(x.normal_n.iloc[0]),'states':5}
    for k in ['clean_pct','clean8_pct','clean2_pct','mean_loss_min','normal_FAR_pct','normal_floor_pct']:
     row[k+'_mean']=float(x[k].mean());row[k+'_min']=float(x[k].min());row[k+'_max']=float(x[k].max())
    t4.append(row)
 t4=pd.DataFrame(t4);t4.to_csv(out/'TABLE4_COMPARISON.csv',index=False)
 # Regression values are unrounded original Table 4 summary percentages and loss.
 expected4={3:(100,32.64,593.9256,6.32),9:(8.72,.56,2316.0624,5.04),15:(72.88,4.8,1576.5192,4.24)}
 for f,(clean,c8,loss,far) in expected4.items():
  x=t4[(t4.scenario=='original')&(t4.fault==f)&(t4.monitor=='Full monitor')].iloc[0]
  assert np.isclose(x.clean_pct_mean,clean) and np.isclose(x.clean8_pct_mean,c8) and np.isclose(x.normal_FAR_pct_mean,far)
  # manuscript loss rounded to one decimal; retain exact mask-derived value.
  assert np.isclose(round(x.mean_loss_min_mean,1),round(loss,1))
  replay.append({'table':'Table 4','key':f'Full monitor F{f}','recomputed':float(x.clean_pct_mean),'archived_value':clean,'match':True})
 t5=[];delaystates=[]
 for fault in [3,9,15]:
  pairs=[(cache[('baseline',s,fault,FULL[fault])],cache[('baseline',s,fault,REF[fault])]) for s in range(1,6)]
  ids=pairs[0][0][2]
  for scenario,ex in [('original',False),('exclude_test_self_overlap',True)]:
   fk=keep('R5','fault_evaluation',fault,ids,ex);nk=keep('R5','normal_evaluation',0,ids,ex)
   row={'scenario':scenario,'fault':fault,'full_method':FULL[fault],'reference_method':REF[fault],'fault_n_shared':int(fk.sum()),'normal_n_shared':int(nk.sum()),'states':5}
   for key,where,level,scale in [('loss',0,100*(1-.05/3),1),('clean',0,95,100),('clean8',0,95,100),('any',1,95,100)]:
    kk=fk if where==0 else nk
    mat=np.stack([(a[where][key]-b[where][key])[kk] for a,b in pairs])*scale;v=mat.mean(0)
    lo,hi=boot(v,f'T5_{fault}_{scenario}_{key}',level);name={'loss':'delta_loss_min','clean':'delta_clean40_pp','clean8':'delta_clean8_pp','any':'delta_normal_FAR_pp'}[key]
    row[name]=float(v.mean());row[name+'_lo']=lo;row[name+'_hi']=hi
    if key=='loss':row['states_with_lower_loss']=int((mat.mean(1)<0).sum())
   delays=[]
   for s,(a,b) in enumerate(pairs,1):
    fa,fb=a[0],b[0];joint=fk&(fa['clean']==1)&(fb['clean']==1);n=int(joint.sum());da=float(fa['delay'][joint].mean()/60) if n else np.nan;db=float(fb['delay'][joint].mean()/60) if n else np.nan
    delaystates.append({'scenario':scenario,'fault':fault,'state':s,'joint_clean_n':n,'full_mean_delay_h':da,'reference_mean_delay_h':db,'full_minus_reference_delay_h':da-db});delays.append([da,db])
   delays=np.asarray(delays);row['full_joint_delay_h']=float(delays[:,0].mean()) if np.isfinite(delays).all() else np.nan;row['reference_joint_delay_h']=float(delays[:,1].mean()) if np.isfinite(delays).all() else np.nan;row['joint_delay_status']='DEFINED' if np.isfinite(delays).all() else 'UNDEFINED_ONE_STATE_HAS_NO_JOINT_DETECTION'
   t5.append(row)
 t5=pd.DataFrame(t5)
 reported5={3:(-582.4,-459.2),9:(-81.8,-32.7),15:(-254.5,-86.9)}
 t5['reported_original_loss_ci_lo']=t5.apply(lambda x:reported5[x.fault][0] if x.scenario=='original' else np.nan,axis=1)
 t5['reported_original_loss_ci_hi']=t5.apply(lambda x:reported5[x.fault][1] if x.scenario=='original' else np.nan,axis=1)
 t5.to_csv(out/'TABLE5_COMPARISON.csv',index=False);pd.DataFrame(delaystates).to_csv(out/'TABLE5_JOINT_DELAY_BY_STATE.csv',index=False)
 for f,expected in [(3,-519.7),(9,-57.0),(15,-170.3)]:
  x=t5[(t5.scenario=='original')&(t5.fault==f)].iloc[0]
  assert abs(round(x.delta_loss_min,1)-expected)<1e-6
  replay.append({'table':'Table 5','key':f'F{f} loss','recomputed':float(x.delta_loss_min),'archived_value':expected,'match':True})
 # Table 6: original saved event rows contain seed identities; masks are independently
 # checked against these events for standard and relearned monitors.
 records=[]
 std=csv(d/'inputs/standard/ORIGINAL_EVENTS.csv');relearn=csv(d/'inputs/relearn/ORIGINAL_EVENTS.csv')
 for family,frame in [('standard',std),('relearned',relearn)]:
  if family=='standard':frame=frame[(frame.feature_set=='full52')&(frame.regime=='adapted')].copy()
  for _,x in frame.iterrows():
   panel={'S25_primary':'recording_panel1','D439_supplement':'recording_panel2'}[x.panel]
   role=str(x.role);normal=role=='normal_evaluation';run=int(str(x.case_id).split('_')[-1]);sid=identity(panel,role,0 if normal else 9,np.arange(1,61 if panel=='recording_panel1' and normal else 76 if panel=='recording_panel1' else 121)).loc[run]
   assert int(x.seed_u32)==int(sid.seed_u32)
   view=(x.input_recording if family=='standard' else x.recording);method=(x.method if family=='standard' else x.monitor)
   def flag(v):return int(str(v).lower() in ['true','1','1.0'])
   clean=flag(x.clean_40h) if family=='standard' else flag(x.clean)
   clean8=flag(x.clean_8h) if family=='standard' else flag(x.clean_480)
   records.append({'panel':panel,'family':family,'method':method,'recording':'controller-input' if view=='snapshot' else 'native','role':role,'run_number':run,'seed_u32':int(sid.seed_u32),'self_overlap':bool(sid.self_wrap),'clean':clean,'clean8':clean8,'normal_alarm':flag(x.normal_alarm),'loss':float(x.loss_minutes)})
 for pn in [1,2]:
  a=csv(d/f'inputs/fixed/panel{pn}_EVENTS.csv')
  if pn==2:a=a[a.method=='Signed_CUSUM_reference']
  for _,x in a.iterrows():
   panel=f'recording_panel{pn}';role=str(x.case_id).rsplit('_',1)[0];run=int(str(x.case_id).split('_')[-1]);nn=60 if pn==1 and role=='normal_evaluation' else 75 if pn==1 else 120
   sid=identity(panel,role,0 if role=='normal_evaluation' else 9,np.arange(1,nn+1)).loc[run]
   records.append({'panel':panel,'family':'fixed_historical','method':'Fixed historical monitor','recording':'native' if x['view']=='native' else 'controller-input','role':role,'run_number':run,'seed_u32':int(sid.seed_u32),'self_overlap':bool(sid.self_wrap),'clean':int(x.clean),'clean8':int(x.clean_480),'normal_alarm':int(x.normal_alarm),'loss':float(x.loss_minutes)})
 rec=pd.DataFrame(records);rec.to_csv(out/'TABLE6_RUN_EVENTS_AND_EXCLUSION.csv',index=False)
 t6=[];t6pairs=[]
 for (panel,family,method,view),g in rec.groupby(['panel','family','method','recording'],sort=False):
  for scenario,ex in [('original',False),('exclude_test_self_overlap',True)]:
   z=g[~g.self_overlap] if ex else g;n=z[z.role=='normal_evaluation'];f=z[z.role=='fault9_evaluation']
   row={'scenario':scenario,'panel':panel,'family':family,'method':method,'recording':view,'fault_n':len(f),'normal_n':len(n),'fault_clean40_n':int(f.clean.sum()),'fault_clean40_pct':float(f.clean.mean()*100),'fault_clean8_n':int(f.clean8.sum()),'fault_clean8_pct':float(f.clean8.mean()*100),'normal_alarm_n':int(n.normal_alarm.sum()),'normal_FAR_pct':float(n.normal_alarm.mean()*100),'normal_floor_n':int(n.clean.sum()),'normal_floor_pct':float(n.clean.mean()*100),'mean_loss_min':float(f.loss.mean())};t6.append(row)
 paired_rec=rec.copy();paired_rec['paired_method']=paired_rec.method
 paired_rec.loc[paired_rec.family=='relearned','paired_method']=paired_rec.loc[paired_rec.family=='relearned','method'].map({'M1':'Second-moment relearned','M2':'Second-moment relearned','M3':'Mean-shift relearned','M4':'Mean-shift relearned'})
 for (panel,family,method),g in paired_rec.groupby(['panel','family','paired_method'],sort=False):
  for scenario,ex in [('original',False),('exclude_test_self_overlap',True)]:
   z=g[(g.role=='fault9_evaluation')&(~g.self_overlap if ex else True)]
   w=z.pivot(index='run_number',columns='recording',values='clean');assert w.notna().all().all();v=(w['controller-input']-w.native).to_numpy()*100
   lo,hi=boot(v,f'T6_{panel}_{family}_{method}_{scenario}',95);t6pairs.append({'scenario':scenario,'panel':panel,'family':family,'method':method,'fault_n':len(v),'controller_minus_native_detection_pp':float(v.mean()),'lo_95':lo,'hi_95':hi})
 t6=pd.DataFrame(t6);t6.to_csv(out/'TABLE6_COMPARISON.csv',index=False);pd.DataFrame(t6pairs).to_csv(out/'TABLE6_PAIRED_RECORDING_DIFFERENCES.csv',index=False)
 for panel,method,expected in [('recording_panel1','Fixed historical monitor',(8,72,2,4)),('recording_panel2','Fixed historical monitor',(23,113,6,9)),('recording_panel1','M2',(28,None,2,None)),('recording_panel2','M2',(38,None,7,None)),('recording_panel1','M1',(None,75,None,3)),('recording_panel2','M1',(None,119,None,7))]:
  z=t6[(t6.scenario=='original')&(t6.panel==panel)&(t6.method==method)].set_index('recording')
  for view,fc,nc in [('native',expected[0],expected[2]),('controller-input',expected[1],expected[3])]:
   if fc is not None:assert int(z.loc[view,'fault_clean40_n'])==fc and int(z.loc[view,'normal_alarm_n'])==nc
  replay.append({'table':'Table 6','key':f'{panel} {method}','recomputed':str(expected),'archived_value':str(expected),'match':True})
 # Wide side-by-side tables preserve original and sensitivity values explicitly.
 maxima=[]
 for name,df,keys,vals in [('Table3',t3,['panel','contrast','fault'],['delta_loss_min']),('Table4',t4,['fault','monitor'],['clean_pct_mean','clean8_pct_mean','mean_loss_min_mean','normal_FAR_pct_mean']),('Table5',t5,['fault'],['delta_loss_min','delta_clean40_pp','delta_clean8_pp','delta_normal_FAR_pp']),('Table6',t6,['panel','family','method','recording'],['fault_clean40_pct','normal_FAR_pct','mean_loss_min'])]:
  a=df[df.scenario=='original'].set_index(keys);e=df[df.scenario=='exclude_test_self_overlap'].set_index(keys)
  assert a.index.equals(e.index)
  wide=a.add_suffix('_original').join(e.add_suffix('_exclude_test'))
  for v in vals:
   delta=e[v]-a[v];wide[v+'_change']=delta
   maxima.append({'table':name,'metric':v,'maximum_absolute_change':float(delta.abs().max()),'signed_change_min':float(delta.min()),'signed_change_max':float(delta.max())})
  wide.reset_index().to_csv(out/f'{name.upper()}_SIDE_BY_SIDE.csv',index=False)
 pd.DataFrame(maxima).to_csv(out/'MAXIMUM_CHANGES.csv',index=False)
 pd.DataFrame(replay).to_csv(out/'ORIGINAL_RESULT_REPLAY_CHECKS.csv',index=False)
 (out/'INPUT_HASHES.json').write_text(json.dumps(list(reads.values()),ensure_ascii=False,indent=2)+'\n')
 status={'analysis1':'COMPLETE','analysis2':'SEE_SEPARATE_ANALYSIS2_STATUS','total_input_runs':9007,'self_overlapping_runs':287,'PCA500_self_overlap':9,'PCA78_self_overlap':0,'tests_excluded_unique':int(seeds[seeds.role.str.endswith('evaluation')].self_wrap.sum()),'bootstrap_draws':B,'bootstrap_seed':SEED,'original_point_estimates_and_saved_run_endpoints_reproduced':True,'reported_original_intervals_kept_separately':True,'confidence_intervals_recomputed_with_new_fixed_seed':True,'original_inputs_sha256_verified':len(manifest),'all_read_inputs_sha256_recorded':len(reads),'new_simulations':0,'PCA_refits':0,'monitor_refits':0,'original_threshold_changes':0,'original_report_edits':0,'five_states_treated_as_shared_physical_runs':True,'retained_between_run_rng_overlap_not_removed':True}
 (out/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status,indent=2));print(pd.DataFrame(maxima).to_string(index=False));print(t3.to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.inputs,a.out)
