#!/usr/bin/env python3
"""Partial Analysis 2 from archived calibration maxima and exact path summaries.

This makes supplemental thresholds only; frozen original files are never written.
Available coverage: repeated validation full monitors; fixed historical Panel 1.
"""
import argparse,hashlib,json,math,io
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd

FULL={3:('mean_signed__full','mean_pool_signed'),9:('moment_signed__full','moment_pool_signed'),15:('moment_squared__full','moment_pool_squared')}
REF={3:'FDA_equal_class_signed_CUSUM',9:'FDA_equal_class_signed_CUSUM',15:'covariance_F15_max_squared_CUSUM'}

def main(root,completed,out,baseline_zip):
 root=Path(root).resolve();completed=Path(completed).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 r=root/'01_RNG_Overlap';d=root/'02_03_Detection_and_Paired_Delay';models=root/'04_Equal_FAR/frozen_inputs/source/current_study/r5_s21_s22/S21/Stage_D_Return/frozen_models';reads={}
 def record(p):
  b=Path(p).read_bytes();reads[str(p)]={'path':str(p),'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return b
 def arr(p):
  record(p)
  with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
 def js(p):return json.loads(record(p).decode('utf-8-sig'))
 record(r/'results/RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv');s=pd.read_csv(r/'results/RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv')
 s['run_number']=s.run_id.str.extract(r'run=(\d+)').fillna(s.run_id.str.extract(r'_(\d+)$')).astype(int)
 s['replication']=s.run_id.str.extract(r'rep=(\d+)').fillna(0).astype(int)
 def ident(panel,role,fault,n,rep=0):
  v=s[(s.panel==panel)&(s.role==role)&(s.fault_id==fault)&(s.replication==rep)].sort_values('run_number')
  assert len(v)==n and np.array_equal(v.run_number,np.arange(1,n+1));return v
 def events(pre,postmax,h):
  crossing=postmax>h;post=crossing[:,-1];clean=(pre<=h)&post;delay=np.where(post,crossing.argmax(1)*3,2400)
  return {'clean':clean,'clean8':clean&(delay<=480),'loss':np.where(clean,delay,2400),'delay':delay}
 def mask_events(x):
  pre=x[:,:160].any(1);post=x[:,160:].any(1);clean=(~pre)&post;delay=np.where(post,x[:,160:].argmax(1)*3,2400)
  return {'clean':clean,'clean8':clean&(delay<=480),'loss':np.where(clean,delay,2400),'delay':delay}
 index=js(completed/'04_EQUAL_FAR/DIAGNOSTIC_PATH_SUMMARIES/INDEX.json')['items']
 pathmap={(x['fault'],x['state'],x['method']):x for x in index}
 # These archived modules bind each role to increasing original run number.
 code=root/'04_Equal_FAR/frozen_inputs/code/r5_continuation.py';ct=record(code).decode()
 assert "key=lambda r:r['run']" in ct and "np.arange(1,n+1)" in ct
 manifest=js(r/'inputs/manifests/R5.json')
 for rep in range(1,6):
  rows=sorted([x for x in manifest if x['replication']==rep and x['role']=='normal_outer_calibration'],key=lambda x:x['run'])
  ids=ident('R5','normal_outer_calibration',0,259,rep)
  assert [x['canonical_label'] for x in rows]==ids.run_id.tolist() and [x['seed_u32'] for x in rows]==ids.seed_u32.tolist()
 thresholds=[];rows=[];regressions=[];all_candidates=[];eventcache={};normalcache={}
 for rep in range(1,6):
  cal=ident('R5','normal_outer_calibration',0,259,rep);ck=~cal.self_wrap.to_numpy();n=int(ck.sum());rank=math.ceil(.95*(n+1))
  nk=~ident('R5','normal_evaluation',0,250).self_wrap.to_numpy()
  for fault,(method,choicekey) in FULL.items():
   model=models/f'rep{rep:02d}'/f'F{fault:02d}';outer=arr(model/'OUTER.npz');sel=js(model/'SELECTION.json');assert outer['peaks'].shape==(259,49)
   assert np.allclose(np.sort(outer['peaks'],axis=0)[246],outer['thresholds'],rtol=1e-12,atol=1e-10)
   new_h=np.sort(outer['peaks'][ck],axis=0)[rank-1];choice=int(sel['choices'][choicekey]);h=float(new_h[choice]);oldh=float(outer['thresholds'][choice])
   for key,j in sel['choices'].items():all_candidates.append({'state':rep,'fault':fault,'method':key,'candidate_index':j,'calibration_original_n':259,'calibration_excluded_n':int((~ck).sum()),'calibration_retained_n':n,'rank_original':247,'rank_new':rank,'rank_proportional_ceiling':math.ceil(247*n/259),'original_threshold':float(outer['thresholds'][j]),'supplemental_threshold':float(new_h[j]),'new_minus_original_threshold':float(new_h[j]-outer['thresholds'][j]),'test_path_available':key==choicekey})
   src=pathmap[(fault,rep,method)];p=completed/'04_EQUAL_FAR/DIAGNOSTIC_PATH_SUMMARIES'/src['file'];b=record(p);assert hashlib.sha256(b).hexdigest()==src['sha256'];a=arr(p)
   original=events(a['fault_prefault_max'],a['fault_post_running_max'],oldh)
   frozen=arr(d/f'inputs/baseline/rep{rep:02d}/F{fault:02d}.npz');j=frozen['methods'].tolist().index(method)
   fm=np.unpackbits(frozen['fault_alarm_bits'],axis=1,count=960,bitorder='little')[:,:,j].astype(bool);nm=np.unpackbits(frozen['normal_alarm_bits'],axis=1,count=960,bitorder='little')[:,:,j].astype(bool);ev=mask_events(fm)
   for k in ['clean','clean8','loss']:assert np.array_equal(original[k],ev[k])
   assert np.array_equal(a['normal_peak']>oldh,nm.any(1));regressions.append({'fault':fault,'state':rep,'method':method,'frozen_endpoint_and_FAR_replay':'PASS'})
   fk=~ident('R5','fault_evaluation',fault,250).self_wrap.to_numpy()
   for scenario,threshold,kk,nkk in [('original',oldh,np.ones(250,bool),np.ones(250,bool)),('exclude_test_self_overlap',oldh,fk,nk),('exclude_test_and_calibration_self_overlap',h,fk,nk)]:
    ee=events(a['fault_prefault_max'],a['fault_post_running_max'],threshold)
    eventcache[(scenario,fault,rep,method)]={k:v[kk] for k,v in ee.items()};normalcache[(scenario,fault,rep,method)]=(a['normal_peak'][nkk]>threshold).astype(int)
    rows.append({'scenario':scenario,'panel':'R5','state':rep,'fault':fault,'method':method,'fault_n':int(kk.sum()),'normal_n':int(nkk.sum()),'threshold':threshold,'calibration_n':n if scenario=='exclude_test_and_calibration_self_overlap' else 259,'calibration_rank':rank if scenario=='exclude_test_and_calibration_self_overlap' else 247,'clean40_n':int(ee['clean'][kk].sum()),'clean40_pct':float(ee['clean'][kk].mean()*100),'clean8_pct':float(ee['clean8'][kk].mean()*100),'mean_loss_min':float(ee['loss'][kk].mean()),'normal_FAR_pct':float((a['normal_peak'][nkk]>threshold).mean()*100),'normal_floor_status':'NOT_AVAILABLE_FROM_FULL_RUN_MAXIMA'})
   thresholds.append({'state':rep,'fault':fault,'method':method,'original_threshold':oldh,'supplemental_threshold':h,'retained_calibration_n':n,'calibration_self_overlap_excluded':int((~ck).sum()),'supplemental_rank':rank,'rank_rule':'ceil(0.95*(n+1)); strict > threshold','threshold_original_file_unchanged':True})
 full_rows=pd.DataFrame(rows);full_rows.to_csv(out/'ANALYSIS2_FULL_MONITOR_BY_STATE.csv',index=False)
 record(baseline_zip)
 with ZipFile(baseline_zip) as z:
  for rep in range(1,6):
   cal=ident('R5','normal_outer_calibration',0,259,rep);ck=~cal.self_wrap.to_numpy();n=int(ck.sum());rank=math.ceil(.95*(n+1));nk=~ident('R5','normal_evaluation',0,250).self_wrap.to_numpy()
   for fault,method in REF.items():
    member=f'scored/R5_state{rep:02d}/F{fault:02d}/'+('COVARIANCE_CALIBRATION.npz' if fault==15 else 'FDA_CALIBRATION.npz');b=z.read(member);reads[str(baseline_zip)+'::'+member]={'path':str(baseline_zip)+'::'+member,'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    with np.load(io.BytesIO(b),allow_pickle=False) as aa:calib={k:aa[k] for k in aa.files}
    oldh=float(calib['threshold']);assert len(calib['outer_peaks'])==259 and np.isclose(np.sort(calib['outer_peaks'])[246],oldh,rtol=1e-12,atol=1e-10)
    h=float(np.sort(calib['outer_peaks'][ck])[rank-1]);src=pathmap[(fault,rep,method)];p=completed/'04_EQUAL_FAR/DIAGNOSTIC_PATH_SUMMARIES'/src['file'];b=record(p);assert hashlib.sha256(b).hexdigest()==src['sha256'];a=arr(p)
    original=events(a['fault_prefault_max'],a['fault_post_running_max'],oldh);frozen=arr(d/f'inputs/baseline/rep{rep:02d}/F{fault:02d}.npz');j=frozen['methods'].tolist().index(method);fm=np.unpackbits(frozen['fault_alarm_bits'],axis=1,count=960,bitorder='little')[:,:,j].astype(bool);nm=np.unpackbits(frozen['normal_alarm_bits'],axis=1,count=960,bitorder='little')[:,:,j].astype(bool);ev=mask_events(fm)
    for k in ['clean','clean8','loss']:assert np.array_equal(original[k],ev[k])
    assert np.array_equal(a['normal_peak']>oldh,nm.any(1));regressions.append({'fault':fault,'state':rep,'method':method,'frozen_endpoint_and_FAR_replay':'PASS'})
    fk=~ident('R5','fault_evaluation',fault,250).self_wrap.to_numpy()
    for scenario,threshold,kk,nkk in [('original',oldh,np.ones(250,bool),np.ones(250,bool)),('exclude_test_self_overlap',oldh,fk,nk),('exclude_test_and_calibration_self_overlap',h,fk,nk)]:
     ee=events(a['fault_prefault_max'],a['fault_post_running_max'],threshold);eventcache[(scenario,fault,rep,method)]={k:v[kk] for k,v in ee.items()};normalcache[(scenario,fault,rep,method)]=(a['normal_peak'][nkk]>threshold).astype(int)
     rows.append({'scenario':scenario,'panel':'R5','state':rep,'fault':fault,'method':method,'fault_n':int(kk.sum()),'normal_n':int(nkk.sum()),'threshold':threshold,'calibration_n':n if scenario=='exclude_test_and_calibration_self_overlap' else 259,'calibration_rank':rank if scenario=='exclude_test_and_calibration_self_overlap' else 247,'clean40_n':int(ee['clean'][kk].sum()),'clean40_pct':float(ee['clean'][kk].mean()*100),'clean8_pct':float(ee['clean8'][kk].mean()*100),'mean_loss_min':float(ee['loss'][kk].mean()),'normal_FAR_pct':float((a['normal_peak'][nkk]>threshold).mean()*100),'normal_floor_status':'NOT_AVAILABLE_FROM_FULL_RUN_MAXIMA'})
    thresholds.append({'state':rep,'fault':fault,'method':method,'original_threshold':oldh,'supplemental_threshold':h,'retained_calibration_n':n,'calibration_self_overlap_excluded':int((~ck).sum()),'supplemental_rank':rank,'rank_rule':'ceil(0.95*(n+1)); strict > threshold','threshold_original_file_unchanged':True})
 rows=pd.DataFrame(rows);rows.to_csv(out/'ANALYSIS2_TABLE5_MONITORS_BY_STATE.csv',index=False)
 agg=rows.groupby(['scenario','fault','method'],sort=False).agg(states=('state','size'),shared_fault_n=('fault_n','first'),shared_normal_n=('normal_n','first'),clean40_pct_mean=('clean40_pct','mean'),clean40_pct_min=('clean40_pct','min'),clean40_pct_max=('clean40_pct','max'),clean8_pct_mean=('clean8_pct','mean'),mean_loss_min_mean=('mean_loss_min','mean'),normal_FAR_pct_mean=('normal_FAR_pct','mean')).reset_index();agg.to_csv(out/'ANALYSIS2_TABLE5_MONITORS_FIVE_STATE.csv',index=False)
 agg[agg.method.isin([v[0] for v in FULL.values()])].to_csv(out/'ANALYSIS2_FULL_MONITOR_FIVE_STATE.csv',index=False)
 def boot(v,key,level):
  rng=np.random.default_rng(np.random.SeedSequence([2026100804,int(hashlib.sha256(key.encode()).hexdigest()[:8],16)]));v=np.asarray(v,float);a=np.empty(20000)
  for start in range(0,20000,1000):a[start:start+1000]=v[rng.integers(0,len(v),(1000,len(v)))].mean(1)
  q=np.quantile(a,[(1-level/100)/2,1-(1-level/100)/2]);return float(q[0]),float(q[1])
 pairrows=[]
 for fault,reference in REF.items():
  full=FULL[fault][0]
  for scenario in ['original','exclude_test_self_overlap','exclude_test_and_calibration_self_overlap']:
   row={'scenario':scenario,'fault':fault,'full_method':full,'reference_method':reference,'states':5}
   pairs=[(eventcache[(scenario,fault,rep,full)],eventcache[(scenario,fault,rep,reference)]) for rep in range(1,6)];row['fault_n_shared']=len(pairs[0][0]['loss']);row['normal_n_shared']=len(normalcache[(scenario,fault,1,full)])
   for key,name,scale,level in [('loss','delta_loss_min',1,100*(1-.05/3)),('clean','delta_clean40_pp',100,95),('clean8','delta_clean8_pp',100,95)]:
    mat=np.stack([np.asarray(a[key],float)-np.asarray(b[key],float) for a,b in pairs])*scale;v=mat.mean(0);lo,hi=boot(v,f'{scenario}_{fault}_{key}',level);row[name]=float(v.mean());row[name+'_lo']=lo;row[name+'_hi']=hi
    if key=='loss':row['states_with_lower_loss']=int((mat.mean(1)<0).sum())
   v=np.stack([normalcache[(scenario,fault,rep,full)]-normalcache[(scenario,fault,rep,reference)] for rep in range(1,6)]).mean(0)*100;lo,hi=boot(v,f'{scenario}_{fault}_far',95);row['delta_normal_FAR_pp']=float(v.mean());row['delta_normal_FAR_pp_lo']=lo;row['delta_normal_FAR_pp_hi']=hi
   delays=[]
   for a,b in pairs:
    joint=a['clean']&b['clean'];delays.append([float(a['delay'][joint].mean()/60) if joint.any() else np.nan,float(b['delay'][joint].mean()/60) if joint.any() else np.nan])
   delays=np.asarray(delays);row['full_joint_delay_h']=float(delays[:,0].mean()) if np.isfinite(delays).all() else np.nan;row['reference_joint_delay_h']=float(delays[:,1].mean()) if np.isfinite(delays).all() else np.nan;row['joint_delay_status']='DEFINED' if np.isfinite(delays).all() else 'UNDEFINED_ONE_STATE_HAS_NO_JOINT_DETECTION';pairrows.append(row)
 pd.DataFrame(pairrows).to_csv(out/'ANALYSIS2_TABLE5_PAIRED_CONTRASTS.csv',index=False)
 pd.DataFrame(thresholds).to_csv(out/'ANALYSIS2_FULL_MONITOR_SUPPLEMENTAL_THRESHOLDS.csv',index=False);pd.DataFrame(all_candidates).to_csv(out/'ANALYSIS2_SELECTED_MONITOR_THRESHOLD_ONLY.csv',index=False);pd.DataFrame(regressions).to_csv(out/'ANALYSIS2_PATH_SUMMARY_REGRESSION.csv',index=False)
 # Full paths exist for the fixed historical monitor on recording Panel 1.
 charts=arr(d/'inputs/fixed/panel1_CHARTS.npz');frozenh=np.asarray(js(d/'inputs/fixed/panel1_THRESHOLDS.json')['thresholds']);cc=ident('recording_panel1','normal_calibration',0,59);ck=~cc.self_wrap.to_numpy();n=int(ck.sum());rank=math.ceil(.95*(n+1));calpeak=charts['normal_calibration'].max(1)
 assert np.allclose(np.sort(calpeak,axis=0)[56],frozenh,rtol=1e-12,atol=1e-10)
 newh=np.sort(calpeak[ck],axis=0)[rank-1];fk=~ident('recording_panel1','fault9_evaluation',9,75).self_wrap.to_numpy();nk=~ident('recording_panel1','normal_evaluation',0,60).self_wrap.to_numpy();fixedrows=[];fixedcache={}
 for scenario,h,fkeep,nkeep in [('original',frozenh,np.ones(75,bool),np.ones(60,bool)),('exclude_test_self_overlap',frozenh,fk,nk),('exclude_test_and_calibration_self_overlap',newh,fk,nk)]:
  for j,view in enumerate(['native','controller-input']):
   f=mask_events(charts['fault9_evaluation'][:,:,j]>h[j]);normal=charts['normal_evaluation'][:,:,j]>h[j];ne=mask_events(normal)
   fixedcache[(scenario,view)]={'clean':f['clean'][fkeep].astype(int),'loss':f['loss'][fkeep],'normal':normal[nkeep].any(1).astype(int)}
   fixedrows.append({'scenario':scenario,'panel':'recording_panel1','method':'Fixed historical monitor','recording':view,'fault_n':int(fkeep.sum()),'normal_n':int(nkeep.sum()),'threshold':float(h[j]),'calibration_n':n if scenario=='exclude_test_and_calibration_self_overlap' else 59,'calibration_rank':rank if scenario=='exclude_test_and_calibration_self_overlap' else 57,'fault_clean40_n':int(f['clean'][fkeep].sum()),'fault_clean40_pct':float(f['clean'][fkeep].mean()*100),'fault_clean8_pct':float(f['clean8'][fkeep].mean()*100),'mean_loss_min':float(f['loss'][fkeep].mean()),'normal_alarm_n':int(normal[nkeep].any(1).sum()),'normal_FAR_pct':float(normal[nkeep].any(1).mean()*100),'normal_floor_pct':float(ne['clean'][nkeep].mean()*100)})
 fixedrows=pd.DataFrame(fixedrows);fixedrows.to_csv(out/'ANALYSIS2_TABLE6_FIXED_PANEL1.csv',index=False)
 fixedpairs=[]
 for scenario in ['original','exclude_test_self_overlap','exclude_test_and_calibration_self_overlap']:
  a=fixedcache[(scenario,'controller-input')];b=fixedcache[(scenario,'native')];row={'scenario':scenario,'panel':'recording_panel1','method':'Fixed historical monitor','fault_n':len(a['clean']),'normal_n':len(a['normal'])}
  for key,name,scale in [('clean','controller_minus_native_detection_pp',100),('loss','controller_minus_native_loss_min',1),('normal','controller_minus_native_normal_FAR_pp',100)]:
   v=(a[key]-b[key])*scale;lo,hi=boot(v,f'fixedpanel1_{scenario}_{key}',95);row[name]=float(v.mean());row[name+'_lo95']=lo;row[name+'_hi95']=hi
  fixedpairs.append(row)
 pd.DataFrame(fixedpairs).to_csv(out/'ANALYSIS2_TABLE6_FIXED_PANEL1_PAIRED.csv',index=False)
 maximum=[]
 for name,df,keys,metrics in [('Table4 full monitors only',agg[agg.method.isin([v[0] for v in FULL.values()])],['fault','method'],['clean40_pct_mean','clean8_pct_mean','mean_loss_min_mean','normal_FAR_pct_mean']),('Table5 all three contrasts',pd.DataFrame(pairrows),['fault'],['delta_loss_min','delta_clean40_pp','delta_clean8_pp','delta_normal_FAR_pp']),('Table6 Panel1 fixed historical only',fixedrows,['recording'],['fault_clean40_pct','normal_FAR_pct','mean_loss_min'])]:
  original=df[df.scenario=='original'].set_index(keys)
  for scenario in ['exclude_test_self_overlap','exclude_test_and_calibration_self_overlap']:
   changed=df[df.scenario==scenario].set_index(keys)
   for metric in metrics:
    delta=changed[metric]-original[metric];maximum.append({'coverage':name,'scenario':scenario,'metric':metric,'maximum_absolute_change':float(delta.abs().max()),'signed_change_min':float(delta.min()),'signed_change_max':float(delta.max())})
 pd.DataFrame(maximum).to_csv(out/'ANALYSIS2_MAXIMUM_CHANGES_AVAILABLE_SUBSETS.csv',index=False)
 gaps=[{'target':'Table 3 single-fit contrasts','status':'NOT_RECONSTRUCTABLE','missing':'full test statistic paths/scores for both comparators (single-fit outer maxima available in nested source snapshot)'},{'target':'Table 3 repeated C1/C2/C3','status':'THRESHOLDS_ONLY_RECONSTRUCTED','missing':'test statistic paths or scores for same-direction Squared and mean-shift comparators; full C2/C3 comparator paths'},{'target':'Table 4 full monitors','status':'COMPLETE_FOR_3_FULL_MONITORS_5_STATES','missing':'normal pre/post maxima needed only for normal detection floors'},{'target':'Table 4 other monitors','status':'PARTIAL_REFERENCES_AVAILABLE','missing':'full test paths/scores for remaining monitors, including I restrictions, Q, and FDA F15; same-direction/mean-pool comparators'},{'target':'Table 5 full-reference contrasts','status':'COMPLETE','missing':''},{'target':'Table 6 recording Panel 1 fixed historical','status':'COMPLETE','missing':''},{'target':'Table 6 Panel 2 fixed historical','status':'INCOMPLETE','missing':'199 calibration paths/maxima; full fault test path/score arrays'},{'target':'Table 6 standard/relearned monitors','status':'INCOMPLETE','missing':'59/199 outer calibration maxima or scores; standard full test paths/scores'}]
 pd.DataFrame(gaps).to_csv(out/'ANALYSIS2_COVERAGE_AND_MISSING_INPUTS.csv',index=False)
 (out/'ANALYSIS2_INPUT_HASHES.json').write_text(json.dumps(list(reads.values()),ensure_ascii=False,indent=2)+'\n')
 status={'status':'PARTIAL_COMPLETE_EXACT_AVAILABLE_SUBSETS','new_simulations':0,'refits':0,'original_threshold_edits':0,'supplemental_thresholds_written_to_new_CSV_only':True,'complete_repeated_full_and_reference_monitors':30,'Table5_all_three_paired_contrasts_complete':True,'complete_fixed_historical_recording_panel1':2,'complete_Tables3_to6_analysis2':False,'rank_rule':'ceil(0.95*(retained_calibration_n+1))','test_normal_limits_not_used':True,'Table5_bootstrap_draws':20000,'reason_for_no_C1_C2_C3_recalibration_claim':'required comparator test paths absent'}
 (out/'ANALYSIS2_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2));print(agg.to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--completed',type=Path,required=True);p.add_argument('--baseline-return-zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.inputs,a.completed,a.out,a.baseline_return_zip)
