"""Complete the remaining fixed-historical Panel 2 recalibration using saved scores."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from historical_core import recurrence,write_csv,loadz,sha
from a4_core import keep_from_seeds,rank95,events,summarize_one,paired_ci,SCENARIOS

def run(root,out):
 p=Path(root)/'inputs/A4_fixed_panel2';out=Path(out);out.mkdir(parents=True,exist_ok=False)
 z=loadz(p/'PAIRED_SIGNED_SCORES.npz');lock=json.loads((p/'THRESHOLDS_LOCK.json').read_text());old=np.array(lock['thresholds'])[[0,3]]
 st={k:np.stack([recurrence(z[k+'_scores'][:,:,j]) for j in range(2)],axis=2) for k in ('normal_calibration','normal_evaluation','fault9_evaluation')}
 peaks=st['normal_calibration'].max(1);np.testing.assert_allclose(np.sort(peaks,axis=0)[189],old,rtol=0,atol=1e-9)
 ck=keep_from_seeds(z['normal_calibration_seeds']);nk=keep_from_seeds(z['normal_evaluation_seeds']);fk=keep_from_seeds(z['fault9_evaluation_seeds']);r=rank95(ck.sum());new=np.sort(peaks[ck],axis=0)[r-1]
 prev=list(csv.DictReader((p/'ALL_RUN_OUTCOMES.csv').open(encoding='utf-8-sig')));checked=0
 for role in ('normal_evaluation','fault9_evaluation'):
  ev=events(st[role]>old)
  for row in prev:
   if row['role']!=role or row['method']!='Signed_CUSUM_reference':continue
   i=int(row['case_id'].split('_')[-1])-1;j=['native','all_fast_input'].index(row['view'])
   for field,key in [('pre_alarm','pre'),('clean_2400','clean40'),('clean_480','clean8'),('loss_minutes','loss')]:
    assert int(row[field])==int(ev[key][i,j]),(row['case_id'],key)
   if role=='normal_evaluation':assert int(row['normal_alarm'])==int(ev['any_alarm'][i,j])
   checked+=1
 rows,evs,m=summarize_one('recording_panel2_fixed_history',1,9,['native_Signed','controller_input_Signed'],st['fault9_evaluation'],st['normal_evaluation'],old,new,fk,nk,ck,np.ones(2),peaks,out)
 pair=[]
 for sc in SCENARIOS:
  ek=m[sc]
  for metric,units in [('loss','minutes'),('clean40','percentage_points'),('clean8','percentage_points')]:
   v=ek[metric][:,1].astype(float)-ek[metric][:,0].astype(float)
   if metric!='loss':v*=100
   pair.append(dict(scenario=sc,contrast='controller_input_minus_native',endpoint=metric,units=units,**paired_ci(v,'fixedpanel2'+sc+metric,.95)))
 write_csv(out/'PAIRED_RECORDING_COMPARISONS.csv',pair)
 (out/'STATUS.json').write_text(json.dumps(dict(status='COMPLETE',saved_scores_replayed=439,original_evaluation_records_reproduced=checked,normal_calibration_original=199,normal_calibration_kept=int(ck.sum()),new_order=r,normal_evaluation_kept=int(nk.sum()),fault_evaluation_kept=int(fk.sum()),new_simulations=0,raw52_not_reloaded=True,original_run_recalibration_sources='fixed scores and original per-run case identities',input_sha256=sha(p/'PAIRED_SIGNED_SCORES.npz')),indent=2))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a.add_argument('--out',type=Path,required=True);ns=a.parse_args();run(ns.root,ns.out)
