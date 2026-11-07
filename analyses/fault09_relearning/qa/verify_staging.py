"""Integrity-only integration checks in a disposable synthetic workspace. No
simulator, real scientific raw inputs, model fit, or held-out panel is accessed."""
from pathlib import Path
from unittest.mock import patch
import csv,hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import run_study as R

def bad(fn,label,checks):
 try:fn()
 except (ValueError,FileNotFoundError):checks.append(label)
 else:raise AssertionError(label)
def row(i,role,rep):
 return dict(case_id=f'FIXTURE_{role}_{i:04}',role=role,rep=str(rep),index=str(i),fault_id='0',key_hex='0x'+format(i+10000*rep,'032x'),stage='development' if rep else 'calibration')
def checkpoint(w,r,arrays=False):
 fp=w/'raw'/r['case_id'];fp.mkdir(parents=True)
 ident=R.sim.identity(r,R.LOCK)
 if arrays:
  base=np.full((960,52),int(r['index']),float);ci=base[:,:41].copy();ci[:,0]+=2;phase=np.zeros((960,4),dtype=int);phase[:,0]=np.arange(1,961)*180;phase[:,3]=3
  np.savez_compressed(fp/'RAW.npz',native_values=base,fast_control_input_measurements=ci,phase=phase,native_time_seconds=np.arange(961)*180.,diagnostic_IDV=np.zeros((960,20),int))
 else:(fp/'RAW.npz').write_bytes(('NOT A TRAJECTORY: '+r['case_id']).encode())
 meta=dict(ident,final_step_count=172800,rng_draw_count=1,shutdown_occurred=False,physical_state_sha256='aa'*32)
 R.put(fp/'RUN_INFO.json',meta)
 rec=dict(meta,outputs={n:R.sha(fp/n) for n in ('RUN_INFO.json','RAW.npz')})
 R.put(fp/'COMPLETE.json',rec);return rec
def main():
 checks=[];R.design() # read only the prospective case ledger
 with tempfile.TemporaryDirectory(prefix='synthetic-stage-only-') as td:
  w=Path(td);m=w/'models/R01';m.mkdir(parents=True)
  rr=[row(i+1,'pca_normal',1) for i in range(384)] # identity-binding fixture, not fit allocation
  rows=[]
  for i,r in enumerate(rr):
   c=checkpoint(w,r,i<2);rows.append(dict(r,raw_sha256=c['outputs']['RAW.npz']))
  R.tables(m/'INPUT_RECEIPTS.csv',rows);R.put(m/'MODEL_SUMMARY.json',{'rep':'1'})
  def seal():R.put(m/'MODEL_LOCK.json',dict(files={n:R.sha(m/n) for n in ('INPUT_RECEIPTS.csv','MODEL_SUMMARY.json')}))
  seal()
  with patch.object(R.learn,'anchors',return_value=np.ones(4)):
   R.verify_model_inputs(w,m,rr);checks.append('identical_model_cohort_passed')
   alternate=[dict(x) for x in rows];alternate[0]['raw_sha256']=rows[1]['raw_sha256']
   R.tables(m/'INPUT_RECEIPTS.csv',alternate);seal()
   bad(lambda:R.verify_model_inputs(w,m,rr),'internally_resealed_model_with_wrong_training_raw_rejected',checks)
   R.tables(m/'INPUT_RECEIPTS.csv',rows);seal()
  def chart(p,c,mod):
   j=int(mod.name[1:]);assert np.all(c[:,:,0]==p[:,:,0]+2)
   return p[:,:,:1]*1000+j*10+np.arange(4)[None,None,:]
  with patch.object(R.learn,'score_batch',side_effect=chart):
   [(start,sc)]=list(R.score_chunks(rr[:2],w,R.model_paths(w)))
   assert start==0 and sc.shape==(2,960,5,4) and np.array_equal(sc[:,20],(1000*np.arange(1,3))[:,None,None]+(10*np.arange(1,6))[None,:,None]+np.arange(4)[None,None,:]);checks.append('original_io_batch_replicate_monitor_order_passed')
  cal=[row(i+1,'normal_calibration',0) for i in range(199)];cs=[checkpoint(w,r) for r in cal];d=w/'calibration';d.mkdir()
  pk=np.random.default_rng(792626).uniform(.1,45,size=(199,5,4));aa=np.array([1.3,2.4,3.5,4.6]);h=np.stack([R.learn.core.thresholds(pk[:,j],aa,190) for j in range(5)])
  R.put(w/'MODELS_LOCK.json',{'test_only':True})
  def threshold_package(keys,hs):
   np.savez_compressed(d/'CALIBRATION.npz',maxima=pk,thresholds=h,case_ids=keys,raw_sha256=[r['outputs']['RAW.npz'] for r in cs])
   R.put(w/'THRESHOLDS_LOCK.json',dict(design=R.LOCK,code='QA',models_lock_sha=R.sha(w/'MODELS_LOCK.json'),calibration_sha=R.sha(d/'CALIBRATION.npz'),rank=190,n=199,new_evaluation_seen=False,h=hs.tolist()))
  ids=[r['case_id'] for r in cal]
  with patch.object(R.learn,'anchors',return_value=aa):
   threshold_package(ids,h);np.testing.assert_array_equal(R.verify_thresholds(w,'QA',cal),h);checks.append('rank190_and_individual_calibration_identities_passed')
   threshold_package(ids[::-1],h);bad(lambda:R.verify_thresholds(w,'QA',cal),'reordered_calibration_rejected',checks)
   threshold_package(ids,h[:,::-1]);bad(lambda:R.verify_thresholds(w,'QA',cal),'threshold_json_arm_swap_rejected',checks)
  bad(lambda:R.stage_guard(w,'development',rr+cal),'early_calibration_records_rejected',checks)
  # On completed resumes a changed or missing outcome/receipt cannot be silently repackaged.
  R.put(w/'FINAL_COMPLETE.json',dict(design=R.LOCK,code='QA',n_physical=2719,models_lock_sha='deliberately_invalid',thresholds_lock_sha='',receipts_sha=''))
  bad(lambda:R.verify_final(w,[{}]*2719,'QA'),'final_model_binding_tamper_rejected',checks)
 print(json.dumps({'status':'PASS','scope':'synthetic, no scientific trajectory accessed','gates':checks,'run_study_sha256':R.sha(ROOT/'run_study.py')},indent=2))
if __name__=='__main__':main()
