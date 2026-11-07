"""Synthetic software tests only. No simulator or actual shared test arrays are available.
Fixture directories: TEP_TEST_A_TOOL, TEP_TEST_B_TOOL, TEP_TEST_B_RETURN, TEP_TEST_C_RETURN.
SciPy is used for build-time CP verification only, never required by the delivered runtime.
"""
from pathlib import Path
import contextlib,csv,copy,hashlib,io,json,os,shutil,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/'payload/code'))
import inputs_d as inp,analysis_d as an,runner_stage_d as run,science,inherited
from storage import *
A=Path(os.environ['TEP_TEST_A_TOOL']);BT=Path(os.environ['TEP_TEST_B_TOOL']);BR=Path(os.environ['TEP_TEST_B_RETURN']);CR=Path(os.environ['TEP_TEST_C_RETURN'])

class NumericTests(unittest.TestCase):
 def test_events_random_match_independent(self):
  m=np.random.default_rng(45).random((7,960,9))<.003
  for k,v in science.events(m).items():np.testing.assert_array_equal(v,an.independent_events(m)[k])
 def test_pre_alarm_retained_as_failure(self):
  m=np.zeros((1,960,1),bool);m[0,10,0]=True;m[0,170,0]=True;e=science.events(m)
  self.assertEqual(e['event_loss_samples'][0,0],800);self.assertEqual(e['detected'][0,0],1);self.assertEqual(e['clean_detection'][0,0],0)
 def test_onset_zero_delay(self):
  m=np.zeros((1,960,1),bool);m[0,160,0]=True;self.assertEqual(science.events(m)['event_loss_samples'][0,0],0)
 def test_last_sample_and_miss(self):
  m=np.zeros((2,960,1),bool);m[0,959,0]=True;e=science.events(m);np.testing.assert_array_equal(e['event_loss_samples'][:,0],[799,800])
 def test_strict_threshold(self):
  mask=np.array([1.,1.+1e-12,1.-1e-12])>1.;np.testing.assert_array_equal(mask,[False,True,False])
 def test_pack_roundtrip(self):
  m=np.random.default_rng(2).integers(0,2,(250,960,9)).astype(bool);np.testing.assert_array_equal(an.unpack(an.pack(m)),m)
 def test_bad_pack_shape(self):
  with self.assertRaises(ValueError):an.pack(np.zeros((249,960,9),bool))
 def test_bad_unpack_dtype(self):
  with self.assertRaises(ValueError):an.unpack(np.zeros((250,120,9),dtype=np.int64))
 def test_CP_against_scipy_all_counts(self):
  from scipy.stats import beta
  for k in range(251):
   lo,hi=an.CP(k,250);a=0. if k==0 else beta.ppf(.025,k,251-k);b=1. if k==250 else beta.ppf(.975,k+1,250-k)
   self.assertLess(max(abs(lo-a),abs(hi-b)),2e-12)
 def test_five_fit_bootstrap_shares_ids(self):
  s=strict_json(BASE/'payload/EVALUATION_SPEC.json');x=np.arange(250,dtype=float);v=np.column_stack([x,-x,2*x,3*x,4*x]);v=np.column_stack([v,v.mean(1)])
  b,d=an.paired_bootstrap(v,s,'fault_03');np.testing.assert_allclose(b[:,:5].mean(1),b[:,5],atol=1e-12);np.testing.assert_array_equal(b[:,0],-b[:,1]);self.assertEqual(len(d),64)
 def test_rng_schedule_roles_and_repeatability(self):
  s=strict_json(BASE/'payload/EVALUATION_SPEC.json');x=np.arange(250)[:,None];a,ad=an.paired_bootstrap(x,s,'fault_03');b,bd=an.paired_bootstrap(x,s,'fault_03');_,cd=an.paired_bootstrap(x,s,'normal_shared');np.testing.assert_array_equal(a,b);self.assertEqual(ad,bd);self.assertNotEqual(ad,cd)
 def test_wrong_test_denominator_refused(self):
  s=strict_json(BASE/'payload/EVALUATION_SPEC.json')
  with self.assertRaises(ValueError):an.paired_bootstrap(np.zeros((1250,5)),s,'fault_03')
 def test_wrong_draw_number_refused(self):
  s=strict_json(BASE/'payload/EVALUATION_SPEC.json');s['bootstrap_replicates']=100
  with self.assertRaises(ValueError):an.paired_bootstrap(np.zeros((250,5)),s,'fault_03')
 def test_unregistered_role_refused(self):
  s=strict_json(BASE/'payload/EVALUATION_SPEC.json')
  with self.assertRaises(ValueError):an.paired_bootstrap(np.zeros((250,5)),s,'choose_best_seed')
 def test_projection_exact_backbone(self):
  rng=np.random.default_rng(3);a=rng.normal(size=(2,960,52));p={'mean':np.zeros(52),'scale':np.ones(52),'loadings':np.eye(52)[:36]};basis=np.eye(52)[:,36:]
  z=science.project(a,p,basis);np.testing.assert_array_equal(z['z'],a[:,:,36:]);np.testing.assert_allclose(z['Q'],(a[:,:,36:]**2).sum(2))
 def test_invalid_raw_shape_refused(self):
  with self.assertRaises(ValueError):science.project(np.zeros((2,100,52)),{},None)
 def test_scientific_module_bytes_preserved(self):
  lock=strict_json(BASE/'payload/reference/STAGE_B_ALL_MODELS_LOCK.json')
  for n,d in lock['binding']['scientific_sources'].items():self.assertEqual(sha(BASE/'payload/code'/n),d)

class GateTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.base=Path(self.t.name);self.root=self.base/'study';self.root.mkdir();self.payload=self.base/'payload';shutil.copytree(BASE/'payload',self.payload)
  self.b=strict_json(self.payload/'BINDING.json');self.b['root']=str(self.root);write_json(self.payload/'BINDING.json',self.b,replace=True)
  shutil.copytree(A/'release',self.root/'release')
  for n in self.b['stage_a_control_hashes']:shutil.copyfile(BT/'payload/reference'/n,self.root/n)
  write_json(self.root/'ACTIVE_SESSION.json',{'active':False});(self.root/'attempts').mkdir()
  self.broot=self.root/self.b['stage_b_subdirectory'];shutil.copytree(BR,self.broot);(self.broot/'attempts').mkdir();(self.broot/'archive').mkdir()
  self.croot=self.root/self.b['stage_c_subdirectory'];shutil.copytree(CR,self.croot);(self.croot/'attempts').mkdir();(self.croot/'archive').mkdir()
 def tearDown(self):self.t.cleanup()
 def gate(self):return inp.gate(self.root,self.payload)
 def test_gate_actual_control_and_model_bytes(self):
  _,_,_,rows,ix=self.gate();self.assertEqual(len(rows),1000);self.assertEqual(len(ix),1000)
 def test_missing_lock_refused(self):
  (self.broot/'ALL_MODELS_LOCK.json').unlink()
  with self.assertRaises((ValueError,FileNotFoundError)):self.gate()
 def test_changed_model_refused(self):
  (self.broot/'models/rep01/F03/MODEL.npz').write_bytes(b'changed')
  with self.assertRaises(ValueError):self.gate()
 def test_changed_seed_index_refused(self):
  (self.root/'SEED_MANIFEST.json').write_bytes(b'[]\n')
  with self.assertRaises(ValueError):self.gate()
 def test_changed_panel_index_refused(self):
  with (self.croot/'TEST_COMPLETED_INDEX.csv').open('ab') as f:f.write(b'\n')
  with self.assertRaises(ValueError):self.gate()
 def test_active_C_refused(self):
  write_json(self.croot/'ACTIVE_SESSION.json',{'active':True},replace=True)
  with self.assertRaises(ValueError):self.gate()
 def test_active_B_refused(self):
  write_json(self.broot/'ACTIVE_SESSION.json',{'active':True},replace=True)
  with self.assertRaises(ValueError):self.gate()
 def test_active_A_refused(self):
  write_json(self.root/'ACTIVE_SESSION.json',{'active':True},replace=True)
  with self.assertRaises(ValueError):self.gate()
 def test_unfinished_C_refused(self):
  (self.croot/'attempts/unfinished').mkdir()
  with self.assertRaises(ValueError):self.gate()
 def test_prior_failure_refused(self):
  write_json(self.croot/'SESSION_ERROR.json',{'error':'synthetic'})
  with self.assertRaises(ValueError):self.gate()
 def test_changed_scoring_source_refused(self):
  with (self.payload/'code/science.py').open('ab') as f:f.write(b'\n')
  with self.assertRaises(ValueError):self.gate()
 def test_missing_committed_raw_refused(self):
  _,_,_,rows,index=self.gate()
  with self.assertRaises(ValueError):inp.validate_row(self.root,self.croot,rows[0],index[rows[0]['canonical_label']],True)
 def test_symlink_input_refused(self):
  _,_,_,rows,index=self.gate();p=self.root/rows[0]['relative'];p.parent.mkdir(parents=True);p.symlink_to(self.base,target_is_directory=True)
  with self.assertRaises(ValueError):inp.validate_row(self.root,self.croot,rows[0],index[rows[0]['canonical_label']],True)
 def test_correct_synthetic_raw_schema_and_corruption(self):
  _,_,_,rows,_=self.gate();row=rows[0];p=self.root/row['relative'];p.mkdir(parents=True);f=inp.cfreeze(self.croot);a=np.zeros((960,52));np.savez_compressed(p/'OUTPUT.npz',values=a,xmv12=np.zeros(960),native_time_seconds=np.arange(961,dtype=float)*180)
  task={'row':row,'payload_manifest_sha256':f['payload_manifest_sha256'],'seed_manifest_sha256':f['seed_manifest_sha256'],'all_models_lock_sha256':f['all_models_lock_sha256'],'stage_c_freeze_sha256':f['_file_sha256']};write_json(p/'TASK.json',task)
  m={**{k:row[k] for k in ('canonical_label','seed_u32','stage','role','fault','run','replication')},'study_id':inp.STUDY,'all_models_lock_sha256':f['all_models_lock_sha256'],'shutdown_occurred':False,'backend':'python','numpy':'2.2.6','tep_version':'2.5.0','preonset_sha256':hashlib.sha256(a[:160].astype('<f8').tobytes()).hexdigest(),'started_utc':'2026-09-15T00:00:00+00:00','ended_utc':'2026-09-15T00:00:01+00:00'};write_json(p/'METADATA.json',m)
  co={**{k:row[k] for k in ('canonical_label','seed_u32','stage')},'status':'COMPUTATION_COMPLETE','output_sha256':sha(p/'OUTPUT.npz'),'metadata_sha256':sha(p/'METADATA.json'),'task_sha256':sha(p/'TASK.json'),'time_max_error_seconds':0.};write_json(p/'COMPLETE.json',co);seal_tree(p)
  ix={**{k:m[k] for k in ('started_utc','ended_utc','preonset_sha256','all_models_lock_sha256')},'shutdown_occurred':'False','files_manifest_sha256':sha(p/'FILES.json'),'output_sha256':sha(p/'OUTPUT.npz'),'metadata_sha256':sha(p/'METADATA.json'),'complete_sha256':sha(p/'COMPLETE.json')}
  self.assertEqual(inp.validate_row(self.root,self.croot,row,ix,True)['fault'],row['fault'])
  with (p/'OUTPUT.npz').open('ab') as h:h.write(b'bad')
  with self.assertRaises(ValueError):inp.validate_row(self.root,self.croot,row,ix,True)

class TransactionTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.base=Path(self.t.name);self.work=self.base/'evaluation';self.work.mkdir()
  for n in ('attempts','archive','evaluated','stop_requests'):(self.work/n).mkdir()
  self.ledger=run.DLedger(self.work/'EVENTS.jsonl');self.ledger.add('SETUP');self.bindings={'synthetic':True}
 def tearDown(self):self.t.cleanup()
 def test_stop_only_current_active_session(self):
  sid='20260915T000000Z_abcdef12';write_json(self.work/'ACTIVE_SESSION.json',{'active':True,'session':sid});self.assertIn('STOP_REQUESTED',run.request_stop(self.work));self.assertTrue((self.work/'stop_requests'/(sid+'.json')).exists())
  write_json(self.work/'ACTIVE_SESSION.json',{'active':False,'session':sid},replace=True);self.assertEqual(run.request_stop(self.work),'NO_ACTIVE_STAGE_D_SESSION')
 def test_stop_path_injection_refused(self):
  write_json(self.work/'ACTIVE_SESSION.json',{'active':True,'session':'../bad'})
  with self.assertRaises(ValueError):run.request_stop(self.work)
 def test_interrupted_same_scoring_task_archived(self):
  p=self.work/'attempts/rep01_x';p.mkdir();write_json(p/'TASK.json',{'fixture':'same original inputs'});self.ledger.add('EVALUATE_START',replication=1,attempt='attempts/rep01_x')
  run.recover(self.work,self.ledger,self.bindings);self.assertTrue((self.work/'archive/rep01_x/TASK.json').exists());self.assertEqual(self.ledger.records[-1]['event'],'INTERRUPTED_SAME_TASK')
 def test_explicit_failure_not_retried(self):
  p=self.work/'attempts/rep01_x';p.mkdir();write_json(p/'FAILED.json',{'failure':'synthetic'});self.ledger.add('EVALUATE_START',replication=1,attempt='attempts/rep01_x')
  with self.assertRaises(ValueError):run.recover(self.work,self.ledger,self.bindings)
  self.assertTrue((p/'FAILED.json').exists())
 def test_missing_committed_result_refused(self):
  self.ledger.add('REP_COMPLETE',replication=1,manifest_sha256='f'*64)
  with self.assertRaises(ValueError):run.recover(self.work,self.ledger,self.bindings)
 def test_unknown_unrecorded_attempt_refused(self):
  (self.work/'attempts/unknown').mkdir()
  with self.assertRaises(ValueError):run.recover(self.work,self.ledger,self.bindings)
 def test_changed_ledger_refused(self):
  with (self.work/'EVENTS.jsonl').open('ab') as f:f.write(b'broken')
  with self.assertRaises(ValueError):run.DLedger(self.work/'EVENTS.jsonl',True)

if __name__=='__main__':unittest.main(verbosity=2)
