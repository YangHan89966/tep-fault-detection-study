"""Synthetic software tests, not TEP generation or performance tests.
Build-time fixture inputs: TEP_TEST_A_RELEASE and TEP_TEST_B_RETURN (extracted original files).
All writes occur under tempfile. Source models are copied/hashed only, never fitted or scored.
"""
from pathlib import Path
import contextlib,copy,fcntl,hashlib,io,json,os,shutil,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE/'payload/code'))
import common_c as c,runner_stage_c as r
from storage import *
A=Path(os.environ['TEP_TEST_A_RELEASE']);B=Path(os.environ['TEP_TEST_B_RETURN'])
class CTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name);self.root=self.d/'study';self.root.mkdir();self.payload=self.d/'payload'
  shutil.copytree(BASE/'payload',self.payload)
  for p in self.payload.rglob('__pycache__'):shutil.rmtree(p)
  b=strict_json(self.payload/'BINDING.json');b['root']=str(self.root);write_json(self.payload/'BINDING.json',b,replace=True);self.b=b
  (self.payload/'MANIFEST.json').unlink();seal_tree(self.payload,'MANIFEST.json')
  shutil.copytree(A,self.root/'release')
  for n in ('FREEZE.json','SEED_MANIFEST.json','SEED_EXCLUSION.json'):shutil.copyfile(self.payload/'reference'/n,self.root/n)
  # Stage A ledger exact bytes are supplied by the original B tool's reference fixture.
  shutil.copyfile(Path(os.environ['TEP_TEST_A_LEDGER']),self.root/'EVENTS.jsonl')
  write_json(self.root/'ACTIVE_SESSION.json',{'active':False});(self.root/'attempts').mkdir();(self.root/'data').mkdir()
  self.bl=self.root/b['stage_b_subdirectory'];shutil.copytree(B,self.bl)
  for n in ('attempts','archive'):(self.bl/n).mkdir()
  self.work=self.root/b['work_subdirectory'];self.rows=c.test_rows(self.root,b)
  self.old_payload=r.PAYLOAD;r.PAYLOAD=self.payload
  r.setup(self.root,self.work,b,{'synthetic_fixture':True})
  self.led=c.CLedger(self.work/'EVENTS.jsonl',established=True)
  self.lease=self.root.with_name(self.root.name+'.operation.lock').open('a+b');fcntl.flock(self.lease,fcntl.LOCK_EX)
 def tearDown(self):
  r.PAYLOAD=self.old_payload;self.lease.close();self.tmp.cleanup()
 def row(self):return self.rows[0]
 def attempt(self,row=None):
  row=row or self.row();p=self.work/'attempts'/('trial_'+str(len(list((self.work/'attempts').iterdir()))));p.mkdir()
  f=c.cfreeze(self.work);write_json(p/'TASK.json',{'root':str(self.root),'row':row,'payload_manifest_sha256':f['payload_manifest_sha256'],'seed_manifest_sha256':f['seed_manifest_sha256'],'all_models_lock_sha256':f['all_models_lock_sha256'],'stage_c_freeze_sha256':f['_file_sha256'],'operation_lock_fd':self.lease.fileno(),'session':'20260915T010000Z_abcd1234'})
  self.led.add('ATTEMPT_START',label=row['canonical_label'],attempt=p.relative_to(self.work).as_posix(),session='synthetic')
  return p
 def complete(self,p,row=None,shutdown=False):
  row=row or strict_json(p/'TASK.json')['row'];v=np.zeros((960,52));np.savez_compressed(p/'OUTPUT.npz',values=v,xmv12=np.zeros(960),native_time_seconds=np.arange(961,dtype=float)*180)
  m={**{k:row[k] for k in ('canonical_label','replication','role','fault','run','seed_u32','stage')},'study_id':c.STUDY,'all_models_lock_sha256':self.b['all_models_lock_sha256'],'shutdown_occurred':shutdown,'backend':'python','numpy':'2.2.6','tep_version':'2.5.0','preonset_sha256':hashlib.sha256(v[:160].astype('<f8').tobytes()).hexdigest(),'wall_seconds':.01,'started_utc':now(),'ended_utc':now()}
  write_json(p/'METADATA.json',m)
  write_json(p/'COMPLETE.json',{'canonical_label':row['canonical_label'],'seed_u32':row['seed_u32'],'stage':row['stage'],'status':'COMPUTATION_COMPLETE','output_sha256':sha(p/'OUTPUT.npz'),'metadata_sha256':sha(p/'METADATA.json'),'task_sha256':sha(p/'TASK.json'),'time_max_error_seconds':0.})
 def test_gate_actual_returned_model_bytes(self):
  b,f,l=c.gate(self.root,self.payload);self.assertEqual(l['configuration_count'],135)
 def test_missing_global_lock(self):
  (self.bl/'ALL_MODELS_LOCK.json').unlink()
  with self.assertRaises((ValueError,FileNotFoundError)):c.gate(self.root,self.payload)
 def test_altered_model(self):
  (self.bl/'models/rep01/F03/MODEL.npz').write_bytes(b'changed')
  with self.assertRaises(ValueError):c.gate(self.root,self.payload)
 def test_altered_seed_manifest(self):
  with (self.root/'SEED_MANIFEST.json').open('ab') as f:f.write(b' ')
  with self.assertRaises(ValueError):c.test_rows(self.root,self.b)
 def test_gate_a_ledger_changed(self):
  with (self.root/'EVENTS.jsonl').open('ab') as f:f.write(b'\n')
  with self.assertRaises(ValueError):c.gate(self.root,self.payload)
 def test_gate_b_active(self):
  write_json(self.bl/'ACTIVE_SESSION.json',{'active':True},replace=True)
  with self.assertRaises(ValueError):c.gate(self.root,self.payload)
 def test_gate_a_active(self):
  write_json(self.root/'ACTIVE_SESSION.json',{'active':True},replace=True)
  with self.assertRaises(ValueError):c.gate(self.root,self.payload)
 def test_gate_unfinished_b(self):
  (self.bl/'attempts/unfinished').mkdir()
  with self.assertRaises(ValueError):c.gate(self.root,self.payload)
 def test_manifest_1000_shared_unique(self):
  self.assertEqual(len(self.rows),1000);self.assertEqual(len({x['seed_u32'] for x in self.rows}),1000)
  self.assertEqual([x['fault'] for x in self.rows[:4]],[0,3,9,15]);self.assertEqual({x['replication'] for x in self.rows},{0})
 def test_development_denied(self):
  d=copy.deepcopy(self.row());d['stage']='development';d['replication']=1
  with self.assertRaises(ValueError):c.validate_test_row(d)
 def test_wrong_role_denied(self):
  d=copy.deepcopy(self.row());d['role']='fault_fit'
  with self.assertRaises(ValueError):c.validate_test_row(d)
 def test_output_path_escape_denied(self):
  d=copy.deepcopy(self.row());d['relative']='../outside'
  with self.assertRaises(ValueError):c.validate_test_row(d)
 def test_modified_seed_identity_denied(self):
  d=copy.deepcopy(self.row());d['seed_u32']+=1
  with self.assertRaises(ValueError):c.validate_test_row(d)
 def test_valid_complete_commit_reuse(self):
  p=self.attempt();self.complete(p);r.promote(self.root,self.work,p,self.row(),self.led)
  before=sha(self.root/self.row()['relative']/'OUTPUT.npz')
  done=r.inventory(self.root,self.work,self.rows,self.led)
  self.assertEqual(len(done),1);r.recover(self.root,self.work,self.rows,self.led)
  self.assertEqual(sha(self.root/self.row()['relative']/'OUTPUT.npz'),before)
 def test_missing_committed_refused(self):
  p=self.attempt();self.complete(p);r.promote(self.root,self.work,p,self.row(),self.led);shutil.rmtree(self.root/self.row()['relative'])
  with self.assertRaises(ValueError):r.inventory(self.root,self.work,self.rows,self.led)
 def test_corrupt_output_refused(self):
  p=self.attempt();self.complete(p);(p/'OUTPUT.npz').write_bytes(b'bad')
  with self.assertRaises(ValueError):r.promote(self.root,self.work,p,self.row(),self.led)
 def test_normal_shutdown_preserved(self):
  p=self.attempt();self.complete(p,shutdown=True)
  with self.assertRaises(ValueError):r.promote(self.root,self.work,p,self.row(),self.led)
  self.assertTrue((p/'OUTPUT.npz').exists())
 def test_worker_failure_no_retry(self):
  p=self.attempt();write_json(p/'FAILED.json',{'error':'synthetic numerical failure'})
  with self.assertRaises(ValueError):r.recover(self.root,self.work,self.rows,self.led)
  self.assertTrue(p.exists())
 def test_positive_exit_no_retry(self):
  p=self.attempt();write_json(p/'WORKER_EXIT.json',{'returncode':2})
  with self.assertRaises(ValueError):r.recover(self.root,self.work,self.rows,self.led)
 def test_interruption_same_seed_preserved(self):
  p=self.attempt();task=(p/'TASK.json').read_bytes();write_json(p/'WORKER_EXIT.json',{'returncode':-9})
  r.recover(self.root,self.work,self.rows,self.led)
  self.assertEqual(len(list((self.work/'attempts').iterdir())),0)
  a=next((self.work/'archive').iterdir());self.assertEqual((a/'TASK.json').read_bytes(),task)
  self.assertEqual(self.led.records[-1]['event'],'INTERRUPTED_SAME_SEED_RETRY')
 def test_complete_orphan_recovery(self):
  p=self.attempt();self.complete(p);r.recover(self.root,self.work,self.rows,self.led)
  self.assertEqual(self.led.records[-1]['event'],'RECOVER_COMMIT')
  self.assertEqual(len(r.inventory(self.root,self.work,self.rows,self.led)),1)
 def test_postrename_orphan_recovery(self):
  p=self.attempt();self.complete(p);seal_tree(p);d=self.root/self.row()['relative'];d.parent.mkdir(parents=True);os.rename(p,d)
  r.recover(self.root,self.work,self.rows,self.led)
  self.assertEqual(self.led.records[-1]['event'],'RECOVER_COMMIT')
 def test_duplicate_commit_refused(self):
  p=self.attempt();self.complete(p);r.promote(self.root,self.work,p,self.row(),self.led)
  self.led.add('ATTEMPT_COMMIT',**self.led.records[-1]['data'])
  with self.assertRaises(ValueError):r.inventory(self.root,self.work,self.rows,self.led)
 def test_model_lock_binding_on_completion(self):
  p=self.attempt();self.complete(p);m=strict_json(p/'METADATA.json');m['all_models_lock_sha256']='0'*64;write_json(p/'METADATA.json',m,replace=True)
  with self.assertRaises(ValueError):c.validate_complete(p,self.row(),c.cfreeze(self.work),True)
 def test_stop_requests_current_session_only(self):
  sid='20260915T010000Z_abcd1234';write_json(self.work/'ACTIVE_SESSION.json',{'session':sid,'active':True})
  r.request_stop(self.root,self.work);self.assertTrue((self.work/'stop_requests'/(sid+'.json')).exists())
  write_json(self.work/'ACTIVE_SESSION.json',{'session':sid,'active':False},replace=True)
  self.assertEqual(r.request_stop(self.root,self.work),'NO_ACTIVE_STAGE_C_SESSION')
 def test_stop_before_dispatch(self):
  sid='20260915T010000Z_abcd1234';write_json(self.work/'stop_requests'/(sid+'.json'),{'request':'stop'})
  with patch.object(r.subprocess,'Popen') as pop:
   out=r.generate(self.root,self.work,self.rows,{},2,8,self.led,sid,self.lease.fileno())
   pop.assert_not_called();self.assertEqual(out,'SESSION_PAUSED_RESUMABLE')
 def test_eight_worker_dispatch_and_completion(self):
  test=self;active=[0,0];calls=[]
  class FakeProc:
   def __init__(self,cmd,**kw):
    calls.append((cmd,kw));self.pid=1000+len(calls);self.returncode=None;self.task=Path(cmd[-1]);test.complete(self.task.parent);active[0]+=1;active[1]=max(active);self.done=False
   def poll(self):
    if not self.done:self.returncode=0;self.done=True;active[0]-=1
    return self.returncode
   def wait(self):return self.poll()
  done={}
  with patch.object(r.subprocess,'Popen',FakeProc):
   status=r.generate(self.root,self.work,self.rows[:9],done,2,8,self.led,'20260915T010000Z_abcd1234',self.lease.fileno())
  self.assertEqual(len(done),9);self.assertEqual(active[1],8);self.assertEqual(len(calls),9)
  for cmd,kw in calls:
   self.assertEqual(cmd[1:3],['-I','-B']);self.assertEqual(kw['env']['OPENBLAS_NUM_THREADS'],'1');self.assertEqual(kw['pass_fds'],(self.lease.fileno(),))
 def test_worker_failure_drains_active(self):
  test=self;calls=[]
  class FakeProc:
   def __init__(self,cmd,**kw):
    calls.append(self);self.pid=2000+len(calls);self.returncode=None;self.p=Path(cmd[-1]).parent
    if len(calls)==1:write_json(self.p/'FAILED.json',{'error':'synthetic failure'})
    else:test.complete(self.p)
   def poll(self):self.returncode=2 if self is calls[0] else 0;return self.returncode
   def wait(self):return self.poll()
  done={}
  with patch.object(r.subprocess,'Popen',FakeProc):
   with self.assertRaises(RuntimeError):r.generate(self.root,self.work,self.rows,done,2,8,self.led,'20260915T010000Z_abcd1234',self.lease.fileno())
  self.assertEqual(len(calls),8);self.assertEqual(len(done),7);self.assertTrue((self.work/'BLOCKED.json').exists())
 def test_complete_short_circuit(self):
  with patch.object(r.subprocess,'Popen') as p:
   self.assertEqual(r.generate(self.root,self.work,self.rows,{row['canonical_label']:{} for row in self.rows},2,8,self.led,'s',self.lease.fileno()),'SHARED_TEST_1000_COMPLETE_NOT_EVALUATED');p.assert_not_called()
 def test_return_archive_contains_no_raw_scores(self):
  p=self.attempt();self.complete(p);m=r.promote(self.root,self.work,p,self.row(),self.led)
  r.write_index(self.root,self.work,self.rows,{self.row()['canonical_label']:m})
  out=self.d/'export';r.export(self.work,out,'synthetic','SESSION_PAUSED_RESUMABLE',None)
  import zipfile
  z=next(out.glob('*.zip'))
  self.assertEqual(sha(z),Path(str(z)+'.sha256').read_text().split()[0])
  with zipfile.ZipFile(z) as q:self.assertIsNone(q.testzip());self.assertFalse(any(n.endswith('.npz') for n in q.namelist()))
 def test_setup_no_unaccounted_tests(self):
  shutil.rmtree(self.work);(self.root/'data/shared_test').mkdir()
  with self.assertRaises(ValueError):r.setup(self.root,self.work,self.b,{'synthetic_fixture':True})
 def test_gate_keeps_a_b_bytes_unchanged(self):
  a=sha(self.root/'EVENTS.jsonl');b=sha(self.bl/'ALL_MODELS_LOCK.json');c.gate(self.root,self.payload)
  self.assertEqual(sha(self.root/'EVENTS.jsonl'),a);self.assertEqual(sha(self.bl/'ALL_MODELS_LOCK.json'),b)
 def test_main_all_1000_synthetic_and_resume(self):
  test=self;calls=[]
  class FakeProc:
   def __init__(self,cmd,**kw):
    calls.append(self);self.pid=3000+len(calls);self.returncode=None;test.complete(Path(cmd[-1]).parent)
   def poll(self):self.returncode=0;return self.returncode
   def wait(self):return self.poll()
  # Release the fixture-held lock so the real coordinator can acquire it.
  fcntl.flock(self.lease,fcntl.LOCK_UN)
  args=['runner_stage_c.py','--mode','GenerateTest','--hours','2','--workers','8','--output-parent',str(self.d/'main_return')]
  with patch.object(sys,'argv',args),patch.object(r,'runtime',return_value={'synthetic_fixture':True}),patch.object(r,'gate',side_effect=lambda root:c.gate(root,self.payload)),patch.object(r.subprocess,'Popen',FakeProc),patch.object(r.time,'sleep',return_value=None),contextlib.redirect_stdout(io.StringIO()):
   self.assertEqual(r.main(),0)
   self.assertEqual(len(calls),1000)
   mark=sha(self.work/'PANEL_COMPLETION.json');idx=sha(self.work/'TEST_COMPLETED_INDEX.csv')
   self.assertEqual(r.main(),0)
   self.assertEqual(len(calls),1000)
   self.assertEqual(sha(self.work/'PANEL_COMPLETION.json'),mark);self.assertEqual(sha(self.work/'TEST_COMPLETED_INDEX.csv'),idx)
  self.assertEqual(strict_json(self.work/'PROGRESS_C.json')['test_completed'],1000)
  self.assertEqual(strict_json(self.work/'PANEL_COMPLETION.json')['test_scored'],0)
 def test_incomplete_finalization_refused(self):
  write_csv(self.work/'TEST_COMPLETED_INDEX.csv',[])
  with self.assertRaises(ValueError):r.finalize(self.work,self.led)
 def worker_stub(self,bad=False):
  import types,worker_c
  p=self.attempt();fake=types.ModuleType('tep');fake.__version__='2.5.0';fake.__file__=str(self.payload/'simulator_source/src/tep/__init__.py')
  simmod=types.ModuleType('tep.simulator');calls=[]
  class FakeSimulator:
   def __init__(self,**kw):calls.append(kw)
   def initialize(self):pass
  simmod.TEPSimulator=FakeSimulator;simmod.ControlMode=types.SimpleNamespace(CLOSED_LOOP='closed')
  h=types.ModuleType('horizon')
  def result(sim,np,duration,fault,onset,interval):
   self.assertEqual((duration,fault,onset,interval),(48.,0,None,180))
   return {'time_seconds':np.arange(961,dtype=float)*180,'measurements':np.zeros((961,41)),'manipulated_vars':np.zeros((961,12)),'shutdown_occurred':bad}
  h.simulate_registered_horizon=result
  # Fixture-only version label: computation uses local NumPy; no pinned-runtime claim.
  with patch.object(np,'__version__','2.2.6'),patch.object(worker_c,'__file__',str(self.payload/'code/worker_c.py')),patch.object(worker_c,'runtime',return_value={}),patch.dict(sys.modules,{'tep':fake,'tep.simulator':simmod,'horizon':h}),patch.object(sys,'argv',['worker_c.py',str(p/'TASK.json')]),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   rc=worker_c.main()
  self.assertEqual(calls[0]['random_seed'],self.row()['seed_u32']);self.assertEqual(calls[0]['backend'],'python')
  return p,rc
 def test_worker_wrapper_with_stub_simulator(self):
  p,rc=self.worker_stub();self.assertEqual(rc,0);c.validate_complete(p,self.row(),c.cfreeze(self.work),True)
 def test_worker_preserves_failed_stub_output(self):
  p,rc=self.worker_stub(bad=True);self.assertEqual(rc,2);self.assertTrue((p/'OUTPUT.npz').exists());self.assertTrue((p/'FAILED.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
