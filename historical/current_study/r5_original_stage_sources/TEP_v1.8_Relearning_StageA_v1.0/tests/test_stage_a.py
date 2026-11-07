"""Synthetic-only operational tests. Never import or execute the TEP simulator.
Run: python -B tests/test_stage_a.py
Fixtures live in TemporaryDirectory, not the user's study root.
"""
from __future__ import annotations
import ast,csv,fcntl,hashlib,io,json,os,shutil,subprocess,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
D=Path(__file__).resolve().parents[1];R=D/'release'
sys.path.insert(0,str(R/'code'))
import storage as st
import seed_allocation as se
import runner_stage_a as run
import worker
import horizon

DV=st.strict_json(R/'provenance/ORIGINAL_DV_SEED_MANIFEST.json')
DV_STATES={r['effective_initial_state'] for r in DV}
FAKE_OLD=set();i=1
while len(FAKE_OLD)<9500:
    if i not in DV_STATES:FAKE_OLD.add(i)
    i+=1
FAKE_EXCLUDED=FAKE_OLD|DV_STATES
ROWS,AUDIT=se.allocate(FAKE_EXCLUDED) # SOFTWARE FIXTURE ALLOCATIONS, NOT THE HOST MANIFEST.


def make_root(base):
    root=Path(base)/'SOFTWARE_FIXTURE_ONLY'
    with patch.object(run,'previous_states',return_value=(FAKE_EXCLUDED,{'software_fixture':True})):
        run.init_root(root,R,{'software_fixture':True,'container_python':sys.version})
    return root


def make_attempt(root,row,name='attempt_fixture',ledger=None):
    attempt=root/'attempts'/name;attempt.mkdir()
    f=st.strict_json(root/'FREEZE.json')
    task={'root':str(root),'row':row,'session':'20260913T000000Z_abcdef12',
      'release_manifest_sha256':f['release_manifest_sha256'],'seed_manifest_sha256':f['seed_manifest_sha256']}
    st.write_json(attempt/'TASK.json',task)
    if ledger:ledger.add('ATTEMPT_START',label=row['canonical_label'],attempt=attempt.relative_to(root).as_posix())
    return attempt


def fake_output(attempt,fail=False):
    row=st.strict_json(attempt/'TASK.json')['row']
    if fail:
        st.write_json(attempt/'FAILED.json',{'software_fixture':True,'error':'injected arithmetic failure'})
        return
    values=np.zeros((960,52),dtype=np.float64);values[:,0]=np.arange(960)
    np.savez_compressed(attempt/'OUTPUT.npz',values=values,xmv12=np.zeros(960),native_time_seconds=np.arange(961,dtype=float)*180.)
    meta={k:row[k] for k in ('canonical_label','replication','role','fault','run','seed_u32','stage')}
    meta.update({'study_id':se.STUDY,'software_fixture':True,'wall_seconds':.01,'shutdown_occurred':False,
      'backend':'python','numpy':'2.2.6','tep_version':'2.5.0','started_utc':st.now(),'ended_utc':st.now(),
      'preonset_sha256':hashlib.sha256(values[:160].astype('<f8').tobytes()).hexdigest()})
    st.write_json(attempt/'METADATA.json',meta)
    st.write_json(attempt/'COMPLETE.json',{'status':'COMPUTATION_COMPLETE','canonical_label':row['canonical_label'],
      'seed_u32':row['seed_u32'],'stage':row['stage'],'output_sha256':st.sha(attempt/'OUTPUT.npz'),
      'metadata_sha256':st.sha(attempt/'METADATA.json'),'task_sha256':st.sha(attempt/'TASK.json'),
      'time_max_error_seconds':0.,'software_fixture':True})


class Tests(unittest.TestCase):
    def test_01_no_tep_imports(self):
        self.assertFalse(any(n=='tep' or n.startswith('tep.') for n in sys.modules))

    def test_02_python312_syntax(self):
        for p in R.rglob('*.py'):ast.parse(p.read_text(),filename=str(p),feature_version=(3,12))
        ast.parse((D/'bootstrap.py').read_text(),feature_version=(3,12))

    def test_03_release_and_original_numeric_hashes(self):
        st.verify_tree(R,'RELEASE_MANIFEST.json')
        ref=st.strict_json(R/'provenance/UNCHANGED_SCIENTIFIC_SOURCE_CHECK.json')
        self.assertEqual(ref['count'],24)
        for e in ref['entries']:self.assertEqual(st.sha(R/e['path']),e['sha256'])

    def test_04_original_dv_derivations(self):
        for r in DV:
            v,h=se.candidate(r['canonical_label'],r['collision_counter'])
            self.assertEqual((v,h),(r['seed_u32'],r['derivation_sha256']))
        self.assertEqual(len(DV_STATES),1915)

    def test_05_new_fixture_manifest_counts_disjoint_and_reproducible(self):
        se.validate_manifest(ROWS)
        again,audit=se.allocate(FAKE_EXCLUDED)
        self.assertEqual(again,ROWS);self.assertEqual(audit,AUDIT)
        self.assertEqual(len({r['seed_u32'] for r in ROWS}),5575)
        self.assertFalse({r['seed_u32'] for r in ROWS}&FAKE_EXCLUDED)
        self.assertEqual(sum(r['stage']=='development' for r in ROWS),4575)
        self.assertEqual(sum(r['stage']=='evaluation' for r in ROWS),1000)
        for rep in range(1,6):self.assertEqual(sum(r['replication']==rep for r in ROWS),915)

    def test_06_collision_resolution_only_changes_counter(self):
        first=ROWS[0];excluded=set(FAKE_EXCLUDED);excluded.remove(min(FAKE_OLD));excluded.add(first['seed_u32'])
        rows,audit=se.allocate(excluded)
        self.assertEqual(rows[0]['canonical_label'],first['canonical_label'])
        self.assertGreater(rows[0]['collision_counter'],first['collision_counter'])
        self.assertGreater(audit['rejected_hash_candidates'],0)
        self.assertNotIn(rows[0]['seed_u32'],excluded)

    def test_07_previous_index_real_parser_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);(q/'provenance').mkdir();shutil.copyfile(R/'provenance/ORIGINAL_DV_SEED_MANIFEST.json',q/'provenance/ORIGINAL_DV_SEED_MANIFEST.json')
            old=q/'FAKE_OLD_9500.csv';b=st.strict_json(R/'LOCAL_BINDING.json')
            with old.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['adapter_scalar_seed_u32','canonical_label','role']);seeds=iter(sorted(FAKE_OLD));index=0
                for role,n in [('normal_training',500),('normal_validation',250),('normal_evaluation',250),('fault_development',4250),('fault_final',4250)]:
                    for _ in range(n):index+=1;w.writerow([next(seeds),'SYNTHETIC'+str(index),role])
            b.update(prior_9500_index=str(old),prior_9500_bytes=old.stat().st_size,prior_9500_sha256=st.sha(old))
            st.write_json(q/'LOCAL_BINDING.json',b)
            found,_=se.previous_states(q);self.assertEqual(found,FAKE_EXCLUDED)
            old.write_bytes(old.read_bytes()+b'CORRUPTION')
            with self.assertRaises(ValueError):se.previous_states(q)

    def test_08_freeze_resume_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);before={n:st.sha(root/n) for n in ('FREEZE.json','SEED_MANIFEST.json','SEED_EXCLUSION.json')}
            with patch.object(run,'previous_states',return_value=(FAKE_EXCLUDED,{'software_fixture':True})):
                run.init_root(root,R,{'software_fixture':'another host inventory does not refreeze'})
            self.assertEqual(before,{n:st.sha(root/n) for n in before})
            self.assertFalse(list((root/'data').rglob('OUTPUT.npz')))
            (root/'SEED_MANIFEST.json').write_bytes(b'[]\n')
            with patch.object(run,'previous_states',return_value=(FAKE_EXCLUDED,{'software_fixture':True})):
                with self.assertRaises(ValueError):run.init_root(root,R,{})

    def test_09_validate_output_shapes_and_normal_shutdown(self):
        a={'time_seconds':np.arange(961,dtype=float)*180,'measurements':np.zeros((961,41)),
          'manipulated_vars':np.zeros((961,12)),'shutdown_occurred':False}
        values,mv,t,err=worker.validate_arrays(a,0,np)
        self.assertEqual(values.shape,(960,52));self.assertEqual(mv.shape,(960,));self.assertEqual(err,0)
        a['shutdown_occurred']=True
        with self.assertRaises(ValueError):worker.validate_arrays(a,0,np)
        worker.validate_arrays(a,15,np) # valid faulty shutdown not filtered out
        a['measurements'][12,1]=np.nan
        with self.assertRaises(ValueError):worker.validate_arrays(a,15,np)

    def test_10_worker_refuses_evaluation_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);row=next(r for r in ROWS if r['stage']=='evaluation');attempt=make_attempt(root,row)
            with patch.object(worker,'__file__',str(root/'release/code/worker.py')),patch.object(sys,'argv',['worker.py',str(attempt/'TASK.json')]):
                with patch('sys.stderr',io.StringIO()):self.assertEqual(worker.main(),2)
            self.assertIn('STAGE_A_DEVELOPMENT_ONLY',st.strict_json(attempt/'FAILED.json')['error'])
            self.assertFalse((attempt/'OUTPUT.npz').exists())
            self.assertFalse(any(n=='tep' or n.startswith('tep.') for n in sys.modules))

    def test_11_commit_inventory_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);row=ROWS[0]
            attempt=make_attempt(root,row,ledger=ledger);fake_output(attempt);run.promote(root,attempt,row,ledger)
            self.assertEqual(len(run.inventory(root,ROWS,ledger)),1)
            output=root/row['relative']/'OUTPUT.npz';output.write_bytes(output.read_bytes()+b'bad')
            with self.assertRaises(ValueError):run.inventory(root,ROWS,ledger)

    def test_12_deleted_committed_run_not_regenerated(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);row=ROWS[0]
            attempt=make_attempt(root,row,ledger=ledger);fake_output(attempt);run.promote(root,attempt,row,ledger)
            shutil.rmtree(root/row['relative'])
            with self.assertRaises(ValueError):run.inventory(root,ROWS,ledger)

    def test_13_partial_recovery_archives_same_seed_and_complete_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);row=ROWS[0]
            attempt=make_attempt(root,row,ledger=ledger);(attempt/'OUTPUT.npz').write_bytes(b'PARTIAL SOFTWARE FIXTURE')
            before=st.sha(root/'SEED_MANIFEST.json');run.recover(root,ROWS,ledger)
            self.assertFalse(attempt.exists());self.assertTrue((root/'archive/attempt_fixture').exists())
            self.assertEqual(before,st.sha(root/'SEED_MANIFEST.json'))
            attempt=make_attempt(root,row,name='completed_fixture',ledger=ledger);fake_output(attempt)
            run.recover(root,ROWS,ledger);self.assertEqual(len(run.inventory(root,ROWS,ledger)),1)

    def test_14_numerical_failure_never_retried(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);attempt=make_attempt(root,ROWS[0],ledger=ledger)
            fake_output(attempt,fail=True)
            with self.assertRaises(ValueError):run.recover(root,ROWS,ledger)
            self.assertTrue((attempt/'FAILED.json').exists());self.assertFalse(list((root/'data').rglob('OUTPUT.npz')))

    def test_15_torn_ledger_stops_preserving_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger';st.Ledger(path).add('SETUP',software_fixture=True)
            raw=path.read_bytes()+b'{incomplete';path.write_bytes(raw)
            with self.assertRaises(ValueError):st.Ledger(path,True)
            self.assertEqual(path.read_bytes(),raw)

    def test_16_stop_request_is_session_specific(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);session='20260913T000000Z_abcdef12'
            st.write_json(root/'ACTIVE_SESSION.json',{'session':session,'active':True})
            self.assertIn('STOP_REQUESTED',run.request_stop(root))
            self.assertTrue((root/'stop_requests'/(session+'.json')).exists())
            st.write_json(root/'ACTIVE_SESSION.json',{'session':session,'active':False},replace=True)
            self.assertEqual(run.request_stop(root),'NO_ACTIVE_SESSION')

    def test_17_inherited_operation_lock(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lease';f=path.open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(0.3)'],pass_fds=(f.fileno(),));f.close()
            with path.open('a+b') as g:
                with self.assertRaises(BlockingIOError):fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
                child.wait();fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_18_scheduler_mock_commits_and_returns_no_raw_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);done={};short=se.ordered_development(ROWS)[:3];counter=[0]
            class FakeProcess:
                def __init__(self,args,**kw):
                    counter[0]+=1;self.pid=100+counter[0];self.returncode=0
                    self.args=args;self.kw=kw;fake_output(Path(args[-1]).parent)
                    if kw.get('pass_fds') is None:raise AssertionError('lease not passed')
                def poll(self):return self.returncode
                def wait(self):return self.returncode
            with (root/'software_fixture_lock').open('a+b') as lease,patch.object(run,'ordered_development',return_value=short),patch.object(run.subprocess,'Popen',FakeProcess):
                status=run.generate(root,ROWS,done,.01,2,ledger,'20260913T000000Z_abcdef12',lease.fileno())
            self.assertEqual(status,'SESSION_PAUSED_RESUMABLE');self.assertEqual(len(done),3)
            self.assertEqual(len(run.inventory(root,ROWS,ledger)),3)
            out=Path(td)/'exports';zpath=run.return_package(root,out,'20260913T000000Z_abcdef12',status)
            import zipfile
            with zipfile.ZipFile(zpath) as z:
                self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.npz') for n in z.namelist()))
                self.assertEqual(len(list(csv.DictReader(io.StringIO(z.read('COMPLETED_INDEX.csv').decode())))),3)
            self.assertEqual(zpath.with_suffix('.zip.sha256').read_text().split()[0],st.sha(zpath))

    def test_19_scheduler_stop_never_dispatches_next_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);done={};short=se.ordered_development(ROWS)[:10];counter=[0]
            class FakeProcess:
                def __init__(self,args,**kw):
                    counter[0]+=1;self.pid=counter[0];self.returncode=0;fake_output(Path(args[-1]).parent)
                    run.request_stop(root)
                def poll(self):return 0
                def wait(self):return 0
            with (root/'software_fixture_lock').open('a+b') as lease,patch.object(run,'ordered_development',return_value=short),patch.object(run.subprocess,'Popen',FakeProcess):
                run.generate(root,ROWS,done,.01,2,ledger,'20260913T000000Z_abcdef12',lease.fileno())
            self.assertLessEqual(len(done),2);self.assertEqual(len(done),2)

    def test_20_scheduler_deadline_stops_new_launches(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);done={};short=se.ordered_development(ROWS)[:10];count=[0]
            class FakeProcess:
                def __init__(self,args,**kw):
                    count[0]+=1;self.pid=count[0];self.returncode=None;self.attempt=Path(args[-1]).parent;self.polled=0
                def poll(self):
                    self.polled+=1
                    if self.polled==1:return None
                    if self.returncode is None:fake_output(self.attempt);self.returncode=0
                    return self.returncode
                def wait(self):
                    if self.returncode is None:fake_output(self.attempt);self.returncode=0
                    return self.returncode
            with (root/'software_fixture_lock').open('a+b') as lease,patch.object(run,'ordered_development',return_value=short),patch.object(run.subprocess,'Popen',FakeProcess):
                run.generate(root,ROWS,done,0.0001,1,ledger,'20260913T000000Z_abcdef12',lease.fileno())
            self.assertEqual(len(done),1);self.assertEqual(count[0],1)

    def test_21_scheduler_preserves_failure_and_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root=make_root(td);ledger=st.Ledger(root/'EVENTS.jsonl',True);done={};short=se.ordered_development(ROWS)[:10];count=[0]
            class FakeProcess:
                def __init__(self,args,**kw):
                    count[0]+=1;self.pid=count[0];self.returncode=2 if count[0]==1 else 0
                    fake_output(Path(args[-1]).parent,fail=self.returncode!=0)
                def poll(self):return self.returncode
                def wait(self):return self.returncode
            with (root/'software_fixture_lock').open('a+b') as lease,patch.object(run,'ordered_development',return_value=short),patch.object(run.subprocess,'Popen',FakeProcess):
                with self.assertRaises(RuntimeError):run.generate(root,ROWS,done,.01,2,ledger,'20260913T000000Z_abcdef12',lease.fileno())
            self.assertEqual(count[0],2);self.assertEqual(len(done),1)
            self.assertTrue((root/'BLOCKED.json').exists())
            self.assertEqual(len(list((root/'attempts').glob('*/FAILED.json'))),1)

    def test_22_horizon_continues_mock_shutdown_without_padding(self):
        class Process:
            def __init__(self):
                self.n=0;self.state=types.SimpleNamespace(yy=np.zeros(50));self._teproc=types.SimpleNamespace(vlr=10*35.3145,vls=5*35.3145,vlc=4*35.3145)
            def get_xmeas(self):
                a=np.zeros(41);a[0]=self.n;a[6]=3100 if self.n>=100 else 1000;return a
            def get_xmv(self):return np.zeros(12)
            def set_xmv(self,*a):pass
            def evaluate(self,t,state):self.n+=1;return np.zeros(50)
            def is_shutdown(self):return self.n>=100
        class Sim:
            def __init__(self):
                self.initialized=True;self.time=0.;self.dt=1./3600;self.step_count=0;self.process=Process();self.control_mode=types.SimpleNamespace(value='closed_loop')
                self.controller=types.SimpleNamespace(calculate=lambda x,m,n:m);self.faults=[]
            def set_disturbance(self,f,on):self.faults.append((f,self.step_count))
            def set_ground_truth(self,f):pass
        sim=Sim();r=horizon.simulate_registered_horizon(sim,np,.1,9,.05,180)
        self.assertEqual(sim.process.n,360);self.assertTrue(r['shutdown_occurred'])
        self.assertEqual(r['shutdown_step'],100);self.assertEqual(r['measurements'][:,0].tolist(),[0,180,360])
        self.assertEqual(len(sim.faults),1);self.assertEqual(r['measurements'].shape,(3,41))
        self.assertTrue(np.allclose(r['time_seconds'],[0,180,360]))
        self.assertFalse(any(n=='tep' or n.startswith('tep.') for n in sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
