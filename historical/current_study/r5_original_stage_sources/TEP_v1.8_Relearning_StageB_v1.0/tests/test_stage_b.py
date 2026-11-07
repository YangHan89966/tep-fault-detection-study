"""Synthetic-only software tests. Never accesses the user's study root or simulator."""
import unittest,tempfile,sys,json,csv,hashlib,shutil,io,contextlib,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'payload/code'))
import runner_stage_b as b
import science
from storage import sha,write_json,seal_tree,strict_json,verify_tree,canonical

class StageBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='TEP_STAGE_B_SYNTHETIC_ONLY_');cls.root=Path(cls.temp.name)
        cls.work=cls.root/'work';cls.work.mkdir()
        for sub in ('models','attempts','archive','stop_requests'):(cls.work/sub).mkdir()
        cls.binding={'synthetic_test_only':True,'fixed_seed':18372}
        cls.ledger=b.BLedger(cls.work/'EVENTS.jsonl')
        cls.loads=[]
        def synth(root,index,rep,role,fault,pca,basis):
            cls.loads.append((rep,role,fault))
            n=b.ROLES[role,fault]
            seed=int.from_bytes(hashlib.sha256(f'{rep}|{role}|{fault}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(seed);z=rng.normal(size=(n,960,16))
            if fault:
                z[:,160:,0]+=.25+.15*(fault==3)
                z[:,160:,1]*=1.3 if fault==15 else 1.05
            return {'z':z,'Q':np.square(z).sum(2),'run_id':np.arange(1,n+1),'record_suffix_start':np.full(n,960,dtype=int),'record_suffix_samples':np.ones(n,dtype=int)}
        cls.synth=staticmethod(synth)
        with patch.object(b,'load_role',side_effect=synth),contextlib.redirect_stdout(io.StringIO()):
            b.process_replication(cls.root,[],1,cls.work,cls.binding,'SYNTHETIC',cls.ledger,None,None,lambda:False)
        cls.model=cls.work/'models/rep01/F03'

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def newdir(self):
        p=self.root/('test_'+__import__('uuid').uuid4().hex);p.mkdir();return p

    def modelcopy(self):
        p=self.newdir()/'F03';shutil.copytree(self.model,p);return p

    def fixture(self):
        root=self.newdir();row={'canonical_label':'SYNTHETIC_ONLY','replication':1,'role':'normal_design','fault':0,'run':1,'seed_u32':123,'stage':'development','relative':'data/rep01/normal_design/F00/R0001'}
        p=root/row['relative'];p.mkdir(parents=True)
        rng=np.random.default_rng(37);values=rng.normal(size=(960,52));pre=hashlib.sha256(values[:160].astype('<f8').tobytes()).hexdigest()
        np.savez_compressed(p/'OUTPUT.npz',values=values,xmv12=np.zeros(960),native_time_seconds=np.arange(961)*180.)
        freeze={'study_id':'SYNTHETIC_ONLY','release_manifest_sha256':'a'*64,'seed_manifest_sha256':'b'*64}
        write_json(p/'TASK.json',{'row':row,**{k:freeze[k] for k in ('release_manifest_sha256','seed_manifest_sha256')}})
        meta={**row,'study_id':'SYNTHETIC_ONLY','backend':'python','numpy':'2.2.6','tep_version':'2.5.0','shutdown_occurred':False,'preonset_sha256':pre}
        write_json(p/'METADATA.json',meta)
        comp={**{k:row[k] for k in ('canonical_label','seed_u32','stage')},'status':'COMPUTATION_COMPLETE','output_sha256':sha(p/'OUTPUT.npz'),'metadata_sha256':sha(p/'METADATA.json'),'task_sha256':sha(p/'TASK.json'),'time_max_error_seconds':0.}
        write_json(p/'COMPLETE.json',comp);seal_tree(p)
        ix={**row,'preonset_sha256':pre,'output_sha256':comp['output_sha256'],'metadata_sha256':comp['metadata_sha256'],'complete_sha256':sha(p/'COMPLETE.json'),'files_manifest_sha256':sha(p/'FILES.json')}
        return root,row,ix,freeze,p

    def test_01_production_index_counts(self):self.assertEqual(len(b.read_index(BASE/'payload/reference/COMPLETED_INDEX.csv')),4575)
    def test_02_duplicate_index_rejected(self):
        rows=list(csv.DictReader(io.StringIO((BASE/'payload/reference/COMPLETED_INDEX.csv').read_text())));rows[-1]=rows[0]
        p=self.newdir()/'index.csv'
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with self.assertRaises(ValueError):b.read_index(p)
    def test_03_missing_index_row_rejected(self):
        p=self.newdir()/'index.csv';s=(BASE/'payload/reference/COMPLETED_INDEX.csv').read_text().splitlines();p.write_text('\n'.join(s[:-1])+'\n')
        with self.assertRaises(ValueError):b.read_index(p)
    def test_04_raw_leaf_pass(self):
        root,row,ix,freeze,p=self.fixture();b.validate_leaf(root,ix,row,freeze)
    def test_05_corrupted_raw_rejected(self):
        root,row,ix,freeze,p=self.fixture();(p/'OUTPUT.npz').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):b.validate_leaf(root,ix,row,freeze)
    def test_06_symlink_rejected(self):
        root,row,ix,freeze,p=self.fixture();(p/'OUTPUT.npz').rename(p/'hidden.npz');(p/'OUTPUT.npz').symlink_to('hidden.npz')
        with self.assertRaises(ValueError):b.validate_leaf(root,ix,row,freeze)
    def test_07_failed_leaf_rejected(self):
        root,row,ix,freeze,p=self.fixture();write_json(p/'FAILED.json',{'synthetic':True})
        with self.assertRaises(ValueError):b.validate_leaf(root,ix,row,freeze)
    def test_08_synthetic_learning_has_27_configs(self):self.assertEqual(len(b.check_replication(self.work/'models/rep01',1,self.binding)),27)
    def test_09_selection_precedes_outer_read(self):
        outer=self.loads.index((1,'normal_outer_calibration',0))
        self.assertTrue(all(self.loads.index((1,'fault_selection',f))<outer for f in b.FAULTS))
    def test_10_choices_integer_witness(self):self.assertEqual(len(b.model_check(self.model,3)),9)
    def test_11_selected_choice_tamper_rejected(self):
        p=self.modelcopy();obj=strict_json(p/'SELECTION.json');obj['choices']['mean_pool_signed']=(obj['choices']['mean_pool_signed']+1)%8
        (p/'SELECTION.json').write_bytes(canonical(obj))
        with self.assertRaises(ValueError):b.model_check(p,3)
    def test_12_threshold_tamper_rejected(self):
        p=self.modelcopy();o=science.load_npz(p/'OUTER.npz');o['thresholds'][0]+=1;np.savez_compressed(p/'OUTER.npz',**o)
        with self.assertRaises(ValueError):b.model_check(p,3)
    def test_13_rank_tamper_rejected(self):
        p=self.modelcopy();o=science.load_npz(p/'OUTER.npz');o['rank']=np.array(246);np.savez_compressed(p/'OUTER.npz',**o)
        with self.assertRaises(ValueError):b.model_check(p,3)
    def test_14_same_axis_links(self):
        s=strict_json(self.model/'SELECTION.json');a=science.load_npz(self.model/'MODEL.npz');k=len(a['mean'])+len(a['moment']);j=s['choices']['mean_pool_signed']
        self.assertEqual(s['choices']['same_mean_signed_axis_absolute'],k+j);self.assertEqual(s['choices']['same_mean_signed_axis_squared'],2*k+j)
    def test_15_scalar_recurrence_witness(self):
        r=strict_json(self.model/'SCALAR_RECURRENCE_CHECK.json');self.assertGreater(r['comparisons'],0);self.assertLessEqual(r['max_absolute_error'],1e-10)
    def test_16_resume_locked_does_not_fit(self):
        before={p.relative_to(self.work/'models').as_posix():sha(p) for p in (self.work/'models').rglob('*') if p.is_file()}
        with patch.object(b,'load_role',side_effect=AssertionError('No reads on locked reuse')),contextlib.redirect_stdout(io.StringIO()):
            b.process_replication(self.root,[],1,self.work,self.binding,'SYNTHETIC_RESUME',self.ledger,None,None,lambda:False)
        self.assertEqual(before,{p.relative_to(self.work/'models').as_posix():sha(p) for p in (self.work/'models').rglob('*') if p.is_file()})
    def test_17_stop_before_next_fit(self):
        with self.assertRaises(b.StopRequested):b.process_replication(self.root,[],2,self.work,self.binding,'SYNTHETIC_STOP',self.ledger,None,None,lambda:True)
        self.assertFalse((self.work/'models/rep02').exists())
    def test_18_global_lock_requires_five(self):
        with self.assertRaises(ValueError):b.finalize(self.work,self.binding,self.ledger)
    def test_19_wrong_binding_rejected(self):
        with self.assertRaises(ValueError):b.check_replication(self.work/'models/rep01',1,{'wrong':True})
    def test_20_global_lock_all_135_and_idempotence(self):
        work=self.newdir()
        for sub in ('models','attempts','archive','stop_requests'):(work/sub).mkdir()
        for rep in range(1,6):
            target=work/'models'/f'rep{rep:02d}';shutil.copytree(self.work/'models/rep01',target)
            obj=strict_json(target/'REPLICATION_LOCK.json');obj['replication']=rep;(target/'REPLICATION_LOCK.json').write_bytes(canonical(obj));(target/'FILES.json').unlink();seal_tree(target)
        # Duplicate synthetic models test aggregation and lock mechanics ONLY, not independent scientific repeats.
        led=b.BLedger(work/'EVENTS.jsonl')
        lock=b.finalize(work,self.binding,led);self.assertEqual(lock['configuration_count'],135);self.assertEqual(lock['test_trajectories_generated'],0)
        before=sha(work/'ALL_MODELS_LOCK.json');b.finalize(work,self.binding,led);self.assertEqual(before,sha(work/'ALL_MODELS_LOCK.json'))
        self.assertEqual(sum(x['event']=='ALL_MODELS_LOCKED' for x in led.records),1)
    def test_21_partial_interruption_preserved(self):
        work=self.newdir()
        for sub in ('models','attempts','archive'):(work/sub).mkdir()
        a=work/'attempts/rep01_SYNTHETIC';a.mkdir();write_json(a/'ATTEMPT.json',{'synthetic_only':True})
        led=b.BLedger(work/'EVENTS.jsonl');b.recover_attempts(work,self.binding,led)
        self.assertTrue((work/'archive/rep01_SYNTHETIC/ATTEMPT.json').exists());self.assertFalse(a.exists())
        self.assertEqual(led.records[-1]['event'],'INTERRUPTED_RELEARN_SAME_INPUTS')
    def test_22_numerical_failure_not_automatically_retried(self):
        work=self.newdir()
        for sub in ('models','attempts','archive'):(work/sub).mkdir()
        a=work/'attempts/rep01_SYNTHETIC';a.mkdir();write_json(a/'FAILED.json',{'synthetic_only':True})
        with self.assertRaises(ValueError):b.recover_attempts(work,self.binding,b.BLedger(work/'EVENTS.jsonl'))
        self.assertTrue((a/'FAILED.json').exists())
    def test_23_test_role_cannot_be_loaded(self):
        with self.assertRaises(ValueError):b.load_role(self.root,[],0,'normal_evaluation',0,None,None)
    def test_24_no_simulator_module_imported(self):self.assertFalse(any(x=='tep' or x.startswith('tep.') for x in sys.modules))
    def test_25_source_unchanged(self):
        binding=strict_json(BASE/'payload/BINDING.json')
        for name,digest in binding['unchanged_scientific_source'].items():self.assertEqual(sha(BASE/'payload/code'/name),digest)
    def test_26_hash_chain_detects_tamper(self):
        p=self.newdir()/'EVENTS.jsonl';led=b.BLedger(p);led.add('MODEL_START',replication=1)
        raw=p.read_bytes().replace(b'"replication":1',b'"replication":2');p.write_bytes(raw)
        with self.assertRaises(ValueError):b.BLedger(p)
    def test_27_project_reuses_actual_fixed_pca(self):
        pca=science.load_npz(BASE/'payload/assets/PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz');basis=np.load(BASE/'payload/assets/RESIDUAL_BASIS_NEWSEED_PCA.npy',allow_pickle=False)
        raw=np.random.default_rng(551).normal(size=(2,960,52))*pca['scale']+pca['mean']
        obj=science.project(raw,pca,basis);self.assertEqual(obj['z'].shape,(2,960,16));self.assertTrue(np.allclose((obj['z']**2).sum(2),obj['Q'],rtol=1e-9,atol=1e-9))
    def test_28_manifest_detects_extra_file(self):
        p=self.newdir();write_json(p/'a.json',{'x':1});seal_tree(p);(p/'extra.txt').write_text('synthetic')
        with self.assertRaises(ValueError):verify_tree(p)
    def test_29_complete_replication_recovery_preserves_hashes(self):
        work=self.newdir()
        for sub in ('models','attempts','archive'):(work/sub).mkdir()
        a=work/'attempts/rep01_SYNTHETIC';shutil.copytree(self.work/'models/rep01',a);before=sha(a/'FILES.json');led=b.BLedger(work/'EVENTS.jsonl')
        b.recover_attempts(work,self.binding,led);self.assertEqual(sha(work/'models/rep01/FILES.json'),before)
    def test_30_stop_request_session_scoped(self):
        work=self.newdir();(work/'stop_requests').mkdir();write_json(work/'ACTIVE_SESSION.json',{'active':True,'session':'20260914T000000Z_abcdef12'})
        with contextlib.redirect_stdout(io.StringIO()):b.request_stop(self.root,work)
        self.assertTrue((work/'stop_requests/20260914T000000Z_abcdef12.json').exists())

    def test_31_full_main_and_return_with_five_synthetic_fits(self):
        area=self.newdir();root=area/'SYNTHETIC_STUDY_ONLY';root.mkdir();root.with_name(root.name+'.operation.lock').touch()
        payload=area/'payload';payload.mkdir();(payload/'assets').mkdir()
        bind=strict_json(BASE/'payload/BINDING.json');bind['root']=str(root)
        write_json(payload/'BINDING.json',bind)
        freeze=strict_json(BASE/'payload/reference/FREEZE.json')
        for name in ('FREEZE.json','EVENTS.jsonl'):shutil.copyfile(BASE/'payload/reference'/name,root/name)
        for p in (BASE/'payload/assets').iterdir():shutil.copyfile(p,payload/'assets'/p.name)
        seal_tree(payload,'MANIFEST.json')
        out=area/'output';index=b.read_index(BASE/'payload/reference/COMPLETED_INDEX.csv')
        inputs={'synthetic_fixture':True,'not_user_data':True,'raw_trajectories_verified':0}
        with patch.object(b,'PAYLOAD',payload),patch.object(b,'environment',return_value={'synthetic_environment_only':True}),patch.object(b,'root_baseline',return_value=(index,{},freeze)),patch.object(b,'verify_all_inputs',return_value=inputs),patch.object(b,'load_role',side_effect=self.synth),patch.object(sys,'argv',['runner_stage_b.py','--output-parent',str(out)]),contextlib.redirect_stdout(io.StringIO()):
            rc=b.main()
        self.assertEqual(rc,0)
        work=root/bind['work_subdirectory'];lock=strict_json(work/'ALL_MODELS_LOCK.json')
        self.assertEqual(lock['replication_count'],5);self.assertEqual(lock['configuration_count'],135)
        self.assertEqual(lock['test_trajectories_generated'],0);self.assertFalse((root/'data/shared_test').exists())
        zips=list(out.glob('*_RETURN.zip'));self.assertEqual(len(zips),1)
        with __import__('zipfile').ZipFile(zips[0]) as z:
            self.assertIsNone(z.testzip());status=json.loads(z.read('SESSION_RESULT.json'))
            self.assertEqual(status['status'],'ALL_135_CONFIGURATIONS_LOCKED_NO_TEST_GENERATED')
            self.assertEqual(sum(n.endswith('/REPLICATION_LOCK.json') for n in z.namelist()),5)
        before=sha(work/'ALL_MODELS_LOCK.json')
        with patch.object(b,'PAYLOAD',payload),patch.object(b,'environment',return_value={'synthetic_environment_only':True}),patch.object(b,'root_baseline',return_value=(index,{},freeze)),patch.object(b,'verify_all_inputs',return_value=inputs),patch.object(b,'load_role',side_effect=AssertionError('No refitting after lock')),patch.object(sys,'argv',['runner_stage_b.py','--output-parent',str(out)]),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(b.main(),0)
        self.assertEqual(before,sha(work/'ALL_MODELS_LOCK.json'))

if __name__=='__main__':unittest.main(verbosity=2)
