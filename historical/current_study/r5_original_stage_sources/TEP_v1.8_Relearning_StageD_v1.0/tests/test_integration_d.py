"""Full runner on synthetic residual inputs, with raw-loading/runtime probes mocked.
Copied frozen models/metadata are software fixtures, never real test trajectories.
"""
from pathlib import Path
import contextlib,csv,hashlib,io,os,sys,unittest
from unittest.mock import patch
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/'tests'));sys.path.insert(0,str(BASE/'payload/code'))
import test_stage_d as fixtures
import analysis_d as an,runner_stage_d as run,science
from storage import *
class Integration(unittest.TestCase):
 def test_synthetic_135_scoring_aggregation_export_and_resume(self):
  f=fixtures.GateTests();f.setUp();old=run.PAYLOAD
  try:
   run.PAYLOAD=f.payload;(f.payload/'MANIFEST.json').unlink();seal_tree(f.payload,'MANIFEST.json')
   f.root.with_name(f.root.name+'.operation.lock').touch();b,broot,croot,rows,index=f.gate()
   original_hashes={p:sha(p) for p in f.root.rglob('*') if p.is_file()}
   objects={}
   for fault in (0,3,9,15):
    rng=np.random.default_rng(881+fault);z=rng.normal(0,1e-5,size=(250,960,16));z[:,160:,:]*=1+fault/10
    objects[fault]={'z':z,'Q':(z*z).sum(2),'run_id':np.arange(1,251),'record_suffix_start':np.full(250,961),'record_suffix_samples':np.zeros(250,int)}
   rawhash={str(f.root/r['relative']/'OUTPUT.npz'):index[r['canonical_label']]['output_sha256'] for r in rows};realsha=run.sha
   def fixture_sha(p):return rawhash.get(str(p)) or realsha(p)
   output=f.base/'returns';work=f.root/b['work_subdirectory'];record=io.StringIO();args=['runner_stage_d.py','--mode','Evaluate','--output-parent',str(output)]
   with patch.object(run,'runtime',return_value={'synthetic_software_fixture':True}),patch.object(run,'verify_raw',return_value=[{'synthetic':True}]*1000),patch.object(run,'read_condition',side_effect=lambda *a:objects[a[4]]),patch.object(run,'sha',side_effect=fixture_sha),patch.object(science,'fit_one',side_effect=AssertionError('forbidden fitting')),patch.object(science,'select_one',side_effect=AssertionError('forbidden selection')),patch.object(science,'calibrate_one',side_effect=AssertionError('forbidden calibration')),patch.object(sys,'argv',args),contextlib.redirect_stdout(record):
    self.assertEqual(run.main(),0)
   self.assertEqual(strict_json(work/'COMPLETION.json')['configurations_scored'],135)
   self.assertEqual(len(list(csv.DictReader(io.StringIO((work/'summary/ALL_135_ENDPOINTS.csv').read_text())))),135)
   self.assertEqual(len(list(csv.DictReader(io.StringIO((work/'summary/ALL_15_PER_FIT_CONTRASTS.csv').read_text())))),15)
   primary=list(csv.DictReader(io.StringIO((work/'summary/PRIMARY_3_FINITE_FIVE_FIT_CONTRASTS.csv').read_text())));self.assertEqual(len(primary),3)
   spec=strict_json(f.payload/'EVALUATION_SPEC.json');vec=science.load_npz(work/'summary/PAIRED_RUN_VECTORS.npz');draw=science.load_npz(work/'summary/BOOTSTRAP_MEANS.npz')
   for row in primary:
    cid=row['contrast'];fault=int(row['fault']);v=vec[cid+'_loss_difference_minutes'];self.assertEqual(v.shape,(250,5));self.assertAlmostEqual(float(row['mean_loss_difference_minutes']),float(v.mean()),places=10)
    seed=int.from_bytes(hashlib.sha256((spec['rng_namespace']+f'fault_{fault:02d}').encode()).digest()[:16],'big');ix=np.random.Generator(np.random.PCG64(seed)).integers(0,250,(20000,250),dtype=np.int64)
    expected=v.mean(1)[ix].mean(1);np.testing.assert_array_equal(expected,draw[cid+'_loss_per_rep_and_mean'][:,5])
    q=np.quantile(expected,[1/120,119/120],method='linear');np.testing.assert_allclose(q,[float(row['bonferroni98_333_low']),float(row['bonferroni98_333_high'])],rtol=0,atol=1e-12)
   before={p:sha(p) for area in ('evaluated','summary') for p in (work/area).rglob('*') if p.is_file()}
   with patch.object(run,'runtime',return_value={'synthetic_software_fixture':True}),patch.object(run,'verify_raw',return_value=[{'synthetic':True}]*1000),patch.object(run,'read_condition',side_effect=AssertionError('unexpected raw scoring read')),patch.object(run,'sha',side_effect=fixture_sha),patch.object(an,'score_pair',side_effect=AssertionError('unexpected rescore')),patch.object(an,'aggregate',side_effect=AssertionError('unexpected reaggregation')),patch.object(sys,'argv',args),contextlib.redirect_stdout(record):self.assertEqual(run.main(),0)
   for p,d in before.items():self.assertEqual(sha(p),d)
   for p,d in original_hashes.items():self.assertEqual(sha(p),d)
   self.assertEqual(len(list(output.glob('*_RETURN.zip'))),2);self.assertFalse(any(n=='tep' or n.startswith('tep.') for n in sys.modules))
   for d in output.glob('*_RETURN'):
    verify_tree(d);self.assertFalse(any(p.name=='OUTPUT.npz' for p in d.rglob('*')))
   (BASE/'reports/SYNTHETIC_INTEGRATION_TRACE.txt').write_text('SYNTHETIC ONLY; runtime/raw verification and raw loading mocked. No TEP inference.\n'+record.getvalue())
   write_json(BASE/'reports/SYNTHETIC_INTEGRATION_RESULT.json',{'result':'PASS','scoring_paths':135,'per_fit_contrasts':15,'finite_five_fit_contrasts':3,'bootstrap_draws_per_primary':20000,'independent_shared_ID_bootstrap_array_equality':True,'unchanged_prior_files':len(original_hashes),'resume_preserved_result_files':len(before),'exports':2,'simulator_imported':False,'fitting_selection_calibration_called':False,'raw_inputs_synthetic':True,'raw_loading_and_runtime_gates_mocked':True,'not_actual_TEP_evaluation':True})
  finally:run.PAYLOAD=old;f.tearDown()
if __name__=='__main__':unittest.main(verbosity=2)
