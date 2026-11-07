"""Synthetic tests only; do not read any real process source."""
from pathlib import Path
import sys,tempfile,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from data_io import OldMatrix
from core import recurrence,prefix_cusum,event_rows

def run(out):
 tests=[]
 def ok(name,value):
  assert value,name;tests.append({'test':name,'passed':True})
 rng=np.random.default_rng(301901) # Public synthetic-interface test RNG, not TEP simulation.
 u=rng.normal(size=(7,960,3));d=recurrence(u)
 ok('scalar_roundtrip',np.array_equal(d,recurrence(u.copy())))
 ok('prefix_close_no_threshold_adjustment',np.allclose(d,prefix_cusum(u),rtol=1e-9,atol=1e-8))
 ok('whole_axis_sign_invariance',np.array_equal(d,recurrence(-u)))
 a=u[:,:,:1];par=(abs(a)-abs(a).mean())/abs(a).std(ddof=1)
 ok('saved_transform_roundtrip',np.array_equal(recurrence(par),recurrence((abs(a.copy())-abs(a).mean())/abs(a).std(ddof=1))))
 mask=np.zeros((1,960),bool);mask[0,0]=True;mask[0,160]=True;r=event_rows(mask,[1],9,'test','fixed')[0]
 ok('prefault_not_success',r['clean_detection']==0 and r['event_loss_samples']==800)
 r=event_rows(np.zeros((1,960),bool),[1],9,'test','fixed')[0]
 ok('miss_penalty',r['event_loss_samples']==800)
 with tempfile.TemporaryDirectory(prefix='s30_test_') as td:
  rows=[]
  for rid in [1,2,3]:
   for t in range(1,5):rows.append([1.,rid,t]+([float(rid+t)]*52 if rid<3 else [float('nan')]*52))
  data=np.array(rows)
  for endian in ['<f8','>f8']:
   p=Path(td)/('little.npy' if endian=='<f8' else 'big.npy');np.save(p,data.astype(endian))
   m=OldMatrix(p,production=False,expected_faults=1,allowed_runs=2,samples=4)
   x=m.read_fault(1);ok('allowed_rows_only_'+endian,np.array_equal(x[:,:,0],[[2,3,4,5],[3,4,5,6]]) and m.access['old_final_process_rows_decoded']==0)
  for kind in ['duplicate','missing','nonfinite','f32']:
   v=data.copy()
   if kind=='duplicate':v[1,:3]=v[0,:3]
   if kind=='missing':v=v[1:]
   if kind=='nonfinite':v[0,4]=np.nan
   if kind=='f32':v=v.astype('f4')
   p=Path(td)/(kind+'.npy');np.save(p,v);rejected=False
   try:
    m=OldMatrix(p,production=False,expected_faults=1,allowed_runs=2,samples=4);m.read_fault(1)
   except (ValueError,AssertionError):rejected=True
   ok('reject_'+kind,rejected)
  p=Path(td)/'little.npy';rejected=False
  try:OldMatrix(p,production=True)
  except (ValueError,AssertionError):rejected=True
  ok('production_identity_rejects_synthetic',rejected)
 report={'status':'PASS_SYNTHETIC_ONLY_NOT_F9_EXPORT','count':len(tests),'tests':tests}
 Path(out).write_text(json.dumps(report,indent=2));print(report['status'],len(tests))
if __name__=='__main__':run(sys.argv[1])
