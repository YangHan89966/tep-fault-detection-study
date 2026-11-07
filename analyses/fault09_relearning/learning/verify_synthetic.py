"""Development I/O + full new fitting smoke test on synthetic arrays ONLY.
Runs in a temporary directory. Uses no TEP simulator, new Philox keys or
scientific trajectories. Inputs deliberately obey two exact CI linear relations.
"""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import pipeline as L

def main():
 arg=argparse.ArgumentParser();arg.add_argument('--out',type=Path,required=True);args=arg.parse_args()
 rng=np.random.default_rng(7615307)  # software fixture, unrelated to scientific keyspace
 mat=rng.normal(size=(24,52))/np.sqrt(24)
 def artificial(fault):
  latent=rng.normal(size=(960,24))
  if fault:latent[160:,0]+=.28; latent[160:,1]*=1.12
  p=latent@mat+.22*rng.normal(size=(960,52)); c=p.copy()
  c[:,:22]+=.07*rng.normal(size=(960,22)); c[:,11]=-p[:,47]; c[:,14]=p[:,48]*.7
  return p,c
 with tempfile.TemporaryDirectory(prefix='f9-software-no-study-') as td:
  t=Path(td);md=t/'model';data=t/'raw';rs=[]
  time=np.arange(961)*180.;phase=np.zeros((960,4),int);phase[:,0]=time[1:];phase[:,3]=3
  for role,n in L.COUNTS.items():
   fault=L.FAULTS[role]
   flags=np.zeros((960,20),np.int8)
   if fault:flags[160:,fault-1]=1
   for i in range(1,n+1):
    p,c=artificial(fault);k=len(rs)+1;cid=f'TEST_ONLY_{role}_{i:03}'
    row=dict(case_id=cid,role=role,fault_id=fault,rep=1,index=i,key_hex=f'{k:032x}')
    d=data/cid;d.mkdir(parents=True)
    np.savez(d/'RAW.npz',native_values=p,fast_control_input_measurements=c[:,:41],native_time_seconds=time,diagnostic_IDV=flags,phase=phase)
    rs.append(row)
  summary=L.fit_replicate(rs,data,md)
  q=[artificial(i) for i in (0,9)]
  scores=np.stack([L.score_pair(*r,md) for r in q])
  expected=L.score_batch(np.stack([e[0] for e in q]),np.stack([e[1] for e in q]),md)
  np.testing.assert_allclose(scores,expected,rtol=1e-12,atol=1e-12)
  pca=L.io.load_npz(md/'PCA_BY_VIEW.npz')
  obs={v:L.core.project_raw(np.stack([pair[i] for pair in q]),L.core.pca_view(pca,v)) for i,v in enumerate(L.VIEWS)}
  historical=L.core.chart(L.core.four_inputs(obs,md))
  np.testing.assert_allclose(scores,historical,rtol=1e-12,atol=1e-12)
  aa=L.anchors(md); mx=rng.uniform(0.2,26,size=(199,4))
  np.testing.assert_array_equal(L.anchored_limits(mx,md),L.core.thresholds(mx,aa,190))
  np.testing.assert_array_equal(aa,L.core.four_anchors(md))
  check={'status':'PASS','scope':'Entirely synthetic data to test file reading, independent per-view PCA fitting, fixed support, axis selection, final-limit formula and complete chart. Not a TEP study or a scientific result.','source_shape':'384 simulated Gaussian fixture arrays each 960x52, two post-fit scoring fixtures','pca_components':summary['pca_components'],'effective_rank':summary['support_rank'],'n_input_receipts':len(rs),'compared_chart_values':scores.size,'max_abs_vs_original_four_condition_scoring':float(abs(scores-historical).max()),'max_abs_vs_score_batch':float(abs(scores-expected).max()),'original_anchored_190of199_limit_reproduced':True}
 args.out.write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check))
if __name__=='__main__':main()
