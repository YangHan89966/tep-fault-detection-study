"""Historical-only replay: paired 78-normal PCA, chosen axes and all D439 alarms.
Command: python verify_history.py --archives source_archives --out QA_HISTORICAL.json
No scientific-output directory/trajectory is accessed.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile
import numpy as np
import pipeline as L
def npz(z,n):
 with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as a:return {k:a[k] for k in a.files}
def j(z,n):return json.loads(z.read(n).decode('utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 d=args.archives;src=next(d.glob('tep-p1-normal-training78*zip'));legacy=next(d.glob('F9_RELEARN_RANK*.zip'));pcasrc=next(d.glob('TEP_F9_Standard*zip'))
 bind={s.name:L.io.sha(s) for s in [src,legacy,pcasrc]};L._verify_code()
 with zipfile.ZipFile(src) as z,zipfile.ZipFile(pcasrc) as zs:
  sources=j(zs,pcasrc.stem+'/source_notes/TRAINING_SOURCE.json')['cases'];old={c['case_id']:c['raw_sha256'] for c in sources}
  manifest=list(csv.DictReader(io.StringIO(z.read(src.stem+'/TRAINING_MANIFEST.csv').decode('utf-8-sig'))))
  assert len(old)==len(manifest)==78
  views={k:[] for k in L.VIEWS}
  for r in sorted(manifest,key=lambda r:int(r['index'])):
   n=src.stem+'/'+r['raw_path'];assert sha(z.read(n))==old[r['case_id']]==r['raw_sha256']
   a=npz(z,n);pair=L.validated_pair(a['native'],a['snapshot'])
   for k in L.VIEWS:views[k].append(pair[k])
  original=L.io.load_npz(L.SELF/'assets/PCA_BY_VIEW.npz')
  fit=L.fit_pca_views({k:np.stack(rows) for k,rows in views.items()})
  pr={}
  for k in L.VIEWS:
   v=L.core.pca_view(fit,k);o=L.core.pca_view(original,k)
   errs={'center':float(np.max(abs(v['mean']-o['mean']))),'scale':float(np.max(abs(v['scale']-o['scale']))),'eigenvalues':float(np.max(abs(v['eigenvalues']-o['eigenvalues']))),'retained_projector':float(np.max(abs(v['loadings']@v['loadings'].T-o['loadings']@o['loadings'].T))),'residual_projector':float(np.max(abs(v['basis']@v['basis'].T-o['basis']@o['basis'].T)))}
   np.testing.assert_allclose(list(errs.values()),0,atol=1e-8,rtol=0)
   np.testing.assert_allclose(L.basis_from_loadings(o['loadings']),o['basis'],atol=2e-13,rtol=0)
   assert int(v['components'])==int(o['components'])
   pr[k]={'retained':int(v['components']),'errors':errs}
 with zipfile.ZipFile(legacy) as z:
  nl=z.namelist(); mlock=j(z,'workspace/MODEL_LOCK.json');folder='workspace/'+mlock['relative_path']+'/'
  spos={};ma={};anchors=[]
  for v in L.VIEWS:
   for n in ['MODEL.npz','DESIGN.npz','FIT_MOMENTS.npz','INNER.npz','SELECTION.json']:
    ub=v+'/'+n;assert sha(z.read(folder+ub))==mlock['files'][ub]
   geometry=npz(z,folder+v+'/DESIGN.npz');mom=npz(z,folder+v+'/FIT_MOMENTS.npz');model=npz(z,folder+v+'/MODEL.npz')
   a,_,small,*_=L.hist.build_axes_regularized(mom['mean_early'],mom['mean_late'],geometry['B'],geometry,L.core.VARIANT)
   a=a[small].copy();a/=np.sqrt(np.einsum('ad,de,ae->a',a,geometry['G'],a))[:,None]
   b=L.hist.learned_axes(mom['contrast'],geometry,'ridge_1e4')[0]
   # Eigenvectors of a real symmetric/Hermitian generalized-coordinate problem may flip signs.
   for name,x in [('mean',a),('moment',b)]:np.testing.assert_allclose(x,model[name],rtol=1e-8,atol=1e-8)
   inn=npz(z,folder+v+'/INNER.npz'); km=len(a); mask=np.unpackbits(inn['selection_alarm_packed'],axis=1,count=960,bitorder='little').astype(bool)
   tot=(L.core.events(mask)['loss_minutes']//3).sum(0)
   np.testing.assert_array_equal(tot,inn['loss_samples_total'])
   ix={'mean':int(np.argmin(tot[:km])),'moment':int(km+np.argmin(tot[km:]))}
   assert ix==j(z,folder+v+'/SELECTION.json')['choices'];spos[v]=ix;ma[v]=inn['anchors']
  for _,v,pool in L.core.CONDITIONS:anchors.append(ma[v][spos[v][pool]])
  out=next(n for n in nl if n.endswith('/D439_supplement.npz')); cal=npz(z,out);h=L.core.thresholds(L.core.chart(cal['scores']).max(1),np.asarray(anchors),190)
  np.testing.assert_array_equal(h,cal['thresholds'])
  ep=npz(z,next(n for n in nl if n.endswith('/D439_supplement_EVALUATION.npz')));cv=L.core.chart(ep['scores'])
  mask=cv>h;np.testing.assert_array_equal(mask,np.unpackbits(ep['alarm_packed'],axis=1,count=960,bitorder='little').astype(bool))
  cs=j(z,next(n for n in nl if n.endswith('/D439_supplement_CASES.json')));normal=[n for n,row in enumerate(cs) if row['role']=='normal_evaluation'];fault=[n for n,row in enumerate(cs) if row['role']=='fault9_evaluation'];assert len(normal)==len(fault)==120
  fe=L.core.events(mask[fault]);ne=L.core.events(mask[normal]); orig= dict(clean40=fe['clean'].sum(0).tolist(),clean8=fe['clean_480'].sum(0).tolist(),normal_any=ne['normal_alarm'].sum(0).tolist(),thresholds=h.tolist())
  assert orig['clean40']==[119,38,112,16] and orig['normal_any']==[7,7,10,11]
 result=dict(status='PASS',scope='Historical evidence only, not the new scientific panel. 78 original paired normal raw arrays re-fit; the historical fitted-state, internal selection, final calibration and evaluation checked from archived moments/scores/masks, not re-simulated raw development or evaluation data.',archives_sha256=bind,pca_new_algorithm_vs_historical=pr,chosen_indices=spos,calibration_rows=199,evaluation_bits_matched=int(mask.size),original_M1_M2_M3_M4=orig)
 args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
