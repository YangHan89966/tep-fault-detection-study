#!/usr/bin/env python3
"""Export fixed scalar score paths only. No model fitting or calibration.
Scientific scope: historical development data and fixed diagnostic score paths.
Only the exact historical matrix and normal cache identities are accepted.
"""
from pathlib import Path
import sys,os,json,zipfile,hashlib,traceback,time,platform,shutil
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from core import recurrence,event_rows,normal_rows,numeric_records,check,sha,js
from data_io import OldMatrix,normal_cache,project,DEFAULT_MATRIX,NORMAL_CANDIDATES,stable_signature


def load(p):
 with np.load(p,allow_pickle=False) as f:return dict(f)

def export_projected(root,out,normal,fd):
 """Same role shapes/projection operation as original S28. All axes for dot parity;
 only selected underlying u paths are saved. Validation never changes thresholds.
 """
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 scope=json.loads((root/'EXPORT_SCOPE.json').read_text());fault=scope['fault'];rows=[]
 for n,tag in enumerate(scope['splits'],1):
  p=root/'inputs'/tag;m=load(p/'MODEL_CORE.npz');des=load(p/'NORMAL_STATE.npz');t=load(p/'TRANSFORM_STATE.npz');o=load(p/'OUTER_RECORDS.npz');ev=load(p/'EVALUATION_RECORDS.npz');ch=json.loads((p/'CHOICES.json').read_text());roles=json.loads((p/'ROLES.json').read_text())
  km=ch['mean_count'];kn=ch['moment_count'];k=km+kn
  axis_roles={'old_signed_axis':int(ch['choices']['mean_pool_signed']),'new_signed_axis':int(ch['choices']['moment_pool_signed']),'new_squared_axis':int(ch['choices']['moment_pool_squared'])%k}
  indices=list(dict.fromkeys(axis_roles.values()));assert len(indices)<=3
  dest=out/tag;dest.mkdir(exist_ok=False)
  candidate_indices=np.array([ind+g*k for g in range(3) for ind in indices])
  state={**axis_roles,'axis_indices':indices,'candidate_names':[ch['candidate_names'][int(i)] for i in candidate_indices],'families':['signed','absolute','squared'],'candidate_indices_family_major':candidate_indices.tolist(),'strict_threshold_comparison':'>','k':0.05,'fault_onset_zero_index':160,'operation':'original_binary64_cp+u-k'}
  packed={'axis_indices':np.array(indices),'thresholds':o['thresholds'][candidate_indices].reshape(3,len(indices)),
          'abs_mean':np.array([t['mean_abs_mean'][i] if i<km else t['moment_abs_mean'][i-km] for i in indices]),
          'abs_sd':np.array([t['mean_abs_sd'][i] if i<km else t['moment_abs_sd'][i-km] for i in indices]),
          'sq_mean':np.array([t['mean_sq_mean'][i] if i<km else t['moment_sq_mean'][i-km] for i in indices]),
          'sq_sd':np.array([t['mean_sq_sd'][i] if i<km else t['moment_sq_sd'][i-km] for i in indices])}
  for source,role in [('normal','outer_calibration'),('normal','audit'),('fault','audit')]:
   ids=np.array(roles[source][role],dtype=np.int16);obj=normal if source=='normal' else fd
   y=(obj['z'][ids-1]-des['centre'])@des['whitener']
   uo=np.einsum('ntd,ad->nta',y,m['mean_axes'],optimize=True)
   un=np.einsum('ntd,ad->nta',y,m['moment_axes'],optimize=True)
   u=np.concatenate([uo,un],axis=2)
   value=u[:,:,indices].copy();key=source+'_'+role
   # Save before validating so any rare rounding mismatch has a complete witness.
   np.savez_compressed(dest/(key+'.npz'),run_id=ids,u=value,axis_indices=np.array(indices))
   fpar=[]
   for g,fam in enumerate(['signed','absolute','squared']):
    def transform(v,prefix):
     if fam=='signed':return v
     if fam=='absolute':return (abs(v)-t[prefix+'_abs_mean'])/t[prefix+'_abs_sd']
     return (v*v-t[prefix+'_sq_mean'])/t[prefix+'_sq_sd']
    full=np.concatenate([recurrence(transform(uo,'mean')),recurrence(transform(un,'moment'))],axis=2)
    # Scalar path reconstruction must restore original per-axis statistics exactly.
    v=value if fam=='signed' else ((abs(value)-packed['abs_mean'])/packed['abs_sd'] if fam=='absolute' else (value*value-packed['sq_mean'])/packed['sq_sd'])
    recovered=recurrence(v)
    check(np.array_equal(recovered,full[:,:,indices]),'Saved score scalar roundtrip differs: '+tag+key+fam)
    for l,ind in enumerate(indices):
     j=ind+g*k;d=full[:,:,ind];h=float(o['thresholds'][j]);mask=d>h
     if source=='normal' and role=='outer_calibration':
      expected=o['peaks'][:,j];peaks=d.max(1);delta=float(np.max(abs(peaks-expected)))
      check(np.array_equal(ids,o['run_id']),'Outer run order mismatch')
      check(np.allclose(peaks,expected,rtol=1e-9,atol=1e-8),'Stored outer score peaks mismatch')
      fpar.append({'family':fam,'axis_index':ind,'outer_peak_error':delta,'n':len(ids)})
     elif source=='normal':
      fields,arr=numeric_records(normal_rows(mask,ids,fault,'frozen'))
      check(list(fields)==list(ev['normal_fields']),'Normal fields changed')
      check(np.array_equal(arr,ev['normal'][j]),'Normal outcome discrepancy '+tag+fam+str(ind))
      fpar.append({'family':fam,'axis_index':ind,'normal_integer_fields':int(arr.size)})
     else:
      fields,arr=numeric_records(event_rows(mask,ids,fault,'audit','frozen'))
      check(list(fields)==list(ev['fault_fields']),'Fault fields changed')
      check(np.array_equal(arr,ev['fault'][j]),'Fault outcome discrepancy '+tag+fam+str(ind))
      fpar.append({'family':fam,'axis_index':ind,'fault_integer_fields':int(arr.size)})
   rows.append({'split':tag,'source':source,'role':role,'path':(Path(out.name)/tag/(key+'.npz')).as_posix(),'n_runs':len(ids),'axes':len(indices),'checks':fpar})
  np.savez_compressed(dest/'TRANSFORM_AND_THRESHOLDS.npz',**packed);js(dest/'AXIS_ROLES.json',state)
  print(f'Fixed-score export: {n}/{len(scope["splits"])} {tag}',flush=True)
 js(out/'PARITY_CHECKS.json',rows)
 return {'status':'PASS_FIXED_SCORE_PATHS_ONLY_NOT_NEW_METHOD_RESULTS','split_count':len(scope['splits']),'roles_per_split':3,'fault':fault,'normalization_reestimated':False,'calibration_reestimated':False,'candidate_selection_changed':False,'evidence_transform_parameters_frozen':True,'scalar_roundtrip_exact':True}

def package(root,parent,success):
 root=Path(root);parent=Path(parent);lines=[]
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name not in ['RESULTS_SHA256.txt','CONSOLE.log']:
   lines.append(sha(p)+'  '+p.relative_to(root).as_posix())
 (root/'RESULTS_SHA256.txt').write_text('\n'.join(lines)+'\n')
 dest=parent/(root.name+('_RESULTS.zip' if success else '_FAILURE.zip'))
 with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file() and p.name!='CONSOLE.log':z.write(p,p.relative_to(root).as_posix())
 print('EXPORT_STATUS='+('FIXED_F9_SCORE_PATHS_EXPORTED' if success else 'FAILED_NO_NEW_F9_RESULT'),flush=True)
 print('ARCHIVE_PATH='+str(dest),flush=True)

def main(root,matrix_path,normal_path):
 root=Path(root);scope=json.loads((root/'EXPORT_SCOPE.json').read_text());check(scope['fault']==9,'Production export is limited to F9')
 check(shutil.disk_usage(root).free>=1024**3,'Need1GiB temporary free space; nothing is deleted')
 pca=load(root/'inputs/PCA_STATE.npz');basis=load(root/'inputs/RESIDUAL_BASIS.npz')['basis']
 candidates=[Path(normal_path)] if normal_path else NORMAL_CANDIDATES
 nf=next((x for x in candidates if x.is_file()),None);check(nf is not None,'Exact OLD normal validation source not found; no substitutes')
 x,ni=normal_cache(nf);normal,ne=project(x,pca,basis);del x
 matrix=OldMatrix(matrix_path);x=matrix.read_fault(9);fd,fe=project(x,pca,basis);del x
 js(root/'INPUT_ACCESS.json',{'normal':ni,'normal_projection_error':ne,'fault_projection_error':fe,'old_matrix':matrix.access,'old_final_process_values_decoded':0,'newseed_access':False})
 result=export_projected(root,root/'scores',normal,fd)
 check(stable_signature(matrix_path)==matrix.signature,'Source changed')
 js(root/'COMPLETE.json',result)
 return result
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output-parent',type=Path,required=True);a.add_argument('--matrix',type=Path,default=DEFAULT_MATRIX);a.add_argument('--normal',type=Path);args=a.parse_args();success=False
 try:
  js(args.root/'RUNTIME.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'threads':1})
  main(args.root,args.matrix,args.normal);success=True
 except BaseException as e:
  js(args.root/'FAILURE.json',{'error':str(e),'type':type(e).__name__,'traceback':traceback.format_exc(),'no_new_fitting':True,'do_not_change_parameters_to_bypass':True});print(traceback.format_exc(),flush=True)
 finally:package(args.root,args.output_parent,success)
 sys.exit(0 if success else 2)
