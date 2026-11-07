#!/usr/bin/env python3
"""Replay original DV frozen scores on a separately exported raw archive.
No simulation, learning, candidate selection, calibration, or interval re-estimation.
Requires NumPy. This utility has synthetic tests; real raw files are not included
in the code snapshot and real raw replay was not done during publication packaging.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, os, platform, sys
from pathlib import Path
sys.dont_write_bytecode=True
# Match the original local runner's single-thread numerical setting before NumPy.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from publication_io import child, readj, sha, verify_manifest

def validate_output(p):
    with np.load(p,allow_pickle=False) as z:
        if set(z.files)!={'values','xmv12','native_time_seconds'}:raise ValueError('unexpected raw array keys')
        vals=z['values'];mv=z['xmv12'];t=z['native_time_seconds']
        if vals.shape!=(960,52) or mv.shape!=(960,) or t.shape!=(961,):raise ValueError('raw array shape')
        if vals.dtype!=np.float64 or mv.dtype!=np.float64 or t.dtype!=np.float64:raise ValueError('raw array dtype')
        if not all(np.isfinite(x).all() for x in (vals,mv,t)):raise ValueError('nonfinite raw data')
        if np.max(np.abs(t-np.arange(961)*180.))>1e-6 or not np.all(np.diff(t)>0):raise ValueError('native clock differs')
        return vals

def original_science(bundle):
    code=bundle/'source_original/code';record=readj(bundle/'provenance/SOURCE_MANIFEST.json')
    expected={x['path']:x for x in record['entries']}
    for fn in ('inherited.py','science.py','storage.py'):
        if sha(code/fn)!=expected['code/'+fn]['sha256']:raise ValueError('original scientific source hash differs')
    sys.path.insert(0,str(code))
    import science
    if Path(science.__file__).resolve()!=(code/'science.py').resolve():raise ValueError('unexpected science module')
    return science

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();bundle=a.bundle.resolve();raw=a.raw_root.resolve();out=a.out.resolve()
    if any(out==p or p in out.parents for p in (bundle,raw)):raise ValueError('choose NEW output outside all inputs')
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'STARTED','raw_to_score':True,'simulation':False,'refit':False,'recalibration':False,
        'python':platform.python_version(),'numpy':np.__version__,'source_environment':'Python3.12.3 / NumPy2.2.6',
        'scope':'Frozen raw test scoring only; not simulator validation or repeated whole-training study.'}
    try:
        m=verify_manifest(raw);verify_manifest(bundle/'frozen_models');verify_manifest(bundle/'results')
        context=readj(raw/'EXPORT_CONTEXT.json');index=readj(raw/'RAW_INDEX.json')
        seeds=readj(bundle/'provenance/SEED_MANIFEST.json')
        if len(index)!=1915 or context['run_count']!=1915:raise ValueError('all1915 original raw runs required')
        by={r['canonical_label']:r for r in index}
        if len(by)!=len(index) or set(by)!={r['canonical_label'] for r in seeds}:raise ValueError('raw role labels differ')
        for field,name in [('model_lock_sha256','MODEL_LOCK.json'),('seed_manifest_sha256','SEED_MANIFEST.json'),('evaluation_lock_sha256','EVALUATION_LOCK.json')]:
            if context[field]!=sha(bundle/'provenance'/name):raise ValueError('raw source lock differs')
        ledger=[json.loads(line) for line in (bundle/'provenance/EVENTS.jsonl').read_text().splitlines()]
        commits={e['data']['label']:e['data'] for e in ledger if e['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT')}
        for r in seeds:
            x=by[r['canonical_label']]
            if any(x[k]!=v for k,v in r.items()):raise ValueError('raw seed/role identity differs')
            p=child(raw,r['relative']+'/OUTPUT.npz')
            if sha(p)!=x['output_sha256']:raise ValueError('raw index hash differs')
            proof=child(raw,'provenance/runs/'+r['relative'])
            cm=commits[r['canonical_label']]
            if cm['relative']!=r['relative'] or sha(proof/'FILES.original.json')!=cm['manifest_sha256']:
                raise ValueError('original raw manifest does not match committed source')
            source_files={e['path']:e for e in readj(proof/'FILES.original.json')['entries']}
            complete=readj(proof/'COMPLETE.original.json')
            if sha(proof/'COMPLETE.original.json')!=source_files['COMPLETE.json']['sha256']:
                raise ValueError('completion record not bound by original manifest')
            if sha(p)!=complete['output_sha256'] or sha(p)!=source_files['OUTPUT.npz']['sha256']:
                raise ValueError('raw output differs from committed source')
            for k in ('canonical_label','stage','seed_u32'):
                if complete[k]!=r[k]:raise ValueError('completion identity differs')
        science=original_science(bundle)
        assets=bundle/'source_original/assets'
        original={x['path']:x for x in readj(bundle/'provenance/SOURCE_MANIFEST.json')['entries']}
        for name in ('PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz','RESIDUAL_BASIS_NEWSEED_PCA.npy'):
            if sha(assets/name)!=original['assets/'+name]['sha256']:raise ValueError('original PCA asset differs')
        pca=science.load_npz(assets/'PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz')
        basis=np.load(assets/'RESIDUAL_BASIS_NEWSEED_PCA.npy',allow_pickle=False)
        # Validate every archived output, including development files, without refitting.
        for i,r in enumerate(seeds):
            validate_output(child(raw,r['relative']+'/OUTPUT.npz'))
            if (i+1)%250==0:print('RAW_SHAPES_VALIDATED=%d/1915'%(i+1),flush=True)
        def project_role(role,fault):
            rows=sorted((r for r in seeds if r['role']==role and r['fault']==fault),key=lambda x:x['run'])
            if len(rows)!=250:raise ValueError('test role count differs')
            # Same250-run batch shape and operation order as the original evaluator.
            x=np.stack([validate_output(child(raw,r['relative']+'/OUTPUT.npz')) for r in rows])
            obj=science.project(x,pca,basis);obj['run_id']=np.array([r['run'] for r in rows]);return obj
        normal=project_role('normal_evaluation',0);bits=fields=0;rows=[]
        for fault in (3,9,15):
            faulty=project_role('fault_evaluation',fault)
            methods,events,mask=science.selected_outcomes(faulty,bundle/f'frozen_models/F{fault:02d}')
            nmethods,nev,nmask=science.selected_outcomes(normal,bundle/f'frozen_models/F{fault:02d}')
            with np.load(bundle/f'results/F{fault:02d}_RECORDS.npz',allow_pickle=False) as saved:
                if methods!=saved['methods'].tolist() or nmethods!=methods:raise ValueError('method ordering differs')
                for name,observed in [('fault_masks',mask),('normal_masks',nmask)]:
                    observed=np.packbits(observed.transpose(2,0,1),axis=2)
                    if not np.array_equal(observed,saved[name]):raise ValueError('raw-to-score alarm mismatch: F%02d %s'%(fault,name))
                    bits+=observed.size*8
                for name,value in events.items():
                    if not np.array_equal(value,saved[name]):raise ValueError('event mismatch: '+name)
                    fields+=value.size
            rows.append({'fault':fault,'methods':len(methods),'fault_runs':250,'normal_runs_shared':250,'alarm_bits':'EXACT_MATCH'})
            print('F%02d_RAW_TO_SCORE=PASS'%fault,flush=True)
        report.update(status='PASS_FROZEN_RAW_TO_SCORE',archive_entries=len(m['entries']),raw_runs_validated=1915,
            test_runs_scored=1000,alarm_bits_compared=bits,event_fields_compared=fields,faults=rows)
    except Exception as e:
        report.update(status='FAILED_DO_NOT_RETUNE',error=type(e).__name__+': '+str(e));raise
    finally:
        (out/'RAW_REPLAY_REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('RAW_REPLAY_STATUS='+report['status'],flush=True)

if __name__=='__main__':main()
