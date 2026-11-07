#!/usr/bin/env python3
"""S28 second-moment fitting on OLD development; no simulator or new seeds."""
from pathlib import Path
import os,sys,json,time,shutil,platform
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from core import check,sha,js,cs,stamp
from data_io import OldMatrix,normal_cache,project,verify_markers,stable_signature
from old_evidence import load_npz
from second_moment import fit as fit_evidence,calibrate as outer_calibrate,evaluate
ROOT=Path(__file__).resolve().parents[1]

def verify_payload(root):
    root=Path(root)
    manifest=root/'PAYLOAD_SHA256.txt'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1);check(sha(root/name)==digest,'Payload mismatch '+name)
    d=json.loads((root/'STUDY_DEFINITION.json').read_text());lock=json.loads((root/'DEFINITION_LOCK.json').read_text())
    check(sha(root/'STUDY_DEFINITION.json')==lock.get('public_copy_sha256',lock['sha256']),'Definition changed')
    return d

def execute(root,run,normal,fault_provider,faults,access,mode):
    root=Path(root);run=Path(run);run.mkdir(parents=True,exist_ok=False);d=verify_payload(root);t=time.time()
    check(set(faults).issubset(set(d['faults'])),'Fault list outside definition')
    work=run/'work';work.mkdir(exist_ok=False)
    total=len(faults)*len(d['splits'])*len(d['variants']);branches=[];count=0
    js(run/'RUNTIME.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'threads':1,'platform':platform.platform(),'no_simulator':True,'new_second_moment_axes':True,'mean_control_axes_fixed':True,'no_newseed_access':True})
    for fault in faults:
        fd=fault_provider(fault);verify_markers(fault,fd,root/'inputs/EXPECTED_RECORD_CONTINUITY.csv')
        np.savez(work/f'fault{fault}.npz',**fd)
        for tag,split in d['splits'].items():
            for variant in d['variants']:
                dest=run/'models'/f'F{fault:02}'/tag/variant
                choice=fit_evidence(root,dest,normal,fd,fault,tag,variant,split)
                branches.append({'fault':fault,'split':tag,'variant':variant,'path':dest.relative_to(run).as_posix(),'choice_sha256':sha(dest/'CHOICES.json')});count+=1
        del fd
    js(run/'ALL_CHOICES_LOCKED.json',{'utc':stamp(),'count':total,'branches':branches,'new_audit_used':False,'definition_sha256':sha(root/'STUDY_DEFINITION.json')})
    for count,b in enumerate(branches,1):
        outer_calibrate(root,run/b['path'],normal,b['fault'],b['split'],b['variant'],d['splits'][b['split']])
    js(run/'ALL_OUTER_LOCKED.json',{'utc':stamp(),'count':total,'threshold_files':{b['path']:sha(run/b['path']/'OUTER_RECORDS.npz') for b in branches},'new_audit_used':False})
    main=[];audits=[];previous=None
    for count,b in enumerate(branches,1):
        if b['fault']!=previous:
            fd=load_npz(work/f"fault{b['fault']}.npz");previous=b['fault']
        rows=evaluate(root,run/b['path'],normal,fd,b['fault'],b['split'],b['variant'],d['splits'][b['split']]);a=json.loads((run/b['path']/'AUDIT.json').read_text())
        main.extend([{'split_id':b['split'],'variant':b['variant'],**r} for r in rows]);audits.append(a)
    cs(run/'MAIN_RESULTS.csv',main)
    from summarize import summarize_run
    summarize_run(run)
    js(run/'INPUT_ACCESS.json',access)
    summary={'status':'PASS_AUTHOR_NUMERICAL_CHECKS_NOT_INDEPENDENT_REVIEW','branches':total,
        'source_selection_integer_fields_verified':sum(json.loads((run/b['path']/'CHOICES.json').read_text())['source_selection_integer_fields_verified'] for b in branches),
        'source_audit_integer_fields_verified':sum(a['source_mean_integer_fields_verified'] for a in audits),
        'prefix_decisions':sum(a['prefix_decisions'] for a in audits),
        'exactly_certified_disagreements':sum(a['exactly_certified_disagreements'] for a in audits),
        'all_original_mean_axes_unchanged':True,'all20_complete':set(faults)==set(d['faults']), 'old_final_process_rows_decoded':0,'new_seed_access':False}
    js(run/'AUDIT.json',summary)
    js(run/'COMPLETE.json',{'utc':stamp(),'mode':mode,'seconds':time.time()-t,'faults':faults,'branches':total,'main_rows':len(main),'all20_complete':set(faults)==set(d['faults']),'status':'COMPLETE_OLD20_RETROSPECTIVE' if len(faults)==20 else 'COMPLETE_LOCAL_F3_F18_ONLY_NOT_ALL20'})
    return summary

def main(run,matrix_path,normal_path):
    d=verify_payload(ROOT);run=Path(run)
    check(shutil.disk_usage(run.parent).free>=3*1024**3,'Need3GiB free temporary space; nothing is deleted automatically')
    pca=load_npz(ROOT/'inputs/PCA_STATE.npz');basis=load_npz(ROOT/'inputs/RESIDUAL_BASIS.npz')['basis']
    normal_file=Path(normal_path)
    check(normal_file.is_file(),'Required historical normal input not found')
    nx,ninfo=normal_cache(normal_file);normal,err=project(nx,pca,basis);del nx
    matrix=OldMatrix(matrix_path);access={'normal':ninfo,'normal_projection_error':err,'matrix':matrix.access,'old_final_process_rows_decoded':0,'new_seed_access':False}
    def provider(fault):
        x=matrix.read_fault(fault);obj,e=project(x,pca,basis);del x
        return obj
    faults=[3,18]+[f for f in d['faults'] if f not in (3,18)]
    summary=execute(ROOT,run,normal,provider,faults,access,'HISTORICAL_20_FAULT_DIRECTION_COMPARISON')
    from secondary_audit import main as secondary_check
    secondary_check(run)
    check(stable_signature(matrix_path)==matrix.signature,'Source changed before completion');verify_payload(ROOT)
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True,help='New output directory')
    p.add_argument('--matrix',type=Path,required=True,help='Exact historical process matrix')
    p.add_argument('--normal',type=Path,required=True,help='Exact historical normal cache')
    a=p.parse_args()
    a.run_root.parent.mkdir(parents=True,exist_ok=True)
    main(a.run_root,a.matrix,a.normal)
