"""Fixed input gates and read-only shared-test loading. No simulator or fitting."""
from __future__ import annotations
import csv,hashlib,sys
from collections import Counter
from pathlib import Path
import numpy as np
from storage import strict_json,sha,regular,safe_child,verify_tree
from common_c import validate_complete,validate_test_row,cfreeze,CLedger,runtime
import science

PAYLOAD=Path(__file__).resolve().parents[1]
STUDY='TEP_V18_RELEARNING_R5_v1.0'
FAULTS=(3,9,15)
METHODS=('Q_CUSUM','mean_pool_absolute','mean_pool_signed','mean_pool_squared','moment_pool_absolute','moment_pool_signed','moment_pool_squared','same_mean_signed_axis_absolute','same_mean_signed_axis_squared')
def require(ok,message):
    if not ok:raise ValueError(message)

def check_expected(root,manifest):
    seen=set()
    for e in manifest['entries']:
        require(e['path'] not in seen,'Duplicate expected path');seen.add(e['path'])
        p=safe_child(root,e['path']);regular(p)
        require(p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'Changed prior file: '+str(p))

def gate(root,payload=PAYLOAD):
    b=strict_json(payload/'BINDING.json')
    require(root==Path(b['root']) and root.is_dir() and not root.is_symlink(),'Original study root differs or is missing')
    for rel,digest in b['stage_a_control_hashes'].items():
        p=safe_child(root,rel);regular(p);require(sha(p)==digest,'Stage A control changed: '+rel)
    require(sha(root/'release/RELEASE_MANIFEST.json')==b['stage_a_release_manifest_sha256'],'Original release changed')
    verify_tree(root/'release','RELEASE_MANIFEST.json')
    broot=safe_child(root,b['stage_b_subdirectory']);croot=safe_child(root,b['stage_c_subdirectory'])
    for sub in (root,broot,croot):
        require(not sub.is_symlink() and sub.is_dir(),'Missing prior stage directory')
        require(strict_json(sub/'ACTIVE_SESSION.json').get('active') is False,'Prior A/B/C stage still active')
        for n in ('BLOCKED.json','SESSION_ERROR.json'):require(not (sub/n).exists(),'Prior stage failure retained: '+str(sub/n))
        at=sub/'attempts';require(at.is_dir() and not at.is_symlink() and not list(at.iterdir()),'Unfinished prior-stage attempt: '+str(at))
    check_expected(broot,strict_json(payload/'reference/STAGE_B_EXPECTED_FILES.json'))
    check_expected(croot,strict_json(payload/'reference/STAGE_C_EXPECTED_FILES.json'))
    lock=strict_json(broot/'ALL_MODELS_LOCK.json')
    require(sha(broot/'ALL_MODELS_LOCK.json')==b['all_models_lock_sha256'] and lock['configuration_count']==135 and lock['replication_count']==5,'135-model lock mismatch')
    require(sha(croot/'PANEL_COMPLETION.json')==b['panel_completion_sha256'],'Completed test panel changed')
    panel=strict_json(croot/'PANEL_COMPLETION.json')
    require(panel['test_count']==1000 and panel['test_scored']==0,'Wrong pre-evaluation completion record')
    require(panel['all_models_lock_sha256']==b['all_models_lock_sha256'],'Panel has a different model lock')
    require(panel['test_completed_index_sha256']==b['test_completed_index_sha256']==sha(croot/'TEST_COMPLETED_INDEX.csv'),'Test index changed')
    require({p.name for p in (broot/'models').iterdir()}=={f'rep{i:02d}' for i in range(1,6)},'Five model directories required')
    for r in lock['replications']:
        p=safe_child(broot,r['path']);verify_tree(p)
        require(sha(p/'FILES.json')==r['manifest_sha256'] and sha(p/'REPLICATION_LOCK.json')==r['replication_lock_sha256'],'Per-repetition lock mismatch')
        for fault in FAULTS:
            q=p/f'F{fault:02d}';s=strict_json(q/'SELECTION.json')
            require(tuple(s['choices'])==METHODS,'Frozen method order/set differs')
            outer=science.load_npz(q/'OUTER.npz')
            require(int(outer['rank'])==247 and int(outer['n'])==259 and outer['thresholds'].shape==(49,) and np.isfinite(outer['thresholds']).all(),'Outer threshold state invalid')
    for name,digest in b['scientific_sources'].items():require(sha(payload/'code'/name)==digest,'Scientific source differs from Stage B: '+name)
    assets=payload/'assets';require(sha(assets/'PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz')==b['pca_sha256'] and sha(assets/'RESIDUAL_BASIS_NEWSEED_PCA.npy')==b['residual_basis_sha256'],'PCA/basis identity differs')
    seeds=strict_json(root/'SEED_MANIFEST.json');require(len(seeds)==5575 and len({r['canonical_label'] for r in seeds})==5575 and len({r['seed_u32'] for r in seeds})==5575,'Frozen allocation not unique')
    rows=sorted((r for r in seeds if r['stage']=='evaluation'),key=lambda r:(r['fault'],r['run']))
    require(len(rows)==1000,'Wrong test row count')
    for r in rows:validate_test_row(r)
    with (croot/'TEST_COMPLETED_INDEX.csv').open(newline='',encoding='utf8') as f:ix=list(csv.DictReader(f))
    require(len(ix)==1000 and len({r['canonical_label'] for r in ix})==1000,'Duplicate or missing completed index rows')
    by={r['canonical_label']:r for r in ix}
    commits={};starts=set();ledger=CLedger(croot/'EVENTS.jsonl',established=True)
    for e in ledger.records:
        if e['event']=='ATTEMPT_START':require(e['data']['label'] not in starts,'Duplicate prior test start');starts.add(e['data']['label'])
        elif e['event']=='ATTEMPT_COMMIT':
            d=e['data'];require(d['label'] in starts and d['label'] not in commits,'Test commit mismatch');commits[d['label']]=d
    require(set(by)==set(commits)==starts and len(starts)==1000,'Generation ledger not complete')
    for r in rows:
        x=by[r['canonical_label']]
        for k in ('canonical_label','replication','role','fault','run','seed_u32','stage','relative'):require(str(r[k])==x[k],'Test identity mismatch: '+k)
        require(x['files_manifest_sha256']==commits[r['canonical_label']]['manifest_sha256'] and x['all_models_lock_sha256']==b['all_models_lock_sha256'],'Raw manifest/model chain differs')
    require(Counter((r['fault'],r['role']) for r in rows)==Counter({(0,'normal_evaluation'):250,(3,'fault_evaluation'):250,(9,'fault_evaluation'):250,(15,'fault_evaluation'):250}),'Shared panel role count differs')
    return b,broot,croot,rows,by

def validate_row(root,croot,row,index,read_values=True):
    p=safe_child(root,row['relative']);require(p.is_dir() and not p.is_symlink(),'Missing original test: '+str(p));verify_tree(p)
    for name,key in [('FILES.json','files_manifest_sha256'),('OUTPUT.npz','output_sha256'),('METADATA.json','metadata_sha256'),('COMPLETE.json','complete_sha256')]:require(sha(p/name)==index[key],'Changed committed test '+name)
    m=validate_complete(p,row,cfreeze(croot),read_values=read_values)
    for k in ('started_utc','ended_utc','preonset_sha256','all_models_lock_sha256'):require(m[k]==index[k],'Test metadata differs from final index')
    require(str(m['shutdown_occurred'])==index['shutdown_occurred'],'Shutdown annotation changed')
    return m

def verify_raw(root,croot,rows,index):
    evidence=[];leaves=set()
    for i,row in enumerate(rows,1):
        m=validate_row(root,croot,row,index[row['canonical_label']],True);p=safe_child(root,row['relative']);leaves.add(p)
        evidence.append({'canonical_label':row['canonical_label'],'fault':row['fault'],'run':row['run'],'output_sha256':index[row['canonical_label']]['output_sha256'],'shutdown_occurred':m['shutdown_occurred']})
        if i%100==0:print(f'VERIFY TEST INPUTS {i}/1000 | no generation',flush=True)
    for p in (root/'data/shared_test').rglob('*'):
        require(not p.is_symlink(),'Shared data symlink')
        if p.is_file():require(p.parent in leaves,'Unregistered shared data file')
    return evidence

def read_condition(root,croot,rows,index,fault,pca,basis):
    chosen=sorted((r for r in rows if r['fault']==fault),key=lambda r:r['run'])
    require([r['run'] for r in chosen]==list(range(1,251)),'Shared IDs must be 1..250')
    a=np.empty((250,960,52),dtype=np.float64)
    for i,row in enumerate(chosen):
        validate_row(root,croot,row,index[row['canonical_label']],False)
        with np.load(root/row['relative']/'OUTPUT.npz',allow_pickle=False) as z:a[i]=z['values']
    obj=science.project(a,pca,basis)
    require(np.array_equal(obj['run_id'],np.arange(1,251)),'Projection run identities differ')
    return obj
