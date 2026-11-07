"""Fault 9 paired recording-specific learning. Scientific engine: archived v1.1.

Replicate input: 78 *new* PCA-normal pairs + 306 separate development pairs.
This module never selects limits or methods on scientific evaluation outcomes.
Outer workflow owns generation, global data identities, five-model locking,
calibration and the evaluation gate. See README.md for public interfaces.
"""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
SELF=Path(__file__).resolve().parent
VENDOR=SELF/'vendor'
for _mod in ('standard_core','common','normal_support','inherited_dim','relearning_core','existing_data'):
    if _mod in sys.modules and Path(sys.modules[_mod].__file__).resolve().parent!=VENDOR:
        raise RuntimeError('Scientific dependency name collision: '+_mod)
sys.path.insert(0,str(VENDOR))
import common as io
import relearning_core as core
import normal_support as ns
from existing_data import check_views
from standard_core import fit_pca
import inherited_dim as hist
VIEWS=('native','snapshot')
MONITORS=('M1_ci_moment_signed','M2_post_moment_signed','M3_ci_mean_signed','M4_post_mean_signed')
COUNTS={'pca_normal':78,'normal_design':40,'normal_anchor':32,'normal_inner':59,'fault_fit':100,'fault_select':75}
FAULTS={'pca_normal':0,'normal_design':0,'normal_anchor':0,'normal_inner':0,'fault_fit':9,'fault_select':9}

def check(cond,why):
    if not cond:raise ValueError(why)

def validated_pair(post,ci):
    a=np.asarray(post);b=np.asarray(ci)
    check(a.dtype==np.float64 and b.dtype==np.float64 and a.shape==b.shape==(960,52),'Both views must be float64 (960,52) with physical-unit columns')
    check(np.all(np.isfinite(a)) and np.all(np.isfinite(b)),'Nonfinite recording')
    check(np.array_equal(a[:,41:],b[:,41:]),'XMV mismatch between views')
    return dict(native=a,snapshot=b)

def raw_pair(raw,fault):
    # Source checks all raw phase/time/IDV rows and reconstructs CI solely by
    # replacing the first 41 columns; the new recorder also emits this CI view.
    pair=check_views(raw,int(fault))
    if 'controller_input_values' in raw:
        check(np.array_equal(raw['controller_input_values'],pair['snapshot']),'Explicit CI record disagrees with paired construction')
    return validated_pair(pair['native'],pair['snapshot'])

def basis_from_loadings(P):
    q=np.linalg.qr(P,mode='complete')[0][:,P.shape[1]:].copy()
    for j in range(q.shape[1]):
        if q[np.argmax(np.abs(q[:,j])),j]<0:q[:,j]*=-1
    check(q.shape[1]>=2 and np.max(np.abs(q.T@P))<1e-11,'Residual QR complement invalid')
    return q

def fit_pca_views(views):
    out={}
    for k in VIEWS:
        x=np.asarray(views[k]); check(x.shape==(78,960,52),'PCA requires exactly78 full normal runs')
        # Preserve the full52 feature extraction and array orientation of save_models.
        p=fit_pca(x[:,:,np.arange(52)])
        p['basis']=basis_from_loadings(p['loadings'])
        out.update({k+'__'+n:v for n,v in p.items()})
    return out

def _verify_code():
    binding=io.read_json(SELF/'assets/SOURCE_BINDING.json')
    io.verify_files(SELF,{k:v for k,v in binding['files'].items() if k.startswith('vendor/')})
    return dict(wrapper_sha256=io.sha(Path(__file__)),source_binding_sha256=io.sha(SELF/'assets/SOURCE_BINDING.json'))

def _load_role(group,raw_root,role,receipts):
    physical={k:[] for k in VIEWS}
    for x in group:
        ident=str(x['case_id']);check(Path(ident).name==ident and ident not in ('.','..'),'Unsafe case_id')
        path=Path(raw_root)/ident/'RAW.npz'; check(path.is_file(),'Missing raw '+ident)
        digest=io.sha(path)
        if x.get('raw_sha256'):check(digest==x['raw_sha256'],'Incorrect input identity')
        with np.load(path,allow_pickle=False) as d:
            pp=raw_pair(d,FAULTS[role])
        for k in VIEWS:physical[k].append(pp[k])
        receipts.append(dict(case_id=ident,key_hex=str(x['key_hex']),index=int(x['index']),rep=x['rep'],role=role,fault_id=FAULTS[role],raw_sha256=digest))
    return {k:np.stack(vs) for k,vs in physical.items()}

def _project(group,raw_root,role,receipts,pca):
    vals=_load_role(group,raw_root,role,receipts)
    return {v:core.project_raw(vals[v],core.pca_view(pca,v)) for v in VIEWS}

def fit_replicate(rows,raw_root,model_dir):
    """Fit both PCA models, both pairs of candidate pools and the four selections.
    rows: exactly 384 disjoint rows of one replicate; science has no test inputs.
    raw_root/<case_id>/RAW.npz: original dual recorder layout. model_dir must be
    absent/empty. No retry/substitution or alternate ridge if a scientific gate fails.
    """
    rows=list(rows);model_dir=Path(model_dir)
    if model_dir.exists():check(model_dir.is_dir() and not any(model_dir.iterdir()),'Use fresh empty model_dir')
    else:model_dir.mkdir(parents=True)
    source=_verify_code()
    check(len(set(str(r['rep']) for r in rows))==1 and Counter(r['role'] for r in rows)==COUNTS,'Expected one complete training replicate; forbidden/missing roles')
    check(len(set(r['case_id'] for r in rows))==384 and len(set(int(str(r['key_hex']),16) for r in rows))==384,'Repeated physical ID or Philox key')
    groups={k:sorted((r for r in rows if r['role']==k),key=lambda r:int(r['index'])) for k in COUNTS}
    for k,g in groups.items():
        check([int(r['index']) for r in g]==list(range(1,COUNTS[k]+1)) and all(int(r['fault_id'])==FAULTS[k] for r in g),'Wrong run indices/fault identity for '+k)
    receipts=[]
    newpca=fit_pca_views(_load_role(groups['pca_normal'],raw_root,'pca_normal',receipts));io.npz(model_dir/'PCA_BY_VIEW.npz',**newpca)
    io.write_json(model_dir/'PCA_LOCK.json',dict(created_utc=io.utc(),pca_sha256=io.sha(model_dir/'PCA_BY_VIEW.npz'),roles_read=['pca_normal'],n=78,explained_cutoff=0.95,**source))
    nd=_project(groups['normal_design'],raw_root,'normal_design',receipts,newpca)
    supports={v:ns.build_support(nd[v]) for v in VIEWS}
    geometry={v:hist.normal_design(ns.apply_support(nd[v],supports[v]),list(range(1,41)),core.VARIANT) for v in VIEWS}
    stage=model_dir/'normal_geometry';stage.mkdir()
    for v in VIEWS:io.npz(stage/(v+'.npz'),**supports[v]);io.npz(stage/(v+'_design.npz'),**geometry[v])
    io.write_json(model_dir/'NORMAL_SUPPORT_LOCK.json',dict(created_utc=io.utc(),files=io.seal(stage),pca_lock_sha256=io.sha(model_dir/'PCA_LOCK.json'),roles_read=['pca_normal','normal_design'],faults_read=False))
    ff=_project(groups['fault_fit'],raw_root,'fault_fit',receipts,newpca)
    for v in VIEWS:
        de,*_=core.fit_state(nd[v],ff[v],model_dir/v,support=supports[v])
        for k in geometry[v]:check(np.array_equal(de[k],geometry[v][k]),'Frozen normal geometry changed')
    del ff,nd
    na,ni,fs=(_project(groups[k],raw_root,k,receipts,newpca) for k in ['normal_anchor','normal_inner','fault_select'])
    chosen={v:core.select_state(na[v],ni[v],fs[v],model_dir/v) for v in VIEWS}
    io.write_csv(model_dir/'INPUT_RECEIPTS.csv',receipts)
    result={'created_utc':io.utc(),'rep':str(rows[0]['rep']),'learned_on':dict(COUNTS),'model_order':MONITORS,
            'selected_indices':chosen,'pca_components':{v:int(newpca[v+'__components']) for v in VIEWS},
            'pca_variance_fraction':{v:float(newpca[v+'__explained']) for v in VIEWS},
            'support_rank':{v:int(supports[v]['effective_dim']) for v in VIEWS},'fixed_B':'covariance over first120 sums/sqrt120 from 40 design runs',
            'outer_evaluation_inputs_read':False,**source}
    io.write_json(model_dir/'MODEL_SUMMARY.json',result)
    io.write_json(model_dir/'MODEL_LOCK.json',dict(created_utc=io.utc(),files=io.seal(model_dir),**source))
    return result

@lru_cache(maxsize=12)
def _load_model(resolved):
    root=Path(resolved);lock=io.read_json(root/'MODEL_LOCK.json');src=_verify_code()
    for k,v in src.items():check(lock[k]==v,'Model and current software do not match')
    io.verify_files(root,lock['files'])
    p=io.load_npz(root/'PCA_BY_VIEW.npz');st={v:core.load_state(root/v) for v in VIEWS}
    ch={v:io.read_json(root/v/'SELECTION.json')['choices'] for v in VIEWS}
    return p,st,ch,core.four_anchors(root)

def _signed_batch(pairs,loaded):
    p,st,ch,_=loaded
    obs={v:core.project_raw(vs,core.pca_view(p,v)) for v,vs in pairs.items()}
    sc={v:core.signed_inputs(obs[v],st[v]) for v in VIEWS}
    return np.stack([sc[v][:,:,ch[v][pool]] for _,v,pool in core.CONDITIONS],axis=-1)

def score_pair(post,ci,model_dir):
    pair=validated_pair(post,ci)
    u=_signed_batch({v:a[None] for v,a in pair.items()},_load_model(str(Path(model_dir).resolve())))
    return core.chart(u)[0]

def score_batch(post,ci,model_dir):
    """Batched equivalent. Scoring up to 8–32 runs together reduces loop overhead."""
    p=np.asarray(post);c=np.asarray(ci)
    check(p.ndim==3 and len(p)==len(c) and len(p)>0,'Bad raw batch')
    for a,b in zip(p,c):validated_pair(a,b)
    return core.chart(_signed_batch(dict(native=p,snapshot=c),_load_model(str(Path(model_dir).resolve()))))

def four_score_paths(raw_or_pair,model_dir):
    if isinstance(raw_or_pair,(list,tuple)):return score_pair(*raw_or_pair,model_dir)
    raw=raw_or_pair
    if 'post' in raw and 'ci' in raw:return score_pair(raw['post'],raw['ci'],model_dir)
    flags=raw['diagnostic_IDV'];fault=9 if np.asarray(flags).any() else 0
    pp=raw_pair(raw,fault);return score_pair(pp['native'],pp['snapshot'],model_dir)

def anchors(model_dir):return np.array(_load_model(str(Path(model_dir).resolve()))[-1],copy=True)

def anchored_limits(peaks,model_dir):
    t=np.asarray(peaks);check(t.shape==(199,4),'Need 199 independent whole-run calibration maxima, four frozen columns')
    return core.thresholds(t,anchors(model_dir),190)
