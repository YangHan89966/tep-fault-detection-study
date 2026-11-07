"""Fixed manuscript dense learning/selection/calibration, NumPy only.
No simulator import, seed creation, model search or access to earlier Final data.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import inherited as old
from storage import write_json,write_csv,seal_tree,verify_tree,sha,strict_json

FAMILIES=('signed','absolute','squared')
VARIANT={'kind':'ridge','relative_floor':0.,'ridge_relative':1e-4}


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def arrays(path,**kwargs):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**kwargs)


def project(values,pca,basis):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=3 or a.shape[1:]!=(960,52) or not np.isfinite(a).all():raise ValueError('bad released trajectory array')
    z=(a-pca['mean'])/pca['scale']
    r=z-(z@pca['loadings'].T)@pca['loadings']
    sub=r@basis
    q=np.sum(r*r,axis=2)
    if not np.allclose(np.sum(sub*sub,axis=2),q,rtol=1e-9,atol=1e-9):raise ValueError('fixed residual projector mismatch')
    starts=[];sizes=[]
    for run in a:
        different=np.flatnonzero(np.any(run[:,:22]!=run[-1,:22],axis=1))
        start=1 if not len(different) else int(different[-1])+2
        starts.append(start);sizes.append(961-start)
    return {'z':sub,'Q':q,'run_id':np.arange(1,len(a)+1,dtype=np.int64),
            'record_suffix_start':np.array(starts),'record_suffix_samples':np.array(sizes)}


def fit_one(nd,fit,fault,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    des=old.normal_design(nd,list(range(1,len(nd['z'])+1)),VARIANT)
    H,mat,windows=old.fit_matrices(fit,nd,des,fault)
    all_axes,labels,small,recipes,endpoints,diag=old.build_axes_regularized(
        mat['mean_early'],mat['mean_late'],des['B'],des,VARIANT)
    mean=all_axes[small].copy()
    variance=np.einsum('ad,de,ae->a',mean,des['G'],mean)
    old.check(np.all(np.isfinite(variance)) and np.all(variance>1e-24),'mean direction unit variance')
    mean=mean/np.sqrt(variance)[:,None]
    moment,mlabels,mdiag=old.learned_axes(H,des,'ridge_1e4')
    pars=old.parameters(nd,des,mean,moment)
    arrays(out/'DESIGN.npz',**des)
    arrays(out/'MODEL.npz',mean=mean,moment=moment)
    arrays(out/'FIT_MOMENTS.npz',**mat)
    arrays(out/'PARAMETERS.npz',**pars)
    write_csv(out/'FIT_WINDOWS.csv',windows)
    write_json(out/'FIT_INFO.json',{'fault':fault,'mean_count':len(mean),'moment_count':len(moment),
      'mean_labels':[labels[j] for j in small],'moment_labels':mlabels,
      'mean_diagnostics':diag,'moment_diagnostics':mdiag,
      'selection_used':False,'outer_used':False,'evaluation_used':False})
    return des,mean,moment,pars


def state(out):
    des=load_npz(out/'DESIGN.npz');a=load_npz(out/'MODEL.npz');p=load_npz(out/'PARAMETERS.npz')
    return des,a['mean'],a['moment'],p


def scores(obj,st):return old.inputs_and_scores(obj,*st)


def events(mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=3 or mask.shape[1]!=960:raise ValueError('mask shape')
    pre=mask[:,:160].any(1);post=mask[:,160:];det=post.any(1)
    first=np.where(det,post.argmax(1),800).astype(np.int64)
    clean=det&~pre;loss=np.where(clean,first,800)
    occ=post.sum(1)
    after=(np.arange(800)[None,:,None]>=first[:,None,:])
    withdraw=(after&~post).sum(1)
    if not np.array_equal(occ+withdraw+first,np.full_like(first,800)):raise ValueError('occupancy identity')
    return {'prefault_alarm':pre.astype(np.int64),'detected':det.astype(np.int64),
            'clean_detection':clean.astype(np.int64),'first_delay_samples_penalized':first,
            'event_loss_samples':loss,'occupancy_points':occ,'postalarm_below_points':withdraw}


def thresholds(peaks,anchors,rank):
    peaks=np.asarray(peaks);anchors=np.asarray(anchors)
    if peaks.ndim!=2 or anchors.shape!=(peaks.shape[1],) or not np.isfinite(peaks).all() or not np.all(anchors>0):raise ValueError('threshold inputs')
    if not 1<=rank<=len(peaks):raise ValueError('rank')
    # Preserve original anchored division/order/multiplication floating order.
    q=np.sort(peaks/anchors,axis=0)[rank-1]
    h=anchors*q
    if not np.isfinite(h).all():raise ValueError('nonfinite threshold')
    return h,q


def select_one(na,ni,fs,fault,out):
    st=state(out);mean,moment=st[1:3];km=len(mean);ka=len(moment);k=km+ka
    anchor_peaks=scores(na,st).max(1)
    anchors=np.quantile(anchor_peaks,.9,axis=0,method='linear')
    old.check(np.all(anchors>0),'nonpositive anchor; no candidates silently dropped')
    inner_peaks=scores(ni,st).max(1);hin,iq=thresholds(inner_peaks,anchors,57)
    ev=events(scores(fs,st)>hin);risk=ev['event_loss_samples'].mean(0)
    def best(ids):
        low=float(risk[ids].min());return next(int(i) for i in ids if risk[i]<=low+1e-12)
    choices={}
    for g,fam in enumerate(FAMILIES):choices['mean_pool_'+fam]=best(list(range(g*k,g*k+km)))
    for g,fam in enumerate(FAMILIES):choices['moment_pool_'+fam]=best(list(range(g*k+km,(g+1)*k)))
    j=choices['mean_pool_signed']
    choices['same_mean_signed_axis_absolute']=k+j
    choices['same_mean_signed_axis_squared']=2*k+j
    choices['Q_CUSUM']=3*k
    nam=old.names(km,ka)
    write_json(out/'SELECTION.json',{'fault':fault,'choices':choices,'candidate_names':nam,
       'rule':'minimum complete-episode event loss, first candidate on exact integer-total tie',
       'outer_used':False,'evaluation_used':False})
    arrays(out/'INNER.npz',anchors=anchors,anchor_peaks=anchor_peaks,inner_peaks=inner_peaks,
           inner_q=iq,inner_thresholds=hin,**ev)
    write_csv(out/'SELECTION_SCORES.csv',[{'candidate':n,'index':j,'loss_minutes':float(risk[j]*3),
      'prealarms':int(ev['prefault_alarm'][:,j].sum()),'clean_detected':int(ev['clean_detection'][:,j].sum())} for j,n in enumerate(nam)])
    return choices


def calibrate_one(no,out,rank=247):
    st=state(out);anchors=load_npz(out/'INNER.npz')['anchors']
    peaks=scores(no,st).max(1);h,q=thresholds(peaks,anchors,rank)
    arrays(out/'OUTER.npz',peaks=peaks,thresholds=h,q=q,rank=np.array(rank),n=np.array(len(no['z'])))
    # Model selection is not revisited after independent outer calibration.
    return h


def selected_outcomes(obj,modeldir):
    st=state(modeldir);sel=strict_json(modeldir/'SELECTION.json');h=load_npz(modeldir/'OUTER.npz')['thresholds']
    methods=list(sel['choices']);idx=np.array([sel['choices'][m] for m in methods])
    mask=scores(obj,st)[:,:,idx]>h[idx]
    return methods,events(mask),mask


def serial_scalar_check(obj,modeldir,limit_runs=2):
    """Independent simple scalar-loop witness; does not affect threshold decisions."""
    st=state(modeldir);sel=strict_json(modeldir/'SELECTION.json');des,mean,moment,pars=st
    witness={k:v[:limit_runs] if isinstance(v,np.ndarray) and len(v)==len(obj['z']) else v for k,v in obj.items()}
    target=scores(witness,st);maxerr=0.;comparisons=0
    # Dot-product inputs computed in source shape; scalar recurrence independently checked.
    uo=old.project_axes(witness,des,mean);un=old.project_axes(witness,des,moment)
    xo=old.transform_inputs(uo,{k:pars['mean_'+k] for k in ('abs_mean','abs_sd','sq_mean','sq_sd')})
    xn=old.transform_inputs(un,{k:pars['moment_'+k] for k in ('abs_mean','abs_sd','sq_mean','sq_sd')})
    inputs=np.concatenate([np.concatenate((a,b),2) for a,b in zip(xo,xn)],2)
    kcount=inputs.shape[2]
    for r in range(len(witness['z'])):
        for j in range(kcount):
            cp=cm=0.
            for t,x in enumerate(inputs[r,:,j]):
                cp=max(0.,(cp+float(x))-.05);cm=max(0.,(cm-float(x))-.05)
                maxerr=max(maxerr,abs(max(cp,cm)-float(target[r,t,j])));comparisons+=1
    if maxerr>1e-10:raise ValueError('scalar CUSUM discrepancy')
    return {'comparisons':comparisons,'max_absolute_error':maxerr}
