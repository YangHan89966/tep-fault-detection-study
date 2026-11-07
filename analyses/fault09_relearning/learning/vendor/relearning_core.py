"""Two established direction pools; Signed CUSUM only, dimension-generalized.
All fitting consumes explicitly assigned normal_design and fault_fit roles.
Selection consumes anchor/inner/selection only. No evaluation-time re-selection.
"""
from __future__ import annotations
import numpy as np
import inherited_dim as old
from common import npz,write_json,write_csv,load_npz,finite
from normal_support import build_support,apply_support,support_from_design,validate_support
VARIANT={'kind':'ridge','relative_floor':0.,'ridge_relative':1e-4}
CONDITIONS=[('M1','snapshot','moment'),('M2','native','moment'),('M3','snapshot','mean'),('M4','native','mean')]

def pca_view(bundle,view):return {k:bundle[view+'__'+k] for k in ('mean','scale','loadings','basis','components','eigenvalues','explained')}
def project_raw(x,pca):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(960,52):raise ValueError('raw must be [n,960,52]')
    finite(x);P=pca['loadings'];basis=pca['basis']
    r=52-P.shape[1]
    if basis.shape!=(52,r):raise ValueError('Wrong record-specific residual basis')
    z=(x-pca['mean'])/pca['scale'];e=z-(z@P)@P.T
    sub=e@basis;q=np.sum(e*e,axis=2)
    if not np.allclose(np.sum(sub*sub,axis=2),q,rtol=1e-9,atol=1e-9):raise ValueError('Residual subspace energy mismatch')
    starts=[];sizes=[]
    for run in x:
        diff=np.flatnonzero(np.any(run[:,:22]!=run[-1,:22],axis=1));start=1 if not len(diff) else int(diff[-1])+2
        starts.append(start);sizes.append(961-start)
    return dict(z=finite(sub),Q=q,run_id=np.arange(1,len(x)+1,dtype=np.int64),record_suffix_start=np.array(starts),record_suffix_samples=np.array(sizes))

def fit_state(nd,fit,out,support=None):
    out.mkdir(parents=True,exist_ok=False)
    if len(nd['z'])!=40 or len(fit['z'])!=100:raise ValueError('Fixed fitting roles/counts violated')
    if support is None:support=build_support(nd)
    validate_support(support)
    original_dim=nd['z'].shape[-1]
    nd=apply_support(nd,support);fit=apply_support(fit,support)
    des=old.normal_design(nd,list(range(1,41)),VARIANT)
    H,mat,windows=old.fit_matrices(fit,nd,des,9)
    axes,labels,small,recipes,endpoints,diag=old.build_axes_regularized(mat['mean_early'],mat['mean_late'],des['B'],des,VARIANT)
    mean=axes[small].copy();v=np.einsum('ad,de,ae->a',mean,des['G'],mean)
    if not np.all(v>1e-24):raise ValueError('Degenerate mean direction')
    mean=mean/np.sqrt(v)[:,None]
    moment,mlabels,mdiag=old.learned_axes(H,des,'ridge_1e4')
    if not (1<=len(mean)<=8 and 1<=len(moment)<=8):raise ValueError('Candidate budget violation')
    des.update(support_projection=support['projection'],support_input_dim=support['input_dim'],
               support_effective_dim=support['effective_dim'],support_identity=support['identity'],
               support_cutoff=support['cutoff'])
    npz(out/'NORMAL_SUPPORT.npz',**support)
    npz(out/'DESIGN.npz',**des);npz(out/'MODEL.npz',mean=mean,moment=moment);npz(out/'FIT_MOMENTS.npz',**mat)
    write_csv(out/'FIT_WINDOWS.csv',windows)
    write_json(out/'FIT_INFO.json',dict(residual_dimension=int(original_dim),effective_dimension=int(nd['z'].shape[2]),normal_support_rule='original normal-only eigenfloor support; authorized v1.1 amendment',mean_count=len(mean),moment_count=len(moment),mean_labels=[labels[j] for j in small],moment_labels=mlabels,mean_diagnostics=diag,moment_diagnostics=mdiag,selection_used=False,evaluation_used=False,signed_only=True))
    return des,mean,moment

def load_state(folder):
    des=load_npz(folder/'DESIGN.npz');m=load_npz(folder/'MODEL.npz');return des,m['mean'],m['moment']
def signed_inputs(obj,state):
    des,mean,moment=state
    # Identity branch returns original object; native floating-point path unchanged.
    obj=apply_support(obj,support_from_design(des))
    # Same block projection shape/order as inherited R5, no single-axis replacement.
    return np.concatenate((old.project_axes(obj,des,mean),old.project_axes(obj,des,moment)),axis=2)
def chart(u):return old.recurrence(u,.05)
def scalar_chart(u):
    u=np.asarray(u,dtype=float);out=np.zeros_like(u)
    for i in range(len(u)):
        for j in range(u.shape[2]):
            cp=cm=0.
            for t,x in enumerate(u[i,:,j]):
                cp=max(0.,(cp+float(x))-.05);cm=max(0.,(cm-float(x))-.05);out[i,t,j]=max(cp,cm)
    return out

def events(mask):
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=3 or m.shape[1]!=960:raise ValueError('Alarm shape')
    pre=m[:,:160].any(1);post=m[:,160:];det=post.any(1)
    ds=np.where(det,post.argmax(1),800).astype(np.int64);clean=det&~pre
    result=dict(normal_alarm=m.any(1),pre_alarm=pre,detected_post=det,clean=clean,delay_samples=ds,loss_minutes=np.where(clean,3*ds,2400),post_occupancy_points=post.sum(1))
    for h in (120,480,2400):result['clean_'+str(h)]=clean&(3*ds<=h)
    return result

def thresholds(peaks,anchor,rank):
    peaks=np.asarray(peaks);anchor=np.asarray(anchor)
    if peaks.ndim!=2 or anchor.shape!=(peaks.shape[1],) or not np.all(anchor>0) or not 1<=rank<=len(peaks):raise ValueError('Bad anchored calibration')
    finite(peaks);finite(anchor)
    h=anchor*np.sort(peaks/anchor,axis=0)[rank-1]
    return finite(h)
def select_state(na,ni,fs,folder):
    if (len(na['z']),len(ni['z']),len(fs['z']))!=(32,59,75):raise ValueError('Fixed selection roles/counts violated')
    st=load_state(folder);km=len(st[1]);ka=len(st[2]);n=km+ka
    ap=chart(signed_inputs(na,st)).max(1);a=np.quantile(ap,.9,axis=0,method='linear')
    ip=chart(signed_inputs(ni,st)).max(1);hin=thresholds(ip,a,57)
    m=chart(signed_inputs(fs,st))>hin;ev=events(m)
    integer_loss=(ev['loss_minutes']//3).sum(0)
    choices={'mean':int(np.argmin(integer_loss[:km])),'moment':int(km+np.argmin(integer_loss[km:]))}
    info=__import__('json').loads((folder/'FIT_INFO.json').read_text())
    labels=info['mean_labels']+info['moment_labels']
    rows=[]
    for j in range(n):
        rows.append(dict(candidate_index=j,pool='mean' if j<km else 'moment',source_labels='|'.join(labels[j]),loss_samples_total=int(integer_loss[j]),mean_loss_minutes=float(integer_loss[j]*3/75),prefault_alarm_n=int(ev['pre_alarm'][:,j].sum()),clean_n=int(ev['clean'][:,j].sum()),selected=bool(j in choices.values())))
    write_csv(folder/'SELECTION_SCORES.csv',rows)
    write_json(folder/'SELECTION.json',dict(choices=choices,mean_count=km,moment_count=ka,candidate_labels=labels,tie_rule='minimum exact integer total, first inherited candidate index',outer_used=False,evaluation_used=False))
    npz(folder/'INNER.npz',anchors=a,anchor_peaks=ap,inner_peaks=ip,inner_thresholds=hin,selection_alarm_packed=np.packbits(m,axis=1,bitorder='little'),selection_alarm_shape=np.array(m.shape),loss_samples_total=integer_loss)
    return choices

def selected_inputs(obj,folder):
    import json
    st=load_state(folder);ch=json.loads((folder/'SELECTION.json').read_text())['choices'];u=signed_inputs(obj,st)
    return u[:,:,[ch['moment'],ch['mean']]]
def four_inputs(objs,models):
    s=selected_inputs(objs['snapshot'],models/'snapshot');n=selected_inputs(objs['native'],models/'native')
    return np.stack((s[:,:,0],n[:,:,0],s[:,:,1],n[:,:,1]),axis=-1)
def four_anchors(models):
    import json
    ans=[]
    for _,view,pool in CONDITIONS:
        ch=json.loads((models/view/'SELECTION.json').read_text())['choices'][pool];ans.append(load_npz(models/view/'INNER.npz')['anchors'][ch])
    return np.array(ans)
