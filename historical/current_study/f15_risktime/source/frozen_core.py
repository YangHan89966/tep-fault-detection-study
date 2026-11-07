"""Frozen numerical functions copied from the v1.7.1 companion snapshot. No fitting or simulation entrypoints."""

import numpy as np

import math

K = 0.05

def check(ok,msg):
 if not ok: raise ValueError(msg)

def features(obj,design):
 return (obj['z']-design['centre'])@design['whitener']

def recurrence(u,k=K):
 u=np.asarray(u,dtype=np.float64)
 cp=np.zeros((u.shape[0],u.shape[2]));cm=cp.copy();out=np.empty_like(u)
 for t in range(u.shape[1]):
  cp=np.maximum(0.,cp+u[:,t,:]-k)
  cm=np.maximum(0.,cm-u[:,t,:]-k)
  out[:,t,:]=np.maximum(cp,cm)
 return out

def upper_cusum(u):
 p=np.concatenate([np.zeros((len(u),1)),np.cumsum(u-K,axis=1)],axis=1)
 return (p-np.minimum.accumulate(p,axis=1))[:,1:]

def project_axes(obj,design,axes):
    return np.einsum('ntd,ad->nta',features(obj,design),axes,optimize=True)

def transform_inputs(u,p):
    a=(np.abs(u)-p['abs_mean'])/p['abs_sd']
    s=(u*u-p['sq_mean'])/p['sq_sd']
    check(np.isfinite(a).all() and np.isfinite(s).all(),'Nonfinite transformed input')
    return u,a,s

def inputs_and_scores(obj,des,old,new,pars,with_inputs=False):
    # Old block is projected, transformed and accumulated in its old shapes.
    # Appending new candidates therefore cannot alter old reduction ordering.
    uo=project_axes(obj,des,old);un=project_axes(obj,des,new)
    xo=transform_inputs(uo,{k:pars["mean_"+k] for k in ("abs_mean","abs_sd","sq_mean","sq_sd")})
    xn=transform_inputs(un,{k:pars["moment_"+k] for k in ("abs_mean","abs_sd","sq_mean","sq_sd")})
    xx=[np.concatenate((a,b),axis=2) for a,b in zip(xo,xn)]
    dd=[np.concatenate((recurrence(a),recurrence(b)),axis=2) for a,b in zip(xo,xn)]
    qi=(np.log1p(obj["Q"])-float(des["q_mean"]))/float(des["q_sd"])
    out=np.concatenate((*dd,upper_cusum(qi)[:,:,None]),axis=2)
    check(np.isfinite(out).all(),"Nonfinite detector statistic")
    return (out,xx) if with_inputs else out

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

def binomial_sum(k,n,p,upper=False):
    if p<=0:return float(upper is False or k==0)
    if p>=1:return float(upper or k==n)
    rng=range(k,n+1) if upper else range(k+1)
    return min(1.,math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p)) for i in rng))

def clopper_pearson(k,n,alpha=.05):
    if not 0<=k<=n or n<=0:raise ValueError('binomial counts')
    def root(target,upper_tail,kk):
        lo=0.;hi=1.
        for _ in range(64):
            mid=(lo+hi)/2;v=binomial_sum(kk,n,mid,upper_tail)
            if (v<target)==upper_tail:lo=mid
            else:hi=mid
        return (lo+hi)/2
    return (0. if k==0 else root(alpha/2,True,k),
            1. if k==n else root(alpha/2,False,k))
