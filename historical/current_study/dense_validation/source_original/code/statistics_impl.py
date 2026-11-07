"""Frozen per-run paired summaries; no SciPy runtime dependency."""
from __future__ import annotations
import hashlib,math
import numpy as np


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


def rng_for(label):
    n=int.from_bytes(hashlib.sha256(('DENSE3915_VALIDATION_v1.0|bootstrap|'+label).encode('ascii')).digest()[:8],'big')
    return np.random.Generator(np.random.PCG64(n))


def bootstrap_means(diffs,label,draws=20000):
    d=np.asarray(diffs,dtype=float)
    if d.ndim==1:d=d[:,None]
    rng=rng_for(label);out=np.empty((draws,d.shape[1]))
    for lo in range(0,draws,100):
        n=min(100,draws-lo);ix=rng.integers(0,len(d),size=(n,len(d)),dtype=np.int64)
        out[lo:lo+n]=d[ix].mean(1)
    return out


def summarize(fault,methods,ev,nmask):
    out=[];n=ev['event_loss_samples'].shape[0];nn=nmask.shape[0]
    for j,m in enumerate(methods):
        row={'fault':fault,'method':m,'fault_n':n,'normal_n':nn,
             'loss_minutes':float(ev['event_loss_samples'][:,j].mean()*3),
             'occupancy_pct':float(ev['occupancy_points'][:,j].mean()/8),
             'postalarm_withdrawal_points':int(ev['postalarm_below_points'][:,j].sum())}
        rates={'normal_FAR':nmask[:,:,j].any(1),'prefault_alarm':ev['prefault_alarm'][:,j].astype(bool),
               'clean_detection':ev['clean_detection'][:,j].astype(bool),'postfault_detection':ev['detected'][:,j].astype(bool)}
        for horizon in (60,120,480):rates[f'clean_by_{horizon}min']=rates['clean_detection']&(3*ev['first_delay_samples_penalized'][:,j]<=horizon)
        for name,a in rates.items():
            k=int(a.sum());low,high=clopper_pearson(k,len(a))
            row.update({name+'_count':k,name+'_pct':100*k/len(a),name+'_CP95_low_pct':100*low,name+'_CP95_high_pct':100*high})
        out.append(row)
    return out


def contrast(fault,c,methods,ev,nmask):
    a=methods.index(c['a']);b=methods.index(c['b'])
    diff=3*(ev['event_loss_samples'][:,a]-ev['event_loss_samples'][:,b]).astype(float)
    boot=bootstrap_means(diff,f'{fault:02d}')[:,0]
    qs=np.quantile(boot,[.025,.975,1/120,1-1/120],method='linear')
    pos=c['predicted_sign'] if 'predicted_sign' in c else c.get('expected_direction')
    predicted=1 if fault==3 else -1
    degenerate=bool(np.all(diff==diff[0]))
    status='EMPIRICAL_DEGENERATE_DESCRIPTIVE_ONLY' if degenerate else ('SUPPORTS_PREDICTED_LOSS_DIRECTION' if (qs[2]>0 if predicted==1 else qs[3]<0) else ('OPPOSITE_LOSS_DIRECTION' if (qs[3]<0 if predicted==1 else qs[2]>0) else 'INCONCLUSIVE_LOSS_CONTRAST'))
    nd=(nmask[:,:,a].any(1).astype(float)-nmask[:,:,b].any(1).astype(float))*100
    nb=bootstrap_means(nd,'normal')[:,0];nq=np.quantile(nb,[.025,.975])
    row={'contrast':c['id'],'fault':fault,'a':c['a'],'b':c['b'],'mean_loss_difference_minutes':float(diff.mean()),
       'marginal95_lower':float(qs[0]),'marginal95_upper':float(qs[1]),
       'bonferroni98_333_lower':float(qs[2]),'bonferroni98_333_upper':float(qs[3]),
       'status':status,'normal_FAR_difference_pp':float(nd.mean()),
       'normal_FAR_difference95_lower_pp':float(nq[0]),'normal_FAR_difference95_upper_pp':float(nq[1]),
       'normal_FAR_equivalence_claimed':False,'training_refit':False,'draws':20000}
    return row,boot,nb
