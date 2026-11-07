import numpy as np
K=0.05

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
