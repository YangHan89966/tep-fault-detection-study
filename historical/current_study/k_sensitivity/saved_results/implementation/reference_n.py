from __future__ import annotations
from typing import Any
import numpy as np
FAMILIES=("signed_level","absolute_deviation","squared_deviation","absolute_increment","lag1_product")

def finite(x: np.ndarray, label: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'NONFINITE:{label}')
    return a

def residual(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    a = finite(x, 'residual_input')
    if a.shape[-1] != 52:
        raise ValueError('RESIDUAL_VARIABLE_COUNT')
    shape = a.shape
    z = (a.reshape(-1,52)-np.asarray(model['mean']))/np.asarray(model['scale'])
    p = np.asarray(model['loadings'])
    return finite((z - (z @ p.T) @ p).reshape(shape), 'PCA_residual')

def feature(r: np.ndarray, center: np.ndarray, family: str) -> np.ndarray:
    r = finite(r, 'feature_input')
    if r.ndim != 3 or r.shape[1] < 2:
        raise ValueError('FEATURE_REQUIRES_RUN_TIME_COORDINATE_TENSOR')
    c = np.asarray(center, dtype=np.float64)
    if c.shape != (r.shape[-1],):
        raise ValueError('RESIDUAL_CENTER_SHAPE')
    if family == 'signed_level':
        a = r.copy()
    elif family == 'absolute_deviation':
        a = np.abs(r-c)
    elif family == 'squared_deviation':
        a = (r-c)**2
    elif family == 'absolute_increment':
        a = np.zeros_like(r)
        a[:,1:,:] = np.abs(r[:,1:,:]-r[:,:-1,:])
    elif family == 'lag1_product':
        a = np.zeros_like(r)
        a[:,1:,:] = (r[:,1:,:]-c)*(r[:,:-1,:]-c)
    else:
        raise ValueError(f'UNKNOWN_FAMILY:{family}')
    return finite(a, f'feature_{family}')

def scalar_input(r: np.ndarray, state: dict) -> np.ndarray:
    if state['status']=='ABSTAIN':
        return np.zeros(r.shape[:2],dtype=np.float64)
    if state['method']=='q_cusum':
        q=np.log1p(np.sum(r*r,axis=2))
        return finite((q-state['q_log_mean'])/state['q_log_sd'],'Q_standardized')
    streams=[]
    for c in state['coordinates']:
        j=c['variable_index'];f=c['family']
        a=feature(r[:,:,[j]],np.array([c['residual_center']]),f)[:,:,0]
        a=c['sign']*(a-c['validation_feature_mean'])/c['validation_feature_sd']
        if f in FAMILIES[3:]: a[:,0]=0.
        streams.append(a)
    return finite(np.stack(streams,axis=-1).mean(axis=-1),'local_scalar')

def cusum(u: np.ndarray, k: float) -> np.ndarray:
    u=finite(u,'cusum_input')
    if u.ndim!=2 or not np.isfinite(k) or k<=0:
        raise ValueError('CUSUM_SHAPE_OR_K')
    out=np.empty_like(u);current=np.zeros(len(u))
    for t in range(u.shape[1]):
        current=np.maximum(0.,current+u[:,t]-float(k))
        out[:,t]=current
    return out
