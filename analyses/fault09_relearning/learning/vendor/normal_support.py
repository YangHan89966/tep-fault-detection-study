"""Authorized normal-only support rule; no fault data, no added ridge/fallback.
The cutoff is the ORIGINAL whitening numerical floor, now used as a support
criterion. Full-rank data take an exact identity path (no numerical rotation).
In reduced rank the original learning routines operate on orthonormal retained
coordinates, preserving their square-whitener/raw-coordinate ridge algebra.
"""
from __future__ import annotations
import numpy as np
from common import finite


def build_support(normal: dict) -> dict:
    z=np.asarray(normal['z'],dtype=np.float64)
    if z.ndim!=3 or z.shape[:2]!=(40,960) or z.shape[2]<2:
        raise ValueError('Support must use exactly40 normal_design runs of960 records')
    finite(z,'normal design residuals')
    r=z.shape[2];C=np.cov(z.reshape(-1,r),rowvar=False,ddof=1)
    eig,U=np.linalg.eigh(C);tau=max(1e-10*float(np.trace(C))/r,1e-14)
    keep=eig>=tau;rank=int(keep.sum())
    if rank<2:raise ValueError('Normal effective rank below2; no fallback')
    if eig[0]<-max(tau,1e-12*float(np.linalg.norm(C,2))):
        raise ValueError('Materially negative normal covariance eigenvalue')
    identity=(rank==r)
    if identity:P=np.eye(r,dtype=np.float64)
    else:
        P=U[:,keep].copy()
        for j in range(rank):
            i=int(np.argmax(np.abs(P[:,j])))
            if P[i,j]<0:P[:,j]*=-1
    ans={'projection':P,'input_dim':np.asarray(r,np.int64),'effective_dim':np.asarray(rank,np.int64),
         'identity':np.asarray(identity),'original_C':C,'original_eigenvalues':eig,
         'cutoff':np.asarray(tau),'keep_eigenvalue_mask':keep}
    validate_support(ans)
    return ans


def validate_support(support: dict) -> None:
    P=np.asarray(support['projection']);r=int(support['input_dim']);d=int(support['effective_dim'])
    if not 2<=d<=r or P.shape!=(r,d):raise ValueError('Invalid normal support dimensions')
    finite(P,'normal support projection')
    if not np.allclose(P.T@P,np.eye(d),rtol=0,atol=2e-12):
        raise ValueError('Normal support columns not orthonormal')
    if bool(support['identity']):
        if r!=d or not np.array_equal(P,np.eye(r)):raise ValueError('Full-rank identity path changed')
    elif d==r:raise ValueError('Full rank must preserve original identity path')


def apply_support(obj: dict, support: dict) -> dict:
    validate_support(support)
    if obj['z'].shape[-1]!=int(support['input_dim']):raise ValueError('Residual/support coordinate mismatch')
    if bool(support['identity']):return obj
    out=dict(obj);out['z']=finite(np.asarray(obj['z'])@support['projection'],'supported residuals')
    # Q and record markers retain their original raw-record meaning. Q is not a
    # monitor in this experiment; it is passed only to the inherited design code.
    return out


def support_from_design(des: dict) -> dict:
    if 'support_projection' not in des:
        d=len(des['centre']);return dict(projection=np.eye(d),input_dim=d,effective_dim=d,identity=True)
    return dict(projection=des['support_projection'],input_dim=des['support_input_dim'],
                effective_dim=des['support_effective_dim'],identity=des['support_identity'])
