"""Exact S22 build_axes_regularized body; import path adapted, no formula change."""
import numpy as np
from core import build_axes,canonical,check
def build_axes_regularized(a,b,B,design,variant):
    alpha=float(variant.get('ridge_relative',0.))
    if alpha==0:
        return build_axes(a,b,B)
    W=design['whitener'];check(W.shape==(16,16),'ridge retains full16')
    Winv=np.linalg.inv(W);C=design['raw_C'];D=design['raw_D']
    ao=np.linalg.solve(W.T,a);bo=np.linalg.solve(W.T,b)
    d=len(a);t=float(np.trace(B))/d
    recipes=[('unweighted',np.eye(d)),('shrink50',.5*B+.5*t*np.eye(d)),
             ('shrink10',.9*B+.1*t*np.eye(d)),('raw',B)]
    axes=[];labels=[];original={};formulas=[];details=[]
    def add(v,label):
        v=canonical(v)
        if v is None:return None
        for i,w in enumerate(axes):
            if min(np.max(np.abs(v-w)),np.max(np.abs(v+w)))<=1e-10:
                labels[i].append(label);return i
        axes.append(v);labels.append([label]);return len(axes)-1
    for name,T in recipes:
        A=Winv.T@T@Winv;A=(A+A.T)/2
        rho=alpha*float(np.trace(A))/d
        Areg=A+rho*np.eye(d)
        check(rho>0 and np.linalg.eigvalsh(Areg)[0]>0,'positive original-space loading')
        ia=np.linalg.solve(Areg,ao);ib=np.linalg.solve(Areg,bo)
        sign=1 if float(ao@ib)>=0 else -1
        vectors=[np.linalg.solve(W,ia+sign*ib),np.linalg.solve(W,ia-sign*ib)]
        # Independent equivalent representation, penalty pulled back WITH W.T W.
        Treg=T+rho*(W.T@W)
        p_a=np.linalg.solve(Treg,a);p_b=np.linalg.solve(Treg,b)
        diffs=[]
        for vo,vw in zip([ia+sign*ib,ia-sign*ib],[p_a+sign*p_b,p_a-sign*p_b]):
            denom=max(float(np.linalg.norm(vo)),1e-30)
            diffs.append(float(np.linalg.norm(vo-W@vw)/denom))
        check(max(diffs)<1e-6,'coordinate loading equivalence')
        # Raw branch remains same after an arbitrary invertible diagonal pre-transform,
        # provided penalty is transformed as a metric rather than replaced by I.
        transform_error=0.
        if name=='raw':
            S=np.diag(np.geomspace(.2,5.,16))
            Ds=S.T@A@S;penalty=rho*S.T@S
            for mu,solution in [(ao,ia),(bo,ib)]:
                alt=S@np.linalg.solve(Ds+penalty,S.T@mu)
                transform_error=max(transform_error,float(np.linalg.norm(alt-solution)/max(np.linalg.norm(solution),1e-30)))
            check(transform_error<1e-6,'RAW penalty undone by coordinate change')
        formulas.append((name,*vectors,sign))
        details.append({'recipe':name,'original_relative_sign':int(sign),'weighted_inner_product':float(ao@ib),
                        'ridge_alpha':alpha,'rho_original_units':rho,'normal_metric_trace':float(np.trace(A)),
                        'regularized_min_eigenvalue':float(np.linalg.eigvalsh(Areg)[0]),
                        'raw16_dimensional_rank':16,'coordinate_formula_max_relative_error':max(diffs),
                        'RAW_arbitrary_rescaling_relative_error':transform_error,
                        'normalized_gain_guarantee':'NOT_CLAIMED; measured later'})
    for name,v,other,s in formulas:original[name]=add(v,name+'_original_merge')
    for name,v,other,s in formulas:add(other,name+'_complementary_merge')
    small=list(range(len(axes)))
    # These are unchanged external controls, NOT eligible protected candidates.
    endpoints=[add(a,'early_endpoint_axis'),add(b,'late_endpoint_axis')]
    check(len(small)>=2 and all(i is not None for i in original.values()) and all(i is not None for i in endpoints),'degenerate pool')
    return np.array(axes),labels,small,original,tuple(sorted(set(endpoints))),details
