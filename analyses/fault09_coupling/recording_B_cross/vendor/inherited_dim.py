# Dimension-generalized copy: only residual-dimension constants changed; see source_notes/DIMENSION_PATCH.diff.
"""Inherited S28 pure numerical function bodies, unchanged; source map in notes.
No old entrypoint, data path, or simulator invocation. Only ridge_1e4 used.
"""
import math
import numpy as np
K=.05; START=160; H=800; T=960; L=120
FAMS=("signed","absolute","squared")
def check(ok,msg):
 if not ok: raise ValueError(msg)


def canonical(v):
 norm=float(np.linalg.norm(v))
 if norm<=1e-12:return None
 v=v/norm;j=int(np.argmax(np.abs(v)))
 if v[j]<0:v=-v
 return v


def features(obj,design):
 return (obj['z']-design['centre'])@design['whitener']


def normal_design(normal,ids,variant):
 at=np.array(ids,dtype=np.int64)-1;x=normal['z'][at];q=normal['Q'][at]
 dim_raw=x.shape[-1];flat=x.reshape(-1,dim_raw);centre=flat.mean(0);C=np.cov(flat,rowvar=False,ddof=1)
 v,U=np.linalg.eigh(C);floor0=max(float(np.trace(C))/dim_raw*1e-10,1e-14)
 tau=max(floor0,float(variant['relative_floor'])*float(np.trace(C))/dim_raw)
 if variant['kind']=='truncate':
  keep=v>=tau;check(keep.sum()>=2,'Retained rank below2; no automatic fallback')
  W=U[:,keep]/np.sqrt(v[keep])
 else:
  keep=np.ones(dim_raw,dtype=bool);W=(U/np.sqrt(np.maximum(v,tau)))@U.T
 y=(x-centre)@W;dim=W.shape[1];G=np.cov(y.reshape(-1,dim),rowvar=False,ddof=1)
 B=np.cov(y[:,:120].sum(1)/math.sqrt(120.),rowvar=False,ddof=1)
 be=np.linalg.eigvalsh(B);check(be[0]>np.trace(B)/dim*1e-12,'RAW temporal covariance SPD rule failed')
 D=np.cov((x[:,:120]-centre).sum(1)/math.sqrt(120.),rowvar=False,ddof=1)
 lq=np.log1p(q);sd=float(lq.std(ddof=1));check(sd>0,'Zero Q design SD')
 return {'centre':centre,'whitener':W,'B':B,'q_mean':float(lq.mean()),'q_sd':sd,'raw_C':C,'raw_D':D,'G':G,'eigenvalues':v,'keep':keep,'tau':tau}


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


def event_rows(mask,ids,fault,role,system,recent_mask=None):
 pre=mask[:,:START].any(axis=1);post=mask[:,START:];det=post.any(axis=1)
 first=np.where(det,post.argmax(axis=1),H).astype(np.int64)
 loss=np.where(pre|~det,H,first);below=(np.arange(H)[None,:]>=first[:,None])&~post
 out=[]
 for i,rid in enumerate(ids):
  row={'fault':fault,'role':role,'system':system,'run_id':int(rid),'prefault_alarm':int(pre[i]),'detected':int(det[i]),'clean_detection':int(det[i] and not pre[i]),'first_delay_samples_penalized':int(first[i]),'event_loss_samples':int(loss[i]),'occupancy_points':int(post[i].sum()),'postalarm_below_points':int(below[i].sum()),'late32h_above_points':int(mask[i,320:].sum()),'last8h_above_points':int(mask[i,800:].sum()),'terminal_above':int(mask[i,-1])}
  row['last8h_recent120_points']=int(recent_mask[i,800:].sum()) if recent_mask is not None else -1
  check(row['occupancy_points']+row['postalarm_below_points']+int(first[i])==H,'occupancy identity')
  out.append(row)
 return out


def normal_rows(mask,ids,fault,system):
 return [{'fault':fault,'system':system,'run_id':int(rid),'alarm':int(mask[i].any()),'above_points':int(mask[i].sum()),'first_alarm_sample':int(mask[i].argmax()+1) if mask[i].any() else -1} for i,rid in enumerate(ids)]


def endpoint_means(y, run_ids, starts, suffixes, policy, fault):
    if y.ndim!=3 or y.shape[:2]!=(len(run_ids),960):raise ValueError('Fit input shape')
    if len(set(map(int,run_ids)))!=len(run_ids):raise ValueError('Duplicate fit IDs')
    if not (len(starts)==len(suffixes)==len(run_ids)):raise ValueError('Fit marker lengths')
    if not np.array_equal(961-np.asarray(starts),suffixes):raise ValueError('Suffix identity')
    threshold=160 if policy=='pre_marker_160' else 2
    rows=[];early=[];late=[];all_default=True
    for i,rid in enumerate(run_ids):
        b=int(starts[i]);length=int(suffixes[i]);mark=int(length>=threshold)
        if policy=='legacy_endpoints':lo1,hi1,lo2,hi2=160,320,800,960;effective_end=960
        else:
            effective_end=b-1 if mark else 960
            available=effective_end-160
            if available<=0:raise ValueError(f'No premarker fault data fault{fault} run{rid}; do not omit')
            q=min(160,available);lo1=160;hi1=160+q
            lo2=effective_end-q;hi2=effective_end
            if policy=='early_only_pre2':lo2,hi2=lo1,hi1
            if (lo1,hi1,lo2,hi2)!=(160,320,800,960):all_default=False
        early.append(y[i,lo1:hi1].mean(axis=0));late.append(y[i,lo2:hi2].mean(axis=0))
        n1=hi1-lo1;n2=hi2-lo2
        # 0-based first marked point b-1. All exposure independent of detector.
        def exposure(lo,hi,minimum):
            return max(0,hi-max(lo,b-1)) if length>=minimum else 0
        rows.append({'fault':fault,'run_id':int(rid),'policy':policy,'record_suffix_start':b,'record_suffix_samples':length,
            'used_last_sample':max(hi1,hi2),'prefix_available_fault_points':effective_end-160,
            'early_first_sample':lo1+1,'early_last_sample':hi1,'early_points':n1,
            'late_first_sample':lo2+1,'late_last_sample':hi2,'late_points':n2,
            'overlap_points':max(0,min(hi1,hi2)-max(lo1,lo2)),
            'early_main160_tail_points':exposure(lo1,hi1,160),'late_main160_tail_points':exposure(lo2,hi2,160),
            'early_any2_tail_points':exposure(lo1,hi1,2),'late_any2_tail_points':exposure(lo2,hi2,2),
            'equal_run_weight':1./len(run_ids)})
    # Preserve historical summation order in unchanged controls.
    if policy=='legacy_endpoints' or all_default:
        a=y[:,160:320].mean(axis=(0,1));b=y[:,800:960].mean(axis=(0,1))
    else:
        a=np.stack(early).mean(axis=0);b=np.stack(late).mean(axis=0)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('Nonfinite fitted endpoint')
    return a,b,rows


def build_axes_regularized(a,b,B,design,variant):
    alpha=float(variant.get('ridge_relative',0.))
    if alpha==0:
        return build_axes(a,b,B)
    W=design['whitener'];check(W.shape==(len(a),len(a)),'ridge retains full residual dimension')
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
            S=np.diag(np.geomspace(.2,5.,d))
            Ds=S.T@A@S;penalty=rho*S.T@S
            for mu,solution in [(ao,ia),(bo,ib)]:
                alt=S@np.linalg.solve(Ds+penalty,S.T@mu)
                transform_error=max(transform_error,float(np.linalg.norm(alt-solution)/max(np.linalg.norm(solution),1e-30)))
            check(transform_error<1e-6,'RAW penalty undone by coordinate change')
        formulas.append((name,*vectors,sign))
        details.append({'recipe':name,'original_relative_sign':int(sign),'weighted_inner_product':float(ao@ib),
                        'ridge_alpha':alpha,'rho_original_units':rho,'normal_metric_trace':float(np.trace(A)),
                        'regularized_min_eigenvalue':float(np.linalg.eigvalsh(Areg)[0]),
                        'raw_dimensional_rank':d,'coordinate_formula_max_relative_error':max(diffs),
                        'RAW_arbitrary_rescaling_relative_error':transform_error,
                        'normalized_gain_guarantee':'NOT_CLAIMED; measured later'})
    for name,v,other,s in formulas:original[name]=add(v,name+'_original_merge')
    for name,v,other,s in formulas:add(other,name+'_complementary_merge')
    small=list(range(len(axes)))
    # These are unchanged external controls, NOT eligible protected candidates.
    endpoints=[add(a,'early_endpoint_axis'),add(b,'late_endpoint_axis')]
    check(len(small)>=2 and all(i is not None for i in original.values()) and all(i is not None for i in endpoints),'degenerate pool')
    return np.array(axes),labels,small,original,tuple(sorted(set(endpoints))),details


def project_axes(obj,design,axes):
    return np.einsum('ntd,ad->nta',features(obj,design),axes,optimize=True)


def transform_parameters(u):
    check(np.isfinite(u).all(),'Nonfinite normal design projection')
    a=np.abs(u);s=u*u
    out={'abs_mean':a.mean((0,1)), 'abs_sd':a.reshape(-1,a.shape[-1]).std(0,ddof=1),
         'sq_mean':s.mean((0,1)), 'sq_sd':s.reshape(-1,s.shape[-1]).std(0,ddof=1)}
    check(all(np.isfinite(v).all() for v in out.values()),'Nonfinite transform moments')
    check(np.all(out['abs_sd']>1e-12) and np.all(out['sq_sd']>1e-12),
          'Non-estimable transformed normal SD; no silent family deletion')
    return out


def transform_inputs(u,p):
    a=(np.abs(u)-p['abs_mean'])/p['abs_sd']
    s=(u*u-p['sq_mean'])/p['sq_sd']
    check(np.isfinite(a).all() and np.isfinite(s).all(),'Nonfinite transformed input')
    return u,a,s


def fit_matrices(fit,normal_design_obj,des,fault):
    """Every value read here belongs to fault FIT or normal DESIGN roles."""
    y=features(fit,des)
    a,b,rows=endpoint_means(y,fit["run_id"],fit["record_suffix_start"],
                          fit["record_suffix_samples"],"pre_marker_160",fault)
    d=y.shape[2];moments=[];within=[];locations=[]
    for i,r in enumerate(rows):
        pieces=[y[i,r["early_first_sample"]-1:r["early_last_sample"]],
                y[i,r["late_first_sample"]-1:r["late_last_sample"]]]
        mm=[];ww=[];ll=[]
        for v in pieces:
            check(len(v)>0 and np.isfinite(v).all(),"Invalid FIT prefix")
            mu=v.mean(0)
            mm.append(v.T@v/len(v))
            dv=v-mu
            ww.append(dv.T@dv/len(v))
            ll.append(np.outer(mu,mu))
        moments.append((mm[0]+mm[1])/2)
        within.append((ww[0]+ww[1])/2)
        locations.append((ll[0]+ll[1])/2)
    F=np.mean(moments,axis=0);V=np.mean(within,axis=0);J=np.mean(locations,axis=0)
    yn=features(normal_design_obj,des).reshape(-1,d)
    N=yn.T@yn/len(yn)
    H=(F-N);H=(H+H.T)/2
    check(np.allclose(F,V+J,rtol=1e-10,atol=1e-9),"FIT moment identity")
    check(np.isfinite(H).all(),"Nonfinite contrast")
    # Return constituent sufficient statistics, not physical covariance claims.
    return H,{"fault_second_moment":F,"normal_second_moment":N,
        "fault_within_window_covariance":V,"fault_endpoint_mean_outer":J,
        "contrast":H,"mean_early":a,"mean_late":b,
        "per_run_second_moment":np.stack(moments)},rows


def metric_list(des,variant):
    B=des["B"];d=len(B);t=float(np.trace(B))/d;W=des["whitener"]
    recipes=[("unweighted",np.eye(d)),("shrink50",.5*B+.5*t*np.eye(d)),
             ("shrink10",.9*B+.1*t*np.eye(d)),("raw",B)]
    ans=[]
    for name,T in recipes:
        rho=0.
        if variant=="ridge_1e4":
            check(W.shape==(d,d),"Ridge requires full residual dimension")
            Wi=np.linalg.inv(W)
            A=Wi.T@T@Wi;A=(A+A.T)/2
            rho=1e-4*float(np.trace(A))/d
            T=T+rho*(W.T@W)
        T=(T+T.T)/2
        check(np.linalg.eigvalsh(T)[0]>0,"Nonpositive normal metric")
        ans.append((name,T,rho))
    return ans


def learned_axes(H,des,variant):
    """Generalized Rayleigh extrema. No amplitude/selection/evaluation tuning."""
    axes=[];labels=[];diagnostics=[];proposals=[]
    G=des["G"]
    for name,T,rho in metric_list(des,variant):
        ev,U=np.linalg.eigh(T)
        W=U/np.sqrt(ev)[None,:]
        E=W.T@H@W;E=(E+E.T)/2
        lam,V=np.linalg.eigh(E)
        for order,j in [("max",-1),("min",0)]:
            raw=W@V[:,j]
            norm=np.sqrt(float(raw@G@raw))
            check(np.isfinite(norm) and norm>1e-12,"Zero amplitude axis normalization")
            c=canonical(raw)/np.sqrt(float(canonical(raw)@G@canonical(raw)))
            scale=max(float(np.linalg.norm(H,2)*np.linalg.norm(c)),
                      float(abs(lam[j])*np.linalg.norm(T,2)*np.linalg.norm(c)),1e-30)
            err=float(np.linalg.norm(H@c-lam[j]*T@c)/scale)
            check(err<1e-7,"Generalized eigensolver residual")
            gap=float(lam[-1]-lam[-2]) if order=="max" else float(lam[1]-lam[0])
            proposals.append((order,name,c))
            diagnostics.append({"recipe":name,"extreme":order,"generalized_eigenvalue":float(lam[j]),
                "next_eigenvalue_gap":gap,"eigengap_relative":gap/max(float(np.max(abs(lam))),1e-30),
                "generalized_residual_relative":err,"ridge_rho_original":rho,
                "actual_normal_variance":float(c@G@c),
                "moment_change":float(c@H@c),"normal_metric_denominator":float(c@T@c)})
    # Match historical ordering: four primary candidates, then four complements.
    for kind in ["max","min"]:
        for order,name,c in proposals:
            if order!=kind:continue
            duplicate=next((i for i,w in enumerate(axes)
                if min(np.max(abs(c-w)),np.max(abs(c+w)))<=1e-10),None)
            if duplicate is None:
                axes.append(c);labels.append([name+"_"+order])
            else:labels[duplicate].append(name+"_"+order)
    check(1<=len(axes)<=8,"Second-moment pool cap")
    return np.stack(axes),labels,diagnostics


def parameters(normal,des,old,new):
    uo=project_axes(normal,des,old);un=project_axes(normal,des,new)
    po=transform_parameters(uo);pn=transform_parameters(un)
    return {**{"mean_"+k:v for k,v in po.items()},**{"moment_"+k:v for k,v in pn.items()}}


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


def names(km,ka):
    base=[f"mean_A{i:02}" for i in range(km)]+[f"moment_A{i:02}" for i in range(ka)]
    return [b+"__"+family for family in FAMS for b in base]+["Q_CUSUM"]


