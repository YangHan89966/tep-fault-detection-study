"""S28: candidate learning from normal-centred second-moment contrasts.
Same data roles and normal metrics as the fixed mean-direction controls.
No simulator, no old final process access, no policy tuning by audit outcomes.
"""
from pathlib import Path
import json
import numpy as np
from core import (K,check,js,cs,sha,stamp,canonical,features,subset,recurrence,
    prefix_cusum,upper_cusum,event_rows,normal_rows,summarize,numeric_records,event_category)
from old_evidence import load_npz,transform_parameters,transform_inputs,project_axes,fault_metrics
from fit_windows import endpoint_means
from numeric_audit import audit_prefix

FAMS=("signed","absolute","squared")

def source(root,fault,tag,variant):
    p=Path(root)/"inputs/frozen_models"/f"F{fault:02}"/tag/variant
    model=load_npz(p/"MODEL.npz")
    info=json.loads((p/"SELECTION.json").read_text())
    des=load_npz(Path(root)/"inputs/normal_design"/tag/(variant+".npz"))
    ref=Path(root)/"inputs/s26_reference"/f"F{fault:02}"/tag/variant
    return p,model,info,des,ref

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
            check(W.shape==(16,16),"Ridge requires original full16")
            Wi=np.linalg.inv(W)
            A=Wi.T@T@Wi;A=(A+A.T)/2
            rho=1e-4*float(np.trace(A))/16
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

def ref_indices(km,ka):
    k=km+ka
    return np.array([g*k+j for g in range(3) for j in range(km)]+[3*k]),np.array(list(range(3*km))+[5*km])

def parity_array(a,fields,b,ref_fields,indices,oldindices):
    n=0
    for f in map(str,ref_fields):
        if f=="last8h_recent120_points":continue
        ia=list(fields).index(f);ib=list(map(str,ref_fields)).index(f)
        check(np.array_equal(a[indices,:,ia],b[oldindices,:,ib]),"S26 reference mismatch: "+f)
        n+=len(indices)*a.shape[1]
    return n

def fit(root,out,normal,fd,fault,tag,variant,split):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    src,m,info,des,ref=source(root,fault,tag,variant)
    km=int(info["small_axis_count"]);old=m["axes"][:km].copy()
    fo=subset(fd,split["fault"]["fit"]);nd=subset(normal,split["normal"]["design"])
    H,mat,rows=fit_matrices(fo,nd,des,fault)
    check(np.allclose(mat["mean_early"],m["early_mean"],rtol=1e-9,atol=1e-9) and
          np.allclose(mat["mean_late"],m["late_mean"],rtol=1e-9,atol=1e-9),
          "Mean and second-moment FIT windows not matched")
    new,labels,diag=learned_axes(H,des,variant);ka=len(new);k=km+ka
    pars=parameters(nd,des,old,new)
    # Reference parameters are read for parity only, not to generate new coefficients.
    rp=load_npz(ref/"TRANSFORM_STATE.npz")
    for key in ("abs_mean","abs_sd","sq_mean","sq_sd"):
        check(np.allclose(pars["mean_"+key],rp[key],rtol=1e-10,atol=1e-10),"Mean transform ref "+key)
    an=inputs_and_scores(subset(normal,split["normal"]["anchor"]),des,old,new,pars).max(1)
    anchors=np.quantile(an,.9,axis=0,method="linear")
    check(np.all(anchors>0),"Nonpositive anchor")
    ip=inputs_and_scores(subset(normal,split["normal"]["inner_calibration"]),des,old,new,pars).max(1)
    iq=np.sort(ip/anchors,axis=0)[56];hin=anchors*iq
    sobj=subset(fd,split["fault"]["selection"])
    ds=inputs_and_scores(sobj,des,old,new,pars)
    fields,sr=fault_metrics(ds,hin,sobj["run_id"],fault)
    ri=load_npz(ref/"INNER_RECORDS.npz");inds,oi=ref_indices(km,ka)
    parity=parity_array(sr,fields,ri["selection"],ri["fields"],inds,oi)
    check(np.allclose(anchors[inds],rp["anchors"][oi],rtol=1e-9,atol=1e-8),"Mean anchors parity")
    check(np.allclose(hin[inds],rp["inner_thresholds"][oi],rtol=1e-9,atol=1e-8),"Mean inner threshold parity")
    risk=sr[:,:,list(fields).index("event_loss_samples")].mean(1)
    def best(pool):
        low=float(risk[pool].min())
        return next(int(j) for j in pool if risk[j]<=low+1e-12)
    mpg={fam:[g*k+j for j in range(km)] for g,fam in enumerate(FAMS)}
    apg={fam:[g*k+j for j in range(km,k)] for g,fam in enumerate(FAMS)}
    j0=int(info["chosen"]["selected_single"][0]);check(best(mpg["signed"])==j0,"Original linear choice parity")
    choices={"fixed_mean_signed":j0,"fixed_mean_squared":2*k+j0}
    for fam in FAMS:choices["mean_pool_"+fam]=best(mpg[fam])
    for fam in FAMS:choices["moment_pool_"+fam]=best(apg[fam])
    meanall=sum(mpg.values(),[]);ampall=sum(apg.values(),[])
    choices.update(mean_pool_three=best(meanall),moment_pool_three=best(ampall),
        mean_pool_three_or_Q=best(meanall+[3*k]),
        combined_three=best(meanall+ampall),
        combined_three_or_Q=best(meanall+ampall+[3*k]),Q_CUSUM=3*k)
    nam=names(km,ka)
    allrows=[]
    for j,n in enumerate(nam):
        allrows.append({"candidate_index":j,"candidate":n,"risk_samples":float(risk[j]),
           "event_loss_min":float(risk[j]*3),"prefault_alarms":int(sr[j,:,list(fields).index("prefault_alarm")].sum()),
           "clean_detections":int(sr[j,:,list(fields).index("clean_detection")].sum()),
           "inner_threshold":float(hin[j])})
    basis=load_npz(Path(root)/"inputs/RESIDUAL_BASIS.npz")["basis"]
    coeff=(basis@des["whitener"]@np.concatenate((old,new)).T).T
    gains=[{"axis":i,"source":"mean" if i<km else "second_moment",
            "coefficient_norm52":float(np.linalg.norm(c)),
            "normal_variance":float(np.concatenate((old,new))[i]@des["G"]@np.concatenate((old,new))[i])}
           for i,c in enumerate(coeff)]
    np.savez_compressed(out/"MODEL.npz",mean_axes=old,moment_axes=new,coefficients52=coeff,**mat)
    np.savez_compressed(out/"TRANSFORM_STATE.npz",**pars,anchors=anchors,inner_q=iq,inner_thresholds=hin)
    np.savez_compressed(out/"INNER_RECORDS.npz",selection=sr,fields=np.array(fields),anchor_peaks=an,
        inner_peaks=ip,selection_run_id=sobj["run_id"])
    cs(out/"FIT_WINDOWS.csv",rows);cs(out/"EIGEN_DIAGNOSTICS.csv",diag);cs(out/"INPUT_GAINS.csv",gains)
    cs(out/"INNER_SCORES.csv",allrows)
    ch={"fault":fault,"split_id":tag,"variant":variant,"mean_count":km,"moment_count":ka,
        "new_labels":labels,"original_axis":j0,"choices":choices,"candidate_names":nam,
        "source_model_sha256":sha(src/"MODEL.npz"),"model_sha256":sha(out/"MODEL.npz"),
        "transform_sha256":sha(out/"TRANSFORM_STATE.npz"),
        "source_selection_integer_fields_verified":parity,"utc":stamp(),"audit_used":False}
    js(out/"CHOICES.json",ch)
    return ch

def calibrate(root,out,normal,fault,tag,variant,split):
    out=Path(out);_,_,_,des,ref=source(root,fault,tag,variant)
    m=load_npz(out/"MODEL.npz");p=load_npz(out/"TRANSFORM_STATE.npz")
    km=len(m["mean_axes"]);ka=len(m["moment_axes"])
    obj=subset(normal,split["normal"]["outer_calibration"])
    peaks=inputs_and_scores(obj,des,m["mean_axes"],m["moment_axes"],p).max(1)
    q=np.sort(peaks/p["anchors"],axis=0)[56];h=p["anchors"]*q
    rp=load_npz(ref/"OUTER_RECORDS.npz");inds,oi=ref_indices(km,ka)
    check(np.allclose(h[inds],rp["thresholds"][oi],rtol=1e-9,atol=1e-8),"Mean outer threshold parity")
    np.savez_compressed(out/"OUTER_RECORDS.npz",peaks=peaks,q=q,thresholds=h,run_id=obj["run_id"])
    cs(out/"THRESHOLDS.csv",[{"candidate":n,"anchor":float(p["anchors"][j]),
        "inner_threshold":float(p["inner_thresholds"][j]),"outer_threshold":float(h[j]),"rank":57,"n":59}
        for j,n in enumerate(names(km,ka))])
    js(out/"OUTER_LOCK.json",{"utc":stamp(),"sha256":sha(out/"OUTER_RECORDS.npz"),"audit_used":False})

def evaluate(root,out,normal,fd,fault,tag,variant,split):
    out=Path(out);_,_,_,des,ref=source(root,fault,tag,variant)
    m=load_npz(out/"MODEL.npz");p=load_npz(out/"TRANSFORM_STATE.npz");oc=load_npz(out/"OUTER_RECORDS.npz")
    ch=json.loads((out/"CHOICES.json").read_text());h=oc["thresholds"]
    old,new=m["mean_axes"],m["moment_axes"];km,ka=len(old),len(new);k=km+ka
    fo=subset(fd,split["fault"]["audit"]);no=subset(normal,split["normal"]["audit"])
    df,xx=inputs_and_scores(fo,des,old,new,p,True);dn=inputs_and_scores(no,des,old,new,p)
    fields,fr=fault_metrics(df,h,fo["run_id"],fault)
    nr=[];summary=[];nam=ch["candidate_names"];audit=[]
    for g,x in enumerate(xx):
        report=audit_prefix(x,df[:,:,g*k:(g+1)*k],prefix_cusum(x),h[g*k:(g+1)*k],
            out=out/"numeric_witnesses",label=FAMS[g],run_ids=fo["run_id"])
        audit.append(report)
    for j,n in enumerate(nam):
        rr=normal_rows(dn[:,:,j]>h[j],no["run_id"],fault,n);nfields,ar=numeric_records(rr);nr.append(ar)
        er=event_rows(df[:,:,j]>h[j],fo["run_id"],fault,"audit",n)
        summary.append({"candidate_index":j,"candidate":n,**summarize(er,rr)})
    nr=np.stack(nr);rr=load_npz(ref/"EVALUATION_RECORDS.npz");inds,oi=ref_indices(km,ka)
    parity=parity_array(fr,fields,rr["fault"],rr["fault_fields"],inds,oi)
    parity+=parity_array(nr,nfields,rr["normal"],rr["normal_fields"],inds,oi)
    main=[];events=[];fmasks=[];nmasks=[];transfer=[]
    for method,j in ch["choices"].items():
        main.append({"method":method,**summary[j]})
        fm=df[:,:,j]>h[j];nm=dn[:,:,j]>h[j];fmasks.append(np.packbits(fm,axis=1));nmasks.append(np.packbits(nm,axis=1))
        er=event_rows(fm,fo["run_id"],fault,"audit",method)
        for i,r in enumerate(er):
            b=int(fo["record_suffix_start"][i]);n=int(fo["record_suffix_samples"][i])
            events.append({"method":method,**r,"record_marker_start":b if n>=160 else -1,
                "tail_samples":n,"category":event_category(r,b,n)})
        eri=event_rows(df[:,:,j]>p["inner_thresholds"][j],fo["run_id"],fault,"audit_inner",method)
        transfer.append({"method":method,"audit_inner_loss_min":float(np.mean([r["event_loss_samples"] for r in eri])*3),
             "audit_outer_loss_min":float(main[-1]["event_loss_min"])})
    np.savez_compressed(out/"EVALUATION_RECORDS.npz",fault=fr,normal=nr,fault_fields=np.array(fields),
         normal_fields=np.array(nfields),fault_run_id=fo["run_id"],normal_run_id=no["run_id"],candidate_names=np.array(nam))
    np.savez_compressed(out/"SELECTED_MASKS.npz",fault=np.stack(fmasks),normal=np.stack(nmasks),
         methods=np.array(list(ch["choices"])),indices=np.array(list(ch["choices"].values())),
         fault_run_id=fo["run_id"],normal_run_id=no["run_id"])
    cs(out/"MAIN_RESULTS.csv",main);cs(out/"ALL_CANDIDATE_RESULTS.csv",summary)
    cs(out/"EVENT_PHASES.csv",events);cs(out/"CALIBRATION_TRANSFER.csv",transfer)
    js(out/"AUDIT.json",{"source_mean_integer_fields_verified":parity,
         "prefix_decisions":sum(a["total_decisions"] for a in audit),
         "exactly_certified_disagreements":sum(a["exactly_certified_disagreements"] for a in audit),
         "whole800_evaluation_retained":True,"utc":stamp()})
    return main
