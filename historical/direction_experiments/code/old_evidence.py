"""S26: fixed-direction evidence ablation. Never fits/rotates a direction.
Primary signed/absolute/squared inputs all use two-sided CUSUM. Upper-sided
magnitude controls are separate and are not eligible for primary selection.
"""
from pathlib import Path
import json
import numpy as np
from core import (K,T,START,H,check,js,cs,sha,stamp,features,recurrence,
                  prefix_cusum,upper_cusum,event_rows,normal_rows,summarize,
                  numeric_records,event_category,subset)
FAMILIES=('signed_two_sided','absolute_two_sided','squared_two_sided',
          'absolute_upper','squared_upper')

def load_npz(p):
    with np.load(p,allow_pickle=False) as a:
        return {key:a[key] for key in a.files}

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

def scores(obj,design,axes,p):
    u=project_axes(obj,design,axes);signed,absolute,squared=transform_inputs(u,p)
    # Exact source recurrence for signed; identical recurrence for primary controls.
    d0=recurrence(signed);d1=recurrence(absolute);d2=recurrence(squared)
    def up(x):
        zero=np.zeros((len(x),1,x.shape[2]));prefix=np.concatenate((zero,np.cumsum(x-K,axis=1)),axis=1)
        return (prefix-np.minimum.accumulate(prefix,axis=1))[:,1:]
    qi=(np.log1p(obj['Q'])-float(design['q_mean']))/float(design['q_sd'])
    d=np.concatenate((d0,d1,d2,up(absolute),up(squared),upper_cusum(qi)[:,:,None]),axis=2)
    check(np.isfinite(d).all(),'Nonfinite statistic')
    return u,d

def candidate_names(k):
    return [f'A{j:02}__{f}' for f in FAMILIES for j in range(k)]+['Q_CUSUM']

def metadata(k,j):
    return ('Q_energy',-1) if j==5*k else (FAMILIES[j//k],j%k)

def fault_metrics(d,h,ids,fault):
    fields=None;arr=[]
    for j in range(d.shape[2]):
        rows=event_rows(d[:,:,j]>h[j],ids,fault,'old_development','candidate')
        fields,v=numeric_records(rows);arr.append(v)
    return fields,np.stack(arr)

def compare_source_records(observed,fields,expected,expected_fields):
    a=list(fields);b=list(map(str,expected_fields));count=0
    for f in b:
        if f=='last8h_recent120_points':continue  # not an S26 endpoint
        check(f in a,'Missing source field '+f)
        check(np.array_equal(observed[:,:,a.index(f)],expected[:,:,b.index(f)]),'Source integer parity: '+f)
        count+=observed.shape[0]*observed.shape[1]
    return count

def state(root,fault,tag,variant,reference_role="audit"):
    p=Path(root)/'inputs/frozen_models'/f'F{fault:02}'/tag/variant
    m=load_npz(p/'MODEL.npz');s=json.loads((p/'SELECTION.json').read_text())
    d=load_npz(Path(root)/'inputs/normal_design'/tag/(variant+'.npz'))
    # Do not decode stored audit outputs at the transformation/selection stage.
    role_keys={
        'fit':['signed_anchor_peaks','signed_inner_peaks','q_inner_peaks','selection_records','fault_fields'],
        'outer':['signed_outer_peaks','q_outer_peaks'],
        'audit':['fault_records','normal_records','fault_fields','normal_fields','source_selected_masks','source_normal_masks']}
    with np.load(p/'REFERENCE.npz',allow_pickle=False) as a:
        r={key:a[key] for key in role_keys[reference_role]}
    k=int(s['small_axis_count'])
    check(m['axes'].shape[0]>=k and k>=1,'No original candidate')
    return p,m,s,d,r,k

def fit_evidence(root,out,normal,fd,fault,tag,variant,split):
    """Only normal design/anchor/inner and fault selection numerical roles used."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    src,m,s,design,ref,k=state(root,fault,tag,variant,"fit");axes=m['axes'][:k].copy()
    j0=int(s['chosen']['selected_single'][0]);check(0<=j0<k,'Original winner outside single pool')
    nd=subset(normal,split['normal']['design']);u=project_axes(nd,design,axes);pars=transform_parameters(u)
    aobj=subset(normal,split['normal']['anchor']);_,da=scores(aobj,design,axes,pars)
    anchor_peaks=da.max(1);anchor=np.quantile(anchor_peaks,.9,axis=0,method='linear')
    check(np.all(anchor>0),'Zero anchor')
    io=subset(normal,split['normal']['inner_calibration']);_,di=scores(io,design,axes,pars);ip=di.max(1)
    qin=np.sort(ip/anchor,axis=0)[56];hin=anchor*qin
    so=subset(fd,split['fault']['selection']);_,ds=scores(so,design,axes,pars)
    fields,sr=fault_metrics(ds,hin,so['run_id'],fault)
    ei=list(fields).index('event_loss_samples');risks=sr[:,:,ei].mean(1)
    # Primary source controls must be recovered before any new choice is accepted.
    check(np.allclose(anchor_peaks[:,:k],ref['signed_anchor_peaks'],rtol=1e-8,atol=1e-8),'Source signed anchor mismatch')
    check(np.allclose(ip[:,:k],ref['signed_inner_peaks'],rtol=1e-8,atol=1e-8),'Source signed inner mismatch')
    check(np.allclose(ip[:,-1],ref['q_inner_peaks'],rtol=1e-8,atol=1e-8),'Source Q inner mismatch')
    parity=compare_source_records(sr[list(range(k))+[5*k]],fields,ref['selection_records'],ref['fault_fields'])
    def best(pool):
        low=float(np.min(risks[pool]));return next(int(j) for j in pool if risks[j]<=low+1e-12)
    check(best(list(range(k)))==j0,'Source selected direction changed')
    chosen={
        'fixed_axis_signed_two_sided':j0,
        'fixed_axis_absolute_two_sided':k+j0,
        'fixed_axis_squared_two_sided':2*k+j0,
        'fixed_axis_absolute_upper':3*k+j0,
        'fixed_axis_squared_upper':4*k+j0,
        'Q_CUSUM':5*k,
        'choose_transform_fixed_axis':best([j0,k+j0,2*k+j0]),
        'choose_three_families_fixed_pool':best(list(range(3*k))),
        'choose_pool_or_Q':best(list(range(3*k))+[5*k]),
        'source_linear_or_Q':best([j0,5*k])}
    names=candidate_names(k)
    cs(out/'INNER_SCORES.csv',[{'candidate':names[j],'family':metadata(k,j)[0],'axis':metadata(k,j)[1],
         'primary_eligible':int(j<3*k),'inner_threshold':float(hin[j]),'selection_n':len(so['run_id']),
         'event_loss_min':float(risks[j]*3),'prefault_alarm_runs':int(sr[j,:,list(fields).index('prefault_alarm')].sum()),
         'success_runs':int(sr[j,:,list(fields).index('clean_detection')].sum())} for j in range(len(names))])
    np.savez_compressed(out/'TRANSFORM_STATE.npz',**pars,anchors=anchor,inner_q=qin,inner_thresholds=hin)
    np.savez_compressed(out/'INNER_RECORDS.npz',anchor_peaks=anchor_peaks,inner_peaks=ip,selection=sr,fields=np.array(fields),run_id=so['run_id'])
    choice={'fault':fault,'split_id':tag,'variant':variant,'original_axis':j0,'axis_count':k,
        'choices':chosen,'candidate_names':names,'axis_source_sha256':sha(src/'MODEL.npz'),
        'state_sha256':sha(out/'TRANSFORM_STATE.npz'),'selection_integer_fields_verified':parity,
        'selection_locked_utc':stamp(),'new_audit_used':False,'no_axis_refit':True}
    js(out/'CHOICES.json',choice)
    return choice

def outer_calibrate(root,out,normal,fault,tag,variant,split):
    out=Path(out);src,m,s,design,ref,k=state(root,fault,tag,variant,"outer")
    pars=load_npz(out/'TRANSFORM_STATE.npz');obj=subset(normal,split['normal']['outer_calibration'])
    _,d=scores(obj,design,m['axes'][:k],pars);p=d.max(1)
    q=np.sort(p/pars['anchors'],axis=0)[56];h=pars['anchors']*q
    check(np.allclose(p[:,:k],ref['signed_outer_peaks'],rtol=1e-8,atol=1e-8),'Source signed outer mismatch')
    check(np.allclose(p[:,-1],ref['q_outer_peaks'],rtol=1e-8,atol=1e-8),'Source Q outer mismatch')
    np.savez_compressed(out/'OUTER_RECORDS.npz',peaks=p,q=q,thresholds=h,run_id=obj['run_id'])
    cs(out/'THRESHOLDS.csv',[{'candidate':n,'normal_anchor':float(pars['anchors'][j]),'inner_threshold':float(pars['inner_thresholds'][j]),
         'outer_threshold':float(h[j]),'rank':57,'calibration_n':59} for j,n in enumerate(candidate_names(k))])
    js(out/'OUTER_LOCK.json',{'utc':stamp(),'sha256':sha(out/'OUTER_RECORDS.npz'),'audit_used':False})

def moment_diagnostics(out,u,z,q,normal_u,normal_z,normal_q,j0):
    """Postselection descriptive decomposition; no new direction or policy selection."""
    windows=[(1,160),(161,320),(321,480),(481,640),(641,800),(801,960),(161,960)]
    k=u.shape[2];rows=[];selected=[];qrows=[]
    def moments(v):
        rm=v.mean(1);within=v.var(1,ddof=0).mean(0);mean=rm.mean(0);between=rm.var(0,ddof=0)
        second=(v*v).mean((0,1));err=float(np.max(abs(second-(mean*mean+within+between))))
        check(np.allclose(second,mean*mean+within+between,rtol=1e-9,atol=1e-9),'Scalar moment decomposition')
        return mean,within,between,second,np.abs(v).mean((0,1)),err
    nm=moments(normal_u);maxerr=nm[-1]
    for lo,hi in windows:
        v=u[:,lo-1:hi];res=moments(v);maxerr=max(maxerr,res[-1])
        for j in range(k):
            rows.append({'first_sample':lo,'last_sample':hi,'axis':j,'original_winner':int(j==j0),
                'mean_u':float(res[0][j]),'within_run_variance':float(res[1][j]),'between_run_mean_variance':float(res[2][j]),
                'mean_u_squared':float(res[3][j]),'mean_absolute_u':float(res[4][j]),
                'delta_second_moment':float(res[3][j]-nm[3][j]),
                'delta_squared_pooled_mean':float(res[0][j]**2-nm[0][j]**2),
                'delta_within_run_variance':float(res[1][j]-nm[1][j]),
                'delta_between_run_mean_variance':float(res[2][j]-nm[2][j])})
        vv=v[:,:,j0];selected.append(np.stack((vv.mean(1),vv.var(1),np.mean(vv*vv,1),np.mean(abs(vv),1)),axis=1))
        def energy_parts(x,energy):
            rm=x.mean(1);globalmean=rm.mean(0)
            a=float(globalmean@globalmean);b=float(np.mean(np.sum((rm-globalmean)**2,axis=1)))
            c=float(np.mean(np.sum((x-rm[:,None])**2,axis=2)));e=float(energy.mean())
            check(np.isclose(e,a+b+c,rtol=1e-7,atol=1e-7),'Q moment decomposition')
            return np.array([e,a,b,c])
        qp=energy_parts(z[:,lo-1:hi],q[:,lo-1:hi]);np_=energy_parts(normal_z,normal_q)
        qrows.append({'first_sample':lo,'last_sample':hi,'mean_Q':float(qp[0]),'pooled_mean_norm_squared':float(qp[1]),'between_run_mean_norm_variance':float(qp[2]),'within_run_trace_variance':float(qp[3]),'delta_Q':float(qp[0]-np_[0]),'delta_mean_norm_squared':float(qp[1]-np_[1]),'delta_between_run_trace':float(qp[2]-np_[2]),'delta_within_run_trace':float(qp[3]-np_[3])})
    cs(out/'AXIS_MOMENT_DECOMPOSITION.csv',rows);cs(out/'Q_MOMENT_DECOMPOSITION.csv',qrows)
    np.savez_compressed(out/'WINNING_AXIS_PER_RUN_MOMENTS.npz',windows=np.array(windows),moments=np.stack(selected),fields=np.array(['mean','variance','second_moment','absolute_mean']))
    return maxerr

def evaluate(root,out,normal,fd,fault,tag,variant,split):
    out=Path(out);src,m,s,des,ref,k=state(root,fault,tag,variant)
    ch=json.loads((out/'CHOICES.json').read_text());p=load_npz(out/'TRANSFORM_STATE.npz');oc=load_npz(out/'OUTER_RECORDS.npz');h=oc['thresholds']
    fo=subset(fd,split['fault']['audit']);no=subset(normal,split['normal']['audit'])
    uf,df=scores(fo,des,m['axes'][:k],p);un,dn=scores(no,des,m['axes'][:k],p)
    fields,fr=fault_metrics(df,h,fo['run_id'],fault);nf=None;normal_rec=[];allrows=[]
    for j,name in enumerate(ch['candidate_names']):
        nr=normal_rows(dn[:,:,j]>h[j],no['run_id'],fault,name);nf,a=numeric_records(nr);normal_rec.append(a)
        r=event_rows(df[:,:,j]>h[j],fo['run_id'],fault,'audit_old_development',name)
        allrows.append({'candidate_index':j,'family':metadata(k,j)[0],'axis':metadata(k,j)[1],**summarize(r,nr)})
    nr=np.stack(normal_rec)
    indices=list(range(k))+[5*k]
    parity=compare_source_records(fr[indices],fields,ref['fault_records'],ref['fault_fields'])
    for field in map(str,ref['normal_fields']):
        check(np.array_equal(nr[indices,:,list(nf).index(field)],ref['normal_records'][:,:,list(ref['normal_fields']).index(field)]),'Source normal integer mismatch '+field)
        parity+=len(indices)*len(no['run_id'])
    j0=ch['original_axis'];sources=[j0,5*k]
    check(np.array_equal(np.packbits((df[:,:,sources]>h[sources]).transpose(2,0,1),axis=2),ref['source_selected_masks']),'Source selected fault mask mismatch')
    check(np.array_equal(np.packbits((dn[:,:,sources]>h[sources]).transpose(2,0,1),axis=2),ref['source_normal_masks']),'Source selected normal mask mismatch')
    # Independent prefix formula check, using the same observed inputs.
    prefixerr=0.;booleans=0
    for g,v in enumerate(transform_inputs(uf,p)):
        alt=prefix_cusum(v);target=df[:,:,g*k:(g+1)*k]
        prefixerr=max(prefixerr,float(np.max(abs(alt-target))))
        check(np.allclose(alt,target,rtol=1e-9,atol=1e-8),'Prefix formula discrepancy')
        check(np.array_equal(alt>h[g*k:(g+1)*k],target>h[g*k:(g+1)*k]),'Prefix decision discrepancy')
        booleans+=target.size
    main=[];event=[];fmask=[];nmask=[];chosen_indices=[]
    for method,j in ch['choices'].items():
        main.append({'method':method,'chosen_candidate':ch['candidate_names'][j],**allrows[j]})
        chosen_indices.append(j);fm=df[:,:,j]>h[j];nm=dn[:,:,j]>h[j];fmask.append(np.packbits(fm,axis=1));nmask.append(np.packbits(nm,axis=1))
        rs=event_rows(fm,fo['run_id'],fault,'audit_old_development',method)
        for t,r in enumerate(rs):
            b=int(fo['record_suffix_start'][t]);tail=int(fo['record_suffix_samples'][t]);marked=tail>=160
            event.append({'method':method,**r,'record_marker_start':b if marked else -1,'tail_samples':tail,'category':event_category(r,b,tail),
                 'pre_marker_above_points':int(fm[t,160:b-1].sum()) if marked else -1,'pre_marker_exposure':b-161 if marked else -1,
                 'tail_above_points':int(fm[t,b-1:].sum()) if marked else -1,'tail_exposure':tail if marked else -1})
    np.savez_compressed(out/'EVALUATION_RECORDS.npz',fault=fr,normal=nr,fault_fields=np.array(fields),normal_fields=np.array(nf),fault_run_id=fo['run_id'],normal_run_id=no['run_id'],candidate_names=np.array(ch['candidate_names']))
    np.savez_compressed(out/'SELECTED_MASKS.npz',methods=np.array(list(ch['choices'])),candidate_indices=np.array(chosen_indices),fault=np.stack(fmask),normal=np.stack(nmask),fault_run_id=fo['run_id'],normal_run_id=no['run_id'])
    cs(out/'ALL_CANDIDATE_RESULTS.csv',allrows);cs(out/'MAIN_RESULTS.csv',main);cs(out/'EVENT_PHASES.csv',event)
    nd=subset(normal,split['normal']['design']);nu=project_axes(nd,des,m['axes'][:k])
    merr=moment_diagnostics(out,uf,fo['z'],fo['Q'],nu,nd['z'],nd['Q'],j0)
    # Same-audit/inner-threshold effect is distinguished from final calibration.
    transfer=[]
    for method,j in ch['choices'].items():
        r=event_rows(df[:,:,j]>p['inner_thresholds'][j],fo['run_id'],fault,'audit_inner',method)
        transfer.append({'method':method,'audit_inner_loss_min':float(np.mean([x['event_loss_samples'] for x in r])*3),'audit_outer_loss_min':main[list(ch['choices']).index(method)]['event_loss_min']})
    cs(out/'CALIBRATION_TRANSFER.csv',transfer)
    audit={'source_integer_fields_verified':parity,'prefix_alarm_booleans_verified':booleans,'prefix_max_error':prefixerr,'moment_identity_max_error':merr,'no_axis_refit':True,'evaluation_utc':stamp()}
    js(out/'AUDIT.json',audit)
    return main,audit
