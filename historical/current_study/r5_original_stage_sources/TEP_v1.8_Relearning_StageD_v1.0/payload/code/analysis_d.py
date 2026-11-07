"""Frozen-state scoring and paired shared-panel summaries. Does not fit models."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import hashlib,csv
import numpy as np
import science, inherited
from storage import strict_json,sha,write_json,write_csv,verify_tree,seal_tree
from inputs_d import PAYLOAD,METHODS,FAULTS,require
from statistics_impl import clopper_pearson

CP=lru_cache(maxsize=1024)(clopper_pearson)
EV_KEYS=('prefault_alarm','detected','clean_detection','first_delay_samples_penalized','event_loss_samples','occupancy_points','postalarm_below_points')

def independent_events(mask):
    """Simple run/monitor integer audit, separate from vectorized science.events."""
    require(mask.ndim==3 and mask.shape[1]==960,'Mask dimensions')
    n,_,k=mask.shape;out={name:np.zeros((n,k),dtype=np.int64) for name in EV_KEYS}
    for i in range(n):
        for j in range(k):
            pre=bool(np.any(mask[i,:160,j]));ix=np.flatnonzero(mask[i,160:,j]);det=bool(len(ix));first=int(ix[0]) if det else 800
            out['prefault_alarm'][i,j]=int(pre);out['detected'][i,j]=int(det);out['clean_detection'][i,j]=int(det and not pre)
            out['first_delay_samples_penalized'][i,j]=first;out['event_loss_samples'][i,j]=first if det and not pre else 800
            out['occupancy_points'][i,j]=len(ix);out['postalarm_below_points'][i,j]=int(np.count_nonzero(~mask[i,160+first:,j]))
    return out

def pack(mask):
    require(mask.dtype==bool and mask.shape==(250,960,9),'Frozen mask shape')
    return np.packbits(mask,axis=1,bitorder='little')

def unpack(bits):
    require(bits.dtype==np.uint8 and bits.shape==(250,120,9),'Packed mask shape/type')
    return np.unpackbits(bits,axis=1,count=960,bitorder='little').astype(bool)

def endpoints(rep,fault,methods,ev,nmask,thresholds,idx):
    out=[]
    for j,m in enumerate(methods):
        rates={'normal_FAR':nmask[:,:,j].any(1),'prefault_alarm':ev['prefault_alarm'][:,j].astype(bool),
               'clean_detection':ev['clean_detection'][:,j].astype(bool),'postfault_detection':ev['detected'][:,j].astype(bool)}
        for horizon in (60,120,480,2400):rates[f'clean_by_{horizon}min']=rates['clean_detection']&(ev['first_delay_samples_penalized'][:,j]*3<=horizon)
        row={'replication':rep,'fault':fault,'method':m,'fault_n':250,'normal_n':250,'candidate_index':int(idx[j]),'frozen_threshold':float(thresholds[j]),'k':.05,
             'loss_minutes':float(ev['event_loss_samples'][:,j].mean()*3),'occupancy_pct':float(ev['occupancy_points'][:,j].mean()/8),
             'postalarm_below_points':int(ev['postalarm_below_points'][:,j].sum())}
        for name,a in rates.items():
            k=int(a.sum());lo,hi=CP(k,250)
            row.update({name+'_count':k,name+'_pct':k/250*100,name+'_CP95_low_pct':100*lo,name+'_CP95_high_pct':100*hi})
        out.append(row)
    return out

def run_event_rows(rep,fault,methods,ev):
    return [{'replication':rep,'fault':fault,'method':m,'run':i+1,**{k:int(ev[k][i,j]) for k in EV_KEYS},'loss_minutes':int(ev['event_loss_samples'][i,j])*3} for j,m in enumerate(methods) for i in range(250)]

def validate_pair(folder,expected_rep=None,expected_fault=None,do_integer_audit=True):
    verify_tree(folder)
    info=strict_json(folder/'SCORING.json');rep=info['replication'];fault=info['fault']
    require(rep in range(1,6) and fault in FAULTS,'Unknown scoring task')
    if expected_rep is not None:require(rep==expected_rep,'Wrong scored repetition')
    if expected_fault is not None:require(fault==expected_fault,'Wrong scored fault')
    require(tuple(info['methods'])==METHODS and info['normal_n']==info['fault_n']==250,'Scored method/denominator mismatch')
    z=science.load_npz(folder/'SCORED_RECORDS.npz');require(np.array_equal(z['run_id'],np.arange(1,251)),'Saved run IDs')
    require(tuple(z['methods'].tolist())==METHODS,'Saved method order')
    fm=unpack(z['fault_alarm_bits']);nm=unpack(z['normal_alarm_bits'])
    ev={k:z['fault_'+k] for k in EV_KEYS};ne={k:z['normal_'+k] for k in EV_KEYS}
    if do_integer_audit:
        for prefix,mask,stored in [('fault',fm,ev),('normal',nm,ne)]:
            calc=independent_events(mask)
            for k in EV_KEYS:require(stored[k].dtype.kind in 'iu' and np.array_equal(stored[k],calc[k]),'Saved alarm/event mismatch '+prefix+' '+k)
    require(np.array_equal(z['normal_alarm_any'],nm.any(1).astype(np.int64)),'Normal FAR vector differs')
    return info,z,ev,nm

def score_pair(rep,fault,normal,obj,modeldir,out,bindings):
    out.mkdir(exist_ok=False)
    # These are the exact Stage B read-only scoring entry points. No selection/calibration call.
    methods,ev,fmask=science.selected_outcomes(obj,modeldir)
    nmeth,nev,nmask=science.selected_outcomes(normal,modeldir)
    require(tuple(methods)==tuple(nmeth)==METHODS,'Frozen method order changed')
    for mask,e in ((fmask,ev),(nmask,nev)):
        audit=independent_events(mask)
        for k in EV_KEYS:require(np.array_equal(e[k],audit[k]),'Independent integer endpoint check failed: '+k)
    sel=strict_json(modeldir/'SELECTION.json');idx=np.array([sel['choices'][m] for m in methods],dtype=np.int64)
    thresholds=science.load_npz(modeldir/'OUTER.npz')['thresholds'][idx]
    witnesses={'normal':science.serial_scalar_check(normal,modeldir,limit_runs=2),'fault':science.serial_scalar_check(obj,modeldir,limit_runs=2),
               'scope':'Two fixed first-run dense scalar-recursion witnesses per role. They do not alter any score or alarm; Q retains the original prefix implementation.'}
    science.arrays(out/'SCORED_RECORDS.npz',run_id=np.arange(1,251,dtype=np.int64),methods=np.array(methods),candidate_index=idx,thresholds=thresholds,
                   fault_alarm_bits=pack(fmask),normal_alarm_bits=pack(nmask),normal_alarm_any=nmask.any(1).astype(np.int64),
                   **{'fault_'+k:v for k,v in ev.items()},**{'normal_'+k:v for k,v in nev.items()})
    write_json(out/'SCORING.json',{'replication':rep,'fault':fault,'methods':methods,'normal_n':250,'fault_n':250,'model_source_relative':f'models/rep{rep:02d}/F{fault:02d}',
       'model_input_sha256':{n:sha(modeldir/n) for n in ('DESIGN.npz','MODEL.npz','PARAMETERS.npz','SELECTION.json','OUTER.npz')},
       'frozen_bindings':bindings,'fitting':False,'recalibration':False,'strict_alarm_rule':'>','alarm_latched':False,'reset_at_onset':False,'all_runs_retained':True,
       'raw_to_score_executed_locally':True,'independent_integer_event_fields_checked':2*250*9*len(EV_KEYS)})
    write_json(out/'SCALAR_RECURRENCE_CHECK.json',witnesses)
    write_csv(out/'ENDPOINTS.csv',endpoints(rep,fault,methods,ev,nmask,thresholds,idx))
    write_csv(out/'PER_FAULT_RUN_EVENTS.csv',run_event_rows(rep,fault,methods,ev))
    seal_tree(out);validate_pair(out,rep,fault,False)

def rng_for(spec,role):
    require(role in spec['rng_roles'],'Unregistered bootstrap role')
    seed=int.from_bytes(hashlib.sha256((spec['rng_namespace']+role).encode('ascii')).digest()[:16],'big')
    return np.random.Generator(np.random.PCG64(seed))

def paired_bootstrap(vectors,spec,role):
    v=np.asarray(vectors,dtype=np.float64)
    require(v.ndim==2 and len(v)==250 and np.isfinite(v).all(),'Paired bootstrap expects 250 shared IDs')
    count=spec['bootstrap_replicates'];require(count==20000,'Frozen 20000 draws required')
    rng=rng_for(spec,role);ans=np.empty((count,v.shape[1]));dig=hashlib.sha256()
    for start in range(0,count,100):
        ix=rng.integers(0,250,size=(min(100,count-start),250),dtype=np.int64);dig.update(ix.astype('<i8').tobytes())
        ans[start:start+len(ix)]=v[ix].mean(1)
    return ans,dig.hexdigest()

def ci(draws,qs):return [float(x) for x in np.quantile(draws,qs,method='linear')]

def aggregate(evaluated,out,spec,contrasts):
    """Reconstruct all endpoint and contrast outputs from saved masks. Never loads raw trajectories."""
    out.mkdir(exist_ok=False);allrows=[];allrun=[];curves=[];by={};allfar=[];farlabels=[]
    for rep in range(1,6):
        qref=None
        for fault in FAULTS:
            info,z,ev,nm=validate_pair(evaluated/f'rep{rep:02d}'/f'F{fault:02d}',rep,fault)
            methods=list(METHODS);by[rep,fault]=(ev,nm)
            q=nm[:,:,methods.index('Q_CUSUM')]
            if qref is None:qref=q.copy()
            else:require(np.array_equal(q,qref),'Within-repetition Q normal alarm paths differ across faults')
            allrows.extend(endpoints(rep,fault,methods,ev,nm,z['thresholds'],z['candidate_index']))
            allrun.extend(run_event_rows(rep,fault,methods,ev))
            for j,m in enumerate(methods):
                clean=ev['clean_detection'][:,j].astype(bool);delay=ev['first_delay_samples_penalized'][:,j]*3
                hist=np.bincount(ev['first_delay_samples_penalized'][clean,j],minlength=801);cumulative=np.cumsum(hist)
                for t in range(801):curves.append({'replication':rep,'fault':fault,'method':m,'minutes':t*3,'clean_count':int(cumulative[t]),'denominator':250})
                allfar.append(nm[:,:,j].any(1).astype(float)*100);farlabels.append([rep,fault,m])
    require(len(allrows)==135 and len(allrun)==33750,'Incomplete full report')
    write_csv(out/'ALL_135_ENDPOINTS.csv',allrows);write_csv(out/'ALL_FAULT_RUN_EVENTS.csv',allrun)
    write_csv(out/'CLEAN_DETECTION_CURVES.csv',curves)
    # All 135 normal monitors share ONE schedule. Do not multiply the normal sample size.
    normalboot,normaldigest=paired_bootstrap(np.stack(allfar,axis=1),spec,'normal_shared')
    per=[];primary=[];drawdata={'normal_all135_means_pp':normalboot};vdata={};schedules={'normal_shared':normaldigest}
    for c in contrasts:
        fault=c['fault'];a=METHODS.index(c['A']);b=METHODS.index(c['B']);dif=[];nd=[]
        for rep in range(1,6):
            ev,nm=by[rep,fault];dif.append((ev['event_loss_samples'][:,a]-ev['event_loss_samples'][:,b]).astype(float)*3)
            nd.append((nm[:,:,a].any(1).astype(float)-nm[:,:,b].any(1).astype(float))*100)
        d=np.stack(dif,axis=1);nv=np.stack(nd,axis=1);meanvector=d.mean(1)
        draws,digest=paired_bootstrap(np.column_stack([d,meanvector]),spec,f'fault_{fault:02d}');schedules[f'fault_{fault:02d}']=digest
        # Arithmetic difference of normal means under identical identity draws is a paired FAR difference.
        normaldiff=np.stack([normalboot[:,farlabels.index([rep,fault,c['A']])]-normalboot[:,farlabels.index([rep,fault,c['B']])] for rep in range(1,6)],axis=1)
        drawdata[c['id']+'_loss_per_rep_and_mean']=draws;drawdata[c['id']+'_normal_FAR_per_rep']=normaldiff
        vdata[c['id']+'_loss_difference_minutes']=d;vdata[c['id']+'_normal_FAR_difference_pp']=nv
        for k,rep in enumerate(range(1,6)):
            lo,hi=ci(draws[:,k],spec['marginal_interval_quantiles']);nlo,nhi=ci(normaldiff[:,k],spec['marginal_interval_quantiles'])
            per.append({'contrast':c['id'],'replication':rep,'fault':fault,'A':c['A'],'B':c['B'],'predicted_sign':c['predicted_sign'],'shared_fault_n':250,
              'mean_loss_difference_minutes':float(d[:,k].mean()),'marginal95_low':lo,'marginal95_high':hi,'empirically_constant_contrast':bool(np.all(d[:,k]==d[0,k])),
              'normal_FAR_difference_pp':float(nv[:,k].mean()),'normal_marginal95_low_pp':nlo,'normal_marginal95_high_pp':nhi,'simultaneous_over_15':False})
        lo,hi=ci(draws[:,5],spec['marginal_interval_quantiles']);alo,ahi=ci(draws[:,5],spec['primary_interval_quantiles']);nlo,nhi=ci(normaldiff.mean(1),spec['marginal_interval_quantiles'])
        deg=bool(np.all(meanvector==meanvector[0]));repmeans=d.mean(0)
        support=(alo>0 if c['predicted_sign']=='positive' else ahi<0)
        opposite=(ahi<0 if c['predicted_sign']=='positive' else alo>0)
        status='EMPIRICAL_DEGENERATE_DESCRIPTIVE_ONLY' if deg else ('SUPPORTS_PREDICTED_FINITE_FIVE_STATE_MEAN' if support else ('OPPOSITE_FINITE_FIVE_STATE_MEAN' if opposite else 'INCONCLUSIVE_FINITE_FIVE_STATE_MEAN'))
        primary.append({'contrast':c['id'],'fault':fault,'A':c['A'],'B':c['B'],'predicted_sign':c['predicted_sign'],'fits':5,'shared_fault_n':250,'shared_normal_n':250,
          'mean_loss_difference_minutes':float(meanvector.mean()),'marginal95_low':lo,'marginal95_high':hi,'bonferroni98_333_low':alo,'bonferroni98_333_high':ahi,
          'fit_median':float(np.median(repmeans)),'fit_min':float(repmeans.min()),'fit_max':float(repmeans.max()),'negative_fit_count':int((repmeans<0).sum()),'zero_fit_count':int((repmeans==0).sum()),'positive_fit_count':int((repmeans>0).sum()),
          'normal_FAR_difference_pp':float(nv.mean()),'normal_marginal95_low_pp':nlo,'normal_marginal95_high_pp':nhi,'status':status,'includes_population_training_uncertainty':False,'equal_true_FAR_claimed':False})
    write_csv(out/'ALL_15_PER_FIT_CONTRASTS.csv',per);write_csv(out/'PRIMARY_3_FINITE_FIVE_FIT_CONTRASTS.csv',primary)
    science.arrays(out/'BOOTSTRAP_MEANS.npz',**drawdata);science.arrays(out/'PAIRED_RUN_VECTORS.npz',**vdata)
    dispersion=[]
    for fault in FAULTS:
        for m in METHODS:
            rows=[r for r in allrows if r['fault']==fault and r['method']==m]
            r={'fault':fault,'method':m,'fits':5,'shared_fault_n':250,'shared_normal_n':250}
            for name in ('loss_minutes','normal_FAR_pct','prefault_alarm_pct','clean_detection_pct','clean_by_60min_pct','clean_by_120min_pct','clean_by_480min_pct','occupancy_pct'):
                v=np.array([x[name] for x in rows]);r.update({name+'_mean':float(v.mean()),name+'_median':float(np.median(v)),name+'_min':float(v.min()),name+'_max':float(v.max())})
            dispersion.append(r)
    write_csv(out/'ALL_27_FIT_DISPERSION.csv',dispersion)
    write_json(out/'INFERENCE_METADATA.json',{'specification':spec,'schedules_sha256_little_endian_int64':schedules,'mean_bootstrap_averages_per_run_first':True,
      'normal_pairing_shared_across_all135':True,'rate_intervals':'Conditional two-sided Clopper-Pearson per individual fit. No binomial CI for a pooled or averaged five-fit rate.',
      'Q_normal_paths_equal_across_faults_within_each_rep':True,'raw_arrays_read_by_aggregation':False,'field_reconstruction':'All saved alarm/event integers recomputed independently before aggregation.',
      'figure_curves':'Empirical clean-detection counts, denominator250 per fit; no simultaneous confidence band.'})
    seal_tree(out);verify_tree(out)
    return primary
