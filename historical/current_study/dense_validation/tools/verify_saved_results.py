#!/usr/bin/env python3
"""Reaggregate the published DV saved masks, events and intervals; never score raw data.
Derived from the archived author verifier. The publication-only wrapper changes paths
and provenance checks, not scientific formulas. Requires NumPy and SciPy.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import beta
import scipy

def sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def canonical(obj)->bytes: return (json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as h: return list(csv.DictReader(h))
def jread(p):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key '+k)
            d[k]=v
        return d
    return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=unique)
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify_manifest(root,filename):
    m=jread(root/filename);paths=set()
    for e in m['entries']:
        rel=Path(e['path']);assert not rel.is_absolute() and '..' not in rel.parts
        assert str(rel) not in paths;paths.add(str(rel));data=(root/rel).read_bytes()
        assert len(data)==e['bytes'] and sha(data)==e['sha256'],str(rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual==paths|{filename},(actual-paths,paths-actual)
    return len(paths)

def main():
    if sys.flags.optimize:
        raise RuntimeError('Run without -O: verification assertions must remain enabled')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();bundle=a.bundle.resolve();out=a.out.resolve()
    if out==bundle or bundle in out.parents:
        raise ValueError('Choose a NEW output directory outside the DV input bundle')
    out.mkdir(parents=True,exist_ok=False);(out/'analysis').mkdir()
    src=bundle/'provenance';data=bundle/'results'
    context=jread(src/'EXPORT_CONTEXT.json')
    report={'status':'STARTED','scope':'Saved-result reaggregation and file identities only; no raw-to-score replay, refitting, simulation or external scientific acceptance.',
            'time_utc':datetime.now(timezone.utc).isoformat(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    report['manifests']={'frozen_model_entries':verify_manifest(bundle/'frozen_models','FILES.json'),
                        'evaluation_entries':verify_manifest(data,'FILES.json')}
    original={r['path']:r for r in jread(src/'SOURCE_MANIFEST.json')['entries']}
    checked=0
    for row in readcsv(src/'SOURCE_EXPORT_INDEX.csv'):
        p=bundle/row['exported_path'];b=p.read_bytes()
        assert sha(b)==row['sha256'],str(p)
        if row['transformation']=='NONE_BYTE_IDENTICAL':
            assert sha(b)==original[row['original_release_path']]['sha256'],str(p)
            checked+=1
    report['unchanged_original_source_files_checked']=checked
    index=readcsv(src/'COMPLETED_INDEX.csv');assert len(index)==1915
    assert len({r['canonical_label'] for r in index})==1915
    indexby={r['canonical_label']:r for r in index}
    seeds=jread(src/'SEED_MANIFEST.json');assert len(seeds)==1915
    assert set(indexby)=={r['canonical_label'] for r in seeds}
    assert len({r['seed_u32'] for r in seeds})==1915
    for r in seeds:
        label=r['canonical_label'];digest=hashlib.sha256((label+'|counter='+str(r['collision_counter'])).encode('ascii')).hexdigest()
        derived=1+int.from_bytes(bytes.fromhex(digest)[:8],'big')%((1<<32)-1)
        assert r['derivation_sha256']==digest and r['seed_u32']==derived and r['effective_initial_state']==int(float(derived))%(1<<32)
        i=indexby[label]
        for k in ('stage','role'):assert i[k]==r[k]
        for k in ('fault','run'):assert int(i[k])==r[k]
    lines=(src/'EVENTS.jsonl').read_bytes();assert lines.endswith(b'\n');events=[];last='0'*64
    for seq,l in enumerate(lines.splitlines()):
        e=json.loads(l);h=e.pop('sha256');assert e['seq']==seq and e['prev']==last and sha(canonical(e))==h
        e['sha256']=h;events.append(e);last=h
    starts={};commits={};counts=Counter(e['event'] for e in events)
    for e in events:
        if e['event']=='ATTEMPT_START':
            lab=e['data']['label'];assert lab not in starts;starts[lab]=e
        if e['event'] in ('ATTEMPT_COMMIT','RECOVER_COMMIT'):
            lab=e['data']['label'];assert lab not in commits and lab in starts;assert starts[lab]['seq']<e['seq'];commits[lab]=e
    assert set(starts)==set(commits)==set(indexby)
    assert not any(('FAIL' in e['event'] or 'INTERRUPTED' in e['event']) for e in events)
    model_event=next(e for e in events if e['event']=='LEARN_LOCKED');eval_starts=[starts[k] for k,r in indexby.items() if r['stage']=='evaluation']
    assert all(e['seq']>model_event['seq'] for e in eval_starts)
    assert counts['EVALUATE_START']==counts['EVALUATE_COMPLETE']==1 and events[-1]['event']=='EVALUATE_COMPLETE'
    preevaluate=next(e for e in events if e['event']=='EVALUATE_START')
    assert all(e['seq']<preevaluate['seq'] for e in commits.values())
    model=jread(src/'MODEL_LOCK.json');evlock=jread(src/'EVALUATION_LOCK.json');freeze=jread(src/'FREEZE.json')
    assert sha((bundle/'frozen_models/FILES.json').read_bytes())==model['manifest_sha256']
    assert sha((data/'FILES.json').read_bytes())==evlock['manifest_sha256']==events[-1]['data']['manifest_sha256']
    assert sha((src/'MODEL_LOCK.json').read_bytes())==evlock['model_lock_sha256']==preevaluate['data']['model_lock_sha256']
    assert model==model_event['data']['lock']
    for k,f in [('release_manifest_sha256','SOURCE_MANIFEST.json'),('seed_manifest_sha256','SEED_MANIFEST.json')]:assert model[k]==sha((src/f).read_bytes())
    for k,f in [('seed_manifest_sha256','SEED_MANIFEST.json'),('seed_exclusion_sha256','SEED_EXCLUSION.json')]:assert freeze[k]==sha((src/f).read_bytes())
    assert freeze['protocol_sha256']==context['original_protocol_sha256']
    report['ledger']={'records':len(events),'event_counts':dict(counts),'hash_chain':'PASS','unmatched_starts':0,'all1000_test_starts_after_model_lock':True,'all1915_commits_before_evaluation':True,'tip':last}
    roles=Counter((r['stage'],r['role'],int(r['fault'])) for r in index)
    report['seed_checks']={'new_derivations_verified':1915,'new_distinct_effective_initializations':1915,'old9500_exclusion':'Local reported result retained; old rows not available here','exclusion_report':jread(src/'SEED_EXCLUSION.json')}
    report['completed_roles']=[{'stage':s,'role':r,'fault':f,'n':n} for (s,r,f),n in roles.items()]
    report['shutdowns_from_returned_index']=Counter((r['stage'],r['fault']) for r in index if r['shutdown_occurred']=='True').total()
    report['model_lock']={'sha256':sha((src/'MODEL_LOCK.json').read_bytes()),'frozen_utc':model['utc'],'configurations':model['configurations'],'matches_locked_manifest':True}
    saved_summary=readcsv(data/'ALL_METHOD_ENDPOINTS.csv');saved_contrasts=readcsv(data/'PRIMARY_CONTRASTS.csv')
    table={(int(r['fault']),r['method']):r for r in saved_summary}; assert len(table)==27
    source_events={(int(r['fault']),r['method'],int(r['run'])):r for r in readcsv(data/'PER_FAULT_RUN_EVENTS.csv')};assert len(source_events)==6750
    phase={(int(r['fault']),r['method'],int(r['run'])):r for r in readcsv(data/'RECORD_PHASES.csv')};assert len(phase)==6750
    protocol=jread(bundle/'protocol/STUDY_PROTOCOL.public.json')
    nbits=0;event_fields=0;csv_event_fields=0;summary_fields=0;interval_max_error=0.;bootstrap_elements=0;phase_cells=0
    recalculated=[];paired_records=[];curve=[];failures=[];cross_normal={};phase_summary=[];contract=[]
    for fault in (3,9,15):
        with np.load(data/f'F{fault:02d}_RECORDS.npz',allow_pickle=False) as d:
            methods=d['methods'].tolist();assert len(methods)==len(set(methods))==9
            for field in ['fault_run_ids','normal_run_ids']:assert np.array_equal(d[field],np.arange(1,251))
            fm=np.unpackbits(d['fault_masks'],axis=2).astype(bool);nm=np.unpackbits(d['normal_masks'],axis=2).astype(bool)
            assert fm.shape==nm.shape==(9,250,960);nbits+=fm.size+nm.size
            pre=fm[:,:,:160].any(2);post=fm[:,:,160:];det=post.any(2);first=np.full((9,250),800,dtype=np.int64)
            for j in range(9):
                for i in range(250):
                    above=np.flatnonzero(post[j,i]); first[j,i]=int(above[0]) if len(above) else 800
            clean=det&~pre;loss=np.where(clean,first,800);occ=post.sum(2);withdraw=800-first-occ
            ev={'prefault_alarm':pre.astype(int),'detected':det.astype(int),'clean_detection':clean.astype(int),'first_delay_samples_penalized':first,'event_loss_samples':loss,'occupancy_points':occ,'postalarm_below_points':withdraw}
            for k,v in ev.items():assert np.array_equal(d[k],v.T),(fault,k);event_fields+=v.size
            for j,m in enumerate(methods):
                saved=table[fault,m];normal_alarm=nm[j].any(1);cross_normal[fault,m]=normal_alarm
                row={'fault':fault,'method':m,'fault_n':250,'normal_n':250,'loss_minutes':float(loss[j].sum()*3/250),'occupancy_pct':float(occ[j].sum()*100/(250*800)),'postalarm_withdrawal_points':int(withdraw[j].sum())}
                for name,vec in {'normal_FAR':normal_alarm,'prefault_alarm':pre[j],'clean_detection':clean[j],'postfault_detection':det[j],**{f'clean_by_{h}min':clean[j]&(first[j]*3<=h) for h in (60,120,480)}}.items():
                    k=int(vec.sum());n=250;lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1));hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
                    row[name+'_count']=k;row[name+'_pct']=100*k/n;row[name+'_CP95_low_pct']=100*lo;row[name+'_CP95_high_pct']=100*hi
                for k,v in row.items():
                    if k=='method':assert saved[k]==v;continue
                    err=abs(float(saved[k])-v);assert err<1e-8,(fault,m,k,err);summary_fields+=1
                    if '_CP95_' in k:interval_max_error=max(interval_max_error,err)
                recalculated.append(row)
                for i in range(250):
                    sr=source_events[fault,m,i+1]
                    for k,v in ev.items():assert int(sr[k])==v[j,i];csv_event_fields+=1
                    pr=phase[fault,m,i+1];cut=960 if pr['native_shutdown']=='0' else min(960,max(0,int(pr['shutdown_sample'])-1))
                    vals={'observed_prefault_alarm':int(pre[j,i]),'first_postfault_alarm_before_shutdown':int(det[j,i] and 160+first[j,i]<cut),'postfault_above_before_shutdown':int(fm[j,i,160:cut].sum()),'postfault_above_after_shutdown':int(fm[j,i,max(160,cut):].sum())}
                    for k,v in vals.items():assert int(pr[k])==v;phase_cells+=1
                    assert (pr['native_shutdown']=='1')==(indexby[f'DENSE3915_VALIDATION_v1.0|role=fault_evaluation|fault={fault:02d}|run={i+1:04d}']['shutdown_occurred']=='True')
                    if not clean[j,i]:failures.append({'fault':fault,'method':m,'run':i+1,'prefault_alarm':int(pre[j,i]),'postfault_detected':int(det[j,i]),'loss_minutes':int(3*loss[j,i])})
                for delay in range(801):
                    curve.append({'fault':fault,'method':m,'minutes':delay*3,'clean_count':int((clean[j]&(first[j]*3<=delay*3)).sum()),'n':250,'clean_pct':float((clean[j]&(first[j]*3<=delay*3)).sum()*100/250)})
                ph=[phase[fault,m,i+1] for i in range(250)];shut=sum(int(r['native_shutdown']) for r in ph)
                phase_summary.append({'fault':fault,'method':m,'shutdown_runs':shut,'unmarked_runs':250-shut,'postfault_alarms_in_unmarked_runs':sum(int(r['first_postfault_alarm_before_shutdown']) for r in ph if r['native_shutdown']=='0'),'above_after_shutdown_points':sum(int(r['postfault_above_after_shutdown']) for r in ph),'long_constancy_suffix_runs':sum(int(r['common_constancy_suffix_n'])>=160 for r in ph)})
            c=next(x for x in protocol['primary_contrasts'] if x['fault']==fault);s=next(x for x in saved_contrasts if int(x['fault'])==fault)
            ia=methods.index(c['a']);ib=methods.index(c['b']);diff=3*(loss[ia]-loss[ib]);nd=(nm[ia].any(1).astype(int)-nm[ib].any(1).astype(int))*100
            label_prefix='DENSE3915_VALIDATION_v1.0|bootstrap|'
            def boot(vec,label):
                seed=int.from_bytes(hashlib.sha256((label_prefix+label).encode('ascii')).digest()[:8],'big');rng=np.random.Generator(np.random.PCG64(seed))
                # A different block size from the study code; same registered RNG sequence.
                vals=[]
                for start in range(0,20000,250):
                    ix=rng.integers(0,250,size=(min(250,20000-start),250),dtype=np.int64)
                    vals.append(vec[ix].sum(1,dtype=np.int64)/250.)
                return np.concatenate(vals)
            bs=boot(diff,f'{fault:02d}');bn=boot(nd,'normal')
            assert np.array_equal(bs,d['loss_contrast_bootstrap']);assert np.array_equal(bn,d['normal_contrast_bootstrap']);bootstrap_elements+=40000
            ci=np.quantile(bs,[.025,.975,1/120,119/120],method='linear');nci=np.quantile(bn,[.025,.975],method='linear')
            vals={'mean_loss_difference_minutes':float(diff.sum()/250),'marginal95_lower':float(ci[0]),'marginal95_upper':float(ci[1]),'bonferroni98_333_lower':float(ci[2]),'bonferroni98_333_upper':float(ci[3]),'normal_FAR_difference_pp':float(nd.sum()/250),'normal_FAR_difference95_lower_pp':float(nci[0]),'normal_FAR_difference95_upper_pp':float(nci[1])}
            for k,v in vals.items():assert abs(float(s[k])-v)<1e-9,(c['id'],k)
            expected=1 if c['expected_sign']=='positive' else -1
            support=ci[2]>0 if expected==1 else ci[3]<0
            opposite=ci[3]<0 if expected==1 else ci[2]>0
            status='SUPPORTS_PREDICTED_LOSS_DIRECTION' if support else ('OPPOSITE_LOSS_DIRECTION' if opposite else 'INCONCLUSIVE_LOSS_CONTRAST')
            assert status==s['status'] and not np.all(diff==diff[0])
            paired_records.append({'contrast':c['id'],'fault':fault,'a':c['a'],'b':c['b'],**vals,'status':status,'paired_lower_loss_runs':int((diff<0).sum()),'paired_equal_loss_runs':int((diff==0).sum()),'paired_higher_loss_runs':int((diff>0).sum()),'normal_a_only':int((nm[ia].any(1)&~nm[ib].any(1)).sum()),'normal_b_only':int((~nm[ia].any(1)&nm[ib].any(1)).sum()),'normal_both':int((nm[ia].any(1)&nm[ib].any(1)).sum())})
    assert np.array_equal(cross_normal[3,'Q_CUSUM'],cross_normal[9,'Q_CUSUM']) and np.array_equal(cross_normal[3,'Q_CUSUM'],cross_normal[15,'Q_CUSUM'])
    report['reaggregation']={'alarm_mask_bits_checked':nbits,'event_array_fields_from_masks':event_fields,'event_CSV_integer_fields':csv_event_fields,'endpoint_numeric_fields':summary_fields,'binomial_interval_max_abs_error_pp':interval_max_error,'fixed_bootstrap_draw_elements_equal':bootstrap_elements,'record_phase_numeric_cells':phase_cells,'same_Q_normal_outcomes_all_three_faults':True,'no_candidate_rescoring_or_refitting':True}
    report['main_contrasts']=paired_records
    report['limitations']=['Source masks are verified for internal consistency; raw 52-variable trajectories were not available here to recompute the monitor scores.','No independent check of original 9500 seed rows; local exclusion record preserved. Distinct initialization states do not establish non-overlapping or statistically independent PRNG substreams.','The 250 normal evaluation runs are shared across all monitors; no pooling as 750 or 6750 independent normal runs.','CIs condition on this frozen PCA and one learned state per method/fault, excluding full retraining uncertainty.','This is original computation verification, not an external independent scientific acceptance.','This command writes only its new output directory. It does not modify manuscripts or research states.']
    writecsv(out/'analysis/RECOMPUTED_ENDPOINTS.csv',recalculated);writecsv(out/'analysis/VERIFIED_PRIMARY_CONTRASTS.csv',paired_records);writecsv(out/'analysis/CLEAN_DETECTION_CURVES.csv',curve);writecsv(out/'analysis/ALL_UNSUCCESSFUL_EVENTS.csv',failures);writecsv(out/'analysis/RECORD_PHASE_SUMMARY.csv',phase_summary)
    report['status']='PASS_SAVED_RESULTS_ONLY'
    dump(out/'VERIFICATION.json',report)
    print(json.dumps({'status':report['status'],'manifests':report['manifests'],'reaggregation':report['reaggregation'],'report':str(out/'VERIFICATION.json')},indent=2))

if __name__=='__main__':main()
