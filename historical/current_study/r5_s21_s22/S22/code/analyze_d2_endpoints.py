#!/usr/bin/env python3
"""Existing-record D2 endpoint sensitivity; no simulation, fitting, tuning or new permutations."""
from __future__ import annotations
import argparse,csv,hashlib,json,datetime,platform
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def contrasts(v):
    a,b,c,d=[v[k] for k in [(0,0),(1,0),(0,1),(1,1)]]
    ans={'path_at_original_limit':b-a,'limit_on_original_path':c-a,'interaction':d-b-c+a,'total':d-a,'path_at_permuted_limit':d-c,'limit_on_permuted_path':d-b}
    if abs(ans['total']-ans['path_at_original_limit']-ans['limit_on_original_path']-ans['interaction'])>1e-8:raise ValueError('factorial identity')
    return ans

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True);ap.add_argument('--replay',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    spec=json.loads(a.protocol.read_text()); a.out.mkdir(parents=True,exist_ok=True)
    if list(a.out.iterdir()):raise ValueError('Nonempty output directory: preserve prior results')
    if sha(a.original/'FAULT_RECORDS.npz')!=spec['scientific_inputs']['fault_records']['sha256']:raise ValueError('input identity')
    with np.load(a.original/'FAULT_RECORDS.npz',allow_pickle=False) as z: old={k:z[k] for k in z}
    with np.load(a.replay/'FAULT_RECORDS.npz',allow_pickle=False) as z: replay={k:z[k] for k in z}
    if set(old)!=set(replay):raise ValueError('replay schema')
    for k in old:
        if not np.array_equal(old[k],replay[k]):raise ValueError('baseline replay mismatch '+k)
    keys,rec,ids=old['keys'],old['records'],old['run_ids']; families=old['family_names'].tolist()
    if rec.shape!=(7200,75,6) or len(np.unique(keys,axis=0))!=7200:raise ValueError('record count')
    pre,det,clean,delay,loss,occ=[rec[:,:,j].astype(np.int64) for j in range(6)]
    if not np.array_equal(clean,(det.astype(bool)&~pre.astype(bool)).astype(int)):raise ValueError('clean event identity')
    if not np.array_equal(loss,np.where(clean,delay,800)):raise ValueError('loss event identity')
    if not np.all(np.isin(pre,[0,1])) or not np.all((delay>=0)&(delay<=800)):raise ValueError('event domains')
    # Recalculate all original repetition endpoints before any new endpoint summaries.
    original_rows=readcsv(a.original/'PER_REPETITION.csv'); normal={}; errors=[]
    if len(original_rows)!=len(keys):raise ValueError('original repetition rows')
    for i,r in enumerate(original_rows):
        key=(int(r['split'][1:]),int(r['block']),int(r['repetition']),families.index(r['family']),int(r['path_order']),int(r['limit_order']))
        if tuple(keys[i])!=key:raise ValueError('original key order')
        if len(np.unique(ids[i]))!=75:raise ValueError('run identities')
        original_metrics={'success_percent':100*clean[i].mean(),'mean_loss_min':3*loss[i].mean(),'prefault_percent':100*pre[i].mean(),'success_within_2h_percent':100*((clean[i]==1)&(delay[i]<=40)).mean(),'success_within_8h_percent':100*((clean[i]==1)&(delay[i]<=160)).mean()}
        errors += [abs(float(r[k])-v) for k,v in original_metrics.items()]
        normal[key]=float(r['normal_far_percent'])
    if max(errors)>1e-9:raise ValueError('original reported endpoints')
    rows=[]
    for H in spec['endpoints']['horizon_minutes']:
        definitions=[(f'horizon_{H}',H,'horizon')]
        if H==2400:definitions += [(f'penalty_{P}',P,'penalty') for P in spec['endpoints']['full_horizon_failure_penalties_minutes'] if P!=2400]
        ok=(clean==1)&(delay*3<=H)
        for name,P,kind in definitions:
            clean_term=np.where(ok,delay*3,0).mean(1);fail_rate=1-ok.mean(1)
            loss_m=clean_term+P*fail_rate
            for i,k in enumerate(keys):
                rows.append({'split':f'D{int(k[0]):02d}','block':int(k[1]),'repetition':int(k[2]),'family':families[int(k[3])],'path_order':int(k[4]),'limit_order':int(k[5]),'endpoint':name,'horizon_min':H,'failure_penalty_min':P,'kind':kind,'n_fault':75,'clean_success_percent':float(100*ok[i].mean()),'failure_percent':float(100*fail_rate[i]),'clean_delay_component_min':float(clean_term[i]),'failure_penalty_component_min':float(P*fail_rate[i]),'mean_loss_min':float(loss_m[i]),'mean_loss_divided_by_H':float(loss_m[i]/H),'prefault_percent':float(100*pre[i].mean()),'normal_full_episode_far_percent':normal[tuple(k)]})
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ['split','block','family','path_order','limit_order','endpoint'])].append(r)
    value_names=['clean_success_percent','failure_percent','clean_delay_component_min','failure_penalty_component_min','mean_loss_min','mean_loss_divided_by_H','prefault_percent','normal_full_episode_far_percent']
    per_split=[]
    for rr in groups.values():
        if len(rr)!=20:raise ValueError('permutation denominator')
        r={k:v for k,v in rr[0].items() if k not in value_names+['repetition']}
        r.update({k:float(np.mean([v[k] for v in rr])) for k in value_names});per_split.append(r)
    g=defaultdict(list)
    for r in per_split:g[(r['split'],r['block'],r['family'],r['endpoint'])].append(r)
    effects=[]
    for rr in g.values():
        if len(rr)!=4:raise ValueError('four cells')
        out={k:rr[0][k] for k in ['split','block','family','endpoint','horizon_min','failure_penalty_min']}
        values={k:contrasts({(r['path_order'],r['limit_order']):r[k] for r in rr}) for k in value_names}
        for effect in values['mean_loss_min']:
            r={**out,'contrast':effect,**{k:v[effect] for k,v in values.items()}}
            if abs(r['mean_loss_min']-r['clean_delay_component_min']-r['failure_penalty_component_min'])>1e-8:raise ValueError('loss decomposition')
            effects.append(r)
    original_effects=readcsv(a.original/'INTERACTION_PER_SPLIT.csv')
    im={(r['split'],int(r['block']),r['family']):r for r in original_effects}
    baseline_errors=[]
    for r in effects:
        if r['endpoint']=='horizon_2400':
            v=float(im[(r['split'],r['block'],r['family'])][r['contrast']]); baseline_errors.append(abs(r['mean_loss_min']-v))
    if max(baseline_errors)>1e-8:raise ValueError('baseline interaction mismatch')
    def summaries(input_rows,keycols,values):
        g=defaultdict(list)
        for r in input_rows:g[tuple(r[k] for k in keycols)].append(r)
        result=[]
        for rr in g.values():
            if len(rr)!=10:raise ValueError('split denominator')
            line={k:rr[0][k] for k in keycols}
            for col in values:
                vals=np.array([r[col] for r in rr]);line.update({col+'_mean':float(vals.mean()),col+'_median':float(np.median(vals)),col+'_min':float(vals.min()),col+'_max':float(vals.max()),col+'_negative_splits':int((vals<-1e-9).sum()),col+'_zero_splits':int((np.abs(vals)<=1e-9).sum()),col+'_positive_splits':int((vals>1e-9).sum())})
            result.append(line)
        return result
    cell_summary=summaries(per_split,['block','family','path_order','limit_order','endpoint','horizon_min','failure_penalty_min'],value_names)
    effect_summary=summaries(effects,['block','family','endpoint','horizon_min','failure_penalty_min','contrast'],value_names)
    for name,rr in [('PER_REPETITION_ENDPOINTS.csv',rows),('PER_SPLIT_ENDPOINTS.csv',per_split),('PER_SPLIT_CONTRASTS.csv',effects),('CELL_SUMMARIES.csv',cell_summary),('CONTRAST_SUMMARIES.csv',effect_summary)]:writecsv(a.out/name,rr)
    # Label the numeric-zero counting convention; it never changes alarm decisions.
    audit={'status':'COMPLETED_EXISTING_RECORD_ENDPOINT_SENSITIVITY','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_records_sha256':sha(a.original/'FAULT_RECORDS.npz'),'replayed_records_sha256':sha(a.replay/'FAULT_RECORDS.npz'),'replay_all_arrays_equal':True,'event_integer_fields_compared':int(rec.size),'key_fields_compared':int(keys.size),'run_id_fields_compared':int(ids.size),'original_endpoint_fields_compared':len(errors),'max_original_endpoint_difference':max(errors),'original_effect_fields_compared':len(baseline_errors),'max_original_effect_difference':max(baseline_errors),'new_repetition_endpoint_rows':len(rows),'new_per_split_rows':len(per_split),'new_per_split_contrast_rows':len(effects),'horizons':spec['endpoints']['horizon_minutes'],'penalties':spec['endpoints']['full_horizon_failure_penalties_minutes'],'summary_numeric_zero_tolerance':1e-9,'alarm_rule_tolerance_added':False,'simulation_executed':False,'refit_executed':False,'independent_raw_52D_replay':False,'fixed_low_dimensional_score_replay':True,'original_files_overwritten':False,'protocol_sha256':sha(a.protocol),'analysis_code_sha256':sha(__file__),'python':platform.python_version(),'numpy':np.__version__}
    (a.out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    for r in effect_summary:
        if r['family']=='signed' and r['block']==1 and r['contrast']=='interaction':print(r['endpoint'],r['mean_loss_min_mean'],r['mean_loss_min_min'],r['mean_loss_min_max'],'signs',r['mean_loss_min_negative_splits'],r['mean_loss_min_zero_splits'],r['mean_loss_min_positive_splits'],'success_interaction_pp',r['clean_success_percent_mean'],'delaycomp',r['clean_delay_component_min_mean'],'penaltycomp',r['failure_penalty_component_min_mean'])
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
