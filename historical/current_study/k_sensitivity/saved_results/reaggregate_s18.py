"""Verify saved S18 outcomes and N paired intervals. No raw data or model fitting.
Usage: python -B reaggregate_s18.py --output NEW_DIRECTORY [--bootstrap]
Requires NumPy. Refuses to overwrite the output directory.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
POLICIES=('k005','k007','auto_dev')
N_METHODS=('adaptive_5family_top5','fixed_ad_top5','independent_top1','q_cusum')
CONTRASTS=((1,0,'007_minus_005'),(2,0,'auto_minus_005'),(2,1,'auto_minus_007'))
FAULTS=(1,2,4,5,6,7,8,10,11,12,13,14,16,17,18,19,20)

def csvrows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def close(actual,expected,label):
    if not np.allclose(actual,expected,rtol=0,atol=1e-9):raise ValueError(f'Mismatch: {label}: {actual} != {expected}')
def events(mask,normal,study):
    pre=mask[...,:160].any(-1);post=mask[...,160:];det=post.any(-1)
    delay=np.where(det,post.argmax(-1),800)*3;clean=det & ~pre
    loss=np.where(clean,delay,2400)
    early=det if study=='N' else clean
    return {'normal_far_pct':normal.any(-1)*100.,'prefault_alarm_pct':pre*100.,
        'detection_pct':det*100.,'clean_detection_pct':clean*100.,
        'early_60_pct':(early & (delay<=60))*100.,'early_120_pct':(early & (delay<=120))*100.,
        'early_480_pct':(early & (delay<=480))*100.,
        'primary_loss_min':delay if study=='N' else loss,
        'miss_penalized_mean_delay_min':delay,'complete_episode_mean_loss_min':loss,
        'occupancy_pct':post.sum(-1)/8.}
def read_masks(path):
    with np.load(path,allow_pickle=False) as d:
        f=np.unpackbits(d['fault_masks'],axis=-1,count=960,bitorder=str(d['bitorder'])).astype(bool)
        n=np.unpackbits(d['normal_masks'],axis=-1,count=960,bitorder=str(d['bitorder'])).astype(bool)
        methods=d['methods'].tolist();policies=d['policies'].tolist();ks=d['k_by_policy'].copy()
        if policies!=list(POLICIES) or d['run_ids'].tolist()!=list(range(1,251)):
            raise ValueError('Unexpected policy/run identity')
    return f,n,methods,ks

def main(out,bootstrap):
    out.mkdir(parents=True,exist_ok=False)
    checks=0; source=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
    for r in source['source_files']:
        if hashlib.sha256((ROOT/r['destination']).read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Source hash: '+r['destination'])
    all_events={}; nr=csvrows('N_PER_FAULT.csv');dr=csvrows('DV_PER_FAULT.csv')
    for study,fs,rows in [('N',FAULTS,nr),('DV',(3,9,15),dr)]:
        for fault in fs:
            f,n,methods,ks=read_masks(ROOT/'masks'/f'{study}_F{fault:02d}_PAIRED.npz')
            ev=events(f,n,study);all_events[study,fault]=(ev,methods)
            for pi,pol in enumerate(POLICIES):
                for j,m in enumerate(methods):
                    row=next(r for r in rows if int(r['fault'])==fault and r['method']==m and r['policy']==pol)
                    for key,v in ev.items():close(v[pi,j].mean(),float(row[key]),f'{study}/{fault}/{m}/{pol}/{key}');checks+=1
                    close(ks[pi,j],float(row['k']),f'{study}/{fault}/{m}/{pol}/k');checks+=1
            if study=='N':
                with np.load(ROOT/'frozen_N_reference'/f'F{fault:02d}_BASELINE.npz',allow_pickle=False) as b:
                    bf=np.unpackbits(b['fault_alarm_packbits'],axis=-1,count=960,bitorder='little').astype(bool)
                    bn=np.unpackbits(b['normal_alarm_packbits'],axis=-1,count=960,bitorder='little').astype(bool)
                if not np.array_equal(f[0],bf[:2]) or not np.array_equal(n[0],bn[:2]):raise ValueError('k005 baseline differs')
                bev=events(bf,bn,'N')
                for j,m in enumerate(N_METHODS[2:],start=2):
                    row=next(r for r in rows if int(r['fault'])==fault and r['method']==m)
                    for key,v in bev.items():close(v[j].mean(),float(row[key]),f'N frozen/{fault}/{m}/{key}');checks+=1
    for row in csvrows('N_MACRO.csv'):
        rr=[r for r in nr if r['method']==row['method'] and (r['policy']==row['policy'] or row['policy']=='frozen_per_fault')]
        if len(rr)!=17:raise ValueError('N macro denominator')
        for key in all_events['N',1][0]:close(np.mean([float(r[key]) for r in rr]),float(row[key]),f'N macro/{key}');checks+=1
    choices=0
    for study in ('N','DV'):
        rows=csvrows(study+'_AUTO_SELECTION.csv');fix=json.loads((ROOT/(study+'_AUTO_SELECTION_FIXED.json')).read_text())
        for fault in sorted(set(int(r['fault']) for r in rows)):
            methods=all_events[study,fault][1]
            for j,m in enumerate(methods):
                candidates=[r for r in rows if int(r['fault'])==fault and r['method']==m]
                if len(candidates)!=5:raise ValueError('Auto grid size')
                win=min(candidates,key=lambda r:(int(r['total_event_loss_samples']),abs(round(100*float(r['k']))-5),float(r['k'])))
                chosen=fix['selected_k'][str(fault)][j] if study=='N' else fix['selected_k'][str(fault)][m]
                close(float(win['k']),chosen,f'{study}/{fault}/{m}/selection');choices+=1
    for row in csvrows('DV_PRIMARY_CONTRASTS.csv'):
        rr={r['method']:r for r in dr if r['fault']==row['fault'] and r['policy']==row['policy']}
        a,b=rr[row['comparison']],rr[row['reference']]
        for k,field in [('contrast_mean_loss_min','primary_loss_min'),('contrast_normal_far_pp','normal_far_pct'),('contrast_clean_120_pp','early_120_pct')]:
            close(float(a[field])-float(b[field]),float(row[k]),'DV contrast/'+k);checks+=1
    result={'source_files_verified':len(source['source_files']),'numeric_cells_checked':checks,'auto_selections_checked':choices,
            'scope':'Saved-alarm reaggregation only. No raw-to-score replay, simulation, training or threshold change.',
            'python':sys.version,'numpy':np.__version__,'N_bootstrap':'not requested'}
    if bootstrap:
        B=20000;keys0=('miss_penalized_mean_delay_min','complete_episode_mean_loss_min','early_60_pct','early_120_pct','prefault_alarm_pct')
        names=('mean_delay_min','event_loss_min','early60_pp','early120_pp','prefault_pp')
        keyset=[(m,c,metric) for _,_,c in CONTRASTS for m in N_METHODS[:2] for metric in names]
        nk=[(m,c,'normal_far_pp') for _,_,c in CONTRASTS for m in N_METHODS[:2]]
        boot=np.zeros((B,len(keyset)));point=np.zeros(len(keyset));nv=np.zeros((250,len(nk)))
        for fault in FAULTS:
            ev,_=all_events['N',fault]
            values=np.stack([ev[k][new,j]-ev[k][old,j] for new,old,_ in CONTRASTS for j in range(2) for k in keys0],axis=1)
            point+=values.mean(0)/17
            rng=np.random.default_rng(2026091320+fault)
            for st in range(0,B,1000):
                w=rng.multinomial(250,np.full(250,1/250),size=min(1000,B-st));boot[st:st+len(w)]+=(w@values)/(250*17)
            nv+=np.stack([ev['normal_far_pct'][new,j]-ev['normal_far_pct'][old,j] for new,old,_ in CONTRASTS for j in range(2)],axis=1)/17
        rng=np.random.default_rng(2026091388);nb=np.empty((B,len(nk)))
        for st in range(0,B,1000):
            w=rng.multinomial(250,np.full(250,1/250),size=min(1000,B-st));nb[st:st+len(w)]=w@nv/250
        expected=csvrows('N_PAIRED_MACRO.csv');results=[]
        for keys,pts,draws in ((keyset,point,boot),(nk,nv.mean(0),nb)):
            ci=np.quantile(draws,[.025,.975],axis=0,method='linear')
            for i,(m,c,metric) in enumerate(keys):
                row=next(r for r in expected if r['method']==m and r['contrast']==c and r['metric']==metric)
                close([pts[i],ci[0,i],ci[1,i]],[float(row[x]) for x in ('estimate','ci_low','ci_high')],f'N bootstrap/{m}/{c}/{metric}')
                results.append({'method':m,'contrast':c,'metric':metric,'estimate':float(pts[i]),'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i])})
        (out/'N_PAIRED_RECOMPUTED.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        result['N_bootstrap']={'draws':B,'contrasts_and_endpoints':len(results),'all_saved_estimates_and_interval_endpoints_match':True}
    (out/'REAGGREGATION_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',action='store_true')
    a=p.parse_args();main(a.output,a.bootstrap)
