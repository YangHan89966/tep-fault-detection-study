"""Read the original, completed Fault 9 return ZIP; no simulation or refitting.

Usage: python replay_relearning.py path/to/F9_PAIRED_RELEARNING_COMPLETE.zip --out replay.json
Python standard library verifies archive, locks and IDs; NumPy reads frozen
arrays, derives endpoint counts and repeats the 20,000-draw paired bootstrap.
"""
import argparse, csv, hashlib, io, json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import numpy as np

EXPECTED = '7839254b073d226faeb7c7dcd7ac9e2f2b37ea19ed0e30b7981722fd5ff20e2b'
sha = lambda b: hashlib.sha256(b).hexdigest()

def check(ok, message):
    if not ok: raise ValueError(message)

def replay(path):
    check(sha(path.read_bytes()) == EXPECTED, 'Different source ZIP')
    with ZipFile(path) as z:
        js=lambda n: json.loads(z.read(n))
        tab=lambda n: list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
        digest=lambda n: sha(z.read(n))
        check(z.testzip() is None and len(z.namelist())==len(set(z.namelist())), 'Archive CRC or duplicate member')
        design,code,models,lim,final,summary=[js(x) for x in ('locks/DESIGN_LOCK.json','locks/CODE_LOCK.json','MODELS_LOCK.json','THRESHOLDS_LOCK.json','FINAL_COMPLETE.json','outcomes/SUMMARY.json')]
        for p,h in design['files'].items(): check(digest('locks/'+p)==h,p)
        check(digest('locks/PROTOCOL.json')==summary['protocol_sha256'] and summary['status']=='COMPLETE' and not summary['no_scientific_claim'],'Wrong scope/protocol')
        check(digest('locks/DESIGN_LOCK.json')==final['design']==models['design']==lim['design']==code['design_lock'],'Design binding')
        check(digest('locks/CODE_LOCK.json')==final['code']==models['code']==lim['code'],'Code binding')
        check(digest('MODELS_LOCK.json')==final['models_lock_sha']==lim['models_lock_sha'] and digest('THRESHOLDS_LOCK.json')==final['thresholds_lock_sha'],'Model/threshold binding')
        times=[design['locked_utc'],models['utc'],lim['utc'],final['utc']]
        check(all(datetime.fromisoformat(a)<datetime.fromisoformat(b) for a,b in zip(times,times[1:])) and not models['new_calibration_or_evaluation_seen'] and not lim['new_evaluation_seen'],'Stage order')
        for name,h in final['all_outputs'].items(): check(digest('outcomes/'+name)==h,name)
        check(digest('RAW_RECEIPTS.csv')==final['receipts_sha'],'Raw ledger binding')
        manifest=tab('locks/CASE_MANIFEST.csv'); cases={r['case_id']:r for r in manifest}
        allrec=tab('RAW_RECEIPTS.csv'); ledger={r['case_id']:r for r in allrec}
        check(len(ledger)==len(allrec)==len(cases)==len(manifest)==len({r['key_hex'] for r in allrec})==len({r['raw_sha256'] for r in allrec})==2719 and set(ledger)==set(cases),'Physical identities')
        for key,r in ledger.items():check(all(r[k]==cases[key][k] for k in ('key_hex','role','rep')),key)
        an=[]
        for j in range(1,6):
            name='R%02d'%j;root='models/'+name+'/'
            check(digest(root+'MODEL_LOCK.json')==models['models'][name],name)
            for n,h in js(root+'MODEL_LOCK.json')['files'].items():check(digest(root+n)==h,n)
            inp=tab(root+'INPUT_RECEIPTS.csv')
            check(len(inp)==384 and {r['case_id'] for r in inp}=={i for i,r in cases.items() if r['stage']=='development' and int(r['rep'])==j},'Development roles')
            for r in inp:check(all(r[k]==ledger[r['case_id']][k] for k in ['role','rep','key_hex','raw_sha256']),r['case_id'])
            a=[]
            for v,pool in [('snapshot','moment'),('native','moment'),('snapshot','mean'),('native','mean')]:
                idx=js(root+v+'/SELECTION.json')['choices'][pool]
                with np.load(io.BytesIO(z.read(root+v+'/INNER.npz')),allow_pickle=False) as d:a.append(d['anchors'][idx])
            an.append(a)
        check(lim['rank']==190 and lim['n']==199 and digest('calibration/CALIBRATION.npz')==lim['calibration_sha'],'Calibration binding')
        with np.load(io.BytesIO(z.read('calibration/CALIBRATION.npz')),allow_pickle=False) as d:
            check(list(d['case_ids'])==[r['case_id'] for r in manifest if r['stage']=='calibration'] and list(d['raw_sha256'])==[ledger[k]['raw_sha256'] for k in d['case_ids']],'Calibration identities')
            an=np.array(an); h=an*np.sort(d['maxima']/an,axis=0)[189]
            check(np.array_equal(h,d['thresholds']) and np.array_equal(h,np.array(lim['h'])),'Threshold reconstruction')
        per=tab('outcomes/PER_RUN.csv'); check(len(per)==12000,'Incomplete outcomes')
        full={};peaks={}
        for role,n in [('normal',400),('fault',200)]:
            ids=[r['case_id'] for r in manifest if r['role']==role+'_evaluation']; index={x:i for i,x in enumerate(ids)}
            check(len(index)==n,role); ev=np.full((n,5,4,6),-1,np.int32); pk=np.full((n,5,4,5),np.nan)
            for r in per:
                if r['role']!=role:continue
                i,j,k=index[r['case_id']],int(r['replicate'])-1,int(r['monitor'][1:])-1; th=h[j,k]
                check(ev[i,j,k,0]==-1 and float(r['h'])==th,'Repeated/mislabeled arm')
                p,b2,b8,b40,a=[float(r[u]) for u in ['pre_max','post2_max','post8_max','post40_max','full48_max']]
                check(max(p,b40)==a and 0<=b2<=b8<=b40,'Inconsistent maxima')
                first,post=int(r['first']),int(r['first_post']); ok=p<=th; clean=ok and b40>th
                check((first>0)==(a>th) and (0<first<=160)==(not ok) and (post>0)==(b40>th) and (not clean or first==post>=161),'Crossing index')
                ev[i,j,k]=[clean,ok and b8>th,ok and b2>th,not ok,a>th,3*(post-161) if clean else 2400]
                check(ev[i,j,k].tolist()==[int(r[u]) for u in ['clean40','clean8','clean2','prefault','any_alarm','loss_min']],'Event mismatch')
                pk[i,j,k]=[p,b2,b8,b40,a]
            check(ev.min()>=0 and np.isfinite(pk).all(),'Missing data'); full[role]=ev;peaks[role]=pk
        rng=np.random.default_rng(2026110406); results={}; reps=[];loomean={}
        original={(r['pool'],r['endpoint']):r for r in tab('outcomes/PAIRED_CONTRASTS.csv')}
        for role,metrics in [('normal',[(4,'normal_48h_FAR'),(0,'normal_clean_floor')]),('fault',[(0,'fault_clean40'),(1,'fault_clean8'),(2,'fault_clean2'),(3,'fault_prefault_alarm'),(5,'fault_mean_penalized_loss')])]:
            n=len(full[role]);draws=rng.integers(n,size=(20000,n),dtype=np.int32);w=np.zeros((20000,n),np.int16)
            np.add.at(w,(np.arange(len(w))[:,None],draws),1)
            for k,name in metrics:
                x=full[role][:,:,:,k]; scale=1 if k==5 else 100
                for pool,a in [('moment',0),('mean',2)]:
                    diff=x[:,:,a]-x[:,:,a+1]; interval=np.quantile(w@diff.sum(1),[.025,.975])*scale/(n*5)
                    vals=[*list(x[:,:,a:a+2].mean((0,1))*scale),float(diff.mean()*scale),*list(interval)]
                    rec=original[pool,name]
                    check(np.allclose(vals,[float(rec[c]) for c in ['ci_estimate','post_estimate','ci_minus_post','lower95','upper95']],rtol=0,atol=1e-10),'Contrast: '+name)
                    results[pool+'/'+name]=dict(zip(['controller_input','post_update','difference','p025','p975'],vals))
                    if pool=='moment' and name=='fault_clean40':loomean={str(j+1):float(np.delete(diff,j,axis=1).mean()*100) for j in range(5)}
        for j in range(5):
            reps.append(dict(replicate=j+1,clean40_fault_n=full['fault'][:,j,:2,0].sum(0).tolist(),clean8_fault_n=full['fault'][:,j,:2,1].sum(0).tolist(),clean2_fault_n=full['fault'][:,j,:2,2].sum(0).tolist(),prefault_n=full['fault'][:,j,:2,3].sum(0).tolist(),normal_alarm_n=full['normal'][:,j,:2,4].sum(0).tolist(),leave_one_out_delta40_pp=loomean[str(j+1)]))
        h5=np.sort(peaks['normal'][:,:,:,4],axis=0)[379];nexc=(peaks['normal'][:,:,:,4]>h5).sum(0)
        ms=(peaks['fault'][:,:,:,0]<=h5)&(peaks['fault'][:,:,:,3]>h5)
        check(np.all(nexc==20),'5% matching')
        matched=(100*ms.mean((0,1))).tolist()
        for r in tab('outcomes/MATCHED_5PCT_CONTRASTS.csv'):
            if r['endpoint']=='fault_clean40' and r['replicate']=='mean_5':
                k=0 if r['pool']=='moment' else 2
                check(np.allclose(matched[k:k+2],[float(r['ci_pct']),float(r['post_pct'])]),'Matched estimates')
        primary=results['moment/fault_clean40'];risk=results['moment/normal_48h_FAR']
        decision={'lower95_delta40_above_zero':bool(primary['p025']>0),'upper95_deltaFAR_at_most_5pp':bool(risk['p975']<=5)}
        check(list(decision.values())==[summary['decision']['detection_lower95_gt_0'],summary['decision']['normal_risk_upper95_le_5pp']],'Decision')
        return {'status':'PASS','source_zip':path.name,'source_sha256':EXPECTED,'audit_scope':'Historical return: manifest keys and raw-file SHA receipts, archive members, learning and calibration locks, and endpoint replay. The 2719 RAW.npz arrays, 960-row scientific chart paths, generation logs per case and the old 1199-key manifest are not included: this replay is not a raw-trajectory rerun or an independent observation of every generation time.','archive_members':len(z.namelist()),'design_models_thresholds_completion_UTC':times,'stage_counts':dict(Counter(r['stage'] for r in manifest)),'physical_keys':len(ledger),'evaluation_rows':len(per),'paired_bootstrap_draws':20000,'pooled_over_replicates':'Equal weight over five fitted replicates evaluated on identical physical cases; resample trajectories separately by role.','all_14_contrasts':results,'moment_counts_CI_then_post':reps,'matched_5pct_clean40_by_M1_M2_M3_M4':matched,'matched_normal_n_per_arm_and_rep':20,'decision':decision}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source_zip',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    s=json.dumps(replay(a.source_zip),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if a.out:a.out.write_text(s,encoding='utf-8')
    else:print(s,end='')
