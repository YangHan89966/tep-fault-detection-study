"""Extract C2 A4 original-panel sufficient statistics. Both inputs are offline ZIPs.
Usage: python extract_v2c2.py TEP_Zenodo_04_A459_20261013_r1.zip TEP_P1_Historical_Code_Models_and_Results_20261026.zip DEST
"""
from pathlib import Path
from zipfile import ZipFile
import argparse,csv,io,json,hashlib
import numpy as np
sha=lambda b:hashlib.sha256(b).hexdigest()
def readcsv(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def znp(b):
    with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def req(b,reason):
    if not b:raise ValueError(reason)
def unique(rows,key):
    d={r[key]:r for r in rows};req(len(d)==len(rows),'duplicate '+key);return d
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('a459',type=Path);p.add_argument('historical',type=Path);p.add_argument('destination',type=Path);q=p.parse_args()
    q.destination.mkdir(parents=True,exist_ok=True)
    rows=[];cal=[];points=[]
    provenance={'sources':{p.name:sha(p.read_bytes()) for p in [q.a459,q.historical]},'state_archives':[]}
    with ZipFile(q.a459) as parent,ZipFile(q.historical) as hist:
        inner,=[s for s in parent.namelist() if s.endswith('/A459_RETURN_20261009T121907Z_59251e.zip')];ib=parent.read(inner)
        provenance['return_zip']={'member':inner,'sha256':sha(ib)}
        with ZipFile(io.BytesIO(ib)) as a4:
            man=unique(json.loads(a4.read('RETURN_MANIFEST.json')),'path')
            def orig(n):
                b=a4.read(n);req((sha(b),len(b))==(man[n]['sha256'],man[n]['bytes']),'Return manifest: '+n);return b
            stage='tep-pca-residual-monitoring/current_study/'
            raw={}
            for tag in ['STAGE_A_COMPLETED_INDEX.csv','STAGE_C_TEST_COMPLETED_INDEX.csv']:
                raw.update(unique(readcsv(hist.read(stage+'r5_raw_identity/'+tag)),'canonical_label'))
            a4_raw={i['canonical_label']:i['sha256'] for i in readcsv(orig('results/R5/SOURCE_RAW_READ_LOG.csv'))}
            for rep in range(1,6):
                root=f'results/R5/rep{rep:02}_F09/'
                base='r5_s21_s22/S21/Stage_D_Return/'
                state=f'rep{rep:02}/F09/'
                sr=hist.read(stage+base+'evaluated/'+state+'SCORED_RECORDS.npz')
                so=hist.read(stage+base+'frozen_models/'+state+'OUTER.npz')
                events=znp(sr);outer=znp(so);src=orig(root+'SUFFICIENT_PATHS.npz');a=znp(src)
                provenance['state_archives'].append(dict(member=root+'SUFFICIENT_PATHS.npz',sha256=sha(src),original_scored_sha256=sha(sr),original_outer_sha256=sha(so)))
                t_ids=readcsv(orig(root+'TEST_IDENTITIES.csv'));c_ids=readcsv(orig(root+'CALIBRATION_IDENTITIES.csv'))
                req(len(t_ids)==500 and len(c_ids)==259,'physical counts')
                for r in t_ids+c_ids:
                    v=raw[r['canonical_label']];req(all(r[k]==v[k] for k in ('run','seed_u32','role','fault','relative','replication')),'identity mismatch')
                    if r in t_ids:req(a4_raw[r['canonical_label']]==v['output_sha256'],'A4 raw/source differs')
                for method in ['moment_pool_signed','mean_pool_signed']:
                    sh=dict(panel='Validation 2',contrast='C2',fault=9,replicate=rep,arm='favoured' if method=='moment_pool_signed' else 'comparator',source_method=method)
                    j=a['methods'].tolist().index(method);m=events['methods'].tolist().index(method);index=events['candidate_index'][m]
                    h=float(a['old_thresholds'][j]);peaks=a['calibration_peaks'][:,j]
                    req(h==float(events['thresholds'][m])==float(outer['thresholds'][index]),'threshold')
                    np.testing.assert_allclose(peaks,outer['peaks'][:,index],rtol=2e-11,atol=1e-10)
                    req(len(peaks)==259 and int(outer['rank'])==247 and int(outer['n'])==259,'calibration')
                    np.testing.assert_allclose(h,a['anchors'][j]*np.sort(peaks/a['anchors'][j])[246],rtol=1e-13,atol=1e-12)
                    chk=dict(**sh,original_h=h,outer_n=259,outer_rank=247)
                    def identity(r):return dict(run_id=int(r['run']),case_id=r['canonical_label'],seed=int(r['seed_u32']),trajectory_sha256=raw[r['canonical_label']]['output_sha256'])
                    for i,r in enumerate(c_ids):
                        req(int(r['run'])==i+1,'calibration order');cal.append(dict(**sh,**identity(r),original_h=h,cal48_max=float(peaks[i])))
                    for role in ['fault','normal']:
                        ids=[r for r in t_ids if r['role']==role+'_evaluation'];req([int(r['run']) for r in ids]==list(range(1,251)) and all(int(r['fault'])==(9 if role=='fault' else 0) for r in ids),'evaluation order')
                        pr=a[role+'_pre_max'][:,j];running=a[role+'_post_running_max'][:,:,j];p8=running[:,160];p40=running[:,-1];entire=np.maximum(pr,p40)
                        req(pr.shape==(250,) and running.shape==(250,800) and np.isfinite(pr).all() and np.isfinite(running).all() and (np.diff(running,axis=1)>=0).all(),'post running max')
                        # An independent artifact of the original Stage D evaluation supplies one alarm bit per record.
                        mask=np.unpackbits(events[role+'_alarm_bits'][:,:,m],axis=1,count=960,bitorder='little').astype(bool)
                        np.testing.assert_array_equal(pr>h,mask[:,:160].any(1))
                        np.testing.assert_array_equal(running>h,np.logical_or.accumulate(mask[:,160:],axis=1))
                        clean=(pr<=h)&(p40>h);loss=np.where(clean,(running>h).argmax(1),800)
                        np.testing.assert_array_equal(loss,events[role+'_event_loss_samples'][:,m])
                        if role=='fault':chk.update(clean40_n=int(clean.sum()),clean8_n=int(((pr<=h)&(p8>h)).sum()),prefault_n=int((pr>h).sum()),mean_penalized_loss_min=float((3*loss).mean()))
                        else:chk.update(normal48_alarm_n=int((entire>h).sum()))
                        for i,r in enumerate(ids):rows.append(dict(**sh,role=role,**identity(r),original_h=h,pre_max=float(pr[i]),post8_max=float(p8[i]),post40_max=float(p40[i]),normal48_max=float(entire[i]) if role=='normal' else ''))
                    points.append(chk)
    req(len(rows)==5000 and len(cal)==2590 and len(points)==10,'row counts')
    for arm,expected in [('favoured',(8.72,5.04)),('comparator',(7.60,6.88))]:
        pts=[r for r in points if r['arm']==arm]
        np.testing.assert_allclose([100*np.mean([r[k] for r in pts])/250 for k in ['clean40_n','normal48_alarm_n']],expected,atol=1e-10)
    for name,tt in [('run_peaks.csv',rows),('calibration_peaks.csv',cal),('original_points.csv',points)]:write(q.destination/name,tt)
    provenance.update(scope='Validation 2 (R5), unchanged original cohorts; 250 shared normal and 250 Fault 9 test runs. Five frozen C2 pairs; original mean and second-moment pools with signed score, k=0.05. No A4 exclusions, new thresholds, simulation, model fitting or raw rescoring.',windows='Pre=max(records 1:160), post8=max(161:321), post40=max(161:960); inclusive endpoints, strict statistic>h. Original rule matches 247th of 259 full-48-h calibration peaks, using historical arithmetic with the fixed common anchor.',checks={'test_model_run_rows':5000,'identical_pre_alarm_booleans':5000,'identical_post_running_crossing_booleans':4000000,'identical_penalized_delays':5000,'calibration_peaks_checked_against_original_outer':2590,'S22_points_percent':{'moment_pool_signed':{'FAR':5.04,'clean40':8.72},'mean_pool_signed':{'FAR':6.88,'clean40':7.60}}},outputs={n:sha((q.destination/n).read_bytes()) for n in ['run_peaks.csv','calibration_peaks.csv','original_points.csv']})
    (q.destination/'v2c2_sources_and_replay.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n',encoding='utf8')
if __name__=='__main__':main()
