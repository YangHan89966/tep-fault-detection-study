"""Extract 250 paired test-run + 259 calibration maxima by model; no scoring or fitting.

Usage: python extract_v2.py A3_RETURN.zip TEP_P1_Historical_Code_Models_and_Results_20261026.zip output_directory
Uses archived continuous CUSUM running maxima (k=.05) to vary thresholds.
Stage-D one-threshold alarm bits are independent checks, not the data from which maxima are computed.
"""
import argparse, csv, hashlib,io,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

def sha(buf): return hashlib.sha256(buf).hexdigest()
def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def insist(ok,why):
    if not ok: raise ValueError(why)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('a3',type=Path);p.add_argument('historical',type=Path);p.add_argument('out',type=Path);args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    hist='tep-pca-residual-monitoring/current_study/r5_s21_s22/S21/Stage_D_Return/'
    rows=[];cal=[];checks=[];bindings=[];panelids={}
    allowed={3:['mean_pool_signed','same_mean_signed_axis_squared'],15:['moment_pool_squared','mean_pool_squared']}
    with ZipFile(args.historical) as zh, ZipFile(args.a3) as za:
        mani={r['relative']:r for r in json.loads(za.read('A3_RETURN/RETURN_FILES_SHA256.json'))}
        for state in range(1,6):
            for fault in [3,15]:
                rel=f'results/A3_K005_A4_SUMMARIES/state{state:02}_F{fault:02}.npz'
                file=za.read('A3_RETURN/'+rel)
                insist(sha(file)==mani[rel]['sha256'] and len(file)==mani[rel]['bytes'],'A3 manifest mismatch')
                gate=f'rep{state:02}/F{fault:02}/'
                rec=zh.read(hist+'evaluated/'+gate+'SCORED_RECORDS.npz')
                of=zh.read(hist+'frozen_models/'+gate+'OUTER.npz')
                bindings.append(dict(member=rel,sha256=sha(file),original_stageD_scored_sha256=sha(rec),original_stageD_outer_sha256=sha(of)))
                with np.load(io.BytesIO(file),allow_pickle=False) as a, np.load(io.BytesIO(rec),allow_pickle=False) as o, np.load(io.BytesIO(of),allow_pickle=False) as outer:
                    insist(sha(rec)==str(a['original_scored_records_sha256']) and sha(of)==str(a['original_outer_sha256']),'Stage D sources not bound')
                    insist(str(a['schema_version'])=='A3_K005_A4_SUFFICIENT_STATISTICS_v1' and str(a['alarm_comparison'])=='CUSUM > threshold' and str(a['validation_status'])=='ORIGINAL_TEST_ALARM_BITS_AND_OUTER_PEAKS_PASSED','A3 metadata mismatch')
                    for k,v in [('pre_records',160),('post_records',800),('sample_minutes',3),('calibration_n',259),('calibration_rank',247),('k',.05),('state',state),('fault',fault)]:
                        insist(float(a[k])==v,k)
                    methods=a['methods'].tolist()
                    insist(set(methods)==set(allowed[fault]),'Unrecognized methods')
                    for j,method in enumerate(methods):
                        pos=o['methods'].tolist().index(method)
                        cx=int(o['candidate_index'][pos])
                        h=float(a['original_thresholds'][j])
                        insist(h==float(a['replayed_thresholds'][j])==float(o['thresholds'][pos])==float(outer['thresholds'][cx]),'Original threshold mismatch')
                        insist(cx==int(a['candidate_indices_zero_based'][j]),'Candidate mismatch')
                        shared=dict(panel='Validation 2',contrast='C1' if fault==3 else 'C3',fault=fault,replicate=state,arm='favoured' if method==allowed[fault][0] else 'comparator',source_method=method)
                        chek=dict(**shared,original_h=h,outer_n=259,outer_rank=247)
                        oc=a['outer_cusum_max'][:,j];anchor=np.quantile(a['anchor_cusum_max'][:,j],.9,method='linear')
                        np.testing.assert_allclose(oc,outer['peaks'][:,cx],rtol=2e-11,atol=1e-10)
                        np.testing.assert_allclose(h,anchor*np.sort(oc/anchor)[246],rtol=1e-13,atol=1e-12)
                        for i,x in enumerate(oc):
                            cal.append(dict(**shared,run_id=int(a['outer_run_id'][i]),case_id=str(a['outer_canonical_label'][i]),seed=int(a['outer_seed_u32'][i]),trajectory_sha256=str(a['outer_output_sha256'][i]),original_h=h,cal48_max=float(x)))
                        for role in ['fault','normal']:
                            ids=a[role+'_run_id'];ss=a[role+'_seed_u32'];canonical=a[role+'_canonical_label'].tolist();hashes=a[role+'_output_sha256'].tolist()
                            identity=tuple(zip(ids.tolist(),canonical,ss.tolist(),hashes))
                            lock=(role, fault if role=='fault' else 0)
                            insist(panelids.setdefault(lock,identity)==identity and len(set(canonical))==len(set(hashes))==250,'Test panels not shared across models')
                            np.testing.assert_array_equal(ids,o['run_id'])
                            pre=a[role+'_cusum_pre_max'][:,j];run=a[role+'_cusum_post_running_max'][:,:,j]
                            insist(pre.shape==(250,) and run.shape==(250,800) and np.isfinite(run).all() and np.isfinite(pre).all() and (np.diff(run,axis=1)>=0).all(),'Invalid continuous running maxima')
                            # 0-based post offset 160 is record 321, i.e. 480 min since record 161.
                            p8=run[:,160];p40=run[:,-1];full=np.maximum(pre,p40)
                            # Restore original strict alarm decisions up to each horizon from the archived alarm bits.
                            mask=np.unpackbits(o[role+'_alarm_bits'][:,:,pos],axis=1,count=960,bitorder='little').astype(bool)
                            np.testing.assert_array_equal(pre>h,mask[:,:160].any(1))
                            np.testing.assert_array_equal(run>h,np.logical_or.accumulate(mask[:,160:],axis=1))
                            clean=(pre<=h)&(p40>h); loss=np.where(clean,np.argmax(run>h,axis=1),800)
                            np.testing.assert_array_equal(loss,o[role+'_event_loss_samples'][:,pos])
                            c8=(pre<=h)&(p8>h)
                            if role=='fault':chek.update(clean40_n=int(clean.sum()),clean8_n=int(c8.sum()),prefault_n=int((pre>h).sum()),mean_penalized_loss_min=float((3*loss).mean()))
                            else:chek.update(normal48_alarm_n=int((full>h).sum()))
                            for i,k in enumerate(ids):
                                rows.append(dict(**shared,role=role,run_id=int(k),case_id=canonical[i],seed=int(ss[i]),trajectory_sha256=hashes[i],original_h=h,pre_max=float(pre[i]),post8_max=float(p8[i]),post40_max=float(p40[i]),normal48_max=float(full[i]) if role=='normal' else ''))
                        checks.append(chek)
    insist(len(checks)==20 and len(rows)==10000 and len(cal)==5180,'Unexpected counts')
    write_csv(args.out/'run_peaks.csv',rows);write_csv(args.out/'calibration_peaks.csv',cal);write_csv(args.out/'original_points.csv',checks)
    digest={'source_archives':{f.name:sha(f.read_bytes()) for f in (args.a3,args.historical)},'A3_state_archive_members':bindings,'scope':'Original Validation 2 test panel, C1 and C3, five training replicates, each including 250 fault + 250 shared normal evaluation runs, with 259 outer calibration runs per frozen monitor. Fixed k=0.05. C2 is not in A3.','extraction':'Continuous prestep/160-record maximum and running post maxima of actual CUSUM statistic. 8h includes post offsets 0 through 160; 40h offsets 0 through 799. Detection=(pre_max<=h)&(post_max>h); FAR=normal48_max>h. No new simulation, refit, recalibration or score approximations.','replay':'For every role/run/model, exact equality of pre-crossing indicator and all 800 post running-crossing indicators to Stage D original alarm bits at h; exact penalized delays, original limits, outer peaks within 2e-11 relative/1e-10 absolute.','model_arms':20,'evaluation_rows':len(rows),'outer_calibration_rows':len(cal),'original_post_running_alarm_indicators_checked':len(rows)*800,'products_sha256':{n:sha((args.out/n).read_bytes()) for n in ('run_peaks.csv','calibration_peaks.csv','original_points.csv')}}
    (args.out/'v2_sources_and_replay.json').write_text(json.dumps(digest,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
