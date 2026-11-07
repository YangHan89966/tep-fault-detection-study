"""Reaggregate Supplementary Data S20 from saved continuous CUSUM paths.
Requires NumPy, pandas and SciPy. Reads data only; no simulation, fitting, raw-process
access, installation or network. Outputs must be written outside this input folder.
"""
from __future__ import annotations
import argparse, hashlib, io, json, platform, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

METHODS = ['mean_pool_squared', 'moment_pool_squared']
RANKS = list(range(234,260))
CHECKS = []

def sha(b): return hashlib.sha256(b).hexdigest()
def check(name, condition, **extra):
    ok=bool(condition);CHECKS.append(dict(check=name,passed=ok,**extra))
    if not ok: raise AssertionError(name)
def close(name, a, b, atol=1e-9):
    a=np.asarray(a);b=np.asarray(b)
    ok=a.shape==b.shape and np.allclose(a,b,rtol=1e-12,atol=atol,equal_nan=False)
    err=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else 0.
    check(name,ok,max_abs_error=err)
def arr(z,n):
    with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as f: return {k:f[k] for k in f.files}
def frame(z,n):return pd.read_csv(io.BytesIO(z.read(n)),float_precision='round_trip')
def events(mask):
    pre=np.any(mask[:,:160,:],axis=1)
    post=mask[:,160:,:]
    det=np.any(post,axis=1)
    first=np.full(det.shape,800,dtype=np.int64)
    hit=np.argmax(post,axis=1)
    first[det]=hit[det]
    clean=det & ~pre
    occupancy=post.sum(axis=1)
    withdrawn=((np.arange(800)[None,:,None]>=first[:,None,:]) & ~post).sum(axis=1)
    return dict(prefault_alarm=pre.astype(int),detected=det.astype(int),clean_detection=clean.astype(int),
      first_delay_samples_penalized=first,event_loss_samples=np.where(clean,first,800),
      occupancy_points=occupancy,postalarm_below_points=withdrawn)
def ci(k,n=250):
    return (0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k))

class FolderReader:
    def __init__(self, root: Path): self.root = root.resolve()
    def read(self, name: str) -> bytes:
        target = (self.root / name).resolve()
        if not target.is_relative_to(self.root): raise ValueError("Unsafe relative path")
        return target.read_bytes()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=args.input.resolve();out=args.output.resolve()
    if out.is_relative_to(root):raise ValueError('Output must be outside the input folder')
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        manifest=json.loads((root/'FILES_SHA256.json').read_text())
        actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
        check('input_exact_manifest_set',set(manifest)==actual-{'FILES_SHA256.json'})
        for n,h in manifest.items():check('member/'+n,sha(FolderReader(root).read(n))==h)
        z=FolderReader(root/'data');tool=FolderReader(root/'reference')
        ident=json.loads(z.read('TOOL_IDENTITY.json'));protocol=json.loads(z.read('PROTOCOL_FIXED.json'))
        check('protocol_identity',sha(z.read('PROTOCOL_FIXED.json'))==ident['protocol_sha256'])
        check('locked_rank_grid_and_k',protocol['threshold_rank_grid']==RANKS and protocol['fixed_k']==.05)
        scores=arr(z,'SCORES_FIXED_F15.npz');baseline=arr(tool,'assets/BASELINE.npz')
        inner=arr(tool,'assets/INNER.npz');outer=arr(tool,'assets/OUTER.npz')
        selection=json.loads(tool.read('assets/SELECTION.json'))['choices']
        check('score_keys',set(scores)=={'fault','normal','methods','run_ids','thresholds','ranks'})
        check('score_identifiers',scores['methods'].tolist()==METHODS and np.array_equal(scores['run_ids'],np.arange(1,251)) and scores['ranks'].tolist()==RANKS)
        for role in ['fault','normal']:check(role+'_score_shape_dtype_finite',scores[role].shape==(250,960,2) and scores[role].dtype==np.float64 and np.isfinite(scores[role]).all())
        ix=np.array([selection[m] for m in METHODS]);anc=inner['anchors'][ix]
        thresholds=np.sort(outer['peaks'][:,ix]/anc,axis=0)[np.array(RANKS)-1]*anc
        check('52_thresholds_exact_frozen_outer_formula',np.array_equal(thresholds,scores['thresholds']))
        throws=frame(z,'THRESHOLDS_FIXED_BEFORE_TEST.csv')
        check('52_threshold_rows_order',list(zip(throws['rank'],throws['method']))==[(r,m) for r in RANKS for m in METHODS])
        check('52_threshold_csv_exact',np.array_equal(throws['threshold'].to_numpy().reshape(26,2),thresholds))
        check('original_h_exact',np.array_equal(thresholds[RANKS.index(247)],outer['thresholds'][ix]))
        check('monotone_thresholds',np.all(np.diff(thresholds,axis=0)>=0))
        endpoints=frame(z,'RISK_TIME_ENDPOINTS.csv');fault=frame(z,'FAULT_PER_RUN.csv');normal=frame(z,'NORMAL_PER_RUN.csv');curves=frame(z,'CLEAN_DETECTION_CURVES.csv');contrasts=frame(z,'PAIRED_RANK_CONTRASTS.csv')
        check('all_output_row_counts',[len(endpoints),len(fault),len(normal),len(curves),len(contrasts)]==[52,13000,13000,41652,260])
        check('endpoint_keys_unique_complete',list(zip(endpoints['rank'],endpoints['method']))==[(r,m) for r in RANKS for m in METHODS])
        diffs={};baseline_bits=0;event_fields=0;perrun_fields=0;curve_counts=0;endpoint_fields=0
        verified_rows=[]
        for ri,rank in enumerate(RANKS):
            nm=scores['normal']>thresholds[ri];fm=scores['fault']>thresholds[ri]
            ev=events(fm);cl=ev['clean_detection'].astype(bool);delay=3*ev['first_delay_samples_penalized']
            vals=dict(normal_FAR_pct=nm.any(1).astype(float)*100,prefault_alarm_pct=ev['prefault_alarm'].astype(float)*100,occupancy_pct=ev['occupancy_points'].astype(float)/8)
            for h in [60,120,480,2400]:vals[f'clean_by_{h}min_pct']=(cl & (delay<=h)).astype(float)*100
            for h in [120,480,2400]:vals[f'loss_{h}min']=np.where(cl & (delay<=h),delay,h).astype(float)
            for k,v in vals.items():diffs.setdefault(k,[]).append(v[:,1]-v[:,0])
            if rank==247:
                bi=np.array([baseline['methods'].tolist().index(m) for m in METHODS])
                for role,mask in [('normal',nm),('fault',fm)]:
                    check('baseline_'+role+'_two_branch_bitmap_exact',np.array_equal(np.packbits(mask.transpose(2,0,1),axis=2),baseline[role+'_masks'][bi]))
                    baseline_bits+=mask.size
                for k,v in ev.items():check('baseline_events/'+k,np.array_equal(v,baseline[k][:,bi]));event_fields+=v.size
            for mi,method in enumerate(METHODS):
                f=fault[(fault['rank']==rank)&(fault.method==method)].sort_values('run_id')
                n=normal[(normal['rank']==rank)&(normal.method==method)].sort_values('run_id')
                check(f'ids/{rank}/{method}',np.array_equal(f.run_id,np.arange(1,251)) and np.array_equal(n.run_id,np.arange(1,251)))
                for k,v in ev.items():check(f'fault/{rank}/{method}/{k}',np.array_equal(f[k],v[:,mi]));perrun_fields+=250
                for h in [120,480,2400]:check(f'fault_loss/{rank}/{method}/{h}',np.array_equal(f[f'loss_{h}min'],vals[f'loss_{h}min'][:,mi]));perrun_fields+=250
                check(f'normal/{rank}/{method}',np.array_equal(n.alarm,nm[:,:,mi].any(1)) and np.array_equal(n.above_points,nm[:,:,mi].sum(1)));perrun_fields+=500
                row=endpoints[(endpoints['rank']==rank)&(endpoints.method==method)].iloc[0]
                check(f'constants/{rank}/{method}',row['normal_n']==250 and row['fault_n']==250 and row['k']==.05 and row['threshold']==thresholds[ri,mi])
                close(f'reference_probability/{rank}/{method}',row.rank_reference_probability,(260-rank)/260)
                for k,v in vals.items():
                    close(f'endpoint/{rank}/{method}/{k}',row[k],v[:,mi].mean());endpoint_fields+=1
                    if k.endswith('_pct') and k!='occupancy_pct':
                        ct=int(np.count_nonzero(v[:,mi]));lo,hi=ci(ct)
                        check(f'count/{rank}/{method}/{k}',row[k+'_count']==ct)
                        close(f'binomial/{rank}/{method}/{k}',[row[k+'_CP95_low'],row[k+'_CP95_high']],np.array([lo,hi])*100,atol=2e-9);endpoint_fields+=3
                cur=curves[(curves['rank']==rank)&(curves.method==method)].sort_values('minutes_after_onset')
                tt=np.arange(0,2401,3);ct=(cl[:,mi,None] & (delay[:,mi,None]<=tt[None,:])).sum(0)
                check(f'curves/{rank}/{method}',np.array_equal(cur.minutes_after_onset,tt) and np.array_equal(cur.clean_count,ct) and np.all(cur.n==250))
                close(f'curve_percent/{rank}/{method}',cur.clean_pct,ct*.4);curve_counts+=len(tt)
                verified_rows.append(dict(rank=rank,method=method,**{k:float(v[:,mi].mean()) for k,v in vals.items()}))
        bootstrap_max_error=0.;seed_spec=json.loads(z.read('BOOTSTRAP_SPEC.json'))
        # Reconstruct entire resampling stream without reading stored intervals as inputs.
        for role,keys in [('normal',['normal_FAR_pct']),('fault',[k for k in diffs if k!='normal_FAR_pct'])]:
            mat=np.concatenate([np.stack(diffs[k],axis=1) for k in keys],axis=1)
            seed=int.from_bytes(hashlib.sha256((protocol['id']+'|'+role).encode('ascii')).digest()[:8],'big')
            check('bootstrap_seed/'+role,seed==seed_spec['seeds'][role])
            rng=np.random.Generator(np.random.PCG64(seed));draws=np.empty((20000,mat.shape[1]))
            for start in range(0,20000,200):
                indices=rng.integers(0,250,size=(200,250),dtype=np.int64)
                weights=np.bincount((indices+250*np.arange(200)[:,None]).ravel(),minlength=200*250).reshape(200,250)
                draws[start:start+200]=weights@mat/250
            qu=np.quantile(draws,[.025,.975],axis=0,method='linear')
            for ki,k in enumerate(keys):
                sub=contrasts[contrasts.metric==k].sort_values('rank')
                check('contrast_ranks/'+k,sub['rank'].tolist()==RANKS)
                segment=slice(ki*26,(ki+1)*26)
                close('contrast_means/'+k,sub.estimate,mat[:,segment].mean(0))
                close('bootstrap_intervals/'+k,sub[['pointwise95_low','pointwise95_high']].to_numpy(),qu[:,segment].T,atol=1e-8)
                bootstrap_max_error=max(bootstrap_max_error,float(np.max(np.abs(sub[['pointwise95_low','pointwise95_high']].to_numpy()-qu[:,segment].T))))
                deg=np.all(mat[:,segment]==mat[0,segment],axis=0)
                check('interval_flags/'+k,np.array_equal(sub.empirically_degenerate,deg) and np.all(sub.bootstrap_draws==20000) and sub.same_calibration_rank_not_equal_true_FAR.all() and sub.exploratory_not_simultaneous.all())
        pd.DataFrame(verified_rows).to_csv(args.output/'VERIFIED_ENDPOINTS.csv',index=False)
        summary=dict(status='PASS_S20_SAVED_SCORE_REAGGREGATION',
          checks_passed=len(CHECKS),checks_failed=0,thresholds_reconstructed=52,
          method_rank_rows=52,paired_contrast_rows=260,
          baseline_alarm_bits_directly_compared=baseline_bits,
          baseline_event_fields_directly_compared=event_fields,
          per_run_value_fields_recomputed=perrun_fields,
          curve_counts_recomputed=curve_counts,
          endpoint_numeric_fields_recomputed=endpoint_fields,
          bootstrap_means_recomputed=20000*260,
          bootstrap_interval_max_abs_error=bootstrap_max_error,
          boundary='Saved continuous-score replay only. Not raw-process replay, simulator validation, repeated training, equal-FAR inference or threshold selection.',
          python=platform.python_version(),numpy=np.__version__)
        (args.output/'VERIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
        (args.output/'CHECKS_DETAIL.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        (args.output/'FAILURE.json').write_text(json.dumps(dict(status='FAIL',error_type=type(exc).__name__,error=str(exc),checks=CHECKS),indent=2)+'\n')
        raise

if __name__=='__main__':main()
