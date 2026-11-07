"""Replay the archived S57 validation and standard-monitor source results.

Adapted from TEP_A459_COMPLETED_20261009/code/replay_completion.py by
selecting its A4 branches (DV, R5 and standard) and removing the A5 branch.
Endpoint recovery and statistical algorithms are unchanged; the original
CODE/a4_core.py supplies the source-exact 20,000-draw paired bootstrap.

Reads saved maxima, calibration statistics, identities, and archived CSVs.
No simulations, fitting, downloads, installation, or raw-trajectory reading.
This covers the DV/R5 validation and standard-monitor branches only, not every
row of S57: fixed historical and relearned recording-monitor branches are
outside this entry point. Original manuscript intervals are not overwritten.

Requires Python 3.10+ and NumPy. Recorded host versions were Python 3.12.3
and NumPy 2.2.6. --out must name a new directory outside the supplied inputs.
"""
# Original replay_completion.py SHA-256: 7f617dfc2a7c2d121dc038ba8281e586f90ecd196efdfa7f47eaa0696c490eda
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

SCENARIOS=('original','exclude_test','exclude_test_and_calibration')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        keys=list(dict.fromkeys(k for r in rs for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(n):return (95*(int(n)+1)+99)//100

def recover_events(premax,cumulative,h):
    """Cumulative maxima retain the first crossing, not sample-level occupancy."""
    pre=premax>h
    post=cumulative>h
    det=post[:,-1,:]
    first=post.argmax(axis=1)
    delay=3*first
    clean=~pre&det
    return dict(pre=pre,detected=det,clean40=clean,clean8=clean&(delay<=480),clean2=clean&(delay<=120),loss=np.where(clean,delay,2400),any_alarm=pre|det)

def same_rows(actual,expected,label,tol=1e-10):
    assert len(actual)==len(expected),(label,len(actual),len(expected))
    maxerr=0.0
    for i,(a,b) in enumerate(zip(actual,expected)):
        assert set(a)==set(b),(label,i,set(a)^set(b))
        for k,v in a.items():
            if isinstance(v,(int,float,np.number)):
                err=abs(float(v)-float(b[k]));maxerr=max(maxerr,err)
                assert err<=tol,(label,i,k,v,b[k])
            else:assert str(v)==b[k],(label,i,k,v,b[k])
    return {'table':label,'rows':len(actual),'max_abs_numeric_error':maxerr,'status':'PASS'}

def verify_seeds(d,z):
    cal=read(d/'CALIBRATION_IDENTITIES.csv')
    seedkey='seed_u32' if 'seed_u32' in cal[0] else 'seed'
    ck=np.array([int(x[seedkey])%32!=0 for x in cal]);np.testing.assert_array_equal(ck,z['calibration_keep'])
    for x,k in zip(cal,ck):assert (x['keep'].lower()=='true')==bool(k)
    if (d/'TEST_IDENTITIES.csv').exists():
        tr=read(d/'TEST_IDENTITIES.csv')
        for role,key in [('fault_evaluation','fault_keep'),('normal_evaluation','normal_keep')]:
            rr=[x for x in tr if x['role']==role]
            assert [int(x['run']) for x in rr]==list(range(1,len(rr)+1))
            keep=np.array([int(x['seed_u32'])%32!=0 for x in rr]);np.testing.assert_array_equal(keep,z[key])
            for x,k in zip(rr,keep):assert (x['keep'].lower()=='true')==bool(k)
    else:
        tr=read(d/'RAW_INPUT_IDENTITIES.csv')
        for role,key in [('fault9_evaluation','fault_keep'),('normal_evaluation','normal_keep')]:
            rr=[x for x in tr if x['role']==role]
            np.testing.assert_array_equal(np.array([int(x['seed'])%32!=0 for x in rr]),z[key])
    return {'calibration_n_original':len(cal),'calibration_n_kept':int(ck.sum()),'new_rank':rank(ck.sum()),'fault_n_original':len(z['fault_keep']),'fault_n_kept':int(z['fault_keep'].sum()),'normal_n_original':len(z['normal_keep']),'normal_n_kept':int(z['normal_keep'].sum())}

def recover_unit(d,out):
    z=load(d/'SUFFICIENT_PATHS.npz');old=read(d/'ENDPOINTS.csv');panel=old[0]['panel'];state=int(old[0]['state']);fault=int(old[0]['fault'])
    names=[str(x) for x in z['methods']];meta=verify_seeds(d,z)
    pk=z['calibration_peaks'];an=z['anchors']
    assert np.isfinite(pk).all() and np.isfinite(an).all() and (an>0).all()
    oldh=an*np.sort(pk/an,axis=0)[rank(len(pk))-1]
    newh=an*np.sort(pk[z['calibration_keep']]/an,axis=0)[rank(meta['calibration_n_kept'])-1]
    np.testing.assert_allclose(oldh,z['old_thresholds'],rtol=0,atol=1e-8)
    np.testing.assert_array_equal(newh,z['new_thresholds'])
    for key in ('fault','normal'):
        c=z[key+'_post_running_max'];assert c.shape==(meta[key+'_n_original'],800,len(names))
        # Inf placeholders used by lagged statistics can be -inf before enough lagged points.
        assert not np.isnan(c).any() and (c[:,1:,:]>=c[:,:-1,:]).all()
    rows=[];evs=[]
    for sc in SCENARIOS:
        h=z['new_thresholds'] if sc==SCENARIOS[2] else z['old_thresholds']
        fk=z['fault_keep'] if sc!='original' else np.ones(meta['fault_n_original'],bool)
        nk=z['normal_keep'] if sc!='original' else np.ones(meta['normal_n_original'],bool)
        fe=recover_events(z['fault_pre_max'],z['fault_post_running_max'],h)
        ne=recover_events(z['normal_pre_max'],z['normal_post_running_max'],h)
        for j,name in enumerate(names):
            rows.append(dict(panel=panel,state=state,fault=fault,scenario=sc,method=name,fault_n=int(fk.sum()),normal_n=int(nk.sum()),threshold=float(h[j]),clean40_n=int(fe['clean40'][fk,j].sum()),clean40_pct=float(fe['clean40'][fk,j].mean()*100),clean8_n=int(fe['clean8'][fk,j].sum()),clean8_pct=float(fe['clean8'][fk,j].mean()*100),clean2_pct=float(fe['clean2'][fk,j].mean()*100),loss_min=float(fe['loss'][fk,j].mean()),normal_alarm_n=int(ne['any_alarm'][nk,j].sum()),normal_FAR_pct=float(ne['any_alarm'][nk,j].mean()*100),normal_floor_pct=float(ne['clean40'][nk,j].mean()*100),prefault_alarm_pct=float(fe['pre'][fk,j].mean()*100)))
            for role,ee,kk in [('fault',fe,fk),('normal',ne,nk)]:
                for i in np.flatnonzero(kk):
                    evs.append(dict(panel=panel,state=state,fault=fault,scenario=sc,method=name,role=role,run_id=int(i+1),**{k:int(v[i,j]) for k,v in ee.items()}))
    checks=[same_rows(rows,old,str(d/'ENDPOINTS.csv')),same_rows(evs,read(d/'EVENTS.csv'),str(d/'EVENTS.csv'))]
    write(out/'ENDPOINTS.csv',rows);write(out/'EVENTS.csv',evs)
    thresholds=[dict(panel=panel,state=state,fault=fault,method=name,old_threshold=float(z['old_thresholds'][j]),new_threshold=float(newh[j]),**meta) for j,name in enumerate(names)]
    return rows,evs,checks,thresholds

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-root',type=Path,
                        default=Path(__file__).resolve().parent/'final_validation',
                        help='Directory containing CODE/ and results/ (default: adjacent final_validation/)')
    parser.add_argument('--out',type=Path,required=True,
                        help='New output directory outside the supplied inputs')
    args=parser.parse_args()
    if sys.flags.optimize:
        parser.error('Run without -O or -OO so all source verification assertions remain enabled.')
    source=args.return_root.expanduser().resolve()
    out=args.out.expanduser().resolve()
    if out==source or source in out.parents:
        parser.error('--out must be outside --return-root to preserve supplied source files.')
    if out.exists():
        parser.error('--out already exists; select a new directory.')
    expected_units={'DV':3,'R5':15,'standard':2}
    units={}
    for filename in ('a4_core.py','historical_core.py'):
        if not (source/'CODE'/filename).is_file():
            parser.error('Missing original shared module: CODE/'+filename)
    for task,expected in expected_units.items():
        src=source/'results'/task
        units[task]=sorted(src.glob('*/SUFFICIENT_PATHS.npz'))
        if task in ('DV','R5'):
            for filename in ('STATE_ENDPOINTS.csv','MEAN_ENDPOINTS.csv','C1_C2_C3_PAIRED_983333.csv','CHANGES_FROM_ORIGINAL.csv'):
                if not (src/filename).is_file():
                    parser.error('Missing archived aggregate table: '+str((src/filename).relative_to(source)))
        if len(units[task])!=expected:
            parser.error(f'{task}: expected {expected} saved units, found {len(units[task])}.')
        for unit in units[task]:
            for filename in ('ENDPOINTS.csv','EVENTS.csv','CALIBRATION_IDENTITIES.csv'):
                if not (unit.parent/filename).is_file():
                    parser.error('Missing required input: '+str((unit.parent/filename).relative_to(source)))
            identity='RAW_INPUT_IDENTITIES.csv' if task=='standard' else 'TEST_IDENTITIES.csv'
            if not (unit.parent/identity).is_file():
                parser.error('Missing required identity table: '+str((unit.parent/identity).relative_to(source)))
    # These modules are copied unchanged from the verified return.
    sys.path.insert(0,str(source/'CODE'))
    import a4_core
    out.mkdir(parents=True,exist_ok=False)
    checks=[];thresholds=[];counts={};csv_equal=[]
    for task in ('DV','R5','standard'):
        src=source/'results'/task;dst=out/task;dst.mkdir()
        rows=[];event_rows=[]
        for unit in units[task]:
            r,e,c,t=recover_unit(unit.parent,dst/unit.parent.name)
            rows.extend(r);event_rows.extend(e);checks.extend(c);thresholds.extend(t)
        counts[task]={'endpoint_rows':len(rows),'event_rows':len(event_rows),'units':len(units[task])}
        if task in ('DV','R5'):
            a4_core.aggregate(rows,event_rows,dst)
            for filename in src.glob('*.csv'):
                generated=dst/filename.name
                if generated.exists():
                    assert generated.read_bytes()==filename.read_bytes(),('source-exact aggregate differs',task,filename.name)
                    csv_equal.append(str(filename.relative_to(source)))
        else:
            write(dst/'ALL_STANDARD_ENDPOINTS.csv',rows)
        for filename in src.glob('*/ENDPOINTS.csv'):
            assert (dst/filename.parent.name/filename.name).read_bytes()==filename.read_bytes()
            csv_equal.append(str(filename.relative_to(source)))
    write(out/'RECALIBRATED_THRESHOLDS.csv',thresholds)
    write(out/'RECOVERY_TABLE_CHECKS.csv',checks)
    receipt={
        'status':'PASS',
        'scope':'S57 DV/R5 validation and standard-monitor branches only',
        'new_simulations':0,'new_fits':0,
        'A4_reconstructed_counts':counts,
        'A4_thresholds_recomputed':len(thresholds),
        'A4_event_rows_recovered':sum(x['event_rows'] for x in counts.values()),
        'host_CSVs_reproduced_byte_identically':len(csv_equal),
        'exact_CSVs':csv_equal,
        'original_raw52_reread_in_this_replay':False,
        'runtime':{'python':sys.version,'numpy':np.__version__},
        'shared_code_sha256':{name:digest(source/'CODE'/name) for name in ('a4_core.py','historical_core.py')},
        'notes':[
            'Pre-onset maxima and post-onset running maxima preserve first crossing and episode metrics, not sample-level occupancy.',
            '20,000-draw bootstrap schedule is copied exactly from the verified host code.',
            'Source-exact re-bootstrap original rows are checks, not replacements for original manuscript confidence intervals.',
            'Fixed historical and relearned recording-monitor branches are outside this entry point.'
        ]
    }
    (out/'REPLAY_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
