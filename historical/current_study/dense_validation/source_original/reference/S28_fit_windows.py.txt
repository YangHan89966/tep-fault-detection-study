"""S22 fixed fit-only windows. No evaluation IDs/outcomes are accepted."""
from __future__ import annotations
import numpy as np

def endpoint_means(y, run_ids, starts, suffixes, policy, fault):
    if y.ndim!=3 or y.shape[:2]!=(len(run_ids),960):raise ValueError('Fit input shape')
    if len(set(map(int,run_ids)))!=len(run_ids):raise ValueError('Duplicate fit IDs')
    if not (len(starts)==len(suffixes)==len(run_ids)):raise ValueError('Fit marker lengths')
    if not np.array_equal(961-np.asarray(starts),suffixes):raise ValueError('Suffix identity')
    threshold=160 if policy=='pre_marker_160' else 2
    rows=[];early=[];late=[];all_default=True
    for i,rid in enumerate(run_ids):
        b=int(starts[i]);length=int(suffixes[i]);mark=int(length>=threshold)
        if policy=='legacy_endpoints':lo1,hi1,lo2,hi2=160,320,800,960;effective_end=960
        else:
            effective_end=b-1 if mark else 960
            available=effective_end-160
            if available<=0:raise ValueError(f'No premarker fault data fault{fault} run{rid}; do not omit')
            q=min(160,available);lo1=160;hi1=160+q
            lo2=effective_end-q;hi2=effective_end
            if policy=='early_only_pre2':lo2,hi2=lo1,hi1
            if (lo1,hi1,lo2,hi2)!=(160,320,800,960):all_default=False
        early.append(y[i,lo1:hi1].mean(axis=0));late.append(y[i,lo2:hi2].mean(axis=0))
        n1=hi1-lo1;n2=hi2-lo2
        # 0-based first marked point b-1. All exposure independent of detector.
        def exposure(lo,hi,minimum):
            return max(0,hi-max(lo,b-1)) if length>=minimum else 0
        rows.append({'fault':fault,'run_id':int(rid),'policy':policy,'record_suffix_start':b,'record_suffix_samples':length,
            'used_last_sample':max(hi1,hi2),'prefix_available_fault_points':effective_end-160,
            'early_first_sample':lo1+1,'early_last_sample':hi1,'early_points':n1,
            'late_first_sample':lo2+1,'late_last_sample':hi2,'late_points':n2,
            'overlap_points':max(0,min(hi1,hi2)-max(lo1,lo2)),
            'early_main160_tail_points':exposure(lo1,hi1,160),'late_main160_tail_points':exposure(lo2,hi2,160),
            'early_any2_tail_points':exposure(lo1,hi1,2),'late_any2_tail_points':exposure(lo2,hi2,2),
            'equal_run_weight':1./len(run_ids)})
    # Preserve historical summation order in unchanged controls.
    if policy=='legacy_endpoints' or all_default:
        a=y[:,160:320].mean(axis=(0,1));b=y[:,800:960].mean(axis=(0,1))
    else:
        a=np.stack(early).mean(axis=0);b=np.stack(late).mean(axis=0)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('Nonfinite fitted endpoint')
    return a,b,rows

def test_windows():
    rng=np.random.default_rng(2026090622);y=rng.standard_normal((5,960,3));ids=np.array([3,5,9,11,25])
    starts=np.array([318,481,818,960,400]);length=961-starts
    checks=[]
    for p in ['pre_marker_160','pre_marker_2','early_only_pre2']:
        a,b,rows=endpoint_means(y,ids,starts,length,p,18);bad=y.copy()
        cutoff=160 if p=='pre_marker_160' else 2
        for i in range(5):
            if length[i]>=cutoff:bad[i,starts[i]-1:]=np.nan
        ap,bp,_=endpoint_means(bad,ids,starts,length,p,18)
        assert np.array_equal(a,ap) and np.array_equal(b,bp)
        assert all(r['early_main160_tail_points']==r['late_main160_tail_points']==0 for r in rows)
        if p!='pre_marker_160':assert all(r['early_any2_tail_points']==r['late_any2_tail_points']==0 for r in rows)
        assert rows[0]['early_points']==157 and rows[0]['late_points']==157
        checks.append(p+'_poisoned_excluded_tail_unchanged')
    a,b,r=endpoint_means(y,ids,np.full(5,960),np.ones(5,int),'pre_marker_160',3)
    assert np.array_equal(a,y[:,160:320].mean((0,1))) and np.array_equal(b,y[:,800:960].mean((0,1)))
    checks.append('no_marker_exact_legacy_identity')
    a,b,_=endpoint_means(y,ids,starts,length,'early_only_pre2',18);assert np.array_equal(a,b);checks.append('early_only_means_identical')
    try:endpoint_means(y,ids,np.array([161,481,818,960,400]),961-np.array([161,481,818,960,400]),'pre_marker_160',18)
    except ValueError:checks.append('empty_prefix_rejected_no_run_dropped')
    else:raise AssertionError('empty prefix allowed')
    return {'status':'PASS_SYNTHETIC_LOGIC_TESTS_NOT_PROCESS_EXPERIMENT','checks':checks}
