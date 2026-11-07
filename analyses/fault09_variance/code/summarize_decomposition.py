"""Pool archived run statistics and run paired, whole-run bootstrap inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from compute_decomposition import FEATURES, FLOW, sha_file


def write_csv(p, rows):
    with Path(p).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(root):
    root = Path(root)
    report = json.loads((root/'results/RECONSTRUCTION_AUDIT.json').read_text())
    if report['status'] != 'PASS' or report['n_runs'] != 400:
        raise ValueError('Incomplete input audit')
    if report['compute_script_sha256'] != sha_file(Path(__file__).with_name('compute_decomposition.py')):
        raise ValueError('Computation code changed')
    names = [x['run_id'] for x in report['runs']]
    zmeans, zcross, vmeans, vcross, echanges = [], [], [], [], []
    sigmax, mmax = [], []
    for run in report['runs']:
        p = root/'checkpoints'/(run['run_id']+'.npz')
        if sha_file(p) != run['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        with np.load(p, allow_pickle=False) as x:
            assert x['feature_names'].tolist() == FEATURES
            assert str(x['run_id']) == run['run_id'] and int(x['n']) == 960
            zmeans.append(x['means']); zcross.append(x['cross'])
            vmeans.append(x['variable_means']); vcross.append(x['variable_cross'])
            echanges.append(x['event_changed_n']); sigmax.append(x['channel_signal_diff_max'])
            mmax.append(x['channel_measurement_diff_max']); sd = x['noise_sd']
    mu, within, vm, vw = map(np.asarray, [zmeans, zcross, vmeans, vcross])
    n, r, total = 960, 400, 384000
    avg = mu.mean(axis=0)
    centered_mu = mu - avg
    total_cross = within.sum(axis=0) + n * centered_mu.T @ centered_mu
    covariance = total_cross / (total-1)
    va = vm.mean(axis=0)
    vd = vm - va
    vcov = (vw.sum(axis=0) + n*np.einsum('rji,rjk->jik',vd,vd))/total
    np.savez_compressed(root/'results/PER_RUN_SUFFICIENT_STATISTICS.npz',
        run_ids=np.asarray(names), feature_names=np.asarray(FEATURES), n_per_run=np.asarray(n),
        score_means=mu, score_within_cross=within,
        variable_means=vm, variable_within_cross=vw, noise_sd=sd,
        event_changed_n=np.asarray(echanges), channel_signal_diff_max=np.asarray(sigmax),
        channel_measurement_diff_max=np.asarray(mmax))
    p = len(FEATURES)
    eye = np.eye(p)
    at = {name: eye[i] for i,name in enumerate(FEATURES)}
    def var(a):
        a = at[a] if isinstance(a,str) else a
        return np.outer(a,a)
    def twice_cov(a,b):
        a = at[a] if isinstance(a,str) else a
        b = at[b] if isinstance(b,str) else b
        return np.outer(a,b)+np.outer(b,a)
    metrics=[]
    def add(name,q,category):
        metrics.append((name,q,category))
    for name in FEATURES[:13]:
        add('variance_'+name,var(name),'score_variance')
    add('post_minus_rho0',var('post_update')-var('rho0'),'paired_difference')
    add('post_minus_CI',var('post_update')-var('controller_input'),'paired_difference')
    for name in FEATURES[3:9]:
        add('reduction_'+name,var('post_update')-var(name),'paired_difference')
    add('substitution_interaction', var('post_all41_to_CI')-var('post_continuous_to_CI')-var('post_analyzers_to_CI')+var('post_update'),'paired_difference')
    add('common_noise_signal_change',var('bridge_all')-var('rho0'),'common_noise_bridge')
    add('common_noise_final_noise_change',var('post_update')-var('bridge_all'),'common_noise_bridge')
    for group in ['C','A','U']:
        add('within_'+group,var('post_group_'+group)-var('rho0_group_'+group),'endpoint_covariance_blocks')
    for ga,gb in [('C','A'),('C','U'),('A','U')]:
        add('cross_'+ga+'_'+gb,twice_cov('post_group_'+ga,'post_group_'+gb)-twice_cov('rho0_group_'+ga,'rho0_group_'+gb),'endpoint_covariance_blocks')
    tj=[at['signal_delta_XMEAS17'],at['signal_delta_continuous_except17'],at['signal_delta_analyzers']]
    t=sum(tj)
    e=at['post_noise_score']-at['independent_noise_score']
    for name,q in [('variance_T',var(t)),('twice_cov_s0_T',twice_cov('rho0',t)),
                   ('variance_E',var(e)),('twice_cov_s0_E',twice_cov('rho0',e)),('twice_cov_T_E',twice_cov(t,e))]:
        add(name,q,'exact_five_terms')
    for name,d in zip(['XMEAS17','other_continuous','analyzers'],tj):
        add('signal_allocation_'+name,twice_cov('rho0',d)+twice_cov(d,t)/2,'signal_covariance_allocation')
    add('noise15_variance',var('CI_noise15_standardized'),'association')
    add('signal17_variance',var('signal_delta17_raw'),'association')
    add('noise15_signal17_covariance',twice_cov('CI_noise15_standardized','signal_delta17_raw')/2,'association')
    mats=np.asarray([x[1] for x in metrics])
    est=np.einsum('ij,kij->k',covariance,mats)
    h=np.einsum('rij,kij->rk',within,mats)+n*np.einsum('ri,kij,rj->rk',centered_mu,mats,centered_mu,optimize=True)
    rng=np.random.default_rng(2026110301)
    weights=rng.multinomial(r,np.full(r,1/r),size=20000)
    bmeans=weights@centered_mu/r
    boot=(weights@h-total*np.einsum('bi,kij,bj->bk',bmeans,mats,bmeans,optimize=True))/(total-1)
    limits=np.quantile(boot,[.025,.975],axis=0,method='linear')
    metrics_rows=[]
    for j,(name,q,category) in enumerate(metrics):
        metrics_rows.append({'category':category,'metric':name,'estimate':float(est[j]),
                             'ci95_low':float(limits[0,j]),'ci95_high':float(limits[1,j]),
                             'bootstrap_unit':'whole_paired_run','draws':20000,'seed':2026110301,
                             'interval_scope':'marginal_descriptive_conditional_on_frozen_model_and_noise_realization'})
    write_csv(root/'results/SCORE_VARIANCES_AND_DECOMPOSITIONS.csv',metrics_rows)
    for category,fn in [('score_variance','GROUP_SUBSTITUTION_AND_BRIDGE.csv'),
                        ('endpoint_covariance_blocks','EXACT_ENDPOINT_BLOCKS.csv'),
                        ('exact_five_terms','EXACT_TIMING_NOISE_TERMS.csv'),
                        ('signal_covariance_allocation','SIGNAL_CHANGE_ALLOCATION.csv')]:
        write_csv(root/'results'/fn,[x for x in metrics_rows if x['category']==category])
    lookup={x['metric']:x for x in metrics_rows}
    metric_pos={x[0]:i for i,x in enumerate(metrics)}
    # One shared bootstrap for the signal variance of all41 channels, ddof0.
    vh=vw[:,:,2,2]+n*vd[:,:,2]**2
    vbmean=weights@vd[:,:,2]/r
    vb=(weights@vh-total*vbmean**2)/total
    vci=np.quantile(vb,[.025,.975],axis=0,method='linear')
    variables=[]
    labels={1:'A feed',2:'D feed',3:'E feed',4:'A and C feed',5:'Recycle flow',6:'Reactor feed rate',
            10:'Purge rate',12:'Separator liquid level',14:'Separator underflow',15:'Stripper liquid level',
            17:'Stripper underflow',19:'Stripper steam flow'}
    for j in range(41):
        vtot,vnoise,vsignal=vcov[j,0,0],vcov[j,1,1],vcov[j,2,2]
        cov2=2*vcov[j,1,2]
        entry={'variable':f'XMEAS{j+1}','description':labels.get(j+1,''),
               'group':'continuous' if j<22 else 'analyzer', 'noise_sd':float(sd[j]),
               'variance_diff_original_units':float(vtot*sd[j]**2),
               'variance_noise_original_units':float(vnoise*sd[j]**2),
               'variance_signal_original_units':float(vsignal*sd[j]**2),
               'twice_cov_original_units':float(cov2*sd[j]**2),
               'variance_diff_noise_normalized':float(vtot),
               'variance_noise_normalized':float(vnoise),'variance_signal_normalized':float(vsignal),
               'twice_cov_normalized':float(cov2),
               'signal_variance_ci95_low_normalized':float(vci[0,j]),
               'signal_variance_ci95_high_normalized':float(vci[1,j]),
               'noise_share_percent':float(100*vnoise/vtot) if vtot>0 else '',
               'signal_share_percent':float(100*vsignal/vtot) if vtot>0 else '',
               'twice_cov_share_percent':float(100*cov2/vtot) if vtot>0 else '',
               'signal_diff_sd_original_units':float(np.sqrt(vsignal)*sd[j]),
               'measurement_diff_max_abs':float(np.max(np.asarray(mmax)[:,j])),
               'signal_diff_max_abs':float(np.max(np.asarray(sigmax)[:,j])),
               'noise_event_changed_observations':int(np.sum(echanges,axis=0)[j]),
               'observations':total,'ddof':0,
               'identity_error_normalized':float(vtot-vnoise-vsignal-cov2)}
        variables.append(entry)
    write_csv(root/'results/ALL41_MEASUREMENT_DIFFERENCE_DECOMPOSITION.csv',variables)
    write_csv(root/'results/FLOWS_AND_LEVELS_DIFFERENCE_DECOMPOSITION.csv',
              [x for i,x in enumerate(variables) if i in set(FLOW)|{11,14}])
    xi=metric_pos['noise15_variance']; yi=metric_pos['signal17_variance']; ci=metric_pos['noise15_signal17_covariance']
    corr=est[ci]/np.sqrt(est[xi]*est[yi])
    bcorr=boot[:,ci]/np.sqrt(boot[:,xi]*boot[:,yi])
    slope=est[ci]/est[xi]
    bslope=boot[:,ci]/boot[:,xi]
    assoc={'correlation':float(corr),'correlation_ci95':np.quantile(bcorr,[.025,.975]).tolist(),
           'slope_raw_signal17_per_one_SD_CI_noise15':float(slope),
           'slope_ci95':np.quantile(bslope,[.025,.975]).tolist(),
           'interpretation':'Descriptive association conditional on original closed-loop trajectories; not an isolated controller intervention.'}
    write_json(root/'results/CONTROL_NOISE_ASSOCIATION.json',assoc)
    sums={cat:float(sum(x['estimate'] for x in metrics_rows if x['category']==cat))
          for cat in ['endpoint_covariance_blocks','exact_five_terms','signal_covariance_allocation']}
    checks={
        'endpoint_block_sum_error':sums['endpoint_covariance_blocks']-lookup['post_minus_rho0']['estimate'],
        'five_term_sum_error':sums['exact_five_terms']-lookup['post_minus_rho0']['estimate'],
        'signal_allocation_sum_error':sums['signal_covariance_allocation']-lookup['common_noise_signal_change']['estimate'],
        'bridge_telescoping_error':lookup['common_noise_signal_change']['estimate']+lookup['common_noise_final_noise_change']['estimate']-lookup['post_minus_rho0']['estimate'],
        'max_variable_variance_identity_error':max(abs(x['identity_error_normalized']) for x in variables),
        'CI_all41_substitution_variance_error':lookup['variance_controller_input']['estimate']-lookup['variance_post_all41_to_CI']['estimate']}
    if any(abs(v)>2e-8 for v in checks.values()):
        raise ValueError('Aggregation identity failed '+str(checks))
    summary={'status':'COMPLETE','study_id':'TEP_P1_F9_RECORDING_VARIANCE_DECOMPOSITION_20261103_v1',
             'n_normal_runs':r,'observations_per_run':n,'score_variance_ddof':1,'measurement_variance_ddof':0,
             'metrics':metrics_rows,'key_variables':[variables[j] for j in [11,14,16]],
             'association':assoc,'identity_checks':checks,
             'all_analyzers_identical':report['analyzer_changed_cells']==0,
             'all_analyzer_events_identical':report['analyzer_event_changed_cells']==0,
             'all_XMV_identical':True,'no_new_simulation':True,
             'bootstrap_method':'20000 shared multinomial count vectors of400 whole physical runs; pooled covariance includes run-mean differences',
             'scope':'Post hoc description of previously analyzed400N panel; fixed historical model, no refit or new performance claims'}
    write_json(root/'results/SUMMARY.json',summary)
    np.savez_compressed(root/'results/BOOTSTRAP_METRICS.npz',metric_names=np.asarray([x[0] for x in metrics]),
                        estimates=est,draws=boot,bootstrap_weights=weights.astype(np.uint16),
                        seed=np.asarray(2026110301),run_ids=np.asarray(names))
    perrun=[]
    for i,run in enumerate(names):
        for j,name in enumerate(FEATURES[:13]):
            perrun.append({'run_id':run,'arm':name,'mean':float(mu[i,j]),'variance_within_ddof1':float(within[i,j,j]/959)})
    write_csv(root/'results/PER_RUN_SCORE_SUMMARY.csv',perrun)
    print(json.dumps({'status':'COMPLETE','variances':{x['metric']:x['estimate'] for x in metrics_rows if x['category']=='score_variance'},
                      'bridge':{x['metric']:x['estimate'] for x in metrics_rows if x['category']=='common_noise_bridge'},
                      'blocks':{x['metric']:x['estimate'] for x in metrics_rows if x['category']=='endpoint_covariance_blocks'},
                      'association':assoc,'checks':checks},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True)
    main(ap.parse_args().root)
