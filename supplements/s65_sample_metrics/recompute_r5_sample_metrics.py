#!/usr/bin/env python3
"""R5 sample occupancy supplement, from archived actual alarm bits only.

This is an extension of A5 on R5 evaluation runs, not the missing historical
D01 / Braatz analysis. It neither fits a model nor changes a threshold.
Run: python recompute_r5_sample_metrics.py --out fresh_results_directory
Dependencies: Python 3.10+ and NumPy. All inputs are supplied in ./inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

B = 20000
SEED = 2026100905
T4 = {
    3: [('baseline', 'mean_signed__full', 'Full monitor: mean-shift Signed'),
        ('repeated', 'same_mean_signed_axis_squared', 'Same direction, Squared'),
        ('baseline', 'mean_signed__I', 'I restriction, Signed'),
        ('baseline', 'FDA_equal_class_signed_CUSUM', 'FDA-CUSUM, Signed'),
        ('baseline', 'Q_CUSUM', 'Q-CUSUM')],
    9: [('baseline', 'moment_signed__full', 'Full monitor: second-moment Signed'),
        ('repeated', 'mean_pool_signed', 'Mean-shift pool, Signed'),
        ('baseline', 'moment_signed__I', 'I restriction, Signed'),
        ('baseline', 'FDA_equal_class_signed_CUSUM', 'FDA-CUSUM, Signed'),
        ('baseline', 'Q_CUSUM', 'Q-CUSUM')],
    15: [('baseline', 'moment_squared__full', 'Full monitor: second-moment Squared'),
         ('repeated', 'mean_pool_squared', 'Mean-shift pool, Squared'),
         ('baseline', 'moment_squared__I', 'I restriction, Squared'),
         ('baseline', 'covariance_F15_max_squared_CUSUM', 'Variance-ratio reference, Squared'),
         ('baseline', 'FDA_equal_class_signed_CUSUM', 'FDA-CUSUM, Signed'),
         ('baseline', 'Q_CUSUM', 'Q-CUSUM')],
}
CONTRASTS = [
    (3, 'C1_Squared_minus_Signed', 1, 0),
    (9, 'C2_moment_minus_mean_Signed', 0, 1),
    (15, 'C3_moment_minus_mean_Squared', 0, 1),
    (3, 'Full_minus_FDA', 0, 3),
    (9, 'Full_minus_FDA', 0, 3),
    (15, 'Full_minus_variance_ratio', 0, 3),
]
FAULT_METRICS = ['postfault_sample_FDR_pct', 'prefault_sample_FAR_pct',
                 'clean40_pct', 'clean8_pct', 'mean_loss_min',
                 'counterfactual_latched_FDR_pct', 'latching_excess_FDR_pp']
NORMAL_METRICS = ['normal_sample_FAR_pct', 'normal_any_alarm_pct',
                  'normal_detection_floor_pct']


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_csv(path, rows):
    assert rows, path
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def events(mask):
    """A byte is an actual S_t > h flag; no permanent alarm latch is imposed."""
    assert mask.shape == (250, 960) and mask.dtype == bool
    pre = mask[:, :160]
    post = mask[:, 160:]
    pre_any = pre.any(1)
    post_any = post.any(1)
    clean = (~pre_any) & post_any
    delay = np.where(post_any, post.argmax(1) * 3, 2400)
    post_latched = np.maximum.accumulate(post, axis=1)
    # Counterfactual full-run latching is a diagnostic, never used as actual FDR.
    all_latched = np.maximum.accumulate(mask, axis=1)
    post_above = post.sum(1)
    post_below_after_first = (post_latched & ~post).sum(1)
    return dict(pre_above=pre.sum(1), post_above=post_above,
                all_above=mask.sum(1), pre_any=pre_any, post_any=post_any,
                any=mask.any(1), clean=clean,
                clean8=clean & (delay <= 480),
                loss=np.where(clean, delay, 2400),
                post_below_after_first=post_below_after_first,
                clean_returns_below=clean & (post_below_after_first > 0),
                latched_post_above=all_latched[:, 160:].sum(1))


def metric_vectors(e, role):
    if role == 'fault':
        fdr = e['post_above'] / 8.0
        latched = e['latched_post_above'] / 8.0
        return np.column_stack([fdr, e['pre_above'] / 1.6,
                                e['clean'] * 100., e['clean8'] * 100.,
                                e['loss'], latched, latched - fdr])
    return np.column_stack([e['all_above'] / 9.6,
                            e['any'] * 100., e['clean'] * 100.])


def bootstrap(vectors, *, role, fault, scenario):
    """One row = one physical run, averaged over five fixed states first."""
    assert vectors.ndim == 2 and np.isfinite(vectors).all()
    # A common seed preserves pairing across compared methods and all states.
    # Normal and fault cohorts use separate resamples; no pairing is asserted.
    role_code = 1 if role == 'fault' else 2
    seed_fault = fault if role == 'fault' else 0
    seed_scenario = 0 if scenario == 'original' else 1
    entropy = [SEED, role_code, seed_fault, seed_scenario]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    means = np.empty((B, vectors.shape[1]))
    h = hashlib.sha256()
    for start in range(0, B, 500):
        ii = rng.integers(0, len(vectors), (min(500, B-start), len(vectors)), dtype=np.int32)
        h.update(ii.astype('<i4', copy=False).tobytes())
        means[start:start+len(ii)] = vectors[ii].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    return lo, hi, entropy, h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, default=Path(__file__).resolve().parent/'inputs')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.inputs.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bindings = json.loads((root/'SOURCE_BINDINGS.json').read_text())
    inputs_read = []
    for item in bindings['supplied_inputs']:
        p = root/item['portable_path']
        blob = p.read_bytes()
        assert len(blob) == item['size_bytes'] and sha(blob) == item['sha256'], p
        inputs_read.append({k:item[k] for k in ['portable_path', 'size_bytes', 'sha256']})
    original_manifest = {x['path']:x for x in json.loads((root/'ORIGINAL_ALARM_INPUT_MANIFEST.json').read_text())}
    identities = read_csv(root/'R5_EVALUATION_IDENTITIES.csv')
    assert len(identities) == 1000
    cohorts = {}
    for fault in [0, 3, 9, 15]:
        role = 'normal_evaluation' if fault == 0 else 'fault_evaluation'
        rs = [x for x in identities if x['role'] == role and int(x['fault_id']) == fault]
        rs.sort(key=lambda x:int(x['run_number']))
        assert [int(x['run_number']) for x in rs] == list(range(1,251))
        assert all(int(x['replication']) == 0 for x in rs)
        cohorts[fault] = rs
    cache = {}
    verification = []
    checked_scalars = 0
    checked_bits = 0
    for fault in [3, 9, 15]:
        for state in range(1,6):
            for family in ['repeated', 'baseline']:
                relative = f'alarms/{family}/rep{state:02d}/F{fault:02d}.npz'
                p = root/relative
                original = original_manifest[f'inputs/{family}/rep{state:02d}/F{fault:02d}.npz']
                blob = p.read_bytes()
                assert len(blob) == original['size_bytes'] and sha(blob) == original['sha256']
                with np.load(p, allow_pickle=False) as z:
                    a = {k:z[k] for k in z.files}
                assert np.array_equal(a['run_id'], np.arange(1,251))
                masks = {}
                for role in ['normal', 'fault']:
                    packed = a[f'{role}_alarm_bits']
                    masks[role] = np.unpackbits(packed, axis=1, count=960, bitorder='little').astype(bool)
                    assert np.array_equal(np.packbits(masks[role], axis=1, bitorder='little'), packed)
                    checked_bits += masks[role].size
                for j, method in enumerate(a['methods'].tolist()):
                    es = {role:events(mask[:,:,j]) for role,mask in masks.items()}
                    cache[(fault,state,family,method)] = (es, float(a['thresholds'][j]), j)
                    if family == 'repeated':
                        checks = [
                            ('normal_alarm_any', 'normal', 'any', 1),
                            ('fault_prefault_alarm', 'fault', 'pre_any', 1),
                            ('fault_detected', 'fault', 'post_any', 1),
                            ('fault_clean_detection', 'fault', 'clean', 1),
                            ('fault_event_loss_samples', 'fault', 'loss', 3),
                            ('fault_occupancy_points', 'fault', 'post_above', 1),
                            ('fault_postalarm_below_points', 'fault', 'post_below_after_first', 1),
                            ('normal_occupancy_points', 'normal', 'post_above', 1),
                        ]
                        for key,role,field,scale in checks:
                            assert np.array_equal(es[role][field], a[key][:,j]*scale), (p,method,key)
                            checked_scalars += len(a['run_id'])
            full = T4[fault][0][1]
            orig_full = {3:'mean_pool_signed',9:'moment_pool_signed',15:'moment_pool_squared'}[fault]
            for bm,rm in [(full,orig_full),('Q_CUSUM','Q_CUSUM')]:
                be,bh,_ = cache[fault,state,'baseline',bm]
                re,rh,_ = cache[fault,state,'repeated',rm]
                assert bh == rh
                for role in ['normal','fault']:
                    for k in be[role]:
                        assert np.array_equal(be[role][k],re[role][k])
                verification.append(dict(check='duplicated_full_or_Q_endpoint_equality',fault=fault,state=state,method=bm,pass_check=True))

    reference = read_csv(root/'REFERENCE_TABLE4_COMPARISON.csv')
    perrun = []
    state_rows = []
    five_rows = []
    latching_rows = []
    resampling = {}
    matrices = {}
    for fault, conditions in T4.items():
        for family,method,label in conditions:
            for role in ['normal','fault']:
                vals = []
                for state in range(1,6):
                    es,h,_ = cache[fault,state,family,method]
                    e = es[role]
                    vals.append(metric_vectors(e,role))
                    for run in range(250):
                        ident = cohorts[fault if role == 'fault' else 0][run]
                        perrun.append(dict(fault=fault,state=state,family=family,method=method,
                            monitor=label,threshold=h,role=role,canonical_run_id=ident['run_id'],
                            run_number=run+1,self_overlap=ident['self_wrap'],
                            prefault_above_points=int(e['pre_above'][run]),
                            postfault_above_points=int(e['post_above'][run]),
                            whole_run_above_points=int(e['all_above'][run]),
                            prefault_any_alarm=int(e['pre_any'][run]),
                            any_alarm=int(e['any'][run]),
                            clean_detection_or_normal_floor=int(e['clean'][run]),
                            clean8_or_normal_floor8=int(e['clean8'][run]),
                            penalized_loss_min=int(e['loss'][run]),
                            actual_below_points_after_first_postalarm=int(e['post_below_after_first'][run]),
                            counterfactual_latched_postfault_above_points=int(e['latched_post_above'][run])))
                matrices[fault,method,role] = np.stack(vals)
            for scenario in ['original','exclude_test_self_overlap']:
                keep = {}
                for role in ['fault','normal']:
                    ids = cohorts[fault if role == 'fault' else 0]
                    keep[role] = np.array([scenario == 'original' or x['self_wrap'].lower() == 'false' for x in ids])
                averaged = dict(scenario=scenario,panel='R5',fault=fault,family=family,
                                method=method,monitor=label,states=5,
                                unique_fault_runs=int(keep['fault'].sum()),
                                unique_normal_runs=int(keep['normal'].sum()),
                                threshold_convention='Original_run_level_rank_247_of_259_unchanged',k=.05)
                for role, names in [('fault',FAULT_METRICS),('normal',NORMAL_METRICS)]:
                    mat = matrices[fault,method,role][:,keep[role]]
                    v = mat.mean(0)
                    lo,hi,entropy,digest = bootstrap(v,role=role,fault=fault,scenario=scenario)
                    resampling[f'{scenario}|{role}|F{fault if role == "fault" else 0:02d}'] = dict(seed_entropy=entropy,indices_sha256_le_int32=digest,n_physical_runs=len(v),draws=B)
                    for j,name in enumerate(names):
                        averaged[name] = float(v[:,j].mean())
                        averaged[name+'_lo95'] = float(lo[j])
                        averaged[name+'_hi95'] = float(hi[j])
                        averaged[name+'_state_min'] = float(mat[:,:,j].mean(1).min())
                        averaged[name+'_state_max'] = float(mat[:,:,j].mean(1).max())
                for state in range(1,6):
                    es,h,_ = cache[fault,state,family,method]
                    row = {k:averaged[k] for k in ['scenario','panel','fault','family','method','monitor','unique_fault_runs','unique_normal_runs','threshold_convention','k']}
                    row.update(state=state,threshold=h)
                    for role,names in [('fault',FAULT_METRICS),('normal',NORMAL_METRICS)]:
                        values = matrices[fault,method,role][state-1,keep[role]].mean(0)
                        row.update({name:float(values[j]) for j,name in enumerate(names)})
                    state_rows.append(row)
                matches = [r for r in reference if r['scenario'] == scenario and int(r['fault']) == fault and r['method'] == method and r['family'] == family]
                assert len(matches) == 1
                r = matches[0]
                for ours,old in [('clean40_pct','clean_pct_mean'),('clean8_pct','clean8_pct_mean'),('mean_loss_min','mean_loss_min_mean'),('normal_any_alarm_pct','normal_FAR_pct_mean'),('normal_detection_floor_pct','normal_floor_pct_mean')]:
                    err = abs(averaged[ours]-float(r[old]))
                    assert err < 1e-9,(fault,method,scenario,ours,err)
                    verification.append(dict(check='archived_Table4_aggregate',fault=fault,method=method,scenario=scenario,metric=ours,max_absolute_difference=err,pass_check=True))
                five_rows.append(averaged)
                total_clean = sum(int(cache[fault,s,family,method][0]['fault']['clean'][keep['fault']].sum()) for s in range(1,6))
                returned_clean = sum(int(cache[fault,s,family,method][0]['fault']['clean_returns_below'][keep['fault']].sum()) for s in range(1,6))
                latching_rows.append(dict(scenario=scenario,fault=fault,method=method,monitor=label,
                    unique_fault_runs=int(keep['fault'].sum()),states=5,
                    actual_postfault_sample_FDR_pct=averaged['postfault_sample_FDR_pct'],
                    counterfactual_latched_postfault_FDR_pct=averaged['counterfactual_latched_FDR_pct'],
                    latching_excess_FDR_pp=averaged['latching_excess_FDR_pp'],
                    clean_state_run_pairs=total_clean,
                    clean_state_run_pairs_with_return_below_h=returned_clean,
                    conditional_return_below_h_pct=100*returned_clean/total_clean if total_clean else '',
                    warning='Latching is a counterfactual coding diagnostic, not an evaluated monitor'))

    contrasts=[]
    for fault,name,ia,ib in CONTRASTS:
        fa,ma,la=T4[fault][ia]
        fb,mb,lb=T4[fault][ib]
        for scenario in ['original','exclude_test_self_overlap']:
            row=dict(scenario=scenario,panel='R5',fault=fault,contrast=name,
                     monitor_A=ma,monitor_B=mb,orientation='A minus B',states=5,
                     interval='Marginal paired physical-run bootstrap 95%; supplementary',bootstrap_draws=B)
            for role,names in [('fault',FAULT_METRICS),('normal',NORMAL_METRICS)]:
                ids=cohorts[fault if role=='fault' else 0]
                keep=np.array([scenario=='original' or x['self_wrap'].lower()=='false' for x in ids])
                a=matrices[fault,ma,role][:,keep]
                b=matrices[fault,mb,role][:,keep]
                diff=(a-b).mean(0)
                lo,hi,_,_=bootstrap(diff,role=role,fault=fault,scenario=scenario)
                row[f'unique_{role}_runs']=int(keep.sum())
                for j,metric in enumerate(names):
                    unit='min' if metric=='mean_loss_min' else 'pp'
                    short=metric.removesuffix('_pct').removesuffix('_pp').removesuffix('_min')
                    row[f'delta_{short}_{unit}']=float(diff[:,j].mean())
                    row[f'delta_{short}_{unit}_lo95']=float(lo[j])
                    row[f'delta_{short}_{unit}_hi95']=float(hi[j])
            contrasts.append(row)

    write_csv(out/'R5_SAMPLE_AND_RUN_METRICS_BY_STATE.csv',state_rows)
    write_csv(out/'R5_SAMPLE_AND_RUN_METRICS_FIVE_STATE.csv',five_rows)
    compact_fields = ['scenario','fault','monitor','states','unique_fault_runs','unique_normal_runs',
                      'postfault_sample_FDR_pct','postfault_sample_FDR_pct_lo95','postfault_sample_FDR_pct_hi95',
                      'prefault_sample_FAR_pct','clean40_pct','clean8_pct','normal_sample_FAR_pct',
                      'normal_any_alarm_pct','mean_loss_min']
    write_csv(out/'R5_SAMPLE_METRICS_COMPACT.csv', [{k:r[k] for k in compact_fields} for r in five_rows])
    write_csv(out/'R5_SAMPLE_CONTRASTS_PAIRED_95.csv',contrasts)
    write_csv(out/'R5_ACTUAL_VS_LATCHED_OCCUPANCY_DIAGNOSTIC.csv',latching_rows)
    write_csv(out/'R5_ACTUAL_ALARM_COUNTS_PER_RUN.csv',perrun)
    write_csv(out/'INPUT_SHA256.csv',inputs_read)
    write_csv(out/'REAGGREGATION_CHECKS.csv',verification)
    spec=dict(analysis='A5 supplementary R5 sample-occupancy extension',
              original_A5_historical_D01_Braatz_complete=False,
              new_simulations=0,model_fits=0,new_thresholds=0,
              normal_sample_denominator=960,fault_prefault_denominator=160,
              fault_postfault_denominator=800,alarm_rule='actual S_t > h; no alarm latch',
              five_state_averaging='Per physical run before resampling; 250 unique evaluation runs per cohort before exclusions',
              uncertainty='20,000 marginal percentile bootstrap resamples of whole physical runs. Five learned states and original thresholds treated as fixed; not evidence of RNG-stream independence.',
              scenarios=['original','exclude_test_self_overlap (unchanged calibration and h)'],
              resampling=resampling)
    (out/'ANALYSIS_SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
    status=dict(complete=True,new_computed_state_rows=len(state_rows),new_computed_five_state_rows=len(five_rows),new_paired_contrast_rows=len(contrasts),per_run_rows=len(perrun),
                original_alarm_bits_unpacked_and_repacked=checked_bits,
                original_integer_endpoint_and_occupancy_values_checked=checked_scalars,
                duplicate_and_archived_Table4_checks=len(verification),all_checks_passed=True)
    (out/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
