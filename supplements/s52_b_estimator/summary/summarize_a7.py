#!/usr/bin/env python3
"""Aggregate saved A7 events only. No simulation, fitting, or calibration.

Requires Python 3 and NumPy. The two input directories are read-only.
New intervals condition on the five fitted/calibrated states and resample the
250 physical test runs, after averaging over states within each run. They are
post-hoc marginal percentile intervals, not a confirmatory familywise result.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ESTIMATORS = ('original', 'batch', 'HAC')
METHODS = ('mean_pool_signed', 'moment_pool_signed')
MEASURES = ('loss_minutes', 'clean40_pct', 'clean8_pct', 'normal_FAR_pct')
BOOTSTRAPS = 20000
SEEDS = {'fault': 2026100907, 'normal': 2026100908}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--a7-dir', type=Path, required=True)
    ap.add_argument('--engine-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def source(p):
        p = p.resolve()
        hashes[str(p)] = {'path': str(p), 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        return p

    def jread(p):
        return json.loads(source(p).read_text(encoding='utf-8-sig'))

    def nread(p):
        with np.load(source(p), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def cread(p):
        with source(p).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def write(name, rows):
        with (args.out / name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    manifest = jread(args.engine_dir / 'A7_INPUTS.json')
    assert [s['state'] for s in manifest['states']] == list(range(1, 6))
    for role in ('fault', 'normal'):
        first = manifest['states'][0][role]
        assert len(first['ids']) == len(set(first['ids'])) == 250
        for s in manifest['states']:
            assert all(s[role][k] == first[k] for k in ('ids', 'sha256s', 'files'))
    source(args.engine_dir / 'relearn_native.py')
    source(args.engine_dir / 'additional_analysis_core.py')
    source(args.engine_dir / 'reference_code/inherited.py')
    source(args.engine_dir / 'reference_code/science.py')
    source(args.engine_dir.parent.parent / 'code/run_one_analysis.py')
    task = jread(args.a7_dir / 'TASK_STATUS.json')
    assert task['scientific_data_status'] == 'COMPLETE'
    old_ep = cread(args.a7_dir / 'A7_ENDPOINTS.csv')
    old_ct = cread(args.a7_dir / 'A7_PAIRED_CONTRASTS.csv')
    old_eig = cread(args.a7_dir / 'A7_B_EIGENVALUES.csv')
    old_replay = cread(args.a7_dir / 'A7_ORIGINAL_B_REPLAY.csv')
    assert (len(old_ep), len(old_ct), len(old_eig), len(old_replay)) == (30, 15, 15, 5)
    assert all(float(r['threshold_max_abs_error']) == 0 and
               r['original_fault_loss_clean_and_normal_FAR_vectors'] == 'PASS' for r in old_replay)
    ep_index = {(int(r['state']), r['metric'], r['method']): r for r in old_ep}
    ct_index = {(int(r['state']), r['metric']): r for r in old_ct}
    eig_index = {(int(r['state']), r['metric']): r for r in old_eig}
    # Arrays indexed by estimator, state, physical run, method, outcome.
    vectors = np.empty((3, 5, 250, 2, 4), dtype=float)
    endpoint_checks = []; replay_checks = []; selections = []; eig_checks = []
    per_state = []; identity_rows = []
    max_endpoint_error = 0.; max_eigen_error = 0.
    for role in ('fault', 'normal'):
        spec = manifest['states'][0][role]
        identity_rows.extend({'role': role, 'event_row_zero_based': i, 'physical_run_id': rid,
                              'raw_sha256': spec['sha256s'][i]} for i, rid in enumerate(spec['ids']))
    for si, state in enumerate(manifest['states']):
        s = state['state']
        baseline = nread(args.engine_dir / state['baseline_events'])
        baseline_des = nread(args.engine_dir / state['design'])
        reference_des = None
        original_choices = None
        for ei, estimator in enumerate(ESTIMATORS):
            directory = args.a7_dir / f'state{s}_{estimator}'
            ev = nread(directory / 'EVALUATION_EVENTS.npz')
            des = nread(directory / 'DESIGN.npz')
            sel = jread(directory / 'SELECTION.json')
            inner = nread(directory / 'INNER.npz')
            outer = nread(directory / 'OUTER.npz')
            model = nread(directory / 'MODEL.npz')
            methods = ev['methods'].tolist()
            assert ev['fault_event_loss_minutes'].shape == ev['fault_clean'].shape == ev['normal_alarm'].shape == (250, 9)
            assert outer['rank'].item() == 247 and outer['n'].item() == 259
            assert sel['evaluation_used'] is False and sel['outer_used'] is False
            assert model['mean'].shape == model['moment'].shape == (8, 16)
            assert len(sel['candidate_names']) == 49
            if estimator == 'original':
                reference_des = des
                original_choices = sel['choices']
                order = [baseline['methods'].tolist().index(m) for m in methods]
                h_error = float(np.max(abs(ev['thresholds'] - baseline['thresholds'][order])))
                assert h_error == 0
                for actual, wanted in (
                    (ev['fault_event_loss_minutes'], baseline['fault_event_loss_samples'][:, order] * 3),
                    (ev['fault_clean'], baseline['fault_clean_detection'][:, order]),
                    (ev['normal_alarm'], baseline['normal_alarm_any'][:, order])):
                    assert np.array_equal(actual, wanted)
                b_error = float(np.max(abs(des['B'] - baseline_des['B'])))
                assert np.allclose(des['B'], baseline_des['B'], rtol=1e-9, atol=1e-10)
                replay_checks.append({'state': s, 'threshold_max_abs_error': h_error,
                                      'B_max_abs_error_vs_saved_original': b_error,
                                      'all_9_methods_loss_clean_normal_vectors': 'PASS'})
            for key in des:
                if key != 'B':
                    assert np.array_equal(des[key], reference_des[key]), (s, estimator, key)
            eigenvalues = np.linalg.eigvalsh((des['B'] + des['B'].T) / 2)
            assert eigenvalues.min() > 0
            values = {'minimum_eigenvalue': eigenvalues[0], 'maximum_eigenvalue': eigenvalues[-1],
                      'condition_number': eigenvalues[-1] / eigenvalues[0]}
            er = max(abs(float(eig_index[s, estimator][k]) - v) for k, v in values.items())
            max_eigen_error = max(max_eigen_error, er)
            assert er < 1e-9
            eig_checks.append({'state': s, 'B_estimator': estimator, **values,
                               'max_abs_error_vs_returned_eigen_summary': er,
                               'B_relative_Frobenius_change_vs_original': float(np.linalg.norm(des['B'] - reference_des['B']) / np.linalg.norm(reference_des['B']))})
            for mi, method in enumerate(METHODS):
                j = methods.index(method)
                loss = ev['fault_event_loss_minutes'][:, j]
                clean = ev['fault_clean'][:, j].astype(bool)
                assert np.all(loss[~clean] == 2400)
                assert np.all((loss[clean] >= 0) & (loss[clean] <= 2397))
                normal = ev['normal_alarm'][:, j].astype(bool)
                clean8 = clean & (loss <= 480)
                vectors[ei, si, :, mi, :] = np.column_stack([loss, clean * 100, clean8 * 100, normal * 100])
                measures = {'threshold': ev['thresholds'][j], 'clean40_n': clean.sum(),
                            'clean40_pct': clean.mean() * 100, 'clean8_n': clean8.sum(),
                            'clean8_pct': clean8.mean() * 100, 'mean_loss_minutes': loss.mean(),
                            'normal_alarm_n': normal.sum(), 'normal_FAR_pct': normal.mean() * 100}
                er = max(abs(float(ep_index[s, estimator, method][k]) - float(v)) for k, v in measures.items())
                max_endpoint_error = max(max_endpoint_error, er)
                assert er < 1e-10
                endpoint_checks.append({'state': s, 'B_estimator': estimator, 'method': method,
                                        **measures, 'max_abs_error_vs_returned_endpoint': er})
                selected = sel['choices'][method]
                candidates = np.arange(mi * 8, mi * 8 + 8)
                total = inner['event_loss_samples'][:, candidates].sum(0)
                assert selected == candidates[np.argmin(total)]
                assert ev['thresholds'][j] == outer['thresholds'][selected]
                selections.append({'state': s, 'B_estimator': estimator, 'method': method,
                                   'candidate': sel['candidate_names'][selected],
                                   'original_candidate': sel['candidate_names'][original_choices[method]],
                                   'candidate_identity_changed': selected != original_choices[method],
                                   'selection_total_loss_samples': int(total.min()),
                                   'selection_min_integer_loss_check': 'PASS'})
            difference = vectors[ei, si, :, 1] - vectors[ei, si, :, 0]
            returned = ct_index[s, estimator]
            assert abs(difference[:, 0].mean() - float(returned['loss_difference_minutes'])) < 1e-10
            per_state.append({'state': s, 'B_estimator': estimator,
                              **{k: difference[:, q].mean() for q, k in enumerate(MEASURES)},
                              'returned_A7_perstate_CI98_333_low': returned['CI98_333_low'],
                              'returned_A7_perstate_CI98_333_high': returned['CI98_333_high'],
                              'interval_status': 'A7_returned_perstate_nominal_98.3333_not_five_state_or_15_comparison_familywise'})

    # Keep state averaging inside physical-run identity before any resampling.
    by_run = vectors.mean(axis=1)  # estimator, physical run, method, outcome
    c2_by_run = by_run[:, :, 1] - by_run[:, :, 0]
    fault_rng = np.random.Generator(np.random.PCG64(SEEDS['fault']))
    normal_rng = np.random.Generator(np.random.PCG64(SEEDS['normal']))
    samples = np.empty((BOOTSTRAPS, 3, 4), dtype=float)
    for start in range(0, BOOTSTRAPS, 1000):
        size = min(1000, BOOTSTRAPS - start)
        fault_indices = fault_rng.integers(0, 250, (size, 250))
        normal_indices = normal_rng.integers(0, 250, (size, 250))
        for ei in range(3):
            samples[start:start + size, ei, :3] = c2_by_run[ei][fault_indices, :3].mean(axis=1)
            samples[start:start + size, ei, 3] = c2_by_run[ei][normal_indices, 3].mean(axis=1)
    aggregate = []; changes_c2 = []; endpoints = []; method_changes = []
    for ei, estimator in enumerate(ESTIMATORS):
        row = {'B_estimator': estimator, 'fixed_states': 5, 'physical_fault_runs': 250, 'physical_normal_runs': 250}
        for q, measure in enumerate(MEASURES):
            values = (vectors[ei, :, :, 1, q] - vectors[ei, :, :, 0, q]).mean(axis=1)
            lo, hi = np.quantile(samples[:, ei, q], [.025, .975], method='linear')
            row.update({measure: c2_by_run[ei, :, q].mean(), measure + '_marginal_CI95_low': lo,
                        measure + '_marginal_CI95_high': hi, measure + '_state_min': values.min(),
                        measure + '_state_max': values.max()})
        row['loss_negative_states'] = int(((vectors[ei, :, :, 1, 0] - vectors[ei, :, :, 0, 0]).mean(1) < 0).sum())
        row['returned_perstate_98_333_CIs_below_zero'] = sum(float(r['returned_A7_perstate_CI98_333_high']) < 0 for r in per_state if r['B_estimator'] == estimator)
        aggregate.append(row)
        for mi, method in enumerate(METHODS):
            row = {'B_estimator': estimator, 'method': method, 'fixed_states': 5, 'physical_runs_per_endpoint': 250}
            for q, measure in enumerate(MEASURES):
                state_values = vectors[ei, :, :, mi, q].mean(1)
                row.update({measure: by_run[ei, :, mi, q].mean(), measure + '_state_min': state_values.min(),
                            measure + '_state_max': state_values.max()})
            endpoints.append(row)
            if ei:
                row = {'B_estimator_minus_original': estimator, 'method': method}
                row.update({measure: (by_run[ei, :, mi, q] - by_run[0, :, mi, q]).mean() for q, measure in enumerate(MEASURES)})
                method_changes.append(row)
        if ei:
            row = {'contrast': f'C2_{estimator}_minus_C2_original'}
            for q, measure in enumerate(MEASURES):
                lo, hi = np.quantile(samples[:, ei, q] - samples[:, 0, q], [.025, .975], method='linear')
                row.update({measure: (c2_by_run[ei, :, q] - c2_by_run[0, :, q]).mean(),
                            measure + '_marginal_CI95_low': lo, measure + '_marginal_CI95_high': hi})
            changes_c2.append(row)
    write('A7_FIVE_STATE_ENDPOINTS.csv', endpoints)
    write('A7_FIVE_STATE_C2_MARGINAL95.csv', aggregate)
    write('A7_CHANGE_IN_C2_MARGINAL95.csv', changes_c2)
    write('A7_METHOD_CHANGES_VS_ORIGINAL.csv', method_changes)
    write('A7_PER_STATE_C2.csv', per_state)
    write('A7_SELECTION_CHANGES.csv', selections)
    write('A7_B_NUMERIC_CHECKS.csv', eig_checks)
    write('A7_ENDPOINT_EVENT_CHECKS.csv', endpoint_checks)
    write('A7_ORIGINAL_REPLAY_RECHECK.csv', replay_checks)
    write('A7_SHARED_PHYSICAL_RUN_IDENTITIES.csv', identity_rows)
    write('INPUT_HASHES.csv', list(hashes.values()))
    verification = {
        'status': 'PASS', 'new_TEP_simulations': False, 'new_fitting_or_calibration': False,
        'shared_ordered_250_fault_and_250_normal_ids_files_hashes_across_states': True,
        'original_replay_states': 5, 'original_replay_methods_per_state': 9,
        'original_replay_max_threshold_abs_error': max(r['threshold_max_abs_error'] for r in replay_checks),
        'endpoint_rows_checked': 30, 'max_endpoint_abs_error': max_endpoint_error,
        'paired_point_estimates_checked': 15, 'eigen_summaries_checked': 15,
        'max_eigen_summary_abs_error': max_eigen_error,
        'selection_integer_loss_minima_checked': 30,
        'unchanged_non_B_design_fields_across_estimators': True,
        'candidate_pool_axes_per_family': 8, 'outer_n': 259, 'outer_rank': 247,
        'B_matrix_scope': 'stored B eigenvalues recomputed; original B compared to saved original; no raw-score B recomputation',
        'new_bootstrap': {'draws': BOOTSTRAPS, 'rng': 'NumPy PCG64', 'seeds': SEEDS,
                          'unit': 'one shared physical evaluation run, average five states before resampling',
                          'shared_indices_across_estimators': True, 'coverage': .95,
                          'quantile_method': 'linear', 'marginal_post_hoc_conditional_not_familywise': True},
        'returned_perstate_intervals': {'coverage': 1 - .05 / 3, 'draws': 20000,
            'seed': '2026100800 + state', 'status': 'retained from RETURN, not rerun; do not equate with original confirmatory familywise success'},
        'numpy_version': np.__version__, 'input_files_hashed': len(hashes)
    }
    (args.out / 'VERIFICATION.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'C2': aggregate, 'change_in_C2': changes_c2}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
