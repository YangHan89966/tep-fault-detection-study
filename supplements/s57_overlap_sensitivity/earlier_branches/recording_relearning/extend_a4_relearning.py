#!/usr/bin/env python3
"""Complete the previously missing A4 recalibration branch for Table-6 M1-M4.

Reads existing scalar scores, calibration maxima, locks and identities only.
No simulator, model fitting, candidate selection or original-file writes.
Existing A4 bootstrap results are not rerun; bootstrap is for the new branch only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

MONITORS = [('M1', 'controller-input', 'second-moment'),
            ('M2', 'native', 'second-moment'),
            ('M3', 'controller-input', 'mean-shift'),
            ('M4', 'native', 'mean-shift')]
SCENARIO = 'exclude_test_and_calibration_self_overlap'
DRAW_COUNT = 20000
RNG_NAMESPACE = 'A4_RELEARNING_CALIBRATION_EXCLUSION_v1|20261008|'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def chart(scores):
    u = np.asarray(scores, dtype=np.float64)
    assert u.ndim == 3 and u.shape[1:] == (960, 4) and np.isfinite(u).all()
    positive = np.zeros((u.shape[0], 4)); negative = positive.copy()
    result = np.empty_like(u)
    for t in range(960):
        positive = np.maximum(0.0, positive + u[:, t, :] - 0.05)
        negative = np.maximum(0.0, negative - u[:, t, :] - 0.05)
        result[:, t, :] = np.maximum(positive, negative)
    return result


def events(mask):
    pre = mask[:, :160, :].any(axis=1)
    post = mask[:, 160:, :]
    detected = post.any(axis=1)
    delay = np.where(detected, post.argmax(axis=1) * 3, 2400)
    clean = (~pre) & detected
    return {'clean40': clean, 'clean8': clean & (delay <= 480),
            'clean2': clean & (delay <= 120), 'pre': pre,
            'normal_alarm': mask.any(axis=1),
            'loss': np.where(clean, delay, 2400).astype(float)}


def thresholds(peaks, anchors, rank):
    assert np.all(anchors > 0) and 1 <= rank <= len(peaks)
    # Preserve the archived scale / order statistic / scale-back operation order.
    return anchors * np.sort(peaks / anchors, axis=0)[rank - 1]


def wilson(k, n):
    z = 1.959963984540054; p = k / n; denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return 100 * max(0., centre - half), 100 * min(1., centre + half)


def indices(panel, role, n):
    seed = int.from_bytes(hashlib.sha256((RNG_NAMESPACE + panel + '|' + role).encode('ascii')).digest()[:16], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n, size=(DRAW_COUNT, n), dtype=np.int64), seed


def main(archive, a4_root, out):
    archive = Path(archive).resolve(); a4_root = Path(a4_root).resolve()
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    source_identities = []

    def record(data, source):
        source_identities.append({'source': source, 'bytes': len(data), 'sha256': sha(data)})
        return data

    def read(path):
        return record(Path(path).read_bytes(), str(path))

    rng_path = a4_root / 'inputs/nine_checks/01_RNG_Overlap/results/RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv'
    inventory = pd.read_csv(io.BytesIO(read(rng_path)))
    assert len(inventory) == 9007 and inventory.self_wrap.dtype == bool
    assert np.array_equal(inventory.self_wrap, inventory.n_draws > inventory.period)
    assert np.array_equal(inventory.self_wrap,
                          np.where(inventory.duration_hours == 25,
                                   inventory.seed_u32 % 64 == 0, inventory.seed_u32 % 32 == 0))
    old_a4 = pd.read_csv(io.BytesIO(read(a4_root / 'results/TABLE6_COMPARISON.csv')))
    old_a4 = old_a4[old_a4.family == 'relearned']
    outer_bytes = read(archive)

    rows = []; threshold_rows = []; pair_rows = []; identities = []; checks = []; schedules = []
    with ZipFile(io.BytesIO(outer_bytes)) as outer:
        candidates = [n for n in outer.namelist() if n.endswith('F9_RELEARN_RANK_20261007T035904Z_ad63023a_RETURN.zip')]
        assert len(candidates) == 1
        inner_member = candidates[0]
        inner_bytes = record(outer.read(inner_member), str(archive) + '!' + inner_member)
        with ZipFile(io.BytesIO(inner_bytes)) as source:
            def member(name):
                return record(source.read(name), str(archive) + '!' + inner_member + '!' + name)

            def js(name):
                return json.loads(member(name))

            def arr(name):
                with np.load(io.BytesIO(member(name)), allow_pickle=False) as data:
                    return {k: data[k] for k in data.files}

            protocol_bytes = member('PROTOCOL.json'); protocol = json.loads(protocol_bytes)
            lock_bytes = member('workspace/THRESHOLDS_LOCK.json'); lock = json.loads(lock_bytes)
            final = js('workspace/FINAL_LOCK.json')
            assert final['threshold_lock_sha256'] == sha(lock_bytes)
            assert lock['protocol_sha256'] == final['protocol_sha256'] == sha(protocol_bytes)
            assert [(m['id'], m['view'], m['pool']) for m in protocol['monitors']] == [
                ('M1', 'snapshot', 'moment'), ('M2', 'native', 'moment'),
                ('M3', 'snapshot', 'mean'), ('M4', 'native', 'mean')]
            assert protocol['pca']['training_n'] == 78 and not protocol['pca']['refit']
            # Source definitions are recorded, never imported or executed.
            member('code/relearning_core.py'); member('code/inherited_dim.py')
            for panel, inventory_panel in [('S25_primary', 'recording_panel1'),
                                            ('D439_supplement', 'recording_panel2')]:
                entry = lock['panels'][panel]
                cal_name = panel + '.npz'
                cal_bytes = member('workspace/' + lock['relative_path'] + '/' + cal_name)
                assert sha(cal_bytes) == lock['files'][cal_name]
                with np.load(io.BytesIO(cal_bytes), allow_pickle=False) as data:
                    cal = {k: data[k] for k in data.files}
                eval_name = panel + '_EVALUATION.npz'
                eval_bytes = member('workspace/' + final['relative_path'] + '/' + eval_name)
                assert sha(eval_bytes) == final['files'][eval_name]
                with np.load(io.BytesIO(eval_bytes), allow_pickle=False) as data:
                    evaluation = {k: data[k] for k in data.files}
                cases_name = panel + '_CASES.json'
                cases_bytes = member('workspace/' + final['relative_path'] + '/' + cases_name)
                assert sha(cases_bytes) == final['files'][cases_name]
                cases = json.loads(cases_bytes)

                inv = inventory[inventory.panel == inventory_panel].set_index('run_id')
                assert inv.index.is_unique
                cal_ids = list(entry['case_ids'])
                cids = [c['case_id'] for c in cases]
                assert len(cal_ids) == len(set(cal_ids)) == len(cal['peaks'])
                assert len(cids) == len(set(cids)) == len(evaluation['scores'])
                assert not set(cal_ids) & set(cids)
                ci = inv.loc[cal_ids]; ei = inv.loc[cids]
                assert (ci.role == 'normal_calibration').all()
                assert ei.role.tolist() == [c['role'] for c in cases]
                assert ei.seed_u32.tolist() == [c['seed_u32'] for c in cases]
                assert ei.fault_id.tolist() == [c['fault_id'] for c in cases]
                assert (ci.duration_hours == 48).all() and (ei.duration_hours == 48).all()
                for case_id in cal_ids + cids:
                    item = inv.loc[case_id]
                    identities.append({'panel': inventory_panel, 'case_id': case_id,
                                       'role': item.role, 'seed_u32': int(item.seed_u32),
                                       'self_overlap': bool(item.self_wrap),
                                       'exclude_from_calibration': case_id in cal_ids and bool(item.self_wrap),
                                       'exclude_from_evaluation': case_id in cids and bool(item.self_wrap),
                                       'shared_by_all_four_monitors': True})

                ckeep = ~ci.self_wrap.to_numpy(); ekeep = ~ei.self_wrap.to_numpy()
                ncal = int(ckeep.sum()); rank = (95 * (ncal + 1) + 99) // 100
                oldh = thresholds(cal['peaks'], cal['anchors'], int(entry['rank']))
                assert np.array_equal(oldh, cal['thresholds'])
                assert np.array_equal(oldh, evaluation['thresholds'])
                assert np.array_equal(oldh, np.asarray(entry['thresholds']))
                calibration_chart = chart(cal['scores'])
                assert np.array_equal(calibration_chart.max(axis=1), cal['peaks'])
                newh = thresholds(cal['peaks'][ckeep], cal['anchors'], rank)
                statistic = chart(evaluation['scores'])
                old_mask = statistic > oldh
                saved_mask = np.unpackbits(evaluation['alarm_packed'], axis=1,
                                           count=960, bitorder='little').astype(bool)
                assert np.array_equal(old_mask, saved_mask)
                checks.append({'panel': panel, 'old_alarm_bits_equal': True,
                               'old_alarm_bits_checked': int(old_mask.size),
                               'calibration_peaks_equal': True,
                               'original_thresholds_equal': True,
                               'evaluation_seeds_and_roles_equal': True})
                for j, (mid, view, pool) in enumerate(MONITORS):
                    threshold_rows.append({'panel': inventory_panel, 'monitor': mid,
                                           'recording': view, 'pool': pool,
                                           'original_calibration_n': len(ci),
                                           'excluded_calibration_n': int((~ckeep).sum()),
                                           'retained_calibration_n': ncal,
                                           'old_rank': int(entry['rank']), 'new_rank': rank,
                                           'old_threshold': float(oldh[j]),
                                           'new_threshold': float(newh[j]),
                                           'threshold_change': float(newh[j] - oldh[j]),
                                           'anchor_unchanged': float(cal['anchors'][j])})

                role_normal = ei.role.to_numpy() == 'normal_evaluation'
                role_fault = ei.role.to_numpy() == 'fault9_evaluation'
                assert np.all(role_normal | role_fault)
                new_event = None
                for scenario, h, keep in [('original', oldh, np.ones(len(ei), dtype=bool)),
                                           ('exclude_test_self_overlap', oldh, ekeep),
                                           (SCENARIO, newh, ekeep)]:
                    ev = events(statistic > h)
                    fk = keep & role_fault; nk = keep & role_normal
                    for j, (mid, view, pool) in enumerate(MONITORS):
                        row = {'scenario': scenario, 'panel': inventory_panel,
                               'monitor': mid, 'recording': view, 'pool': pool,
                               'fault_n': int(fk.sum()), 'normal_n': int(nk.sum()),
                               'threshold': float(h[j]),
                               'calibration_n': ncal if scenario == SCENARIO else len(ci),
                               'calibration_rank': rank if scenario == SCENARIO else int(entry['rank']),
                               'mean_loss_min': float(ev['loss'][fk, j].mean())}
                        for key in ['clean40', 'clean8', 'clean2', 'pre']:
                            count = int(ev[key][fk, j].sum()); lo, hi = wilson(count, int(fk.sum()))
                            row['fault_' + key + '_n'] = count
                            row['fault_' + key + '_pct'] = 100 * count / int(fk.sum())
                            row['fault_' + key + '_Wilson95_lo_pct'] = lo
                            row['fault_' + key + '_Wilson95_hi_pct'] = hi
                        for key, label in [('normal_alarm', 'normal_alarm'), ('clean40', 'normal_floor')]:
                            count = int(ev[key][nk, j].sum()); lo, hi = wilson(count, int(nk.sum()))
                            row[label + '_n'] = count; row[label + '_pct'] = 100 * count / int(nk.sum())
                            row[label + '_Wilson95_lo_pct'] = lo; row[label + '_Wilson95_hi_pct'] = hi
                        if scenario != SCENARIO:
                            old = old_a4[(old_a4.scenario == scenario) & (old_a4.panel == inventory_panel)
                                         & (old_a4.method == mid)].iloc[0]
                            for key in ['fault_n', 'normal_n', 'fault_clean40_n', 'fault_clean8_n', 'normal_alarm_n']:
                                assert row[key] == old[key], (panel, mid, scenario, key)
                            assert np.isclose(row['mean_loss_min'], old.mean_loss_min, rtol=0, atol=1e-10)
                            assert row['normal_floor_n'] == old.normal_floor_n
                        rows.append(row)
                    if scenario == SCENARIO:
                        new_event = ev

                fk = ekeep & role_fault; nk = ekeep & role_normal
                fi, fs = indices(panel, 'fault9_evaluation', int(fk.sum()))
                ni, ns = indices(panel, 'normal_evaluation', int(nk.sum()))
                schedules.append({'panel': panel, 'fault_seed': str(fs), 'normal_seed': str(ns),
                                  'fault_index_sha256': sha(fi.tobytes()),
                                  'normal_index_sha256': sha(ni.tobytes()),
                                  'shared_across_recording_pairs_and_endpoints': True})
                for a, b, pool in [(0, 1, 'second-moment'), (2, 3, 'mean-shift')]:
                    row = {'scenario': SCENARIO, 'panel': inventory_panel, 'pool': pool,
                           'contrast': MONITORS[a][0] + '-' + MONITORS[b][0],
                           'direction': 'controller-input minus native',
                           'fault_n': int(fk.sum()), 'normal_n': int(nk.sum()),
                           'bootstrap_draws': DRAW_COUNT, 'interval_percent': 95}
                    for key, label, use, ii, scale in [
                        ('clean40', 'delta_clean40_pp', fk, fi, 100),
                        ('clean8', 'delta_clean8_pp', fk, fi, 100),
                        ('loss', 'delta_loss_min', fk, fi, 1),
                        ('normal_alarm', 'delta_normal_FAR_pp', nk, ni, 100)]:
                        va = new_event[key][use, a].astype(float)
                        vb = new_event[key][use, b].astype(float)
                        difference = scale * (va - vb)
                        bootstrap = difference[ii].mean(axis=1)
                        lo, hi = np.quantile(bootstrap, [.025, .975], method='linear')
                        row[label] = float(difference.mean())
                        row[label + '_lo95'] = float(lo); row[label + '_hi95'] = float(hi)
                        row[label + '_empirically_degenerate'] = bool(np.all(difference == difference[0]))
                        if key == 'clean40':
                            row['controller_only_detected_n'] = int(((va == 1) & (vb == 0)).sum())
                            row['native_only_detected_n'] = int(((va == 0) & (vb == 1)).sum())
                    pair_rows.append(row)

    pd.DataFrame(rows).to_csv(out / 'TABLE6_RELEARNING_THREE_SCENARIOS.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(out / 'RELEARNING_SUPPLEMENTAL_THRESHOLDS.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out / 'RELEARNING_RECALIBRATED_PAIRED_CONTRASTS.csv', index=False)
    pd.DataFrame(identities).to_csv(out / 'SHARED_CASE_IDENTITIES_AND_EXCLUSIONS.csv', index=False)
    (out / 'SOURCE_IDENTITIES.json').write_text(json.dumps(source_identities, ensure_ascii=False, indent=2) + '\n')
    spec = {'scope': 'New A4 recalibration branch for Table-6 supervised relearning M1-M4, both existing panels',
            'bootstrap_draws': DRAW_COUNT, 'rng_namespace': RNG_NAMESPACE,
            'intervals': 'marginal 95% whole physical-run paired percentile, quantile method linear',
            'individual_rate_intervals': 'Wilson 95%, retained recording-study convention',
            'schedules': schedules, 'rank': 'ceil(0.95*(retained calibration n+1)), exact integer calculation',
            'threshold_operations': 'existing positive anchor * order_rank(peaks / anchor)',
            'no_model_refit_or_reselection': True,
            'learning_and_calibration_uncertainty_included': False,
            'original_A4_bootstrap_replayed': False,
            'interpretation': 'Post hoc sensitivity on previously evaluated panels; not a new prespecified primary test'}
    (out / 'RESAMPLING_AND_ANALYSIS_SPEC.json').write_text(json.dumps(spec, indent=2) + '\n')
    status = {'status': 'COMPLETE_TABLE6_SUPERVISED_RELEARNING_RECALIBRATION_BRANCH',
              'monitors': 4, 'panels': 2, 'recalibrated_conditions': 8,
              'paired_recording_contrasts': 4, 'old_reference_checks': checks,
              'original_and_test_only_point_estimates_match_existing_A4': True,
              'new_simulations': 0, 'PCA_refits': 0, 'monitor_refits': 0,
              'candidate_reselections': 0, 'original_threshold_edits': 0,
              'remaining_A4_branches': ['recalibrated Table3 comparators',
                                      'remaining Table4 monitors',
                                      'Table6 fixed historical Panel2',
                                      'Table6 standard monitors'],
              'independence_established': False}
    (out / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(pd.DataFrame(rows).query('scenario == @SCENARIO')[
        ['panel', 'monitor', 'fault_n', 'fault_clean40_n', 'fault_clean8_n',
         'normal_n', 'normal_alarm_n', 'mean_loss_min']].to_string(index=False))
    print(pd.DataFrame(pair_rows)[['panel', 'contrast', 'delta_clean40_pp',
                                  'delta_clean40_pp_lo95', 'delta_clean40_pp_hi95']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--relearn-archive', type=Path, required=True)
    parser.add_argument('--a4-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.relearn_archive, args.a4_root, args.out)
