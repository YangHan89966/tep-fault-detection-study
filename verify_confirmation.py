#!/usr/bin/env python3
"""Verify saved confirmation tables, without simulation or model fitting."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent / 'evidence/2026_10_24'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, message):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-11,
                         abs_tol=1e-8), message)


def read_csv(name):
    with (ROOT / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    extracted = json.loads((ROOT / 'EXTRACTED_FILES.json').read_text())
    for entry in extracted['files']:
        path = Path(entry['path'])
        require(not path.is_absolute() and '..' not in path.parts,
                'Unsafe extract path')
        data = (ROOT / path).read_bytes()
        require(len(data) == entry['size_bytes'], 'Size mismatch: ' + str(path))
        require(hashlib.sha256(data).hexdigest() == entry['sha256'],
                'Hash mismatch: ' + str(path))
    results = json.loads((ROOT / 'results/RESULTS.json').read_text())
    for name, expected in results['csv_sha256'].items():
        require(hashlib.sha256((ROOT / 'results' / name).read_bytes()).hexdigest()
                == expected, 'Result-record CSV mismatch: ' + name)
    manifest = read_csv('MANIFEST.csv')
    require(len(manifest) == 1199, 'Manifest total')
    require(len({r['run_id'] for r in manifest}) == 1199, 'Duplicate run ID')
    require(len({r['key_hex'] for r in manifest}) == 1199, 'Duplicate RNG key')
    require({int(r['ordinal']) for r in manifest} == set(range(1, 1200)),
            'Ordinal coverage')
    require(Counter((r['role'], r['fault_id']) for r in manifest) == {
        ('normal_calibration', '0'): 199, ('normal_evaluation', '0'): 400,
        ('fault_evaluation', '3'): 200, ('fault_evaluation', '9'): 200,
        ('fault_evaluation', '15'): 200}, 'Role/fault counts')
    lock = json.loads((ROOT / 'THRESHOLDS_LOCK.json').read_text())
    require(lock['n_calibration'] == 199 and lock['order_1based'] == 190,
            'Calibration rule')
    require(lock['manifest_sha256'] == hashlib.sha256(
        (ROOT / 'MANIFEST.csv').read_bytes()).hexdigest(), 'Locked manifest')
    require(set(lock['calibration_run_ids']) == {
        r['run_id'] for r in manifest if r['role'] == 'normal_calibration'},
        'Calibration run coverage')

    events = read_csv('results/per_run_events.csv')
    require(len(events) == 13200, 'Event count')
    require(len({(e['run_id'], e['arm_id']) for e in events}) == len(events),
            'Duplicate run/arm event')
    by_run = {r['run_id']: r for r in manifest}
    by_arm = defaultdict(list)
    for event in events:
        require(event['run_id'] in by_run, 'Unregistered event')
        registered = by_run[event['run_id']]
        require((event['role'], event['fault_id']) ==
                (registered['role'], registered['fault_id']), 'Event role mismatch')
        by_arm[event['arm_id']].append(event)
    evaluation_ids = {r['run_id'] for r in manifest if r['role'] != 'normal_calibration'}
    require({e['run_id'] for e in events} == evaluation_ids, 'Evaluation coverage')
    arms = read_csv('results/arm_summary.csv')
    require(len(arms) == 22 and {a['arm_id'] for a in arms} == set(by_arm),
            'Monitoring-arm coverage')
    losses = {}
    for arm in arms:
        records = by_arm[arm['arm_id']]
        normal = [e for e in records if e['role'] == 'normal_evaluation']
        fault = [e for e in records if e['role'] == 'fault_evaluation']
        require(len(normal) == 400 and len(fault) == 200, 'Arm physical sample size')
        require({e['fault_id'] for e in fault} == {arm['fault_id']}, 'Arm fault mapping')
        for field, group, out in [('normal_alarm', normal, 'normal_alarm_count'),
                                  ('prefault_alarm', fault, 'fault_prefault_alarm_count'),
                                  ('clean40', fault, 'clean40_count'),
                                  ('clean2', fault, 'clean2_count'),
                                  ('clean8', fault, 'clean8_count')]:
            count = sum(int(e[field]) for e in group)
            near(count, arm[out], 'Count mismatch: ' + arm['arm_id'] + '/' + field)
            pct = out[:-6] + '_pct'
            near(100 * count / len(group), arm[pct], 'Rate mismatch: ' + pct)
        for e in fault:
            clean = bool(int(e['clean40']))
            loss = float(e['loss_minutes'])
            require(0 <= loss <= 2400, 'Loss outside registered range')
            near(loss, float(e['delay_minutes']) if clean else 2400, 'Penalized loss')
            require(not (clean and int(e['prefault_alarm'])), 'Unclean detection')
            for field, limit in [('clean2', 120), ('clean8', 480)]:
                require(int(e[field]) == int(clean and float(e['delay_minutes']) <= limit),
                        'Window endpoint mismatch')
        near(sum(float(e['loss_minutes']) for e in fault) / 200,
             arm['mean_loss_minutes'], 'Mean loss mismatch')
        losses[arm['arm_id']] = {e['run_id']: float(e['loss_minutes']) for e in fault}

    method_faults = {'mean_pool_signed': '3', 'same_mean_signed_axis_squared': '3',
                     'moment_pool_squared': '15', 'mean_pool_squared': '15'}
    endpoint_map = {'loss_minutes': 'mean_loss_minutes', 'normal_alarm': 'normal_alarm_pct',
                    'clean': 'clean40_pct', 'clean_120': 'clean2_pct', 'clean_480': 'clean8_pct'}
    five_state = read_csv('results/five_state_averages.csv')
    expected_five_state = {(fault_id, method, endpoint)
                           for method, fault_id in method_faults.items()
                           for endpoint in endpoint_map}
    require(Counter((r['fault_id'], r['method'], r['endpoint']) for r in five_state)
            == Counter(expected_five_state), 'Five-state family coverage/uniqueness')
    for row in five_state:
        matched = [a for a in arms if a['fault_id'] == row['fault_id'] and
                   a['arm_id'].endswith('_' + row['method'])]
        require(Counter(a['state'] for a in matched) == Counter(map(str, range(1, 6))),
                'Five-state mapping')
        require(int(row['fixed_states']) == 5, 'Five-state fixed-state count')
        field = endpoint_map[row['endpoint']]
        near(sum(float(a[field]) for a in matched) / 5, row['estimate'],
             'Five-state mean mismatch')
        require(int(row['physical_n']) == (400 if row['endpoint'] == 'normal_alarm' else 200),
                'Five-state physical sample size')

    contrast_methods = {
        'C1_same_axis_squared_minus_signed': ('3', 'same_mean_signed_axis_squared', 'mean_pool_signed'),
        'C3_moment_minus_mean_squared': ('15', 'moment_pool_squared', 'mean_pool_squared'),
        'F9_controller_input_minus_native': ('9', 'controller_input', 'native')}
    primary = read_csv('results/primary_contrasts.csv')
    require(Counter(r['contrast'] for r in primary) == Counter(contrast_methods.keys()),
            'Primary family coverage/uniqueness')
    for row in primary:
        fault_id, left, right = contrast_methods[row['contrast']]
        pair_means = []
        ids = [r['run_id'] for r in manifest if r['role'] == 'fault_evaluation'
               and r['fault_id'] == fault_id]
        for run_id in ids:
            values = []
            for method in (left, right):
                selected = [a for a in arms if a['fault_id'] == fault_id and
                            a['arm_id'].endswith('_' + method)]
                require(len(selected) == (1 if fault_id == '9' else 5), 'Paired-state mapping')
                values.append(sum(losses[a['arm_id']][run_id] for a in selected) / len(selected))
            pair_means.append(values[0] - values[1])
        near(sum(pair_means) / 200, row['estimate'], 'Paired physical-run contrast')
        require(int(row['physical_n']) == 200, 'Primary physical sample size')
    require(results['physical_trajectories'] == 1199 and results['shutdown_run_count'] == 0,
            'Result panel count')
    require(not read_csv('results/shutdown_runs.csv'), 'Unexpected shutdown records')
    print(json.dumps({'status': 'PASS', 'physical_runs': 1199, 'evaluation_runs': 1000,
                      'event_rows': len(events), 'monitoring_arms': 22,
                      'checks': 'hashes, identities, roles, saved-event counts, rates, losses and primary point contrasts',
                      'intervals_recomputed': False, 'scientific_simulation_executed': False}))


if __name__ == '__main__':
    main()
