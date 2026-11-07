"""Reaggregate archived S52 events; never simulate, fit, or change source data.

Requires Python 3.8+ and NumPy. Use --help for input paths. Archived seeds
reproduce both the per-replicate 98.3333% intervals and the five-replicate
95% intervals in Table S52. This independent verifier uses only saved arrays.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np

ESTIMATORS = ['original', 'batch', 'HAC']
METHODS = ['mean_pool_signed', 'moment_pool_signed']
COLUMNS = {'Mean loss (min)': 'loss', '40 h clean detection (%)': 'clean40',
           '8 h clean detection (%)': 'clean8', 'Normal FAR (%)': 'far'}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def paired_interval(difference, seed, alpha=0.05):
    difference = np.asarray(difference, dtype=float)
    if difference.shape != (250,) or not np.isfinite(difference).all():
        raise ValueError('Expected 250 finite paired physical-run differences')
    rng = np.random.default_rng(seed)
    samples = np.empty(20000)
    for start in range(0, 20000, 1000):
        indices = rng.integers(0, len(difference), (1000, len(difference)))
        samples[start:start + 1000] = difference[indices].mean(axis=1)
    return [float(difference.mean()),
            *np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist()]


def numbers(text):
    return [float(value) for value in re.findall(r'[-+]?\d+(?:\.\d+)?',
                                                text.replace('\u2212', '-'))]


def main():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a7', type=Path, default=package / 'results',
                        help='Directory containing A7_ENDPOINTS.csv and state1_original etc.')
    parser.add_argument('--table', type=Path, default=package / 'table' / 'Table_S52.csv',
                        help='Table_S52.csv, or Data S1 ZIP containing Data_S1/Table_S52.csv')
    parser.add_argument('--manifest', type=Path, default=package / 'source' / 'engines' / 'A2_A7' / 'A7_INPUTS.json',
                        help='Optional A7_INPUTS.json to verify shared fault/normal run order')
    parser.add_argument('--summary', type=Path, default=package / 'summary' / 'results' / 'A7_FIVE_STATE_C2_MARGINAL95.csv',
                        help='Archived five-replicate paired summary CSV')
    parser.add_argument('--out', type=Path, required=True,
                        help='New output directory; source data are never overwritten')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_endpoints = read_csv(args.a7 / 'A7_ENDPOINTS.csv')
    source_contrasts = read_csv(args.a7 / 'A7_PAIRED_CONTRASTS.csv')
    if args.table.suffix.lower() == '.zip':
        with zipfile.ZipFile(args.table) as archive:
            table = list(csv.DictReader(io.StringIO(
                archive.read('Data_S1/Table_S52.csv').decode('utf-8-sig'))))
    else:
        table = read_csv(args.table)
    if len(source_endpoints) != 30 or len(source_contrasts) != 15 or len(table) != 9:
        raise ValueError('Unexpected source/table row counts')
    report = {'purpose': 'Independent validation of archived S52 events',
              'source_endpoint_mismatches': [], 'source_replicate_CI_mismatches': [],
              'table_mean_range_mismatches': [], 'table_C2_point_mismatches': [],
              'state_checks': [], 'recomputed_bootstrap': [],
              'bootstrap_seeds': {'fault': 2026100907, 'normal': 2026100908}, 'bootstrap_draws': 20000,
              'historical_Table_S52_bootstrap_stream_identified': True,
              'table_CI_mismatches': [], 'archived_summary_mismatches': []}
    archived_summary = read_csv(args.summary)
    archived_fields = {'loss': 'loss_minutes', 'clean40': 'clean40_pct',
                       'clean8': 'clean8_pct', 'far': 'normal_FAR_pct'}
    if args.manifest:
        states = json.loads(args.manifest.read_text(encoding='utf-8'))['states']
        if [row['state'] for row in states] != [1, 2, 3, 4, 5]:
            raise ValueError('Unexpected manifest replicate order')
        report['shared_run_order'] = {
            role: all(row[role] == states[0][role] for row in states)
            for role in ('fault', 'normal')}
        if not all(report['shared_run_order'].values()):
            raise ValueError('Shared test panel order differs between replicates')
        report['run_ids'] = {role: states[0][role]['ids'] for role in ('fault', 'normal')}
    summaries = {metric: {method: {key: [] for key in COLUMNS.values()}
                 for method in METHODS} for metric in ESTIMATORS}
    differences = {metric: {key: [] for key in COLUMNS.values()} for metric in ESTIMATORS}
    for metric in ESTIMATORS:
        for state in range(1, 6):
            directory = args.a7 / ('state%d_%s' % (state, metric))
            events = load_npz(directory / 'EVALUATION_EVENTS.npz')
            selection = json.loads((directory / 'SELECTION.json').read_text(encoding='utf-8'))
            outer = load_npz(directory / 'OUTER.npz')
            inner = load_npz(directory / 'INNER.npz')
            names = events['methods'].tolist()
            positions = [names.index(method) for method in METHODS]
            loss = events['fault_event_loss_minutes'][:, positions]
            clean = events['fault_clean'][:, positions]
            normal = events['normal_alarm'][:, positions].astype(int)
            if loss.shape != (250, 2) or clean.shape != (250, 2) or normal.shape != (250, 2):
                raise ValueError('Unexpected event matrix shape')
            if not ((loss >= 0) & (loss <= 2400) & (loss % 3 == 0)).all():
                raise ValueError('Invalid three-minute event loss')
            if not (loss[clean == 0] == 2400).all() or not (loss[clean != 0] < 2400).all():
                raise ValueError('Loss/clean event inconsistency')
            values = {'loss': loss, 'clean40': clean * 100,
                      'clean8': (clean.astype(bool) & (loss <= 480)) * 100,
                      'far': normal * 100}
            if int(outer['n']) != 259 or int(outer['rank']) != 247:
                raise ValueError('Unexpected outer calibration sample/rank')
            limits = inner['anchors'] * np.sort(outer['peaks'] / inner['anchors'], axis=0)[246]
            selected = limits[[selection['choices'][method] for method in names]]
            report['state_checks'].append({
                'state': state, 'estimator': metric, 'fault_runs': 250, 'normal_runs': 250,
                'outer_limit_max_abs_error': float(np.max(abs(limits - outer['thresholds']))),
                'selected_limit_max_abs_error': float(np.max(abs(selected - events['thresholds'])))})
            for position, method in enumerate(METHODS):
                row = next(row for row in source_endpoints if row['state'] == str(state)
                           and row['metric'] == metric and row['method'] == method)
                actual = {'threshold': events['thresholds'][positions[position]],
                          'clean40_n': clean[:, position].sum(),
                          'clean40_pct': values['clean40'][:, position].mean(),
                          'clean8_n': (values['clean8'][:, position] / 100).sum(),
                          'clean8_pct': values['clean8'][:, position].mean(),
                          'mean_loss_minutes': loss[:, position].mean(),
                          'normal_alarm_n': normal[:, position].sum(),
                          'normal_FAR_pct': values['far'][:, position].mean()}
                for key, value in actual.items():
                    if not np.isclose(value, float(row[key]), rtol=0, atol=1e-10):
                        report['source_endpoint_mismatches'].append(
                            [state, metric, method, key, float(value), float(row[key])])
                for key, value in values.items():
                    summaries[metric][method][key].append(float(value[:, position].mean()))
            observed = paired_interval(loss[:, 1] - loss[:, 0], 2026100800 + state, 0.05 / 3)
            row = next(row for row in source_contrasts if row['state'] == str(state)
                       and row['metric'] == metric)
            expected = [float(row[key]) for key in
                        ('loss_difference_minutes', 'CI98_333_low', 'CI98_333_high')]
            if not np.allclose(observed, expected, rtol=0, atol=1e-10):
                report['source_replicate_CI_mismatches'].append([state, metric, observed, expected])
            for key, value in values.items():
                differences[metric][key].append(value[:, 1] - value[:, 0])
    for group, metric in enumerate(ESTIMATORS):
        for position, method in enumerate(METHODS):
            for column, key in COLUMNS.items():
                value = np.asarray(summaries[metric][method][key])
                actual = [float(value.mean()), float(value.min()), float(value.max())]
                expected = numbers(table[group * 3 + position][column])
                if not all(round(a, 1) == b for a, b in zip(actual, expected)):
                    report['table_mean_range_mismatches'].append([metric, method, key, actual, expected])
        for column, key in COLUMNS.items():
            # Average the five differences for each physical run before resampling.
            paired = np.asarray(differences[metric][key]).mean(axis=0)
            actual = paired_interval(paired, 2026100908 if key == 'far' else 2026100907)
            expected = numbers(table[group * 3 + 2][column])
            if round(actual[0], 1) != expected[0]:
                report['table_C2_point_mismatches'].append([metric, key, actual[0], expected[0]])
            if not all(round(a, 1) == b for a, b in zip(actual[1:], expected[1:])):
                report['table_CI_mismatches'].append([metric, key, actual[1:], expected[1:]])
            historical = next(row for row in archived_summary if row['B_estimator'] == metric)
            stem = archived_fields[key]
            saved = [float(historical[field]) for field in
                     (stem, stem + '_marginal_CI95_low', stem + '_marginal_CI95_high')]
            if not np.allclose(actual, saved, rtol=0, atol=1e-10):
                report['archived_summary_mismatches'].append([metric, key, actual, saved])
            report['recomputed_bootstrap'].append({
                'estimator': metric, 'endpoint': key, 'mean': actual[0],
                'CI95_low': actual[1], 'CI95_high': actual[2],
                'manuscript_mean': expected[0], 'manuscript_CI95_low': expected[1],
                'manuscript_CI95_high': expected[2],
                'CI_matches_manuscript_at_1_decimal':
                    all(round(a, 1) == b for a, b in zip(actual[1:], expected[1:]))})
    report['interpretation'] = (
        'Five-replicate percentile intervals use the archived summarize_a7.py seeds: '
        'fault 2026100907 and normal 2026100908, NumPy PCG64, 20000 draws. '
        'The same physical-run indices are shared across estimators and fault endpoints. '
        'Intervals are conditional on the fitted replicates and calibration; they do not '
        'measure relearning variability. Event row order follows A7_INPUTS.json.')
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (args.out / 'C2_recomputed_archived_seeds.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['recomputed_bootstrap'][0]))
        writer.writeheader()
        writer.writerows(report['recomputed_bootstrap'])
    checks = ['source_endpoint_mismatches', 'source_replicate_CI_mismatches',
              'table_mean_range_mismatches', 'table_C2_point_mismatches',
              'table_CI_mismatches', 'archived_summary_mismatches']
    print(json.dumps({key: len(report[key]) for key in checks}, indent=2))
    print('Historical Table S52 bootstrap stream: identified; see validation.json.')


if __name__ == '__main__':
    main()
