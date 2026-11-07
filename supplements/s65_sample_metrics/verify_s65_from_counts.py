#!/usr/bin/env python3
"""Verify archived S65 aggregates from archived per-run counts; no simulation.

Requires Python 3.8+ and NumPy. This is a verification utility added on
2026-11-07, not the original alarm-bit analysis. Output is written only to
the requested report file; archived scientific files are never modified.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

FAULT_METRICS = ['postfault_sample_FDR_pct', 'prefault_sample_FAR_pct',
                 'clean40_pct', 'clean8_pct', 'mean_loss_min',
                 'counterfactual_latched_FDR_pct', 'latching_excess_FDR_pp']
NORMAL_METRICS = ['normal_sample_FAR_pct', 'normal_any_alarm_pct',
                  'normal_detection_floor_pct']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def display_round(value, digits):
    # Decimal half-up avoids the binary float representation of 72.975
    # being formatted as 72.97 rather than the submitted decimal 72.98.
    return format(Decimal(str(value)).quantize(Decimal(10) ** -digits,
                                               rounding=ROUND_HALF_UP), '.%df' % digits)


def check_close(got, expected, label, stats, tolerance=1e-9):
    error = abs(float(got) - float(expected))
    stats['numeric_checks'] += 1
    stats['maximum_absolute_difference'] = max(stats['maximum_absolute_difference'], error)
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError((label, got, expected, error))


def bootstrap(vectors, role, fault, scenario):
    """Same fixed seed and whole-run resampling protocol as archived analysis."""
    entropy = [2026100905, 1 if role == 'fault' else 2,
               fault if role == 'fault' else 0,
               0 if scenario == 'original' else 1]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    means = np.empty((20000, vectors.shape[1]))
    digest = hashlib.sha256()
    for start in range(0, 20000, 500):
        indices = rng.integers(0, len(vectors), (500, len(vectors)), dtype=np.int32)
        digest.update(indices.astype('<i4', copy=False).tobytes())
        means[start:start + 500] = vectors[indices].mean(1)
    low, high = np.quantile(means, [0.025, 0.975], axis=0)
    return low, high, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = read_csv(root / 'results/R5_ACTUAL_ALARM_COUNTS_PER_RUN.csv')
    assert len(rows) == 40000, len(rows)
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        fault, state, run = (int(row[k]) for k in ['fault', 'state', 'run_number'])
        key = (fault, row['family'], row['method'], row['role'], state)
        identity = key + (run,)
        assert identity not in seen, identity
        seen.add(identity)
        assert row['self_overlap'] in ['True', 'False']
        pre, post, whole, latched, below = [int(row[k]) for k in [
            'prefault_above_points', 'postfault_above_points', 'whole_run_above_points',
            'counterfactual_latched_postfault_above_points',
            'actual_below_points_after_first_postalarm']]
        assert 0 <= pre <= 160 and 0 <= post <= 800
        assert whole == pre + post and post <= latched <= 800 and 0 <= below <= 800
        assert int(row['prefault_any_alarm']) == int(pre > 0)
        assert int(row['any_alarm']) == int(whole > 0)
        clean = int(row['clean_detection_or_normal_floor'])
        clean8 = int(row['clean8_or_normal_floor8'])
        loss = int(row['penalized_loss_min'])
        assert clean == int(pre == 0 and post > 0)
        assert loss % 3 == 0 and 0 <= loss <= 2400
        assert (clean and loss < 2400) or (not clean and loss == 2400)
        assert clean8 == int(clean and loss <= 480)
        if row['role'] == 'fault':
            values = [post / 8., pre / 1.6, clean * 100., clean8 * 100.,
                      loss, latched / 8., (latched - post) / 8.]
        else:
            assert row['role'] == 'normal'
            values = [whole / 9.6, int(whole > 0) * 100., clean * 100.]
        grouped[key].append((run, row['canonical_run_id'],
                             row['self_overlap'] == 'True', values, row))
    matrices = {}
    identities = {}
    exclusions = {}
    thresholds = {}
    for key, group in grouped.items():
        group.sort(key=lambda item: item[0])
        assert [item[0] for item in group] == list(range(1, 251)), key
        base = key[:-1]
        identity = [(item[0], item[1], item[2]) for item in group]
        if base in identities:
            assert identities[base] == identity, ('state_identity_mismatch', key)
        else:
            identities[base] = identity
            exclusions[base] = np.array([item[2] for item in group])
        matrices[key] = np.array([item[3] for item in group], dtype=float)
        threshold = set(item[4]['threshold'] for item in group)
        assert len(threshold) == 1, key
        thresholds[key] = float(next(iter(threshold)))
    stats = dict(numeric_checks=0, maximum_absolute_difference=0.0,
                 count_rows=len(rows), unique_count_keys=len(seen),
                 state_role_groups=len(grouped), bootstrap_draws=20000,
                 simulation_runs=0, alarm_bits_replayed=False)
    bs_hashes = {}
    aggregates = read_csv(root / 'results/R5_SAMPLE_AND_RUN_METRICS_FIVE_STATE.csv')
    state_rows = read_csv(root / 'results/R5_SAMPLE_AND_RUN_METRICS_BY_STATE.csv')
    states_by_key = {(r['scenario'], int(r['fault']), r['family'], r['method'], int(r['state'])): r
                     for r in state_rows}
    assert len(aggregates) == 32 and len(state_rows) == 160
    aggregate_by_key = {}
    for row in aggregates:
        fault, scenario = int(row['fault']), row['scenario']
        aggregate_by_key[(scenario, fault, row['method'])] = row
        for role, names in [('fault', FAULT_METRICS), ('normal', NORMAL_METRICS)]:
            base = (fault, row['family'], row['method'], role)
            matrix = np.stack([matrices[base + (s,)] for s in range(1, 6)])
            keep = np.ones(250, dtype=bool) if scenario == 'original' else ~exclusions[base]
            assert int(keep.sum()) == int(row['unique_' + role + '_runs'])
            matrix = matrix[:, keep]
            vectors = matrix.mean(0)
            low, high, digest = bootstrap(vectors, role, fault, scenario)
            bs_hashes['%s_%d_%s' % (role, fault if role == 'fault' else 0, scenario)] = digest
            for j, name in enumerate(names):
                values = [vectors[:, j].mean(), low[j], high[j],
                          matrix[:, :, j].mean(1).min(), matrix[:, :, j].mean(1).max()]
                for suffix, value in zip(['', '_lo95', '_hi95', '_state_min', '_state_max'], values):
                    check_close(value, row[name + suffix], name + suffix, stats)
                for state in range(1, 6):
                    state_row = states_by_key[(scenario, fault, row['family'], row['method'], state)]
                    check_close(matrix[state - 1, :, j].mean(), state_row[name], name, stats)
                    check_close(thresholds[base + (state,)], state_row['threshold'], 'threshold', stats)
    contrasts = read_csv(root / 'results/R5_SAMPLE_CONTRASTS_PAIRED_95.csv')
    assert len(contrasts) == 12
    method_family = {(r['fault'], r['method']): r['family'] for r in aggregates}
    for row in contrasts:
        fault, scenario = int(row['fault']), row['scenario']
        for role, names in [('fault', FAULT_METRICS), ('normal', NORMAL_METRICS)]:
            bases = [(fault, method_family[(str(fault), row['monitor_' + code])], row['monitor_' + code], role)
                     for code in ['A', 'B']]
            assert identities[bases[0]] == identities[bases[1]]
            keep = np.ones(250, dtype=bool) if scenario == 'original' else ~exclusions[bases[0]]
            a, b = [np.stack([matrices[base + (s,)] for s in range(1, 6)])[:, keep] for base in bases]
            difference = (a - b).mean(0)
            low, high, unused_digest = bootstrap(difference, role, fault, scenario)
            for j, name in enumerate(names):
                short = name
                for suffix in ['_pct', '_pp', '_min']:
                    if short.endswith(suffix):
                        short = short[:-len(suffix)]
                field = 'delta_' + short + ('_min' if name == 'mean_loss_min' else '_pp')
                for suffix, value in zip(['', '_lo95', '_hi95'], [difference[:, j].mean(), low[j], high[j]]):
                    check_close(value, row[field + suffix], field + suffix, stats)
    diagnostic = read_csv(root / 'results/R5_ACTUAL_VS_LATCHED_OCCUPANCY_DIAGNOSTIC.csv')
    for row in diagnostic:
        source = aggregate_by_key[(row['scenario'], int(row['fault']), row['method'])]
        for field, original in [('actual_postfault_sample_FDR_pct', 'postfault_sample_FDR_pct'),
                                ('counterfactual_latched_postfault_FDR_pct', 'counterfactual_latched_FDR_pct'),
                                ('latching_excess_FDR_pp', 'latching_excess_FDR_pp')]:
            check_close(row[field], source[original], field, stats)
        relevant = [r for r in rows if r['fault'] == row['fault'] and r['method'] == row['method']
                    and r['role'] == 'fault' and (row['scenario'] == 'original' or r['self_overlap'] == 'False')]
        clean = sum(int(r['clean_detection_or_normal_floor']) for r in relevant)
        returned = sum(int(r['clean_detection_or_normal_floor']) and int(r['actual_below_points_after_first_postalarm']) > 0 for r in relevant)
        check_close(clean, row['clean_state_run_pairs'], 'clean_pairs', stats)
        check_close(returned, row['clean_state_run_pairs_with_return_below_h'], 'returned_pairs', stats)
    compact = read_csv(root / 'results/R5_SAMPLE_METRICS_COMPACT.csv')
    assert len(compact) == len(aggregates)
    for a, b in zip(compact, aggregates):
        assert all(a[k] == b[k] for k in a), ('compact_row_mismatch', a)
    # Verify the submitted table's displayed rounding, preserving its row order.
    table_path = root / 'reference/Table_S65.csv'
    table_checks = []
    if table_path.exists():
        submitted = read_csv(table_path)
        original = [r for r in aggregates if r['scenario'] == 'original']
        assert len(submitted) == len(original) == 16
        for published, source in zip(submitted, original):
            expected = '%s [%s, %s]' % tuple(display_round(source[k], 2) for k in
                ['postfault_sample_FDR_pct', 'postfault_sample_FDR_pct_lo95', 'postfault_sample_FDR_pct_hi95'])
            assert published['Post-fault sample-level detection (%) [95% interval]'] == expected
            for display, field, digits in [
                ('Normal sample-level FAR (%)', 'normal_sample_FAR_pct', 2),
                ('40 h clean detection (%)', 'clean40_pct', 1),
                ('Normal run-level FAR (%)', 'normal_any_alarm_pct', 1)]:
                check_close(float(published[display]), display_round(source[field], digits), display, stats)
            table_checks.append(dict(fault=int(source['fault']), method=source['method'], displayed_detection=expected))
    spec = json.loads((root / 'results/ANALYSIS_SPEC.json').read_text())
    for key, expected in spec['resampling'].items():
        scenario, role, fault_label = key.split('|')
        hash_key = '%s_%d_%s' % (role, int(fault_label[1:]), scenario)
        assert bs_hashes[hash_key] == expected['indices_sha256_le_int32'], key
    stats.update(all_passed=True, state_rows=160, five_state_rows=32, contrast_rows=12,
                 bootstrap_index_hashes=bs_hashes, submitted_table_rows_checked=len(table_checks),
                 archived_bootstrap_index_hashes_verified=len(spec['resampling']),
                 submitted_table_checks=table_checks,
                 scope='Archived integer counts to aggregates and confidence intervals; no reconstruction of time-ordered alarm bits.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ['bootstrap_index_hashes', 'submitted_table_checks']}, indent=2))


if __name__ == '__main__':
    main()
