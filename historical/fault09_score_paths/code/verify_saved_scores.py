#!/usr/bin/env python3
"""Validate supplied scalar paths without raw data, fitting or simulation."""
from pathlib import Path
import argparse, hashlib, io, json, sys, zipfile
import numpy as np
from core import recurrence, normal_rows, event_rows, numeric_records

def main(source, report):
    source = Path(source)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    def read(name):
        return archive.read(name) if archive else (source / name).read_bytes()
    def js(name): return json.loads(read(name))
    def npz(name):
        with np.load(io.BytesIO(read(name)), allow_pickle=False) as data:
            return dict(data)
    checks = []
    def check(name, value):
        if not bool(value): raise ValueError(name)
        checks.append(name)
    try:
        manifest = js('PAYLOAD_MANIFEST.json')
        check('manifest_excludes_hash_manifests', all(x not in manifest for x in
              ('PAYLOAD_MANIFEST.json', 'RESULTS_SHA256.txt')))
        for name, expected in manifest.items():
            check('sha256:' + name, hashlib.sha256(read(name)).hexdigest() == expected)
        listed = {}
        for line in read('RESULTS_SHA256.txt').decode().splitlines():
            h, name = line.split('  ', 1); listed[name] = h
        check('hash_manifests_agree', listed == manifest)
        scope = js('EXPORT_SCOPE.json')
        rows = []; peak_errors = []; exact_peak_checks = 0
        for tag in scope['splits']:
            choices = js(f'inputs/{tag}/CHOICES.json')
            roles = js(f'inputs/{tag}/ROLES.json')
            state = js(f'scores/{tag}/AXIS_ROLES.json')
            packed = npz(f'scores/{tag}/TRANSFORM_AND_THRESHOLDS.npz')
            outer = npz(f'inputs/{tag}/OUTER_RECORDS.npz')
            evaluation = npz(f'inputs/{tag}/EVALUATION_RECORDS.npz')
            indices = np.asarray(state['axis_indices'])
            ktotal = choices['mean_count'] + choices['moment_count']
            check(tag + ':axis_indices', np.array_equal(indices, packed['axis_indices']))
            check(tag + ':candidate_indices', state['candidate_indices_family_major'] ==
                  [int(i + g * ktotal) for g in range(3) for i in indices])
            for src, role in [('normal', 'outer_calibration'), ('normal', 'audit'), ('fault', 'audit')]:
                key = f'{src}_{role}'
                scores = npz(f'scores/{tag}/{key}.npz')
                ids = scores['run_id']; u = scores['u']
                check(tag + ':' + key + ':run_ids', np.array_equal(ids, roles[src][role]))
                check(tag + ':' + key + ':shape', u.shape == (len(ids), 960, len(indices)))
                check(tag + ':' + key + ':axes', np.array_equal(scores['axis_indices'], indices))
                check(tag + ':' + key + ':finite', np.isfinite(u).all())
                for g, family in enumerate(('signed', 'absolute', 'squared')):
                    v = u if family == 'signed' else (
                        (abs(u) - packed['abs_mean']) / packed['abs_sd'] if family == 'absolute'
                        else (u*u - packed['sq_mean']) / packed['sq_sd'])
                    d = recurrence(v, k=state['k'])
                    for pos, index in enumerate(indices):
                        j = int(index + g * ktotal)
                        threshold = packed['thresholds'][g, pos]
                        label = f'{tag}:{key}:{family}:{int(index)}'
                        check(label + ':threshold', threshold == outer['thresholds'][j])
                        if src == 'normal' and role == 'outer_calibration':
                            peaks = d[:, :, pos].max(axis=1)
                            expected = outer['peaks'][:, j]
                            check(label + ':outer_order', np.array_equal(ids, outer['run_id']))
                            check(label + ':outer_peaks', np.allclose(peaks, expected, rtol=1e-9, atol=1e-8))
                            peak_errors.append(float(np.max(abs(peaks - expected))))
                            exact_peak_checks += int(np.array_equal(peaks, expected))
                        else:
                            mask = d[:, :, pos] > threshold
                            result = normal_rows(mask, ids, scope['fault'], 'fixed') if src == 'normal' else event_rows(mask, ids, scope['fault'], role, 'fixed')
                            fields, values = numeric_records(result)
                            check(label + ':fields', list(fields) == list(evaluation[src + '_fields']))
                            check(label + ':integer_outcomes', np.array_equal(values, evaluation[src][j]))
                        rows.append({'split': tag, 'source': src, 'role': role,
                                     'family': family, 'axis_index': int(index), 'passed': True})
        result = {'status': 'PASS_SAVED_SCORE_VERIFICATION', 'split_count': len(scope['splits']),
                  'manifest_members_verified': len(manifest), 'checks_passed': len(checks),
                  'axis_family_role_comparisons': len(rows), 'outer_peak_comparisons': len(peak_errors),
                  'outer_peak_comparisons_exact': exact_peak_checks,
                  'outer_peak_max_abs_error': max(peak_errors, default=0),
                  'normal_and_fault_integer_outcomes_exact': True,
                  'raw_source_export_performed': False, 'results': rows}
        Path(report).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(result['status'], 'comparisons=' + str(len(rows)), 'checks=' + str(len(checks)))
    finally:
        if archive: archive.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', default='saved_score_verification.json')
    args = parser.parse_args(); main(args.source, args.report)
