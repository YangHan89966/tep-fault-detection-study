#!/usr/bin/env python3
"""Dispatch one existing-data analysis; original numerical source stays unchanged."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def configure_plot(out):
    info = {'requested_font': 'Times New Roman', 'requested_dpi': 600,
            'font_files_copied': False, 'status': 'NOT_YET_DRAWN'}
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import font_manager as fm
        info['matplotlib_version'] = matplotlib.__version__
        windows_font = Path('/mnt/c/Windows/Fonts/times.ttf')
        if windows_font.is_file():
            fm.fontManager.addfont(str(windows_font))
            info['windows_font_registered_read_only'] = str(windows_font)
        try:
            info['resolved_font_path'] = fm.findfont(fm.FontProperties(family='Times New Roman'), fallback_to_default=False)
            info['font_status'] = 'REQUESTED_FONT_AVAILABLE'
        except ValueError as exc:
            info['font_status'] = 'REQUESTED_FONT_MISSING'
            info['font_error'] = str(exc)
    except ImportError as exc:
        info['font_status'] = 'MATPLOTLIB_UNAVAILABLE'
        info['font_error'] = str(exc)
    except Exception as exc:
        # A font/cache/backend problem must not prevent the numeric A2 work.
        info['font_status'] = 'PLOT_CONFIGURATION_PROBLEM'
        info['font_error'] = repr(exc)
    dump(out / 'A2_PLOT_STATUS.json', info)
    return info


def instrument_loader(module, receipts):
    original = module.load_role
    def load_role(spec, expected_n, d=52, root='.'):
        print(f"Reading existing {spec.get('role', 'role')}: {expected_n} complete runs", flush=True)
        result = original(spec, expected_n, d=d, root=root)
        files = spec.get('files') or [spec.get('file')]
        hashes = spec.get('sha256s')
        if not hashes or len(files) != len(hashes) or len(files) != len(spec['ids']):
            raise ValueError('The portable R5 role must have one committed hash and global identity per file')
        for name, identity, expected in zip(files, spec['ids'], hashes):
            p = (Path(root) / name).resolve()
            key = str(p)
            if key in receipts:
                if receipts[key]['sha256'] != expected or receipts[key]['canonical_label'] != identity:
                    raise ValueError('Conflicting raw identity on repeated read')
                receipts[key]['read_count'] += 1
            else:
                receipts[key] = {'source': key, 'canonical_label': identity,
                                 'sha256': expected, 'bytes': p.stat().st_size, 'read_count': 1,
                                 'hash_checked_by_original_loader': True}
        print(f"Read/hash-verified {expected_n} runs; unique original files so far: {len(receipts)}", flush=True)
        return result
    module.load_role = load_role


def a2(package, raw_root, out, receipts):
    engine = package / 'engines/A2_A7'
    sys.path.insert(0, str(engine))
    import supervised_raw_analyses as original
    instrument_loader(original, receipts)
    manifest = original.resolve_raw_roots(json.loads((engine / 'A2_INPUTS.json').read_text()), str(raw_root))
    if [r['state'] for r in manifest['states']] != [1, 2, 3, 4, 5]:
        raise ValueError('A2 requires five original states')
    context = {'state': None, 'fits': 0, 'by_family': {}}
    old_load_state = original.load_state
    old_score = original.score_new_axis
    def load_state(row, root, faults):
        context.update(state=row['state'], by_family={})
        print(f"A2 state {row['state']}/5: loading fixed roles", flush=True)
        return old_load_state(row, root, faults)
    def score_new_axis(y, v, family):
        context['fits'] += 1
        count = context['by_family'].get(family, 0) + 1
        context['by_family'][family] = count
        repeat = (count - 1) // 5 + 1
        n = (5, 10, 20, 50, 100)[(count - 1) % 5]
        fault = 3 if family == 'signed' else 15
        print(f"A2 fit {context['fits']}/500: state={context['state']}, fault={fault}, repeat={repeat}/10, n={n}", flush=True)
        return old_score(y, v, family)
    original.load_state = load_state
    original.score_new_axis = score_new_axis
    plot = configure_plot(out)
    plot_exception = None
    try:
        # This calls only A2, avoiding the original CLI's unused raw-role hashing.
        original.a2(manifest, engine, out)
    except Exception as exc:
        # In the unchanged a2 function all three tables are written only after
        # every fit and n=100 gate passes; the sole remaining block is plotting.
        complete_tables = all((out / name).is_file() for name in
                              ('A2_LEARNING_CURVE.csv', 'A2_N100_DIRECTION_REPRODUCTION.csv', 'A2_90PCT_THRESHOLD.csv'))
        if not complete_tables:
            raise
        plot_exception = repr(exc)
        print('A2 numeric tables are complete; plot could not finish: ' + repr(exc), flush=True)
    curve = csv_rows(out / 'A2_LEARNING_CURVE.csv')
    gates = csv_rows(out / 'A2_N100_DIRECTION_REPRODUCTION.csv')
    expected = {(s, f, rep, n) for s in range(1, 6) for f in (3, 15)
                for rep in range(1, 11) for n in (5, 10, 20, 50, 100)}
    actual = {(int(r['state']), int(r['fault']), int(r['repeat']), int(r['n'])) for r in curve}
    if len(curve) != 500 or actual != expected or context['fits'] != 500:
        raise ValueError('A2 did not complete all 500 fixed-subset fits')
    if len(gates) != 100 or any(float(r['relative_direction_error']) > 1e-6 for r in gates):
        raise ValueError('A2 n=100 direction gates incomplete')
    if len(receipts) != 3405:
        raise ValueError('A2 did not read precisely its 3405 unique required raw files')
    plot['status'] = 'COMPLETE_600_DPI_REQUESTED_FONT' if (out / 'A2_LEARNING_CURVE.png').is_file() and plot_exception is None else 'PLOT_PENDING_NUMERIC_DATA_PRESERVED'
    if plot_exception:
        plot['plot_exception'] = plot_exception
    dump(out / 'A2_PLOT_STATUS.json', plot)
    return {'status': 'COMPLETE' if plot['status'].startswith('COMPLETE') else 'COMPLETE_DATA_PLOT_PENDING',
            'scientific_data_status': 'COMPLETE', 'fits': 500, 'n100_direction_checks': 100,
            'unique_raw_files_read': len(receipts), 'plot': plot,
            'subset_rng_definition': 'default_rng(202610080000 + state*100 + repeat).permutation(100); prefixes 5,10,20,50,100',
            'new_TEP_simulations': False, 'full_monitor_optional_branch_run': False}


def a6(package, raw_root, out, receipts):
    engine = package / 'engines/A6'
    sys.path.insert(0, str(engine / 'code'))
    import run_A3_A6_existing as original
    source, rows, commits = original.inputs(engine / 'frozen_inputs')
    store = original.ExistingRaw([raw_root], rows, commits)
    old_role = original.ExistingRaw.role
    def role(self, role_name, fault, state):
        print(f'A6 reading state={state}, role={role_name}, fault={fault}', flush=True)
        value = old_role(self, role_name, fault, state)
        print(f'A6 role complete: {len(value)} runs; unique files {len(self.read_log)}/4155', flush=True)
        return value
    original.ExistingRaw.role = role
    try:
        available = store.availability(original.wanted(rows, 'A6'))
        original.csv_write(out / 'A6_REQUIRED_EXISTING_RAW_INPUTS.csv', available)
        if any(r['status'] == 'MISSING_EXISTING_RAW' for r in available):
            raise FileNotFoundError('A6 original raw files missing')
        assets = source / 'current_study/dense_validation/source_original/assets'
        pca = original.npz_read(assets / 'PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz')
        basis = original.np.load(assets / 'RESIDUAL_BASIS_NEWSEED_PCA.npy', allow_pickle=False)
        inherited = original.module('more_a6_inherited_numerical_only', source / 'current_study/dense_validation/source_original/code/inherited.py')
        # Directly call A6. The original CLI's unconditional saved A3 point is
        # intentionally skipped, and neither its A3 function nor A3 files run.
        result = original.a6(store, engine / 'frozen_inputs', source, pca, basis, inherited, out)
        result.update(store.after_read())
        original.inputs(engine / 'frozen_inputs')
        if result['original_array_read_n'] != 4155 or result['after_read_unchanged_n'] != 4155:
            raise ValueError('A6 complete original-array gate failed')
        if len(csv_rows(out / 'A6_FDA52_VS_RESIDUAL.csv')) != 15 or len(csv_rows(out / 'A6_FROZEN_RESIDUAL_REGRESSION.csv')) != 15:
            raise ValueError('A6 comparison or original FDA regression incomplete')
        result.update(scientific_data_status='COMPLETE', new_TEP_simulations=False, A3_executed=False)
        return result
    finally:
        receipts.update(store.read_log)
        store.close()


def a7(package, raw_root, out, receipts):
    engine = package / 'engines/A2_A7'
    sys.path.insert(0, str(engine))
    import relearn_native as original
    instrument_loader(original, receipts)
    old_build = original.build_one
    old_check = original.check_original
    old_metrics = original.temporal_metrics
    baseline_checks = []
    context = {}
    def build_one(des, mat, nd, directory):
        context['directory'] = directory.name
        print('A7 learning/selection/calibration: ' + directory.name, flush=True)
        return old_build(des, mat, nd, directory)
    def check_original(methods, ef, en, h, saved):
        old_check(methods, ef, en, h, saved)
        order = [list(saved['methods']).index(m) for m in methods]
        error = float(original.np.max(original.np.abs(h - saved['thresholds'][order])))
        baseline_checks.append({'state_metric': context['directory'], 'threshold_max_abs_error': error,
                                'original_fault_loss_clean_and_normal_FAR_vectors': 'PASS'})
        original.csv_write(out / 'A7_ORIGINAL_B_REPLAY.csv', baseline_checks)
        print('A7 original B baseline gate passed: ' + context['directory'], flush=True)
    def metrics(y, bandwidth=120):
        print('A7 estimating original B, 320 run-bounded blocks, and Bartlett HAC (bandwidth 120)', flush=True)
        value = old_metrics(y, bandwidth=bandwidth)
        print('A7 three temporal matrices ready', flush=True)
        return value
    original.build_one = build_one
    original.check_original = check_original
    original.temporal_metrics = metrics
    sys.argv = ['relearn_native.py', 'A7', '--manifest', str(engine / 'A7_INPUTS.json'),
                '--raw-root', str(raw_root), '--out', str(out)]
    original.main()
    endpoint_rows = csv_rows(out / 'A7_ENDPOINTS.csv')
    comparisons = csv_rows(out / 'A7_PAIRED_CONTRASTS.csv')
    diagnostics = csv_rows(out / 'A7_B_EIGENVALUES.csv')
    if len(endpoint_rows) != 30 or len(comparisons) != 15 or len(diagnostics) != 15 or len(baseline_checks) != 5:
        raise ValueError('A7 expected five states times three B estimators are incomplete')
    if len(receipts) != 2825:
        raise ValueError('A7 did not read precisely its 2825 unique original raw files')
    return {'status': 'COMPLETE', 'scientific_data_status': 'COMPLETE', 'endpoint_rows': 30,
            'comparison_rows': 15, 'original_B_replay_checks': 5, 'unique_raw_files_read': len(receipts),
            'B_estimators': ['original', 'batch', 'HAC'], 'new_TEP_simulations': False,
            'batch_definition': '320 nonoverlapping blocks from 40 existing runs; not 320 independent runs',
            'HAC_definition': 'within-run lag pairs only; max(0,1-lag/120); denominator 40*960',
            'A8_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('A2', 'A6', 'A7'), required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root.expanduser().resolve(); out = args.out.expanduser().resolve()
    if out.exists() or out == raw_root or raw_root in out.parents or (package / 'engines') in out.parents:
        raise ValueError('A new output folder outside original inputs is required')
    out.mkdir(parents=True)
    receipts = {}
    try:
        result = {'A2': a2, 'A6': a6, 'A7': a7}[args.task](package, raw_root, out, receipts)
        result.update(task=args.task, exit_code=0, original_scientific_source_modified=False)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result = {'task': args.task, 'status': 'FAILED_NOT_COMPLETE', 'scientific_data_status': 'NOT_COMPLETE',
                  'exit_code': 3, 'error': repr(exc), 'new_TEP_simulations': False,
                  'partial_outputs_retained': True, 'original_scientific_source_modified': False}
        code = 3
    finally:
        dump(out / 'TRAJECTORY_READ_RECEIPTS.json', list(receipts.values()))
    dump(out / 'TASK_STATUS.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
