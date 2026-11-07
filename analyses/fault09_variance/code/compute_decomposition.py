"""Offline variance decomposition of the archived 400 normal TEP trajectories.

No process simulation, model fitting, threshold tuning, or fault evaluation.
Requires the original normal-evaluation zip and the extracted graded-coupling
package. Existing source code and inputs are read only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

FLOW = np.array([1, 2, 3, 4, 5, 6, 10, 14, 17, 19]) - 1
FEATURES = [
    'rho0', 'controller_input', 'post_update',
    'post_continuous_to_CI', 'post_analyzers_to_CI', 'post_all41_to_CI',
    'post_XMEAS17_to_CI', 'post_all10flows_to_CI', 'post_continuous_except17_to_CI',
    'bridge_continuous', 'bridge_analyzers', 'bridge_all', 'bridge_XMEAS17',
    'rho0_group_C', 'rho0_group_A', 'rho0_group_U',
    'post_group_C', 'post_group_A', 'post_group_U',
    'signal_delta_XMEAS17', 'signal_delta_continuous_except17', 'signal_delta_analyzers',
    'post_noise_score', 'independent_noise_score', 'CI_noiseless_score',
    'CI_noise15_standardized', 'signal_delta17_raw'
]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def initialize(coupling_root, raw_zip, out):
    global ROOT, ZIPP, OUTPUT, recover, fetch_events, sequential_noise, replacement_uniforms
    global replacement_key, fixed, NOISE_SD, file_index
    ROOT, ZIPP, OUTPUT = Path(coupling_root), Path(raw_zip), Path(out)
    sys.path.insert(0, str(ROOT / 'code'))
    from recover_noise import recover, _fetch_events as fetch_events, NOISE_SD
    from run_coupling import sequential_noise, replacement_uniforms, replacement_key
    from score_core import load_historical_map
    fixed = load_historical_map(ROOT / 'assets/FIXED_AFFINE_MAPS.npz')
    with (ROOT / 'reference/FILE_MANIFEST.csv').open(encoding='utf-8-sig', newline='') as f:
        file_index = {x['path']: x for x in csv.DictReader(f)}


def one_run(run_id):
    started = time.perf_counter()
    dest = OUTPUT / 'checkpoints' / (run_id + '.npz')
    jdest = dest.with_suffix('.json')
    if dest.exists() and jdest.exists():
        saved = json.loads(jdest.read_text())
        if sha_file(dest) == saved['checkpoint_sha256']:
            return saved
        raise ValueError('Checkpoint identity mismatch ' + run_id)
    with zipfile.ZipFile(ZIPP) as archive:
        by = {name: archive.read('independent_stream_panel/runs/' + run_id + '/' + name)
              for name in ['RAW.npz', 'RUN_INFO.json']}
    for name, data in by.items():
        rec = file_index['runs/' + run_id + '/' + name]
        if sha_bytes(data) != rec['sha256'] or len(data) != int(rec['size_bytes']):
            raise ValueError('Original input identity mismatch ' + run_id + '/' + name)
    with np.load(io.BytesIO(by['RAW.npz']), allow_pickle=False) as a:
        raw = {k: a[k] for k in a.files}
    meta = json.loads(by['RUN_INFO.json'])
    if meta['run_id'] != run_id or meta['role'] != 'normal_evaluation' or meta['fault_id'] != 0:
        raise ValueError('Wrong population ' + run_id)
    native = raw['native_values']
    ci = native.copy()
    ci[:, :41] = raw['fast_control_input_measurements']
    if not np.array_equal(ci, raw['controller_input_values']):
        raise ValueError('CI reconstruction differs from full saved record')
    if not np.array_equal(native[:, 41:], ci[:, 41:]):
        raise ValueError('Manipulated variables are not identical')
    rec = recover(raw, meta, verify_full_rng_trace=False)
    en_ci = sequential_noise(rec['original_uniforms'], NOISE_SD, rec['has_noise'])
    if not np.array_equal(en_ci, rec['input_noise']):
        raise ValueError('CI noise summation differs')
    native_ids = rec['native_noise_event_id']
    # The continuous noise block ends each saved model evaluation.
    expected_native = raw['diagnostic_rng_draw_count'][:, None] - 264 + 12 * np.arange(22)[None, :]
    if not np.array_equal(native_ids[:, :22], expected_native):
        raise ValueError('Independent native-event identity check failed')
    native_uniforms = fetch_events(int(meta['rng_key_hex'], 16), native_ids)
    en_post = sequential_noise(native_uniforms, NOISE_SD, native_ids >= 0)
    independent_uniforms = replacement_uniforms(replacement_key(run_id), rec['noise_event_id'])
    en_ind = sequential_noise(independent_uniforms, NOISE_SD, rec['has_noise'])
    rho0 = ci.copy()
    rho0[:, :41] += en_ind - en_ci
    signal_ci = ci[:, :41] - en_ci
    signal_post = native[:, :41] - en_post
    dm = native[:, :41] - ci[:, :41]
    dn = en_post - en_ci
    ds = signal_post - signal_ci
    identity_error = float(np.max(np.abs(dm - dn - ds)))
    if identity_error > 2e-10:
        raise ValueError('Measurement decomposition failed ' + str(identity_error))
    beta, intercept = fixed['beta'], fixed['intercept']
    s0, sci, sp = (x @ beta + intercept for x in [rho0, ci, native])
    scorefile = ROOT / 'scores' / (run_id + '.npz')
    scoremeta = json.loads(scorefile.with_suffix('.json').read_text())
    if sha_file(scorefile) != scoremeta['score_sha256']:
        raise ValueError('Archived coupling score hash mismatch')
    with np.load(scorefile, allow_pickle=False) as a:
        old = a['scores'][:, [0, 4, 5]]
    replay_error = float(np.max(np.abs(old - np.column_stack([s0, sci, sp]))))
    if replay_error > 1e-9:
        raise ValueError('Three original score endpoints differ ' + str(replay_error))
    def substitute(indices):
        view = native.copy()
        view[:, indices] = ci[:, indices]
        return view @ beta + intercept
    dc = ds[:, :22] @ beta[:22]
    da = ds[:, 22:] @ beta[22:41]
    d17 = ds[:, 16] * beta[16]
    dco = dc - d17
    cexcept = np.delete(np.arange(22), 16)
    group0 = [rho0[:, :22] @ beta[:22], rho0[:, 22:41] @ beta[22:41], rho0[:, 41:] @ beta[41:]]
    groupp = [native[:, :22] @ beta[:22], native[:, 22:41] @ beta[22:41], native[:, 41:] @ beta[41:]]
    signal_view = ci.copy()
    signal_view[:, :41] = signal_ci
    vals = [s0, sci, sp, substitute(np.arange(22)), substitute(np.arange(22, 41)),
            substitute(np.arange(41)), substitute([16]), substitute(FLOW), substitute(cexcept),
            s0 + dc, s0 + da, s0 + dc + da, s0 + d17,
            *group0, *groupp, d17, dco, da,
            en_post @ beta[:41], en_ind @ beta[:41], signal_view @ beta + intercept,
            en_ci[:, 14] / NOISE_SD[14], ds[:, 16]]
    z = np.column_stack(vals)
    assert z.shape == (960, len(FEATURES)) and np.isfinite(z).all()
    means = z.mean(axis=0)
    centered = z - means
    cross = centered.T @ centered
    v = np.stack([dm, dn, ds], axis=-1) / NOISE_SD[None, :, None]
    vmean = v.mean(axis=0)
    vc = v - vmean
    vcross = np.einsum('tji,tjk->jik', vc, vc)
    channel_signal_diff_max = np.max(np.abs(ds), axis=0)
    channel_measurement_diff_max = np.max(np.abs(dm), axis=0)
    event_changed_n = np.sum(native_ids != rec['noise_event_id'], axis=0)
    np.savez_compressed(dest, feature_names=np.asarray(FEATURES), means=means, cross=cross,
                        variable_means=vmean, variable_cross=vcross,
                        channel_signal_diff_max=channel_signal_diff_max,
                        channel_measurement_diff_max=channel_measurement_diff_max,
                        event_changed_n=event_changed_n, noise_sd=NOISE_SD,
                        n=np.asarray(960), run_id=np.asarray(run_id))
    report = {'run_id': run_id, 'status': 'PASS', 'role': meta['role'],
              'source_raw_sha256': sha_bytes(by['RAW.npz']), 'source_info_sha256': sha_bytes(by['RUN_INFO.json']),
              'source_score_sha256': sha_file(scorefile),
              'score_replay_max_abs_error': replay_error, 'decomposition_max_abs_error': identity_error,
              'analyzer_changed_cells': int(np.count_nonzero(dm[:, 22:])),
              'analyzer_event_changed_cells': int(np.sum(event_changed_n[22:])),
              'xmv_changed_cells': 0, 'native_continuous_event_check': 'PASS',
              'recovery': rec['diagnostics'], 'seconds': time.perf_counter() - started,
              'checkpoint_sha256': sha_file(dest)}
    dump_json(jdest, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coupling-root', required=True)
    p.add_argument('--raw-zip', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    out, source = Path(args.out), Path(args.coupling_root)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    raw_hash = sha_file(args.raw_zip)
    if raw_hash != '5b61386a96e5709b1cd1de52aaac3c0f63d906543bbf80e21abae1e13947dbf9':
        raise ValueError('Normal-evaluation archive does not match original input')
    with (source / 'MANIFEST.csv').open(encoding='utf-8-sig', newline='') as f:
        cases = [x for x in csv.DictReader(f) if x['role'] == 'normal_evaluation']
    ids = [x['run_id'] for x in cases]
    if len(ids) != 400 or len(set(ids)) != 400:
        raise ValueError('Expected all400 normal evaluation runs')
    started = time.perf_counter()
    reports = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                                initargs=(args.coupling_root, args.raw_zip, args.out)) as pool:
        for report in pool.map(one_run, ids):
            reports.append(report)
            if len(reports) % 40 == 0:
                print(json.dumps({'completed': len(reports), 'total': 400, 'seconds': round(time.perf_counter()-started, 1)}, ensure_ascii=False), flush=True)
    dump_json(out / 'results/RECONSTRUCTION_AUDIT.json', {
        'status': 'PASS', 'n_runs': 400, 'n_observations': 384000,
        'source_archive_sha256': raw_hash, 'compute_script_sha256': sha_file(__file__),
        'fixed_map_sha256': sha_file(source / 'assets/FIXED_AFFINE_MAPS.npz'),
        'max_score_replay_error': max(x['score_replay_max_abs_error'] for x in reports),
        'max_measurement_decomposition_error': max(x['decomposition_max_abs_error'] for x in reports),
        'analyzer_changed_cells': sum(x['analyzer_changed_cells'] for x in reports),
        'analyzer_event_changed_cells': sum(x['analyzer_event_changed_cells'] for x in reports),
        'xmv_changed_cells': 0, 'rng_record_anchors_checked': 384000,
        'physical_simulation_performed': False, 'fitting_or_recalibration_performed': False,
        'elapsed_seconds': time.perf_counter()-started, 'runs': reports})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
