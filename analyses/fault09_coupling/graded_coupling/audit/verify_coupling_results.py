"""Independent numerical audit of the frozen graded-noise-coupling analysis.

No production experiment or numerical-analysis modules are imported. The main
CUSUM reconstruction uses cumulative sums and running minima in extended
precision. A separate literal scalar recurrence checks floating-point alarm
equivalence. Bootstrap replication uses multiplicity weights and integer run
differences instead of the production gather-and-average implementation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ARMS = ('rho_0', 'rho_025', 'rho_05', 'rho_075', 'rho_1', 'actual_post_update')
CONTRASTS = ((0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (0, 5))
ENDPOINTS = ('clean40', 'clean8', 'clean2', 'loss_minutes', 'normal_alarm', 'normal_floor')
STUDY = 'TEP_P1_F9_GRADED_NOISE_COUPLING_20261030_v1'
TOL = 1e-9


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b, label):
    require(np.array_equal(a, b), label)


def close(a, b, label, tolerance=TOL):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, label + ' shape')
    require(np.isfinite(a).all() and np.isfinite(b).all(), label + ' finite')
    error = float(np.max(np.abs(a.astype(np.longdouble) - b.astype(np.longdouble)))) if a.size else 0.
    require(error <= tolerance, label + f' absolute difference {error:.12g}')
    return error


def reflected_cusum(scores):
    """Mathematically independent solution of the two Lindley recurrences."""
    x = np.asarray(scores, dtype=np.longdouble)
    zeros = np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.longdouble)
    paths = []
    for sign in (1, -1):
        cumulative = np.cumsum(sign * x - np.longdouble(.05), axis=1)
        running_minimum = np.minimum.accumulate(np.concatenate((zeros, cumulative), axis=1), axis=1)[:, 1:]
        paths.append(cumulative - running_minimum)
    return np.maximum(paths[0], paths[1])


def scalar_cusum(scores):
    """Literal per-arm arithmetic used only as a floating-point cross-check."""
    out = np.empty_like(scores, dtype=np.float64)
    for run in range(scores.shape[0]):
        for arm in range(scores.shape[2]):
            positive = negative = 0.
            for index, value in enumerate(scores[run, :, arm]):
                positive = max(0., (positive + float(value)) - .05)
                negative = max(0., (negative - float(value)) - .05)
                out[run, index, arm] = max(positive, negative)
    return out


def independent_events(alarm):
    n, _, arms = alarm.shape
    result = {name: np.zeros((n, arms), dtype=np.int64) for name in
              ('any_alarm', 'pre_alarm', 'post_alarm', 'clean40', 'clean8', 'clean2',
               'first_post_delay_minutes', 'loss_minutes')}
    for run in range(n):
        for arm in range(arms):
            hits = np.flatnonzero(alarm[run, :, arm])
            before = bool(np.any(hits < 160))
            after = hits[hits >= 160]
            delay = 2400 if len(after) == 0 else 3 * (int(after[0]) - 160)
            clean = not before and len(after) > 0
            values = (bool(len(hits)), before, bool(len(after)), clean,
                      clean and delay <= 480, clean and delay <= 120,
                      delay, delay if clean else 2400)
            for name, value in zip(result, values):
                result[name][run, arm] = value
    return result


def wilson(count, n):
    z = 1.959963984540054
    centre = (count + z*z/2) / (n + z*z)
    radius = z * np.sqrt(count * (n-count) / n + z*z/4) / (n+z*z)
    return max(0., centre-radius), min(1., centre+radius)


def weighted_bootstrap(events, indices, contrasts, scale):
    differences = np.column_stack([events[:, a]-events[:, b] for a, b in contrasts]).astype(np.int64) * scale
    draws, n = indices.shape
    result = np.empty((draws, len(contrasts)), dtype=np.float64)
    for first in range(0, draws, 250):
        batch = indices[first:first+250]
        weights = np.vstack([np.bincount(row, minlength=n) for row in batch]).astype(np.int64)
        result[first:first+len(batch)] = (weights @ differences) / n
    return result, differences.sum(axis=0) / n


def sorted_percentiles(values):
    ordered = np.sort(values, axis=0)
    answers = []
    for probability in (.025, .975):
        position = (len(ordered)-1) * probability
        lower = int(np.floor(position)); fraction = position-lower
        upper = min(lower+1, len(ordered)-1)
        answers.append((1-fraction)*ordered[lower] + fraction*ordered[upper])
    return np.stack(answers)


def audit(root, output, source_root=None):
    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    if source_root is not None:
        source_root = Path(source_root).resolve()
    protocol = read_json(root/'PROTOCOL.json')
    protocol_hash = digest(root/'PROTOCOL.json')
    require(protocol['study_id'] == STUDY, 'Study identity')
    require(protocol_hash == read_json(root/'PROTOCOL_LOCK.json')['protocol_sha256'], 'Frozen protocol hash')
    code_lock = read_json(root/'CODE_LOCK.json')
    code_hash = digest(root/'CODE_LOCK.json')
    # The public distribution documents one path-only test-file derivation.
    # Its verifier checks all historical identities and all current public bytes.
    public_spec = importlib.util.spec_from_file_location('_public_coupling_identity', root/'public_identity.py')
    public_identity = importlib.util.module_from_spec(public_spec)
    public_spec.loader.exec_module(public_identity)
    public_identity.verify_frozen_files(root)
    cases = sorted(read_csv(root/'MANIFEST.csv'), key=lambda c: int(c['ordinal']))
    require(len(cases) == 799 and len({c['run_id'] for c in cases}) == 799, 'Unique complete panel')
    require(digest(root/'MANIFEST.csv') == protocol['panel']['manifest_sha256'], 'Manifest binding')
    groups = {role: [c for c in cases if c['role'] == role] for role in
              ('normal_calibration', 'normal_evaluation', 'fault_evaluation')}
    require({k: len(v) for k, v in groups.items()} ==
            {'normal_calibration':199, 'normal_evaluation':400, 'fault_evaluation':200}, 'Panel allocation')
    lock = read_json(root/'results/THRESHOLDS_LOCK.json')
    lock_hash = digest(root/'results/THRESHOLDS_LOCK.json')
    require(lock['status'] == 'LOCKED', 'Threshold lock status')
    require(lock['binding']['study_id'] == STUDY, 'Threshold study binding')
    require(lock['binding']['protocol_sha256'] == protocol_hash, 'Threshold protocol binding')
    require(lock['binding']['code_lock_sha256'] == code_hash, 'Threshold code binding')
    require(lock['binding']['manifest_sha256'] == digest(root/'MANIFEST.csv'), 'Threshold manifest binding')
    require(lock['binding']['analysis_code_sha256'] == digest(root/'code/analyze_coupling.py'), 'Analysis identity')
    h = np.asarray(lock['thresholds'], dtype=np.float64)
    require(h.shape == (6,) and np.isfinite(h).all(), 'Six finite thresholds')
    require(lock['arm_names'] == list(ARMS), 'Threshold arm order')
    frozen_input_files = {x['path']: x for x in read_csv(root/'reference/FILE_MANIFEST.csv')}
    saved_events = {(r['run_id'], r['arm']): r for r in read_csv(root/'results/Per_run_events.csv')}
    require(len(saved_events) == 600*6, 'Unique evaluation event rows')
    original_events = {(r['run_id'], r['arm_id']):r for r in read_csv(root/'reference/per_run_events.csv')}
    original_lock = read_json(root/'reference/THRESHOLDS_LOCK.json')
    original_h = np.asarray(original_lock['thresholds'])[[21,20]]
    same(h[[4,5]], original_h, 'Original endpoint thresholds')
    expected_cal_ids = [r['run_id'] for r in groups['normal_calibration']]
    same(np.asarray(expected_cal_ids), np.asarray([r['run_id'] for r in lock['calibration_score_files']]), 'Calibration file ordering')
    frozen_score_hashes = {r['run_id']: r['sha256'] for r in lock['calibration_score_files']}
    source_files_checked = original_alarm_bits = noise_anchors = source_trace_steps = 0
    max_recorded_chart_error = 0.
    score_arrays = {}
    reference_endpoints = {}
    replacement_keys = set()
    max_temp_error = 0.
    for case in cases:
        rid = case['run_id']
        report = read_json(root/'scores'/(rid+'.json'))
        require(report['status'] == 'COMPLETE' and report['run_id'] == rid, 'Completed score identity '+rid)
        require(report['protocol_sha256'] == protocol_hash and report['code_lock_sha256'] == code_hash, 'Score protocol/code binding '+rid)
        expected_lock = None if case['role'] == 'normal_calibration' else lock_hash
        require(report['threshold_lock_sha256_at_generation'] == expected_lock, 'Generation threshold binding '+rid)
        require(report['original_key_hex'] == case['key_hex'], 'Original stream key '+rid)
        key = int.from_bytes(hashlib.sha256((STUDY+'|replacement|'+rid).encode()).digest()[:16], 'big')
        require(report['replacement_key_hex'] == f'0x{key:032x}', 'Replacement stream key '+rid)
        replacement_keys.add(key)
        score_path = root/'scores'/(rid+'.npz')
        score_hash = digest(score_path)
        require(report['score_sha256'] == score_hash, 'Score hash '+rid)
        if rid in frozen_score_hashes:
            require(frozen_score_hashes[rid] == score_hash, 'Calibration score immutable '+rid)
        item = read_npz(score_path)
        require(item['scores'].shape == (960,6) and item['scores'].dtype == np.float64 and np.isfinite(item['scores']).all(), 'Finite score matrix '+rid)
        require(tuple(item['arm_names']) == ARMS, 'Score arm order '+rid)
        score_arrays[rid] = item['scores']
        rec = report['noise_recovery']
        require(rec['status'] == 'PASS_EXACT_RNG_SCHEDULE_RECOVERY', 'Noise recovery status '+rid)
        require(rec['record_count_checks'] == 960 and rec['sample_times_bitwise_equal'] is True, 'Noise anchors '+rid)
        require(len(rec['full_rng_trace_sha256']) == 64, 'Complete RNG trace recorded '+rid)
        require(rec['key_hex'] == case['key_hex'], 'Recovered original stream identity '+rid)
        require(rec['temperature_max_abs_error'] <= 1e-12, 'Recovered fault temperature '+rid)
        max_temp_error = max(max_temp_error, rec['temperature_max_abs_error'])
        require(report['original_alarm_bits_checked'] == 1920, 'Endpoint alarm replay accounting '+rid)
        max_recorded_chart_error = max(max_recorded_chart_error, report['endpoint_chart_max_abs_error'])
        noise_anchors += rec['record_count_checks']; source_trace_steps += 172800
        for name, expected_hash in report['source_hashes'].items():
            relative = 'runs/'+rid+'/'+name
            require(expected_hash == frozen_input_files[relative]['sha256'], 'Source identity declaration '+relative)
            if source_root is not None:
                p = source_root/relative
                require(p.stat().st_size == int(frozen_input_files[relative]['size_bytes']) and digest(p) == expected_hash, 'Source bytes '+relative)
                source_files_checked += 1
        if source_root is not None:
            meta = read_json(source_root/'runs'/rid/'RUN_INFO.json')
            require(rec['full_rng_trace_sha256'] == meta['rng_consumption_trace_sha256'], 'Source complete RNG trace '+rid)
            require(rec['final_draw_count'] == meta['rng_draw_count'], 'Source final RNG count '+rid)
            require(rec['final_uniform_u32'] == meta['rng_state']['storage']['last_consumed_u32'], 'Source last uniform '+rid)
            reference_endpoints[rid] = read_npz(source_root/'runs'/rid/'CHARTS.npz')['charts'][:,[21,20]]
    require(len(replacement_keys) == 799, 'Replacement keys distinct')
    original_keys = {int(r['key_hex'],16) for r in read_csv(root/'reference/MANIFEST.csv')}
    require(not (replacement_keys & original_keys), 'Separate replacement stream namespace')
    reconstructed, scalar, events_by_group = {}, {}, {}
    maximum_math_discrepancy = 0.
    alarm_bits = 0
    for role, members in groups.items():
        scores = np.stack([score_arrays[c['run_id']] for c in members])
        reflected = reflected_cusum(scores)
        literal = scalar_cusum(scores)
        maximum_math_discrepancy = max(maximum_math_discrepancy, close(reflected,literal,'Closed form and literal recurrence '+role))
        # Use literal arithmetic for exact equality at calibration-order-statistic ties.
        if role != 'normal_calibration':
            same(reflected>h, literal>h, 'All evaluation alarm bits '+role)
            alarm_bits += literal.size
        reconstructed[role], scalar[role] = reflected, literal
        if source_root is not None:
            for i, c in enumerate(members):
                old = reference_endpoints[c['run_id']]
                same(literal[i,:,[4,5]].T > original_h, old > original_h, 'Archived endpoint alarm bits '+c['run_id'])
                original_alarm_bits += 1920
        if role != 'normal_calibration':
            ev = independent_events(literal>h)
            events_by_group[role] = ev
            for i, c in enumerate(members):
                for j, arm in enumerate(ARMS):
                    row = saved_events[(c['run_id'],arm)]
                    require(row['role'] == role and int(row['fault_id']) == int(c['fault_id']), 'Event row identity')
                    for field, values in ev.items():
                        require(int(row[field]) == int(values[i,j]), 'Saved event '+c['run_id']+'/'+arm+'/'+field)
                for j, view in ((4,'controller_input'),(5,'native')):
                    old = original_events[(c['run_id'],'F9_historical_'+view)]
                    for new, saved in (('any_alarm','normal_alarm'),('pre_alarm','prefault_alarm'),('clean40','clean40'),('clean8','clean8'),('clean2','clean2'),('first_post_delay_minutes','delay_minutes'),('loss_minutes','loss_minutes')):
                        require(int(ev[new][i,j]) == int(old[saved]), 'Archived endpoint events '+c['run_id']+'/'+saved)
    cal = read_npz(root/'results/CALIBRATION.npz')
    require(digest(root/'results/CALIBRATION.npz') == lock['calibration_npz_sha256'], 'Calibration artifact hash')
    peaks = scalar['normal_calibration'].max(axis=1)
    independent_h = np.partition(peaks,189,axis=0)[189]
    same(independent_h,h,'Independent rank190 thresholds')
    same(peaks,cal['peaks'],'All calibration maxima')
    same(cal['thresholds'],h,'Saved calibration thresholds')
    h_closed = np.partition(reconstructed['normal_calibration'].max(axis=1),189,axis=0)[189]
    closed_threshold_error = close(h_closed,h,'Closed form rank190 thresholds')
    old_peaks = read_npz(root/'reference/CALIBRATION_MAXIMA.npz')['maxima'][:,[21,20]]
    same(peaks[:,[4,5]],old_peaks,'Archived endpoint calibration maxima')
    evaluation = read_npz(root/'results/EVALUATION.npz')
    normal, fault = events_by_group['normal_evaluation'], events_by_group['fault_evaluation']
    for prefix,ev,members in (('normal',normal,groups['normal_evaluation']),('fault',fault,groups['fault_evaluation'])):
        same(evaluation[prefix+'_run_ids'],np.asarray([r['run_id'] for r in members]),'Evaluation run ordering '+prefix)
        for name,data in ev.items():same(evaluation[prefix+'_'+name],data,'Saved evaluation arrays '+prefix+'/'+name)
    main = read_csv(root/'results/Main_results.csv')
    require([r['arm'] for r in main] == list(ARMS),'Main table arm order')
    normal_scores = np.stack([score_arrays[c['run_id']] for c in groups['normal_evaluation']])
    independent_main = []
    main_max_error = 0.
    for j,row in enumerate(main):
        series=normal_scores[:,:,j].ravel().astype(np.longdouble)
        mean=series.sum()/series.size
        variance=np.dot(series-mean,series-mean)/(series.size-1)
        metrics={'threshold':h[j], 'n_fault':200, 'n_normal':400,'normal_score_mean':mean,'normal_score_variance_ddof1':variance,'mean_penalized_loss_minutes':fault['loss_minutes'][:,j].sum()/200}
        for label,arr,n in (('clean40',fault['clean40'],200),('clean8',fault['clean8'],200),('clean2',fault['clean2'],200),('prefault_alarm',fault['pre_alarm'],200),('normal_alarm',normal['any_alarm'],400),('normal_floor',normal['clean40'],400)):
            count=int(arr[:,j].sum()); lo,hi=wilson(count,n)
            metrics.update({label+'_n':count,label+'_percent':100*count/n,label+'_wilson95_low_percent':100*lo,label+'_wilson95_high_percent':100*hi})
        for key,value in metrics.items():main_max_error=max(main_max_error,close(np.array(value),np.array(float(row[key])),'Main table '+row['arm']+'/'+key))
        independent_main.append({'arm':row['arm'],**{k:float(v) for k,v in metrics.items()}})
    boot=read_npz(root/'results/BOOTSTRAP.npz')
    same(boot['contrast_indices'],np.asarray(CONTRASTS),'Bootstrap contrasts')
    same(boot['endpoints'],np.asarray(ENDPOINTS),'Bootstrap endpoint order')
    require(int(boot['seed']) == 2026103009,'Bootstrap seed')
    generator=np.random.default_rng(2026103009)
    ni=generator.integers(0,400,size=(20000,400),dtype=np.int32)
    fi=generator.integers(0,200,size=(20000,200),dtype=np.int32)
    same(ni,boot['normal_indices'],'Normal bootstrap indices')
    same(fi,boot['fault_indices'],'Fault bootstrap indices')
    all_distributions=np.empty((20000,6,6)); estimates=np.empty((6,6))
    for endpoint_index,name in enumerate(ENDPOINTS):
        source,indices=(normal['any_alarm'],ni) if name=='normal_alarm' else ((normal['clean40'],ni) if name=='normal_floor' else (fault[name],fi))
        all_distributions[:,:,endpoint_index],estimates[:,endpoint_index]=weighted_bootstrap(source,indices,CONTRASTS,1 if name=='loss_minutes' else 100)
    bounds=sorted_percentiles(all_distributions)
    distribution_error=close(all_distributions,boot['contrast_distributions'],'All independent bootstrap distributions')
    estimate_error=close(estimates,boot['contrast_estimates'],'Independent contrast estimates')
    interval_error=close(bounds,boot['contrast_bounds95'],'Independent percentile bounds')
    contrasts=read_csv(root/'results/Paired_contrasts.csv')
    require(len(contrasts)==36,'Contrast table completeness')
    csv_contrasts={(r['first_arm'],r['second_arm'],r['endpoint']):r for r in contrasts}
    require(len(csv_contrasts)==36,'Contrast table unique identities')
    for c,(a,b) in enumerate(CONTRASTS):
        for e,name in enumerate(ENDPOINTS):
            row=csv_contrasts[(ARMS[a],ARMS[b],name)]
            for key,value in [('estimate',estimates[c,e]),('ci95_low',bounds[0,c,e]),('ci95_high',bounds[1,c,e])]:
                close(np.array(float(row[key])),np.array(value),'Contrast CSV '+key)
    summary=read_json(root/'results/SUMMARY.json')
    require(summary['status']=='COMPLETE' and summary['threshold_lock_sha256']==lock_hash,'Completed summary lock')
    require(summary['binding']['code_lock_sha256']==code_hash and summary['binding']['protocol_sha256']==protocol_hash,'Summary frozen binding')
    require(len(summary['main_results'])==6 and len(summary['paired_contrasts'])==36,'Summary table cardinality')
    for saved,independent in zip(summary['main_results'],independent_main):
        require(saved['arm']==independent['arm'],'Summary main arm ordering')
        for key,value in independent.items():
            if key!='arm':close(np.asarray(saved[key]),np.asarray(value),'Summary main '+key)
    for saved in summary['paired_contrasts']:
        expected=csv_contrasts[(saved['first_arm'],saved['second_arm'],saved['endpoint'])]
        for key in ('estimate','ci95_low','ci95_high'):
            close(np.asarray(saved[key]),np.asarray(float(expected[key])),'Summary contrast '+key)
    for record in summary['evaluation_score_files']:
        require(digest(root/record['path'])==record['sha256'],'Summary evaluation file hash '+record['run_id'])
    require(len(summary['evaluation_score_files'])==600,'Summary evaluation file count')
    chronology=[protocol['created_utc'],code_lock['locked_utc'],lock['created_utc'],summary['completed_utc']]
    timestamps=[dt.datetime.fromisoformat(x) for x in chronology]
    require(all(a<=b for a,b in zip(timestamps,timestamps[1:])),'Protocol/code/calibration/result timestamp sequence')
    report={'status':'PASS','study_id':STUDY,'independent_implementation':True,
      'production_numerical_modules_imported':False,'physical_cases':799,'calibration_cases':199,'evaluation_cases':600,
      'arms':6,'evaluation_alarm_bits_crosschecked':alarm_bits,'evaluation_per_run_rows_verified':3600,
      'source_files_rehashed':source_files_checked,'archived_endpoint_alarm_bits_verified':original_alarm_bits,
      'original_endpoint_per_run_event_rows_verified':1200,'noise_recovery_record_anchors_bound':noise_anchors,
      'full_rng_trace_steps_bound':source_trace_steps,'complete_source_trace_hashes_bound':799,
      'replacement_keys_verified':799,'max_recovered_temperature_error':max_temp_error,
      'max_reported_original_chart_error':max_recorded_chart_error,
      'max_cusum_closed_form_vs_scalar_difference':maximum_math_discrepancy,
      'max_closed_form_threshold_difference':closed_threshold_error,'max_main_table_difference':main_max_error,
      'bootstrap_draws':20000,'bootstrap_contrast_endpoint_rows_verified':36,
      'max_bootstrap_distribution_difference':distribution_error,'max_bootstrap_estimate_difference':estimate_error,
      'max_bootstrap_interval_difference':interval_error,
      'protocol_sha256':protocol_hash,'code_lock_sha256':code_hash,'threshold_lock_sha256':lock_hash,
      'calibration_before_evaluation_evidence':'All600 evaluation score reports bind the verified calibration lock hash; protocol/code/calibration/result timestamp order verified. No unrecorded per-run timestamp is claimed.',
      'source_scope':'Sourcebytes and original trace/hash references rechecked' if source_root else 'Source identity declarations checked against frozen manifests; source bytes not supplied to this audit',
      'scope_limit':'Audits saved scores through reported results. Noise reconstruction is supported separately by complete source-capture tests and per-run trace bindings; this audit does not rerun all physical trajectories.',
      'wall_seconds':time.perf_counter()-started,'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    output.mkdir(parents=True,exist_ok=True)
    (output/'INDEPENDENT_RESULTS_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'INDEPENDENT_MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(independent_main[0]));writer.writeheader();writer.writerows(independent_main)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    arguments=parser.parse_args()
    try:
        audit(arguments.root,arguments.output,arguments.source_root)
    except Exception as exc:
        arguments.output.mkdir(parents=True,exist_ok=True)
        (arguments.output/'INDEPENDENT_RESULTS_AUDIT.json').write_text(json.dumps({'status':'FAIL','error':repr(exc)},indent=2)+'\n')
        raise
