"""A7 B estimators / A8 native PCA500 swap: exact existing-data downstream run.

No simulation. A7 changes B only, keeping the original PCA/design whitening.
A8 swaps only PCA and basis, relearning with original run roles/ranks/k. Its
full16 implementation requires all native residual covariance eigenvalues to
pass the amended existing floor; if not, it stops rather than inventing a rule.

Manifest states have raw52 roles normal_design,anchor,inner,selection,outer,
normal,fault, and fault_fit (A8 only). Other keys: state,pca,basis,design,
fit_moments,baseline_events (A7),old_events (A8),old_pca (A8).
A8 uses one state, outer199, normal120, fault120; A7 five original states,
outer259, normal250, fault250. Baselines are original event arrays, not curves.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from additional_analysis_core import *
sys.path.insert(0, str(Path(__file__).parent / 'reference_code'))
import inherited as old
import science


def objects(raw, pca, basis):
    return science.project(raw, pca, basis)


def build_one(des, mat, nd, directory):
    directory.mkdir(parents=True, exist_ok=False)
    all_axes, labels, small, recipes, ends, diagnostics = old.build_axes_regularized(
        mat['mean_early'], mat['mean_late'], des['B'], des, science.VARIANT)
    mean = all_axes[small].copy()
    var = np.einsum('ad,de,ae->a', mean, des['G'], mean)
    mean /= np.sqrt(var)[:, None]
    moment, _, _ = old.learned_axes(mat['contrast'], des, 'ridge_1e4')
    pars = old.parameters(nd, des, mean, moment)
    np.savez_compressed(directory / 'DESIGN.npz', **des)
    np.savez_compressed(directory / 'MODEL.npz', mean=mean, moment=moment)
    np.savez_compressed(directory / 'PARAMETERS.npz', **pars)


def selected_eval(obj, directory):
    methods, ev, mask = science.selected_outcomes(obj, directory)
    return methods, ev, mask


def check_original(methods, ef, en, h, saved):
    order = [list(saved['methods']).index(m) for m in methods]
    if not np.allclose(h, saved['thresholds'][order], rtol=1e-9, atol=1e-8):
        raise ValueError('original k=.05 thresholds not reproduced')
    if not np.array_equal(ef['event_loss_samples'], saved['fault_event_loss_samples'][:, order]):
        raise ValueError('original fault loss not reproduced')
    if not np.array_equal(ef['clean_detection'], saved['fault_clean_detection'][:, order]):
        raise ValueError('original fault detection not reproduced')
    if not np.array_equal(en['prefault_alarm'] | en['detected'], saved['normal_alarm_any'][:, order]):
        raise ValueError('original normal FAR not reproduced')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('task', choices=['A7', 'A8'])
    ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--raw-root'); ap.add_argument('--development-root'); ap.add_argument('--panel2-root'); args = ap.parse_args()
    mp = Path(args.manifest).resolve(); root = mp.parent
    m = resolve_raw_roots(json.loads(mp.read_text()), args.raw_root, args.development_root, args.panel2_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.task == 'A7' and [r['state'] for r in m['states']] != [1, 2, 3, 4, 5]:
        raise ValueError('A7 requires original five states')
    if args.task == 'A8' and len(m['states']) != 1:
        raise ValueError('A8 requires one original native recording pipeline')
    results = []; diagnostics = []; comparisons = []; hashes = []
    for row in m['states']:
        pca = npz_read(root / row['pca']); basis = np.load(root / row['basis'], allow_pickle=False)
        if len(pca['loadings']) != 36 or basis.shape != (52, 16):
            raise ValueError('this native-only runner requires original full16 residuals')
        count = 250 if args.task == 'A7' else 120; outer_n = 259 if args.task == 'A7' else 199
        rank = 247 if args.task == 'A7' else 190
        raw = {}; obj = {}; identities = []
        roles = [('normal_design', 40), ('anchor', 32), ('inner', 59), ('selection', 75),
                 ('outer', outer_n), ('normal', count), ('fault', count)]
        if args.task == 'A8':
            roles.append(('fault_fit', 100))
        for role, n in roles:
            raw[role], ids = load_role(row[role], n, root=root); identities.append(ids)
            obj[role] = objects(raw[role], pca, basis)
        assert_disjoint(*identities)
        if args.task == 'A7':
            des = npz_read(root / row['design']); mat = npz_read(root / row['fit_moments'])
            measured = old.normal_design(obj['normal_design'], list(range(1, 41)), science.VARIANT)
            for key in ('centre', 'whitener', 'G', 'B'):
                if not np.allclose(measured[key], des[key], atol=1e-9, rtol=1e-8):
                    raise ValueError('A7 original normal design does not match: ' + key)
        else:
            previous = npz_read(root / row['old_pca'])
            if 'basis' not in previous:
                raise ValueError('A8 original PCA78 input must carry its exact native residual basis')
            original_objs = {role: objects(x, previous, previous['basis']) for role, x in raw.items()}
            original_des = old.normal_design(original_objs['normal_design'], list(range(1, 41)), science.VARIANT)
            _, original_mat, _ = old.fit_matrices(original_objs['fault_fit'], original_objs['normal_design'], original_des, 9)
            original_directory = out / 'A8_ORIGINAL_PCA78_REPRODUCTION'
            build_one(original_des, original_mat, original_objs['normal_design'], original_directory)
            original_choices = science.select_one(original_objs['anchor'], original_objs['inner'], original_objs['selection'], 9, original_directory)
            original_h = science.calibrate_one(original_objs['outer'], original_directory, rank=190)
            original_methods, original_ef, original_mask = selected_eval(original_objs['fault'], original_directory)
            _, original_en, original_nmask = selected_eval(original_objs['normal'], original_directory)
            old_ev = npz_read(root / row['old_events'])
            target_j = original_methods.index('moment_pool_signed')
            if not np.isclose(original_h[original_choices['moment_pool_signed']], float(old_ev['threshold']), rtol=1e-9, atol=1e-8):
                raise ValueError('A8 original PCA78 M2 threshold not reproduced')
            for actual, target in [(original_ef['event_loss_samples'][:, target_j] * 3, old_ev['fault_event_loss_minutes']),
                                   (original_ef['clean_detection'][:, target_j], old_ev['fault_clean']),
                                   (original_nmask[:, :, target_j].any(1), old_ev['normal_alarm'])]:
                if not np.array_equal(actual, target):
                    raise ValueError('A8 original PCA78 M2 event vectors not reproduced')
            # Implement the recorded amended rule and stop on unsupported
            # native truncation, rather than silently treating a floored zero as data.
            zz = obj['normal_design']['z'].reshape(-1, 16)
            C = np.cov(zz, rowvar=False, ddof=1)
            eig = np.linalg.eigvalsh(C); tau = max(1e-10 * np.trace(C) / 16, 1e-14)
            if np.any(eig < tau):
                raise ValueError('PCA500 native needs reduced-rank amended source implementation; no automatic fallback')
            des = old.normal_design(obj['normal_design'], list(range(1, 41)), science.VARIANT)
            _, mat, _ = old.fit_matrices(obj['fault_fit'], obj['normal_design'], des, 9)
            angles = principal_angles(previous, pca)
            csv_write(out / 'A8_PCA_RESIDUAL_PRINCIPAL_ANGLES.csv', [
                {'coordinate_system': coordinate, 'principal_angle_index': j + 1, 'angle_degrees': float(a)}
                for coordinate, a in angles.items() for j, a in enumerate(a)])
        y = old.features(obj['normal_design'], des)
        metric_variants = temporal_metrics(y) if args.task == 'A7' else {'PCA500_swap': des['B']}
        for name, B in metric_variants.items():
            if name == 'original' and not np.allclose(B, des['B'], atol=1e-10, rtol=1e-9):
                raise ValueError('original B reconstruction failed')
            changed = dict(des); changed['B'] = B
            ev = np.linalg.eigvalsh((B + B.T) / 2)
            diagnostics.append({'state': row['state'], 'metric': name, 'minimum_eigenvalue': ev[0],
                                'maximum_eigenvalue': ev[-1], 'condition_number': ev[-1] / ev[0] if ev[0] > 0 else 'INF'})
            if ev[0] <= 0:
                raise ValueError('nonpositive B estimator; no undocumented regularization')
            directory = out / f'state{row["state"]}_{name}'
            build_one(changed, mat, obj['normal_design'], directory)
            choices = science.select_one(obj['anchor'], obj['inner'], obj['selection'], 9, directory)
            h_all = science.calibrate_one(obj['outer'], directory, rank=rank)
            methods, ef, mask = selected_eval(obj['fault'], directory)
            nmethods, en, nmask = selected_eval(obj['normal'], directory)
            if nmethods != methods:
                raise ValueError('normal/fault method order mismatch')
            selection_order = [choices[m] for m in methods]; h = h_all[selection_order]
            if args.task == 'A7' and name == 'original':
                check_original(methods, ef, en, h, npz_read(root / row['baseline_events']))
            losses = ef['event_loss_samples'] * 3
            a = methods.index('mean_pool_signed'); b = methods.index('moment_pool_signed')
            d, lo, hi = bootstrap_mean(losses[:, b] - losses[:, a], seed=2026100800 + row['state'], alpha=.05 / 3)
            comparisons.append({'state': row['state'], 'metric': name, 'contrast': 'C2_moment_minus_mean',
                                'loss_difference_minutes': d, 'CI98_333_low': lo, 'CI98_333_high': hi})
            for j, method in enumerate(methods):
                if method not in ('mean_pool_signed', 'moment_pool_signed'):
                    continue
                clean = ef['clean_detection'][:, j].astype(bool); first = ef['first_delay_samples_penalized'][:, j] * 3
                results.append({'state': row['state'], 'metric': name, 'method': method, 'threshold': h[j],
                    'clean40_n': int(clean.sum()), 'clean40_pct': clean.mean() * 100,
                    'clean8_n': int((clean & (first <= 480)).sum()), 'clean8_pct': (clean & (first <= 480)).mean() * 100,
                    'mean_loss_minutes': losses[:, j].mean(), 'normal_alarm_n': int(nmask[:, :, j].any(1).sum()),
                    'normal_FAR_pct': nmask[:, :, j].any(1).mean() * 100})
            np.savez_compressed(directory / 'EVALUATION_EVENTS.npz', methods=np.asarray(methods), thresholds=h,
                                fault_event_loss_minutes=losses, fault_clean=ef['clean_detection'],
                                normal_alarm=nmask.any(1))
            if args.task == 'A8':
                old_ev = npz_read(root / row['old_events'])
                old_loss = old_ev['fault_event_loss_minutes']; old_clean = old_ev['fault_clean']
                if old_loss.shape != (120,) or old_clean.shape != (120,) or old_ev['normal_alarm'].shape != (120,):
                    raise ValueError('A8 old events must be 120 original run-ID-ordered vectors for M2')
                if old_clean.sum() != 38 or old_ev['normal_alarm'].sum() != 7:
                    raise ValueError('A8 old event vectors do not reproduce M2 38/120 and FAR7/120')
                for label, vector in [('loss_minutes', losses[:, b] - old_loss),
                                      ('clean40_percentage_points', (ef['clean_detection'][:, b] - old_clean) * 100),
                                      ('clean8_percentage_points', (ef['clean_detection'][:, b] * (ef['first_delay_samples_penalized'][:, b] * 3 <= 480) - old_ev['fault_clean8']) * 100),
                                      ('normal_FAR_percentage_points', (nmask[:, :, b].any(1).astype(int) - old_ev['normal_alarm']) * 100)]:
                    avg, ci_lo, ci_hi = bootstrap_mean(vector, seed=20261008)
                    comparisons.append({'state': row['state'], 'metric': name, 'contrast': 'PCA500_minus_PCA78_' + label,
                                        'difference': avg, 'CI95_low': ci_lo, 'CI95_high': ci_hi})
    csv_write(out / f'{args.task}_ENDPOINTS.csv', results)
    csv_write(out / f'{args.task}_PAIRED_CONTRASTS.csv', comparisons)
    csv_write(out / f'{args.task}_B_EIGENVALUES.csv', diagnostics)
    for row in m['states']:
        for value in row.values():
            names = value.get('files', [value.get('file')]) if isinstance(value, dict) else [value] if isinstance(value, str) else []
            for name in names:
                if name and (root / name).is_file():
                    p = root / name; hashes.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    csv_write(out / 'INPUT_HASHES.csv', hashes)


if __name__ == '__main__':
    main()
