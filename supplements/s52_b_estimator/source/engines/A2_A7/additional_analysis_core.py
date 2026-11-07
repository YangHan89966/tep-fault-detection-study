"""Numerical tools for A2/A3/A6/A7/A8; no simulator imports or execution.

Input contracts preserve entire runs. A saved alarm mask, a CUSUM maximum,
or a whole-100-run mean is never substituted for missing pre-CUSUM scores
or a subset's own mean. All 52-D angles refer to standardized-variable space.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np


def resolve_raw_roots(manifest, raw_root=None, development_root=None, panel2_root=None):
    """Replace known path placeholders once; retain original template on disk."""
    roots = {'__SUPPLY_R5_RAW_ROOT__': raw_root,
             '__SUPPLY_RELEARN_RAW_ROOT__': development_root or raw_root,
             '__SUPPLY_PANEL2_RAW_ROOT__': panel2_root or raw_root}
    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            for marker, directory in roots.items():
                if directory:
                    value = value.replace(marker, str(Path(directory).expanduser().resolve()))
        return value
    return walk(manifest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty table cannot be reported as a completed result')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def npz_read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_role(spec, expected_n, d=52, root='.'):
    """Read explicit file list or aggregate; never scan for data or simulate.

    spec keys: role, ids, files OR file, optional key, sha256s. Each file's
    array is values/native/X or measurements plus manipulated_vars. A 961-row
    file is accepted ONLY when drop_initial=true is explicitly recorded.
    """
    ids = tuple(spec['ids'])
    if len(ids) != expected_n or len(set(ids)) != expected_n:
        raise ValueError('wrong run-ID count or duplicate IDs')
    paths = spec.get('files', [spec.get('file')])
    if any(p is None for p in paths):
        raise ValueError('explicit data paths required')
    blocks = []
    for i, name in enumerate(paths):
        p = Path(root) / name
        if spec.get('sha256s') and sha256(p) != spec['sha256s'][i]:
            raise ValueError(f'input hash mismatch: {p}')
        z = npz_read(p)
        key = spec.get('key')
        if key:
            a = z[key]
        elif 'values' in z:
            a = z['values']
        elif 'native' in z:
            a = z['native']
        elif 'native_values' in z:
            a = z['native_values']
        elif 'X' in z:
            a = z['X']
        elif {'measurements', 'manipulated_vars'} <= set(z):
            a = np.concatenate([z['measurements'][..., :41], z['manipulated_vars'][..., :11]], axis=-1)
        else:
            raise ValueError(f'unknown raw layout at {p}; specify key explicitly')
        if a.ndim == 2:
            a = a[None]
        if spec.get('drop_initial') and a.shape[-2] == 961:
            a = a[:, 1:]
        blocks.append(np.asarray(a, dtype=np.float64))
    x = np.concatenate(blocks)
    if x.shape != (expected_n, 960, d) or not np.isfinite(x).all():
        raise ValueError(f'expected ({expected_n},960,{d}), got {x.shape}')
    return x, ids


def assert_disjoint(*id_sets):
    seen = set()
    for ids in id_sets:
        if seen.intersection(ids):
            raise ValueError('different development/calibration/evaluation roles share run IDs')
        seen.update(ids)


def cusum(s, k=.05):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 2:
        s = s[..., None]
    if s.ndim != 3 or s.shape[1] != 960 or not np.isfinite(s).all():
        raise ValueError('pre-CUSUM s_t must be complete finite (runs,960,monitors)')
    cp = np.zeros((len(s), s.shape[2])); cm = cp.copy()
    out = np.empty_like(s)
    for t in range(960):
        cp = np.maximum(0, (cp + s[:, t]) - k)
        cm = np.maximum(0, (cm - s[:, t]) - k)
        out[:, t] = np.maximum(cp, cm)
    return out


def event_values(statistics, threshold):
    mask = np.asarray(statistics) > np.asarray(threshold)
    if mask.ndim == 2:
        mask = mask[..., None]
    pre = mask[:, :160].any(1); post = mask[:, 160:]
    det = post.any(1); first = np.where(det, post.argmax(1), 800)
    clean = det & ~pre
    return {'clean40': clean, 'clean8': clean & (first * 3 <= 480),
            'loss': np.where(clean, first * 3, 2400), 'any': mask.any(1),
            'floor': clean, 'first_delay_minutes': first * 3}


def endpoints(s_fault, s_normal, h, k=.05):
    ef = event_values(cusum(s_fault, k), h)
    en = event_values(cusum(s_normal, k), h)
    return ef, {'clean40_pct': ef['clean40'].mean(0) * 100,
                'clean8_pct': ef['clean8'].mean(0) * 100,
                'mean_loss_minutes': ef['loss'].mean(0),
                'normal_FAR_pct': en['any'].mean(0) * 100,
                'normal_floor_pct': en['floor'].mean(0) * 100}


def calibrate_scores(s_outer, s_anchor=None, rank=247, k=.05):
    peaks = cusum(s_outer, k).max(1)
    if not 1 <= rank <= len(peaks):
        raise ValueError('invalid calibration order statistic')
    # Match manuscript's floating order when anchor trajectories are available.
    if s_anchor is None:
        return np.sort(peaks, axis=0)[rank - 1]
    anchor = np.quantile(cusum(s_anchor, k).max(1), .9, axis=0, method='linear')
    if np.any(anchor <= 0):
        raise ValueError('nonpositive anchor')
    return anchor * np.sort(peaks / anchor, axis=0)[rank - 1]


def bootstrap_mean(d, seed=20261008, draws=20000, alpha=.05):
    d = np.asarray(d, dtype=float)
    if d.ndim != 1:
        raise ValueError('one paired per-run difference vector required')
    rng = np.random.default_rng(seed); out = np.empty(draws)
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        out[start:start + n] = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), *np.quantile(out, [alpha / 2, 1 - alpha / 2]).tolist()


def raw_standardize(x, pca):
    return (x - pca['mean']) / pca['scale']


def residual_coordinates(x, pca, basis):
    z = raw_standardize(x, pca)
    r = z - (z @ pca['loadings'].T) @ pca['loadings']
    if basis.shape != (52, r.shape[-1] - len(pca['loadings'])):
        raise ValueError('residual basis dimension does not match PCA rank')
    q = np.sum(r * r, axis=-1)
    rz = r @ basis
    if not np.allclose(np.sum(rz * rz, -1), q, rtol=1e-9, atol=1e-9):
        raise ValueError('residual basis/PCA mismatch')
    return rz, q


def projected_scores(x, pca, basis, des, v, family='signed', pars=None):
    rz, _ = residual_coordinates(x, pca, basis)
    y = (rz - des['centre']) @ des['whitener']
    v = np.asarray(v)
    normal_variance = float(v @ des['G'] @ v)
    if not np.isclose(normal_variance, 1, rtol=1e-8, atol=1e-10):
        raise ValueError('saved axis does not have frozen normal-design unit variance')
    # Frozen axes already include the denominator, preserving source arithmetic.
    u = np.einsum('ntd,ad->nta', y, v[None], optimize=True)[..., 0]
    if family == 'signed':
        return u
    if family == 'squared':
        return (u * u - pars['sq_mean']) / pars['sq_sd']
    if family == 'absolute':
        return (abs(u) - pars['abs_mean']) / pars['abs_sd']
    raise ValueError('unknown expression')


def endpoint_moments(y, windows=None):
    """Run/endpoint weights equal. Return per-run means AND moments for A2."""
    means = []; seconds = []
    for i, run in enumerate(y):
        w = windows[i] if windows is not None else {
            'early_first_sample': 161, 'early_last_sample': 320,
            'late_first_sample': 801, 'late_last_sample': 960}
        mm = []; ff = []
        for label in ('early', 'late'):
            v = run[int(w[label + '_first_sample']) - 1:int(w[label + '_last_sample'])]
            if not len(v):
                raise ValueError('empty fitting endpoint')
            mm.append(v.mean(0)); ff.append(v.T @ v / len(v))
        means.append((mm[0] + mm[1]) / 2)
        seconds.append((ff[0] + ff[1]) / 2)
    return np.stack(means), np.stack(seconds)


def fda_direction(normal, fault, windows=None):
    """A6 full52 definition and A2 generic centered FDA, all in input coordinates."""
    flat = normal.reshape(-1, normal.shape[-1])
    mn = flat.mean(0); cn = (flat - mn).T @ (flat - mn) / len(flat)
    per_mu, per_F = endpoint_moments(fault, windows)
    mf = per_mu.mean(0); cf = per_F.mean(0) - np.outer(mf, mf)
    sw = (cn + cf) / 2; sw = (sw + sw.T) / 2
    rho = 1e-4 * np.trace(sw) / len(sw)
    if rho <= 0 or np.linalg.norm(mf - mn) < 1e-12:
        raise ValueError('degenerate FDA; no favorable fallback')
    w = np.linalg.solve(sw + rho * np.eye(len(sw)), mf - mn)
    g = np.cov(flat, rowvar=False, ddof=1)
    w /= np.sqrt(w @ g @ w)
    if w[np.argmax(abs(w))] < 0:
        w = -w
    return w, mn, {'rho': rho, 'mu_fault': mf, 'C_fault': cf, 'C_normal': cn, 'G': g}


def angle(v, w):
    """Acute unoriented angle; score sign does not change two-sided CUSUM."""
    return float(np.degrees(np.arccos(np.clip(abs(v @ w) / np.linalg.norm(v) / np.linalg.norm(w), 0, 1))))


def temporal_metrics(y, bandwidth=120):
    """Original B, whole-run batch means, and run-boundary-aware Bartlett HAC.

    HAC convention: Bartlett bandwidth=120, w_l=max(0,1-l/120), Gamma_l averages
    within-run lag pairs only, denominator n_runs*960 (Newey-West biased form).
    No concatenated cross-run lag pairs. Native design centering is frozen.
    """
    n, t, d = y.shape
    if (n, t) != (40, 960):
        raise ValueError('temporal metrics require 40 complete normal design runs')
    original = np.cov(y[:, :120].sum(1) / np.sqrt(120), rowvar=False, ddof=1)
    sums = y.reshape(n, 8, 120, d).sum(2).reshape(n * 8, d) / np.sqrt(120)
    batch = np.cov(sums, rowvar=False, ddof=1)
    flat = y.reshape(-1, d); centered = y - flat.mean(0)
    hac = centered.reshape(-1, d).T @ centered.reshape(-1, d) / (n * t)
    for lag in range(1, bandwidth + 1):
        gamma = np.einsum('ntd,nte->de', centered[:, lag:], centered[:, :-lag]) / (n * t)
        hac += max(0, 1 - lag / bandwidth) * (gamma + gamma.T)
    return {'original': original, 'batch': batch, 'HAC': (hac + hac.T) / 2}


def principal_angles(pca_a, pca_b):
    """Compare score-coefficient and displacement subspaces in stated metrics.

    The second model is the common standardization reference (PCA500 for A8).
    Direct basis comparison identifies model-specific standardized axes merely
    by variable index; models with different scales do not share those axes.
    Coefficients are covectors: reference D_b D_model^-1 R_model. Physical
    residual displacements are vectors: reference D_b^-1 D_model R_model.
    Their spaces differ. Raw coefficients use units chosen for each variable.
    """
    ans = {}
    for coordinate in ('model_specific_standardized_axes_identified_by_variable_index',
                       'raw_score_coefficient_unit_dependent',
                       'PCA500_reference_standardized_score_coefficient',
                       'PCA500_reference_standardized_residual_displacement'):
        residuals = []
        for p in (pca_a, pca_b):
            _, _, vh = np.linalg.svd(p['loadings'], full_matrices=True)
            r = vh[len(p['loadings']):].T
            if coordinate == 'raw_score_coefficient_unit_dependent':
                r = r / p['scale'][:, None]
            elif coordinate == 'PCA500_reference_standardized_score_coefficient':
                r = (pca_b['scale'] / p['scale'])[:, None] * r
            elif coordinate == 'PCA500_reference_standardized_residual_displacement':
                r = (p['scale'] / pca_b['scale'])[:, None] * r
            residuals.append(np.linalg.qr(r)[0])
        singular = np.linalg.svd(residuals[0].T @ residuals[1], compute_uv=False)
        ans[coordinate] = np.degrees(np.arccos(np.clip(singular, 0, 1)))
    return ans
