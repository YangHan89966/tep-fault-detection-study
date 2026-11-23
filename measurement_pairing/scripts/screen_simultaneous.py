"""Recompute the simultaneous 95% intervals in Figure 4 (SI Section S6).
The crossed bootstrap resamples 36 independent fits and the 120 evaluation
trajectories per fault. All 76 paired contrasts use the same replicate draw.
"""
import argparse
from pathlib import Path
import csv
import os

import numpy as np

study = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=study.parent / "replay_output/screening_simultaneous_clean8h_rank95.csv")
args = parser.parse_args()
root, out = study / "family_b", args.output
out.parent.mkdir(parents=True, exist_ok=True)
SEED, B, BATCH = 20261113, 20000, 500
COLS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 18, 19, 20, 21]  # zero-based XMEAS indices screened
z = np.load(os.path.join(root, "statistics/E17/first_alarm_rank95.npz"), allow_pickle=False)
arch = {(r["fault"], r["screen_direction"], int(r["measurement_zero_based"])): r
        for r in csv.DictReader(open(os.path.join(root, "statistics/E16_S/screening.csv"), newline=""))
        if r["rank"] == "95" and r["endpoint"] == "clean8h"}


def clean8(key):
    a = z[key].astype(int)
    return ((a >= 160) & (a <= 319)).astype(np.int8)


labels, D = [], {3: [], 9: []}
for f in (3, 9):
    for d, base in (("S1", "X1"), ("S2", "aligned")):
        b0 = clean8(f"F{f:02d}_Fixed246_{base}")
        for c in COLS:
            D[f].append(clean8(f"F{f:02d}_Fixed246_{d}_C{c:02d}") - b0)
            labels.append((f, d, c))
D = {f: np.stack(v).astype(np.float64) for f, v in D.items()}
nrep, nrun = D[3].shape[1:]
est = np.concatenate([D[f].mean(axis=(1, 2)) for f in (3, 9)])
rng = np.random.Generator(np.random.Philox(SEED))
boot = np.empty((B, est.size))
for s in range(0, B, BATCH):
    m = min(BATCH, B - s)
    wr = rng.multinomial(nrep, np.full(nrep, 1 / nrep), size=m).astype(np.float64)
    parts = []
    for f in (3, 9):
        wt = rng.multinomial(nrun, np.full(nrun, 1 / nrun), size=m).astype(np.float64)
        parts.append(np.einsum("bct,bt->bc", np.einsum("br,crt->bct", wr, D[f]), wt) / (nrep * nrun))
    boot[s:s + m] = np.concatenate(parts, axis=1)
sd = boot.std(axis=0, ddof=1)
ok = sd > 0
t = np.abs(boot[:, ok] - est[ok]) / sd[ok]
q = np.quantile(t.max(axis=1), 0.95)
lo_pw, hi_pw = np.quantile(boot, [0.025, 0.975], axis=0)
p = np.ones(est.size)
p[ok] = np.maximum((t >= (np.abs(est[ok]) / sd[ok])).mean(axis=0), 1 / B)
order = np.argsort(p)
holm = np.empty_like(p)
running = 0.0
for k, i in enumerate(order):
    running = max(running, min(1.0, (est.size - k) * p[i]))
    holm[i] = running
rows, maxdiff = [], 0.0
for i, (f, d, c) in enumerate(labels):
    a = arch[(str(f), d, c)]
    maxdiff = max(maxdiff, abs(est[i] - float(a["estimate"])) * 100,
                  abs(lo_pw[i] - float(a["lo"])) * 100, abs(hi_pw[i] - float(a["hi"])) * 100)
    rows.append(dict(fault=f, screen_direction=d, measurement_zero_based=c, measurement_name=f"XMEAS({c + 1})",
                     estimate_pp=f"{100 * est[i]:.4f}", archived_lo_pp=f"{100 * float(a['lo']):.4f}",
                     archived_hi_pp=f"{100 * float(a['hi']):.4f}", new_pointwise_lo_pp=f"{100 * lo_pw[i]:.4f}",
                     new_pointwise_hi_pp=f"{100 * hi_pw[i]:.4f}", simultaneous_lo_pp=f"{100 * (est[i] - q * sd[i]):.4f}",
                     simultaneous_hi_pp=f"{100 * (est[i] + q * sd[i]):.4f}", bootstrap_p=f"{p[i]:.6g}", holm_p=f"{holm[i]:.6g}",
                     simultaneous_excludes_zero=int(ok[i] and abs(est[i]) > q * sd[i])))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"contrasts {est.size}; draws {B}; seed {SEED}; max-t critical value {q:.3f}")
print(f"largest difference from screening.csv (estimate or pointwise bound): {maxdiff:.2f} pp")
for r in rows:
    if r["simultaneous_excludes_zero"] or float(r["holm_p"]) < 0.05:
        print(r["fault"], r["screen_direction"], r["measurement_name"], r["estimate_pp"],
              r["simultaneous_lo_pp"], r["simultaneous_hi_pp"], "holm", r["holm_p"])
