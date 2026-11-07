# Supplementary Data S20 — Fault-15 risk/time sensitivity

Companion to manuscript and Supplementary Information **v1.7.4**, Section S20,
Tables S48–S49. This later, post-outcome exploratory analysis retains the original
fixed state and `k = 0.05` for the mean-pool and second-moment-pool Squared branches.
It is not an additional prospective validation or a refitting experiment.

Every integer outer-calibration rank from **234 through 259** is retained: 26 ranks,
52 method/rank rows, 250 normal evaluation runs and 250 Fault-15 evaluation runs.
Rank 247 is the unchanged original DV reference. The supplied results preserve
all thresholds, per-run events, clean-detection curves and adverse early outcomes.

## Recheck the saved results

Requires Python 3.10+ and NumPy. The original execution used Python 3.12.3 and
NumPy 2.2.6; `requirements.txt` records that original NumPy version. The packaged
validation report records the environment actually used for the publication recheck.
Run from this directory, or invoke the script by its path from any directory:

```bash
python recheck_saved.py
python recheck_saved.py --report recheck_report.json
```

To also check the tables directly against your copy of the submitted SI:

```bash
python recheck_saved.py --si-docx TEP_SI_v1.7.4_CILS.docx --report si_recheck_report.json
```

The script verifies packaged file hashes and byte identity of the returned
numerical material; reconstructs all 52 thresholds from the saved 259 outer
peaks and anchors; rechecks the two released branches against their original
rank-247 alarm bitmaps and seven event fields; and reaggregates the saved
continuous score trajectories. It recomputes every endpoint, Clopper–Pearson
interval, normal/fault per-run row and full clean-detection curve. It also reruns
the specified 20,000 paired whole-run bootstrap draws, preserving the common
schedule across ranks and separate normal/fault schedules, to check all 260
contrasts and pointwise intervals.

All integer identities and saved arrays are compared exactly. Floating-point CSV
values use `rtol=1e-12`, `atol=1e-10`. Counts and the printed losses/intervals in
S48–S49 must agree exactly at their displayed precision. A failed check stops the
script. Temporary reaggregation files are removed automatically; an optional
report is created only at a new filename. There is no simulation, fitting,
threshold selection, network access or change to the saved results.

## Contents

| Location | Contents |
|---|---|
| `results/RISK_TIME_ENDPOINTS.csv` | All 52 rows, counts, rates, marginal exact intervals and horizon-specific mean losses |
| `results/PAIRED_RANK_CONTRASTS.csv` | All 260 moment-minus-mean point estimates and pointwise paired bootstrap intervals |
| `results/THRESHOLDS_FIXED_BEFORE_TEST.csv` | All 52 original thresholds and rank-reference probabilities |
| `results/FAULT_PER_RUN.csv` | 13,000 run/method/rank rows and event/loss fields |
| `results/NORMAL_PER_RUN.csv` | 13,000 run/method/rank rows, any-alarm flags and exceedance counts |
| `results/CLEAN_DETECTION_CURVES.csv` | 41,652 rows: every 3 minutes from 0 through 2,400 minutes |
| `results/SCORES_FIXED_F15.npz` | Two fixed detector-score trajectories per role, each shaped `(250, 960, 2)`; methods, run IDs, all ranks and thresholds |
| `results/PROTOCOL_FIXED.json`, `results/BOOTSTRAP_SPEC.json` | Fixed analysis design, interval definitions, deterministic seeds and limitations |
| `frozen_assets/` | Unchanged frozen numerical state, candidate identities, calibration peaks/anchors, original baseline bitmaps and relative raw-input identity index extracted from the source archive |
| `source/frozen_core.py` | Byte-identical frozen scientific functions from the supplied archive |
| `source/risk_time.py` | Unchanged aggregation/bootstrap function bodies extracted from the original runner; portable imports added |
| `manuscript_tables/` | S48/S49 table-cell exports from the current v1.7.4 SI and its document SHA-256 |
| `provenance/` | Original upstream checks/logs and identities, extraction map, historical hashes, and run status with two machine-path fields removed |
| `validation/` | Publication saved-score recheck report |
| `PACKAGE_FILES_SHA256.json` | Integrity hashes for distributed files other than this manifest and validation reports |

`ORIGINAL_RETURN_FILES_SHA256.json` and `ORIGINAL_PYZ_MEMBER_HASHES.json` are
historical identity records, not manifests of this publication directory. Some
listed original files are deliberately absent here. The original host launcher
and raw-input execution wrapper are omitted. `SOURCE_EXTRACTION.json` identifies
the precise original function ranges. Only the executable path and output-directory
path were removed from the original run status; its scientific/status fields are
unchanged. Numerical return files and frozen assets are byte-identical to their
verified supplied originals.

## Definitions and interpretation

An alarm is the strict comparison `score > threshold`, without latching or an
onset reset. Normal FAR uses all 960 records (48 hours). Fault onset is the 161st
record, leaving 800 post-fault records (40 hours). Delay zero denotes the first
post-fault record. A clean detection requires no alarm among the first 160 records
and a post-fault alarm by the specified horizon; all 250 runs remain in the denominator.
For horizon H, loss is the first delay in minutes when clean detection occurs by H,
and otherwise H. Detection exactly at H can have the same loss as failure. These
finite-window losses do not encode industrial economic costs.

The threshold at rank r is calculated as
`anchor * sorted(outer_peak / anchor)[r-1]`. The probability `(260-r)/260` is an
ideal continuous exchangeable marginal reference. A common rank does not imply
equal realized or true false-alarm risk. No interpolation, empirical-FAR matching,
equivalence margin or test-selected replacement threshold is supplied.

The rate intervals are marginal and conditional on frozen state and independent
Bernoulli runs. Bootstrap intervals are paired, pointwise, exploratory and not
simultaneous or multiplicity adjusted; they include no fitting/recalibration
uncertainty. A zero empirical bootstrap interval is not evidence of population
equivalence or absence of uncertainty. This later bootstrap stream does not
replace the original DV adjusted intervals.

Across every rank, both branches have **zero clean detections within 60 minutes**.
The mean branch has zero two-hour clean detections; the moment branch has at most
2/250. At rank 259, moment has only 3/250 clean detections by eight hours, with an
eight-hour loss difference of −0.600 minutes [−1.560, 0.000]; both two-hour mean
losses are 120 minutes. At that endpoint the 40-hour clean counts are 0/250 and
171/250, while normal alarm counts are 1/250 and 2/250. The long-horizon results
therefore do not establish reliable early warning or matched-risk superiority.

## Reproduction boundary

This package supports score-to-event, fixed-calibration-to-threshold and
result-to-manuscript-table reproduction. It does **not** contain the 52-variable raw
process trajectories, and the recheck does not independently regenerate the score
paths from those trajectories. Frozen PCA, projection/scaler and model assets plus
readable scientific functions are included for inspection and use alongside the
separately obtained original raw-data companion; no such raw trajectories are included in this kit. The relative input index and original raw
archive digest are retained in the provenance/design material.

The upstream return reports replay of 791 existing trajectories and exact
agreement of 4,320,000 original alarm bits across nine branches. That historical
check is retained as provenance. The publication recheck directly checks **960,000
bits across the two released Squared score branches**, not all nine branches.
Neither check establishes new simulator fidelity, training stability, a fresh
confirmation sample, or general performance beyond the defined data and design.
