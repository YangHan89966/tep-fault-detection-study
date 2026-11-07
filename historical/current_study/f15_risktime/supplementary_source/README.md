# Supplementary Data S20 — fixed-k Fault-15 risk–time sensitivity

This archive supports manuscript Section S20 and Tables S48–S49. It supplements, and never replaces, the original DV primary assessment and the separate S18 k-sensitivity analysis.

## What it contains

`data/` retains all 26 specified calibration ranks (234 through 259), 52 method–rank rows, all 260 paired contrast rows, 13,000 per-run rows for each test role, 41,652 curve rows, both continuous CUSUM score paths, the exact protocol and resampling specification. Every data file is an unchanged member of the original F15 result archive. `reference/assets/` retains the original calibration peaks/anchors, candidate-selection indices and baseline packed alarms/events needed for reaggregation. Their bytes come from the issued fixed-k tool; they are frozen inputs, not fitted anew by this archive.

The two methods are `mean_pool_squared` and `moment_pool_squared`, both at **k = 0.05**. All 250 normal and 250 Fault-15 test IDs are retained. Rank 247 is the original DV reference. Ranks 234 and 259 are the specified range endpoints, not selected operating points. No empirical-FAR matching or interpolation is performed.

## Reproduce the saved-score calculations

Extract the archive, then use a Python environment with NumPy, pandas and SciPy already available. From the extracted folder run:

```sh
python -B reaggregate_s20.py --output ../S20_REAGGREGATION_NEW
```

The output directory must not already exist and must be outside the input folder. The script installs nothing, accesses no network, and does not import the original runner. It verifies the input manifest and reconstructs all 52 limits, original-rank alarm bits/events, per-run endpoints, conditional binomial intervals, complete detection curves, and 20,000 paired bootstrap draws for every contrast. It writes `VERIFICATION.json`, `CHECKS_DETAIL.json` and `VERIFIED_ENDPOINTS.csv`. A failure produces `FAILURE.json` and a nonzero exit status; do not alter the archived inputs to force agreement.

This command was tested in the integration environment (Python 3.13.5, NumPy 2.3.5). Library requirements are stated rather than silently installed. The original local scoring used the environment recorded in the original local return. Minor floating-point differences are checked with stated numeric tolerances, while original thresholds, packed alarm bits and event counts require exact agreement.

## Scope and interpretation

These are **post-outcome exploratory** results on already inspected data, not a new prospective test. PCA, directions, expression, scaling and k stay fixed; the calibration rank varies. The paired bootstrap intervals are pointwise, conditional on the fitted states and realized calibration sample, not simultaneous across ranks and not inclusive of retraining uncertainty. Rate intervals are conditional marginal Clopper–Pearson intervals. An interval spanning zero does not establish equivalence; sparse normal alarms do not prove equal true FAR.

The normal alarm horizon is all 960 records (48 hours). Fault outcomes use the 800 post-fault records (40 hours), with the 160 pre-fault records also checked. Clean detection excludes a pre-alarmed run from success but never removes it from the denominator. Finite-window penalties are evaluation conventions, not measured industrial costs. The rank-247 exploratory intervals use this analysis's resampling stream; they must not overwrite the original DV adjusted primary intervals.

## What is not reproduced here

The archive begins at saved continuous score paths, not at the raw 52-variable process trajectories. It does not rerun the simulator, refit PCA or directions, certify local raw-to-score execution, quantify training stability, or select a replacement threshold. The local 791-trajectory replay and full nine-branch original-alarm checks are documented in the separate evidence archive. Necessary raw-process inputs and access are described separately in the input-availability statement.

## Provenance and permissions

`SOURCE_IDENTITIES.json` identifies the original return, issued tool and frozen assets. `FILES_SHA256.json` covers every shipped member except itself. Source identities establish traceability, not scientific independence or public availability. This package does not grant or expand third-party redistribution rights; applicable permissions and data-access terms remain component-specific. No secrets, production seed bank, raw trajectories or original local logs are included in this supplement.
