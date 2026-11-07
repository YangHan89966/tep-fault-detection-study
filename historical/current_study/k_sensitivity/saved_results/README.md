# Supplementary Data S18 — post-outcome CUSUM reference-value sensitivity

Companion to manuscript/SI v1.7.2, Section S18 and Tables S44–S46. Data-source version: the completed k extra-test combined report v1.0 (13 September 2026). This is a post-outcome analysis of existing N and DV trajectories, not another independent test population. No original primary state or result is replaced.

## What is included

- `N_MACRO.csv`: all eight N rows — the two Top-5 procedures under three policies plus original frozen Top-1/Q reference rows.
- `N_PER_FAULT.csv`: 136 N method–fault–policy rows; `DV_PER_FAULT.csv`: all 81 DV configuration–policy rows. No adverse valid branch is removed.
- `N_AUTO_SELECTION.csv`, `DV_AUTO_SELECTION.csv` and `*_FIXED.json`: the five candidate scores and all 61 selected values, including tie-breaking and data-role definitions.
- `N_PAIRED_MACRO.csv/json`: 36 exploratory N contrast/endpoint rows and calculation metadata. `DV_PAIRED_SENSITIVITY.json`: complete saved DV paired sensitivity outputs. `DV_PRIMARY_CONTRASTS.csv`: the three original comparison definitions under each policy; their point signs are not new confirmatory tests.
- `masks/`: original packed fault and normal alarm arrays from the valid N return (second run) and valid DV return (first run). The first run's failed N attempt supplies no result.
- `frozen_N_reference/`: unchanged N arrays included with the sensitivity runner; their method order is Adaptive Top-5, Fixed-AD Top-5, Independent Top-1, Q-CUSUM. These allow verification of the comparator rows, including pre-fault penalties.
- `implementation/`: the original pure projection/score/recurrence function copies; not a complete raw-data orchestrator.
- `SOURCE_PROVENANCE.json`: source archive identities, exact original member hashes, valid scopes and source environment. `SHA256SUMS.txt` covers this supplementary archive's members except itself.

## Definitions that must remain separate

All episodes contain 960 records, sampled every three minutes. The first affected sample is 161 (zero-based index 160). N's primary delay penalizes misses by 2,400 minutes but does not additionally penalize pre-fault alarms. N's early detection counts post-fault detections even after a pre-alarm. The auxiliary complete-episode loss penalizes either a pre-alarm or miss by 2,400 minutes. DV's primary loss already uses the latter definition, and its early detection requires no pre-alarm. All faulty runs remain in each denominator. Normal FAR is any alarm in 960 records of a separate normal run; it is not the first-160-record pre-fault rate.

N fixes 17 fault identities with 250 runs each and equal-weight macro aggregation. Its 250 normal identities are shared across monitors, not 4,250 independent normal observations. DV retains three separate faults, each with 250 faulty runs, and a shared 250-run normal sample. Do not pool N/DV denominators.

Every new k has a normally recalibrated threshold. N uses the inherited normal-validation percentile rule; DV selects using the inner 59/rank57 role, then applies independent outer 259/rank247 calibration. Automatic selection searches {0.01,0.03,0.05,0.07,0.10}, minimizes development complete-episode loss, breaks ties by distance to 0.05 and then smaller k, and stays fixed online. It does not reproduce the original Top-1 eight-point search or its lexicographic criterion.

## Reaggregate saved outcomes

With Python and NumPy installed, extract this archive outside any original research directory, then run:

```sh
python -B reaggregate_s18.py --output NEW_DIRECTORY --bootstrap
```

The output directory must not already exist. The script verifies exact copied-source hashes, computes N/DV endpoints from the masks, checks the 61 selections, checks all nine DV comparison point rows, and reproduces the saved N 20,000-draw paired intervals. Omitting `--bootstrap` performs only the point/selection checks. It does not simulate, retrain, select replacement methods, change a threshold, score raw process trajectories, or contact a network service. It validates numerical consistency of supplied records, not all scientific assumptions. The saved DV intervals are supplied as original results; this entry point does not regenerate their bootstrap draws.

Paired intervals are unadjusted exploratory 95% percentile intervals conditional on learned states and chosen k, not the 98.3333% intervals of the original DV primary test. They do not account for repeated training or post-outcome choice of analysis. Shared normal identities remain paired, and individual time samples are not independent resampling units.

## Reproduction limits

Original process trajectories and complete frozen model assets are not duplicated here. They belong to the separately documented N/DV companion archives. The supplied masks suffice to reaggregate the reported saved outcomes, not to certify raw-to-score correctness or independently regenerate the process. Repository accessibility must be verified separately; this archive makes no claim that an updated GitHub or Zenodo release is online.

The local sensitivity records specify Python 3.12.3 / NumPy 2.2.6. Integration reaggregation was performed with Python 3.13.5 / NumPy 2.3.5. Those are different task environments. Figure numbering and data in the original N/DV results remain unchanged.
