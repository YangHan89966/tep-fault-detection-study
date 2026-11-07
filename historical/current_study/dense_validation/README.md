# Targeted dense validation: Faults 3, 9 and 15

DV is a completed historical study of 1,915 physical trajectories: 915 downstream development/calibration trajectories and 1,000 tests. It conditions on the original assessment's normal-training PCA. A set of 27 configurations was fixed before testing; DV does not add methods or trajectories to the separate 4,500-run original assessment.

## Contents

- `frozen_models/`: fitted states and saved numerical checks.
- `results/`: saved alarm arrays, run-level events, endpoint tables and primary contrasts.
- `analysis/`: saved reaggregations and plotting data.
- `source_original/`: numerical implementations, fixed PCA assets and simulator source.
- `protocol/` and `provenance/`: study definitions and historical source, seed, model and evaluation identities.
- `tools/`: saved-result recalculation and raw-data scoring entry points.

## Recalculate saved outcomes

From this directory, with NumPy and SciPy available:

```sh
python -B tools/verify_saved_results.py --out NEW_OUTPUT_DIRECTORY
```

This entry point recalculates endpoints and recorded bootstrap comparisons from saved alarm outcomes. It is distinct from rescoring process arrays. Input identity checks must be satisfied; a successful archive-integrity check alone does not establish scientific replay.

## Score separately supplied raw data

```sh
python -B tools/replay_raw.py --raw-root PATH_TO_DV_RAW_DATA --out NEW_OUTPUT_DIRECTORY
```

The raw-data archive must contain the matching 1,915 trajectories and its identity records. The calculation uses fixed projection/scoring functions and model states without learning or recalibrating. It compares calculated alarm records with the saved reference. It must not resolve a mismatch by changing thresholds or dropping trajectories.

The original raw process arrays are not included in this code component. Historical hashes identify the corresponding original objects; the root file manifest describes this archive's present files.

## Interpretation

Each fault test group contains 250 trajectories; the 250 normal tests are shared across monitors. Conditional intervals concern the fixed PCA and fitted monitor states, not repeated-training uncertainty. Preserved adverse results remain part of the evidence. Different nominally calibrated monitors can have different realized normal false-alarm rates. Distinct random initial states alone do not establish independent, non-overlapping random substreams.
