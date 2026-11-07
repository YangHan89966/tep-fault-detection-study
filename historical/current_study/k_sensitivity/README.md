# CUSUM reference-value sensitivity

This historical analysis reuses existing N and DV trajectories. It was exploratory and specified after the original outcomes were inspected; it creates no new test population. The fixed comparisons are k=0.05, k=0.07 and development selection from {0.01, 0.03, 0.05, 0.07, 0.10}, with corresponding normal calibration. PCA, direction construction, orientations, transformations and score scalings remain fixed.

## Saved-result calculation

From this directory, with the documented NumPy environment:

```sh
python -B saved_results/reaggregate_s18.py --output NEW_DIRECTORY --bootstrap
python -B verify_saved_sources.py
```

The output directory must be new. The reaggregation script reads saved alarm arrays and recomputes N/DV endpoints, recorded selections and saved N paired intervals. Omitting `--bootstrap` skips N interval recalculation. Saved DV interval results are supplied; the script does not regenerate their draws. These calculations do not require raw process arrays, simulation or model refitting.

## Contents

| Path | Purpose |
|---|---|
| `saved_results/` | Full-precision tables, paired alarm arrays, fixed comparator assets and reaggregation code |
| `source/scientific_runner.py` | Extracted scientific functions for loading, calibration, selection, scoring and aggregation |
| `source/reference_n.py`, `source/reference_dv.py` | Historical reference implementations |
| `source/assets/` | Fixed PCA and N/DV states, reference arrays and input identity indexes |
| `source/EXTRACTION_MANIFEST.json` | Source identities and extraction boundary |
| `replay_existing_trajectories.py` | Wrapper accepting explicit roots for matching external trajectories |

## Existing-trajectory scoring

Inspect supplied source and asset identities:

```sh
python -B replay_existing_trajectories.py --inspect
```

To score matching external data:

```sh
python -B replay_existing_trajectories.py --only N --auto-k --n-development-root PATH_TO_N_DEVELOPMENT --n-final-root PATH_TO_N_FINAL --output NEW_N_OUTPUT
python -B replay_existing_trajectories.py --only DV --auto-k --dv-root PATH_TO_DV_ROOT --output NEW_DV_OUTPUT
```

`N_INDEX.json`, `N_AUTO_INDEX.json`, `DV_INDEX.json` and `DV_AUTO_INDEX.json` under `source/assets/` define exact relative paths and identities. Raw data are not included here. A fresh complete raw-data replay in another environment requires its own validation against the stored references. Expected dimensions are 960 by 52, float64, with finite process values.

## Statistical scope

N includes 17 fault groups with 250 trajectories each and equal-weight macro aggregation; its 250 normal test identities are shared across fault-specific monitors. DV includes Faults 3, 9 and 15 with 250 tests per fault and a shared 250-run normal panel. N and DV remain separate datasets.

N's primary loss penalizes misses by 2,400 minutes while retaining detections after prefault alarms; its auxiliary complete-episode loss also penalizes prefault alarms. DV uses the complete-episode definition and requires clean early detection. Normal false-alarm rate means any alarm during a separate complete normal run. The supplementary definitions retain the precise endpoint and resampling conventions.

The recorded sensitivity execution used Python 3.12.3 and NumPy 2.2.6; supplementary reaggregation used Python 3.13.5 and NumPy 2.3.5. Environment differences can affect floating-point replay. Preserved reference outcomes remain the comparison targets. Recalculation checks consistency of the saved computation; it does not establish independent raw-to-score replication or future learning performance.
