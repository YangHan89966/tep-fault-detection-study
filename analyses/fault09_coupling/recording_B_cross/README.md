# Recording mode × temporal covariance estimator: Fault 9

This module contains the completed follow-up sensitivity analysis of Fault 9. It compares post-update (`native`) and controller-input (`snapshot`) recordings using the original temporal covariance estimate B and a batch-means estimate. Each setting includes a second-moment and a mean-shift candidate pool, giving eight signed two-sided CUSUM monitors.

Study identifier: `TEP_P1_F9_RECORDING_B_CROSS_20261030_v1`.

## Results

All monitors use the same 306 development trajectories, 199 normal calibration trajectories, and paired evaluation panel of 120 normal and 120 Fault-9 trajectories. No additional process trajectories were simulated for this comparison.

| Recording | B estimator | Pool | Clean detections by 40 h | By 8 h | Normal alarms / 120 | Mean penalized loss (min) |
|---|---|---|---:|---:|---:|---:|
| Post-update | Original | Second moment | 38/120 | 4/120 | 7 | 2056.275 |
| Controller-input | Original | Second moment | 119/120 | 85/120 | 7 | 401.050 |
| Post-update | Batch means | Second moment | 46/120 | 3/120 | 3 | 2023.400 |
| Controller-input | Batch means | Second moment | 118/120 | 81/120 | 2 | 436.200 |
| Post-update | Original | Mean shift | 16/120 | 0/120 | 11 | 2295.125 |
| Controller-input | Original | Mean shift | 112/120 | 44/120 | 10 | 887.975 |
| Post-update | Batch means | Mean shift | 16/120 | 0/120 | 7 | 2295.150 |
| Controller-input | Batch means | Mean shift | 95/120 | 29/120 | 6 | 1268.475 |

A clean detection excludes trajectories with a pre-fault alarm. Non-detection or a pre-fault alarm receives a 2400-minute loss. The recording difference remains large with the batch-means estimate. Empirical normal alarm rates change as well; the two estimators use the same calibration rule, rather than matched observed false-alarm rates.

This is a post-hoc sensitivity analysis on previously analyzed recording panel 2. The 95% intervals condition on the fitted states and realized calibration. They do not include training or calibration variability or resolve dependence from shared simulator random-stream segments. Recording-specific PCA and support spaces are retained, so the comparison is between the corresponding fitted pipelines.

## Contents

- `PROTOCOL.json` and `EXPERIMENT_SPECIFICATION.txt`: fixed design and endpoint definitions.
- `results/`: all eight monitor results; per-trajectory events; calibration and evaluation scores; model and threshold locks; candidate states; paired bootstrap draws and contrasts.
- `assets/`: frozen input identities, PCA state, and the 40 normal residual trajectories needed to independently recompute the two B matrices.
- `reference/`: original numerical baseline evidence used by the reproduction gate. Historical narrative reports are omitted from this public module.
- `vendor/`: numerical routines required by the runner.
- `code/`: raw-data recalculation and paired analysis scripts.
- `audit/`: independent scalar CUSUM/event checks, independent paired-bootstrap verification, and public-file integrity checks.
- `validation/`: verification reports generated for this release.
- `PUBLICATION_CHANGES.json`: original/public metadata digest mapping and omission records.
- `PUBLIC_MANIFEST.json`: SHA-256 digests of the distributed public files.

## Reproduce the published outputs

Requirements are Python 3.10 or later and NumPy 1.26 or later, below version 3. The execution used Python 3.12.3 and NumPy 1.26.4. The public checks were also run using the environment recorded in `validation/RELEASE_VALIDATION.json`.

From this module directory, install the dependency if necessary:

```sh
python -m pip install -r requirements.txt
```

Run both independent checks:

```sh
python audit/verify_cross_results.py --returned . --reference-package . --output verification_results/numerical
python audit/verify_bootstrap.py --output verification_results/bootstrap
```

The first command validates public and historical digests, recomputes 1,843,200 evaluation alarm decisions from saved scores using a separate scalar CUSUM implementation, checks all 1,920 per-trajectory event rows and eight result summaries, reproduces the baseline, and recomputes the batch covariance from the included 40 normal residual trajectories. The second independently reconstructs all 84 contrast/endpoint estimates and percentile intervals from the saved common bootstrap indices. Neither check imports the runner's numerical implementation.

To regenerate the paired tables in a separate directory:

```sh
python code/analyze_cross.py results/PER_RUN_EVENTS.csv --output analysis_replay
```

The archived results remain in `results/`; regenerated files are written to the requested output directory.

## Recalculate from the original raw trajectories

The complete raw-data replay additionally needs the original two workspaces:

| Source workspace | Required trajectories | Identity manifest |
|---|---:|---|
| `TEP_F9_Supervised_Relearning_v1_20261006/workspace` | 306 development | `assets/EXPECTED_PARENT.json` |
| `TEP_F9_Dynamic_Validation_v1_20261006/workspace` | 199 normal calibration + 120 normal evaluation + 120 Fault-9 evaluation | `assets/INPUT_MANIFEST.json`, panel `D439_supplement` |

These original 52-variable raw trajectories are external dependencies, not files in this module. Raw input identity is checked against the frozen hashes. The included score-level data are sufficient for the two independent result-verification commands above.

Copy `SOURCE_PATHS.json` to a new file named `source_paths.local.json` and fill the paths to the two source workspaces, for example:

```json
{
  "relearning_workspace": "/path/to/development/workspace",
  "D439_supplement": "/path/to/validation/workspace"
}
```

Then use fresh output directories:

```sh
python code/run_cross.py --config source_paths.local.json --preflight --output outputs/input_check
python code/run_cross.py --config source_paths.local.json --output outputs/raw_recalculation
```

The runner first reproduces the four original monitors, then fits, selects, calibrates and evaluates the batch-means monitors. It reads source trajectories without modifying them. A standalone software self-test is also available:

```sh
python code/run_cross.py --self-test --output verification_results/software
```

## Publication metadata and evidence identity

All distributed CSV and NPZ files retain their source bytes. Machine-specific paths in `results/INPUT_PREFLIGHT.json` were replaced by descriptive placeholders. The historical `results/FINAL_LOCK.json` is retained unchanged; its original digest for the edited preflight document is connected to the public digest through `PUBLICATION_CHANGES.json`.

`reference/FINAL_LOCK.json` is explicitly marked as a public projection: two narrative-report references were removed, while all listed scientific-file digests remain original. Its source-document digest and the omitted reports' digests are retained in the publication mapping. This projection is not presented as an unmodified historical lock. Public verification accepts only the explicitly registered preflight transformation when checking a historical file digest and otherwise requires an exact match. `PUBLIC_MANIFEST.json` covers the distributed files, including the mapping and verification code.

See `LICENSE_NOTICE.txt` for licensing and dependency notices.
