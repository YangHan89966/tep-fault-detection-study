# Supplementary source analyses

These three components supply the saved inputs and analysis programs for Tables S52, S57, D14 and S65 in the final V10 paper. Files retain their source bytes and component-relative paths.

| Directory | Analysis | Supplied inputs |
| --- | --- | --- |
| [s52_b_estimator](s52_b_estimator/README.md) | Table S52: estimation of temporal covariance B | Fifteen fitted states, calibration records, evaluation events, fitting inputs and aggregation code |
| [s57_overlap_sensitivity](s57_overlap_sensitivity/README.md) | Tables S57 and D14: self-overlap exclusions and recalibration | Saved monitoring statistics, run identities, exclusion masks, thresholds, original intervals and replay code |
| [s65_sample_metrics](s65_sample_metrics/README.md) | Table S65: sample-level alarm occupancy | Thirty packed alarm archives, cohort identities, per-run counts, aggregation code and an independent count verifier |

The S52 and S65 table files match the V10 Data S1 files exactly. Historical source filenames retain their original version labels.

## Saved-result reproduction

Use Python with NumPy; the checked environment is Python 3.12.14 and NumPy 2.3.5. Run each command from its component directory. Choose a new output path outside this repository.

**S52:**

```sh
python -B verification/verify_table_s52.py --out PATH_TO_NEW_OUTPUT_DIRECTORY
```

**S57 and D14:**

```sh
python -B replay_S57_saved.py --out PATH_TO_NEW_OUTPUT_DIRECTORY
```

The S57 README also supplies separate commands for fixed recording Panel 2, supervised recording relearning and leave-one-out calibration. The earlier test-only/partial source files are historical records; the final-validation command above is the supported entry for DV, R5 and standard-monitor results.

**S65:**

```sh
python -B recompute_r5_sample_metrics.py --out PATH_TO_NEW_OUTPUT_DIRECTORY
python -B verify_s65_from_counts.py --report PATH_TO_NEW_REPORT.json
```

These commands operate on supplied fitted states, saved statistics or alarm masks. They do not generate new process trajectories or refit models. Full trajectory-based fitting and scoring require the separately identified raw inputs. S52's optional raw-input analysis is documented in its component README.

## Source archives

Each directory was extracted from the archive below, removing only its outer `S52/`, `S57/` or `S65/` directory. Component manifests and scientific provenance remain included.

| Archive | SHA-256 |
| --- | --- |
| `TEP_P1_S52_B_Estimator_Sources_Public_20261107.zip` | `aeda0182825d5b79ceda1cba0c6fbfa8f17c26a5a97cfbbe0d8cae1f94622d82` |
| `TEP_P1_S57_Overlap_Sensitivity_Sources_Public_20261107.zip` | `4cedaf4c00f95ac6a7d3f5aeb1cf14c34f78cc24f7fb11292bcae19879bc5f0b` |
| `TEP_P1_S65_Sample_Metrics_Sources_Public_20261107.zip` | `fb26c17134dcc0f2f295b87ece783eecdcd7e33bdc3e553e130ed5cae416cbcb` |

The three archives contain 422 files. Source attribution, original component terms and the repository's [licence scope](../LICENSE_SCOPE.md) apply.

