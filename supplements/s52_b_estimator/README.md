# Table S52: sensitivity to the estimator of B

This supplement contains the existing-data analysis for Fault 9 using three temporal covariance estimators: original, batch means, and Bartlett HAC. It includes all five fitted states for each estimator, frozen fitting inputs, calibration and selection records, run-level evaluation outcomes, and the original numerical implementation. No new Tennessee Eastman simulations were generated for this supplement.

## Contents

- `results/state{1..5}_{original,batch,HAC}/`: 15 complete fitted states. Each has `DESIGN.npz`, `MODEL.npz`, `PARAMETERS.npz`, `INNER.npz`, `OUTER.npz`, `SELECTION.json`, `SELECTION_SCORES.csv`, and `EVALUATION_EVENTS.npz`.
- `results/A7_ENDPOINTS.csv`: 30 signed-pool endpoint rows, covering two direction pools in each fitted state.
- `results/A7_PAIRED_CONTRASTS.csv`: 15 state-specific paired loss contrasts with 98.3333% bootstrap intervals.
- `results/A7_B_EIGENVALUES.csv`: the 15 estimator diagnostics.
- `results/A7_ORIGINAL_B_REPLAY.csv`: five checks against the original frozen baseline.
- `source/engines/A2_A7/`: unchanged source code, input manifest, fixed PCA/residual basis, five normal-design and fault-fitting moment arrays, and five baseline event archives.
- `summary/`: the unchanged final aggregation script and original five-state summary tables, including the reported marginal 95% intervals.
- `table/Table_S52.csv`: the submitted table, retained without numerical changes.
- `RAW_INPUT_IDENTITY.csv`: the 2,825 distinct raw trajectories required by the manifest, with their canonical identifiers, relative paths, byte counts, SHA-256 hashes, and recorded loader read counts.
- `ROLE_COUNTS.csv`: run-role counts for each state.
- `SOURCE_MAP.csv`: archive-relative provenance and checksums of the unchanged supplied files.
- `SHA256SUMS.csv`: checksums of all files in this supplement, excluding the checksum file itself.

## Design and pairing

Each state uses 40 normal design runs, 32 normal anchor runs, 59 inner-calibration runs, 75 fault-selection runs, and 259 outer-calibration runs. Evaluation uses the same 250 fault runs and 250 normal runs across the five states and the three estimators. The test panel is shared; it does not become 1,250 independent runs when five fitted states are evaluated.

The original B estimator uses the covariance of the first 120-point sums divided by the square root of 120. Batch means uses eight nonoverlapping 120-point blocks in each of 40 design runs: 320 blocks nested within 40 runs. HAC uses Bartlett weights `max(0, 1 - lag/120)`, within-run lag pairs, and denominator `40 * 960`; cross-run lag pairs are excluded. All three estimators retain the same PCA and design whitening, then relearn directions, select candidates, and recalibrate thresholds. The outer threshold is the 247th order statistic of 259 calibration maxima. The CUSUM reference value is 0.05.

The saved `EVALUATION_EVENTS.npz` arrays contain `methods`, `thresholds`, `fault_event_loss_minutes`, `fault_clean`, and `normal_alarm`. Array row order is the corresponding `fault.ids` or `normal.ids` order in `source/engines/A2_A7/A7_INPUTS.json`. The method names `mean_pool_signed` and `moment_pool_signed` correspond to the table's signed mean-shift and second-moment pools. C2 is second-moment minus mean-shift. Loss is in minutes; a run without clean detection has loss 2,400 minutes. For clean detections, the saved loss is the first detection delay, so clean detection within 8 h is obtained from `fault_clean` and loss at most 480 minutes.

The fixed residual-basis filename contains `PCA500`; its original filename is retained. The actual analysis uses the frozen PCA and residual-basis arrays supplied together in `source/engines/A2_A7/actual_inputs/`, and the basis has been checked for orthogonality to those PCA loadings. Filename labels are not used to infer the estimator or fitting sample size.

## Check the saved results without raw trajectories

Use Python 3.10 or later and NumPy 1.22 or later. From this directory, run:

```text
python verification/verify_table_s52.py --out verification_output
```

This checks the saved arrays and recomputes aggregate results; it does not simulate trajectories or fit new scientific models. Use a new output directory. The supplied numerical source files and results are unchanged.

The 30 source endpoint rows, 15 state-specific paired loss intervals, all table point summaries and ranges, and all 12 table C2 point estimates and 95% intervals have been checked against the saved evaluation events. The selected thresholds exactly reproduce the saved rank-247 calibration values. For all five original-B states, the complete saved nine-method thresholds, loss, detection, and normal-alarm vectors match their frozen baselines.

The submitted table's 95% intervals are across-state, run-paired marginal intervals. For each physical test run, the script averages the paired contrast across the five fitted states, then resamples the 250 runs with 20,000 bootstrap draws. It uses NumPy PCG64, seed 2026100907 for fault outcomes and 2026100908 for normal false-alarm outcomes, with shared indices across estimators and linear percentile quantiles. These intervals condition on the five fitted states; they are post hoc marginal intervals, not a simultaneous familywise analysis. The original per-state 98.3333% intervals are retained separately and must not be interpreted as the table's five-state 95% intervals.

The final aggregation source and seeds are included. To rerun that unchanged aggregation directly, use:

```text
python summary/summarize_a7.py --a7-dir results --engine-dir source/engines/A2_A7 --out NEW_SUMMARY_DIRECTORY
```

The dispatcher in `source/code/` is retained because the original aggregation records its checksum; it is not executed by the aggregation.

## Re-run the original analysis from archived trajectories

The raw trajectories are provided separately in the associated Zenodo record, `10.5281/zenodo.22084007`, in the `tep-r5-raw-rep01` through `rep05` and `tep-r5-raw-shared_test` archive groups. Extract those archives so that the supplied root contains `data/rep01/`, ..., `data/rep05/`, and `data/shared_test/`. Use the exact trajectory bytes identified in `RAW_INPUT_IDENTITY.csv`. The loader checks the original SHA-256 hashes before reading arrays.

```text
python source/engines/A2_A7/relearn_native.py A7 --manifest source/engines/A2_A7/A7_INPUTS.json --raw-root PATH_TO_EXTRACTED_R5_ROOT --out NEW_OUTPUT_DIRECTORY
```

Use a new output directory outside the supplied inputs and results. This command performs the retained downstream fitting, selection, calibration, and evaluation on existing raw trajectories; it does not call a simulator. It is more expensive than checking the saved results. The raw trajectories were not downloaded or re-read during preparation of this supplement; their identity catalog was cross-checked against the original run's loader receipts. All 17 supplied fixed input assets were rehashed and matched the original run's input-hash records.

## Provenance

Scientific code, numerical arrays, submitted table, and scientific result tables are copied byte for byte. The public raw-input identity table is a documented derivative of the original manifest and loader receipts: canonical identifiers, counts, and hashes are preserved, while host-specific absolute path strings are represented by portable paths. Original machine execution logs and unrelated analyses are not part of this supplement. Documentation, public checksums, and the independent verification script were added for this release.
