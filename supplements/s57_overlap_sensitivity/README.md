# Self-overlap exclusion and recalibration sources (Tables S57 and D14)

This component contains saved monitoring statistics, run identities, exclusion masks, supplementary thresholds, recorded results and analysis code for the self-overlap sensitivity analysis. The original monitors, learning states and preprocessing remain fixed. The three scenarios are the original evaluation, exclusion of self-overlapping test runs, and exclusion of self-overlapping test and calibration runs with recalibration.

## Files and scope

- `final_validation/`: final single-fit validation (DV), five-state validation (R5), and the five full-variable adapted standard monitors in each recording view and panel. Each unit includes sufficient statistics, run-level events, endpoints, calibration identities and exclusion flags. `final_validation/CODE/a4_core.py` contains the final paired bootstrap schedule.
- `fixed_panel2/`: saved paired signed scores, original outcome records, case identities, thresholds, code and results for the fixed historical monitor in recording Panel 2.
- `earlier_branches/`: preserved code and results for the original test exclusion analysis, full-monitor/reference comparisons, fixed Panel 1, supervised recording relearning and the Panel 1 calibration leave-one-out analysis. These directories retain their historical filenames; the earlier partial analysis is not the final validation implementation.
- `shared_inputs/`: the archived run inventory used for exclusions.
- `final_summary/`: final table summaries and the reconciliation of previously reported point estimates.
- `original_reported_intervals/`: original source tables for retained study confidence intervals.
- `TABLE_VALUE_SOURCE_MAP.csv`: 96 displayed cells in S57 and 96 in D14 mapped to their archived numerical source; all match at the reported precision.
- `SOURCE_FILE_MANIFEST.csv`: source member, original checksum, delivered checksum and any metadata-only path transformation for each delivered file.

## Replaying the final validation results

Use Python 3.10 or later with NumPy 1.22 or later. The archived final run used Python 3.12.3 and NumPy 2.2.6. From this directory, run:

```bash
python replay_S57_saved.py --out /path/to/new_output_directory
```

The output directory must not already exist. This entry point reconstructs the saved endpoints for DV, R5 and the standard-monitor panels from pre-onset maxima and post-onset running maxima; checks calibration ranks, masks and archived events; and uses the original final bootstrap code for the DV/R5 contrasts. It does not simulate trajectories or refit models. The archived 20,000-draw resampling schedule averages fixed-state paired differences within each physical run before resampling runs.

The entry point is an A4-only adaptation of the archived saved-results replay. Its numerical recovery helpers are unchanged. It does not invoke the sample-level analysis, fixed historical recording monitors or supervised relearning branches. For recording Panel 2, the original separate command is:

```bash
python fixed_panel2/code/recompute_A4_fixed_panel2.py --root fixed_panel2 --out /path/to/new_panel2_output
```

The original test-only and partial full-reference scripts retain their historical input arguments and rely on historical alarm archives that are not included in this component. They are provided as provenance code with recorded outputs; this component does not establish that their full input layout is available in another deposit. The final validation entry point is the tested replay route for the completed DV/R5 and standard-monitor branches.

## Replaying supervised recording relearning

The required original scientific members have been repackaged in `shared_inputs/recording_relearning_inputs.zip`. Its nested layout is compatible with the unchanged source scripts. The 11 selected members preserve their original bytes and checksums, including protocol, threshold locks, model definitions, calibration statistics, evaluation scores and case identities. Editorial material is not included. NumPy and pandas are required for these commands:

```bash
python earlier_branches/recording_relearning/extend_a4_relearning.py --relearn-archive shared_inputs/recording_relearning_inputs.zip --a4-root shared_inputs/A4_reference --out /path/to/new_relearning_output
python earlier_branches/calibration_leave_one_out/enumerate_panel1_calibration.py --relearn-archive shared_inputs/recording_relearning_inputs.zip --inventory shared_inputs/RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv --out /path/to/new_leave_one_out_output
```

Use a new output directory for each command. These commands operate on frozen scores and calibration samples and do not refit the models. The Panel 1 leave-one-out analysis is a finite enumeration of 59 deletions; its frequencies are not sampling probabilities.

## Interpretation

The original-study confidence intervals are retained as reported in the study. Replaying an original-scenario row using the sensitivity analysis seed is a consistency check, not a replacement for that reported interval. The final exclusion-scenario intervals use the A459 schedule in `a4_core.py`; older supplementary scripts use different recorded seeds.

Cumulative maxima support first-crossing and run-level metrics. They must not be used to estimate sample-level alarm occupancy. Exclusion of within-run self-overlap does not establish independence between different simulated runs.

No host-specific source paths are needed to use these files. Where present in standard-panel identity tables, such paths were replaced by relative case paths; run identifiers, numerical data and source checksums were preserved. Scientific code, result tables and arrays are otherwise copied from their identified archives.

## Verification of this component

All 96 displayed numerical cells of Table S57 and all 96 of Table D14 match the identified archived results at the manuscript precision. A saved-statistics replay reconstructed 167,568 run-event rows and 127 supplementary thresholds for the final validation and standard-monitor branches, reproducing 28 archived CSV files byte for byte. Separate saved-score replays reproduced 3 fixed-Panel-2, 4 supervised-relearning and 3 leave-one-out CSV files byte for byte. These checks used Python 3.12.14 and NumPy 2.3.5. The detailed scope and receipts are recorded in `VERIFICATION.json`. No new trajectories were simulated and no model was refitted.
