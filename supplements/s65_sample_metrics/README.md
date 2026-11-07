# Table S65: R5 sample-level alarm occupancy

This package supplies the saved alarm masks, run identities, original analysis
code, per-run integer counts and aggregates underlying Table S65. It supports
Faults 3, 9 and 15, sixteen monitor conditions and five fixed learned states.
`reference/Table_S65.csv` is the unchanged table from
`TEP_P1_Data_S1_Submission_v4_20261106.zip`.

## Reproduce the analysis

Use Python 3.10 or later with NumPy. From this directory, run:

```sh
python recompute_r5_sample_metrics.py --out replay_results
python verify_s65_from_counts.py --report counts_verification.json
```

The first command requires a fresh output directory. It reads the thirty
supplied compressed alarm NPZ files and checks their SHA-256 digests before
recomputing the results. It does not run the process simulator, fit a model or
change an alarm threshold. The second command independently reaggregates the
archived per-run counts, including the confidence intervals, and checks all
sixteen displayed Table S65 rows. The verification utility was added on
2026-11-07; the scientific analysis code and numerical inputs are unchanged.

## Definitions

- Actual sample-level detection is the percentage of post-fault samples with
  `S_t > h`, among samples 161–960 (800 samples). No permanent alarm latch is
  applied. Prefault sample FAR uses samples 1–160. Normal sample FAR uses all
  960 samples.
- Clean 40 h detection excludes runs with a prefault alarm and requires a
  post-fault alarm within the observation horizon. Sample occupancy and
  run-level detection are different quantities.
- Five learned-state values are averaged within each physical run before
  resampling. Intervals are marginal percentile intervals from 20,000
  whole-run bootstrap draws, conditional on the five fixed states and original
  thresholds. Fault and normal cohorts use separate resamples.
- The original scenario contains 250 physical evaluation runs per cohort.
  The alternative removes self-overlapping test runs, leaving 242 normal
  runs and 241, 234 and 245 fault runs for Faults 3, 9 and 15. Calibration and
  thresholds remain unchanged. This alternative does not implement
  calibration-set exclusions or threshold recalibration.
- Table S65 uses the original scenario. The three full-pool occupancy values
  are 72.9750%, 0.9138% and 26.9711% for Faults 3, 9 and 15, respectively.
  Decimal rounding gives 72.98%, 0.91% and 26.97%.

## Files and verification

`inputs/alarms/` contains thirty unchanged saved NPZ files (536,236 bytes in
total), covering both source families, five states and three faults.
`inputs/R5_EVALUATION_IDENTITIES.csv` supplies the 1,000 cohort identities and
test-exclusion labels. `results/` contains 40,000 per-run count rows, 160
state-level rows, 32 five-state rows, twelve paired contrasts and the latching
diagnostic. `verification/` records the new checks. `SOURCE_MAP.csv` and
`SHA256SUMS.csv` document file origins and digests.

The original-input replay reproduced all ten generated archived files
byte for byte. The public-input replay reproduced all nine numerical/specification
outputs byte for byte; only `INPUT_SHA256.csv` differs because the public
provenance manifest has been cleaned. Both replays checked 120,000,000 alarm
bits, 270,000 saved endpoint/occupancy values and 190 duplicated-monitor or
archived-table comparisons. The independent count verification also checked
all aggregation results and bootstrap intervals against the archived outputs.

The original script and scientific NPZ/CSV files are unchanged. Two provenance
JSON files were prepared for public use: absolute computer paths were removed
from `ORIGINAL_ALARM_INPUT_MANIFEST.json` and `SOURCE_BINDINGS.json`, and the
manifest size and checksum in the bindings were updated. The public
`results/INPUT_SHA256.csv` describes these supplied inputs. The corresponding
original input hashes and prior replay record are retained separately under
`provenance/original_run/`. These metadata changes do not alter any scientific
arrays, counts, thresholds or results.

## Scope

This package reproduces the R5 saved-alarm analysis. It does not regenerate
the process trajectories or refit the learned monitoring directions; those
are separate stages of the study. Stored flags are treated as the archived
actual alarm decisions, rather than being recomputed from raw statistic paths.

The legacy field `original_A5_historical_D01_Braatz_complete: false` in
`ANALYSIS_SPEC.json` and `inputs/ORIGINAL_MISSING_A5_INPUTS.csv` refer to a
different historical D01/Braatz analysis. The latter is retained as an original
input checked by the source script. These historical notes do not indicate a
missing input for this R5/Table S65 reproduction. The source code and the
required R5 alarm inputs for this package are complete.
