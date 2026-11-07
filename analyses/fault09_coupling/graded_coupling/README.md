# Graded measurement-noise coupling for Fault 9

This module quantifies how a fixed historical Fault 9 monitor responds when the
noise in recorded measurements shares progressively more components with the
noise used by the controller. It uses all 799 applicable trajectories from the
previous independent-stream confirmation panel: 199 normal calibration runs,
400 normal evaluation runs, and 200 Fault 9 evaluation runs.

Each measurement noise value uses the original transform
`sigma * (sum of 12 uniform components - 6)`.
The five coupling levels retain 0, 3, 6, 9, or 12 original components and replace
the remainder with components from a separate keyed Philox stream. This preserves
the original marginal noise distribution, variance, and measurement hold schedule.
The physical process, controller outputs, historical projection, and CUSUM rule
are fixed. Actual post-update records form a separate sixth arm; their physical
measurement time differs from the five constructed coupling arms.

The study is a post hoc mechanism sensitivity analysis on an already analyzed
panel. Its intervals describe trajectory variation conditional on the fixed
monitor and realized calibration. The five coupling settings are an idealized
experimental family rather than measurements of an industrial installation.

## Results and contents

Forty-hour clean detection rises from 60/200 at zero shared components to 176/200
at complete sharing. At three-quarter sharing, it is 136/200, with normal alarms
in 20/400 trajectories. The main table reports normal alarms alongside every
detection result.

| Path | Contents |
| --- | --- |
| `results/Main_results.csv` | Six-arm detection, normal alarms, score variance, thresholds, and penalized loss |
| `results/Paired_contrasts.csv` | Thirty-six descriptive paired contrasts and 95% intervals |
| `results/Per_run_events.csv` | All 3,600 evaluation run-by-arm event records |
| `results/CALIBRATION.npz` | Calibration maxima, case identities, and thresholds |
| `results/EVALUATION.npz` | Evaluation case identities and event arrays |
| `results/BOOTSTRAP.npz` | All 20,000 paired resampling indices and contrast distributions |
| `results/figures/` | Three-panel figure in PNG, PDF, and SVG, with its caption |
| `scores/` | All 799 six-arm score matrices and source/noise-recovery records |
| `assets/` | Original historical affine monitor |
| `code/` | Noise recovery, score construction, calibration, analysis, and noise diagnostics |
| `audit/` and `checks/` | Independent numerical verification and original engineering checks |
| `reference/` | Identities and comparison outputs from the original confirmation panel |
| `engine/` | Pinned process source, observer code, upstream attribution, and license |
| `PROTOCOL.json`, `PROTOCOL_LOCK.json`, `CODE_LOCK.json` | Original pre-calculation scientific rules and freeze evidence |

## Environment

The verified environment is CPython 3.12.14 and NumPy 2.3.5. Figure generation
uses Matplotlib 3.10.8. Install the versions listed in `requirements.txt` in an
isolated Python environment. `public_replay.py` sets numerical-library thread
counts to one and writes recalculations to a separate directory.

Run the following commands from this module's directory. Choose unused output
directories outside the module.

```bash
python public_replay.py verify-files
python public_replay.py audit --output ../graded_coupling_audit
python public_replay.py synthetic --output ../graded_coupling_synthetic
python public_replay.py recalculate --output ../graded_coupling_recalculation
```

The independent audit reconstructs CUSUM paths, calibration thresholds, all
evaluation events, and paired bootstrap results from the stored scores. It does
not import the production numerical routines. `recalculate` runs the archived
analysis and noise diagnostics in a copy, compares every result CSV and NPZ
array with the archive, and runs the independent audit on that copy.

## Reconstructing scores from the original process records

The original measurements are supplied separately by these source archives;
their SHA-256 identities are listed in `SOURCE_ARCHIVES.json`:

- `TEP_P1_Normal_Calibration_Data_v1_20261026.zip`
- `TEP_P1_Normal_Evaluation_Data_v1_20261026.zip`
- `TEP_P1_Fault09_Data_v1_20261026.zip`

Extract the three archives into the same directory so that their
`independent_stream_panel/runs/` directories merge. Then run:

```bash
python public_replay.py rebuild --source-root /path/to/independent_stream_panel --output ../graded_coupling_reconstruction --workers 4
```

This command constructs a new working copy, recovers each measurement's original
noise components from its keyed random-call schedule, rebuilds all 799 score
files, calibrates the six thresholds, and evaluates the fixed panel. It compares
the reconstructed scores and results with the archived numerical payload and
runs an independent audit that also rehashes all supplied source files. It uses
the original trajectories; it does not generate additional process trajectories.

## Endpoint and interval definitions

A clean detection requires no alarm in records 1–160 and a first subsequent
crossing in records 161–960. Delay is `3 × (first post-fault index − 160)` minutes
using zero-based indices. The 8 h and 2 h endpoints include delays of exactly
480 and 120 minutes. A prefault alarm or failure to detect incurs a 2,400-minute
loss. Normal alarms cover the full 960-record, 48-hour trajectory.

Each arm uses the 190th ascending trajectory maximum among the same 199 normal
calibration runs. Actual normal alarm rates are reported separately. Paired
contrasts use 20,000 whole-trajectory bootstrap draws with seed `2026103009`;
normal and fault trajectories are resampled separately, with common indices
across all arms within each role. Contrast intervals are marginal 95% percentile
intervals. Single-arm rate intervals in the figure are pointwise Wilson intervals.

## Public-distribution provenance

Distribution version 1.1.0 preserves the original study identifier and protocol
version. All scientific CSV/NPZ files and all per-run score records are unchanged.
`PUBLIC_DERIVATIONS.json` records the original and public hashes for the one
frozen synthetic-test file whose machine-specific root was replaced with a
package-relative path, and for the public verification adapters. The original
`CODE_LOCK.json` is retained verbatim. Public verification checks all 61 frozen
file identities: 60 against their unchanged bytes and one against the explicitly
documented path-only derivation. Content checking remains active throughout.

`PUBLIC_RELEASE_MANIFEST.json` covers the files in this public module.
`PUBLIC_RELEASE_VALIDATION.json` records its portability and numerical checks.
For this distribution, the complete saved-score reanalysis and independent
result audit passed. A separate raw-record reconstruction check completed 207
cases (199 calibration and 8 normal evaluation); all 207 score files were
byte-identical to the archive. This distribution check stopped at that subset;
the original completed 799-case construction and its verification remain in the
scientific archive.
The completed source package's manifest and completion record are retained under
`provenance/` as historical records of that source package.

The upstream source and its license, contributor credits, and AI-assistance
acknowledgements are preserved in `engine/vendor/`. This distribution does not
change or replace those license terms.
