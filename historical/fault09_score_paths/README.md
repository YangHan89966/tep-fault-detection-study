# Historical Fault 9 scalar score paths

This archive contains previously exported scalar score paths for ten fixed
development splits (D01-D10) of the Tennessee Eastman Process data. The archive
identifier S30 distinguishes this diagnostic export from its upstream model
snapshot, S28. It does not denote an additional experiment.

The material is a historical diagnostic resource: it contains no newly simulated
trajectories, no refitted models, no new calibration and no new performance
comparison. Candidate selections, normalization, transform parameters and alarm
thresholds were fixed before the export. The ten splits reuse the historical
development pool; do not count them as ten independent newly generated datasets.

## Contents and coverage

- `inputs/PCA_STATE.npz` and `inputs/RESIDUAL_BASIS.npz`: fixed PCA and residual
  projection states.
- `inputs/D01` through `inputs/D10`: model cores, normal-design states, transform
  states, original outer-calibration and evaluation records, selected candidate
  definitions (`CHOICES.json`) and exact run-role assignments (`ROLES.json`).
- `scores/D01` through `scores/D10`: scalar paths (`u`), run IDs, selected axis
  indices and fixed transform parameters/thresholds. Each split includes 59 normal
  outer-calibration runs, 60 normal audit runs and 75 Fault 9 audit runs, all from
  the historical run IDs 1-250. Each path retains 960 samples. Duplicate selected
  axis indices are stored once: D01 and D06 contain two axes, other splits three.
- `scores/PARITY_CHECKS.json`: original export comparisons, with paths expressed
  relative to this archive root.
- `SOURCE_BINDING.json`: the upstream archive identity and snapshot hashes.
  Its `MODEL.npz` hashes identify the original upstream model files, not the
  extracted `MODEL_CORE.npz` files. Do not compare those different objects as
  though they were identical files. The archive's current member hashes are in
  `PAYLOAD_MANIFEST.json` and `RESULTS_SHA256.txt`.
- `INPUT_ACCESS.json`: identities and access/projection checks for the historical
  raw sources used by the original export. `RUNTIME.json` records that export's
  software environment. `COMPLETE.json` records its completed export checks.
- `code/`: numerical helpers, the original-source export program, its synthetic
  self-test and a saved-path verification program.

There are 92 numerical NPZ files: 52 model/state/record files under `inputs/` and
40 score/transform files under `scores/`. These files are preserved byte for byte
from the supplied historical archive. Raw 52-variable process trajectories are
not included. This archive alone cannot regenerate the simulator trajectories,
refit the historical models, or establish full raw-data coverage for the paper.

## Score definitions

The stored `u` has shape (runs, 960, selected axes). Axis and candidate identifiers
are zero based. Historical keys `old_signed_axis`, `new_signed_axis` and
`new_squared_axis` mean, respectively, the selected mean-pool signed axis,
moment-pool signed axis and moment-pool squared axis. The words `old` and `new`
in these keys distinguish the historical candidate families; they do not imply
new data generation in this release. The `new_labels` field in `CHOICES.json`
similarly lists moment-pool direction labels.

For each axis, the signed family uses `u`; the absolute family uses
`(abs(u) - abs_mean) / abs_sd`; the squared family uses
`(u*u - sq_mean) / sq_sd`, with the saved per-axis statistics. The two-sided
CUSUM recurrence starts at zero and uses k = 0.05:

    cp[t] = max(0, cp[t-1] + v[t] - k)
    cm[t] = max(0, cm[t-1] - v[t] - k)
    d[t]  = max(cp[t], cm[t])

The alarm comparison is strictly `d > threshold`. Fault onset is zero-based
index 160 (sample 161); the post-fault horizon is 800 samples. At the historical
3-minute sampling interval this is 40 hours. The helper `event_rows` preserves
the historical definitions: a clean detection requires no pre-fault alarm;
non-detection or a pre-fault alarm receives an 800-sample event-loss penalty.
Do not substitute another recurrence ordering or threshold comparison when
checking numerical parity.

## Verify the supplied numerical results

Use Python 3 with NumPy and pandas. The recorded export environment used Python
3.12.3, NumPy 2.2.6 and pandas 2.2.3; other versions may require compatibility
testing. From the extracted archive root:

    python code/export_selftest.py synthetic_selftest.json
    python code/verify_saved_scores.py . --report saved_score_verification.json

The first command runs synthetic-interface tests only. The second verifies all
member hashes in the manifest, reconstructs CUSUM paths from the saved scalar
scores, checks the recorded run order, compares normal and fault integer outcomes
exactly, and checks outer-calibration peaks with the original tolerances
(rtol = 1e-9, atol = 1e-8). Its report states whether those peaks also agree
exactly. It performs no fitting, calibration or simulation. Output reports are
new files and are not listed in the distributed manifest.

The verification program also accepts the ZIP path as its first argument, if
the code files have been extracted separately. The public manifest covers every
distributed file except the two hash manifests themselves; neither manifest
includes itself or the other manifest, avoiding circular hash definitions.

## Re-export from original raw sources, if available

`code/export_scores.py` requires the exact original raw-source identities. Both
sources are external to this archive:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `faulty_testing_full_20faults.float64.npy` | 4224000128 | `486532e9a139461396405d2ab00dae75a653fc5f0de502f1d4c3dfb6ec99197e` |
| `normal_validation_runs_1_250.pkl` | 105601801 | `112c344ec6394ca48ce70e35acd1e2b55efdce9c792f933a8c9b81541e0125c2` |

The matrix reader scans metadata but decodes only Fault 9 development process
rows (run IDs 1-250). It does not decode the held-out final process values.
The pickle adapter is restricted to the stated historical source and is not a
general-purpose safe loader for arbitrary pickle files.

Create a separate `reexport/` directory, copy `inputs/`, `code/`,
`EXPORT_SCOPE.json` and `SOURCE_BINDING.json` into it, and leave its `scores/`
directory absent. Then provide explicit paths to your source files:

    python code/export_scores.py --root reexport --output-parent . \
      --matrix data/faulty_testing_full_20faults.float64.npy \
      --normal data/normal_validation_runs_1_250.pkl

The `data/` paths above are examples relative to the current directory. The
export fails if source sizes/hashes do not match, and it must not be made to
accept substitutes by changing those checks. The source data and a new export
have additional storage requirements beyond this small diagnostic archive.
This packaging revision does not claim a new end-to-end raw-source export.

## Rights and provenance

The supplied historical archive contained no standalone license or copyright
notice. This revision does not assign a new license or infer permission to
redistribute third-party source data. Source identities, scientific definitions
and original numerical validation records are retained. Licensing must follow
the rights and license statement accompanying the final deposit; no third-party
attribution has been removed from the supplied code.
