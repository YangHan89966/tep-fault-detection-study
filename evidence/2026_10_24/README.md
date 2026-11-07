# Independent-stream confirmation panel

This directory provides the fixed-panel results, model assets, analysis code and simulator source for the Tennessee Eastman Faults 3, 9 and 15 confirmation experiment.

The panel contains 1,199 physical trajectories: 199 normal calibration runs, 400 normal evaluation runs, and 200 fault evaluation runs for each fault. Paired recording views and five fitted states share these physical runs.

## Verify saved outcomes

From the repository root, run:

```sh
python3 -I -S -B verify_confirmation.py
```

This standard-library check verifies the source hashes and run identities, then recounts 13,200 run-arm events across 22 monitoring arms. It recomputes alarm counts, detection rates, mean penalized losses and primary point contrasts. It does not recompute bootstrap intervals or simulate trajectories.

The files at this level and in `results/` are the unchanged numerical records used by this check. `EXTRACTED_FILES.json` records their original source identities.

## Models and analysis code

`independent_stream_panel/` preserves all 112 files from the code, results and documentation archive dated 26 October 2026. It includes fixed PCA and model assets, calibration maxima, scoring and analysis code, the pinned simulator source, run provenance, protocols and six result CSVs.

Its release-format metadata differs from the original numerical records at this directory level. Both sets are retained in their own layouts; the numerical CSV values are unchanged. The release's original `FILE_MANIFEST.csv` also lists raw-data companion files supplied separately.

The 1,199 raw recording and chart files are external to this Git checkout. For full saved-chart analysis, combine the code archive and the corresponding raw-data companion archives in their original `independent_stream_panel/` layout. Then use the commands and dependencies in that directory's README. This repository's event recount can run without those external arrays.

## Attribution

The simulator is `tep` 2.5.0, pinned to commit `9a6c8e5fcef4a2850778704e7793c87b0a187005`. Its original licence and attribution remain in `independent_stream_panel/engine/vendor/`.

