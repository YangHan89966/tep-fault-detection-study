# Historical inputs

Included inputs comprise scientific code, selected fixed PCA and monitor states, calibration records, saved scores, event arrays, result tables and source identities. `data/RAW_DATA_SOURCES.md` and `data/DATA_MANIFEST.csv` identify the original historical TEP dataset.

Separate process-data inputs include the original historical matrices, the 1,915 dense-validation trajectories and the 5,575 R5 trajectories. R5 comprises 4,575 development/calibration trajectories and one shared 1,000-trajectory test panel. Their identity records are under `data/` and `current_study/r5_raw_identity/`.

`fault09_score_paths/` includes the S30 scalar-score component. `direction_experiments/` contains a compact S28 subset; full candidate model states and event payloads are in `TEP_P1_Historical_Direction_Experiments_S28_20261026.zip`. The source inventory records the exact source archive hash. These named companion archives require separate access; this directory does not establish their current online availability.

Saved scores and raw-input receipts have different purposes. Receipts identify the original inputs and support provenance checks; raw-to-score replay requires the matching numerical process arrays. The later independent-stream confirmation panel is a separate experiment and cannot replace historical inputs.
