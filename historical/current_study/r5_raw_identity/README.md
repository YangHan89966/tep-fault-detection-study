# R5 trajectory identities and model chronology

The original Stage-A index contains 4,575 completed development/calibration trajectories. The Stage-C index contains 1,000 completed shared tests. The fixed allocation covers five groups of 915 development/calibration runs and one shared test panel, for 5,575 unique trajectories.

These files are indices and completion records, not raw numerical arrays. Model-lock records connect the completed development data to all 135 configurations before test generation.

## Raw export format

The existing-data exporter preserves each original `OUTPUT.npz` file and groups the files into independently extractable, size-bounded ZIP parts for `rep01` through `rep05` and `shared_test`. The actual parts are listed in the export catalog.

Each portable `MANIFEST.json` identifies the exported files. `source_receipts/COMPLETE.json` retains the hashes of the original task and metadata files. Host-specific absolute-path strings in exported `METADATA.json` are redacted, with JSON-pointer changes recorded in `SOURCE_IDENTITY.json`; raw arrays and numerical metadata are unchanged. The raw parts do not include original task, log or lock files.

The preparation-time record in [data/R5_RAW_ARCHIVES.json](../../data/R5_RAW_ARCHIVES.json) does not establish later export or upload status. Consult the actual `R5_EXPORT_CATALOG.json` and deposit contents when obtaining the raw inputs.
