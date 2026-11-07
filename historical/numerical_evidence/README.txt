Historical numerical evidence for Tennessee Eastman fault monitoring
Archive date: 26 October 2026

Scope
This component contains saved numerical evidence from four historical studies: Fault 9 validation, Fault 9 development, subsequent Fault 9 relearning, and Fault 3/Fault 15 studies with sensitivity and correlation analyses. It is separate from the later 1,199-run independent-stream experiment.

Contents
- 512 NPZ files containing numerical model, calibration, monitoring-score, support and event arrays.
- 42 CSV files containing result tables, sampling identities and input provenance.
- FILE_MANIFEST.json lists each file, its current hash, its original numerical-file hash and its original source-archive hash.
- DATA_DICTIONARY.json lists NPZ keys, array shapes and dtypes, and CSV columns and row counts. Original field names and scientific identifiers are retained.
- TRANSFORMATIONS.json identifies path-redacted CSV columns and exact duplicate nested copies omitted from this component.
- historical_coverage contains the historical component inventory and expected original raw-data identities.

Preservation and transformations
All 512 NPZ files and 37 CSV files are byte-identical to their numerical source files. Five CSV files contain 2,984 original-host path cells replaced by opaque external-input identifiers. Every other parsed cell, including values, hashes, seeds and study roles, was compared and is unchanged. These identifiers are reference labels, not downloadable files or executable paths. They do not assert that the referenced inputs are included.

Twenty nested numerical copies were omitted only where the same normalized logical path in the same source component had identical bytes. Different study roles and different logical paths were retained even when file contents matched. No numerical sample or study role was removed by value-based deduplication.

Reproduction limits
This is a numerical-evidence collection, not a standalone executable replay package. Historical analysis wrappers, dependency environments and operational instructions are not included in this component. No analysis code was executed and no simulation, refitting or new scientific evaluation was performed for this archive. Inspect arrays with NumPy using allow_pickle=False; the inspected arrays do not require object deserialization.

The four sources contain no RECORDING.npz files for the 1,335 catalogued original historical trajectories. The expected identity catalog is not raw data, and saved scores or fitted models do not substitute for the missing process arrays. Complete raw-to-score reproduction and refitting of these historical studies are not established. The newer independent-stream dataset is not a replacement for those original inputs. Other storage locations have not been assessed here, so no permanent-loss claim is made.

This component does not contain all historical material cited in the first paper. The accompanying coverage inventory explicitly identifies other packages and unresolved dependencies. Its source hashes establish the inspected archive identities, not that the original complete archives have been republished here.

Rights
No redistribution license for original material is granted by this archive. Any applicable third-party rights remain in effect. No third-party software code is included in this numerical component.
