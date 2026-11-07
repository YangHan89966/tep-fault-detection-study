# Independent-stream confirmation panel: data and analysis

This dataset supports a targeted follow-up for Tennessee Eastman Faults 3, 9 and 15. It is one component of the accompanying study, not a complete archive of all historical experiments in the manuscript.

## Contents and sample size

The fixed panel comprises 1,199 physical trajectories: 199 normal calibration trajectories, 400 normal evaluation trajectories, and 200 evaluation trajectories for each of Faults 3, 9 and 15. Each trajectory uses a fixed 48-hour recording horizon with 960 retained samples at three-minute intervals; faults begin at hour 8. Physical shutdowns, if present, remain flagged and follow the protocol-defined horizon-continuation rule rather than being excluded or replaced. Native and controller-input recordings are paired views of the same physical trajectory. The five fixed model states and 22 detector arms do not multiply the physical sample size.

- `MANIFEST.csv`: fixed run IDs, roles, fault IDs, ordinals and Philox keys.
- `runs/<run_id>/RAW.npz`: paired recordings and diagnostics.
- `runs/<run_id>/CHARTS.npz`: saved 960 by 22 CUSUM arrays.
- `runs/<run_id>/RUN_INFO.json`: scientific execution metadata and actual timestamps.
- `assets/`: fixed PCA, residual basis, model representations, choices and score parameters.
- `calibration/`: 199 by 22 maxima and the threshold vector, using the 190th smallest maximum in each arm.
- `results/`: six archived numerical result tables and a result summary.
- `code/`: scoring, random-source implementation and portable analysis replay.
- `engine/`: pinned simulator source and author recording/horizon components.
- `PROTOCOL.json` and `ARM_METADATA.json`: scientific design and detector definitions.
- `RUN_PROVENANCE.csv` and `HISTORICAL_PROVENANCE.json`: recovery membership, original identity anchors and release transformations.
- `SCIENTIFIC_FILES.csv`: hashes of byte-preserved numerical files.
- `FILE_MANIFEST.csv`: checksums of the current reader-facing release (excluding this manifest itself).

The ZIP files are independently readable archives, not byte-split volumes. Extract all of them into the same empty parent directory. Their shared top-level directory is `independent_stream_panel`; no file is intentionally repeated between archives. All archives are needed for complete replay. No network access, additional simulation or retraining is required to analyze the saved panel.

## Replay

Use Python 3.12 with the versions in `requirements.txt` (NumPy 2.3.5 and SciPy 1.17.0). From the shared parent directory:

    python independent_stream_panel/code/analyze_confirmation.py --root independent_stream_panel --out replay_results

To recompute every saved chart directly from the stored raw recordings and fixed scoring parameters, run:

    python independent_stream_panel/code/verify_raw_charts.py --root independent_stream_panel --out raw_chart_verification.json

This second check also verifies all release hashes, recalculates all 1,199 chart matrices, and requires exact elementwise equality; it performs no simulation or refitting. The JSON output must be outside the dataset.

Choose a new output directory outside the dataset. The analyzer verifies the release checksums and scientific identities, reconstructs calibration thresholds, and recomputes endpoints and paired bootstrap intervals from the stored charts. It refuses incomplete or changed inputs and does not overwrite archived results. Compare the six generated CSV files against `results/` using SHA256. The release verification and replay tests are summarized in `REPLAY_VALIDATION.json`.

The included random-source and simulator files document the generation implementation. A portable workflow for generating fresh trajectories is not included or validated for this archive. Analysis replay is distinct from fresh simulation and from reproducing the training history of the fixed models.

## Provenance and recovery

The scientific design, representations, run identities and random keys were fixed before this panel was generated. This follow-up was not externally preregistered. Of the 1,199 trajectories, 1,039 original saved records were recovered; 160 missing records were reconstructed using their original run IDs, keys and fixed protocol. Those 160 are not new independent samples. Their membership and actual timestamps are retained in `RUN_PROVENANCE.csv`. The six restored result CSV files matched the historical SHA256 anchors retained from the original analysis; this does not imply that reconstructed metadata or the repackaged archive has the same whole-file hash as an earlier archive.

This release retains all scientific NPZ/NPY arrays and the six result CSV files byte-for-byte. Machine-specific directory and host identifiers were removed from run metadata. Reader documentation and a checksum manifest accompany the scientific data; non-scientific administrative records are omitted. Original protocol and threshold hashes in run metadata are historical anchors, not hashes of the edited reader-facing JSON files. Portable verification explicitly distinguishes these identities.

## Scope and interpretation

Results condition on the frozen representations and the realized normal calibration set. Physical trajectories, not model-state cells or recording views, are the resampling units. The new common 190/199 threshold design does not retroactively replace earlier calibration protocols. Nominal threshold matching does not establish equal realized false-alarm rates; use the per-arm normal evaluation results.

The panel does not establish independent relearning success for Fault 9, rescue an earlier failed two-hour prediction, or establish universal behavior for all implementations of the Tennessee Eastman process. Earlier manuscript datasets, including 1,335 historical raw records whose original arrays remain outside this package, must not be inferred to be present. The separate historical numerical-evidence ZIP supplies saved models, scores and tables, not these missing process arrays. The manuscript-wide coverage inventory identifies that companion component and the remaining material outside this archive.

## Software attribution and licensing

The simulator is `tep` 2.5.0 at commit `9a6c8e5fcef4a2850778704e7793c87b0a187005`. Upstream copyright, license, citation and development-attribution notices are preserved in `engine/vendor/LICENSE` and `engine/vendor/README.md`. They apply to those third-party files and must not be removed. These notices apply to the third-party files. No additional redistribution license for author-created code or data is granted by this archive.

AI tools assisted with code, analysis protocols, figures and text preparation. Automated verification covers only the computational checks explicitly described in this archive; it does not independently validate every scientific conclusion.
