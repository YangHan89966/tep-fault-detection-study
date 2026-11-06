# Tennessee Eastman fault-detection study

Code, saved results and figure data for **Effects of monitoring statistic, projection direction and measurement–controller pairing on the detectability of Tennessee Eastman Faults 3, 9 and 15**.

This is a new, private research companion based on the final V10 scientific materials. Its Git history starts here; it does not fork or mirror the working repository. Old manuscripts, submission correspondence, internal discussions and second-paper materials are not part of this distribution.

## Start here

Python 3.12 is recommended. Source programs and small tables are directly readable. Larger scientific inputs and exact document/figure bytes are stored in checked binary bundles, to keep them out of repeated Git text diffs. Materialize those included files before running analyses:

```sh
python unpack.py
python verify_repository.py
python verify_paper.py
python verify_confirmation.py
python -m pip install -r requirements.txt
python verify_fault09_extensions.py --replay
```

unpack.py uses only the standard library, verifies every bundle and extracted file, and refuses to overwrite a different existing file. It does not download external raw data or execute bundled code. See [REPRODUCTION.md](REPRODUCTION.md) for saved-result replays and plotting commands.

## Included

- **paper/**: exact V10 manuscript, supplementary information, Highlights, seven TIFFs, Data S1 and 83 readable table CSVs. All 68 SI tables correspond to 6,110 normalized Data S1 cells.
- **data/source_tables/**: ten exact numerical source CSVs and their source mapping.
- **evidence/2026_10_24/**: 1,199 recorded-run identities, parameters and 13,200 saved confirmation event rows with a standard-library recount.
- **evidence/2026_11_05/**: recording-specific Fault-9 relearning and a separate Rust fixed-monitor comparison, with frozen code, models, saved endpoints, protocols and replay programs.
- **analyses/fault09_coupling/**: recording-by-B and graded-noise-coupling studies, including frozen models, scores, noise records, independent audits and component notices.
- **analyses/detection_far/**: calibration/evaluation maxima, curve calculation, paired comparisons, independent checks and supplementary plotting code.
- **analyses/f15_xmeas17/**: returned case outcomes and the saved-outcome table/timing replay.
- **historical/**: selected foundational method and analysis code, fitted states, saved results, parameters and component-specific provenance.
- **licenses/** and distributed LICENSE/NOTICE files: original and third-party terms, within their documented scope.

The inventory distinguishes exact source files from added documentation and verification tooling. The source working repository remains private and unchanged. No old Git objects, branches or PR conversations are imported.

## What the checks establish

Integrity checks test the distributed bytes. Saved-result checks replay specified calculations from supplied models, scores, maxima or endpoints. These are separate from complete raw-trajectory reproduction.

The supplied raw receipts are not raw trajectories. Complete generation logs, all historical/new raw matrices, and all source-to-executable bindings have not been independently verified here. The asserted Zenodo record is not treated as an accessible or complete deposit. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the specific gaps and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for rights.

The final V10 source documents still contain their original repository/deposit statements and author metadata. Their presence is not an endorsement of unresolved availability or author-approval claims. They are preserved unchanged; a separately approved manuscript revision must align them with the final release.

**Private preparation, not public publication or a journal submission.** Public visibility, a Zenodo deposit, a finalized citation and a blanket license are not established by this repository.
