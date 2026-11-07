# Tennessee Eastman fault-detection study

Code, fitted models, saved results and figure data for **Effects of monitoring statistic, projection direction and measurement–controller pairing on the detectability of Tennessee Eastman Faults 3, 9 and 15**.

## Start here

Download or clone this repository, then use Python 3.12:

```sh
python unpack.py
python verify_repository.py
python verify_paper.py
python verify_confirmation.py
python -m pip install -r requirements.txt
python verify_fault09_extensions.py --replay
```

`unpack.py` restores the larger files from the included bundles and checks their SHA-256 hashes. It uses the Python standard library. It preserves existing matching files and stops if an existing file differs.

## Contents

| Directory | Contents |
| --- | --- |
| [paper/](paper/) | V10 manuscript, supplementary information, Highlights, seven figures and Data S1 with 83 readable CSV tables |
| [data/](data/) | Ten numerical source tables and their mapping to the manuscript |
| [evidence/2026_10_24/](evidence/2026_10_24/) | Independent-stream confirmation panel: code, models, calibration, 1,199 run identities and 13,200 saved evaluation event rows |
| [analyses/](analyses/) | Recording-specific Fault 9 relearning, Rust fixed-monitor comparison, noise coupling, recording variance, detection–false-alarm curves and Fault 15 XMEAS17 analysis |
| [historical/](historical/) | Historical PCA/residual monitoring code, fitted states, saved numerical evidence and selected direction experiments |
| [supplements/](supplements/) | Source inputs and replay programs for temporal-covariance estimation (S52), self-overlap sensitivity (S57 and D14), and sample-level alarm occupancy (S65) |

See [REPRODUCTION.md](REPRODUCTION.md) for the checked calculations and their commands, [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the supplied inputs, and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for component licenses.

The main numerical checks cover 68 supplementary tables and 6,110 normalized table cells, the confirmation-panel event counts, the Fault 9 saved-endpoint comparisons, the saved-score coupling analyses, and the three supplementary source analyses. These checks use the distributed scientific files; complete process-trajectory regeneration requires the additional raw-data archives identified in the component instructions.

This repository has an independent Git history. The research development repositories remain separate. The manuscript files retain their source bytes; their repository links refer to the version in which they were written.
