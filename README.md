# Tennessee Eastman fault-detection study

Code, frozen models and saved results for our manuscript *Effects of monitoring statistic, projection direction and measurement–controller pairing on the detectability of Tennessee Eastman Faults 3, 9 and 15* (Han et al., submitted). The raw trajectories and the full archive are on Zenodo: [10.5281/zenodo.22084007](https://doi.org/10.5281/zenodo.22084007).

## What the study is about

Faults 3, 9 and 15 of the Tennessee Eastman process are hard to detect, and many monitoring studies leave them out. We built CUSUM charts on PCA residuals and changed one thing at a time, first the monitoring statistic and then the projection direction, to find out why.

For Fault 9, the main obstacle turned out to be how the simulator records its data rather than the monitor. The reference Fortran driver stores each valve output next to the measurement the controller actually used; the tep package stores it next to a measurement taken 3 s later, with fresh noise. With the controller's measurements, relearned monitors detected 98% of Fault 9 runs within 40 h; with the later measurements, 15%.

## Quick start

You need Python 3.12. From the repository root:

```sh
python unpack.py                # restore the larger files from bundles/ and check their SHA-256
python verify_repository.py     # check every file byte for byte
python verify_paper.py          # compare the 68 SI tables with Data S1 (6,110 cells)
python verify_confirmation.py   # recount the confirmation panel: 1,199 runs, 13,200 event rows
python -m pip install -r requirements.txt
python verify_fault09_extensions.py --replay   # Fault 9 relearning (14 contrasts) and the 194-run TEP-Rust comparison
```

The first four scripts need only the standard library. The replay needs NumPy, and the historical calculations also use SciPy and pandas. `unpack.py` leaves matching files alone and stops if an existing file differs from the expected one.

## What is where

| Folder | Contents |
| --- | --- |
| [paper/](paper/) | The submitted manuscript, Supplementary Information, Highlights, seven figures and Data S1 (83 CSV tables). Run `unpack.py` first to restore the Word and TIFF files. |
| [data/](data/) | Ten numerical source tables and how they map to the manuscript |
| [evidence/2026_10_24/](evidence/2026_10_24/) | Confirmation panel on independent random-number streams: code, models, calibration, the 1,199 run identities and 13,200 saved evaluation events |
| [analyses/](analyses/) | Fault 9 relearning on independent streams, the TEP-Rust comparison, partially shared measurement noise and the recording × B analysis, recording-variance decomposition, detection–false-alarm curves and the Fault 15 XMEAS17 exchange |
| [historical/](historical/) | Code, fitted states and saved numerical evidence from the earlier studies, and selected direction experiments |
| [supplements/](supplements/) | Inputs and replay scripts behind Tables S52 (three estimators of the long-run covariance B), S57 and D14 (runs with overlapping random-number streams excluded) and S65 (sample-level alarm rates) |

## Going further

[REPRODUCTION.md](REPRODUCTION.md) lists every check with its command and what it covers, including the detection–false-alarm grids, the noise-sharing replay and the S52, S57 and S65 replays. All of these recalculate results from saved inputs: frozen models, scores and alarm records. Regenerating the trajectories themselves, or rebuilding scores from the raw measurements, needs the raw-data archives on Zenodo; [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) says which ones.

The historical analyses use the extended data set of Rieth et al. (2017), [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1).

## Licence

Code written for this study is released under the MIT License, and data, tables and documentation under CC BY 4.0. Third-party code, including the tep package source and the TEP-Rust module, keeps its own licence and notices. The manuscript, Highlights and figure files in `paper/` are not covered by these licences. Details are in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).

## How to cite

If you use this code or data, please cite the archived record:

Han, Y., Zhang, X., 2026. Data and code for: Effects of monitoring statistic, projection direction and measurement–controller pairing on the detectability of Tennessee Eastman Faults 3, 9 and 15 [dataset]. Zenodo, version 1.0.0. https://doi.org/10.5281/zenodo.22084007

We will add the article reference once it is published.
