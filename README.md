# Measurement–controller pairing and fault detection in the Tennessee Eastman process

This branch contains the November 2026 simulation analyses and the figure sources from 23 November. The [article repository](https://github.com/YangHan89966/tep-fault-detection-study/blob/main/README.md) and [Zenodo v1.1.0](https://doi.org/10.5281/zenodo.24519585) provide the article figures and tables.

A controller output can be stored with the measurement that the controller used or with a later measurement. We recorded the same Tennessee Eastman trajectories both ways and fitted and calibrated each monitor on its own recording. Across five fits on independent random streams, the learned-direction monitor detected 98.2% of 200 Fault 9 runs within 40 h with the controller's measurement and 14.6% with the measurement taken 3 s later. At 8 h the rates were 43.2% and 1.3%. The percentages of 400 normal runs with an alarm in 48 h were 3.65% and 6.25%.

## Find a result

| Material | Where to start |
| --- | --- |
| The six figures and numerical inputs from 23 November | [Figure and analysis guide](measurement_pairing/README.md) |
| 23 November supplementary table numbering and the original Data S1 | [Table crosswalk](measurement_pairing/tables/SI_table_crosswalk.csv) |
| Few-label autoregressive monitors (Family B), simulation protocols and rank-95 first alarms | [Family B data and screen replay](measurement_pairing/family_b/README.md) |
| Pairing in the multiphase pilot plant and the industrial level loop | [Source records and analysis commands](measurement_pairing/plant_records/README.md) |
| Learned-direction CUSUMs (Family A), independent-stream confirmation and TEP-Rust | [Reproduction guide](REPRODUCTION.md) and [data availability](DATA_AVAILABILITY.md) |
| 83 original CSV tables, 7 November 2026 | [Data S1](paper/README.md) |

## Reproduce the 23 November figures

Run these commands from the repository root with Python 3.12.

```sh
python -m pip install -r requirements.txt
python measurement_pairing/figures/main.py
```

The command writes six figures in PDF, PNG and TIFF and a table of plotted values to `replay_output/main/`. The original values for comparison are in [expected_main_values.tsv](measurement_pairing/figures/inputs/expected_main_values.tsv).

## Reproduce the earlier experiments

```sh
python unpack.py
python verify_repository.py
python verify_paper.py
python verify_confirmation.py
python verify_fault09_extensions.py --replay
```

The first four commands use the Python standard library. They restore and verify the bundled files, compare Data S1 with the original table archive and its checksums, and recount 13,200 events from the 1,199-run confirmation panel. The last command also replays the independent Fault 9 relearning and the 194-run TEP-Rust comparison. The full list of checks is in [REPRODUCTION.md](REPRODUCTION.md).

## Data citation and licenses

Han, Y. and Zhang, X. (2026). *Data and code for “Measurement–controller pairing and fault detection in the Tennessee Eastman process”*. Zenodo, version 1.1.0. [10.5281/zenodo.24519585](https://doi.org/10.5281/zenodo.24519585).

For the simulation results, plant records and figure sources in this repository, see the [data availability guide](DATA_AVAILABILITY.md).

The original Family A materials are also available in the [7 November 2026 release, version 1.0.0](https://doi.org/10.5281/zenodo.22084007). The historical TEP data are from Rieth et al. (2017). [Data set](https://doi.org/10.7910/DVN/6C3JR1).

Study code uses the MIT License. The archived study data, tables and documentation use CC BY 4.0. Third-party records and software retain their original terms. [License scope](LICENSE_SCOPE.md).
