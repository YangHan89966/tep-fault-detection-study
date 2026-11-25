# Reproducing the article results

## Figures and tables

From the repository root, install the plotting dependencies and generate Figures 1–6.

```sh
python -m pip install -r article/figures/requirements.txt
python article/figures/draw_figures.py --output redrawn_figures
```

The [figure guide](article/figures/README.md) identifies the inputs for each panel. Figure 3(c) shows the effect of changing the monitored representation and the results for slow feature analysis. Figure 5 compares the post-update record, the same record with D-feed flow reconstructed, and the controller-input record. Reference PDFs are under `figures/article/` in the supplementary [Zenodo archive](https://doi.org/10.5281/zenodo.24519585).

The [table index](article/table_index.csv) gives captions, notes and filenames for Tables 1–4 and S1–S42. Its paths are relative to the `article/` directory. The extended supplement uses a separate numbering system.

## Alarm rates and model fitting

The [analysis guide](article/analyses/README.md) gives commands for recalculating monitor results from the archived run-level records. The full analysis sources below are in `Measurement_Pairing_Additional_Experiments.zip` in [Zenodo v1.1.0](https://doi.org/10.5281/zenodo.24519585).

| Analysis | Path in the ZIP |
| --- | --- |
| Normal-trained slow feature analysis | `monitor_comparisons/representation/sfa/run.py` and `model.py` in the same directory |
| Window-statistics XGBoost classifier | `monitor_comparisons/representation/xgb/window_xgb.py` |
| Endpoints for the comparisons in Table S36 | `monitor_comparisons/representation/report/e2_endpoints.csv` |
| Detection after reconstructing D-feed flow | `dfeed_reconstruction_monitoring/README.md`, with refitting code in `code/fit_monitor.py` and rates in `results/rates.csv` under that directory |

## Earlier studies

The [earlier repository snapshot](https://github.com/YangHan89966/tep-fault-detection-study/tree/study-archive-2026-11-25) contains the original simulator and supplementary analyses. Follow the instructions and table numbering in that snapshot when reproducing its 7 November materials or its 23 November figures. The latter include a noise-sharing analysis as Figure 5.
