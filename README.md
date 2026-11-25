# Measurement–controller pairing and fault detection in the Tennessee Eastman process

This study compares two records of the same process trajectory. Each valve output is paired either with the measurement used by its controller or with a measurement taken 3 s later. For Fault 9, a CUSUM fitted to directions in the PCA residual subspace detected 98.2% and 14.6% of runs within 40 h under the two pairings, respectively.

The [article data and code on Zenodo, version 1.1.0](https://doi.org/10.5281/zenodo.24519585) include the supplementary information, analysis scripts and run-level results.

| Material | Where to find it |
| --- | --- |
| Figures 1–6 | [Plotting script, inputs and instructions](article/figures/README.md) |
| Figure 3(c), representation and slow feature analysis | [Five 40 h contrasts and their intervals](article/figures/data/representation_and_sfa_contrasts.csv) |
| Figure 5, detection after D-feed flow reconstruction | [Plotted values](article/figures/data/dfeed_clean_detection.csv) and [Table S42](article/tables/Table_S42.csv) |
| Slow feature analysis and the XGBoost classifier | [Table S36](article/tables/Table_S36.csv) and [analysis code](article/analyses/README.md) |
| Article Tables 1–4 and supplementary Tables S1–S42 | [Table captions, notes and CSV filenames](article/table_index.csv) |
| Fitted monitors, alarm records and simulation trajectories | [Data availability](DATA_AVAILABILITY.md) |

To redraw all six figures from the repository root, run

```sh
python -m pip install -r article/figures/requirements.txt
python article/figures/draw_figures.py --output redrawn_figures
```

The command writes six PDFs and a table of plotted values. [Reproduction instructions](REPRODUCTION.md) cover the monitor comparisons and D-feed reconstruction.

The [7 November 2026 Family A data release](https://doi.org/10.5281/zenodo.22084007) provides the original simulation trajectories, fitted monitors and alarm records.
