# Main figures

The six main figures accompany *Measurement–controller pairing and fault detection in the Tennessee Eastman process*. The reference PDFs are included under `figures/article/` in `Supplementary_Information_and_Source_Tables.zip` in the [Zenodo data and code archive](https://doi.org/10.5281/zenodo.24519585). The Python script redraws each figure from the inputs in `data/`. PDF rendering and font metrics can vary with software versions. The plotting script reads local files and does not need network access.

From this directory, run the following in a Python environment with the packages in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python draw_figures.py --output redrawn_figures
```

The script writes six PDFs and a tab-separated file containing the numeric values calculated for the plots. The individual source tables retain their column names and units.

| Figure | Content and input data |
|---|---|
| 1 | Process flowsheet and controller connections, defined in the script. |
| 2 | Recording order and separator level against valve position. `paired_measurements.csv.gz` supplies the paired records. |
| 3 | Family A counts and five-fit means, Family B horizon-specific rates, and five 40 h contrasts with 95% intervals. Inputs are `family_a_120_run_counts.csv`, `family_a_five_fit_rates.csv`, `family_b_detection_rates.csv`, and `representation_and_sfa_contrasts.csv`. |
| 4 | Group and single-measurement exchanges. `summary_by_horizon.csv.gz`, `exchange_main_effects_rank95.csv`, and `screening_simultaneous_clean8h_rank95.csv` supply the estimates and intervals. |
| 5 | Detection after D-feed flow reconstruction. `dfeed_clean_detection.csv` contains the arm-specific counts, estimates, and intervals for each fault and horizon. |
| 6 | Pilot-plant and industrial level loops. `lag_metrics_aggregate.csv` and `primary_comparisons.csv` contain the published error measures. |

The study code is released under the MIT license in `LICENSE`. Study-derived tables and documentation are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source records retain their providers’ terms. The pilot-plant data attribution is in `data/DATA_ATTRIBUTION.txt`. The industrial summary uses the [SACAC level-loop benchmark](https://sacac.org.za/resources/).
