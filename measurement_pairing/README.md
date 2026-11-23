# Reproducing the measurement–controller pairing study

| Result | Source |
| --- | --- |
| Figure 1 | Process and loop diagram in [figures/main.py](figures/main.py) |
| Figure 2 | [Paired measurements](figures/inputs/paired_measurements.csv.gz), eight normal trajectories, records 161–960 in the scatter plot |
| Figure 3 | Family A five-fit means and clean-detection counts from historical [Table S51](../paper/Data_S1/Table_S51.csv) and [Table S55](../paper/Data_S1/Table_S55.csv). Family B uses the [horizon summary](family_b/statistics/E17/summary_by_horizon.csv.gz), rank 95, fixed template |
| Figure 4 | The [76 bidirectional measurement contrasts](family_b/statistics/E16_S/screening_simultaneous_clean8h_rank95.csv) and their [bootstrap replay](scripts/screen_simultaneous.py) |
| Figure 5 | Historical [Table S61](../paper/Data_S1/Table_S61.csv), calibration limits and detection under partial sharing of measurement noise |
| Figure 6 | [Pilot](plant_records/2024_Pilot/published/lag_metrics_aggregate.csv) and [industrial](plant_records/2003_Industrial/published/primary_comparisons.csv) comparison tables |
| Submission and extended SI tables | [SI table crosswalk](tables/SI_table_crosswalk.csv), using repository-relative paths |
| Submission SI Figures S6 and S7 (extended S24 and S25) | [Plant-record guide](plant_records/README.md) |

Install the dependencies from the [repository requirements](../requirements.txt). Run the plotting commands from the repository root.

```sh
python measurement_pairing/figures/main.py
python measurement_pairing/figures/extended_si.py
```

The first command creates Figures 1–6 and `check_values.tsv` under `replay_output/main/`. Compare the plotted values with [expected_main_values.tsv](figures/inputs/expected_main_values.tsv). Rendering uses Matplotlib and a serif font and can vary between systems.

The second command reads the [supplementary plot sources](figures/inputs/supplementary_sources.zip) directly. It reproduces extended SI Figures S2, S3, S5, S6, S10, S12–S14 and S16–S21 under `replay_output/extended_si/`. Both commands accept `--output DIRECTORY`.

## Reading the table crosswalk

The crosswalk contains extended SI Tables S1–S95, the 34 tables retained and renumbered in the submission SI, and the 15 additional Data S1 tables D1–D15. A blank submission number indicates a table outside the submission SI. Each row gives a CSV for every applicable edition. Archived CSV filenames keep their original numbers.

The extended SI updates textual cross-references in historical S14, S57 and S67 (extended S12, S67 and S94). The numerical entries agree. The two editions of S1 have separate [submission](tables/submission/Table_S01.csv) and [extended](tables/extended/Table_S01.csv) timelines. The additional [extended Table S2](tables/extended/Table_S02.csv) gives the learned-direction comparisons. Family B table exports follow the wording of the SI; the underlying first-alarm arrays, estimates and protocol summaries are in the [Family B directory](family_b/README.md).
