# Measurement and output timing in two plant records

Both analyses use normal-operation records. The pilot analysis compares a stored output with a documented controller relation across row offsets. The industrial analysis fits a PI-type relation on the first half of a level trace and compares alternative measurement offsets on held-out rows.

| Case | Source and attribution | Published comparison |
| --- | --- | --- |
| Multiphase pilot | Giovanni Mazzuto (2025), *A dataset of tests in a multiphase experimental plant simulating oil and gas transportation system*, v2, CC BY 4.0. Data collectors Marco Ortenzi, Vincenzo Foti and Filippo Emanuele Ciarapica. Data curators Ilaria Pietrangeli and Maurizio Bevilacqua. [Dataset](https://doi.org/10.5281/zenodo.16978349) · [Data article](https://doi.org/10.1016/j.dib.2025.112038) | [Lag metrics](2024_Pilot/published/lag_metrics_aggregate.csv) |
| Industrial level loop | SACAC control-loop repository, record `other-L-paper-horch-2003`, anonymous pulp-and-paper plant. [Repository](https://sacac.org.za/resources/) · [Bauer et al. (2019)](https://doi.org/10.1021/acs.iecr.8b06354) | [Held-out error by offset](2003_Industrial/published/primary_comparisons.csv) |

Run these commands from the repository root. The download scripts verify the original files against their SHA-256 hashes. The analysis scripts write derived tables into the two case-specific `results/` directories.

```sh
python measurement_pairing/plant_records/2024_Pilot/fetch_source.py
python measurement_pairing/plant_records/2003_Industrial/fetch_source.py
python measurement_pairing/plant_records/2024_Pilot/analysis.py
python measurement_pairing/plant_records/2003_Industrial/analysis.py
python measurement_pairing/figures/plant_si.py
```

For inputs already downloaded, use `--data FILE` on either analysis, and `--output DIRECTORY` to choose a results directory. The industrial downloader also accepts `--archive FILE` with the complete official SACAC ZIP. The plotting command accepts `--results-root DIRECTORY` and `--industrial-csv FILE` and produces submission Figures S6 and S7 in `replay_output/plant_si/`.

The pilot includes 38 distinct active records and 91,941 common rows. In 67,513 unsaturated rows, the preceding-row level reconstructs the stored output with RMSE 0.5763 µA, compared with 28.137 µA for the same row. The industrial source contains 904 samples, including 445 held-out targets. The stored pairing gives a held-out RMSE of 0.00429 percentage points. These results concern timing relationships within the published fields.
