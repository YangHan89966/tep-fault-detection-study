# Recompute the monitoring results

Download `Measurement_Pairing_Additional_Experiments.zip` from the [study data and code archive, version 1.1.0](https://doi.org/10.5281/zenodo.24519585). Run the following commands from the repository root. Both scripts read saved first alarms directly from the ZIP. The matched false-alarm comparisons also use saved trajectory maxima.

```sh
python -m pip install -r requirements.txt
python article/analyses/monitor_comparisons.py Measurement_Pairing_Additional_Experiments.zip --output results/monitor_rows.csv
python article/analyses/dfeed_detection.py Measurement_Pairing_Additional_Experiments.zip --output results
```

| Output | Calculation |
| --- | --- |
| `monitor_rows.csv` | For 16 monitor and panel combinations, the proportions detected within 8 and 40 h and the proportions of normal trajectories with an alarm in 48 h. These include the normal-trained slow-feature analysis (SFA) and the 500-tree window-statistics XGBoost classifier with a CUSUM reference value of 0.5. The relevant main-text comparisons appear in Tables 2 and 3. In the archived output, D1 denotes the five-fit Fault 9 panel and D2 the independent confirmation panel. |
| `dfeed_rates.csv` | Recounts the clean-detection rates and intervals in Figure 5 and the corresponding normal false-alarm rates in Table S42 for Faults 3 and 9. PU is the post-update record, RC replaces its D-feed flow measurement with the reconstructed value, and CI is the controller-input record. |
| `dfeed_recovery.csv` | Fraction of the CI–PU detection difference recovered by reconstruction at 2, 8 and 40 h, with paired 95% intervals. |

The D-feed analysis pairs the records of each trajectory. Each fault has 36 fits scored on the same 120 fault and 120 normal trajectories. The intervals resample the fit and trajectory dimensions together in 20,000 bootstrap draws. Fault onset is zero-based row 160. Only a first alarm from onset through the end of the specified window counts as a clean detection.

## Model fitting and source records

The same ZIP contains the study's fitting code and the full analysis records.

| Analysis | Path inside the ZIP |
| --- | --- |
| SFA training and scoring | `monitor_comparisons/representation/sfa/model.py` and `monitor_comparisons/representation/sfa/run.py` |
| Window-statistics classifier | `monitor_comparisons/representation/xgb/window_xgb.py`. The fitted models and per-run results are under the adjacent `results/` directory. |
| D-feed flow reconstruction | `dfeed_reconstruction_monitoring/code/reference/dfeed.py` contains the observer. `dfeed_reconstruction_monitoring/code/fit_monitor.py` fits, calibrates and evaluates monitors after substituting the reconstructed flow. |
| Data for refitting | `monitor_comparisons/representation/README.md` identifies the source-trajectory downloads for the comparison monitors. For D-feed, `dfeed_reconstruction_monitoring/README.md` explains how to obtain the original trajectories from the dataset creators. `dfeed_reconstruction_monitoring/code/reference/input_archives_sha256.csv` identifies the 24 required archives. |
