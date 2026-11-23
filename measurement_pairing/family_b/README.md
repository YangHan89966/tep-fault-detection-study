# Family B recording comparisons

The few-label autoregressive density monitors used 57 normal and 16 fault training runs per fault and replicate, with 99 separate normal runs for calibration. The 36 fits share the evaluation panels of 120 fault runs per fault. Rank 95 is the 95th of the 99 ordered normal calibration maxima. Clean detection requires a first alarm after onset and within the stated horizon.

| Directory | Material |
| --- | --- |
| [statistics/E16/](statistics/E16/) | Recording exchanges, deletions, two-by-two contrasts, first-alarm arrays and summaries |
| [statistics/E16_S/](statistics/E16_S/) | Bidirectional single-measurement screen for Faults 3 and 9, including simultaneous intervals |
| [statistics/E17/](statistics/E17/) | Extension of the same trajectories to 40 h after onset, group manifest, rank-95 first alarms and results by horizon |
| [protocols/](protocols/) | English summaries identifying original protocols by filename, SHA-256 and freeze time |

In the stored arrays, `aligned` and `native` denote controller-input and post-update records. `Fixed246` denotes the three-family template. `RSTAR` denotes out-of-fold selection. The time-extension predictions have IDs P1 through P4. In the measurement screen, `C00` refers to XMEAS1, `C01` to XMEAS2, and so on. The protocols give the two exchange directions and the backgrounds used.

First-alarm files are NumPy NPZ archives. For the time extension, fault onset is at zero-based index 160. Clean detection within 8 h uses indices 160–319 and within 40 h uses 160–959. Each compressed CSV or JSON file expands to the original table or manifest.

## Recompute Figure 4

From the repository root, run the following command after installing the requirements.

```sh
python measurement_pairing/scripts/screen_simultaneous.py
```

This reads the rank-95 first alarms and archived pointwise estimates. It writes the 76 contrasts and simultaneous intervals to `replay_output/screening_simultaneous_clean8h_rank95.csv` using 20,000 crossed-bootstrap draws. Compare with the [reference CSV](statistics/E16_S/screening_simultaneous_clean8h_rank95.csv). The [time-extension manifest](statistics/E17/GROUP_MANIFEST.json.gz) records the groups used in the extended evaluation.
