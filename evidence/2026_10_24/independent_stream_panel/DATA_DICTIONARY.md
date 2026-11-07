# Array and table dictionary

All NPZ/NPY files can be read with NumPy using `allow_pickle=False`. No object arrays or serialized Python model objects are required.

## Recorded trajectory arrays

`RAW.npz` contains:

- `native_values`: float64, 960 by 52; 41 XMEAS channels followed by 11 XMV channels.
- `controller_input_values`: float64, 960 by 52; controller-input recording paired with the same manipulated-variable sequence.
- `fast_control_input_measurements`: float64, 960 by 41; cached measurements supplied to fast control.
- `native_xmv12`: float64, length 960; the twelfth manipulated variable retained separately.
- `record_time_seconds`: float64, length 960; retained sample times.
- `native_time_seconds`: float64, length 961; includes the initial time record, unlike the retained 960 measurement rows.
- `diagnostic_D_feed_temperature`: float64, length 960.
- `diagnostic_IDV`: int32, 960 by 20; fault indicators.
- `diagnostic_rng_draw_count`: int64, length 960; random draw counter.
- `phase`: int64, 960 by 4; recording phase diagnostics.
- `ctrl11_accumulated_integral`: float64, length 960; controller integral diagnostic.

`CHARTS.npz` contains `charts`, float64, 960 by 22. Column definitions are in `ARM_METADATA.json`. The first 160 records form the prefault interval; indices 160 to 959 (zero-based) form the 40-hour post-onset interval. First affected recorded-sample delay is defined as zero; this is a recorded-sample convention, not continuous-time detection.

`CALIBRATION_MAXIMA.npz` contains `maxima`, float64, 199 by 22. Thresholds are the 190th smallest values per column, without interpolation. Alarm comparisons are strictly greater than the threshold. CUSUM state resets at the beginning of an episode only, not at fault onset or alarm.

## Archived result tables

- `arm_summary.csv`: detector-arm detection, delay, loss and false-alarm summaries.
- `five_state_averages.csv`: averages across fixed model states, not independent additional trajectories.
- `per_run_events.csv`: run-level/arm-level outcomes used for paired summaries.
- `primary_contrasts.csv`: the three prespecified 40-hour penalized-loss comparisons.
- `secondary_contrasts.csv`: descriptive endpoint comparisons, including shorter windows and normal alarms.
- `shutdown_runs.csv`: physical shutdown flags; shutdown cases remain in the panel.

Loss uses a 2,400-minute penalty when a run does not achieve a clean detection. A clean detection requires no prefault alarm and at least one post-onset threshold crossing. Primary interval calculations use 20,000 paired bootstrap draws and an approximate Bonferroni-adjusted 98.3333333333333% percentile interval across three contrasts. These intervals are conditional on the fixed representations and realized calibration.
