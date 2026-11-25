# Detection and false-alarm curves

The supplied calibration and evaluation maxima reproduce the archived operating points and the full detection–false-alarm grids. The normal false-alarm rate counts runs with any alarm within 48 h. Clean detection requires no alarm in records 1–160 and a crossing within the specified fault window (records 161–321 for 8 h and 161–960 for 40 h). Each new threshold is applied to the saved maxima from the same runs.

Run the independent grid comparison with Python, NumPy and pandas:

```sh
python review/crosscheck_all_grids.py .
```

This check compares 7,314 grid rows, including denominators and replicate counts. To regenerate curves and figures, run these commands in a working copy:

```sh
python core/curves.py --root .
python figures/plot_detection_far.py results/curves_mean.csv figures
```

The independently calibrated curves and the curves matched on the normal evaluation sample use different thresholds. At 20 alarms among 400 normal confirmation runs, the 40 h detection contrasts were 100.0% versus 26.0% for Fault 3, 68.4% versus 15.4% for Fault 15, and 86.5% versus 19.0% for Fault 9. Faults 3 and 15 average five models on the same 200 fault runs. Fault 9 uses one fixed monitor. The operating points and pointwise intervals are reported in [the analysis notes](analysis_notes/CURVES_METHODS_RESULTS_EN.txt) and in `results/matched_anchors.csv` and `results/paired_contrasts.csv`.

Curve calculations require NumPy; plotting additionally requires Matplotlib. `review/independent_audit.py` repeats the independent endpoint and bootstrap checks. `analysis_notes/` explains the thresholds and interval interpretation. Historical table filenames are retained; use the paper's Data S1 for final numbering.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
