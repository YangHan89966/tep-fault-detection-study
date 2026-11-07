# Detection and false-alarm curves

The supplied calibration and evaluation maxima reproduce the archived operating points and the full detection–false-alarm grids. The figures retain the normal-run false-alarm denominator and the clean-detection rule used in the study.

Run the independent grid comparison with Python, NumPy and pandas:

```sh
python review/crosscheck_all_grids.py .
```

This check compares 7,314 grid rows, including denominators and replicate counts. To regenerate curves and figures, run these commands in a working copy:

```sh
python core/curves.py --root .
python figures/plot_detection_far.py results/curves_mean.csv figures
```

Curve calculations require NumPy; plotting additionally requires Matplotlib. `review/independent_audit.py` repeats the independent endpoint and bootstrap checks. `analysis_notes/` explains the thresholds and interval interpretation. Historical table filenames are retained; use the paper's Data S1 for final numbering.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
