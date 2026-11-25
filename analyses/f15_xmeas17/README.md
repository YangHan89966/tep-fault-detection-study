# Fault 15 XMEAS17 substitution

The source archive contains 1,920 saved outcomes from the XMEAS17 substitution comparisons on 120 normal and 120 fault runs. Fixed monitors and their original thresholds are preserved.

Recalculate the eight arm summaries and timing diagnostics with Python's standard library, using a working copy of this directory:

```sh
python code/review_return.py
```

The command regenerates four CSV files in `results/`. The original archive, input identities and reported outcomes remain unchanged. The full raw-record audit in `code/independent_check.py` additionally requires the original tool and source archives described in its command-line help.

In each exchange direction, changing XMEAS17 alone while holding the covariance chart fixed produced alarms in all 120 normal runs and before fault onset in all 120 fault runs. All 120 fault runs in each direction also crossed the limit after onset. The first-alarm criterion excludes runs that alarmed before onset from clean detection.

The fixed-projection diagnostic in `results/X17_FROZEN_RESPONSE.json` reports a median within-run RMS difference of 0.4933 measurement units for XMEAS17 over the 120 normal runs. The corresponding RMS change before squaring the projected score was 20.9803 for the controller-input full-pool monitor and 0.07675 for the post-update monitor. This diagnostic measures sensitivity to an input channel at the original monitor settings.

`ANALYSIS_NOTES_EN.txt` explains the measurement substitution and its interpretation. Historical table filenames are retained; use the paper's Data S1 for final numbering.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
