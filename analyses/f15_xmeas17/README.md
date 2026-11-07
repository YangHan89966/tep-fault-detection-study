# Fault 15 XMEAS17 substitution

The source archive contains 1,920 saved outcomes from the XMEAS17 substitution comparisons on 120 normal and 120 fault runs. Fixed monitors and their original thresholds are preserved.

Recalculate the eight arm summaries and timing diagnostics with Python's standard library, using a working copy of this directory:

```sh
python code/review_return.py
```

The command regenerates four CSV files in `results/`. The original archive, input identities and reported outcomes remain unchanged. The full raw-record audit in `code/independent_check.py` additionally requires the original tool and source archives described in its command-line help.

`ANALYSIS_NOTES_EN.txt` explains the measurement substitution and its interpretation. Historical table filenames are retained; use the paper's Data S1 for final numbering.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
