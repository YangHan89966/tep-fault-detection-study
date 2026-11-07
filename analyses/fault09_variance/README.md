# Fault 9 recording variance decomposition

The 400 saved normal-run checkpoints contain the sufficient statistics used to decompose the score-variance difference between recording modes. Models, recording pairs and raw trajectories are identified in the protocol and source records.

Recompute the tables and 20,000-draw whole-run bootstrap with Python and NumPy from a working copy of this directory:

```sh
python code/summarize_decomposition.py --root .
python code/export_group_covariances.py
```

These commands write the reconstructed results in `results/`. `README.txt` describes the individual tables and the separate raw-record reconstruction command. `ANALYSIS_NOTES_EN.txt` gives the scientific interpretation.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
