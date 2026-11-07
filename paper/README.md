# Paper and supporting tables

The files submitted for publication in November 2026: the manuscript, the Supplementary Information, the Highlights, the seven figures and Data S1. They are byte-identical to the submitted files.

The Word and TIFF files are stored in `bundles/`, so run this from the repository root before opening them:

```sh
python unpack.py
```

- `Manuscript.docx`: the manuscript
- `Supplementary_Information.docx`: supplementary methods and results
- `Highlights.docx`: highlights
- `Figure_1.tif` to `Figure_7.tif`: the figures
- `Data_S1/`: the 83 CSV tables (S1–S68 and D1–D15) with an index and a readme, also supplied as `Data_S1.zip`
- `SOURCE_MANIFEST.json`: sizes and SHA-256 checksums of these files

From the repository root, `python verify_paper.py` checks these files and compares all 68 supplementary tables in the Word document with Data S1, 6,110 cells in total.

The full archive, including the raw trajectories, is on Zenodo: https://doi.org/10.5281/zenodo.22084007
