# Archived manuscript and supporting tables, 7 November 2026

This directory preserves the manuscript, Supplementary Information, Highlights, seven figures and Data S1 from the 7 November 2026 release. See the [main branch](https://github.com/YangHan89966/tep-fault-detection-study/blob/main/README.md) for the revised study.

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
