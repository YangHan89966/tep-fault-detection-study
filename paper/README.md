# Paper and supporting tables

Scientific files from the final V10 manuscript package, dated 6 November 2026. The three Word documents, seven figures and Data S1 archive retain their original bytes.

Before opening the documents or figures, run this command from the repository root:

```sh
python unpack.py
```

This restores the DOCX and TIFF files from the included bundles. Access the current repository at [tep-fault-detection-study](https://github.com/YangHan89966/tep-fault-detection-study) and the archived study materials on [Zenodo](https://doi.org/10.5281/zenodo.22084007).

- `Manuscript.docx`: main manuscript.
- `Supplementary_Information.docx`: supplementary methods and results.
- `Highlights.docx`: research highlights.
- `Figure_1.tif` through `Figure_7.tif`: figure files.
- `Data_S1/`: 83 CSV tables, with an index and readme; also supplied as `Data_S1.zip`.
- `SOURCE_MANIFEST.json`: file sizes, SHA-256 checksums and archive member mapping.

Run `python verify_paper.py` from the repository root to check file integrity, Data S1 extraction and all 68 supplementary tables against the Word document. This comparison covers 6,110 normalized table cells.
