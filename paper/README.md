# Historical manuscript and Data S1, 7 November 2026

This directory preserves the manuscript, Supplementary Information (SI), Highlights, seven figures and Data S1 from the earlier version of this study. Its title is *Effects of monitoring statistic, projection direction and measurement–controller pairing on the detectability of Tennessee Eastman Faults 3, 9 and 15*. The record of that version is [Zenodo v1.0.0](https://doi.org/10.5281/zenodo.22084007).

Readers using the measurement–controller pairing manuscript can locate each current table by its submission or extended SI number in the [table crosswalk](../measurement_pairing/tables/SI_table_crosswalk.csv). The current six figures can be regenerated from the [figure sources](../measurement_pairing/README.md).

| File | Historical material |
| --- | --- |
| `Manuscript.docx`, `Supplementary_Information.docx`, `Highlights.docx` | Earlier manuscript and accompanying documents |
| `Figure_1.tif` through `Figure_7.tif` | Earlier figures |
| [Data_S1/](Data_S1/) and `Data_S1.zip` | Original tables S1–S68 and D1–D15, with an index |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | Byte counts and checksums of the archived files |

From the repository root, run `python unpack.py` to restore the larger files and `python verify_paper.py` to check them against the historical snapshot.
