# Data

This folder records data identities and access information. It does not contain the complete raw trajectory collection.

`DATA_MANIFEST.csv` lists the original historical RData and derived-file identities. Obtain the RData files from the [cited source](RAW_DATA_SOURCES.md), place them in a local raw-data directory, and verify them from the repository root:

```sh
python verify_external_data.py --raw-dir data/raw
```

The original pickle caches and 4.224 GB transport matrix are also external to this snapshot.

`DV_RAW_ARCHIVE.json` identifies the separate 1,915-run DV export. `R5_RAW_ARCHIVES.json` records the R5 allocation, export format and preparation-time status. Actual R5 part names and hashes come from the export catalog. See [input availability](../INPUT_AVAILABILITY.md) for the distinction between included records and externally accessible files.
