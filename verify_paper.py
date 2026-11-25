#!/usr/bin/env python3
"""Check the 7 November 2026 Data S1 tables against their original ZIP and index."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "af2485b2061b56a94e56c3b72dad319d61b37531d38b57de6569a1398a28d889"


def verify(root=ROOT):
    paper = Path(root) / "paper"
    tables = paper / "Data_S1"
    manifest = json.loads((paper / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    labels = {f"Table_{series}{n:02}.csv": f"Table {series}{n}"
              for series, count in (("S", 68), ("D", 15)) for n in range(1, count + 1)}
    expected = {f"Data_S1/{name}" for name in (*labels, "README.txt", "index.csv")}
    records = manifest["included_files"]
    if len(records) != len(expected) or {row["path"] for row in records} != expected:
        raise ValueError("Unexpected file list in Data S1 source manifest")

    archive_path = paper / "Data_S1.zip"
    source = manifest["source_archive"]
    if (source["name"] != archive_path.name or source["sha256"] != ARCHIVE_SHA256
            or archive_path.stat().st_size != source["size_bytes"]
            or hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256):
        raise ValueError("Data S1 source ZIP checksum or size mismatch")

    present = {p.relative_to(paper).as_posix() for p in tables.rglob("*") if p.is_file()}
    if present - {"Data_S1/README.md"} != expected:
        raise ValueError("Data S1 directory differs from the source inventory")
    with ZipFile(archive_path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Data S1 ZIP differs from the source inventory")
        for row in records:
            name = row["path"]
            data = (paper / name).read_bytes()
            if archive.read(name) != data:
                raise ValueError(f"{name} differs from its original ZIP member")
            if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"{name} differs from its source manifest")
            git_blob = f"blob {len(data)}\0".encode() + data
            if hashlib.sha1(git_blob).hexdigest() != row["git_blob_sha1"]:
                raise ValueError(f"Git blob checksum mismatch for {name}")

    with (tables / "index.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    if len(index) != len(labels) or {row["file"]: row["table"] for row in index} != labels:
        raise ValueError("Table names and labels differ from the Data S1 index")
    cells = 0
    for name in sorted(tables.glob("Table_S*.csv")):
        with name.open(encoding="utf-8-sig", newline="") as stream:
            cells += sum(len(row) for row in csv.reader(stream))
    if cells != 6110:
        raise ValueError(f"Unexpected cell count in Tables S1–S68: {cells}")
    return {"status": "PASS", "data_s1_extracts": len(expected),
            "table_csv_files": len(labels), "si_table_check": {"tables": 68, "normalized_cells": cells}}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), sort_keys=True))
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
