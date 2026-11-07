#!/usr/bin/env python3
"""Verify externally obtained TEP raw files against the manuscript data manifest."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/DATA_MANIFEST.csv')
    parser.add_argument('--raw-dir', required=True)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    failed = 0
    with Path(args.manifest).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['data_class'] != 'raw':
                continue
            path = raw_dir / row['basename']
            if not path.is_file():
                print(f"MISSING {path}")
                failed += 1
                continue
            size_ok = str(path.stat().st_size) == row['size_bytes'].replace(',', '')
            digest = sha256_file(path)
            hash_ok = digest == row['sha256']
            print(f"{row['basename']}: size={'PASS' if size_ok else 'FAIL'} sha256={'PASS' if hash_ok else 'FAIL'}")
            if not (size_ok and hash_ok):
                failed += 1
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
