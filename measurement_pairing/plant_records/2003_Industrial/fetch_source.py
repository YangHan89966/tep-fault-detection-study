#!/usr/bin/env python3
"""Download the published SACAC level-loop record and verify exact source bytes."""
from pathlib import Path
import argparse
import hashlib
import io
import urllib.request
import zipfile

BASE = Path(__file__).resolve().parent
FILES = {
    'other-L-paper-horch-2003.csv': '76da7b972818f9116afc27d31c1afb44196d2f66325a7c5aba27be5a8047b2c0',
    'other-L-paper-horch-2003.txt': 'af1e5c064f7143822a40333dba4464db5eac40a48ff8a954ad68f021076b7814',
}
SOURCE = 'https://sacac.org.za/wp-content/uploads/2023/07/'
ARCHIVE = 'https://sacac.org.za/Resources/DataRepositories/DataRepositories.zip'


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'IndustrialPairingReanalysis/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE / 'data')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(args.archive) if args.archive else None
    for name, expected in FILES.items():
        path = args.out / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f'Verified existing source: {name}')
            continue
        if archive is None:
            try:
                data = download(SOURCE + name)
            except Exception as exc:
                print(f'Direct download unavailable ({type(exc).__name__}); trying official archive.')
                archive = zipfile.ZipFile(io.BytesIO(download(ARCHIVE)))
                data = archive.read('Other/other-L-paper-horch-2003/' + name)
        else:
            data = archive.read('Other/other-L-paper-horch-2003/' + name)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Source changed for {name}: expected {expected}, got {actual}. Review the new source before analysis.')
        path.write_bytes(data)
        print(f'Downloaded and verified: {name}')


if __name__ == '__main__':
    main()
