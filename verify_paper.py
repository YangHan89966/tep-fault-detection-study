#!/usr/bin/env python3
"""Verify paper bytes and compare supplementary tables with Data S1."""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
ORIGINALS = {'Manuscript.docx', 'Supplementary_Information.docx', 'Highlights.docx', 'Data_S1.zip'} | {f'Figure_{i}.tif' for i in range(1, 8)}
DATA = {'Data_S1/README.txt', 'Data_S1/index.csv'} | {f'Data_S1/Table_S{i:02}.csv' for i in range(1, 69)} | {f'Data_S1/Table_D{i:02}.csv' for i in range(1, 16)}
SOURCE_HASH = '435203d1b55302b1ea03d4ad55905cc75cdeeb453e7bfbb47937100ceb51bb59'


def checked_zip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {}
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename or item.filename in files:
                raise ValueError('Invalid or duplicate ZIP member')
            if item.flag_bits & 1:
                raise ValueError('Encrypted ZIP member')
            files[item.filename] = archive.read(item)
        return files


def normalize(value):
    return re.sub(r'\s+', ' ', value.replace('\u00ad', '').replace('\u200b', '')).strip()


def compare_si_tables(data, data_dir):
    document = ET.fromstring(checked_zip(data)['word/document.xml'])
    tables = document.findall('.//' + W + 'tbl')
    if len(tables) != 68:
        raise ValueError('SI table count mismatch')
    cells = 0
    for i, table in enumerate(tables, 1):
        docrows = []
        for tr in table.findall(W + 'tr'):
            row = []
            for tc in tr.findall(W + 'tc'):
                if tc.find(W + 'tcPr/' + W + 'vMerge') is not None:
                    raise ValueError('Unexpected vertical table merge')
                paragraphs = []
                for p in tc.findall(W + 'p'):
                    paragraphs.append(''.join(' ' if n.tag in {W+'br', W+'tab'} else n.text or ''
                                              for n in p.iter() if n.tag in {W+'t', W+'br', W+'tab'}))
                span = tc.find(W + 'tcPr/' + W + 'gridSpan')
                width = int(span.get(W+'val')) if span is not None else 1
                if not 1 <= width <= 100:
                    raise ValueError('Invalid table span')
                row.extend([normalize(' '.join(paragraphs))] + [''] * (width - 1))
            docrows.append(row)
        with (data_dir/f'Table_S{i:02}.csv').open(encoding='utf-8', newline='') as stream:
            csvrows = [[normalize(c) for c in row] for row in csv.reader(stream)]
        if docrows != csvrows:
            raise ValueError(f'SI/Data S1 cell mismatch: S{i:02}')
        cells += sum(len(row) for row in csvrows)
    if cells != 6110:
        raise ValueError('SI normalized cell count mismatch')
    return {'tables': len(tables), 'normalized_cells': cells}


def check_docx(data):
    entries = checked_zip(data)
    if '[Content_Types].xml' not in entries or 'word/document.xml' not in entries:
        raise ValueError('Incomplete DOCX')
    for name, value in entries.items():
        if 'vbaproject' in name.lower():
            raise ValueError('Unexpected macro')
        if name.startswith('word/') and name.endswith('.xml'):
            tree = ET.fromstring(value)
            if any(tree.find('.//' + W + tag) is not None for tag in ('ins', 'del', 'commentRangeStart', 'commentRangeEnd', 'comment')):
                raise ValueError('Unexpected tracked changes or comments')


def verify(root=ROOT):
    base = Path(root)/'paper'
    manifest = json.loads((base/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    source = manifest['source_archive']
    if manifest['snapshot'] != 'final_v11' or source['sha256'] != SOURCE_HASH or source['size_bytes'] != 14235720:
        raise ValueError('Unexpected source snapshot')
    records = manifest['included_files']
    if len(records) != 96 or {r['path'] for r in records} != ORIGINALS | DATA:
        raise ValueError('Paper file set mismatch')
    for row in records:
        data = (base/row['path']).read_bytes()
        git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256'] or git_hash != row['git_blob_sha1']:
            raise ValueError('File checksum mismatch: ' + row['path'])
        if row['path'].endswith('.docx'):
            check_docx(data)
    entries = checked_zip((base/'Data_S1.zip').read_bytes())
    actual = {p.relative_to(base).as_posix() for p in (base/'Data_S1').rglob('*') if p.is_file()}
    if set(entries) != DATA or actual != DATA:
        raise ValueError('Data S1 member set mismatch')
    for name, data in entries.items():
        if (base/name).read_bytes() != data:
            raise ValueError('Data S1 extraction differs from ZIP: ' + name)
    tables = compare_si_tables((base/'Supplementary_Information.docx').read_bytes(), base/'Data_S1')
    return {'status': 'PASS', 'original_files': 11, 'docx_checked': 3,
            'data_s1_extracts': 85, 'table_csv_files': 83, 'si_table_check': tables}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), sort_keys=True))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        print('FAIL: ' + str(exc), file=sys.stderr)
        sys.exit(1)
