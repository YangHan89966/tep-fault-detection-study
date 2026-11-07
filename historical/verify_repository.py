#!/usr/bin/env python3
"""Verify allowlisted repository files without importing or executing research code."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import pathlib
import stat
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def safe_file(relative: str) -> pathlib.Path:
    rel = pathlib.PurePosixPath(relative)
    if (not relative or '\\' in relative or rel.is_absolute() or
            any(x in {'', '.', '..', '.git'} or ':' in x for x in relative.split('/'))):
        raise ValueError('unsafe manifest path')
    path = ROOT.joinpath(*rel.parts)
    for p in [path, *path.parents]:
        s = p.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 0x400:
            raise ValueError('link or reparse point in manifest path')
        if p == ROOT:
            break
    if not path.is_file():
        raise ValueError('missing regular file')
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile-only', action='store_true',
                    help='Legacy compatibility flag; this verifier always checks hashes and AST.')
    ap.parse_args()
    errors = []
    sums = {}
    try:
        for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            digest, relative = line.split('  ', 1)
            if relative in sums or len(digest) != 64:
                raise ValueError('duplicate entry or bad digest')
            int(digest, 16)
            sums[relative] = digest
        if 'FILE_MANIFEST.csv' not in sums or 'SHA256SUMS.txt' in sums:
            raise ValueError('manifest coverage error')
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}))
        return 1
    py_count = json_count = 0
    sizes = {}
    for relative, expected in sums.items():
        try:
            p = safe_file(relative)
            raw = p.read_bytes()
            sizes[relative] = len(raw)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('SHA-256 mismatch')
            if p.suffix == '.py':
                ast.parse(raw, filename=relative)
                py_count += 1
            elif p.suffix == '.json':
                json.loads(raw.decode('utf-8-sig'))
                json_count += 1
        except (OSError, ValueError, SyntaxError) as exc:
            errors.append({'path': relative, 'error': str(exc)})
    try:
        rows = list(csv.DictReader(io.StringIO((ROOT / 'FILE_MANIFEST.csv').read_text(encoding='utf-8'))))
        paths = [r['relative_path'] for r in rows]
        if len(paths) != len(set(paths)) or set(paths) != set(sums) - {'FILE_MANIFEST.csv'}:
            raise ValueError('FILE_MANIFEST / SHA256SUMS coverage differs')
        for row in rows:
            rel = row['relative_path']
            if row['sha256'] != sums.get(rel) or int(row['size_bytes']) != sizes.get(rel):
                errors.append({'path': rel, 'error': 'file-manifest mismatch'})
    except (OSError, ValueError, KeyError) as exc:
        errors.append({'path': 'FILE_MANIFEST.csv', 'error': str(exc)})
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(sums) | {'SHA256SUMS.txt'}:
        errors.append({'path': '.', 'error': 'Unlisted or missing distribution files'})
    result = {'status': 'PASS' if not errors else 'FAIL',
              'manifest_files_hashed': len(sums), 'python_files_ast_parsed': py_count,
              'json_files_parsed': json_count, 'errors': errors,
              'research_code_executed': False, 'package_import_test': False,
              'unlisted_local_files_verified': True, 'live_remote_checked': False,
              'nested_historical_manifest_replay': False,
              'note': 'Current-distribution integrity and syntax only; not scientific replay or a licence determination.'}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
