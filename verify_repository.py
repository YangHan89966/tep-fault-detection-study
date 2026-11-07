"""Verify the exact bytes of the repository and restored scientific files."""
from pathlib import Path, PurePosixPath
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    if len(rows) != len({r['path'] for r in rows}):
        raise ValueError('Duplicate manifest paths')
    missing, changed = [], []
    total = 0
    for row in rows:
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in row['path']:
            raise ValueError('Invalid manifest path')
        path = ROOT.joinpath(*rel.parts)
        if not path.is_file():
            missing.append(row['path'])
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            changed.append(row['path'])
    if missing or changed:
        print(json.dumps({'status': 'FAIL', 'missing': missing, 'changed': changed}, indent=2))
        raise SystemExit(1)
    print(json.dumps({'status': 'PASS', 'files': len(rows), 'bytes': total}))


if __name__ == '__main__':
    main()
