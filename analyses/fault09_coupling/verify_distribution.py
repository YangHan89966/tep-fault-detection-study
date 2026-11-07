"""Verify all files recorded in this public distribution's SHA-256 manifest."""
from pathlib import Path
import hashlib
import json
import sys

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'PUBLIC_FILE_MANIFEST.json').read_text(encoding='utf-8'))
    failed = []
    for row in manifest['files']:
        path = root / row['path']
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            failed.append(row['path'] + ': missing or invalid file')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != row['size_bytes'] or digest.hexdigest() != row['sha256']:
            failed.append(row['path'] + ': size or SHA-256 mismatch')
    result = {'status': 'FAIL' if failed else 'PASS',
              'files_checked': len(manifest['files']), 'failures': failed}
    print(json.dumps(result, indent=2))
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
