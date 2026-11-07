"""Restore checked scientific assets from the included bundle parts."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or '\\' in name
            or ':' in name or str(p) != name):
        raise ValueError('Invalid bundle path: ' + name)
    dest = ROOT.joinpath(*p.parts)
    if not dest.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Path leaves repository: ' + name)
    return dest


def main():
    manifest = json.loads((ROOT / 'BUNDLES.json').read_text(encoding='utf-8'))
    restored = reused = 0
    for archive in manifest['archives']:
        with tempfile.TemporaryFile() as assembled:
            for part in archive['parts']:
                data = checked_path(part['path']).read_bytes()
                if len(data) != part['size_bytes'] or digest(data) != part['sha256']:
                    raise ValueError('Bundle part mismatch: ' + part['path'])
                assembled.write(data)
            assembled.seek(0)
            h = hashlib.sha256()
            for block in iter(lambda: assembled.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != archive['sha256']:
                raise ValueError('Assembled bundle mismatch')
            assembled.seek(0)
            with zipfile.ZipFile(assembled) as z:
                expected = {r['path']: r for r in archive['members']}
                if len(z.infolist()) != len(expected) or set(z.namelist()) != set(expected):
                    raise ValueError('Bundle member-set mismatch')
                pending = []
                for name, row in expected.items():
                    dest = checked_path(name)
                    data = z.read(name)
                    if len(data) != row['size_bytes'] or digest(data) != row['sha256']:
                        raise ValueError('Bundle member mismatch: ' + name)
                    if dest.exists():
                        if not dest.is_file() or digest(dest.read_bytes()) != row['sha256']:
                            raise ValueError('Existing file differs: ' + name)
                        reused += 1
                    else:
                        pending.append((dest, data))
                for dest, data in pending:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open('xb') as handle:
                        handle.write(data)
                    restored += 1
    print(json.dumps({'status': 'PASS', 'restored': restored, 'already_present': reused}))


if __name__ == '__main__':
    main()
