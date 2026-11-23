"""Obtain and verify the unchanged public Zenodo v2 source archive."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://zenodo.org/records/16978349/files/Data.zip?download=1'
SHA256 = 'e5fa6872449d4b538065b498ff547ebfddbe751831c8cceb852f30935a185f4f'
SIZE = 17658806
ROOT = Path(__file__).resolve().parent


def verify(path):
    if path.stat().st_size != SIZE:
        raise ValueError('Source size differs from the audited v2 release')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError('Source SHA256 differs from the audited v2 release')


def main():
    target = ROOT / 'data' / 'Data.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target)
        print('Verified existing source archive:', target)
        return
    temporary = target.with_suffix('.download')
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'PID-pairing-reproducibility/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        verify(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded and verified:', target)


if __name__ == '__main__':
    main()
