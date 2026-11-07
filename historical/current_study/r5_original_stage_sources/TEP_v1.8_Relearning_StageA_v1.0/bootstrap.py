#!/usr/bin/env python3
"""Verify the pinned release before importing package-local operation code."""
from pathlib import Path
import hashlib,json,runpy,sys
EXPECTED = '0befb68b1fbca396154cfdad53e8926d64b27784bb8890fa731740d8757273a5'
HERE=Path(__file__).resolve().parent
ROOT=HERE/'release'
if sys.dont_write_bytecode is not True or not sys.flags.isolated:
    raise RuntimeError('Launch this tool with Python -I -B')
manifest=ROOT/'RELEASE_MANIFEST.json'
if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED:
    raise RuntimeError('Release manifest changed; stop and return the Windows log')
record=json.loads(manifest.read_bytes());seen=set()
for entry in record['entries']:
    rel=entry['path'];parts=Path(rel).parts
    if Path(rel).is_absolute() or not parts or '..' in parts or rel in seen:
        raise RuntimeError('Invalid release entry')
    p=ROOT
    for part in parts:
        p=p/part
        if p.is_symlink():raise RuntimeError('No symlinks permitted in bundled release')
    if not p.is_file() or p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        raise RuntimeError('Release input changed: '+rel)
    seen.add(rel)
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
if actual!=seen|{'RELEASE_MANIFEST.json'}:raise RuntimeError('Release contains unexpected/missing files')
sys.path.insert(0,str(ROOT/'code'))
runpy.run_path(str(ROOT/'code/runner_stage_a.py'),run_name='__main__')
