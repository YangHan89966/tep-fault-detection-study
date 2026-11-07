"""Verified local Stage D loader. Reads no process or test data itself."""
from pathlib import Path
import hashlib,json,os,runpy,sys
ROOT=Path(__file__).resolve().parent
PAYLOAD=ROOT/'payload'
EXPECTED='72ad094017bd09fe7f911cf85effd10ee2cc9030f8e6e7d54b07be8d4a1a930e'
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise RuntimeError('Use the provided -I -B launcher')
manifest=PAYLOAD/'MANIFEST.json'
if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED:
    raise RuntimeError('Stage D payload manifest differs; extract the original ZIP again')
obj=json.loads(manifest.read_text('utf-8'));seen=set()
for item in obj['entries']:
    rel=item['path'];p=PAYLOAD
    if rel in seen or Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise RuntimeError('Invalid payload path')
    seen.add(rel)
    for part in Path(rel).parts:
        p=p/part
        if p.is_symlink():raise RuntimeError('Payload symlink')
    if not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        raise RuntimeError('Changed or missing payload file: '+rel)
actual={p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob('*') if p.is_file()}
if actual!=seen|{'MANIFEST.json'}:raise RuntimeError('Unexpected payload files')
os.chdir('/')
sys.path.insert(0,str(PAYLOAD/'code'))
sys.argv[0]=str(PAYLOAD/'code/runner_stage_d.py')
runpy.run_path(sys.argv[0],run_name='__main__')
