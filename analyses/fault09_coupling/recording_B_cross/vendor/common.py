"""Local file integrity and finite-study bookkeeping; no network or installation."""
from __future__ import annotations
import csv, datetime, hashlib, json, os, platform, uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)
def write_csv(p,rows):
    if not rows:raise ValueError('Empty result rows require explicit schema')
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    with q.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(q,p)
def npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    with q.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(q,p)
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def finite(x,label='array'):
    if not np.isfinite(x).all():raise ValueError('Nonfinite '+label)
    return x

def seal(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='FILES.json' and '__pycache__' not in p.parts}
def verify_files(folder,entries):
    for name,d in entries.items():
        p=Path(folder)/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(Path(folder).resolve()) or sha(p)!=d:raise ValueError('Changed/missing file: '+str(p))
def environment():return dict(python=platform.python_version(),numpy=np.__version__,executable=os.path.realpath(os.sys.executable),system=platform.system(),machine=platform.machine(),byteorder=os.sys.byteorder)
def safe_dir(p):
    p=Path(p)
    if p.is_symlink():raise ValueError('Symlink output refused')
    for par in p.parents:
        if par==ROOT:break
        if par.is_symlink():raise ValueError('Symlink output ancestor refused')
    if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Output outside new package')
    p.mkdir(parents=True,exist_ok=True);return p
