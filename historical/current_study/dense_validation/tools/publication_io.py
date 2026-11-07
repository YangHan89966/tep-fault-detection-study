"""Standard-library publication integrity helpers; no scientific execution."""
from __future__ import annotations
import csv, hashlib, io, json, os, stat
from pathlib import Path, PurePosixPath

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def canonical(obj):return (json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n').encode('ascii')
def loads(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key: '+k)
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def readj(p):return loads(Path(p).read_bytes())
def reject_links(p):
    for q in [Path(p),*Path(p).parents]:
        if not os.path.lexists(q):continue
        st=q.lstat()
        if stat.S_ISLNK(st.st_mode) or getattr(st,'st_file_attributes',0)&0x400:
            raise ValueError('symlink/reparse point refused: '+q.name)
def child(root,rel):
    parts=rel.split('/')
    if not rel or '\\' in rel or PurePosixPath(rel).is_absolute() or any(x in ('','.','..','.git') or ':' in x for x in parts):
        raise ValueError('unsafe relative path: '+rel)
    p=Path(root).joinpath(*parts);reject_links(p);return p
def identity(p):
    reject_links(p)
    if not p.exists():return None
    if not p.is_file():raise ValueError('expected file: '+str(p))
    return {'size_bytes':p.stat().st_size,'sha256':sha(p)}
def verify_manifest(root,name='FILES.json',exact=True):
    m=readj(child(root,name));seen=set()
    for e in m['entries']:
        if e['path'] in seen or e['path']==name:raise ValueError('duplicate/self manifest entry')
        seen.add(e['path']);p=child(root,e['path'])
        if identity(p)!={'size_bytes':e['bytes'],'sha256':e['sha256']}:raise ValueError('file hash mismatch: '+e['path'])
    if exact:
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual!=seen|{name}:raise ValueError('unexpected or missing files in manifest tree')
    return m
