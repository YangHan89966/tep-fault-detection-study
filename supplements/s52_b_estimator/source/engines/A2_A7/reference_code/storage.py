"""Local-only integrity and transaction helpers. No simulator or scientific imports."""
from __future__ import annotations
import csv, hashlib, io, json, os, stat, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def canonical(obj: Any) -> bytes:
    return (json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n').encode('ascii')


def strict_json(path: Path) -> Any:
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(Path(path).read_text('utf-8'),object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def atomic_bytes(path: Path, data: bytes, replace: bool=False) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or (path.exists() and not replace):raise FileExistsError(str(path))
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.writing',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        if not replace and path.exists():raise FileExistsError(str(path))
        os.replace(tmp,path)
        try:
            d=os.open(path.parent,os.O_RDONLY);os.fsync(d);os.close(d)
        except OSError:pass
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_json(path: Path,obj: Any,replace: bool=False) -> None:
    atomic_bytes(path,canonical(obj),replace)


def csv_bytes(rows: list[dict]) -> bytes:
    s=io.StringIO(newline='')
    if rows:
        w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return s.getvalue().encode('utf-8')


def write_csv(path: Path,rows: list[dict],replace: bool=False) -> None:
    atomic_bytes(path,csv_bytes(rows),replace)


def regular(path: Path) -> None:
    st=path.lstat()
    if not stat.S_ISREG(st.st_mode):raise ValueError('expected regular non-symlink file: '+str(path))


def safe_child(root: Path, rel: str) -> Path:
    parts=Path(rel).parts
    if not parts or Path(rel).is_absolute() or any(x in ('.','..') for x in parts):raise ValueError('invalid relative path')
    p=root
    for x in parts:
        p=p/x
        if p.is_symlink():raise ValueError('symlink in owned tree')
    return p


def seal_tree(root: Path,name: str='FILES.json') -> dict:
    files=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in transaction')
        if p.is_file() and p.relative_to(root).as_posix()!=name:
            files.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    record={'entries':files,'self_excluded':name}
    write_json(root/name,record)
    return record


def verify_tree(root: Path,name: str='FILES.json',exact: bool=True) -> dict:
    regular(root/name);obj=strict_json(root/name)
    if not isinstance(obj,dict) or not isinstance(obj.get('entries'),list):raise ValueError('invalid file manifest')
    seen=set()
    for r in obj['entries']:
        rel=r['path']
        if rel in seen or rel==name:raise ValueError('duplicate manifest member')
        seen.add(rel);p=safe_child(root,rel);regular(p)
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('file identity mismatch: '+rel)
    if exact:
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual!=seen|{name}:raise ValueError('unexpected or missing transaction file')
    return obj


class Ledger:
    """Append-only hash chained local operation record; no authorizations/secrets."""
    allowed={'SETUP','SESSION_START','ATTEMPT_START','ATTEMPT_COMMIT','RECOVER_COMMIT',
             'INTERRUPTED_SAME_SEED_RETRY','SESSION_END','SESSION_FAILURE','LEARN_START',
             'LEARN_LOCKED','EVALUATE_START','EVALUATE_COMPLETE'}
    def __init__(self,path: Path,established: bool=False):
        self.path=path;self.records=[];self.last='0'*64
        if path.exists():
            raw=path.read_bytes()
            if raw and not raw.endswith(b'\n'):raise ValueError('torn ledger; retain for repair')
            for line in raw.splitlines():
                r=json.loads(line)
                sig=r.pop('sha256',None)
                if r.get('event') not in self.allowed or r.get('seq')!=len(self.records) or r.get('prev')!=self.last:raise ValueError('ledger schema/order')
                if hashlib.sha256(canonical(r)).hexdigest()!=sig:raise ValueError('ledger hash')
                r['sha256']=sig;self.records.append(r);self.last=sig
        if established and not self.records:raise ValueError('established batch has missing/empty ledger')
    def add(self,event: str,**data):
        if event not in self.allowed:raise ValueError('unknown ledger event')
        r={'seq':len(self.records),'prev':self.last,'utc':now(),'event':event,'data':data}
        r['sha256']=hashlib.sha256(canonical(r)).hexdigest()
        with self.path.open('ab') as f:f.write(canonical(r));f.flush();os.fsync(f.fileno())
        self.records.append(r);self.last=r['sha256']
        return r
