"""One locked manifest row -> paired 960x52 records. The stage scheduler owns ordering.
No direct scientific CLI: the study's locked driver calls run_row in independent processes.
"""
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import datetime as dt,hashlib,importlib.util,json,os,re,sys,time,uuid
for env in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[env]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
INDEX_SHA='7298717b1a3c86387fad99c4e4e53ed9b3743100dd8f2b335810d7f5cafa0982'
ADAPTER_SHA='d1d218e847152d08e7cf6c9299b48e98c7738cd491c781ea3123df9315ef6450'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def writej(p,obj):
    tmp=p.with_suffix('.new');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def getj(p):return json.loads(p.read_text(encoding='utf-8'))
@lru_cache(maxsize=1)
def pinned_engine():
    root=HERE/'pinned';p=root/'ENGINE_FILES.json'
    if sha(p)!=INDEX_SHA:raise ValueError('Pinned engine index SHA mismatch.')
    for rel,d in getj(p)['files'].items():
        if sha(root/rel)!=d:raise ValueError('Pinned engine source mismatch: '+rel)
    if sys.version_info[:2]!=(3,12) or np.__version__!='2.3.5' or sys.implementation.name!='cpython':
        raise RuntimeError('Use CPython 3.12 with NumPy 2.3.5, as in the checked reference runtime.')
    adapter=root/'code/rng_engine.py';spec=importlib.util.spec_from_file_location('_p1_philox_frozen_adapter',adapter)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);mod.load_engine(root/'engine')
    return mod
@contextmanager
def exclusive(folder):
    with (folder/'WORKER.lock').open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
def identity(row,lock):
    if not isinstance(lock,str) or not re.fullmatch('[0-9a-f]{64}',lock):raise ValueError('Caller must pass frozen protocol lock SHA256.')
    key=str(row['key_hex']); fid=int(row['fault_id']); rid=str(row.get('case_id') or row.get('run_id') or '')
    if not (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}',rid) and fid in (0,9) and re.fullmatch('0[xX][0-9a-fA-F]{32}',key)):
        raise ValueError('Invalid scientific case ID, fault ID or 128-bit stream key: '+rid)
    role=str(row['role']);rep=row.get('rep',row.get('replicate',row.get('training_rep')))
    if not role or rep is None:raise ValueError('Manifest must specify role and replicate (0/shared for shared rows).')
    return dict(run_id=rid,role=role,replicate=str(rep),fault_id=fid,key_hex=key.lower(),protocol_lock_sha256=lock,engine_index_sha256=INDEX_SHA,adapter_sha256=ADAPTER_SHA)
def completed(folder,identity):
    path=folder/'COMPLETE.json'
    if not path.exists():return None
    rec=getj(path)
    if any(rec.get(k)!=v for k,v in identity.items()) or set(rec.get('outputs',{}))!={'RAW.npz','RUN_INFO.json'}:
        raise ValueError('Existing completion identity differs: '+folder.name)
    if any(sha(folder/k)!=v for k,v in rec['outputs'].items()):raise ValueError('Completed files changed: '+folder.name)
    return rec
def run_row(row,raw_dir,protocol_lock_sha256):
    """Returns the verified COMPLETE record; skips an existing matching, SHA-verified completion."""
    target=identity(row,protocol_lock_sha256);folder=Path(raw_dir)/target['run_id'];folder.mkdir(exist_ok=True,parents=True)
    with exclusive(folder):
        previous=completed(folder,target)
        if previous:return previous
        eng=pinned_engine()
        started=folder/'STARTED.json'
        if started.exists():
            prior=getj(started)
            if any(prior.get(k)!=v for k,v in target.items()):raise ValueError('Incomplete run belongs to a different study/key: '+folder.name)
        elif (folder/'RAW.npz').exists() or (folder/'RUN_INFO.json').exists():
            raise ValueError('Unidentified partial run; refusing to overwrite '+folder.name)
        old=[p for p in folder.iterdir() if p.is_file() and p.name!='WORKER.lock']
        if old:
            retain=folder/'incomplete_attempts'/uuid.uuid4().hex;retain.mkdir(parents=True)
            for p in old:p.rename(retain/p.name)
        utc0=now();writej(started,dict(**target,attempt_started_utc=utc0))
        try:
            arrays,meta=eng.run_one(target['fault_id'],target['key_hex'],duration_hours=48.,onset_hours=8.,mode='dual',rng_kind='philox',source_dir=HERE/'pinned/engine')
            if meta['final_step_count']!=172800 or arrays['native_values'].shape!=(960,52) or meta['rng_draw_count']>=2**64:raise ValueError('Wrong registered horizon or stream bound')
            if meta['adapter_sha256']!=ADAPTER_SHA:raise ValueError('Unexpected adapter')
            meta.pop('source_directory',None)
            with (folder/'RAW.new').open('wb') as f:np.savez_compressed(f,**arrays)
            os.replace(folder/'RAW.new',folder/'RAW.npz')
            utc1=now();info=dict(meta);info.update(target);info.update(started_utc=utc0,completed_utc=utc1)
            writej(folder/'RUN_INFO.json',info)
            rec=dict(**target,started_utc=utc0,completed_utc=utc1,outputs={n:sha(folder/n) for n in ['RAW.npz','RUN_INFO.json']},rng_draw_count=meta['rng_draw_count'],wall_seconds=meta['wall_seconds'],shutdown_occurred=meta['shutdown_occurred'],physical_state_sha256=meta['physical_state_sha256'])
            writej(folder/'COMPLETE.json',rec);return rec
        except BaseException as e:
            writej(folder/'FAILED.json',dict(**target,failed_utc=now(),error=repr(e)));raise
