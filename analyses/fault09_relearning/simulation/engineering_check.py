"""Portable engineering-only 48 h equivalence check (python engineering_check.py). No panel key, score or scientific output."""
import hashlib,json,sys,time
from pathlib import Path
from scientific_runner import pinned_engine
import numpy as np
root=Path(__file__).resolve().parent
# Shared with the prior full-horizon non-study QA; never used as a scientific key.
QA_KEY=0x51414F4E4C595F5445505F524E475F33
ARRAY_SHA='f8834970a1cf83f9923276342df513d114399f51969b80a4ad3b04fcd99dbf3a'
META_SHA='126ada5bf426b83d2efc274dad7dafd1e656eedcba5371866e8db8c0e12c86fd'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(out, hours=48):
    refroot=root/'reference';mp=refroot/'RNG_FULL_HORIZON_QA.json';ap=refroot/'QA_NORMAL48_DUAL.npz'
    if sha(mp)!=META_SHA or sha(ap)!=ARRAY_SHA:raise ValueError('Engineering truth SHA mismatch')
    jsonref=json.loads(mp.read_text());mref=jsonref['normal48_metadata']['dual']
    assert int(jsonref['normal_key_hex'],16)==QA_KEY
    start=time.perf_counter();a,m=pinned_engine().run_one(0,QA_KEY,hours,mode='dual',source_dir=root/'pinned/engine');elapsed=time.perf_counter()-start
    with np.load(ap,allow_pickle=False) as z:
        if set(a)!=set(z.files):raise ValueError('Different engineering array names')
        exact={k:v.dtype==z[k].dtype and v.shape==z[k][:len(v)].shape and v.tobytes()==z[k][:len(v)].tobytes() for k,v in a.items()}
    same={k:m[k]==mref[k] for k in ('rng_draw_count','rng_consumption_trace_sha256','full_step_trace_sha256','physical_state_sha256')} if hours==48 else {}
    ans={'engineering_only':True,'key_hex':hex(QA_KEY),'duration_hours':hours,'seconds':elapsed,'exact_array_bytes':exact,'exact_full_run_digests':same,'pass':all(exact.values()) and all(same.values()),'python':sys.version,'numpy':np.__version__,'draw_count':m['rng_draw_count'],'full_step_trace_sha256':m['full_step_trace_sha256']}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(ans,indent=2)+'\n',encoding='utf-8');print(json.dumps(ans),flush=True)
    if not ans['pass']:raise RuntimeError('Engineering gate failed; do not run this machine as a study worker.')
    return ans
if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else root/'engineering/PORTABLE_ENGINE_QA.json')
