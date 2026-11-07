"""Subprocess entry point. Keeps worker imports portable on Windows spawn."""
import os,subprocess,sys,shutil,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
def resolve_work(s):
    if os.name!='nt' and len(s)>2 and s[1]==':' and os.environ.get('WSL_DISTRO_NAME'):
        s=subprocess.check_output(['wslpath','-u',s],text=True).strip()
    return Path(s).expanduser().resolve()
def start(argv):
    env=dict(os.environ,PYTHONPATH=str(HERE),PYTHONUNBUFFERED='1',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[k]='1'
    # Relative work paths are interpreted in the directory from which the launcher was invoked.
    args=list(argv);w=None
    for i,v in enumerate(list(args)):
        if v=='--work' and i+1<len(args):w=resolve_work(args[i+1]);args[i+1]=str(w)
        elif v.startswith('--work='):w=resolve_work(v.split('=',1)[1]);args[i]='--work='+str(w)
    if w is not None and w!=(HERE/'study').resolve() and '--status' not in args and '--help' not in args and '-h' not in args:
        # Seed a *fresh* alternative working directory once; never merge archived
        # records into an existing study. The main runner then verifies every receipt.
        src=HERE/'study'/'raw'; dst=w/'raw'
        if not dst.exists() and src.is_dir():
            w.mkdir(parents=True,exist_ok=True)
            staging=w/('.seed_'+uuid.uuid4().hex)
            try:
                staging.mkdir();copied=0
                for ident in ('R01_pca_normal_001','R01_pca_normal_002','R01_normal_design_001','R01_normal_design_002','R01_fault_fit_001','R01_fault_fit_002','R01_fault_fit_003','R01_fault_select_001'):
                    old=src/ident
                    if all((old/f).is_file() for f in ('RAW.npz','RUN_INFO.json','COMPLETE.json')):
                        new=staging/ident;new.mkdir()
                        for f in ('RAW.npz','RUN_INFO.json','COMPLETE.json','STARTED.json'):
                            if (old/f).is_file():shutil.copy2(old/f,new/f)
                        copied+=1
                if copied and not dst.exists():staging.rename(dst)
            finally:
                if staging.exists():shutil.rmtree(staging)
    if not any(v=='--workers' or v.startswith('--workers=') for v in args):
        args=['--workers','8']+args
    p=subprocess.run([sys.executable,'-u',str(HERE/'run_study.py'),*args],cwd=HERE,env=env)
    return p.returncode
if __name__=='__main__':
    try:sys.exit(start(sys.argv[1:]))
    except KeyboardInterrupt:sys.exit(130)
