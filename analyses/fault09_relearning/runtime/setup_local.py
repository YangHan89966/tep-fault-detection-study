"""Create only the package-local Windows venv for the pinned NumPy wheel."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]/'.f9-venv-win'
if sys.implementation.name!='cpython' or sys.version_info[:2]!=(3,12):
    raise SystemExit('A 64-bit CPython 3.12 installation is required.')
if not (root/'pyvenv.cfg').is_file():
    if root.exists():raise SystemExit('Local venv path already exists without pyvenv.cfg; inspect it before retrying: '+str(root))
    subprocess.run([sys.executable,'-m','venv',str(root)],check=True)
py=root/'Scripts'/'python.exe'
subprocess.run([str(py),'-c','import sys; assert sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,12)'],check=True)
subprocess.run([str(py),'-m','pip','--require-virtualenv','--disable-pip-version-check','install','--only-binary=:all:','numpy==2.3.5'],check=True)
subprocess.run([str(py),'-c','import numpy; assert numpy.__version__=="2.3.5"; print("Pinned private environment ready.")'],check=True)
