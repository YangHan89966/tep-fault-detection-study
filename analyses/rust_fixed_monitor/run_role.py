from pathlib import Path
import os,sys,subprocess,concurrent.futures
p=Path(__file__).resolve().parent
r=sys.argv[1]; n=int(sys.argv[2])
if r in ('normal','fault') and not (p/'limits.json').is_file():raise RuntimeError("Cannot run evaluation before locking limits")
with concurrent.futures.ThreadPoolExecutor(n) as e:
 list(e.map(lambda i:subprocess.run(['node',str(p/'driver.mjs'),r,str(n),str(i)],check=True),range(n)))
