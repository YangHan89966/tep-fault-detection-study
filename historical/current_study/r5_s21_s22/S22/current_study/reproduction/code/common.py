from pathlib import Path
import io,csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/'audit/ANALYSIS_CONTRACT.json').read_text())
FAMILIES=('signed','absolute','squared')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read_npz(z,name):
 with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def csv_rows(data):return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 assert rows
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_json(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def recurrence(x,k=.05):
 x=np.asarray(x,dtype=np.float64)
 assert x.ndim==3 and np.isfinite(x).all()
 cp=np.zeros((len(x),x.shape[2]));cm=cp.copy();out=np.empty_like(x)
 for t in range(x.shape[1]):
  cp=np.maximum(0.,(cp+x[:,t,:])-k);cm=np.maximum(0.,(cm-x[:,t,:])-k)
  out[:,t,:]=np.maximum(cp,cm)
 return out
def scalar_recurrence(x,k=.05):
 out=np.empty_like(x)
 for j in range(x.shape[-1]):
  cp=cm=0.
  for t,u in enumerate(x[:,j]):
   cp=max(0.,(cp+float(u))-k);cm=max(0.,(cm-float(u))-k);out[t,j]=max(cp,cm)
 return out
def event_arrays(stat,threshold):
 mask=stat>np.asarray(threshold)[None,None,:]
 pre=mask[:,:160,:].any(axis=1);post=mask[:,160:,:];det=post.any(axis=1)
 delay=np.where(det,post.argmax(axis=1),800);success=det&~pre;loss=np.where(success,delay,800)
 return {'pre':pre.astype('int16'),'detected':det.astype('int16'),'success':success.astype('int16'),
  'delay':delay.astype('int16'),'loss':loss.astype('int16'),'occupancy':post.sum(axis=1).astype('int16')}
def normal_arrays(stat,threshold):
 mask=stat>np.asarray(threshold)[None,None,:]
 return {'alarm':mask.any(axis=1).astype('int16'),'above':mask.sum(axis=1).astype('int16')}
def summary_stat(values,prefix):
 a=np.asarray(values,dtype=float)
 return {prefix+'_median':float(np.median(a)),prefix+'_min':float(a.min()),prefix+'_max':float(a.max())}
