"""Exact-identity numeric decoding and frozen historical monitoring. No pickle execution."""
from pathlib import Path
import hashlib, pickletools, json, csv
import numpy as np

PICKLES={
 '880d4ed850578c23d6f7db0fc65ff95477d70c0669e8f0e984a1be345c2e185d':('train',0,1,500,500),
 '112c344ec6394ca48ce70e35acd1e2b55efdce9c792f933a8c9b81541e0125c2':('test',0,1,250,960),
 'd33390993bb2028dfd4407e7db4a9bbe684feb51cf1e513e5f44fa060442ec40':('test',0,251,250,960),
 'ba11696da5ac3356b4e47ce81bf99b513ee85878084af21f9e40b8563d0ec887':('test',3,1,250,960)}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**23),b''):h.update(b)
 return h.hexdigest()
def loadz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def read_literal(path):
 """Only decode reviewed literal numeric buffers in four hash-pinned archives."""
 blob=Path(path).read_bytes();hs=hashlib.sha256(blob).hexdigest()
 if hs not in PICKLES:raise ValueError('Unrecognized pickle identity; refusing to decode')
 mode,fid,first,n,T=PICKLES[hs];N=n*T
 blocks=[a for o,a,p in pickletools.genops(blob) if o.name=='BYTEARRAY8']
 if mode=='train':
  assert sorted(map(len,blocks))==[432,1000000,108000000]
  place=np.frombuffer(next(b for b in blocks if len(b)==432),dtype='<i8')
  assert np.array_equal(place,np.r_[0,1,np.arange(3,55)])
  a=np.frombuffer(next(b for b in blocks if len(b)==108000000),dtype='<f8').reshape(54,N).T
  faults=a[:,0];ids=a[:,1];samples=np.frombuffer(next(b for b in blocks if len(b)==1000000),dtype='<i4');x=a[:,2:]
 else:
  assert list(map(len,blocks))==[1920000,101760000,424,1920000]
  place=np.frombuffer(blocks[2],dtype='<i8');assert np.array_equal(place,np.r_[1,np.arange(3,55)])
  meta=np.frombuffer(blocks[0],dtype='<i4').reshape(2,N).T
  a=np.frombuffer(blocks[1],dtype='<f8').reshape(53,N).T
  faults=meta[:,0];samples=meta[:,1];ids=a[:,0];x=a[:,1:]
 assert np.all(faults==fid)
 assert np.array_equal(ids,np.repeat(np.arange(first,first+n),T))
 assert np.array_equal(samples,np.tile(np.arange(1,T+1),n))
 assert x.shape==(N,52) and np.isfinite(x).all()
 return x.reshape(n,T,52).copy(),np.arange(first,first+n),dict(source_sha256=hs,decoder='literal numeric buffers; no pickle execution',fault=fid,first_run=first,n_runs=n,records=T)

def recurrence(x,upper=False,k=.05):
 a=np.asarray(x,float);one=a.ndim==1
 if one:a=a[None,:]
 cp=np.zeros(a.shape[0]);cm=cp.copy();out=np.empty_like(a)
 for t in range(a.shape[1]):
  cp=np.maximum(0,cp+a[:,t]-k)
  if upper:out[:,t]=cp
  else:
   cm=np.maximum(0,cm-a[:,t]-k);out[:,t]=np.maximum(cp,cm)
 return out[0] if one else out

class HistoricalMonitor:
 def __init__(self,root):
  r=Path(root);self.pca=loadz(r/'PCA_STATE.npz');self.basis=loadz(r/'RESIDUAL_BASIS.npz')['basis'];self.des=loadz(r/'DESIGN.npz')
  self.eigen=loadz(r/'PCA_DERIVED_RETAINED_EIGENVALUES.npz')['eigenvalues']
  self.models={f:loadz(r/f'F{f:02}/MODEL.npz') for f in (3,9,15)}
  self.ts={f:loadz(r/f'F{f:02}/TRANSFORM_STATE.npz') for f in (3,9,15)}
  self.choices={f:json.loads((r/f'F{f:02}/CHOICES.json').read_text()) for f in (3,9,15)}
  self.orig_h={};self.idx={3:2,9:9,15:41}
  for f,c in self.idx.items():
   out=loadz(r/f'F{f:02}/OUTER_RECORDS.npz');self.orig_h[f'F{f:02}']=float(out['thresholds'][c])
  self.orig_h['Q']=float(out['thresholds'][48])
 def score(self,x):
  """Per-run multiplication order; residual definition follows original S28."""
  x=np.asarray(x,float)
  if x.ndim==2:x=x[None,:,:]
  rawout={k:[] for k in ('F03','F09','F15','Q','T2','SPE')}
  for xx in x:
   ss=(xx-self.pca['mean'])/self.pca['std'];pc=ss@self.pca['components'].T
   z=ss@self.basis;res=ss-pc@self.pca['components'];q=np.sum(res*res,axis=1)
   y=(z-self.des['centre'])@self.des['whitener']
   u3=y@self.models[3]['mean_axes'][2];u9=y@self.models[9]['moment_axes'][1];u15=y@self.models[15]['moment_axes'][1]
   rawout['F03'].append(recurrence(u3));rawout['F09'].append(recurrence(u9))
   rawout['F15'].append(recurrence((u15*u15-self.ts[15]['moment_sq_mean'][1])/self.ts[15]['moment_sq_sd'][1]))
   rawout['Q'].append(recurrence((np.log1p(q)-float(self.des['q_mean']))/float(self.des['q_sd']),True))
   rawout['SPE'].append(q);rawout['T2'].append(np.sum(pc*pc/self.eigen,axis=1))
  return {k:np.asarray(a) for k,a in rawout.items()}

def sample_metrics(s,h,fault):
 mask=np.asarray(s)>h;pre=mask[:160].any();anypost=mask[160:].any();clean=(not pre) and anypost
 return dict(threshold=float(h),normal_sample_FAR_percent=100*float(mask.mean()) if not fault else '',
 prefault_sample_FAR_percent=100*float(mask[:160].mean()),postfault_sample_FDR_percent=100*float(mask[160:].mean()) if fault else '',
 normal_above_points=int(mask.sum()) if not fault else '',prefault_above_points=int(mask[:160].sum()),
 postfault_above_points=int(mask[160:].sum()) if fault else '',normal_run_alarm=int(mask.any()) if not fault else '',
 clean_detection=int(clean),loss_min=3*int(mask[160:].argmax()) if clean else 2400)
