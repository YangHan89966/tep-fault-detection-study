"""Exact OLD20 numerical access. Metadata scan cannot decode final process values."""
from pathlib import Path
import ast,struct,hashlib,io,os,csv,mmap
import numpy as np
from core import check,sha
from dev_reader import DataUnpickler,COLS,validate_array
import pandas as pd
MATRIX_NAME='faulty_testing_full_20faults.float64.npy'
MATRIX_SIZE=4224000128
MATRIX_SHA='486532e9a139461396405d2ab00dae75a653fc5f0de502f1d4c3dfb6ec99197e'
NORMAL_SIZE=105601801
NORMAL_SHA='112c344ec6394ca48ce70e35acd1e2b55efdce9c792f933a8c9b81541e0125c2'

def stable_signature(path):
 s=Path(path).stat();return (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)

def normal_cache(path):
 p=Path(path);before=stable_signature(p)
 check(p.stat().st_size==NORMAL_SIZE,'Normal file length mismatch')
 b=p.read_bytes();check(hashlib.sha256(b).hexdigest()==NORMAL_SHA,'Normal file SHA mismatch')
 obj=DataUnpickler(io.BytesIO(b)).load()
 check(type(obj) is pd.DataFrame and list(obj.columns)==COLS,'Normal cache schema')
 x=validate_array(obj.sort_values(['simulationRun','sample'],kind='mergesort').to_numpy(dtype=np.float64),0)
 check(stable_signature(p)==before,'Normal source changed while reading')
 return x,{'path':str(p),'bytes':NORMAL_SIZE,'sha256':NORMAL_SHA,'runs':250,'normal_evaluation_opened':False}

class OldMatrix:
 def __init__(self,path,*,production=True,expected_faults=20,allowed_runs=250,samples=960):
  self.path=Path(path);self.signature=stable_signature(path)
  if production:
   check(self.path.name==MATRIX_NAME and self.path.stat().st_size==MATRIX_SIZE,'Not the exact old matrix')
   print('Verifying exact OLD 4.224GB source identity. No simulation.',flush=True)
   check(sha(path)==MATRIX_SHA,'Old matrix SHA-256 mismatch')
  with self.path.open('rb') as f:
   check(f.read(6)==b'\x93NUMPY','Not NPY');v=f.read(2)
   if v[0]==1:hlen=struct.unpack('<H',f.read(2))[0]
   elif v[0] in (2,3):hlen=struct.unpack('<I',f.read(4))[0]
   else:raise ValueError('Unsupported NPY header')
   check(hlen<=65536,'Oversized NPY header')
   h=ast.literal_eval(f.read(hlen).decode('utf8' if v[0]==3 else 'latin1'));off=f.tell()
  check(set(h)=={'descr','fortran_order','shape'} and not h['fortran_order'],'C-order source required')
  shape=tuple(h['shape']);dtype=np.dtype(h['descr'])
  check(dtype.kind=='f' and dtype.itemsize==8 and len(shape)==2 and shape[1]==55,'Require55-column float64 source')
  if production:check(shape==(9600000,55),'Old source shape mismatch')
  check(off+np.prod(shape)*8==self.path.stat().st_size,'Truncated NPY source')
  self.mm=np.memmap(self.path,mode='r',dtype=dtype,offset=off,shape=shape,order='C')
  self.nf=expected_faults;self.nr=allowed_runs;self.T=samples
  self.rows=np.full((expected_faults,allowed_runs,samples),-1,dtype=np.int64);flat=self.rows.ravel();allowed_count=0
  for lo in range(0,len(self.mm),65536):
   # Explicitly only first THREE metadata columns, never entire source row.
   m=np.asarray(self.mm[lo:lo+65536,:3]);fault,run,sample=m.T
   eligible=np.isfinite(fault)&np.isfinite(run)&(fault>=1)&(fault<=expected_faults)&(run>=1)&(run<=allowed_runs)
   m=m[eligible];ridx=np.flatnonzero(eligible)+lo
   check(np.all(np.isfinite(m)) and np.all(m==np.floor(m)),'Malformed eligible metadata')
   v=m.astype(np.int64);check(np.all((v[:,2]>=1)&(v[:,2]<=samples)),'Bad eligible sample ID')
   keys=((v[:,0]-1)*allowed_runs+v[:,1]-1)*samples+v[:,2]-1
   check(len(np.unique(keys))==len(keys) and np.all(flat[keys]==-1),'Duplicate development key')
   flat[keys]=ridx;allowed_count+=len(keys)
   # Avoid keeping the entire4.2GB metadata scan resident on small PCs.
   if hasattr(self.mm._mmap,'madvise') and hasattr(mmap,'MADV_DONTNEED'):
    self.mm._mmap.madvise(mmap.MADV_DONTNEED)
  check(allowed_count==len(flat) and np.all(flat>=0),'Missing old development observations')
  check(np.all(np.diff(self.rows,axis=2)>0),'Within-run samples not in source order')
  check(stable_signature(path)==self.signature,'Old matrix changed during index construction')
  self.access={'metadata_rows_scanned':len(self.mm),'indexed_development_rows':allowed_count,'process_rows_decoded':{},'old_final_process_rows_decoded':0}
  print('OLD development index complete:',allowed_count,'rows; final process rows decoded: 0',flush=True)
 def read_fault(self,fault):
  check(1<=fault<=self.nf,'Fault outside predefined set')
  idx=self.rows[fault-1].reshape(-1)
  # Advanced index has only allowed rows and only process columns3..54.
  a=self.mm[idx[:,None],np.arange(3,55)[None,:]].astype(np.float64,copy=False)
  check(np.isfinite(a).all(),'Nonfinite permitted process value')
  if hasattr(self.mm._mmap,'madvise') and hasattr(mmap,'MADV_DONTNEED'):
   self.mm._mmap.madvise(mmap.MADV_DONTNEED)
  self.access['process_rows_decoded'][str(fault)]=self.nr*self.T
  check(stable_signature(self.path)==self.signature,'Matrix source changed')
  return a.reshape(self.nr,self.T,52)

def project(x,pca,basis):
 check(x.shape[1:]==(960,52),'Wrong process trace dimensions')
 n=len(x);out=np.empty((n,960,16));q=np.empty((n,960));err=0.
 starts=[];P=pca['components']
 for i in range(n):
  z=(x[i]-pca['mean'])/pca['std'];out[i]=z@basis
  r=z-(z@P.T)@P;q[i]=np.sum(r*r,axis=1)
  err=max(err,float(np.max(abs(r-out[i]@basis.T))))
  change=np.any(x[i,1:,:22]!=x[i,:-1,:22],axis=1)
  at=np.flatnonzero(change);starts.append(int(at[-1])+2 if len(at) else 1)
 starts=np.asarray(starts,dtype=np.int16)
 check(np.allclose(q,(out*out).sum(2),rtol=1e-7,atol=1e-7),'Residual projection/Q parity')
 return {'z':out,'Q':q,'record_suffix_start':starts,'record_suffix_samples':961-starts},err

def verify_markers(fault,projected,expected_csv):
 expected=[r for r in csv.DictReader(Path(expected_csv).open()) if int(r['fault'])==fault]
 expected=sorted(expected,key=lambda r:int(r['run']))
 check(len(expected)==250,'Continuity source coverage')
 v=np.array([int(r['first22_constant_suffix_start']) for r in expected])
 check(np.array_equal(v,projected['record_suffix_start']),'Continuity source mismatch Fault'+str(fault))
