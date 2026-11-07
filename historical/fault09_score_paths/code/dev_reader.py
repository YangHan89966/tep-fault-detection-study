"""Hash-bound development-only reader with historical pandas metadata compatibility.
Never opens old final/evaluation pickle members. F18 only interprets first250 runs.
This is not a safe general-purpose loader for arbitrary pickle.
"""
from pathlib import Path
import io,hashlib,json,pickle,pickletools,zipfile
import numpy as np
import pandas as pd
VARS=[f"xmeas_{i}" for i in range(1,42)]+[f"xmv_{i}" for i in range(1,12)]
COLS=["faultNumber","simulationRun","sample"]+VARS
ALLOWED = {('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),
 ('pandas._libs.internals','_unpickle_block'),('numpy._core.numeric','_frombuffer'),
 ('numpy.core.numeric','_frombuffer'),('numpy','dtype'),('builtins','slice'),
 ('pandas.core.indexes.base','_new_Index'),('pandas.core.indexes.base','Index'),
 ('numpy._core.multiarray','_reconstruct'),('numpy.core.multiarray','_reconstruct'),
 ('numpy','ndarray'),('pandas.core.indexes.range','RangeIndex'),
 ('pandas','DataFrame'),('pandas','Index'),('pandas','RangeIndex'),('pandas._libs.arrays','__pyx_unpickle_NDArrayBacked'),
 ('pandas.arrays','StringArray'),('pandas','StringDtype')}
def old_column_string_dtype(storage="python", na_value=float("nan")):
 # These hash-pinned caches contain pandas StringDtype('python', nan) only
 # for the 55 nonmissing column labels. pandas 2.2 lacks its second argument.
 # Normalize that label metadata, never scientific numeric array contents.
 if storage != "python" or not isinstance(na_value, float) or not np.isnan(na_value):
  raise pickle.UnpicklingError("Unexpected column-label StringDtype metadata")
 return pd.StringDtype(storage="python")
class ColumnLabels:
 """Decode only the known 55 nonmissing column labels from newer pandas metadata."""
 def __setstate__(self, state):
  if not isinstance(state, tuple) or len(state)!=2:
   raise pickle.UnpicklingError("Unexpected StringArray label state")
  labels=state[1]
  if not isinstance(labels,np.ndarray) or labels.shape!=(55,) or list(labels)!=COLS:
   raise pickle.UnpicklingError("Label adapter accepts only exact scientific column names")
  self.values=labels

def old_column_array(klass, checksum, state):
 if klass is not pd.arrays.StringArray or checksum!=82904607 or state is not None:
  raise pickle.UnpicklingError("Unexpected column array reducer")
 return ColumnLabels()

def old_metadata_index(cls, attrs):
 if isinstance(attrs.get('data'),ColumnLabels):
  if cls is not pd.Index or set(attrs)!={'data','name'}:
   raise pickle.UnpicklingError("Unexpected column index schema")
  return pd.Index(attrs['data'].values,dtype=object,name=attrs['name'])
 from pandas.core.indexes.base import _new_Index
 return _new_Index(cls,attrs)

class DataUnpickler(pickle.Unpickler):
 def find_class(self,module:str,name:str):
  if (module,name)==('pandas','StringDtype'): return old_column_string_dtype
  if (module,name)==('pandas._libs.arrays','__pyx_unpickle_NDArrayBacked'): return old_column_array
  if (module,name)==('pandas.core.indexes.base','_new_Index'): return old_metadata_index
  if (module,name) not in ALLOWED: raise pickle.UnpicklingError(f'Forbidden global: {module}.{name}')
  return super().find_class(module,name)
SPEC={
 'normal':('normal_validation_runs_1_250.pkl',105601801,'112c344ec6394ca48ce70e35acd1e2b55efdce9c792f933a8c9b81541e0125c2','df515b0dc80f5ba670d5bf49545c418ac8cb176d138614a08236e20c4e683715',0),
 'fault3':('fault3_development_runs_1_250.pkl',105601930,'ba11696da5ac3356b4e47ce81bf99b513ee85878084af21f9e40b8563d0ec887','76a6a7b75b9832bb92c86a305d7159261c3ae61042801cbac544c64e91c029e7',3)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def validate_array(a,fault):
 if a.shape!=(240000,55):raise ValueError('wrong shape')
 if not np.isfinite(a).all():raise ValueError('nonfinite dev values')
 if not np.all(a[:,0]==fault):raise ValueError('wrong fault')
 if not np.array_equal(a[:,1],np.repeat(np.arange(1,251),960)):raise ValueError('dev run order')
 if not np.array_equal(a[:,2],np.tile(np.arange(1,961),250)):raise ValueError('sample order')
 return a[:,3:].reshape(250,960,52).copy()
def read_dev_pickle(path,kind):
 name,size,digest,azip,fault=SPEC[kind]
 if sha(path)!=azip:raise ValueError('wrong archive '+kind)
 with zipfile.ZipFile(path) as z:
  # Explicit allowlisted member only, never iterate through scientific members.
  blob=z.read(name)
 if len(blob)!=size or hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('payload identity')
 obj=DataUnpickler(io.BytesIO(blob)).load()
 if type(obj) is not pd.DataFrame or list(obj.columns)!=COLS:raise ValueError('frame schema')
 source_buffers={hashlib.sha256(b).hexdigest() for op,b,pos in pickletools.genops(blob) if op.name in ('BYTEARRAY8','BINBYTES','BINBYTES8') and len(b)>10000}
 decoded=[hashlib.sha256(b.values.tobytes(order='A')).hexdigest() for b in obj._mgr.blocks]
 if not all(h in source_buffers for h in decoded):raise ValueError('numeric buffers changed')
 obj=obj.sort_values(['simulationRun','sample'],kind='mergesort')
 x=validate_array(obj.to_numpy(dtype=np.float64),fault)
 return x,{'source':kind,'archive_sha256':azip,'decoded_member':name,'member_bytes':size,'member_sha256':digest,'decoded_run_min':1,'decoded_run_max':250,'numeric_buffers_match':True,'final_member_opened':False}
def read_f18_prefix(path):
 h='e7eb4bd83e65f1060e23b3e3c52e9af233b12b4bd8d744a41dd4ccc289638b7c'
 if sha(path)!=h:raise ValueError('F18 archive identity')
 with zipfile.ZipFile(path) as z:
  with z.open('old_fault18_runs_1_500.npy') as f:
   version=np.lib.format.read_magic(f)
   if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
   elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
   else:raise ValueError('unsupported npy version')
   if shape!=(480000,55) or order or dtype!=np.dtype('float64'):raise ValueError('F18 prefix layout')
   b=f.read(240000*55*8)
   if len(b)!=105600000:raise ValueError('incomplete dev prefix')
   # Close here. Do not read, load or interpret remaining250-run numerical payload.
 a=np.frombuffer(b,dtype=dtype).reshape(240000,55)
 x=validate_array(a,18)
 return x,{'source':'fault18','archive_sha256':h,'decoded_member':'old_fault18_runs_1_500.npy, C-order development prefix ONLY','decoded_numeric_bytes':len(b),'development_prefix_sha256':hashlib.sha256(b).hexdigest(),'decoded_run_min':1,'decoded_run_max':250,'whole_member_crc_not_claimed':True,'archive_hash_verified_instead':True,'old_final_rows_interpreted':0}
