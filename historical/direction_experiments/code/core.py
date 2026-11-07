"""S23 numerical core; selected S22 helpers retained. NumPy only; no JIT dependency.
No simulator imports. Input units, k, ranks, candidate tie ordering and task loss unchanged.
"""
from pathlib import Path
import csv,json,datetime,hashlib,math,itertools
import numpy as np
K=.05;T=960;START=160;H=800;L=120
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def check(ok,msg):
 if not ok: raise ValueError(msg)

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def js(p,o):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def cs(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows: p.write_text('');return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def features(obj,design):
 return (obj['z']-design['centre'])@design['whitener']

def canonical(v):
 norm=float(np.linalg.norm(v))
 if norm<=1e-12:return None
 v=v/norm;j=int(np.argmax(np.abs(v)))
 if v[j]<0:v=-v
 return v

def build_axes(a,b,B):
 axes=[];labels=[];recipe_original={};details=[]
 def add(v,label):
  v=canonical(v)
  if v is None:return None
  for i,w in enumerate(axes):
   if np.max(np.abs(v-w))<=1e-10 or np.max(np.abs(v+w))<=1e-10:
    labels[i].append(label);return i
  axes.append(v);labels.append([label]);return len(axes)-1
 trace=float(np.trace(B))/len(a)
 recipes=[('unweighted',np.eye(len(a))),('shrink50',.5*B+.5*trace*np.eye(len(a))),('shrink10',.9*B+.1*trace*np.eye(len(a))),('raw',B)]
 # First add the historical choice for every recipe; then its complementary merge.
 formulas=[]
 for name,C in recipes:
  ia=np.linalg.solve(C,a);ib=np.linalg.solve(C,b);dot=float(a@ib);s=1 if dot>=0 else -1
  formulas.append((name,ia+s*ib,ia-s*ib,s,dot))
 for name,v,other,s,dot in formulas:
  recipe_original[name]=add(v,name+'_original_merge')
  details.append({'recipe':name,'original_relative_sign':s,'weighted_inner_product':dot})
 for name,v,other,s,dot in formulas:add(other,name+'_complementary_merge')
 small=list(range(len(axes)))
 stages=[add(a,'early_endpoint_axis'),add(b,'late_endpoint_axis')]
 check(len(small)>=2 and all(i is not None for i in recipe_original.values()) and all(i is not None for i in stages),'degenerate candidate pool')
 return np.array(axes),labels,small,recipe_original,tuple(sorted(set(stages))),details

def sys_name(tup):return 'Q' if tup==(-1,) else '_'.join(f'A{i:02}' for i in tup)

def sys_peaks(tup,peak,qpeak,anchor,qanchor):
 if tup==(-1,):return qpeak/qanchor
 return np.max(peak[:,list(tup)]/anchor[list(tup)],axis=1)

def sys_mask(tup,d,qd,anchor,qanchor,q):
 if tup==(-1,):return qd>q*qanchor
 return np.any(d[:,:,list(tup)]>q*anchor[list(tup)],axis=2)

def event_rows(mask,ids,fault,role,system,recent_mask=None):
 pre=mask[:,:START].any(axis=1);post=mask[:,START:];det=post.any(axis=1)
 first=np.where(det,post.argmax(axis=1),H).astype(np.int64)
 loss=np.where(pre|~det,H,first);below=(np.arange(H)[None,:]>=first[:,None])&~post
 out=[]
 for i,rid in enumerate(ids):
  row={'fault':fault,'role':role,'system':system,'run_id':int(rid),'prefault_alarm':int(pre[i]),'detected':int(det[i]),'clean_detection':int(det[i] and not pre[i]),'first_delay_samples_penalized':int(first[i]),'event_loss_samples':int(loss[i]),'occupancy_points':int(post[i].sum()),'postalarm_below_points':int(below[i].sum()),'late32h_above_points':int(mask[i,320:].sum()),'last8h_above_points':int(mask[i,800:].sum()),'terminal_above':int(mask[i,-1])}
  row['last8h_recent120_points']=int(recent_mask[i,800:].sum()) if recent_mask is not None else -1
  check(row['occupancy_points']+row['postalarm_below_points']+int(first[i])==H,'occupancy identity')
  out.append(row)
 return out

def normal_rows(mask,ids,fault,system):
 return [{'fault':fault,'system':system,'run_id':int(rid),'alarm':int(mask[i].any()),'above_points':int(mask[i].sum()),'first_alarm_sample':int(mask[i].argmax()+1) if mask[i].any() else -1} for i,rid in enumerate(ids)]

def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return centre-w,centre+w

def summarize(fr,nr):
 n=len(fr);dn=len(nr);clean=[r for r in fr if not r['prefault_alarm']];d=sum(r['detected'] for r in clean);na=sum(r['alarm'] for r in nr)
 losses=np.array([r['event_loss_samples'] for r in fr],float);normci=wilson(na,dn)
 row={'fault':fr[0]['fault'],'system':fr[0]['system'],'audit_fault_n':n,'normal_audit_n':dn,'normal_alarm_runs':na,'normal_FAR_percent':na/dn*100,'normal_Wilson_lower_percent':normci[0]*100,'normal_Wilson_upper_percent':normci[1]*100,'prefault_alarm_runs':sum(r['prefault_alarm'] for r in fr),'detected_runs':sum(r['detected'] for r in fr),'clean_detected_runs':sum(r['clean_detection'] for r in fr),'eligible_n':len(clean),'conditional_detection_percent':d/len(clean)*100 if clean else None,'event_loss_samples':float(losses.mean()),'event_loss_min':float(losses.mean()*3),'penalized_delay_min':float(np.mean([r['first_delay_samples_penalized'] for r in fr])*3),'median_penalized_delay_min':float(np.median([r['first_delay_samples_penalized'] for r in fr])*3),'occupancy_percent':sum(r['occupancy_points'] for r in fr)/(n*H)*100,'withdrawal_runs':sum(r['postalarm_below_points']>0 for r in fr),'withdrawal_points':sum(r['postalarm_below_points'] for r in fr),'late32h_occupancy_percent':sum(r['late32h_above_points'] for r in fr)/(n*640)*100,'last8h_occupancy_percent':sum(r['last8h_above_points'] for r in fr)/(n*160)*100,'last8h_recent120_percent':sum(r['last8h_recent120_points'] for r in fr)/(n*160)*100 if fr[0]['last8h_recent120_points']>=0 else None,'terminal_above_runs':sum(r['terminal_above'] for r in fr)}
 for minutes in (60,120,240,480):
  row[f'clean_detection_by_{minutes}min_percent']=sum(r['clean_detection'] and 3*r['first_delay_samples_penalized']<=minutes for r in fr)/n*100
 return row

def upper_cusum(u):
 p=np.concatenate([np.zeros((len(u),1)),np.cumsum(u-K,axis=1)],axis=1)
 return (p-np.minimum.accumulate(p,axis=1))[:,1:]

# Array recurrence performs the same floating operations in the same order as S22
# for each individual trace, without requiring numba/JIT in the user's environment.
def recurrence(u,k=K):
 u=np.asarray(u,dtype=np.float64)
 cp=np.zeros((u.shape[0],u.shape[2]));cm=cp.copy();out=np.empty_like(u)
 for t in range(u.shape[1]):
  cp=np.maximum(0.,cp+u[:,t,:]-k)
  cm=np.maximum(0.,cm-u[:,t,:]-k)
  out[:,t,:]=np.maximum(cp,cm)
 return out

def prefix_cusum(u,k=K):
 zero=np.zeros((len(u),1,u.shape[2]));result=None
 for sign in (1.,-1.):
  p=np.concatenate((zero,np.cumsum(sign*u-k,axis=1)),axis=1)
  d=(p-np.minimum.accumulate(p,axis=1))[:,1:]
  result=d if result is None else np.maximum(result,d)
 return result

def recent_positive(u,k=K,L=L):
 zero=np.zeros((len(u),1,u.shape[2]));p=np.concatenate((zero,np.cumsum(u-k,axis=1)),axis=1)
 out=(p-np.minimum.accumulate(p,axis=1))[:,1:]
 if u.shape[1]>=L:
  w=np.lib.stride_tricks.sliding_window_view(p,L+1,axis=1)
  out[:,L-1:]=p[:,L:]-w.min(axis=-1)
 return out

def recent(u,k=K,L=L):
 return np.maximum(recent_positive(u,k,L),recent_positive(-u,k,L))

def all_scores(obj,design,axis):
 y=features(obj,design);u=np.einsum('ntd,ad->nta',y,axis,optimize=True)
 d=recurrence(u,K);alt=prefix_cusum(u,K)
 check(np.allclose(d,alt,rtol=1e-9,atol=1e-8),'Independent CUSUM formula mismatch')
 err=float(np.max(np.abs(d-alt)))
 qi=(np.log1p(obj['Q'])-design['q_mean'])/design['q_sd']
 return u,d,qi,upper_cusum(qi),err

def normal_design(normal,ids,variant):
 at=np.array(ids,dtype=np.int64)-1;x=normal['z'][at];q=normal['Q'][at]
 flat=x.reshape(-1,16);centre=flat.mean(0);C=np.cov(flat,rowvar=False,ddof=1)
 v,U=np.linalg.eigh(C);floor0=max(float(np.trace(C))/16*1e-10,1e-14)
 tau=max(floor0,float(variant['relative_floor'])*float(np.trace(C))/16)
 if variant['kind']=='truncate':
  keep=v>=tau;check(keep.sum()>=2,'Retained rank below2; no automatic fallback')
  W=U[:,keep]/np.sqrt(v[keep])
 else:
  keep=np.ones(16,dtype=bool);W=(U/np.sqrt(np.maximum(v,tau)))@U.T
 y=(x-centre)@W;dim=W.shape[1];G=np.cov(y.reshape(-1,dim),rowvar=False,ddof=1)
 B=np.cov(y[:,:120].sum(1)/math.sqrt(120.),rowvar=False,ddof=1)
 be=np.linalg.eigvalsh(B);check(be[0]>np.trace(B)/dim*1e-12,'RAW temporal covariance SPD rule failed')
 D=np.cov((x[:,:120]-centre).sum(1)/math.sqrt(120.),rowvar=False,ddof=1)
 lq=np.log1p(q);sd=float(lq.std(ddof=1));check(sd>0,'Zero Q design SD')
 return {'centre':centre,'whitener':W,'B':B,'q_mean':float(lq.mean()),'q_sd':sd,'raw_C':C,'raw_D':D,'G':G,'eigenvalues':v,'keep':keep,'tau':tau}

def subset(obj,ids):
 ids=np.asarray(ids,dtype=np.int64);at=ids-1
 out={'z':obj['z'][at],'Q':obj['Q'][at],'run_id':ids.astype(np.int16)}
 for k in ('record_suffix_start','record_suffix_samples'):
  if k in obj:out[k]=obj[k][at]
 return out

def numeric_records(records,excluded=('fault','role','system')):
 fields=[k for k in records[0] if k not in excluded]
 a=np.array([[r[k] for k in fields] for r in records],dtype=np.int32)
 return fields,a

def event_category(row,marker_start,suffix):
 if row['prefault_alarm']:return 'prefault_alarm'
 marked=suffix>=160
 if not row['detected']:return 'marked_not_detected' if marked else 'unmarked_not_detected'
 if not marked:return 'unmarked_detected'
 return 'before_marker' if row['first_delay_samples_penalized']+161<marker_start else 'at_or_after_marker'
