"""All synthetic, full panel dimensions; independent scalar first-crossing / signed
discordance and frequency-weighted bootstrap QA. Never reads or writes study results."""
import sys,tempfile,csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import inference as I

def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def direct(t,h):
 f=np.flatnonzero(t>h);q=f[f>=160]
 anyalarm=bool(len(f));pref=anyalarm and f[0]<160
 first=1+int(f[0]) if anyalarm else 0
 post=1+int(q[0]) if len(q) else 0
 clean=bool(len(q)) and not pref
 loss= (post-161)*3 if clean else 2400
 return dict(any_alarm=int(anyalarm),prefault=int(pref),clean40=int(clean),clean8=int(clean and loss<=480),clean2=int(clean and loss<=120),loss_min=loss,first=first,first_post=post)
def main():
 I._expected() # validate protocol + identity file bytes only, no actual study access
 h=(np.arange(20,dtype=float).reshape(5,4)*.09)+1
 ids=[['SOFTWARE_ONLY_%d_%04d'%(n,i+1) for i in range(n)] for n in (400,200)]
 roles=['normal','fault'];inds={r:i for i,r in enumerate(roles)}
 names={'normal':('any_alarm','clean40'),'fault':('clean40','clean8','clean2','prefault','loss_min')}
 inv={'normal_48h_FAR':'any_alarm','normal_clean_floor':'clean40','fault_clean40':'clean40','fault_clean8':'clean8','fault_clean2':'clean2','fault_prefault_alarm':'prefault','fault_mean_penalized_loss':'loss_min'}
 with tempfile.TemporaryDirectory(prefix='f9-inference-software-only-') as tmp:
  z=Path(tmp);arrays={};orig={};sens={}
  for role,ns in zip(roles,(400,200)):
   mm=np.lib.format.open_memmap(z/(role+'.npy'),mode='w+',dtype=np.float64,shape=(ns,960,5,4))
   for i in range(ns):
    for j in range(5):
     for a in range(4):
      # 160th is the last pre-onset record; post delays 0,40,41,160,161,799.
      off=[0,40,41,160,161,799][(i+j+2*a)%6]
      val=(1+(i*(a+1)+3*j)%25)/15 if role=='fault' else (1+(i*(a+3)+j)%41)/40
      mm[i,160+off,j,a]=h[j,a]*val
      mm[i,159,j,a]=h[j,a]*(1.1 if (i+3*j+a)%79==0 else 1.0 if i%7==0 else .15)
   mm.flush();arrays[role]=mm
  h5=np.array([[sorted(np.max(arrays['normal'][:,:,j,a],axis=1))[379] for a in range(4)] for j in range(5)])
  for role,n in zip(roles,(400,200)):
   for loc,threshold in [(orig,h),(sens,h5)]:
    for nm in names[role]+('first','first_post'):
     loc[(role,nm)]=np.zeros((n,5,4),dtype=int)
    for i in range(n):
     for j in range(5):
      for a in range(4):
       ev=direct(arrays[role][i,:,j,a],threshold[j,a])
       for nm in names[role]+('first','first_post'):loc[role,nm][i,j,a]=ev[nm]
  # Physically reorder the incoming arrays and ID sequence, preserve frozen order in oracle.
  inpaths=[]
  for ii,(role,n) in enumerate(zip(roles,(400,200))):
   q=z/(role+'_reversed.npy');x=np.lib.format.open_memmap(q,mode='w+',dtype=np.float64,shape=(n,960,5,4));x[:]=arrays[role][::-1];x.flush();del x;inpaths.append(q)
  out=z/'software_results';d=I._impl(inpaths[0],inpaths[1],h,out,ids[0][::-1],ids[1][::-1],ids,True)
  assert d['status']=='SOFTWARE_QA_ONLY' and d['no_scientific_claim']
  er=read(out/'PER_RUN.csv');assert len(er)==12000
  for row in er:
   role=row['role'];n=inds[role];i=int(row['case_id'].split('_')[-1])-1;j=int(row['replicate'])-1;a=int(row['monitor'][1])-1
   for nm in names[role]+('first','first_post'):assert int(row[nm])==orig[role,nm][i,j,a]
  mxerr=0.;pairs=read(out/'PAIRED_CONTRASTS.csv');assert len(pairs)==14
  rng=np.random.default_rng(2026110406);weights={}
  for role,n in zip(roles,(400,200)):
   ix=rng.integers(n,size=(20000,n),dtype=np.int32)
   w=np.zeros((20000,n),np.int16);np.add.at(w,(np.arange(20000)[:,None],ix),1);weights[role]=w
  for row in pairs:
   role=row['role'];name=inv[row['endpoint']];a=0 if row['pool']=='moment' else 2; qual=orig[role,name]
   delta=qual[:,:,a]-qual[:,:,a+1];factor=100 if name!='loss_min' else 1
   point=factor*delta.sum()/delta.size;assert abs(point-float(row['ci_minus_post']))<1e-12
   dist=(weights[role]@delta.sum(1))*(factor/delta.size)
   qq=np.quantile(dist,[.025,.975]);err=max(abs(qq-np.array([float(row['lower95']),float(row['upper95'])])));mxerr=max(mxerr,float(err));assert err<1e-10
  flags=d['decision']
  assert flags['detection_lower95_gt_0']==(float(d['primary_detection']['lower95'])>0)
  assert flags['normal_risk_upper95_le_5pp']==(float(d['normal_risk']['upper95'])<=5.)
  assert flags['supports_gain_without_material_excess_of_normal_alarms']==all([flags['normal_risk_upper95_le_5pp'],flags['detection_lower95_gt_0']])
  for rr in read(out/'LEAVE_ONE_OUT.csv'):
   j=int(rr['replicate'])-1;a=0 if rr['pool']=='moment' else 2;nn=inv[rr['endpoint']];x=orig[rr['role'],nn];delta=x[:,:,a]-x[:,:,a+1];scale=1 if nn=='loss_min' else 100
   np.testing.assert_allclose(float(rr['omit_this_replicate_ci_minus_post']),scale*np.delete(delta,j,axis=1).mean(),rtol=0,atol=1e-11)
  count_ties=set()
  for rr in read(out/'MATCHED_5PCT.csv'):
   j=int(rr['replicate'])-1;a=int(rr['monitor'][1])-1;r=rr['role'];nn=inv[rr['endpoint']];n=len(ids[inds[r]])
   assert float(rr['h'])==h5[j,a] and int(rr['numerator_sum'])==sens[r,nn][:,j,a].sum()
   xmax=arrays['normal'][:,:,j,a].max(1); nt=int(sum(xmax==h5[j,a]));na=int(sum(xmax>h5[j,a]))
   assert int(rr['normal_peaks_equal_to_h'])==nt and int(rr['actual_normal_crossings'])==na and na<=20
   if nt>1:count_ties.add((j,a))
  assert len(count_ties)>0 and any(int(rr['actual_normal_crossings'])<20 for rr in read(out/'MATCHED_5PCT.csv'))
  for r in read(out/'MATCHED_5PCT_CONTRASTS.csv'):
   x=sens[r['role'],inv[r['endpoint']]];a=0 if r['arms']=='M1-M2' else 2;delt=x[:,:,a]-x[:,:,a+1];j=r['replicate'];me=np.mean(delt if j=='mean_5' else delt[:,int(j)-1])*100
   assert abs(float(r['ci_minus_post_pp'])-me)<1e-10
  qa={'status':'PASS','fixture':'synthetic only, 400N/200F, 960 rows, five replicates, four arms; matched normal ties, exact equality at original threshold, early crossing boundaries, no crossing, pre-fault exclusion, and deliberately permuted input run order','per_run_output_rows_recounted':len(er),'original_paired_contrasts_checked':len(pairs),'paired_draws':20000,'comparison':'Scalar first crossing per trajectory/arm, separate independently computed resampling frequencies, input-ID permutation, original and descriptive sensitivity all checked. No new scientific trajectories read.','largest_bootstrap_bound_abs_diff':mxerr,'matched_threshold_models_with_peak_ties':len(count_ties),'pipeline_sha256':hashlib.sha256((ROOT/'inference.py').read_bytes()).hexdigest()}
  (ROOT/'qa/INFERENCE_QA.json').write_text(json.dumps(qa,indent=2,allow_nan=False)+'\n');print(qa)
if __name__=='__main__':main()
