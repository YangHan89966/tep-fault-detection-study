"""D4: controlled Gaussian AR(1) counterexamples; not a TEP-method confirmation."""
from common import *
from scipy.stats import beta
import argparse,time,math


def cp_interval(x,n):
 return (0. if x==0 else float(beta.ppf(.025,x,n-x+1)),1. if x==n else float(beta.ppf(.975,x+1,n-x)))


def innovations(phi,n,role):
 code=int(round((phi+1)*1000))
 seed=np.random.SeedSequence(CONTRACT['D4']['seed_sequence_label']+[role,code])
 return np.random.Generator(np.random.PCG64(seed)).standard_normal((n,960))


def ar_path(e,phi_normal,phi_after):
 x=np.empty_like(e);x[:,0]=e[:,0]
 for t in range(1,960):
  phi=phi_normal if t<160 else phi_after
  x[:,t]=phi*x[:,t-1]+math.sqrt(1-phi*phi)*e[:,t]
 return x


def express(u):
 return np.stack([u,(np.abs(u)-math.sqrt(2/math.pi))/math.sqrt(1-2/math.pi),(u*u-1)/math.sqrt(2)],axis=2)


def cumulative_ratio(phi,L=120,squared=False):
 p=phi*phi if squared else phi
 return float(1+2*sum((1-j/L)*p**j for j in range(1,L)))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/D4');args=ap.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=True);design=CONTRACT['D4']
 ncal=design['n_calibration'];neval=design['n_evaluation_per_scenario']
 scenarios=design['scenarios'];phis=sorted(set(s['phi_normal'] for s in scenarios))
 rows=[];nr=[];record_arrays=[];record_meta=[];normal_record_arrays=[];normal_meta=[];checks=[];theory=[]
 scalar_count=0;start=time.monotonic()
 for phi in phis:
  cal=ar_path(innovations(phi,ncal,1),phi,phi)
  calstats=recurrence(express(cal));peaks=calstats.max(axis=1);threshold=np.sort(peaks,axis=0)[design['calibration_order_1based']-1]
  np.savez_compressed(out/f'normal_calibration_phi_{phi:+.1f}.npz',peaks=peaks,threshold=threshold,normal_phi=phi)
  del calstats,cal
  e=innovations(phi,neval,2);normal=ar_path(e,phi,phi)
  nvalues=express(normal);nstats=recurrence(nvalues);nm=normal_arrays(nstats,threshold);null=event_arrays(nstats,threshold)
  for j,fam in enumerate(FAMILIES):
   x=int(nm['alarm'][:,j].sum());lo,hi=cp_interval(x,neval)
   nr.append(dict(phi_normal=phi,family=fam,n=neval,normal_alarms=x,normal_far_percent=100*x/neval,
    cp_low_percent=100*lo,cp_high_percent=100*hi,threshold=float(threshold[j]),null_post_success_percent=100*float(null['success'][:,j].mean())))
   normal_meta.append([phi,j]);normal_record_arrays.append(np.stack([nm['alarm'][:,j],null['pre'][:,j],null['success'][:,j],null['loss'][:,j]],axis=1))
  for rowi in [0,1]:
   assert np.array_equal(nstats[rowi],scalar_recurrence(nvalues[rowi]));scalar_count+=nstats[rowi].size
  theory.append(dict(phi=phi,point_mean=0,point_variance=1,lag1_signed=phi,lag1_standardized_square=phi*phi,
   sum_variance_ratio_L120_signed=cumulative_ratio(phi),sum_variance_ratio_L120_squared=cumulative_ratio(phi,squared=True)))
  for sci,sc in enumerate(scenarios):
   if sc['phi_normal']!=phi:continue
   after=sc['phi_fault'];exact_reversal=phi!=0 and after==-phi and sc['post_mean']==0 and sc['post_scale']==1
   if exact_reversal:
    signal=normal.copy();signal[:,160:]*=np.where(np.arange(800)%2==0,-1.,1.)[None,:]
   elif after==phi:signal=normal.copy()
   else:signal=ar_path(e,phi,after)
   signal[:,160:]=signal[:,160:]*sc['post_scale']+sc['post_mean']
   v=express(signal);stats=recurrence(v);fm=event_arrays(stats,threshold)
   assert np.array_equal(v[:,:160],nvalues[:,:160])
   if exact_reversal:
    assert np.array_equal(v[:,:,1:],nvalues[:,:,1:])
    assert np.array_equal(stats[:,:,1:],nstats[:,:,1:])
    for name in fm:assert np.array_equal(fm[name][:,1:],null[name][:,1:])
    checks.append(dict(scenario=sci,phi_normal=phi,phi_fault=after,identical_amplitude_input_values=neval*960*2,identical_amplitude_cumulative_values=neval*960*2,
     claim='Exact shared-innovation coupling; each marginal distribution N(0,1); no evidence of sign-reversal can exist in these identical even-transformed paths.'))
   if sc['post_mean']==0 and sc['post_scale']==1 and after==phi:assert np.array_equal(stats,nstats)
   assert np.array_equal(stats[0],scalar_recurrence(v[0]));scalar_count+=stats[0].size
   for j,fam in enumerate(FAMILIES):
    success=int(fm['success'][:,j].sum());lo,hi=cp_interval(success,neval)
    rr=dict(scenario_id=f'G{sci+1:02}',**sc,family=fam,n=neval,successes=success,success_percent=100*success/neval,
     success_cp_low_percent=100*lo,success_cp_high_percent=100*hi,prefault_alarms=int(fm['pre'][:,j].sum()),
     prefault_percent=100*float(fm['pre'][:,j].mean()),mean_penalized_delay_samples=float(fm['loss'][:,j].mean()),
     success_within40_samples_percent=100*float(((fm['success'][:,j]==1)&(fm['delay'][:,j]<=40)).mean()),
     success_within160_samples_percent=100*float(((fm['success'][:,j]==1)&(fm['delay'][:,j]<=160)).mean()),
     normal_far_percent=100*float(nm['alarm'][:,j].mean()),threshold=float(threshold[j]),exact_amplitude_coupling=exact_reversal)
    rows.append(rr);record_meta.append([sci,j]);record_arrays.append(np.stack([fm[k][:,j] for k in ['pre','detected','success','delay','loss','occupancy']],axis=1))
   print('D4',f'G{sci+1:02}',phi,'->',after,sc['kind'],'complete',round(time.monotonic()-start,1),flush=True)
  del normal,e,nvalues,nstats
 write_csv(out/'ALL_SCENARIOS.csv',rows);write_csv(out/'NORMAL_EVALUATION.csv',nr);write_csv(out/'POPULATION_IDENTITIES.csv',theory)
 np.savez_compressed(out/'RUN_RECORDS.npz',keys=np.asarray(record_meta,dtype=np.int16),key_fields=np.array(['scenario_zero_index','family_index']),families=np.array(FAMILIES),
  records=np.asarray(record_arrays,dtype=np.int16),fields=np.array(['prefault','detected','success','delay_samples','penalized_loss_samples','post_occupancy_points']))
 np.savez_compressed(out/'NORMAL_RUN_RECORDS.npz',keys=np.asarray(normal_meta),records=np.asarray(normal_record_arrays,dtype=np.int16),
  fields=np.array(['normal_alarm','prefault','null_post_success','null_post_penalized_loss_samples']))
 write_json(out/'EXACT_AMPLITUDE_CHECKS.json',checks)
 audit=dict(status='COMPLETED_CONTROLLED_MODEL_NOT_TEP_CONFIRMATION',scenarios=len(scenarios),method_rows=len(rows),
  normal_models=len(phis),n_calibration=ncal,n_evaluation=neval,scalar_cumulative_values_verified=scalar_count,
  exact_sign_reversal_pairs=len(checks),all_prespecified_scenarios_reported=True,original_TEP_model_fitting=False,
  noise_realizations_shared_across_scenarios_with_same_normal_phi=True,known_normal_scales=True,
  reported_CP_intervals='conditional on one realized calibration threshold and model; not full calibration/learning uncertainty',
  wall_seconds=time.monotonic()-start)
 write_json(out/'AUDIT.json',audit)
 if out==ROOT/'results/D4':write_json(ROOT/'audit/D4_AUDIT.json',audit)
 print('D4 COMPLETE',len(rows),round(time.monotonic()-start,1),flush=True)
 for r in rows:
  if r['phi_normal']==-.6 and r['phi_fault']==.6 or r['kind']!='dependence' and r['phi_normal']==0:print(r['scenario_id'],r['family'],r['success_percent'],r['mean_penalized_delay_samples'],flush=True)
if __name__=='__main__':main()
