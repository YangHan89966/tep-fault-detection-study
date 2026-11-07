"""Independent recount using an empirical sweep of half-open [pre,post) intervals,
and an integer, frequency-weighted paired whole-run bootstrap (no curves.py import)."""
from pathlib import Path
import csv,json,hashlib,sys
from collections import defaultdict
import numpy as np

root=Path(__file__).resolve().parents[1];res=root/'results'
def tab(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sub(name):return tab(res/(name+'.csv'))
def key(r):return r['panel'],r['contrast'],int(r['replicate']),r['arm']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
ep=sub('evaluation_peaks');cp=sub('calibration_peaks')
ev=defaultdict(list);ca=defaultdict(list)
for x in ep:ev[key(x)].append(x)
for x in cp:ca[key(x)].append(x)
assert set(ev)==set(ca)
def fl(rows,k):return np.array([float(x[k]) for x in rows])
siteinfo={}; panels={}; raw={}; tie_e=tie_c=0
for b in sorted(ev):
 rs=sorted(ev[b],key=lambda r:r['case_id']);norm=[r for r in rs if r['role']=='normal'];fault=[r for r in rs if r['role']=='fault']; cal=sorted(ca[b],key=lambda r:r['case_id'])
 for role,rows in [('fault',fault),('normal',norm)]:
  ids=[r['case_id'] for r in rows]; assert len(ids)==len(set(ids)) and panels.setdefault((*b[:2],role),ids)==ids
 ids=[r['case_id'] for r in cal]; assert len(set(ids))==len(ids) and panels.setdefault((*b[:3],'cal'),ids)==ids
 x=fl(norm,'normal48_max'); pre=fl(fault,'pre_max'); e8=fl(fault,'post8_max'); e40=fl(fault,'post40_max');cc=fl(cal,'cal48_max')
 assert all(np.isfinite(v).all() for v in [x,pre,e8,e40,cc]) and np.all(e8<=e40)
 assert len(set(r['original_h'] for r in (rs+cal)))==1
 hh=float(norm[0]['original_h'])
 raw[b]=(x,pre,e8,e40,cc,hh)
 nc,nf,nk=len(x),len(pre),len(cc)
 siteinfo.setdefault(b[:2],[nc,nf,set(),nk])[2].add(b[2])
 tie_e+=nc-len(set(x));tie_c+=nk-len(set(cc))
 assert abs(sorted(cc)[246 if nk==259 else 189]-hh)<1e-9
 # The F9 input exports do not contain early/late maxima on the normal rows;
 # other source arms may include them. If present, enforce the full-run identity.
 for row in norm:
  if row['pre_max'] and row['post40_max']: assert float(row['normal48_max'])==max(float(row['pre_max']),float(row['post40_max']))
# Assemble-key audit vs separately created source records, without invoking their scoring routines.
scount={'eval':0,'cal':0}
allindex={'eval':{(*key(r),r['role'],r['case_id']):r for r in ep},'cal':{(*key(r),r['case_id']):r for r in cp}}
for subdir in ['confirm','v2','v2c2','f9']:
 for ty,name in [('eval','run_peaks.csv'),('cal','calibration_peaks.csv')]:
  if subdir=='f9':
   allrows=tab(root/'f9/F9_RUN_MAXIMA.csv')
   rows=[]
   for a in allrows:
    if (a['role']=='normal_calibration')!=(ty=='cal'):continue
    role='normal' if a['role']=='normal_evaluation' else 'fault'
    rows.append(dict(a,panel=a['panel'],contrast='F9_recording',replicate=1,arm='favoured' if a['recording']=='controller_input' else 'comparator',role=role,case_id=a['run_id'],cal48_max=a['normal48_max'],source_method=a['source_arm']))
  else:rows=tab(root/subdir/name)
  for r in rows:
   if subdir in ['v2','v2c2']:r['panel']='validation2'
   ky=(*key(r),r['role'],r['case_id']) if ty=='eval' else (*key(r),r['case_id'])
   v=allindex[ty][ky]
   fields=['original_h','source_method']+(['pre_max','post8_max','post40_max','normal48_max'] if ty=='eval' else ['cal48_max'])
   for nm in fields: assert v[nm]==r[nm],(ky,nm)
   scount[ty]+=1
assert scount=={'eval':len(ep),'cal':len(cp)}
# At h an individual run registers clean detection iff h lies within [pre,post).
# Count interval starts minus ends using sorted binary searches.
def active(pre,post,h):
 eligible=post>pre
 return np.searchsorted(np.sort(pre[eligible]),h,side='right')-np.searchsorted(np.sort(post[eligible]),h,side='right')
detail=sub('curves_by_model'); means=sub('curves_mean');actual=defaultdict(list);mn={}; mismatched_target=0
for r in detail:actual[(*key(r),r['kind'])].append(r)
countrows=0; pools=defaultdict(list); hmaxerr=0.
for b in raw:
 n,pr,e8,e40,c,oh=raw[b];N=len(n);F=len(pr)
 grid={'matched_evaluation':([*range(N+1)],np.r_[np.sort(n)[::-1],np.nextafter(np.min(n),-np.inf)]),'calibration_rank':([*range(1,len(c)+1)],np.sort(c)),'original':([0],np.asarray([oh]))}
 for kind,(idx,h) in grid.items():
  u=sorted(actual[b+(kind,)],key=lambda r:int(r['index'])); assert len(u)==len(idx) and list(map(lambda r:int(r['index']),u))==idx
  delta=float(np.max(np.abs(fl(u,'threshold')-h))); assert delta<1e-12;hmaxerr=max(delta,hmaxerr)
  cc=np.stack([N-np.searchsorted(np.sort(n),h,side='right'),active(pr,e40,h),active(pr,e8,h),F-np.searchsorted(np.sort(pr),h,side='right')],axis=1)
  vv=cc*(100/np.array([N,F,F,F]))
  for jj,r in enumerate(u):
   for z,k in enumerate(['normal_alarm_n','clean40_n','clean8_n','fault_prefault_n']):assert int(r[k])==cc[jj,z]
   for z,k in enumerate(['far_pct','clean40_pct','clean8_pct','prefault_pct']):assert abs(float(r[k])-vv[jj,z])<1e-11
   pools[(*b[:2],b[3],kind,idx[jj])].append(vv[jj])
  if kind=='matched_evaluation':mismatched_target+=int(np.count_nonzero(cc[:,0]!=idx))
  countrows+=len(u)
assert countrows==len(detail)
for k,v in pools.items():mn[k]=np.mean(np.asarray(v),axis=0)
assert len(mn)==len(means)
for r in means:
 ky=(r['panel'],r['contrast'],r['arm'],r['kind'],int(r['index']))
 assert np.allclose(mn[ky], [float(r[nm]) for nm in ['far_pct','clean40_pct','clean8_pct','prefault_pct']],atol=1e-11,rtol=0)
anchors=sub('matched_anchors');boots=sub('paired_contrasts')
pk=lambda r:(r['panel'],r['contrast'],int(r['nominal_anchor_pct']),r['horizon'])
bl={pk(r):r for r in boots}; assert len(bl)==len(boots)==len(anchors)==42
orig=sub('original_points')
for r in orig:
 site=(r['panel'],r['contrast']); assert np.allclose(mn[(*site,r['arm'],'original',0)], [float(r[k]) for k in ('far_pct','clean40_pct','clean8_pct','prefault_pct')],rtol=0,atol=1e-11)
for r in anchors:
 site=(r['panel'],r['contrast']);N,F,reps,C=siteinfo[site];q=int(r['nominal_anchor_pct']);k=N*q//100; hor=1 if r['horizon']=='40h' else 2
 av=mn[(*site,'favoured','matched_evaluation',k)]; bv=mn[(*site,'comparator','matched_evaluation',k)]
 wanted=[k,100*k/N,av[0],bv[0],av[hor],bv[hor],av[hor]-bv[hor],av[3],bv[3]]
 names=['k_allowed','empirical_target_pct','favoured_far_pct','comparator_far_pct','favoured_detect_pct','comparator_detect_pct','favoured_minus_comparator_pp','favoured_prefault_pct','comparator_prefault_pct']
 assert np.allclose(wanted,[float(r[c]) for c in names],rtol=0,atol=1e-11)
 assert abs(float(bl[pk(r)]['favoured_minus_comparator_pp'])-(av[hor]-bv[hor]))<1e-11
# Independent, distribution-preserving count weights. Generate the same specified run draws.
# Threshold = first sorted value at which cumulative resampled frequencies reaches N-k.
seeds=dict(zip([('confirmation','C1'),('confirmation','C3'),('confirmation','F9_recording'),('validation2','C1'),('validation2','C2'),('validation2','C3'),('recording_panel_2','F9_recording')],range(2026110403,2026110410)))
agg=[];qerr=0.
def counts(x,n):
 ans=np.zeros((len(x),n),np.int16)
 np.add.at(ans,(np.arange(len(x))[:,None],x),1)
 return ans
for site in seeds:
 N,F,reps,C=siteinfo[site];seed=seeds[site];rng=np.random.default_rng(seed);B=5000
 wnorm=counts(rng.integers(N,size=(B,N),dtype=np.int32),N);wfault=counts(rng.integers(F,size=(B,F),dtype=np.int32),F)
 sums=np.zeros((B,3,2),dtype=np.int32)
 names=(1,5,10)
 for rep in sorted(reps):
  for sign,arm in [(1,'favoured'),(-1,'comparator')]:
   n,p,e,f,_,_=raw[(*site,rep,arm)]
   order=np.argsort(n);nst=n[order]
   cw=np.cumsum(wnorm[:,order],axis=1,dtype=np.int16)
   for iq,q in enumerate(names):
    # Search on resample multiplicities, not sample np.partition.
    h=nst[np.argmax(cw>=N-N*q//100,axis=1),None]
    for i,post in enumerate([f,e]):
     sums[:,iq,i]+=sign*(((h>=p)&(h<post))*wfault).sum(axis=1,dtype=np.int32)
 quant=np.quantile(sums,[.025,.975],axis=0)*(100./(len(reps)*F))
 for i,q in enumerate(names):
  for t,hor in enumerate(['40h','8h']):
   key_=(*site,q,hor);r=bl[key_]; assert int(r['seed'])==seed and int(r['bootstrap_draws'])==B
   qerr=max(qerr,max(np.abs(np.asarray([float(r['percentile_low_pp']),float(r['percentile_high_pp'])])-quant[:,i,t])))
   if q==5:
    agg.append({'panel':site[0],'contrast':site[1],'horizon':hor,'attained_far':N*q//100/N*100,'favoured_minus_comparator_pp':float(r['favoured_minus_comparator_pp']),'paired_run_percentiles_pp':[float(z) for z in quant[:,i,t]]})
assert qerr<1e-10
audit={'status':'PASS','inputs_and_table_hashes':{p.name:sha(p) for p in sorted(res.glob('*.csv'))},'scope':'Independent count sweep for all seven paired panel/contrast sets. No import of production assembly, threshold, endpoint, aggregation, or bootstrap code. Source maxima were previously verified by extraction.','eval_rows_matched_to_source_exports':scount['eval'],'outer_calibration_rows_matched_to_exports':scount['cal'],'model_arm_count':len(raw),'per_model_operating_rows_all_four_counts_exact':countrows,'mean_operating_rows_recomputed':len(mn),'original_mean_rows_checked':len(orig),'threshold_largest_abs_error':hmaxerr,'full_grid_normal_order_statistics':{'normal_evaluation_extra_equal_values_within_arm':tie_e,'calibration_extra_equal_values_within_arm':tie_c,'rows_missing_target_normal_crossing_count':mismatched_target},'anchor_rows_reproduced':len(anchors),'bootstrap':{'draws_per_panel_contrast':B,'bootstrap_interval_rows_reproduced':len(bl),'max_percentile_endpoint_abs_diff_pp':qerr,'independent_method':'Run frequencies in separately resampled normal and fault cohorts, common across detector arms and repeats. Weighted normal empirical CDF cutoffs; integer paired sums over physical runs, division only at end. Resampling produces ties even if original maxima are distinct.'},'five_percent_nominal_anchor_results':agg,'interpretation':['Descriptive thresholds chosen only from same evaluation normals, lower endpoint of each attainable empirical-FAR bin (except uppermost bin boundary), not thresholds set independently in advance; original operating points and independent calibration-rank curves remain distinct. A separate threshold is used in every detector replicate and arm.','Confirmation: C1, C3 and the F9 recording contrast retain large positive 40h clean-detection differences at common 1%,5%,10% FAR. At 5% FAR the C3 8h difference is 2.3 pp with lower endpoint exactly zero; the 1% 8h interval crosses zero.','Validation 2 C2: post hoc 40h clean-detection differences of 1.28, 3.12 and 4.88 pp at actually attainable 0.8%, 4.8%, 10% FAR, with 95% pointwise percentile intervals crossing zero at the first two. This diagnostic is not the prespecified penalized-loss endpoint and cannot turn failure of original C2 into confirmation.','Historic shared RNG limits uncertainty interpretation. All intervals pointwise, conditional on trained models; cannot infer ROC dominance or infer a uniform 8h C3 advantage at very low FAR.']}
(root/'review/INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('PASS',countrows,len(mn),tie_e,tie_c,mismatched_target,qerr,len(raw),scount)
