"""Independent audit of the returned F15-only results. No kit code imported."""
import csv,hashlib,io,json,sys
from collections import defaultdict
from zipfile import ZipFile
from pathlib import Path
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kit',required=True,type=Path)
parser.add_argument('--return-zip',required=True,type=Path)
parser.add_argument('--a10',required=True,type=Path)
parser.add_argument('--a1',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
K,R,a10file,a1file=args.kit,args.return_zip,args.a10,args.a1
a1,a10,rr,kit = map(ZipFile,(a1file,a10file,R,K))
assert rr.testzip() is None and kit.testzip() is None
hash_=lambda b:hashlib.sha256(b).hexdigest()
def j(z,n):return json.loads(z.read(n).decode('utf-8-sig'))
def tab(z,n):return list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
def linked(z,prefix,table,key):
 rows=tab(z,prefix+table)
 names={prefix+r[key] for r in rows}
 assert len(names)==len(rows) and (set(z.namelist())-{prefix+table}==names)
 for r in rows:
  d=z.read(prefix+r[key]); assert int(r['bytes'])==len(d) and hash_(d)==r['sha256']
 return len(rows)
r_cnt=linked(rr,'','FILE_MANIFEST.csv','name')
p='F15_XMEAS17/'
k_cnt=linked(kit,p,'PACKAGE_MANIFEST.csv','relative_path')
bind=j(rr,'BINDING.json');state=j(rr,'STATUS.json'); prot=j(rr,'PROTOCOL.json')
assert state['status']=='COMPLETE' and state['study']==prot['study_id'] and state['hybrid_results_generated'] and not any(state[k] for k in ['model_refit','recalibrated','new_process_simulation'])
assert rr.read('PROTOCOL.json')==kit.read(p+'PROTOCOL.json')
expected={n[len(p):] for n in kit.namelist() if n.startswith(p) and n.endswith('.py') and '/' not in n[len(p):]}
assert expected==set(bind['script_hashes'])
for n,h in {**bind['script_hashes'],'PROTOCOL.json':bind['protocol_sha256'],'assets/F15_TARGET_240.csv':bind['case_catalog_sha256'],'assets/mewmc/provenance.json':bind['model_manifests']['MEWMC'],'assets/fullpool/source_sha256.json':bind['model_manifests']['full_pool']}.items():
 assert h==hash_(kit.read(p+n))
cv=tab(kit,p+'assets/F15_TARGET_240.csv')
ids={v['case_id']:v for v in cv}; assert len(ids)==240==len({v['expected_sha256'] for v in cv})
a1man=j(a1,'NEW_STUDY_RETURN/A1_results/EVALUATION_MANIFEST.json')['roles']
for role in ('normal','fault15'):
 for i,v in enumerate(a1man[role],1):
  c=ids[v['case_id']]
  assert c['expected_sha256']==v['raw_sha256'] and int(c['seed'])==v['seed'] and int(c['run_index'])==i and int(c['fault_id'])==v['fault_id'] and c['role']==v['role']
vtab=tab(rr,'VERIFIED_INPUTS.csv'); assert set(ids)=={x['case_id'] for x in vtab} and len(vtab)==240
for v in vtab:
 c=ids[v['case_id']]; assert v['verified']=='True' and all(c[k]==v[k] for k in ('expected_sha256','role','run_index'))
b=tab(rr,'PER_RUN.csv'); assert len(b)==1920
origref=tab(kit,p+'assets/original/ORIGINAL_PER_RUN.csv')
key=lambda r:(r['method'],r['arm'],r['role'],r['case_id'])
obs={}; index=defaultdict(dict)
for r in b:
 met,arm,role,id=key(r)
 assert key(r) not in obs and id in ids and role==('normal' if int(ids[id]['fault_id'])==0 else 'fault')
 f,fp=int(r['first']),int(r['first_post'])
 assert 0<=f<=960 and (fp==0 or 161<=fp<=960)
 assert ((f==0 and fp==0) or (1<=f<=160) or (f==fp>=161))
 hit=bool(f); pre=hit and f<161; clean=hit and not pre; cost=3*(f-161) if clean else 2400
 row={'clean40':int(clean),'clean8':int(clean and cost<=480),'clean2':int(clean and cost<=120),'prefault':int(pre),'any_alarm':int(hit),'loss_min':cost}
 assert all(int(r[k])==v for k,v in row.items()) and (float(r['run_max'])>float(r['h']))==hit
 row.update(first=f,first_post=fp,h=float(r['h']),run_max=float(r['run_max']));obs[key(r)]=row;index[met,arm,role][id]=row
for r in origref:
 o=obs[key(r)]
 for k in ('first','first_post','prefault','clean40','clean8','clean2','any_alarm','loss_min'):assert o[k]==int(r[k])
 assert o['h']==float(r['h'])
 if r['run_max']!='':assert np.isclose(float(r['run_max']),o['run_max'],rtol=1e-9,atol=1e-9)
auds={key(r):r for r in tab(rr,'ORIGINAL_REPLAY_AUDIT.csv')}; assert len(auds)==960
peakmax=0.; bits=0; refs={}; ep={}
def firston(x,offset=0):
 where=np.flatnonzero(x);return int(where[0])+1+offset if len(where) else 0
for side,v in [('post','native'),('ci','snapshot')]:
 for role,fault in [('normal',0),('fault',15)]:
  # Direct historical A1 alarm arrays and A10 first/peak arrays, not generated code.
  with np.load(io.BytesIO(a1.read('NEW_STUDY_RETURN/A1_results/'+v+'_F15_EVENTS.npz'))) as orig:
   col=list(orig['methods']).index('full_moment_squared')
   for i,cid in enumerate(orig[role+'_case_ids']):
    chk=('full_pool',side,role,str(cid));o=obs[chk]; spec=auds[chk]
    alarms=np.unpackbits(orig[role+'_alarm_packed'][i,:,col],bitorder='little')
    f,pf=firston(alarms),firston(alarms[160:],160); h=float(orig['thresholds'][col])
    assert (f,pf,h)==(o['first'],o['first_post'],o['h']) and str(f)==spec['original_first'] and str(pf)==spec['original_first_post']
    assert int(spec['alarm_bits_checked'])==len(alarms)==960 and spec['original_peak_first_and_pre_verified']=='alarm_mask_exact'
    bits+=len(alarms)
    assert o['h']==obs[('full_pool',side+'_with_'+('ci' if side=='post' else 'post')+'17',role,str(cid))]['h']
  n=[n for n in a10.namelist() if f'/RECORDING/rep01_{v}/F{fault:02}/' in n and n.endswith('/FIRST_ALARMS.npz')]
  assert len(n)==1
  with np.load(io.BytesIO(a10.read(n[0]))) as orig:
   col=18
   for i,cid in enumerate(orig['case_ids']):
    chk=('MEWMC',side,role,str(cid));o=obs[chk];spec=auds[chk];c=ids[str(cid)]
    assert int(c['seed'])==int(orig['seed'][i]) and int(c['run_index'])==int(orig['run_ids'][i])
    f,pf,pre=[int(orig[k][i,col]) for k in ('first_alarm_record','first_post_alarm_record','prefault_alarm')]
    assert (f,pf,pre)==(o['first'],o['first_post'],o['prefault']) and int(spec['original_first'])==f and int(spec['original_first_post'])==pf and spec['original_peak_first_and_pre_verified']=='True'
    err=abs(o['run_max']-float(orig['run_max'][i,col]))
    assert err==float(spec['max_abs_peak_error']) and err<=1e-9*(abs(o['run_max'])+1)
    assert o['h']==float(orig['thresholds'][col]) and o['h']==obs[('MEWMC',side+'_with_'+('ci' if side=='post' else 'post')+'17',role,str(cid))]['h']
    peakmax=max(peakmax,err)
armrows=tab(rr,'ARM_SUMMARY.csv');assert len(armrows)==8
summaries=[]; onsets={}
for s in armrows:
 m,a=s['method'],s['arm'];ff=list(index[m,a,'fault'].values());nn=list(index[m,a,'normal'].values())
 assert len(ff)==len(nn)==120 and len(index)==32//2
 vals={k:sum(v[k] for v in ff) for k in ('clean2','clean8','clean40')}
 vals.update(n_fault=len(ff),n_normal=len(nn),fault_prefault=sum(v['prefault'] for v in ff),normal_any=sum(v['any_alarm'] for v in nn),normal_clean_floor=sum(v['clean40'] for v in nn),mean_loss_min=sum(v['loss_min'] for v in ff)/len(ff))
 assert all(abs(float(s[k])-v)<1e-10 for k,v in vals.items())
 summaries.append(dict(method=m,arm=a,h=ff[0]['h'],**vals))
 if '_with_' in a:
  onsets[m+'/'+a]={}
  for role,grp in [('normal',nn),('fault',ff)]:
   pos=[v['first'] for v in grp if v['first']>0]
   onsets[m+'/'+a][role]={'n_any_alarm':len(pos),'positive_first_alarm_record_range':[min(pos),max(pos)],'n_post_onset_alarms':sum(v['first_post']>0 for v in grp)}
ct=tab(rr,'PAIRED_CONTRASTS.csv');assert len(ct)==42
rng=np.random.default_rng(2026110407); w=dict((t,rng.multinomial(120,[1/120]*120,size=20000)) for t in ('normal','fault'))
paired=[]; err_q=0.
for r in ct:
 m,a,b,t,k=r['method'],r['arm_A'],r['arm_B'],r['role'],r['endpoint']; aa,bb=index[m,a,t],index[m,b,t]
 assert set(aa)==set(bb) and len(aa)==120
 seq=sorted(aa)
 va=np.array([aa[i][k] for i in seq]);vb=np.array([bb[i][k] for i in seq])
 d=(va-vb)*(1 if k=='loss_min' else 100)
 assert abs(float(d.mean())-float(r['A_minus_B']))<1e-12
 qq=np.quantile((w[t]*d).sum(axis=1)/len(seq),[.025,.975])
 err_q=max(err_q,max(abs(float(r[j])-v) for j,v in zip(['resampling_p025','resampling_p975'],qq)))
 pp={'contrast':a+' minus '+b,'method':m,'role':t,'endpoint':k,'delta_'+('min' if k=='loss_min' else 'pp'):float(d.mean()),'resampling_q025_q975':list(qq)}
 if k!='loss_min':
  four=[int(np.count_nonzero((va==x)&(vb==y))) for x,y in [(1,0),(0,1),(1,1),(0,0)]]
  assert four==[int(r[j]) for j in ('A_only','B_only','both','neither')]
  pp['n_A_only_B_only_both_neither']=four
 if (t,k) in [('fault','clean40'),('normal','any_alarm')]:paired.append(pp)
assert err_q<1e-8
dif=tab(rr,'INPUT_DIFFERENCES.csv'); assert len(dif)==240 and set(ids)=={i['case_id'] for i in dif}
assert all(i['rows']=='960' and i['XMEAS17_nonidentical_records']=='960' and float(i['max_abs_XMV_difference'])==0 and i['role']==ids[i['case_id']]['role'] for i in dif)
audit={'status':'PASS','return_file':R.name,'return_sha256':hash_(R.read_bytes()),'issued_kit_sha256':hash_(K.read_bytes()),'return_payload_manifest_entries':r_cnt,'issued_kit_payload_manifest_entries':k_cnt,'bound_scripts':len(expected),'protocol_sha256':bind['protocol_sha256'],'binding':'Protocol bytes, 6 Python scripts, target catalog, 2 model manifests identical to the issued ZIP; package manifests independently checked.','physical_panel':'120 F15 and 120 normal cases with original A1 evaluation manifest IDs, seeds, SHA-256 sums, and original A10 IDs, indexes and seeds; all 240 raw hashes marked verified by the bound local reader.','per_run_rows_rederived_from_first_crossings':len(obs),'summary_rows_exact':len(armrows),'all_contrast_point_estimates_and_binary_discordances_match':len(ct),'whole_run_paired_resamples':20000,'largest_recomputed_percentile_difference':err_q,'historical_original_checks':{'direct_A10_first_pre_first_post_peak_threshold_verification_runs':480,'max_abs_A10_peak_error':peakmax,'direct_A1_alarm_masks_to_returned_first_and_first_post_runs':480,'original_A1_mask_bits_used_here_to_derive_events':bits,'A1_alarm_bits_exact_to_raw_replay_as_reported_by_original_computer':state['full_pool_archived_alarm_bits_verified']},'xmeas17_different_points_reported':sum(int(i['XMEAS17_nonidentical_records']) for i in dif),'all_XMV_identical_as_reported':True,'eight_arms':summaries,'key_paired_contrasts':paired,'hybrid_first_alarm_diagnostics':onsets,'scope':['Return archive has run-level results, not raw arrays or hybrid chart paths. Raw hashes, single-channel changes and full-curve comparison were verified by the matched local script; here independent recomputation covers endpoints, contrasts, resampling limits and source-event checks.','All 960 XMEAS17 records were swapped including rows 1-160 before the fault. clean40=0 with all prefault alarms does not mean failure to alarm after onset.','Frozen model and threshold per side; original CI vs post includes different fitted models. No causal separation, no apportioned original gain, no matched-FAR comparison; prior shared RNG segments preclude unconditional interval coverage.']}
args.output.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(audit['return_sha256'],audit['largest_recomputed_percentile_difference'],audit['historical_original_checks'])
