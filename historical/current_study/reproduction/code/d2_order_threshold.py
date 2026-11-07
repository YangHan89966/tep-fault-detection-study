"""D2: crossed path-order and limit-origin diagnostic on fixed F9 score paths."""
from common import *
import zipfile,argparse,time


def transformed(u,tr,ai):
 return np.stack([u,(np.abs(u)-tr['abs_mean'][ai])/tr['abs_sd'][ai],
                  (u*u-tr['sq_mean'][ai])/tr['sq_sd'][ai]],axis=2)


def perm_indices(source,rid,rep,block):
 key=f"{CONTRACT['D2']['permutation_label']}|{source}|{int(rid)}|{rep}|{block}".encode('ascii')
 seed=int.from_bytes(hashlib.sha256(key).digest()[:16],'little')
 order=np.random.Generator(np.random.PCG64(seed)).permutation(800//block)
 return 160+(order[:,None]*block+np.arange(block)[None,:]).reshape(-1)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'results/D2');args=ap.parse_args()
 out=args.out;out.mkdir(parents=True,exist_ok=True)
 src=args.source or Path(CONTRACT['sources']['s30_paths']['path']);assert sha(src)==CONTRACT['sources']['s30_paths']['sha256']
 rows=[];fkeys=[];fids=[];fout=[];parity=0;scalar_checks=0;sort_checks=0
 start=time.monotonic()
 with zipfile.ZipFile(src) as z:
  for si in range(1,11):
   split=f'D{si:02}';role=json.loads(z.read(f'scores/{split}/AXIS_ROLES.json'))
   tr=read_npz(z,f'scores/{split}/TRANSFORM_AND_THRESHOLDS.npz')
   ts=read_npz(z,f'inputs/{split}/TRANSFORM_STATE.npz')
   old=read_npz(z,f'inputs/{split}/EVALUATION_RECORDS.npz')
   ai=role['axis_indices'].index(role['new_signed_axis']);na=len(role['axis_indices'])
   ci=[role['candidate_indices_family_major'][j*na+ai] for j in range(3)]
   anchor=ts['anchors'][ci];h0=tr['thresholds'][:,ai]
   data={key:read_npz(z,f'scores/{split}/{key}.npz') for key in ['normal_outer_calibration','normal_audit','fault_audit']}
   u={key:a['u'][:,:,ai].copy() for key,a in data.items()}
   vals0={key:transformed(x,tr,ai) for key,x in u.items()};s0={key:recurrence(x) for key,x in vals0.items()}
   calc=np.sort(s0['normal_outer_calibration'].max(axis=1)/anchor,axis=0)[56]*anchor
   assert np.array_equal(calc,h0),(split,calc,h0)
   f0=event_arrays(s0['fault_audit'],h0);n0=normal_arrays(s0['normal_audit'],h0)
   fields=list(old['fault_fields']);nf=list(old['normal_fields'])
   for j in range(3):
    for name,archname in [('pre','prefault_alarm'),('success','clean_detection'),('delay','first_delay_samples_penalized'),('loss','event_loss_samples'),('occupancy','occupancy_points')]:
     assert np.array_equal(f0[name][:,j],old['fault'][ci[j],:,fields.index(archname)]),(split,j,name)
     parity+=75
    assert np.array_equal(n0['alarm'][:,j],old['normal'][ci[j],:,nf.index('alarm')]);parity+=60
   for key in vals0:
    assert np.array_equal(s0[key][0],scalar_recurrence(vals0[key][0]));scalar_checks+=s0[key][0].size
   for block in [1,20,80]:
    for rep in range(20):
     vals={};sp={}
     for key,x in u.items():
      xx=x.copy();source='fault9' if key=='fault_audit' else 'normal'
      for ri,rid in enumerate(data[key]['run_id']):xx[ri,160:]=x[ri,perm_indices(source,rid,rep,block)]
      assert np.array_equal(xx[:,:160],x[:,:160])
      if rep==0:
       assert np.array_equal(np.sort(xx[:,160:],axis=1),np.sort(x[:,160:],axis=1));sort_checks+=len(xx)
      vals[key]=transformed(xx,tr,ai);sp[key]=recurrence(vals[key])
      if rep==0:
       assert np.array_equal(sp[key][0],scalar_recurrence(vals[key][0]));scalar_checks+=sp[key][0].size
     hp=np.sort(sp['normal_outer_calibration'].max(axis=1)/anchor,axis=0)[56]*anchor
     for po in [0,1]:
      ss=s0 if po==0 else sp
      for lo,h in [(0,h0),(1,hp)]:
       ff=event_arrays(ss['fault_audit'],h);nn=normal_arrays(ss['normal_audit'],h)
       for j,fam in enumerate(FAMILIES):
        rows.append(dict(split=split,block=block,repetition=rep,family=fam,path_order=po,limit_order=lo,
         threshold=float(h[j]),threshold_ratio=float(h[j]/h0[j]),success_percent=100*float(ff['success'][:,j].mean()),
         mean_loss_min=3*float(ff['loss'][:,j].mean()),prefault_percent=100*float(ff['pre'][:,j].mean()),
         normal_far_percent=100*float(nn['alarm'][:,j].mean()),
         success_within_2h_percent=100*float(((ff['success'][:,j]==1)&(ff['delay'][:,j]<=40)).mean()),
         success_within_8h_percent=100*float(((ff['success'][:,j]==1)&(ff['delay'][:,j]<=160)).mean())))
        fkeys.append([si,block,rep,j,po,lo]);fids.append(data['fault_audit']['run_id'])
        fout.append(np.stack([ff[k][:,j] for k in ['pre','detected','success','delay','loss','occupancy']],axis=1))
    print('D2',split,'block',block,'done',round(time.monotonic()-start,1),flush=True)
 write_csv(out/'PER_REPETITION.csv',rows)
 np.savez_compressed(out/'FAULT_RECORDS.npz',keys=np.asarray(fkeys,dtype=np.int16),key_fields=np.array(['split','block','repetition','family','path_order','limit_order']),
  family_names=np.array(FAMILIES),run_ids=np.asarray(fids,dtype=np.int16),records=np.asarray(fout,dtype=np.int16),
  record_fields=np.array(['prefault','detected','success','delay_samples','penalized_loss_samples','post_occupancy_points']))
 # Average randomizations within split before describing the ten overlapping splits.
 keys=['threshold','threshold_ratio','success_percent','mean_loss_min','prefault_percent','normal_far_percent','success_within_2h_percent','success_within_8h_percent']
 spl=[]
 for si in range(1,11):
  for block in [1,20,80]:
   for fam in FAMILIES:
    for po in [0,1]:
     for lo in [0,1]:
      rr=[r for r in rows if r['split']==f'D{si:02}' and r['block']==block and r['family']==fam and r['path_order']==po and r['limit_order']==lo]
      assert len(rr)==20
      spl.append(dict(split=f'D{si:02}',block=block,family=fam,path_order=po,limit_order=lo,**{k:float(np.mean([r[k] for r in rr])) for k in keys}))
 write_csv(out/'PER_SPLIT.csv',spl);summary=[]
 for block in [1,20,80]:
  for fam in FAMILIES:
   for po in [0,1]:
    for lo in [0,1]:
     rr=[r for r in spl if r['block']==block and r['family']==fam and r['path_order']==po and r['limit_order']==lo]
     line=dict(block=block,family=fam,path_order=po,limit_order=lo)
     for k in keys:line.update(summary_stat([r[k] for r in rr],k));line[k+'_mean']=float(np.mean([r[k] for r in rr]))
     summary.append(line)
 write_csv(out/'FACTORIAL_SUMMARY.csv',summary)
 contrasts=[]
 for si in range(1,11):
  for block in [1,20,80]:
   for fam in FAMILIES:
    r={(x['path_order'],x['limit_order']):x for x in spl if x['split']==f'D{si:02}' and x['block']==block and x['family']==fam}
    A,B,C,D=[r[k]['mean_loss_min'] for k in [(0,0),(1,0),(0,1),(1,1)]]
    components={'path_at_original_limit':B-A,'limit_on_original_path':C-A,'interaction':D-B-C+A,
     'total':D-A,'path_at_permuted_limit':D-C,'limit_on_permuted_path':D-B}
    assert abs(components['path_at_original_limit']+components['limit_on_original_path']+components['interaction']-components['total'])<1e-8
    contrasts.append(dict(split=f'D{si:02}',block=block,family=fam,**components))
 write_csv(out/'INTERACTION_PER_SPLIT.csv',contrasts);cr=[]
 for block in [1,20,80]:
  for fam in FAMILIES:
   rr=[r for r in contrasts if r['block']==block and r['family']==fam]
   for k in ['path_at_original_limit','limit_on_original_path','interaction','total','path_at_permuted_limit','limit_on_permuted_path']:
    a=[r[k] for r in rr];cr.append(dict(block=block,family=fam,contrast=k,mean_min=float(np.mean(a)),**summary_stat(a,'difference_min'),negative_splits=sum(v<0 for v in a),zero_splits=sum(v==0 for v in a),positive_splits=sum(v>0 for v in a)))
 write_csv(out/'INTERACTION_SUMMARY.csv',cr)
 audit=dict(status='COMPLETED_AUTHOR_DIAGNOSTIC',source_sha256=sha(src),configuration_rows=len(rows),scalar_path_values_checked=scalar_checks,
  original_integer_fields_checked=parity,exact_sorted_value_rows_checked=sort_checks,original_thresholds_exactly_restored=30,
  all_800_points_retained=True,original_source_overwritten=False,wall_seconds=time.monotonic()-start)
 write_json(out/'AUDIT.json',audit)
 if out==ROOT/'results/D2':write_json(ROOT/'audit/D2_AUDIT.json',audit)
 print('D2 COMPLETE',len(rows),round(time.monotonic()-start,1),flush=True)
 for r in summary:
  if r['block']==1 and r['family']=='signed':print('cell',r['path_order'],r['limit_order'],r['success_percent_median'],r['mean_loss_min_median'],r['normal_far_percent_median'],flush=True)
if __name__=='__main__':main()
