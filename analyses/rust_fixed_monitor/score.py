from pathlib import Path
import csv,json,hashlib,sys
import numpy as np
p=Path(__file__).resolve().parent; P=json.loads((p/'protocol.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
ident=sha(p/'FIXED_AFFINE_MAPS.npz')
assert ident=='7a20407a0fafa0028931de8d73c32b2ceae9585175e59dadd836777a97efcf66'
af=np.load(p/'FIXED_AFFINE_MAPS.npz',allow_pickle=False);ix=af['channels'].tolist().index('H_D01_moment_A01_signed')
def allrole(role):
 out=[]; diag=[]; ids=[a for a in P['cases'] if a['role']==role]
 for c in ids:
  base=p/'runs'/role/str(c['id'])
  a=np.fromfile(base.with_suffix('.bin'),dtype=np.float64).reshape(960,93);meta=json.loads(base.with_suffix('.json').read_text())
  assert np.isfinite(a).all() and meta['seed']==c['seed'] and meta['outcome']=='completed'
  for v,start in [('CI',0),('Rust_native',41)]:
   raw=np.column_stack((a[:,start:start+41],a[:,82:93]));s=raw@af['beta'][:,ix]+af['intercept'][ix]
   cp=cm=0.;chart=[]
   for x in s:
    cp=max(0.,(cp+x)-.05);cm=max(0.,(cm-x)-.05);chart.append(max(cp,cm))
   chart=np.array(chart)
   d=dict(role=role,id=c['id'],seed=c['seed'],view=v,premax=chart[:160].max(),post8max=chart[160:321].max(),post40max=chart[160:].max(),runmax=chart.max(),scoremean=s.mean(),scorevar=s.var(),x12_mv7_r=np.corrcoef(raw[:,11],raw[:,47])[0,1],x15_mv8_r=np.corrcoef(raw[:,14],raw[:,48])[0,1],loop12_width=np.ptp(raw[:,47]-(2.06*100/70)*raw[:,11]),loop15_width=np.ptp(raw[:,48]-(1.62*100/70)*raw[:,14]))
   out.append((d,chart))
 return out
def save_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
if sys.argv[1]=='cal':
 c=allrole('cal'); assert len(c)==118
 h={v:float(sorted(d['runmax'] for d,t in c if d['view']==v)[56]) for v in ['CI','Rust_native']}
 l=dict(h=h,rank=57,n=59,model_sha256=ident,protocol_sha256=sha(p/'protocol.json'))
 (p/'limits.json').write_text(json.dumps(l,indent=2));save_csv(p/'calibration.csv',[d for d,t in c]); print('LOCKED',l)
else:
 h=json.loads((p/'limits.json').read_text())['h']
 out=[]
 for role in ['normal','fault']:
  for d,path in allrole(role):
   hit=np.flatnonzero(path>h[d['view']]);first=int(hit[0]+1) if len(hit) else 0
   d['first']=first;d['any']=int(first>0);d['pre']=int(0<first<=160);d['clean40']=int(first>=161);d['clean8']=int(161<=first<=321);d['clean2']=int(161<=first<=201);d['loss']=3*(first-161) if first>=161 else 2400
   out.append(d)
 if sys.argv[1]=='audit-summary':
  with (p/'evaluation.csv').open(newline='') as archive:
   preserved=list(csv.DictReader(archive))
  assert len(preserved)==len(out) and all(row=={k:str(v) for k,v in fresh.items()} for row,fresh in zip(preserved,out)), 'Evaluation CSV would change'
 else: save_csv(p/'evaluation.csv',out)
 summary={}
 for role in ['normal','fault']:
  for v in ['CI','Rust_native']:
   g=[r for r in out if r['role']==role and r['view']==v]
   summary[role+'_'+v]={k:int(sum(r[k] for r in g)) for k in ('any','pre','clean40','clean8','clean2')};summary[role+'_'+v].update(n=len(g),mean_loss=np.mean([r['loss'] for r in g]), pooled_variance=np.mean([r['scorevar'] for r in g])+np.var([r['scoremean'] for r in g]),**{k:[float(np.min([r[k] for r in g])),float(np.median([r[k] for r in g])),float(np.max([r[k] for r in g]))] for k in ('x12_mv7_r','x15_mv8_r','loop12_width','loop15_width')})
  for metric in (['any'] if role=='normal' else ['clean40','clean8','clean2']):
   a,b=[np.array([r[metric] for r in out if r['role']==role and r['view']==v],bool) for v in ['CI','Rust_native']]
   summary[role+'_'+metric]={'CI_only':int((a&~b).sum()),'native_only':int((~a&b).sum()),'both':int((a&b).sum()),'neither':int((~a&~b).sum())}
 # Post hoc clock sensitivity; the declared first affected RECORD remains the primary origin.
 clock=json.loads((p/'CLOCK_TRIP_AUDIT.json').read_text())
 assert clock['strict_8h_last_record']==320 and clock['protocol_8h_last_record']==321
 groups={v:[r for r in out if r['role']=='fault' and r['view']==v] for v in ['CI','Rust_native']}
 for v,g in groups.items(): assert len(g)==75
 summary['clock_sensitivity']={'scope':'Post hoc sensitivity on the identical first alarms; primary event counts and original h are unchanged.',
      'protocol_delay_origin':'record 161 (first affected recorded sample); delay = (first record - 161) x 180 s.',
      'injection_origin':'scheduled IDV(9) at 8 h; instrumented step 28801, at 28800.000000004 s. Emission row j is timestamped at approximately 180*j-1 s.',
      'actual_onset_sec':clock['activation_observed_sec'],'record161_sec':clock['record_stamps_sec']['161'],
      'record320_sec':clock['record_stamps_sec']['320'],'record321_sec':clock['record_stamps_sec']['321'],
      'fault_first_alarm_on_record321_ids':{v:[int(r['id']) for r in g if r['first']==321] for v,g in groups.items()},
      'protocol_record_origin':{v:{'clean2':int(sum(161<=r['first']<=201 for r in g)), 'clean8':int(sum(161<=r['first']<=321 for r in g)), 'clean40':int(sum(r['first']>=161 for r in g))} for v,g in groups.items()},
      'strict_injection_clock':{v:{'clean2':int(sum(161<=r['first']<=200 for r in g)), 'clean8':int(sum(161<=r['first']<=320 for r in g)), 'clean40':int(sum(r['first']>=161 for r in g))} for v,g in groups.items()}}
 summary['status_diagnostic']={'study_runs':194,'status_completed':int(sum(json.loads(file.read_text())['outcome']=='completed' for role in ['cal','normal','fault'] for file in (p/'runs'/role).glob('*.json'))),
     'trip0':'A trip freezes process evolution but output continues to horizon; the binary still reports tripped. In the archived runs the recorded outcome is completed.',
     'independent_off_panel_positive_control':clock['positive_controls']}
 assert summary['status_diagnostic']['status_completed']==194
 (p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
