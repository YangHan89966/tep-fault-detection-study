"""Independent, table-level Rust endpoint check. Python 3 standard library. No regeneration."""
from pathlib import Path
import csv,json,hashlib,collections
root=Path(__file__).resolve().parent
read=lambda f: list(csv.DictReader((root/f).open(encoding='utf-8-sig')))
js=lambda f:json.loads((root/f).read_text())
h=lambda f:hashlib.sha256((root/f).read_bytes()).hexdigest()
def check(x): 
 if not x: raise ValueError('Inconsistent evidence bundle')
limits=js('limits.json');meta=read('RUN_MANIFEST.csv');evals=read('evaluation.csv');cal=read('calibration.csv');s=js('summary.json')
check(h('FIXED_AFFINE_MAPS.npz')==limits['model_sha256'] and h('protocol.json')==limits['protocol_sha256'])
view=('CI','Rust_native'); clock=js('CLOCK_TRIP_AUDIT.json');normal={};fault={}
check(limits['rank']==57 and limits['n']==59 and len({x['seed'] for x in meta})==194)
check(len(meta)==194 and all(int(x['rows'])==960 and x['outcome']=='completed' for x in meta))
roles=collections.Counter(x['role'] for x in meta);check(roles=={'normal':60,'fault':75,'cal':59})
casekeys={(x['role'],x['id'],x['seed']) for x in meta}
check(len(cal+evals)==388 and len({(x['role'],x['id'],x['seed'],x['view']) for x in cal+evals})==388)
check({(x['role'],x['id'],x['seed'],x['view']) for x in cal+evals}=={(*k,v) for k in casekeys for v in view})
for v in view:
 check(limits['h'][v]==sorted(float(x['runmax']) for x in cal if x['view']==v)[56])
 for role,store in [('normal',normal),('fault',fault)]:
  rows=[x for x in evals if x['role']==role and x['view']==v]
  store[v]={int(x['id']):int(x['first']) for x in rows}
  for x in rows:
   first=int(x['first']);check(bool(first)==(float(x['runmax'])>limits['h'][v]));check(int(x['clean40'])==(161<=first<=960));check(int(x['pre'])==(0<first<=160))
def paired(a,b,pred):
 check(set(a)==set(b));pairs=collections.Counter((pred(a[i]),pred(b[i])) for i in a)
 return dict(both=pairs[True,True],controller_input_only=pairs[True,False],native_only=pairs[False,True],neither=pairs[False,False])
pa=paired(*[fault[v] for v in view],lambda t:161<=t<=960);pn=paired(*[normal[v] for v in view],lambda t:t>0)
check(list(pa.values())==[21,49,0,5] and list(pn.values())==[0,2,2,56])
h8_record=[sum(161<=t<=clock['protocol_8h_last_record'] for t in fault[v].values()) for v in view]
h8_clock=[sum(161<=t<=clock['strict_8h_last_record'] for t in fault[v].values()) for v in view]
check(h8_record==[22,0] and h8_clock==[21,0])
check([i for i,t in fault['CI'].items() if t==321]==[53]);check(all(not 0<t<=160 for v in view for t in fault[v].values()))
check(all(s['fault_'+v]['clean40']==sum(161<=t<=960 for t in fault[v].values()) for v in view))
qa=js('native_replay.json')['replays'];check(len(qa)==2 and sum(x['elements'] for x in qa)==99840 and all(x['different']==0 for x in qa))
print(json.dumps(dict(status='PASS',unit='physical trajectories, Rust driver, historical score',from_files=['evaluation.csv','calibration.csv','RUN_MANIFEST.csv','limits.json','summary.json','CLOCK_TRIP_AUDIT.json','native_replay.json'],roles=roles,fault40=pa,normal48=pn,record_based_8h=h8_record,strict_clock_8h=h8_clock,clean2=[sum(161<=t<=201 for t in fault[v].values()) for v in view],boundary_case=53,default_recorder_replays='99,840 values, two runs, no differences as reported in replay audit',limitation='Recounts archived run-level endpoints, checks original limit identities; does not rerun raw simulator or audit unseen raw bytes.'),indent=2))
