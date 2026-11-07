"""Read-only checks of returned Stage B locks/arrays; no model fitting or simulator."""
from pathlib import Path
import csv,json,zipfile,hashlib,collections,datetime,sys,argparse
import numpy as np
if sys.flags.optimize:raise RuntimeError('Run without -O; verification assertions are required')
parser=argparse.ArgumentParser(description='Check saved Stage B model selection/calibration arithmetic against the original source archive. No simulator or fitting is invoked.')
parser.add_argument('--return-root',type=Path,required=True,help='Extracted original Stage B result directory')
parser.add_argument('--stage-b-payload',type=Path,required=True,help='Original Stage B payload directory')
parser.add_argument('--stage-a-release',type=Path,required=True,help='Original Stage A release directory')
parser.add_argument('--original-return-zip',type=Path,required=True,help='Original Stage B ZIP bound by the historical report')
parser.add_argument('--out',type=Path,required=True,help='New JSON report file outside all input trees')
args=parser.parse_args()
R=args.return_root.resolve(strict=True);B=args.stage_b_payload.resolve(strict=True);A=args.stage_a_release.resolve(strict=True)
report_path=args.out.resolve()
if report_path.exists():raise FileExistsError('Choose a new output report path')
if any(report_path==p or p in report_path.parents for p in (R,B,A)):raise ValueError('Report output must be outside input trees')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text('utf-8'))
def canon(o):return (json.dumps(o,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':'))+'\n').encode('ascii')
def tree(p,name='FILES.json'):
 m=j(p/name); seen=set()
 for e in m['entries']:
  rel=e['path'];assert rel not in seen and not Path(rel).is_absolute() and '..' not in Path(rel).parts;seen.add(rel)
  q=p/rel;assert q.is_file() and not q.is_symlink();assert q.stat().st_size==e['bytes'] and sha(q)==e['sha256'],str(q)
 assert {q.relative_to(p).as_posix() for q in p.rglob('*') if q.is_file()}==seen|{name}
 return len(seen)
zp=args.original_return_zip.resolve(strict=True)
assert sha(zp)=='30c3748e9281404055c2c8d2a1ba6820eb906a090bcfe84fe6e3bd57a03af7fa', 'An original Stage B archive is required; current distribution hashes are not substitutes'
with zipfile.ZipFile(zp) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
outer_count=tree(R);tree(B,'MANIFEST.json');tree(A,'RELEASE_MANIFEST.json')
assert sha(B/'MANIFEST.json')==sha(R/'TOOL_MANIFEST.json')
assert sha(B/'BINDING.json')==sha(R/'TOOL_BINDING.json')
lock=j(R/'ALL_MODELS_LOCK.json');binding=j(R/'BINDING.json');assert lock['binding']==binding
assert lock['configuration_count']==135 and lock['replication_count']==5
assert lock['test_trajectories_generated']==lock['test_trajectories_scored']==0
assert lock['PCA_retrained'] is False
assert sha(R/'ALL_CONFIGURATIONS.csv')==lock['configuration_table_sha256']
for name,h in binding['scientific_sources'].items(): assert sha(B/'code'/name)==h==sha(A/'code'/name)
for name,key in [('PCA_MODEL_NEWSEED_NORMAL_TRAINING.npz','PCA_sha256'),('RESIDUAL_BASIS_NEWSEED_PCA.npy','residual_basis_sha256')]:assert sha(B/'assets'/name)==binding[key]
for file,key in [('FREEZE.json','stage_a_freeze_sha256'),('EVENTS.jsonl','stage_a_event_ledger_sha256'),('COMPLETED_INDEX.csv','completed_index_sha256'),('SEED_MANIFEST.json','seed_manifest_sha256'),('EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json','experiment_protocol_sha256')]:assert sha(B/'reference'/file)==binding[key]
prev='0'*64;events=[]
for n,line in enumerate((R/'EVENTS.jsonl').read_bytes().splitlines(keepends=True)):
 assert line.endswith(b'\n');d=json.loads(line);h=d.pop('sha256');assert d['seq']==n and d['prev']==prev and hashlib.sha256(canon(d)).hexdigest()==h
 prev=h;events.append(d)
counts=collections.Counter(d['event'] for d in events)
assert counts==collections.Counter({'SESSION_START':1,'INPUTS_VERIFIED':1,'MODEL_START':5,'MODEL_LOCKED':5,'ALL_MODELS_LOCKED':1,'SESSION_END':1}),counts
assert events[0]['data']['source_binding']==binding
assert events[1]['data']['count']==4575
assert events[-2]['data']['lock_sha256']==sha(R/'ALL_MODELS_LOCK.json')
assert events[-1]['data']['status']=='ALL_135_CONFIGURATIONS_LOCKED_NO_TEST_GENERATED'
assert j(R/'ACTIVE_SESSION.json')['active'] is False
assert j(R/'SESSION_RESULT.json')['error'] is None
reconstructed=[];max_norm_error=0.;rep_files=[];checks=0;scalars=0
for rep,r in enumerate(lock['replications'],1):
 p=R/r['path'];assert rep==r['replication'] and r['configurations']==27
 assert sha(p/'FILES.json')==r['manifest_sha256'] and sha(p/'REPLICATION_LOCK.json')==r['replication_lock_sha256']
 rep_files.append(tree(p));lk=j(p/'REPLICATION_LOCK.json');assert lk['binding']==binding and lk['replication']==rep and lk['configuration_count']==27
 assert lk['PCA_retrained'] is False and lk['test_generated'] is False and lk['test_scored'] is False
 assert events[rep*2]['event']=='MODEL_START' and events[rep*2]['data']['replication']==rep
 assert events[rep*2+1]['event']=='MODEL_LOCKED' and events[rep*2+1]['data']['manifest_sha256']==sha(p/'FILES.json')
 assert datetime.datetime.fromisoformat(lk['locked_utc'])<=datetime.datetime.fromisoformat(lock['locked_utc'])
 rows=[]
 for fault in (3,9,15):
  f=p/f'F{fault:02d}';info=j(f/'FIT_INFO.json');sel=j(f/'SELECTION.json');ch=j(f/'CHOICE_BEFORE_OUTER.json')
  assert info['fault']==sel['fault']==fault and not any(info[k] for k in ['selection_used','outer_used','evaluation_used'])
  assert not sel['outer_used'] and not sel['evaluation_used'] and not ch['test_data_read'] and not ch['outer_data_read_so_far']
  assert ch['selection_sha256']==sha(f/'SELECTION.json')
  md=dict(np.load(f/'MODEL.npz',allow_pickle=False));des=dict(np.load(f/'DESIGN.npz',allow_pickle=False))
  km,ka=len(md['mean']),len(md['moment']);assert km==info['mean_count'] and ka==info['moment_count']
  assert km==ka==8; k=km+ka;c=k*3+1
  names=[f'{pool}_A{i:02d}__{fam}' for fam in ('signed','absolute','squared') for pool,n in [('mean',km),('moment',ka)] for i in range(n)]+['Q_CUSUM']
  assert names==sel['candidate_names']
  for axes in md.values():
   v=np.einsum('ad,de,ae->a',axes,des['G'],axes);err=float(np.abs(v-1).max());max_norm_error=max(err,max_norm_error);assert err<1e-8
  for a in f.glob('*.npz'):
   with np.load(a,allow_pickle=False) as q:
    for key in q.files:assert np.isfinite(q[key]).all(),(a,key)
  inn=dict(np.load(f/'INNER.npz',allow_pickle=False));out=dict(np.load(f/'OUTER.npz',allow_pickle=False))
  assert inn['anchor_peaks'].shape==(32,c) and inn['inner_peaks'].shape==(59,c) and out['peaks'].shape==(259,c)
  a=np.quantile(inn['anchor_peaks'],.9,axis=0,method='linear');assert np.array_equal(a,inn['anchors']) and np.all(a>0)
  assert np.array_equal(inn['clean_detection'],(inn['detected'].astype(bool)&~inn['prefault_alarm'].astype(bool)).astype(int))
  loss=inn['event_loss_samples'];assert loss.shape==(75,c) and np.array_equal(loss,np.where(inn['clean_detection'],inn['first_delay_samples_penalized'],800))
  totals=np.sum(loss,axis=0,dtype=np.int64);choices={}
  for g,fam in enumerate(('signed','absolute','squared')):
   choices['mean_pool_'+fam]=min(range(g*k,g*k+km),key=lambda i:(int(totals[i]),i))
   choices['moment_pool_'+fam]=min(range(g*k+km,(g+1)*k),key=lambda i:(int(totals[i]),i))
  s=choices['mean_pool_signed'];choices.update(same_mean_signed_axis_absolute=k+s,same_mean_signed_axis_squared=2*k+s,Q_CUSUM=3*k)
  assert choices==sel['choices']
  for peaks,q,h,rank in [(inn['inner_peaks'],inn['inner_q'],inn['inner_thresholds'],57),(out['peaks'],out['q'],out['thresholds'],247)]:
   q2=np.sort(peaks/a,axis=0)[rank-1];assert np.array_equal(q,q2) and np.array_equal(h,a*q2);checks+=len(h)
  assert int(out['rank'])==247 and int(out['n'])==259
  # Config ordering follows the archived table, not sorted JSON keys.
  methods=[p+'_'+fam for p in ('mean_pool','moment_pool') for fam in ('signed','absolute','squared')]+['same_mean_signed_axis_absolute','same_mean_signed_axis_squared','Q_CUSUM']
  rows += [{'fault':fault,'method':m,'candidate_index':choices[m],'candidate_name':names[choices[m]],'k':.05,'threshold':float(out['thresholds'][choices[m]]),'chart':'upper' if m=='Q_CUSUM' else 'two-sided','selection_total_loss_samples':int(totals[choices[m]]),'selection_runs':75,'outer_runs':259,'outer_rank':247} for m in methods]
  sc=j(f/'SCALAR_RECURRENCE_CHECK.json');assert sc['no_test_data'] is True and sc['source_role']=='fault_selection' and sc['max_absolute_error']==0;scalars+=sc['comparisons']
 assert rows==lk['configurations'];reconstructed.extend({'replication':rep,**r} for r in rows)
import io
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(reconstructed[0]),lineterminator='\n');w.writeheader();w.writerows(reconstructed)
assert s.getvalue().encode('utf-8')==(R/'ALL_CONFIGURATIONS.csv').read_bytes()
seeds=j(B/'reference/SEED_MANIFEST.json');test=[r for r in seeds if r['stage']=='evaluation'];assert len(test)==1000
assert len({r['seed_u32'] for r in seeds})==5575 and len({r['canonical_label'] for r in seeds})==5575
assert collections.Counter((r['replication'],r['role'],r['fault']) for r in test)==collections.Counter({(0,'normal_evaluation',0):250,(0,'fault_evaluation',3):250,(0,'fault_evaluation',9):250,(0,'fault_evaluation',15):250})
report={'status':'PASS_RETURNED_MODEL_LOCK_AND_SAVED_SELECTION_CALIBRATION_CHECKS','return_sha256':sha(zp),'return_bytes':zp.stat().st_size,'outer_manifest_entries':outer_count,'rep_manifest_entries':rep_files,'ledger_events':len(events),'event_counts':dict(counts),'replications':5,'locked_configurations':135,'candidate_counts_all_15':{'mean':8,'moment':8},'threshold_values_recomputed':checks,'maximum_axis_actual_variance_error':max_norm_error,'local_reported_scalar_recurrence_comparisons':scalars,'global_lock_sha256':sha(R/'ALL_MODELS_LOCK.json'),'test_generated':0,'test_scored':0,'runtime_of_readback':{'python':sys.version,'numpy':np.__version__},'raw_process_arrays_reread_here':False,'simulator_run_here':False,'learning_repeated_here':False,'scope':'Saved model/selection/calibration arithmetic and archive/chronology consistency. Local raw input checks and scalar recurrence remain reported evidence, not independently repeated here.'}
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,indent=2))
