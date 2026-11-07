"""Deterministic allocation for five disjoint downstream sets and one shared test set.
No process simulation. Read the pinned old scalar-index CSV locally, never copy its rows.
"""
from __future__ import annotations
import csv, hashlib, io
from collections import Counter
from pathlib import Path
from storage import canonical, regular, strict_json, sha

STUDY='TEP_V18_RELEARNING_R5_v1.0'
MOD=1 << 32
DEVELOPMENT_ROLES=(('normal_design',0,40),('normal_anchor',0,32),
 ('normal_inner_calibration',0,59),('normal_outer_calibration',0,259),
 ('fault_fit',3,100),('fault_selection',3,75),('fault_fit',9,100),
 ('fault_selection',9,75),('fault_fit',15,100),('fault_selection',15,75))
TEST_ROLES=(('normal_evaluation',0,250),('fault_evaluation',3,250),
 ('fault_evaluation',9,250),('fault_evaluation',15,250))


def effective(seed: int) -> int:
    if type(seed) is not int or not 0 < seed < MOD:
        raise ValueError('nonzero uint32 initialization required')
    return int(float(seed)) % MOD


def candidate(label: str,counter: int) -> tuple[int,str]:
    digest=hashlib.sha256((label+'|counter='+str(counter)).encode('ascii')).digest()
    return 1+int.from_bytes(digest[:8],'big')%(MOD-1),digest.hex()


def previous_states(release: Path) -> tuple[set[int],dict]:
    binding=strict_json(release/'LOCAL_BINDING.json')
    path=Path(binding['prior_9500_index']);regular(path);raw=path.read_bytes()
    if len(raw)!=binding['prior_9500_bytes'] or hashlib.sha256(raw).hexdigest()!=binding['prior_9500_sha256']:
        raise ValueError('Previous 9500 index differs. Do not modify or substitute it.')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not {'adapter_scalar_seed_u32','canonical_label','role'}<=set(reader.fieldnames or []):
        raise ValueError('Prior index columns')
    old=set();labels=set();counts=Counter()
    for r in reader:
        g=effective(int(r['adapter_scalar_seed_u32']))
        if g in old or r['canonical_label'] in labels:raise ValueError('Prior index duplicate')
        old.add(g);labels.add(r['canonical_label']);counts[r['role']]+=1
    if len(old)!=9500 or counts!=Counter(normal_training=500,normal_validation=250,
       normal_evaluation=250,fault_development=4250,fault_final=4250):raise ValueError('Prior 9500 count/roles')
    dvpath=release/'provenance/ORIGINAL_DV_SEED_MANIFEST.json';regular(dvpath)
    if sha(dvpath)!=binding['original_dv_seed_manifest_sha256']:raise ValueError('Original DV seed-index identity')
    dv=strict_json(dvpath);dvset=set();dvlabels=set();dvc=Counter()
    for r in dv:
        g=effective(r['seed_u32'])
        v,h=candidate(r['canonical_label'],r['collision_counter'])
        if (g!=r['effective_initial_state'] or v!=r['seed_u32'] or h!=r['derivation_sha256']
            or g in dvset or r['canonical_label'] in dvlabels):raise ValueError('Original DV seed definition/uniqueness')
        dvset.add(g);dvlabels.add(r['canonical_label']);dvc[(r['role'],r['fault'])]+=1
    expected=Counter({(a,b):n for a,b,n in DEVELOPMENT_ROLES+TEST_ROLES})
    if len(dvset)!=1915 or dvc!=expected or old & dvset:raise ValueError('Original DV roles or overlap with prior 9500')
    excluded=old|dvset
    audit={'prior_9500_index_sha256':binding['prior_9500_sha256'],'prior_9500_rows':9500,
      'original_DV_seed_manifest_sha256':sha(dvpath),'original_DV_rows':1915,
      'previous_unique_effective_states':len(excluded),
      'previous_effective_states_digest':hashlib.sha256(b''.join(g.to_bytes(4,'big') for g in sorted(excluded))).hexdigest(),
      'old_9500_seed_rows_copied':False,'old_process_data_read':False,
      'initialization_definition':'int(float(nonzero uint32)) modulo 2^32, before first TESUB7',
      'limit':'Distinct initializations do not prove independent or non-overlapping PRNG substreams.'}
    return excluded,audit


def allocate(excluded: set[int]) -> tuple[list[dict],dict]:
    if len(excluded)!=11415:raise ValueError('Require exactly 9500 + 1915 previously used distinct initializations')
    used=set(excluded);rows=[];rejected=0
    groups=[(rep,role,fault,n,'development') for rep in range(1,6) for role,fault,n in DEVELOPMENT_ROLES]
    groups.extend((0,role,fault,n,'evaluation') for role,fault,n in TEST_ROLES)
    for rep,role,fault,n,stage in groups:
        for run in range(1,n+1):
            label=f'{STUDY}|rep={rep:02d}|role={role}|fault={fault:02d}|run={run:04d}'
            for counter in range(100000):
                seed,digest=candidate(label,counter);state=effective(seed)
                if state not in used:break
                rejected+=1
            else:raise RuntimeError('Collision resolution exhausted; no random fallback')
            used.add(state)
            prefix=f'data/rep{rep:02d}' if stage=='development' else 'data/shared_test'
            rows.append({'canonical_label':label,'replication':rep,'role':role,'fault':fault,'run':run,
              'stage':stage,'relative':f'{prefix}/{role}/F{fault:02d}/R{run:04d}',
              'seed_u32':seed,'effective_initial_state':state,'collision_counter':counter,'derivation_sha256':digest})
    validate_manifest(rows)
    audit={'new_total':5575,'new_development':4575,'new_evaluation':1000,
      'new_unique_effective_states':len(used)-len(excluded),'new_previous_overlap':0,
      'new_cross_role_or_replication_overlap':0,'rejected_hash_candidates':rejected,
      'seed_manifest_sha256':hashlib.sha256(canonical(rows)).hexdigest(),
      'mapping':'SHA256(ASCII canonical_label + |counter=c); 1 + big-endian first8bytes mod(2^32-1). Counter increments only for excluded/repeated initialization.',
      'assignment_order':'rep1..5; serialized DEVELOPMENT_ROLES then run1..n; shared rep00 TEST_ROLES then run1..n',
      'generation_order':'development only: run ordinal, then replication, then serialized role order; completion order may differ',
      'test_generation':'Disabled in Stage A; later requires all five 27-configuration locks.'}
    return rows,audit


def validate_manifest(rows: list[dict]) -> None:
    if not isinstance(rows,list) or len(rows)!=5575:raise ValueError('5575-entry manifest required')
    labels=set();paths=set();states=set();counts=Counter();ordinals={}
    for r in rows:
        rep,role,fault,run,stage=(r[k] for k in ['replication','role','fault','run','stage'])
        label=f'{STUDY}|rep={rep:02d}|role={role}|fault={fault:02d}|run={run:04d}'
        prefix=f'data/rep{rep:02d}' if stage=='development' else 'data/shared_test'
        if r['canonical_label']!=label or r['relative']!=f'{prefix}/{role}/F{fault:02d}/R{run:04d}':raise ValueError('Manifest label/path')
        seed,digest=candidate(label,r['collision_counter']);g=effective(r['seed_u32'])
        if seed!=r['seed_u32'] or digest!=r['derivation_sha256'] or g!=r['effective_initial_state']:raise ValueError('Manifest derivation')
        if label in labels or r['relative'] in paths or g in states:raise ValueError('Manifest duplicate')
        labels.add(label);paths.add(r['relative']);states.add(g);key=(rep,role,fault,stage)
        counts[key]+=1;ordinals.setdefault(key,set()).add(run)
    expected=Counter({(rep,role,fault,'development'):n for rep in range(1,6) for role,fault,n in DEVELOPMENT_ROLES})
    expected.update({(0,role,fault,'evaluation'):n for role,fault,n in TEST_ROLES})
    if counts!=expected:raise ValueError('Manifest role/replication counts')
    if any(ordinals[k]!=set(range(1,n+1)) for k,n in expected.items()):raise ValueError('Manifest run ordinals')


def ordered_development(rows: list[dict]) -> list[dict]:
    order={(role,fault):i for i,(role,fault,_) in enumerate(DEVELOPMENT_ROLES)}
    return sorted((r for r in rows if r['stage']=='development'),
      key=lambda r:(r['run'],r['replication'],order[(r['role'],r['fault'])]))
