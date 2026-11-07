"""Independent per-contrast paired resampling from saved indices and CSV rows."""
from pathlib import Path
import argparse, csv, json, hashlib, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_integrity import verify_public_bundle
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT/'verification_results/bootstrap')
args = parser.parse_args()
_, manifest_files = verify_public_bundle(ROOT)
R = ROOT / 'results'
readcsv = lambda p: list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
rows = readcsv(R / 'PER_RUN_EVENTS.csv')
lookup = {}
for row in rows:
    key = (row['role'], row['case_id'], '/'.join(row[k] for k in ('estimator','recording','pool')))
    assert key not in lookup
    lookup[key] = row
with np.load(R / 'CROSS_BOOTSTRAP_INDICES.npz', allow_pickle=False) as z:
    ids = {role:z[prefix+'_case_ids'].tolist() for role,prefix in [('normal_evaluation','normal'),('fault9_evaluation','fault')]}
    idx = {role:z[prefix+'_indices'].copy() for role,prefix in [('normal_evaluation','normal'),('fault9_evaluation','fault')]}
rng = np.random.default_rng(20261030)
for role in ('normal_evaluation','fault9_evaluation'):
    assert len(set(ids[role])) == 120
    np.testing.assert_array_equal(idx[role],rng.integers(0,120,size=(20000,120),dtype=np.int32))
definitions=json.loads((R/'CROSS_CONTRAST_DEFINITIONS.json').read_text())['contrasts']
definitions={x['name']:x for x in definitions}
errors=[]
for contrast in readcsv(R/'CROSS_CONTRASTS.csv'):
    role=contrast['role']; endpoint=contrast['endpoint']
    terms=definitions[contrast['contrast']]['terms']
    # Difference each paired trajectory first; resample that single difference.
    difference=np.array([sum(t['weight']*float(lookup[(role,case,t['arm'])][endpoint]) for t in terms) for case in ids[role]])
    scale=1.0 if endpoint=='loss_minutes' else 100.0
    boot=np.array([difference[block].mean(axis=1) for block in np.array_split(idx[role],20)]).reshape(-1)*scale
    actual=np.r_[difference.mean()*scale,np.quantile(boot,[.025,.975],method='linear')]
    saved=np.array([float(contrast[k]) for k in ['estimate','ci95_low','ci95_high']])
    np.testing.assert_allclose(actual,saved,rtol=0,atol=1e-9)
    errors.append(float(np.max(np.abs(actual-saved))))
assert len(errors)==84
report={'status':'PASS','contrast_endpoint_rows_verified':84,'paired_draws':20000,'seed':20261030,'saved_indices_equal_regenerated_indices':True,'max_absolute_estimate_or_ci_error':max(errors),'method':'Form within-case weighted differences first, resample those differences using the saved common indices; no imports from analysis or runner code.','scope':'Conditional marginal intervals only; no claim of independent random streams or learning/calibration uncertainty coverage.'}
report['public_manifest_files_verified']=manifest_files
args.output.mkdir(parents=True, exist_ok=True)
out=args.output/'BOOTSTRAP_INDEPENDENT_VERIFICATION.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
