"""Independent synthetic-only checks; never reads scientific score outputs."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import run_coupling as candidate
from recover_noise import NOISE_SD, uniform_words

checks={}
key=0x9e3779b97f4a7c15a54ff53a5f1d36f1
ref_raw=np.random.Philox(key=key,counter=0).random_raw(2000)
ref_uniform=(ref_raw>>np.uint64(32)).astype(np.float64)/2**32
for start,count in [(0,12),(1,12),(3,19),(4,0),(23,100),(1990,10)]:
    assert np.array_equal(uniform_words(key,start,count),ref_uniform[start:start+count])
checks['counter_word_mapping_independent_reference']=True

p,cases=candidate.verify_protocol(ROOT)
keys=[candidate.replacement_key(c['run_id']) for c in cases]
assert len(keys)==len(set(keys))==799
for c,k in zip(cases,keys):
    digest=hashlib.sha256((p['study_id']+'|replacement|'+c['run_id']).encode()).hexdigest()
    assert k==int(digest[:32],16)
checks['799_replacement_keys_match_frozen_formula_and_are_disjoint']=True

# Synthetic fast channels change each row; slow channels hold for 2 or 5 rows.
# All channel events occupy distinct, nonoverlapping 12-word stream intervals.
ids=np.empty((960,41),dtype=np.int64)
cursor=7
for j in range(41):
    hold=1 if j<22 else (2 if j<36 else 5)
    for t in range(0,960,hold):
        ids[t:t+hold,j]=cursor
        cursor+=12
    if j>=22:ids[:hold,j]=-1
mask=ids>=0
word_count=int(ids.max())+12
raw_original=np.random.Philox(key=key,counter=0).random_raw(word_count)
u_original=(raw_original>>np.uint64(32)).astype(np.float64)/2**32
replacement_key=candidate.replacement_key('synthetic_engineering_case')
raw_replacement=np.random.Philox(key=replacement_key,counter=0).random_raw(word_count)
u_replacement=(raw_replacement>>np.uint64(32)).astype(np.float64)/2**32
u=np.zeros((960,41,12));v=np.zeros_like(u)
for t in range(960):
    for j in range(41):
        if mask[t,j]:
            idx=int(ids[t,j]);u[t,j]=u_original[idx:idx+12];v[t,j]=u_replacement[idx:idx+12]
actual_v=candidate.replacement_uniforms(replacement_key,ids)
assert np.array_equal(actual_v,v)
for j in range(22,41):
    held=ids[1:,j]==ids[:-1,j]
    assert np.array_equal(actual_v[1:,j][held],actual_v[:-1,j][held])
checks['replacement_uniforms_match_independent_contiguous_stream']=True
checks['held_analyzer_events_reuse_exact_replacement_components']=True
checks['initialization_events_do_not_gain_noise']=bool(np.all(actual_v[~mask]==0))

noise=(u.sum(axis=-1)-6)*NOISE_SD;noise[~mask]=0
rng=np.random.default_rng(20261030007)
latent=rng.normal(size=(960,52))
controller=latent[:,:41]+noise
native=latent+rng.normal(scale=.02,size=(960,52))
raw={'native_values':native.copy(),'fast_control_input_measurements':controller.copy()}
raw_before={k:z.copy() for k,z in raw.items()}
recovered={'original_uniforms':u,'has_noise':mask,'noise_sd':NOISE_SD.copy(),
           'noise_event_id':ids,'input_noise':noise}
beta=rng.normal(size=52);intercept=.14
score,moments=candidate.make_scores(raw,recovered,{'beta':beta,'intercept':intercept},replacement_key)

# Independent reference: construct each noisy measurement from the latent signal,
# then sum one row's 52 weighted coordinates using long-double arithmetic.
reference=np.empty((960,6))
for ai,m in enumerate([0,3,6,9,12]):
    independent_noise=(u[...,:m].sum(axis=-1)+v[...,m:].sum(axis=-1)-6)*NOISE_SD
    independent_noise[~mask]=0
    independent_view=latent.copy()
    independent_view[:,:41]+=independent_noise
    independent_view[:,41:]=native[:,41:]
    reference[:,ai]=np.sum(independent_view.astype(np.longdouble)*beta.astype(np.longdouble),axis=1)+intercept
    for j in range(41):
        unique,idx=np.unique(ids[:,j],return_index=True);idx=idx[unique>=0]
        assert moments[ai,j,0]==len(idx)
        a=noise[idx,j]/NOISE_SD[j];b=independent_noise[idx,j]/NOISE_SD[j]
        wanted=np.array([len(idx),np.sum(a),np.sum(b),np.sum(a*a),np.sum(b*b),np.sum(a*b)])
        np.testing.assert_allclose(moments[ai,j],wanted,rtol=1e-14,atol=1e-12)
reference[:,5]=np.sum(native.astype(np.longdouble)*beta.astype(np.longdouble),axis=1)+intercept
max_error=float(np.max(np.abs(score-reference)))
assert max_error<1e-10
checks['all_six_scores_match_independent_latent_signal_reference']=True
checks['noise_diagnostics_count_unique_events_not_held_repeats']=True
checks['score_reference_max_abs_error']=max_error
snapshot=native.copy();snapshot[:,:41]=controller
assert np.array_equal(score[:,4],snapshot@beta+intercept)
assert np.array_equal(score[:,5],native@beta+intercept)
assert all(np.array_equal(raw[k],raw_before[k]) for k in raw)
checks['rho1_and_actual_native_scores_bit_exact']=True
checks['source_arrays_not_mutated']=True

# Distribution stress check with disjoint whole events: each arm is a sum of 12
# independent U variables with exactly m shared. Assess expected noise variance
# and correlation without accessing any process or archived observation data.
n=250000
original_words=np.random.Philox(key=key,counter=0).random_raw(n*12)
replacement_words=np.random.Philox(key=replacement_key,counter=0).random_raw(n*12)
au=((original_words>>np.uint64(32)).astype(float)/2**32).reshape(n,12)
bv=((replacement_words>>np.uint64(32)).astype(float)/2**32).reshape(n,12)
original_noise=au.sum(axis=1)-6
metrics=[]
for m in [0,3,6,9,12]:
    z=au[:,:m].sum(axis=1)+bv[:,m:].sum(axis=1)-6
    variance=float(np.var(z,ddof=1));corr=float(np.corrcoef(original_noise,z)[0,1])
    assert abs(variance-1)<.015
    assert abs(corr-m/12)<.012
    assert abs(float(z.mean()))<.012
    metrics.append({'rho':m/12,'variance':variance,'correlation_to_original':corr,'mean':float(z.mean())})
checks['synthetic_noise_variance_and_nominal_correlation_sanity']=True

result={'status':'PASS_SYNTHETIC_NUMERICAL_CHECKS','no_scientific_score_outputs_read':True,
        'checks':checks,'synthetic_noise_sample_count':n,'synthetic_noise_metrics':metrics,
        'reviewed_files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ['code/run_coupling.py','code/recover_noise.py','PROTOCOL.json']}}
(OUT/'SYNTHETIC_NUMERICAL_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
