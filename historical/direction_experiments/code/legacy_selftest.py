"""Synthetic checks, never process performance or external scientific acceptance."""
from pathlib import Path
import sys,tempfile,json
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.dont_write_bytecode=True
import numpy as np
from core import recurrence,prefix_cusum,event_rows,event_category,normal_rows,K,sha,js,check
from old_evidence import transform_parameters,transform_inputs,candidate_names,metadata
from data_io import OldMatrix

def main(output):
    results=[]
    def test(name,func):
        func();results.append({'test':name,'passed':True})
    rng=np.random.default_rng(260901);u=rng.normal(size=(15,960,3))
    def exact():check(np.allclose(recurrence(u),prefix_cusum(u),atol=1e-10,rtol=1e-10),'CUSUM')
    test('two_sided_prefix_vs_recurrence',exact)
    test('signed_whole_axis_sign_invariance',lambda:check(np.array_equal(recurrence(u),recurrence(-u)),'Sign'))
    pars=transform_parameters(u);tr=transform_inputs(u,pars);neg=transform_inputs(-u,transform_parameters(-u))
    test('AD_and_SD_axis_sign_invariance',lambda:check(np.array_equal(tr[1],neg[1]) and np.array_equal(tr[2],neg[2]),'ADSD sign'))
    test('AD_and_SD_normal_mean_zero',lambda:check(all(np.max(abs(x.mean((0,1))))<1e-12 for x in tr[1:]),'Mean'))
    test('AD_and_SD_normal_sd_one',lambda:check(all(np.max(abs(x.reshape(-1,3).std(0,ddof=1)-1))<1e-12 for x in tr[1:]),'SD'))
    def scalar_trace():
        x=rng.normal(size=(2,40,1));ref=recurrence(x);v=[]
        for seq in x:
            cp=cm=0.;line=[]
            for y in seq[:,0]:cp=max(0.,cp+y-K);cm=max(0.,cm-y-K);line.append(max(cp,cm))
            v.append(line)
        check(np.array_equal(ref[:,:,0],np.array(v)),'scalar trace')
    test('scalar_reference_CUSUM',scalar_trace)
    def prealarm():
        m=np.zeros((4,960),bool);m[0,159]=True;m[0,200]=True;m[1,160]=True;m[2,959]=True
        r=event_rows(m,np.arange(1,5),1,'test','x')
        check([x['event_loss_samples'] for x in r]==[800,0,799,800],'Full denominator')
        check([x['clean_detection'] for x in r]==[0,1,1,0],'clean detection')
        check([event_category(x,200,761) for x in r]==['prefault_alarm','before_marker','at_or_after_marker','marked_not_detected'],'phases')
    test('prealarm_miss_firstpoint_and_phase_boundaries',prealarm)
    def moments():
        rm=u.mean(1);part=rm.mean(0)**2+rm.var(0)+u.var(1).mean(0)
        check(np.allclose(part,(u*u).mean((0,1)),atol=1e-12),'moments')
    test('within_between_and_pooled_mean_identity',moments)
    def rank():
        a=np.arange(1,60,dtype=float);h=np.sort(a)[56];check(h==57 and int((a>h).sum())==2,'rank57 strict inequality')
    test('rank57of59_and_strict_threshold',rank)
    def null_reject():
        try:transform_parameters(np.zeros((3,20,1)))
        except ValueError:return
        raise AssertionError('Degenerate transform did not stop')
    test('zero_transform_variance_rejected',null_reject)
    def nan_reject():
        v=u.copy();v[0,0,0]=np.nan
        try:transform_parameters(v)
        except ValueError:return
        raise AssertionError('NaN accepted')
    test('nonfinite_normal_transform_rejected',nan_reject)
    def family_indices():
        n=candidate_names(8);check(len(n)==41 and n[-1]=='Q_CUSUM' and metadata(8,24)==('absolute_upper',0),'index')
    test('primary_and_upper_controls_index_isolation',family_indices)
    # Known normal metadata layout; forbidden final process block may contain NaN.
    with tempfile.TemporaryDirectory(prefix='s26_tests_') as tmp:
        tmp=Path(tmp);rows=[]
        for f in (1,2):
            for run in (1,2,3):
                for t in range(1,5):rows.append([f,run,t]+([np.nan]*52 if run==3 else [float(f+run+t)]*52))
        arr=np.array(rows,float)
        for dtype in ('<f8','>f8'):
            path=tmp/('toy_'+dtype.replace('<','le').replace('>','be')+'.npy');np.save(path,arr.astype(dtype))
            def allowed(path=path):
                m=OldMatrix(path,production=False,expected_faults=2,allowed_runs=2,samples=4)
                check(np.isfinite(m.read_fault(1)).all() and np.isfinite(m.read_fault(2)).all(),'forbidden rows decoded')
                check(m.access['old_final_process_rows_decoded']==0,'forbidden scope')
            test('forbidden_final_nan_is_not_decoded_'+dtype,allowed)
        def malformed(kind):
            b=arr.copy()
            if kind=='duplicate':b[1,:3]=b[0,:3]
            elif kind=='bad_allowed_value':b[0,3]=np.nan
            elif kind=='bad_dtype':b=b.astype(np.float32)
            elif kind=='fortran':b=np.asfortranarray(b)
            p=tmp/(kind+'.npy');np.save(p,b)
            try:
                m=OldMatrix(p,production=False,expected_faults=2,allowed_runs=2,samples=4);m.read_fault(1)
            except ValueError:return
            raise AssertionError('Malformed source accepted: '+kind)
        for kind in ['duplicate','bad_allowed_value','bad_dtype','fortran']:
            test('reject_'+kind,lambda kind=kind:malformed(kind))
    js(Path(output),{'status':'PASS_SYNTHETIC_IMPLEMENTATION_CHECKS_ONLY','tests':results,'count':len(results),'no_process_performance_evidence':True})
    print('SELFTEST_PASS',len(results),flush=True)
if __name__=='__main__':main(sys.argv[1])
