"""Synthetic unit and input-boundary tests; not fault performance."""
from pathlib import Path
import sys,json,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from core import check,js,canonical
from second_moment import learned_axes,fit_matrices,metric_list
from fit_windows import test_windows
from numeric_audit import exact_trace_audit
import legacy_selftest
def main(out):
    tests=[]
    def run(name,fn):
        fn();tests.append({"name":name,"passed":True})
    rng=np.random.default_rng(280906)
    R=rng.normal(size=(16,16));C=R@R.T+np.eye(16);v,U=np.linalg.eigh(C);W=(U/np.sqrt(v))@U.T
    G=W.T@C@W
    R=rng.normal(size=(16,16));D=R@R.T+np.eye(16);B=W.T@D@W
    des={"centre":np.zeros(16),"whitener":W,"G":G,"B":B,"raw_C":C,"raw_D":D}
    mu=rng.normal(size=16);H=np.outer(mu,mu)
    def rank_one():
        axes,lab,di=learned_axes(H,des,"ridge_1e4")
        for name,T,rho in metric_list(des,"ridge_1e4"):
            expected=np.linalg.solve(T,mu)
            expected=canonical(expected)/np.sqrt(canonical(expected)@G@canonical(expected))
            idx=next(i for i,labels in enumerate(lab) if name+"_max" in labels)
            check(min(np.linalg.norm(axes[idx]-expected),np.linalg.norm(axes[idx]+expected))<1e-8,"Rank-one mean equivalence")
    run("rank_one_location_second_moment_recovers_inverse_mean_direction",rank_one)
    def extrema():
        H=np.diag(np.linspace(-2,3,16));ax,lab,di=learned_axes(H,des,"ridge_1e4")
        for item in di:
            T=next(t for n,t,r in metric_list(des,"ridge_1e4") if n==item["recipe"])
            vv=rng.normal(size=(200,16))
            ratios=np.einsum("ai,ij,aj->a",vv,H,vv)/np.einsum("ai,ij,aj->a",vv,T,vv)
            e=item["generalized_eigenvalue"]
            check((ratios.max()<=e+1e-9) if item["extreme"]=="max" else (ratios.min()>=e-1e-9),"Rayleigh ordering")
    run("generalized_extrema_bound_random_rayleigh_quotients",extrema)
    def zero_mean():
        d={"whitener":np.eye(16),"G":np.eye(16),"B":np.eye(16),"raw_C":np.eye(16),"raw_D":np.eye(16)}
        h=np.zeros((16,16));h[2,2]=5
        axes,labels,_=learned_axes(h,d,"baseline")
        idx=next(i for i,l in enumerate(labels) if "unweighted_max" in l)
        check(np.isclose(abs(axes[idx,2]),1),"Variance direction missing")
    run("zero_mean_amplitude_change_yields_nonzero_informative_axis",zero_mean)
    def affine():
        T=metric_list(des,"ridge_1e4")[-1][1];h=H
        def solve(h,t):
            ev,u=np.linalg.eigh(t);w=u/np.sqrt(ev)
            l,v=np.linalg.eigh((w.T@h@w+w.T@h.T@w)/2)
            return w@v[:,-1]
        p=solve(h,T)
        S=np.diag(np.geomspace(.2,4,16));q=S@solve(S.T@h@S,S.T@T@S)
        check(min(np.linalg.norm(canonical(p)-canonical(q)),np.linalg.norm(canonical(p)+canonical(q)))<1e-8,"Metric pullback")
    run("invertible_coordinate_pullback_keeps_physical_generalized_axis",affine)
    def moments():
        y=rng.normal(size=(4,960,16))
        fit={"z":y,"run_id":np.array([2,4,7,9]),"record_suffix_start":np.array([280,490,818,960]),
             "record_suffix_samples":961-np.array([280,490,818,960])}
        nd={"z":rng.normal(size=(6,960,16))}
        h,mat,rows=fit_matrices(fit,nd,des,18)
        poison={**fit,"z":y.copy()}
        for i in range(4):
            if fit["record_suffix_samples"][i]>=160:poison["z"][i,fit["record_suffix_start"][i]-1:]=np.nan
        hp,mp,rp=fit_matrices(poison,nd,des,18)
        check(np.array_equal(h,hp),"Forbidden FIT tail was used")
        check(rows[0]["early_points"]==119 and rows[0]["late_points"]==119,"Short prefix")
        yy=(fit["z"]-des["centre"])@des["whitener"]
        terms=[]
        for i,r in enumerate(rows):
            mm=[]
            for lo,hi in [(r["early_first_sample"]-1,r["early_last_sample"]),(r["late_first_sample"]-1,r["late_last_sample"])]:
                z=yy[i,lo:hi]
                scalar=sum(np.outer(zj,zj) for zj in z)/len(z)
                mm.append(scalar)
            terms.append((mm[0]+mm[1])/2)
        check(np.allclose(np.mean(terms,0),mat["fault_second_moment"],atol=1e-12),"Independent outer-product moments")
        check(np.allclose(mat["fault_second_moment"],mat["fault_within_window_covariance"]+mat["fault_endpoint_mean_outer"],atol=1e-12),"Decomposition")
    run("equal_run_moments_tail_poison_short_prefix_and_scalar_outer_product",moments)
    def caps():
        for variant in ["baseline","ridge_1e4"]:
            axes,labels,d=learned_axes(np.diag(np.arange(16.)),des,variant)
            check(1<=len(axes)<=8 and len(d)==8,"Eight-slot cap")
            check(np.allclose(np.einsum("ai,ij,aj->a",axes,G,axes),1),"Unit actual normal variance")
    run("candidate_cap_duplicate_accounting_actual_unit_variance",caps)
    def repeated():
        dd={"whitener":np.eye(16),"G":np.eye(16),"B":np.eye(16),"raw_C":np.eye(16),"raw_D":np.eye(16)}
        a,l,di=learned_axes(np.zeros((16,16)),dd,"baseline")
        check(all(x["next_eigenvalue_gap"]==0 for x in di),"Degenerate gap not recorded")
    run("zero_contrast_degeneracy_recorded_not_claimed_unique",repeated)
    legacy_path=Path(out).with_name("LEGACY_SELFTEST.json")
    legacy_selftest.main(legacy_path)
    wins=test_windows()
    from repair_selftest import run as numeric_tests
    numeric_result=numeric_tests(Path(out).with_name("NUMERIC_REGRESSIONS.json"))
    js(Path(out),{"status":"PASS_SYNTHETIC_ONLY","new_tests":tests,"new_test_count":len(tests),
        "legacy_tests":json.loads(legacy_path.read_text())["count"],"window_checks":wins,
        "numeric_regression_tests":numeric_result["repair_count"],"new_process_results":False})
    print("S28_SELFTEST_PASS",len(tests),flush=True)
if __name__=="__main__":main(sys.argv[1])
