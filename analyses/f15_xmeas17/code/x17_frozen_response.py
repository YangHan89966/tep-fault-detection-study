"""Post hoc pre-chart sensitivity; usage: python x17_frozen_response.py KIT COMPLETE.zip"""
import csv, io, json, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
import score_fullpool as fp, score_mewmc as mc
with zipfile.ZipFile(sys.argv[2]) as z:
    rows = list(csv.DictReader(io.StringIO(z.read('INPUT_DIFFERENCES.csv').decode('utf-8-sig'))))
norm=[r for r in rows if r['role']=='normal_evaluation']; assert len(norm)==len({r['case_id'] for r in norm})==120 and all(int(r['rows'])==960 for r in norm)
rms=np.array([float(r['XMEAS17_RMS_difference']) for r in norm]); assert np.isfinite(rms).all() and (rms>=0).all()
out={'status':'Post hoc descriptive check; pre-chart map only. No inference of causality or endpoint effects.','normal_runs':len(norm),'records_per_run':960,'aggregation':'median across runs of the within-run RMS displacement over all 960 records','median_X17_RMS_raw':float(np.median(rms)),'sides':{}}
for side in ('post','ci'):
    stats={}; out['sides'][side]=stats
    for kind in ('mewmc','fullpool'):
        if kind=='mewmc': p,d,_=mc._state(mc._view(side))
        else: p,d,m,par,ix,_=fp._state(side)
        P=p['loadings']; M=((np.eye(52)-P@P.T)@p['basis']); M=M if bool(d['support_identity']) else M@d['support_projection']; effect=(M@d['whitener'])[16]/p['scale'][16]
        x=np.tile(p['mean'],(960,1)); h=.25; xp=x.copy();xm=x.copy();xp[:,16]+=h;xm[:,16]-=h
        if kind=='mewmc':
            diff=(mc.feature_path(xp,side)[0,0]-mc.feature_path(xm,side)[0,0])/(2*h); coeff=float(np.linalg.norm(effect)); key='norm_of_vector_displacement_per_raw_unit'
        else:
            effect=effect@m['moment'][ix]; coeff=float(effect);key='signed_presquare_projection_per_raw_unit'
            def project(raw):
                z=(raw[None,:,:]-p['mean'])/p['scale']; s=(z-(z@P)@P.T)@p['basis']; s=s if bool(d['support_identity']) else s@d['support_projection']; features=(s-d['centre'])@d['whitener'];return np.einsum('ntd,ad->nta',features,m['moment'],optimize=True)[0,0,ix]
            diff=(project(xp)-project(xm))/(2*h)
        np.testing.assert_allclose(effect,diff,rtol=1e-10,atol=1e-13)
        stats[kind]={key:coeff,'median_normal_run_RMS_prechart_displacement':float(np.median(np.abs(coeff)*rms)),'max_abs_central_difference_error':float(np.max(np.abs(diff-effect)))}
print(json.dumps(out,indent=2,allow_nan=False))
