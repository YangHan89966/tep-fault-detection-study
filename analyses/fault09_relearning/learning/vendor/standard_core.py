"""Fixed PCA/DPCA/Q-CUSUM comparison on existing paired recordings. No simulator."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
METHODS=('PCA_T2','PCA_SPE','Q_CUSUM','DPCA_T2','DPCA_SPE')
VIEWS=('native','snapshot')
FEATURES=('full52','drop_six')
DROP=(11,14,16,47,48,51)
N=960

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_csv(p,rows):
    if not rows: raise ValueError('Refuse empty table')
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def finite(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite numerical input')
    return x
def augment(x):
    x=finite(x)
    if x.ndim not in (2,3) or x.shape[-2]<3:raise ValueError('Separate time dimension with at least 3 records required')
    return np.concatenate((x[...,2:,:],x[...,1:-1,:],x[...,:-2,:]),axis=-1)
def upper(u,k=.05):
    u=finite(u)
    if u.ndim!=2 or k<=0:raise ValueError('Upper CUSUM requires time x channels and positive k')
    state=np.zeros(u.shape[1]);out=np.empty_like(u)
    for t,row in enumerate(u):
        state=np.maximum(0.,state+row-float(k));out[t]=state
    return out
def signed(u):
    u=finite(u);up=np.zeros(u.shape[1]);lo=up.copy();out=np.empty_like(u)
    for t,row in enumerate(u):
        up=np.maximum(0.,up+row-.05);lo=np.maximum(0.,lo-row-.05);out[t]=np.maximum(up,lo)
    return out
def fit_pca(x):
    x=finite(x)
    if x.ndim!=3 or len(x)<2:raise ValueError('Normal fitting must have separate runs')
    f=x.reshape(-1,x.shape[-1]);mu=f.mean(axis=0);sd=f.std(axis=0,ddof=1)
    if (sd<=0).any():raise ValueError('Constant training variable')
    z=(f-mu)/sd;cov=z.T@z/(len(z)-1)
    vals,vec=np.linalg.eigh(cov);idx=np.argsort(vals)[::-1];vals=vals[idx];vec=vec[:,idx]
    if vals.min()<-1e-10*max(1.,vals.max()):raise ValueError('Negative covariance spectrum')
    vals=np.maximum(vals,0.);cum=np.cumsum(vals)/vals.sum();k=int(np.searchsorted(cum,.95)+1)
    if vals[k-1]<=0:raise ValueError('Invalid retained eigenvalue')
    return dict(mean=mu,scale=sd,loadings=vec[:,:k],eigenvalues=vals,components=np.asarray(k),explained=np.asarray(cum[k-1]))
def pca_statistics(x,model):
    x=finite(x);z=(x-model['mean'])/model['scale'];pc=z@model['loadings'];e=z-pc@model['loadings'].T
    k=int(model['components']);return np.sum(pc*pc/model['eigenvalues'][:k],axis=-1),np.sum(e*e,axis=-1)
def save_models(out,training):
    payload={};meta=[]
    for feature in FEATURES:
        keep=np.array([i for i in range(52) if feature=='full52' or i not in DROP])
        for vi,view in enumerate(VIEWS):
            x=training[vi][:,:,keep];base=feature+'__'+view
            pca=fit_pca(x);dpca=fit_pca(augment(x));_,q=pca_statistics(x,pca)
            lq=np.log1p(q);qm=float(lq.mean());qs=float(lq.std(ddof=1))
            if qs<=0:raise ValueError('Invalid Q scale')
            payload[base+'__keep']=keep;payload[base+'__q_mu']=np.array(qm);payload[base+'__q_sd']=np.array(qs)
            for typ,m in [('PCA',pca),('DPCA',dpca)]:
                for key,a in m.items():payload[base+'__'+typ+'__'+key]=a
            meta.append(dict(feature_set=feature,training_recording=view,training_runs=len(x),variables=len(keep),pca_components=int(pca['components']),pca_explained=float(pca['explained']),dpca_components=int(dpca['components']),dpca_explained=float(dpca['explained']),q_log_mean=qm,q_log_sd=qs,q_k=.05))
    np.savez_compressed(out,**payload);return meta

def channel_specs():
    return [dict(feature_set=f,training_recording=s,input_recording=t,method=m,regime=('adapted' if s==t else 'fixed_cross')) for f in FEATURES for s in VIEWS for t in VIEWS for m in METHODS]

def statistics_one(views,models):
    """960x40, order exactly channel_specs. Fixed source scalers applied to both inputs."""
    out=np.zeros((N,40));col=0
    for feature in FEATURES:
        for source in VIEWS:
            b=feature+'__'+source
            p={k:models[b+'__PCA__'+k] for k in ('mean','scale','loadings','eigenvalues','components')}
            d={k:models[b+'__DPCA__'+k] for k in ('mean','scale','loadings','eigenvalues','components')}
            for target in VIEWS:
                x=finite(views[target])
                if x.shape!=(N,52):raise ValueError('Record shape must be 960x52')
                x=x[:,models[b+'__keep']]
                t2,q=pca_statistics(x,p);dt2,dq=pca_statistics(augment(x),d)
                u=(np.log1p(q)-models[b+'__q_mu'])/models[b+'__q_sd']
                out[:,col]=t2;out[:,col+1]=q;out[:,col+2]=upper(u[:,None])[:,0]
                out[2:,col+3]=dt2;out[2:,col+4]=dq;col+=5
    return finite(out)
def thresholds(maxima,rank):
    x=finite(maxima)
    if x.ndim!=2 or x.shape[1]!=40 or not 1<=rank<=len(x):raise ValueError('Bad calibration allocation')
    q=np.sort(x,axis=0)[rank-1]
    specs=channel_specs();out=np.empty(40)
    for i,s in enumerate(specs):
        src=dict(s,input_recording=s['training_recording'],regime='adapted')
        out[i]=q[specs.index(src)]
    if (out<=0).any():raise ValueError('Nonpositive limit')
    return out

def event_arrays(mask):
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=3 or m.shape[1:]!=(N,40):raise ValueError('Bad packed alarm shape')
    pre=m[:,:160].any(1);post=m[:,160:];det=post.any(1);delay=np.where(det,post.argmax(1)*3,2400);clean=det&~pre
    return {'normal_alarm':m.any(1),'pre_alarm':pre,'post_detected':det,'clean_2h':clean&(delay<=120),'clean_8h':clean&(delay<=480),'clean_40h':clean,'delay_minutes':delay,'loss_minutes':np.where(clean,delay,2400),'post_point_exceedance_percent':100*post.mean(1)}

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,c-h)*100,min(100.,c+h)*100

def summarize(mask,cases,h,panel):
    roles=np.array([c['role'] for c in cases]);nix=roles=='normal_evaluation';fix=roles=='fault9_evaluation'
    ev=event_arrays(mask);n=int(nix.sum());nf=int(fix.sum());rows=[]
    for j,s in enumerate(channel_specs()):
        r=dict(panel=panel,**s,normal_n=n,fault_n=nf,threshold=float(h[j]))
        for key in ('normal_alarm','pre_alarm','post_detected','clean_2h','clean_8h','clean_40h'):
            ix=nix if key=='normal_alarm' else fix;count=int(ev[key][ix,j].sum());den=n if key=='normal_alarm' else nf
            r[key+'_n']=count;r[key+'_percent']=100*count/den
            lo,hi=wilson(count,den);r[key+'_lo95']=lo;r[key+'_hi95']=hi
        r['mean_loss_minutes']=float(ev['loss_minutes'][fix,j].mean());r['post_point_exceedance_percent']=float(ev['post_point_exceedance_percent'][fix,j].mean());rows.append(r)
    return rows,ev

def contrasts(ev,cases,panel,draws=20000):
    roles=np.array([c['role'] for c in cases]);ni=np.flatnonzero(roles=='normal_evaluation');fi=np.flatnonzero(roles=='fault9_evaluation')
    rng=np.random.Generator(np.random.PCG64(2026100633+(0 if panel=='S25_primary' else 1)));ibn=rng.integers(len(ni),size=(draws,len(ni)));ibf=rng.integers(len(fi),size=(draws,len(fi)))
    specs=channel_specs();rows=[]
    for f in FEATURES:
        for m in METHODS:
            def index(source,target):
                return specs.index(dict(feature_set=f,training_recording=source,input_recording=target,method=m,regime='adapted' if source==target else 'fixed_cross'))
            pairs=[('adapted_snapshot_minus_native',index('snapshot','snapshot'),index('native','native')),
                   ('fixed_native_model_snapshot_minus_native_input',index('native','snapshot'),index('native','native')),
                   ('fixed_snapshot_model_native_minus_snapshot_input',index('snapshot','native'),index('snapshot','snapshot'))]
            for label,a,b in pairs:
                for key in ('normal_alarm','clean_2h','clean_8h','clean_40h','loss_minutes','post_point_exceedance_percent'):
                    ix,ib=(ni,ibn) if key=='normal_alarm' else (fi,ibf)
                    d=ev[key][ix,a].astype(float)-ev[key][ix,b].astype(float);factor=100 if key in ('normal_alarm','clean_2h','clean_8h','clean_40h') else 1
                    boots=d[ib].mean(1)*factor;lo,hi=np.quantile(boots,[.025,.975])
                    rows.append(dict(panel=panel,feature_set=f,method=m,contrast=label,endpoint=key,estimate=float(d.mean()*factor),lo95=float(lo),hi95=float(hi),unit='minutes' if key=='loss_minutes' else 'percentage_points',n=len(ix),draws=draws,scope='marginal, conditional, follow-up'))
    return rows

def check_views(raw,expected_fault):
    n=finite(raw['native_values']);s=finite(raw['fast_control_input_measurements']);phase=raw['phase'];t=finite(raw['native_time_seconds']);idv=raw['diagnostic_IDV']
    if n.shape!=(960,52) or s.shape!=(960,41) or phase.shape!=(960,4):raise ValueError('Raw recording dimensions')
    if not np.all(phase[:,3]==3) or not np.array_equal(phase[:,0],np.arange(1,961)*180):raise ValueError('Raw recording phase')
    if t.shape!=(961,) or not np.allclose(t,np.arange(961)*180.,atol=1e-5,rtol=0):raise ValueError('Raw recording times')
    if idv.shape[0]!=960 or idv.shape[1]<9:raise ValueError('Fault flag dimensions')
    want=np.zeros_like(idv)
    if expected_fault==9:want[160:,8]=1
    elif expected_fault!=0:raise ValueError('Only Fault 9 permitted')
    if not np.array_equal(idv,want):raise ValueError('Fault flag mismatch')
    snap=n.copy();snap[:,:41]=s
    return {'native':n,'snapshot':snap}

def reference_check(views,expected,beta,intercept,limits):
    y=np.stack([views[v]@beta+intercept for v in VIEWS],axis=1);e=finite(expected)
    if y.shape!=e.shape or not np.allclose(y,e,rtol=1e-10,atol=1e-8):raise ValueError('Historical raw-to-scalar mismatch')
    a=signed(y)>limits;b=signed(e)>limits
    if not np.array_equal(a,b):raise ValueError('Historical Signed raw-to-alarm mismatch')
    return float(np.max(np.abs(y-e)))
