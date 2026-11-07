"""Clean detection versus run-level FAR. Python 3 + NumPy; no model fitting."""
import csv, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def assemble(root):
    root=Path(root);ev=[];ca=[]
    cols='panel contrast fault replicate arm source_method role case_id original_h pre_max post8_max post40_max normal48_max'.split()
    cc='panel contrast fault replicate arm source_method case_id original_h cal48_max'.split()
    for sub in ['confirm','v2','v2c2']:
        for file,fields,dest in [('run_peaks.csv',cols,ev),('calibration_peaks.csv',cc,ca)]:
            for rr in read(root/sub/file):
                row={k:rr[k] for k in fields}
                if sub!='confirm':row['panel']='validation2'
                dest.append(row)
    for r in read(root/'f9/F9_RUN_MAXIMA.csv'):
        core=dict(panel=r['panel'],contrast='F9_recording',fault=9,replicate=1,arm='favoured' if r['recording']=='controller_input' else 'comparator',source_method=r['source_arm'],case_id=r['run_id'],original_h=r['original_h'])
        if r['role']=='normal_calibration':
            ca.append(dict(core,cal48_max=r['normal48_max']))
        else:
            ev.append(dict(core,role='normal' if r['role']=='normal_evaluation' else 'fault',**{k:r[k] for k in ['pre_max','post8_max','post40_max','normal48_max']}))
    result=root/'results';result.mkdir(exist_ok=True)
    write(result/'evaluation_peaks.csv',ev);write(result/'calibration_peaks.csv',ca)
    return ev,ca

SITES=[('confirmation','C1'),('confirmation','C3'),('confirmation','F9_recording'),('validation2','C1'),('validation2','C2'),('validation2','C3'),('recording_panel_2','F9_recording')]
ARMS=['favoured','comparator']
def dataset(e0,c0,site):
    ev=[x for x in e0 if (x['panel'],x['contrast'])==site];cal=[x for x in c0 if (x['panel'],x['contrast'])==site]
    reps=sorted(set(int(x['replicate']) for x in ev))
    n0=400 if site[0]=='confirmation' else 250 if site[0]=='validation2' else 120
    f0=200 if site[0]=='confirmation' else n0
    csize=259 if site[0]=='validation2' else 199
    assert len(reps)==(1 if site[1]=='F9_recording' else 5)
    out={k:[] for k in ('normal','pre','eight','forty','cal','h')};ref={}
    for rep in reps:
        assoc=None
        for a in ARMS:
            def subset(rows): return sorted((x for x in rows if int(x['replicate'])==rep and x['arm']==a),key=lambda x:x['case_id'])
            normal=[x for x in subset(ev) if x['role']=='normal']
            fault=[x for x in subset(ev) if x['role']=='fault'];cs=subset(cal)
            assert (len(normal),len(fault),len(cs))==(n0,f0,csize)
            for name,rows in [('normal',normal),('fault',fault)]:
                ids=tuple(r['case_id'] for r in rows)
                assert len(set(ids))==len(ids)
                if name not in ref:ref[name]=ids
                assert ids==ref[name],(site,rep,a,name)
            ci=tuple(x['case_id'] for x in cs)
            assert len(set(ci))==len(ci) and (assoc is None or ci==assoc)
            assoc=ci
            hh={float(x['original_h']) for x in normal+fault+cs};assert len(hh)==1
            h=hh.pop()
            for key,rows,col in [('normal',normal,'normal48_max'),('pre',fault,'pre_max'),('eight',fault,'post8_max'),('forty',fault,'post40_max'),('cal',cs,'cal48_max')]:
                ar=np.asarray([float(x[col]) for x in rows]); assert np.all(np.isfinite(ar));out[key].append(ar)
            assert np.isclose(np.sort(out['cal'][-1])[246 if csize==259 else 189],h,atol=1e-9,rtol=1e-10),(site,rep,a,h)
            out['h'].append(h)
    D={k:np.asarray(v).reshape(len(reps),2,-1) for k,v in out.items()}
    assert np.all(D['forty']>=D['eight'])
    return D,reps,n0,f0,csize

def run(ev,ca,result):
    result=Path(result);result.mkdir(exist_ok=True)
    detail=[];mean=[];anchor=[];boots=[]
    for si,site in enumerate(SITES):
        D,reps,N,F,C=dataset(ev,ca,site)
        common=dict(panel=site[0],contrast=site[1],n_normal=N,n_fault=F,n_rep=len(reps))
        cal=np.sort(D['cal'],axis=-1);sn=np.sort(D['normal'],axis=-1)
        heval=np.concatenate([sn[:,:,::-1],np.nextafter(sn[:,:,:1],-np.inf)],axis=2)
        ops={}
        for kind,H,ii in [('original',D['h'],[0]),('calibration_rank',cal,list(range(1,C+1))),('matched_evaluation',heval,list(range(N+1)))]:
            rates=np.empty((len(reps),2,len(ii),4))
            for j,rep in enumerate(reps):
                for ia,a in enumerate(ARMS):
                    h=H[j,ia][None,:];p=D['pre'][j,ia,:,None]
                    counts=np.column_stack([(D['normal'][j,ia,:,None]>h).sum(axis=0),((p<=h)&(D['forty'][j,ia,:,None]>h)).sum(axis=0),
                                            ((p<=h)&(D['eight'][j,ia,:,None]>h)).sum(axis=0),(p>h).sum(axis=0)])
                    rates[j,ia]=100.*counts/np.array([N,F,F,F])
                    for t,ix in enumerate(ii):
                        d=dict(**common,replicate=rep,arm=a,kind=kind,index=ix,threshold=float(h[0,t]),normal_alarm_n=int(counts[t,0]),
                               clean40_n=int(counts[t,1]),clean8_n=int(counts[t,2]),fault_prefault_n=int(counts[t,3]),
                               far_pct=rates[j,ia,t,0],clean40_pct=rates[j,ia,t,1],clean8_pct=rates[j,ia,t,2],prefault_pct=rates[j,ia,t,3])
                        detail.append(d)
            avg=rates.mean(axis=0)
            for t,ix in enumerate(ii):
                for ia,a in enumerate(ARMS):
                    item=dict(**common,arm=a,kind=kind,index=ix,far_pct=avg[ia,t,0],clean40_pct=avg[ia,t,1],clean8_pct=avg[ia,t,2],prefault_pct=avg[ia,t,3])
                    mean.append(item)
            ops[kind]=avg
            if kind=='matched_evaluation':
                assert (rates[:,:,:,0]<=np.arange(N+1)[None,None,:]/N*100+1e-10).all()
        # Original points are independently verified from the run/event archives by extraction.
        mean_data=ops['matched_evaluation']
        select=[p*N//100 for p in (1,5,10)]
        for alpha,k in zip((1,5,10),select):
            for ih,endpoint in [(1,'40h'),(2,'8h')]:
                anchor.append(dict(**common,nominal_anchor_pct=alpha,k_allowed=k,empirical_target_pct=100*k/N,horizon=endpoint,
                                   favoured_far_pct=mean_data[0,k,0],comparator_far_pct=mean_data[1,k,0],
                                   favoured_detect_pct=mean_data[0,k,ih],comparator_detect_pct=mean_data[1,k,ih],
                                   favoured_minus_comparator_pp=mean_data[0,k,ih]-mean_data[1,k,ih],
                                   favoured_prefault_pct=mean_data[0,k,3],comparator_prefault_pct=mean_data[1,k,3]))
        # Paired bootstrap: refit only the descriptive threshold on each resampled normal panel.
        # No threshold uses a faulty run; preserve each frozen model and the physical-run pairing.
        rng=np.random.default_rng(2026110403+si);B=5000
        bn=rng.integers(N,size=(B,N),dtype=np.int32);bf=rng.integers(F,size=(B,F),dtype=np.int32)
        dist_counts=np.zeros((B,len(select),2),dtype=np.int32)
        for j in range(len(reps)):
            for ia in range(2):
                n=D['normal'][j,ia];p=D['pre'][j,ia];e=D['eight'][j,ia];f=D['forty'][j,ia]
                for start in range(0,B,250):
                    end=min(start+250,B);nstar=n[bn[start:end]]
                    ords=[N-k-1 for k in select]
                    hs=np.partition(nstar,ords,axis=1)[:,ords]
                    fs=bf[start:end];pre=p[fs];fstar=f[fs];estar=e[fs]
                    for ik in range(len(select)):
                        h=hs[:,ik,None];ok=pre<=h
                        dist_counts[start:end,ik,0]+=(1 if ia==0 else -1)*(ok&(fstar>h)).sum(axis=1,dtype=np.int32)
                        dist_counts[start:end,ik,1]+=(1 if ia==0 else -1)*(ok&(estar>h)).sum(axis=1,dtype=np.int32)
        dist=dist_counts.astype(np.float64)*(100./(F*len(reps)))
        for ik,(a,k) in enumerate(zip((1,5,10),select)):
            for ih,hz in enumerate(('40h','8h')):
                v=dist[:,ik,ih];low,high=np.quantile(v,[.025,.975])
                boots.append(dict(**common,nominal_anchor_pct=a,empirical_target_pct=100*k/N,horizon=hz,
                                  favoured_minus_comparator_pp=mean_data[0,k,ih+1]-mean_data[1,k,ih+1],
                                  percentile_low_pp=low,percentile_high_pp=high,bootstrap_draws=B,seed=2026110403+si))
    write(result/'curves_by_model.csv',detail);write(result/'curves_mean.csv',mean)
    write(result/'matched_anchors.csv',anchor);write(result/'paired_contrasts.csv',boots)
    write(result/'original_points.csv',[r for r in mean if r['kind']=='original'])
    print('Curves:',len(detail),'individual rows;',len(mean),'averaged rows;',len(boots),'paired bootstrap contrasts.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--from-source-tables',action='store_true',help='Assemble the four source-specific extrema exports first.')
    args=p.parse_args(); dest=args.root/'results'
    ev,ca=assemble(args.root) if args.from_source_tables else (read(dest/'evaluation_peaks.csv'),read(dest/'calibration_peaks.csv'))
    run(ev,ca,dest)
