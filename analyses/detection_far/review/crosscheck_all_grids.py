import pandas as pd,numpy as np
from pathlib import Path
import sys,json
# Independent cross-check from panel-specific source maxima, bypassing the unified analysis.
base=Path(sys.argv[1])
a=pd.concat([pd.read_csv(base/p/'run_peaks.csv') for p in ['v2','confirm','v2c2']])
cal=pd.concat([pd.read_csv(base/p/'calibration_peaks.csv') for p in ['v2','confirm','v2c2']])
f9=pd.read_csv(base/'f9/F9_RUN_MAXIMA.csv').assign(contrast='F9_recording',replicate=1)
f9['arm']=f9.recording.map({'controller_input':'favoured','post_update':'comparator'})
f9['role']=f9.role.map({'normal_evaluation':'normal','fault_evaluation':'fault','normal_calibration':'cal'})
a=pd.concat([a,f9.loc[f9.role!='cal']]);cal=pd.concat([cal,f9.loc[f9.role=='cal'].rename(columns={'normal48_max':'cal48_max'})])
cols=['panel','contrast','arm','replicate']
cg=cal.groupby(cols);records=[]
for key,g in a.groupby(cols):
 n=g.loc[g.role=='normal'].normal48_max.to_numpy(); f=g.loc[g.role=='fault'];cc=cg.get_group(key);h0=g.original_h.iloc[0]
 dd={'calibration_rank':np.sort(cc.cal48_max),'matched_evaluation':np.r_[np.sort(n)[::-1],np.nextafter(n.min(),-np.inf)],'original':np.array([h0])}
 for kind,hh in dd.items():
  T=hh[:,None];pre0=f.pre_max.to_numpy()[None,:]<=T
  ar=100*(n[None,:]>T).mean(axis=1);d8=100*(pre0 & (f.post8_max.to_numpy()[None,:]>T)).mean(axis=1);d40=100*(pre0 & (f.post40_max.to_numpy()[None,:]>T)).mean(axis=1)
  for i in range(hh.size):records.append([*key,kind,i+(kind=='calibration_rank'),ar[i],d8[i],d40[i],len(n),len(f)])
r=pd.DataFrame(records,columns=[*cols,'kind','index','far_pct','clean8_pct','clean40_pct','n_normal','n_fault'])
r.panel=r.panel.replace({'Validation 2':'validation2'});r=r.groupby(['panel','contrast','arm','kind','index'],as_index=False).agg({'replicate':'nunique','far_pct':'mean','clean8_pct':'mean','clean40_pct':'mean','n_normal':'first','n_fault':'first'}).rename(columns={'replicate':'n_rep'})

keys=['panel','contrast','arm','kind','index']
gold=pd.read_csv(base/'results/curves_mean.csv')
comp=r.merge(gold,on=keys,how='outer',indicator=True,suffixes=('_a','_b'),validate='1:1')
assert (comp._merge=='both').all()
err={k:float((comp[k+'_a']-comp[k+'_b']).abs().max()) for k in ('far_pct','clean8_pct','clean40_pct')}
assert max(err.values())<1e-11
for k in ('n_normal','n_fault','n_rep'):assert (comp[k+'_a']==comp[k+'_b']).all()
print(json.dumps({'status':'PASS','unique_grid_rows_compared':len(comp),'max_abs_diff_percentage_points':err,'denominators_and_replicate_counts':'exact'},indent=2))
