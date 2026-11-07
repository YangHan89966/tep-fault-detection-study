#!/usr/bin/env python3
"""Rebuild P0-F06 from the packaged, audited local source CSV.
No experiment execution, refitting, or retuning occurs.
"""
from pathlib import Path
import csv, statistics, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
SRC=HERE/'P0-F06_source_data.csv'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

with SRC.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
faults=[int(r['fault_id']) for r in rows]
families=[r['family_label'] for r in rows]
occ=[100*float(r['post_fault_threshold_occupancy']) for r in rows]
delay=[float(r['all_run_penalized_median_delay_min']) for r in rows]
far=[100*float(r['normal_run_level_far']) for r in rows]
full=[100*float(r['full_horizon_run_detection']) for r in rows]
macro_occ=statistics.mean(occ); macro_delay=statistics.median(delay); macro_far=statistics.mean(far)
assert len(rows)==17 and all(abs(v-100)<1e-9 for v in full)
cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']
family_order=['Signed level','Absolute deviation','Lag-1 product']
family_color={fam:cycle[i] for i,fam in enumerate(family_order)}
family_hatch={'Signed level':'//','Absolute deviation':'','Lag-1 product':'xx'}
colors=[family_color[f] for f in families]; hatches=[family_hatch[f] for f in families]
fig,axes=plt.subplots(3,1,figsize=(7.01,5.85),sharex=True,gridspec_kw={'hspace':0.16})
fig.subplots_adjust(left=0.105,right=0.985,top=0.925,bottom=0.11)
x=list(range(len(faults))); bar_kw=dict(width=0.74,linewidth=0.55,edgecolor='black')
for ax,vals,ylabel,ylim,yticks,ref,lab,letter in [
    (axes[0],occ,'Threshold occupancy (%)',(0,105),[0,20,40,60,80,100],macro_occ,f'Macro mean {macro_occ:.2f}%','A'),
    (axes[1],delay,'Penalized median delay (min)',(0,182),[0,50,100,150],macro_delay,f'Macro median {macro_delay:.0f} min','B'),
    (axes[2],far,'Normal run FAR (%)',(0,10),[0,2,4,6,8,10],macro_far,f'Macro mean {macro_far:.2f}%','C')]:
    bars=ax.bar(x,vals,color=colors,**bar_kw)
    for b,h in zip(bars,hatches): b.set_hatch(h)
    ax.set_ylabel(ylabel,fontsize=8.5,labelpad=6); ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.axhline(ref,linestyle='--',linewidth=0.9,color='0.25')
    off=1.1 if ax is axes[0] else (3 if ax is axes[1] else 0.22)
    ax.text(0.992,ref+off,lab,ha='right',va='bottom',fontsize=7.6,transform=ax.get_yaxis_transform(),bbox=dict(facecolor='white',edgecolor='none',alpha=0.82,pad=0.6))
    ax.text(0.008,0.965,letter,transform=ax.transAxes,fontweight='bold',fontsize=9.5,va='top')
    ax.grid(axis='y',linewidth=0.45,alpha=0.25); ax.tick_params(axis='both',labelsize=7.8,length=3)
    for sp in ax.spines.values(): sp.set_linewidth(0.75)
i18=faults.index(18)
axes[0].annotate('Fault 18: 61.8%',xy=(i18,occ[i18]),xytext=(i18-3.2,72),fontsize=7.6,ha='left',va='center',arrowprops=dict(arrowstyle='->',lw=0.8,color='black'))
for fid in [10,13,18,20]:
    idx=faults.index(fid); axes[1].text(idx,delay[idx]+4.0,f'{delay[idx]:g}',ha='center',va='bottom',fontsize=7.1,fontweight='bold' if fid==18 else 'normal')
axes[2].set_xlabel('Reserved TEP fault',fontsize=8.5); axes[2].set_xticks(x,[str(f) for f in faults])
for tick,fid in zip(axes[2].get_xticklabels(),faults):
    if fid in {10,13,18,20}: tick.set_fontweight('bold')
handles=[Patch(facecolor=family_color[f],edgecolor='black',hatch=family_hatch[f],linewidth=0.55,label=f) for f in family_order]
fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(0.53,0.995),frameon=False,fontsize=8.0,handlelength=1.6,columnspacing=1.5)
for ext,kwargs in [('pdf',{}),('svg',{}),('png',{'dpi':600})]:
    fig.savefig(HERE/f'P0-F06_frozen_proposed_top5_by_reserved_fault.{ext}',bbox_inches='tight',**kwargs)
plt.close(fig)
files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
with (HERE/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
    for p in files: f.write(f'{sha256(p)}  {p.name}\n')
print(f'PASS: rebuilt P0-F06 from {SRC.name}; rows={len(rows)}')
