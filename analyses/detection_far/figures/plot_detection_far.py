"""Plot frozen operating points and empirical threshold sweeps; no smoothing/interpolation.

python plot_detection_far.py curves_mean.csv FIGURE_DIR
Requires matplotlib and NumPy. Feed the complete curves (one row per arm/rank).
"""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.1,"axes.titlesize":10.0,
 "axes.labelsize":8.8,"axes.linewidth":.65,"svg.fonttype":"none","pdf.fonttype":42,
 "xtick.labelsize":8.0,"ytick.labelsize":8.0,"axes.edgecolor":"#56606A","axes.labelcolor":"#252E37",
 "text.color":"#232C34","xtick.color":"#37454D","ytick.color":"#37454D",
 "lines.solid_capstyle":"round","savefig.facecolor":"white","axes.unicode_minus":False})
B="#007283";G="#5C6268";B2="#28323A"; G2="#666F75"
CASES={
 "C1":("Fault 3  |  Score form","Signed score","Squared score"),
 "C3":("Fault 15  |  Direction","Second-moment direction","Mean-shift direction"),
 "C2":("Fault 9  |  Direction","Second-moment direction","Mean-shift direction"),
 "F9_recording":("Fault 9  |  Recording","Controller-input","Post-update")}
JOBS=[('confirmation','calibration_rank','Fig_1_Confirmation',('C1','C3','F9_recording')),
      ('validation2','calibration_rank','Fig_2_Validation2',('C1','C2','C3')),
      ('confirmation','matched_evaluation','Fig_S1_Confirmation_matched',('C1','C3','F9_recording'))]
LN = dict(favoured='-',comparator=(0,(3.8,2.3)))
def colors(c):return (B2,G2) if c=='C2' else (B,G)
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
    for r in rs:
        for k in ('far_pct','clean40_pct','clean8_pct'):
            r[k]=float(r[k]); assert 0<=r[k]<=100.000000001, (k,r)
        for k in ('n_normal','n_fault','n_rep'):r[k]=int(r[k])
        if r['kind']!='original':r['index']=int(float(r['index']))
    return rs
def draw(rs,panel,kind,stem,contrasts,out):
    fig,axs=plt.subplots(2,3,figsize=(7.60,5.45),sharex=True,sharey=True)
    fig.subplots_adjust(left=.095,right=.982,bottom=.185,top=.696,wspace=.205,hspace=.235)
    matched=kind=='matched_evaluation';validation=panel=='validation2'
    title="Validation 2" if validation else "Independent-stream confirmation"
    method="Matched evaluation false-alarm rates" if matched else ("Thresholds from separate calibration runs" if validation else "Thresholds from independent calibration runs")
    fig.text(.050,.965,title,fontsize=14.3,fontweight='bold',va='top')
    fig.text(.050,.909,method,fontsize=9.6,va='top',color='#48535A')
    letters='abcdef'
    meta=[]
    for ci,c in enumerate(contrasts):
        fcol,ccol=colors(c);title_c,an,bn=CASES[c]
        pos=axs[0,ci].get_position();x=pos.x0
        fig.text(x,.838,title_c,ha='left',va='center',fontsize=9.65,fontweight='bold')
        for y,lab,co,ls in [(.793,an,fcol,LN['favoured']),(.752,bn,ccol,LN['comparator'])]:
            fig.add_artist(Line2D([x,x+.034],[y,y],transform=fig.transFigure,color=co,linestyle=ls,lw=1.65))
            fig.text(x+.042,y,lab,va='center',ha='left',fontsize=7.9,color='#273039')
        for row,yk in enumerate(['clean40_pct','clean8_pct']):
            ax=axs[row,ci]
            for arm,co,m in [('favoured',fcol,'o'),('comparator',ccol,'s')]:
                same=[r for r in rs if (r['panel'],r['contrast'],r['arm'])==(panel,c,arm)]
                part=sorted([r for r in same if r['kind']==kind],key=lambda r:r['index'],reverse=not matched)
                orig=[r for r in same if r['kind']=='original']
                if len(part)<2 or len(orig)!=1:raise ValueError(f'{panel} {c} {arm}: {len(part)} curve rows; {len(orig)} original points')
                o=orig[0];meta.append((o['n_normal'],o['n_fault'],o['n_rep']))
                xx=np.array([r['far_pct'] for r in part]); yy=np.array([r[yk] for r in part])
                # Follow threshold order, not TPR order; keep duplicate FAR values and reversals.
                assert np.all(np.diff(xx)>=-1e-9)
                ax.plot(xx,yy,ls=LN[arm],color=co,lw=1.2,zorder=3 if arm=='favoured' else 2)
                ax.plot(o['far_pct'],o[yk],m,mfc=co if arm=='favoured' else 'white',mec=co,
                        mew=.95,ms=5.25,ls='none',zorder=6)
            ax.set_xlim(-.17,15.16);ax.set_ylim(-2,102)
            ax.xaxis.set_major_locator(FixedLocator([0,5,10,15]));ax.yaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
            ax.grid(axis='y',color='#E2E5E8',lw=.55,zorder=0);ax.set_axisbelow(True)
            for spine in ('top','right'):ax.spines[spine].set_visible(False)
            ax.tick_params(length=2.7,width=.65,pad=2.8)
            ax.text(-.005,1.034,letters[row*3+ci],fontsize=9.2,ha='left',va='bottom',transform=ax.transAxes,fontweight='bold')
            if ci==0:ax.set_ylabel(('40 h' if row==0 else '8 h')+' clean detection (%)',labelpad=4)
    fig.text(.537,.093,'Normal runs with any alarm in 48 h (%)',ha='center',va='center',fontsize=9.8)
    fig.text(.05,.034,'Markers: original frozen operating points.  Lines join adjacent threshold settings in order.',ha='left',va='center',fontsize=8.1,color='#48535A')
    settings={'pdf':{},'svg':{},'png':{'dpi':320}}
    for ext,kw in settings.items():fig.savefig(out/(stem+'.'+ext),**kw)
    plt.close(fig)
    ns=set((n,f) for n,f,_ in meta);assert len(ns)==1
    n,f=ns.pop();reps={c:{r['n_rep'] for r in rs if (r['panel'],r['contrast'],r['kind'])==(panel,c,'original')} for c in contrasts}
    assert all(len(s)==1 for s in reps.values());reps={c:s.pop() for c,s in reps.items()}; expected={'C1':5,'C2':5,'C3':5,'F9_recording':1}
    assert all(v==expected[k] for k,v in reps.items())
    ctext="C1 compares signed and squared scores on the same selected mean-shift direction; C3 compares second-moment and mean-shift directions with a squared score. "
    ctext+=("C2 compares second-moment and mean-shift directions with a signed score; its direction labels do not imply confirmation of the prediction. " if validation else "The Fault 9 comparison applies the same fixed historical monitor to controller-input and post-update records. ")
    desc=("Each point uses a separate limit for each arm and training replicate at a common ascending order-statistic rank of the calibration-run maxima. " if not matched else
        "For each target number of false alarms, the two arms and each training replicate receive separate thresholds determined from the same normal evaluation maxima. Thus the comparison matches observed false-alarm rates descriptively; it does not estimate performance at independently calibrated limits. ")
    repsdesc=('The three contrasts are means over five frozen training replicates, all evaluated on those same runs. ' if validation else
       'C1 and C3 are means over five frozen training replicates evaluated on the same trajectories; Fault 9 uses one frozen historical monitor. ')
    return (stem+" — "+title+". "+method+". "+ctext+desc+
      f"Each false-alarm rate is based on {n} normal runs, and each fault-specific rate on {f} fault runs. "+repsdesc+
      "A clean detection requires no alarm before onset and a strict threshold crossing within the indicated interval after onset (40 h: top; 8 h: bottom). All fault runs remain in the denominator. The original limits (filled circles for the solid-line method; open squares for the dashed-line method) use rank "+
      ("247 of 259" if validation else "190 of 199")+" of the separate normal calibration maxima. Lines connect adjacent thresholds without smoothing or enforcing monotonicity. The display is restricted to 0–15% empirical false alarms; the data table includes the full range.")
def main():
    pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('csv',type=Path);pa.add_argument('output',type=Path);args=pa.parse_args()
    rs=read(args.csv);args.output.mkdir(parents=True,exist_ok=True)
    caps=[draw(rs,*job,args.output) for job in JOBS]
    (args.output/'FIGURE_CAPTIONS_EN.txt').write_text('\n\n'.join(caps)+'\n',encoding='utf-8')
if __name__=='__main__':main()
