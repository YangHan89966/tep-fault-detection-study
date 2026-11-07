from pathlib import Path
import csv, tempfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import fitz

HERE=Path(__file__).resolve().parent
with open(HERE/"P2-SIF04_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
for r in R:
    r["fault_id"]=int(r["fault_id"])
    for k in ["run_detection_pct","threshold_occupancy_pct","median_detected_run_delay_min"]:
        r[k]=float(r[k])

faults=[3,9,15]
methods=["pca_q_cusum","lda_residual_cusum","top5"]
labels={"pca_q_cusum":"Q-CUSUM","lda_residual_cusum":"LDA residual","top5":"Frozen Top-5"}
D={(r["fault_id"],r["method_key"]):r for r in R}

plt.rcParams.update({
    "font.size":8.0,"axes.titlesize":9.0,"axes.labelsize":8.0,
    "xtick.labelsize":7.2,"ytick.labelsize":7.2,"legend.fontsize":6.8
})
TMP=Path(tempfile.mkdtemp(prefix="p2sif04_panels_"))

def grouped(metric,ylabel,title,ylim=None,log=False,legend=False,fmt="{:.1f}"):
    fig=plt.figure(figsize=(7.0,2.35))
    ax=fig.add_axes([.10,.24,.86,.64])
    x=np.arange(len(faults)); width=.23
    for off,m in zip([-width,0,width],methods):
        vals=[D[(fid,m)][metric] for fid in faults]
        bars=ax.bar(x+off,vals,width=width,label=labels[m])
        for rect,val in zip(bars,vals):
            ax.annotate(fmt.format(val),(rect.get_x()+rect.get_width()/2,rect.get_height()),
                        xytext=(0,3),textcoords="offset points",
                        ha="center",va="bottom",fontsize=6.4)
    ax.set_xticks(x,["Fault 3","Fault 9","Fault 15"])
    ax.set_ylabel(ylabel)
    ax.set_title(title,loc="left",fontweight="bold")
    if ylim is not None: ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log"); ax.set_ylim(70,2300)
        ax.set_yticks([100,200,500,1000,2000])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True,axis="y",alpha=.20)
    if legend: ax.legend(loc="upper center",bbox_to_anchor=(.5,1.03),ncol=3,frameon=False)
    return fig

f=grouped("run_detection_pct","Run detection (%)","A  Frozen-final run detection",(0,108),legend=True)
f.savefig(TMP/"A.pdf",bbox_inches="tight"); plt.close(f)
f=grouped("threshold_occupancy_pct","Threshold occupancy (%)","B  Frozen-final sustained threshold occupancy",(0,102))
f.savefig(TMP/"B.pdf",bbox_inches="tight"); plt.close(f)
f=grouped("median_detected_run_delay_min","Median detected-run delay (min)",
          "C  Frozen-final detected-run latency (log scale)",log=True,fmt="{:.0f}")
f.axes[0].text(.02,.94,"Delay is conditional on detected runs; interpret together with Panel A.",
               transform=f.axes[0].transAxes,ha="left",va="top",fontsize=6.5)
f.savefig(TMP/"C.pdf",bbox_inches="tight"); plt.close(f)

mm=72/25.4
pw,ph=178*mm,157*mm
mx=7*mm; top=7*mm; bottom=7*mm; gap=4*mm
panel_h=(ph-top-bottom-2*gap)/3
doc=fitz.open(); page=doc.new_page(width=pw,height=ph)
for i,name in enumerate(["A","B","C"]):
    y0=top+i*(panel_h+gap)
    s=fitz.open(TMP/f"{name}.pdf")
    page.show_pdf_page(fitz.Rect(mx,y0,pw-mx,y0+panel_h),s,0,keep_proportion=True)
    s.close()
pdf=HERE/"P2-SIF04_performance_profiles_faults3_9_15.pdf"
doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P2-SIF04_performance_profiles_faults3_9_15.svg").write_text(
    d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(
    HERE/"P2-SIF04_performance_profiles_faults3_9_15.png")
d.close()
shutil.rmtree(TMP,ignore_errors=True)
