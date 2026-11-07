from pathlib import Path
import csv, tempfile, shutil, numpy as np, matplotlib.pyplot as plt, fitz

HERE=Path(__file__).resolve().parent
with open(HERE/"P0-F10_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
def rows(panel,series=None):
    q=[r for r in R if r["panel"]==panel]
    if series is not None: q=[r for r in q if r["series"]==series]
    return q
plt.rcParams.update({"font.size":8.0,"axes.titlesize":9.0,"axes.labelsize":8.0,
                     "xtick.labelsize":7.4,"ytick.labelsize":7.4,"legend.fontsize":7.0})
TMP=Path(tempfile.mkdtemp(prefix="p0f10_panels_"))
def save(fig,name):
    fig.savefig(TMP/f"{name}.pdf",bbox_inches="tight"); plt.close(fig)

ld=sorted(rows("A","macro worst-combination run detection"),key=lambda r:float(r["x_value"]))
lo=sorted(rows("A","macro worst-combination threshold occupancy"),key=lambda r:float(r["x_value"]))
ll=sorted(rows("A","macro worst-combination penalized median delay"),key=lambda r:float(r["x_value"]))
x=[float(r["x_value"]) for r in ld]; yd=[float(r["value"]) for r in ld]
yo=[float(r["value"]) for r in lo]; yl=[float(r["value"]) for r in ll]
f16d=float(rows("A","Fault 16 worst run detection")[0]["value"])
f16o=float(rows("A","Fault 16 worst threshold occupancy")[0]["value"])
f16l=float(rows("A","Fault 16 worst penalized median delay")[0]["value"])
fig=plt.figure(figsize=(6.40,2.75)); ax=fig.add_axes([.09,.20,.62,.66]); ax2=ax.twinx()
p1,=ax.plot(x,yd,marker="o",linewidth=1.5,label="Run detection")
p2,=ax.plot(x,yo,marker="s",linewidth=1.5,linestyle="--",label="Threshold occupancy")
p3,=ax2.plot(x,yl,marker="^",linewidth=1.5,linestyle="-.",label="Delay (min)")
ax.set_xlim(.8,3.2); ax.set_xticks([1,2,3]); ax.set_ylim(82,101.5); ax2.set_ylim(0,280)
ax.set_xlabel("Simultaneously lost selected evidence coordinates")
ax.set_ylabel("Worst-combination macro rate (%)")
ax.set_title("A  Simultaneous selected-coordinate loss",loc="left",fontweight="bold"); ax.grid(True,alpha=.22)
for xx,yy in zip(x,yo):
    off={1:(0,-13),2:(-10,-13),3:(0,-13)}[int(xx)]
    ax.annotate(f"{yy:.1f}",(xx,yy),xytext=off,textcoords="offset points",ha="center",fontsize=6.8)
for xx,yy in zip(x,yl):
    off={1:(0,8),2:(22,9),3:(-15,8)}[int(xx)]
    ax2.annotate(f"{yy:.0f}",(xx,yy),xytext=off,textcoords="offset points",ha="center",fontsize=6.8)
ax.legend([p1,p2,p3],[p1.get_label(),p2.get_label(),p3.get_label()],loc="lower left",frameon=False,ncol=3)
fig.text(.76,.73,"Fault 16, worst 3-coordinate case",fontsize=7.2,fontweight="bold")
fig.text(.76,.62,f"Run detection      {f16d:.1f}%",fontsize=7.0)
fig.text(.76,.54,f"Threshold occupancy {f16o:.2f}%",fontsize=7.0)
fig.text(.76,.46,f"Penalized delay     {f16l:.0f} min",fontsize=7.0)
fig.text(.76,.27,"All one- and two-coordinate\nloss combinations retained\n100% run detection.",fontsize=6.9,va="top")
save(fig,"A")

def contamination(panel,cond,title):
    rr=rows(panel,"macro normal run-level FAR")
    t1=sorted([r for r in rr if r["condition"]=="Top-1 reference"],key=lambda r:float(r["x_value"]))
    t5=sorted([r for r in rr if r["condition"]==cond],key=lambda r:float(r["x_value"]))
    xs=[float(r["x_value"]) for r in t1]; y1=[float(r["value"]) for r in t1]; y5=[float(r["value"]) for r in t5]
    fig=plt.figure(figsize=(3.20,2.55)); ax=fig.add_axes([.17,.21,.78,.66])
    ax.plot(xs,y1,marker="o",linewidth=1.6,label="Top-1 reference")
    ax.plot(xs,y5,marker="s",linestyle="--",linewidth=1.6,label="Top-5")
    ax.set_xticks(xs,["0.5","1.0","2.0"]); ax.set_ylim(0,85)
    ax.set_xlabel(r"Contamination scale $\sigma$"); ax.set_ylabel("Macro normal run FAR (%)")
    ax.set_title(title,loc="left",fontweight="bold"); ax.grid(True,alpha=.22); ax.legend(loc="upper left",frameon=False)
    for xx,yy in zip(xs,y5):
        off=(-3,8) if xx==2.0 else (0,8)
        ax.annotate(f"{yy:.1f}",(xx,yy),xytext=off,textcoords="offset points",ha="center",fontsize=6.8)
    return fig
save(contamination("B","Top-5 exact total-energy","B  Exact total-energy matching"),"B")
save(contamination("C","Top-5 common mode","C  Common-mode equal-index matching"),"C")

mm=72/25.4; pw=178*mm; ph=118*mm; mx=8*mm; mt=6*mm; mb=6*mm; gx=6*mm; gy=4*mm; th=52*mm
bw=(pw-2*mx-gx)/2
rects=[fitz.Rect(mx,mt,pw-mx,mt+th),
       fitz.Rect(mx,mt+th+gy,mx+bw,ph-mb),
       fitz.Rect(mx+bw+gx,mt+th+gy,pw-mx,ph-mb)]
doc=fitz.open(); page=doc.new_page(width=pw,height=ph)
for rect,name in zip(rects,["A","B","C"]):
    s=fitz.open(TMP/f"{name}.pdf"); page.show_pdf_page(rect,s,0,keep_proportion=True); s.close()
pdf=HERE/"P0-F10_matched_multi_coordinate_loss_and_contamination_robustness.pdf"; doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P0-F10_matched_multi_coordinate_loss_and_contamination_robustness.svg").write_text(
    d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(
    HERE/"P0-F10_matched_multi_coordinate_loss_and_contamination_robustness.png")
d.close(); shutil.rmtree(TMP,ignore_errors=True)
