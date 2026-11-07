from pathlib import Path
import csv, tempfile, shutil
import numpy as np
import matplotlib.pyplot as plt
import fitz

HERE=Path(__file__).resolve().parent
with open(HERE/"P1-F11_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
P={r["evidence_class"]:r for r in R}
primary,bank,gate=P["primary"],P["secondary"],P["exploratory"]

pl=float(primary["normal_run_far_pct"])
of=float(bank["normal_run_far_pct"])
gf=float(gate["normal_run_far_pct"])
glo=float(gate["normal_run_far_ci_low_pct"])
ghi=float(gate["normal_run_far_ci_high_pct"])
ordet=float(bank["reserved_fault_run_detection_pct"])
gdet=float(gate["reserved_fault_run_detection_pct"])
orocc=float(bank["threshold_occupancy_pct"])
gocc=float(gate["threshold_occupancy_pct"])
ordelay=float(bank["median_fault_level_allrun_penalized_mean_delay_min"])
gdelay=float(gate["median_fault_level_allrun_penalized_mean_delay_min"])
rel=100*(of-gf)/of

plt.rcParams.update({
    "font.size":8.1,"axes.titlesize":9.2,"axes.labelsize":8.1,
    "xtick.labelsize":7.1,"ytick.labelsize":7.1
})
TMP=Path(tempfile.mkdtemp(prefix="p1f11_panels_"))

fig=plt.figure(figsize=(3.60,2.85))
ax=fig.add_axes([.15,.23,.82,.65])
x=np.arange(3)
ax.bar(x,[pl,of,gf],width=.62)
ax.errorbar([2],[gf],yerr=[[gf-glo],[ghi-gf]],fmt="none",capsize=4,linewidth=1.2)
ax.axvline(.5,linestyle=":",linewidth=1.0)
ax.set_xticks(x,["Primary local\nmacro mean","Hypothetical\nOR bank","Exploratory\njoint gate"])
ax.set_ylabel("Normal run-level FAR (%)")
ax.set_ylim(0,42)
ax.set_title("A  Distinct FAR estimands - do not pool",loc="left",fontweight="bold")
ax.grid(True,axis="y",alpha=.22)
ax.text(0,pl+1.0,f"{pl:.3f}%",ha="center",va="bottom",fontsize=7.0,fontweight="bold")
ax.text(1,of+1.0,f"{of:.1f}%",ha="center",va="bottom",fontsize=7.0,fontweight="bold")
ax.text(2,7.0,f"{gf:.1f}%\n95% CI {glo:.2f}-{ghi:.2f}%",ha="center",va="bottom",fontsize=6.7,fontweight="bold")
ax.text(1.98,17.0,f"{rel:.1f}% relative FAR\nreduction vs OR bank",ha="right",va="center",fontsize=6.6)
ax.text(.08,.88,"local-monitor\nestimand",transform=ax.transAxes,ha="center",va="top",fontsize=6.2)
ax.text(.72,.88,"system-level rules",transform=ax.transAxes,ha="center",va="top",fontsize=6.2)
fig.savefig(TMP/"A.pdf",bbox_inches="tight")
plt.close(fig)

fig=plt.figure(figsize=(3.60,2.85))
ax=fig.add_axes([.16,.23,.68,.65])
ax2=ax.twinx()
xx=np.arange(2); width=.30
ax.bar(xx-width/2,[ordet,gdet],width=width)
ax.bar(xx+width/2,[orocc,gocc],width=width)
ax2.plot(xx,[ordelay,gdelay],marker="D",linestyle="--",linewidth=1.2)
ax.set_xticks(xx,["OR bank","Joint gate"])
ax.set_ylim(94,101.2)
ax2.set_ylim(0,90)
ax.set_ylabel("Reserved-fault macro rate (%)")
ax2.set_ylabel("Delay summary (min)")
ax.set_title("B  System-level coverage and latency trade-off",loc="left",fontweight="bold")
ax.grid(True,axis="y",alpha=.22)
for x0,y0 in zip(xx,[ordet,gdet]):
    ax.text(x0-width/2,y0+.12,f"{y0:.0f}%\nrun",ha="center",va="bottom",fontsize=6.3)
for x0,y0 in zip(xx,[orocc,gocc]):
    ax.text(x0+width/2,y0+.12,f"{y0:.2f}%\nocc.",ha="center",va="bottom",fontsize=6.25)
ax2.annotate(f"{ordelay:.1f} min",(0,ordelay),xytext=(-34,13),textcoords="offset points",fontsize=6.5)
ax2.annotate(f"{gdelay:.1f} min",(1,gdelay),xytext=(7,5),textcoords="offset points",fontsize=6.5)
fig.text(.16,.04,"Both retain 250/250 detection for every reserved fault; the gate lowers normal FAR but delays alarm.",
         fontsize=6.15,ha="left",va="bottom")
fig.savefig(TMP/"B.pdf",bbox_inches="tight")
plt.close(fig)

mm=72/25.4
pw,ph=178*mm,82*mm
mx,my,gx=7*mm,6*mm,5*mm
cw=(pw-2*mx-gx)/2
doc=fitz.open()
page=doc.new_page(width=pw,height=ph)
for rect,name in [
    (fitz.Rect(mx,my,mx+cw,ph-my),"A"),
    (fitz.Rect(mx+cw+gx,my,pw-mx,ph-my),"B")
]:
    s=fitz.open(TMP/f"{name}.pdf")
    page.show_pdf_page(rect,s,0,keep_proportion=True)
    s.close()
pdf=HERE/"P1-F11_exploratory_joint_gate_operating_point.pdf"
doc.save(pdf)
doc.close()

d=fitz.open(pdf)
(HERE/"P1-F11_exploratory_joint_gate_operating_point.svg").write_text(
    d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(
    HERE/"P1-F11_exploratory_joint_gate_operating_point.png")
d.close()
shutil.rmtree(TMP,ignore_errors=True)
