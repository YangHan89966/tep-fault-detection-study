from pathlib import Path
import csv, tempfile, shutil, numpy as np, matplotlib.pyplot as plt, fitz
HERE=Path(__file__).resolve().parent
with open(HERE/"P1-F04_source_data.csv",encoding="utf-8",newline="") as f:
    R=sorted(list(csv.DictReader(f)),key=lambda r:int(r["top_k"]))
K=np.array([int(r["top_k"]) for r in R]); far=np.array([float(r["normal_run_level_far_pct"]) for r in R])
occ=np.array([float(r["threshold_occupancy_pct"]) for r in R]); delay=np.array([float(r["median_alarm_delay_min"]) for r in R])
primary=np.array([r["is_frozen_primary"]=="True" for r in R]); pi=int(np.where(primary)[0][0])
ko=int(K[np.argmax(occ)]); kf=int(K[np.argmin(far)]); kd=int(K[np.argmin(delay)])
plt.rcParams.update({"font.size":8.0,"axes.titlesize":9.0,"axes.labelsize":8.0,
                     "xtick.labelsize":7.0,"ytick.labelsize":7.0,"legend.fontsize":6.6})
TMP=Path(tempfile.mkdtemp(prefix="p1f04_panels_"))
fig=plt.figure(figsize=(3.55,2.75)); ax=fig.add_axes([.14,.20,.72,.68]); axr=ax.twinx()
p1,=ax.plot(K,occ,marker="s",linestyle="--",linewidth=1.4,label="Threshold occupancy")
p2,=axr.plot(K,far,marker="o",linestyle="-",linewidth=1.4,label="Normal run FAR")
ax.axvline(5,linestyle=":",linewidth=1.2); ax.scatter([5],[occ[pi]],marker="*",s=95,zorder=5); axr.scatter([5],[far[pi]],marker="*",s=95,zorder=5)
ax.set_xlim(0,41); ax.set_xticks([1,2,3,4,5,6,7,10,15,20,30,40]); ax.set_ylim(74,92); axr.set_ylim(0,8)
ax.set_xlabel("Number of selected residuals (K)"); ax.set_ylabel("Post-fault threshold occupancy (%)"); axr.set_ylabel("Normal run-level FAR (%)")
ax.set_title("A  FAR and sustained occupancy versus K",loc="left",fontweight="bold"); ax.grid(True,alpha=.22)
ax.legend([p1,p2],[p1.get_label(),p2.get_label()],loc="lower right",frameon=False)
ax.text(.98,.96,f"K=5 frozen: {occ[pi]:.2f}% occupancy / {far[pi]:.2f}% FAR\nObserved maxima/minima: occupancy K={ko}; FAR K={kf}",
        transform=ax.transAxes,ha="right",va="top",fontsize=6.35)
fig.savefig(TMP/"A.pdf",bbox_inches="tight"); plt.close(fig)
fig=plt.figure(figsize=(3.55,2.75)); ax=fig.add_axes([.15,.20,.81,.68])
ax.plot(K,delay,marker="o",linewidth=1.4); ax.axvline(5,linestyle=":",linewidth=1.2); ax.scatter([5],[delay[pi]],marker="*",s=95,zorder=5)
ax.set_xlim(0,41); ax.set_xticks([1,2,3,4,5,6,7,10,15,20,30,40]); ax.set_ylim(205,485)
ax.set_xlabel("Number of selected residuals (K)"); ax.set_ylabel("Median alarm delay (min)")
ax.set_title("B  Alarm delay versus K",loc="left",fontweight="bold"); ax.grid(True,alpha=.22)
ax.text(.03,.96,f"Run detection = 250/250 for every K\nK=5 frozen: {delay[pi]:.0f} min | observed minimum: K={kd}, {delay.min():.0f} min",
        transform=ax.transAxes,ha="left",va="top",fontsize=6.35)
fig.savefig(TMP/"B.pdf",bbox_inches="tight"); plt.close(fig)
mm=72/25.4; pw=178*mm; ph=76*mm; mx=7*mm; my=6*mm; gx=5*mm; cw=(pw-2*mx-gx)/2
doc=fitz.open(); page=doc.new_page(width=pw,height=ph)
for rect,name in [(fitz.Rect(mx,my,mx+cw,ph-my),"A"),(fitz.Rect(mx+cw+gx,my,pw-mx,ph-my),"B")]:
    s=fitz.open(TMP/f"{name}.pdf"); page.show_pdf_page(rect,s,0,keep_proportion=True); s.close()
pdf=HERE/"P1-F04_post_freeze_fault3_topk_sensitivity.pdf"; doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P1-F04_post_freeze_fault3_topk_sensitivity.svg").write_text(d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(HERE/"P1-F04_post_freeze_fault3_topk_sensitivity.png")
d.close(); shutil.rmtree(TMP,ignore_errors=True)
