from pathlib import Path
import csv, tempfile, shutil, matplotlib.pyplot as plt, fitz
HERE=Path(__file__).resolve().parent
with open(HERE/"P1-FT07_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
macro=[r for r in R if r["record_type"]=="macro"]
order=["frozen_multievidence_top5_cusum","cva_rr_q_residual_cusum","cva_rr_l1_residual_cusum"]
M={r["method_key"]:r for r in macro}; methods=[M[k] for k in order]
markers=["o","s","^"]
plt.rcParams.update({"font.size":8.0,"axes.titlesize":9.0,"axes.labelsize":8.0,
                     "xtick.labelsize":7.0,"ytick.labelsize":7.0,"legend.fontsize":6.8})
TMP=Path(tempfile.mkdtemp(prefix="p1ft07_panels_"))
fig=plt.figure(figsize=(3.55,2.75)); ax=fig.add_axes([.16,.20,.80,.68])
for i,r in enumerate(methods):
    vals=[float(r[k]) for k in ["det15_pct","det30_pct","det60_pct","det120_pct","full_detection_pct"]]
    ax.plot(range(5),vals,marker=markers[i],linewidth=1.5,markersize=4,label=r["method_short"])
ax.set_xticks(range(5),["15","30","60","120","Full"]); ax.set_ylim(20,103)
ax.set_ylabel("Macro run detection (%)"); ax.set_xlabel("Time after fault onset (min)")
ax.set_title("A  Dynamic baselines lead early; Proposed leads late",loc="left",fontweight="bold")
ax.grid(True,alpha=.22); ax.legend(loc="lower right",frameon=False)
ax.text(.03,.93,"CVA-RR advantage: 15-60 min",transform=ax.transAxes,fontsize=6.5,va="top")
ax.text(.56,.36,"Proposed advantage:\n120 min + full coverage",transform=ax.transAxes,fontsize=6.5)
fig.savefig(TMP/"A.pdf",bbox_inches="tight"); plt.close(fig)
fig=plt.figure(figsize=(3.55,2.75)); ax=fig.add_axes([.17,.22,.78,.65])
offs={"Proposed Top-5":(6,7),"CVA-RR Q":(-52,-2),"CVA-RR L1":(6,7)}
for i,r in enumerate(methods):
    x=float(r["normal_far_pct"]); y=float(r["occupancy_pct"])
    ax.scatter([x],[y],s=60,marker=markers[i])
    ax.annotate(r["method_short"],(x,y),xytext=offs[r["method_short"]],
                textcoords="offset points",fontsize=6.5,fontweight="bold")
ax.set_xlim(4.0,5.5); ax.set_ylim(88.8,96.2)
ax.set_xlabel("Macro normal run FAR (%)"); ax.set_ylabel("Macro threshold occupancy (%)")
ax.set_title("B  Sustained operating region and tail penalty",loc="left",fontweight="bold")
ax.grid(True,alpha=.22)
ax.text(.03,.03,"Mean all-run penalized delay (min): Proposed 67.9 | Q 216.9 | L1 212.7\n"
        "Macro median delay = 48 min for all three methods",
        transform=ax.transAxes,fontsize=6.15,va="bottom")
fig.savefig(TMP/"B.pdf",bbox_inches="tight"); plt.close(fig)
mm=72/25.4; pw=178*mm; ph=78*mm; mx=7*mm; my=6*mm; gx=5*mm; cw=(pw-2*mx-gx)/2
doc=fitz.open(); page=doc.new_page(width=pw,height=ph)
for rect,name in [(fitz.Rect(mx,my,mx+cw,ph-my),"A"),(fitz.Rect(mx+cw+gx,my,pw-mx,ph-my),"B")]:
    s=fitz.open(TMP/f"{name}.pdf"); page.show_pdf_page(rect,s,0,keep_proportion=True); s.close()
pdf=HERE/"P1-FT07_dynamic_cva_vs_proposed_main_figure.pdf"; doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P1-FT07_dynamic_cva_vs_proposed_main_figure.svg").write_text(d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(HERE/"P1-FT07_dynamic_cva_vs_proposed_main_figure.png")
d.close(); shutil.rmtree(TMP,ignore_errors=True)
