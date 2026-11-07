from pathlib import Path
import csv, numpy as np, matplotlib.pyplot as plt, fitz, tempfile, shutil

HERE = Path(__file__).resolve().parent
with open(HERE/"P0-F09_source_data.csv", encoding="utf-8", newline="") as f:
    R = list(csv.DictReader(f))

def get(panel, branch=None, fault=None, metric=None, submetric=None):
    rows=[r for r in R if r["panel"]==panel]
    if branch is not None: rows=[r for r in rows if r["branch_key"]==branch]
    if fault is not None: rows=[r for r in rows if r["fault_id"]==str(fault)]
    if metric is not None: rows=[r for r in rows if r["metric"]==metric]
    if submetric is not None: rows=[r for r in rows if r["submetric"]==submetric]
    return rows

plt.rcParams.update({"font.size":7.5,"axes.titlesize":8.5,"axes.labelsize":7.5,
                     "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.8})
TMP=Path(tempfile.mkdtemp(prefix="p0f09_panels_"))
W,H=3.38,2.55
short={"adaptive_5family":"Adaptive-5","fixed_absolute_deviation":"Fixed-AD",
       "restricted_ad_or_sl":"AD-or-SL","fixed_signed_level":"Fixed-SL"}

def save(fig,name):
    fig.savefig(TMP/f"{name}.pdf",bbox_inches="tight"); plt.close(fig)

# A
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0.14,0.18,0.82,0.70])
branches=["adaptive_5family","fixed_absolute_deviation","restricted_ad_or_sl","fixed_signed_level"]
markers=["o","s","^","D"]; subs=["15 min","30 min","60 min","120 min","full horizon"]
for i,b in enumerate(branches):
    vals=[float(get("A",branch=b,submetric=s)[0]["value"]) for s in subs]
    ax.plot(np.arange(5),vals,marker=markers[i],linewidth=1.4,markersize=4,label=short[b])
ax.set_xticks(np.arange(5),["15","30","60","120","full"]); ax.set_ylim(0,104)
ax.set_ylabel("All-panel run detection (%)"); ax.set_xlabel("Time after fault onset (min)")
ax.set_title("A  U1: simplification changes the timing profile",loc="left",fontweight="bold")
ax.grid(True,alpha=.22); ax.legend(loc="lower right",frameon=False); save(fig,"A")

def v(f,b,m): return float(get("B",branch=b,fault=f,metric=m)[0]["value"])
# B
def v(f,b,m): return float(get("B",branch=b,fault=f,metric=m)[0]["value"])
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([.035,.07,.93,.86]); ax.axis("off")
ax.text(0,1.02,"B  U1: localized benefit and localized harm",transform=ax.transAxes,
        fontsize=8.5,fontweight="bold",va="bottom")
ax.text(.02,.83,"Fault 5  |  Adaptive-5 lag-1 product vs Fixed-AD",transform=ax.transAxes,fontweight="bold")
ax.text(.04,.67,
        f"15-min detection  {v(5,'adaptive_5family','detection_within_15min'):5.1f}% vs {v(5,'fixed_absolute_deviation','detection_within_15min'):5.1f}%"
        f"     |     median delay  {v(5,'adaptive_5family','penalized_median_delay'):.0f} vs {v(5,'fixed_absolute_deviation','penalized_median_delay'):.0f} min",
        transform=ax.transAxes,fontsize=6.7,family="monospace")
ax.text(.04,.54,
        f"Normal FAR  {v(5,'adaptive_5family','normal_run_level_far'):.1f}% vs {v(5,'fixed_absolute_deviation','normal_run_level_far'):.1f}%"
        "     → localized early-speed gain with a small FAR cost.",
        transform=ax.transAxes,fontsize=6.7)
ax.plot([.02,.98],[.45,.45],transform=ax.transAxes,linewidth=.8)
ax.text(.02,.34,"Fault 18  |  Adaptive-5 signed level vs Fixed-AD",transform=ax.transAxes,fontweight="bold")
ax.text(.04,.20,
        f"Occupancy  {v(18,'adaptive_5family','threshold_occupancy'):5.2f}% vs {v(18,'fixed_absolute_deviation','threshold_occupancy'):5.2f}%"
        f"     |     median delay  {v(18,'adaptive_5family','penalized_median_delay'):.0f} vs {v(18,'fixed_absolute_deviation','penalized_median_delay'):.0f} min",
        transform=ax.transAxes,fontsize=6.7,family="monospace")
ax.text(.04,.08,
        f"Normal FAR  {v(18,'adaptive_5family','normal_run_level_far'):.1f}% vs {v(18,'fixed_absolute_deviation','normal_run_level_far'):.1f}%"
        "     → family score can be misaligned with final sustained coverage.",
        transform=ax.transAxes,fontsize=6.5)
save(fig,"B")

# C
faults=sorted({r["fault_id"] for r in get("C",branch="adaptive_5family")},key=int)
dat=[]
for f in faults:
    dat.append((f,float(get("C",branch="adaptive_5family",fault=f,metric="mean_top5_jaccard")[0]["value"]),
                float(get("C",branch="adaptive_5family",fault=f,metric="family_recovery_rate")[0]["value"])*100,
                float(get("C",branch="adaptive_5family",fault=f,metric="exact_top5_set_recovery_rate")[0]["value"])))
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([.15,.18,.81,.70])
ax.scatter([d[1] for d in dat],[d[2] for d in dat],s=[35+85*d[3] for d in dat],alpha=.78)
for f,j,fr,e in dat:
    if f=="17": ax.annotate("Fault 17",(j,fr),xytext=(5,-14),textcoords="offset points",fontweight="bold")
    if f=="20": ax.annotate("Fault 20",(j,fr),xytext=(5,3),textcoords="offset points",fontweight="bold")
ax.axvline(.8570028011204481,linestyle="--",linewidth=.9); ax.axhline(96.47058823529412,linestyle="--",linewidth=.9)
ax.text(.54,69,"Adaptive-5 overall\nfamily recovery 96.47%\nexact Top-5 recovery 62.35%\nmean Jaccard 0.857",fontsize=6.5,va="center")
ax.set_xlim(.42,1.03); ax.set_ylim(35,103); ax.set_xlabel("Mean Top-5 Jaccard overlap"); ax.set_ylabel("Original-family recovery (%)")
ax.set_title("C  U2: family identity is more stable than coordinates",loc="left",fontweight="bold")
ax.grid(True,alpha=.22); ax.text(.02,.02,"Marker size ∝ exact Top-5 recovery",transform=ax.transAxes,fontsize=6.4); save(fig,"C")

# D
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([.15,.18,.81,.70])
branches=["adaptive_5family","fixed_absolute_deviation","restricted_ad_or_sl"]; markers=["o","s","^"]
offs={"adaptive_5family":(-45,-10),"fixed_absolute_deviation":(5,-10),"restricted_ad_or_sl":(5,5)}
for i,b in enumerate(branches):
    xm={r["submetric"]:float(r["value"])*100 for r in get("D",branch=b,metric="detection_within_15min_rate")}
    ym={r["submetric"]:float(r["value"])*100 for r in get("D",branch=b,metric="threshold_occupancy")}
    reps=sorted(set(xm)&set(ym),key=lambda q:int(q.split("_")[1]))
    x=np.array([xm[q] for q in reps]); y=np.array([ym[q] for q in reps])
    ax.scatter(x,y,marker=markers[i],s=28,alpha=.65,label=short[b]); mx,my=x.mean(),y.mean()
    ax.scatter([mx],[my],marker=markers[i],s=90,facecolors="none",linewidths=1.2)
    ax.annotate(short[b],(mx,my),xytext=offs[b],textcoords="offset points",fontsize=6.4)
ax.set_xlabel("15-min run detection (%)"); ax.set_ylabel("Post-fault threshold occupancy (%)")
ax.set_title("D  U2: repeated-split operating regions",loc="left",fontweight="bold")
ax.grid(True,alpha=.22); ax.legend(loc="lower left",frameon=False)
ax.text(.98,.03,"10 splits; open markers = means",transform=ax.transAxes,ha="right",fontsize=6.2); save(fig,"D")

# Compose 178 x 144 mm vector PDF.
mm=72/25.4; pw=178*mm; ph=144*mm; mx=9*mm; mt=8*mm; mb=8*mm; gx=6*mm; gy=7*mm
cw=(pw-2*mx-gx)/2; ch=(ph-mt-mb-gy)/2
rects=[fitz.Rect(mx,mt,mx+cw,mt+ch),fitz.Rect(mx+cw+gx,mt,pw-mx,mt+ch),
       fitz.Rect(mx,mt+ch+gy,mx+cw,ph-mb),fitz.Rect(mx+cw+gx,mt+ch+gy,pw-mx,ph-mb)]
doc=fitz.open(); p=doc.new_page(width=pw,height=ph)
for rect,name in zip(rects,["A","B","C","D"]):
    s=fitz.open(TMP/f"{name}.pdf"); p.show_pdf_page(rect,s,0,keep_proportion=True); s.close()
pdf=HERE/"P0-F09_family_simplification_full_pipeline_stability.pdf"; doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P0-F09_family_simplification_full_pipeline_stability.svg").write_text(d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(HERE/"P0-F09_family_simplification_full_pipeline_stability.png")
d.close()
shutil.rmtree(TMP, ignore_errors=True)

