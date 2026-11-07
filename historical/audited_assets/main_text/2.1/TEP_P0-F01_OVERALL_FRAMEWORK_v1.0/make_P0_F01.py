from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
with open(HERE/"P0-F01_source_nodes.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
nodes=[(r["node_id"],float(r["x"]),float(r["y"]),float(r["width"]),float(r["height"]),
        r["evidence_class"],r["title"],r["subtitle"]) for r in R]

P={"normal":"#4C78A8","development":"#F28E2B","validation":"#7A6DAA",
   "freeze":"#66589C","primary":"#3B8F70","postfreeze":"#E2EEE9",
   "secondary":"#EEF0F2","exploratory":"#F2EFF8","text":"#26323D",
   "edge":"#56616C","firewall":"#2F3439","red":"#B54A4A"}

fig=plt.figure(figsize=(178/25.4,110/25.4))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.text(.025,.965,"Development and freezing",fontsize=10.2,fontweight="bold",color=P["normal"],va="top")
ax.text(.735,.965,"Evaluation / audits",fontsize=10.2,fontweight="bold",color=P["primary"],va="top")

for nid,x,y,w,h,kind,title,subtitle in nodes:
    if kind in {"postfreeze","secondary","exploratory"}:
        fill=P[kind]; ec={"postfreeze":"#6E8F84","secondary":"#6E7781","exploratory":"#7A6DAA"}[kind]; tc=P["text"]
    else:
        fill=P[kind]; ec=fill; tc="white"
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.014",
                                facecolor=fill,edgecolor=ec,linewidth=1.15))
    ax.text(x+w/2,y+h*.64,title,ha="center",va="center",fontsize=7.6,fontweight="bold",color=tc)
    ax.text(x+w/2,y+h*.29,subtitle,ha="center",va="center",fontsize=5.85,color=tc,linespacing=1.0)

M={r[0]:r for r in nodes}
def a(n,side,frac=.5):
    _,x,y,w,h,*_=M[n]
    if side=="left": return (x,y+h*frac)
    if side=="right": return (x+w,y+h*frac)
    if side=="top": return (x+w*frac,y+h)
    return (x+w*frac,y)

def ar(p1,p2,ls="-",color=None,lw=1.2,rad=0):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=9.5,
                                 shrinkA=1.5,shrinkB=1.5,linewidth=lw,
                                 linestyle=ls,color=color or P["edge"],
                                 connectionstyle=f"arc3,rad={rad}"))

ar(a("N1","right"),a("N2","left")); ar(a("N2","bottom"),a("N4","top"))
ar(a("N3","right"),a("N4","left")); ar(a("N4","right"),a("N5","left"))
ar(a("N6","right"),a("N7","left")); ar(a("N5","bottom"),a("N8","top"))
ar(a("N7","right"),a("N8","left"))

fw=.685
ax.plot([fw,fw],[.18,.90],color=P["firewall"],linewidth=1.8)
ax.text(.676,.84,"DATA\nFIREWALL",ha="right",va="center",fontsize=7.6,
        fontweight="bold",color=P["firewall"],linespacing=.95)
ax.text(.696,.80,"final access\nonly after freeze",ha="left",va="center",
        fontsize=5.4,color=P["firewall"],linespacing=1.0)

ar(a("N8","right"),(.725,.635),color=P["primary"],lw=1.6)
ax.text(.653,.58,"frozen\nstate",fontsize=5.5,ha="center",va="center",color=P["primary"])
ar(a("N9","bottom"),a("N10","top"),color=P["primary"],lw=1.6)

ax.text(.735,.555,"Primary endpoints:",fontsize=5.55,fontweight="bold",
        ha="left",va="top",color=P["text"])
ax.text(.735,.536,"FAR | early detection\nthreshold occupancy | penalized delay",
        fontsize=5.35,ha="left",va="top",color=P["text"],linespacing=1.0)

ar(a("N10","bottom",.88),a("N11","top",.88),ls="--",color="#6E7781")
ar(a("N10","right",.35),a("N12","right",.65),ls="--",color="#6E7781",rad=-0.22)
ar(a("N12","bottom"),a("N13","top"),ls=":",color=P["validation"])

ax.annotate("",xy=(.655,.445),xytext=(.735,.445),
            arrowprops=dict(arrowstyle="-|>",lw=1.25,color=P["red"],linestyle="--"))
ax.text(.695,.462,"no feedback",ha="center",va="bottom",
        fontsize=6.0,fontweight="bold",color=P["red"])

ly=.025
ax.add_line(Line2D([.025,.075],[ly,ly],color=P["primary"],lw=1.7))
ax.text(.082,ly,"primary frozen path",fontsize=5.7,va="center")
ax.add_line(Line2D([.245,.295],[ly,ly],color="#6E7781",lw=1.25,ls="--"))
ax.text(.302,ly,"post-freeze / secondary",fontsize=5.7,va="center")
ax.add_line(Line2D([.49,.54],[ly,ly],color=P["validation"],lw=1.3,ls=":"))
ax.text(.547,ly,"exploratory",fontsize=5.7,va="center")

fig.savefig(HERE/"P0-F01_leakage_controlled_framework.pdf")
fig.savefig(HERE/"P0-F01_leakage_controlled_framework.svg")
fig.savefig(HERE/"P0-F01_leakage_controlled_framework.png",dpi=600)
plt.close(fig)
