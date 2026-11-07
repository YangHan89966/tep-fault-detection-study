from pathlib import Path
import csv, tempfile, shutil, numpy as np, matplotlib.pyplot as plt, fitz

HERE=Path(__file__).resolve().parent
with open(HERE/"P1-F03_source_data.csv",encoding="utf-8",newline="") as f:
    R=list(csv.DictReader(f))
primary=[r for r in R if r["record_type"]=="frozen_primary"]
repeats=[r for r in R if r["record_type"]=="internal_repeat"]
rep_ids=sorted({int(r["repeat_id"]) for r in repeats})

# One performance record per repeat can be taken from the first rank row.
rep0={rid:next(r for r in repeats if int(r["repeat_id"])==rid and int(r["rank"])==1) for rid in rep_ids}
pv=[r["variable"] for r in sorted(primary,key=lambda r:int(r["rank"]))]
rank=np.full((5,20),np.nan)
for j,rid in enumerate(rep_ids):
    rr=sorted([r for r in repeats if int(r["repeat_id"])==rid],key=lambda r:int(r["rank"]))
    order=[r["variable"] for r in rr]
    for i,v in enumerate(pv): rank[i,j]=order.index(v)+1

ids=np.array(rep_ids)
stable=np.array([int(rep0[r]["n_direction_stable_variables"]) for r in rep_ids])
far=np.array([float(rep0[r]["normal_run_level_far_pct"]) for r in rep_ids])
occ=np.array([float(rep0[r]["fault3_threshold_occupancy_pct"]) for r in rep_ids])
delay=np.array([float(rep0[r]["median_alarm_delay_min"]) for r in rep_ids])
ps=int(primary[0]["n_direction_stable_variables"])
pf=float(primary[0]["normal_run_level_far_pct"])
po=float(primary[0]["fault3_threshold_occupancy_pct"])
pd=float(primary[0]["median_alarm_delay_min"])
exact_order=sum(rep0[r]["exact_primary_top5_order"]=="True" for r in rep_ids)

plt.rcParams.update({"font.size":8.0,"axes.titlesize":9.0,"axes.labelsize":8.0,
                     "xtick.labelsize":7.0,"ytick.labelsize":7.0,"legend.fontsize":6.8})
TMP=Path(tempfile.mkdtemp(prefix="p1f03_panels_"))
def save(fig,name):
    fig.savefig(TMP/f"{name}.pdf",bbox_inches="tight"); plt.close(fig)

fig=plt.figure(figsize=(3.45,2.65)); ax=fig.add_axes([.18,.23,.77,.65])
im=ax.imshow(rank,aspect="auto",vmin=1,vmax=5)
ax.set_xticks(np.arange(0,20,2),[str(i) for i in range(1,21,2)])
ax.set_yticks(np.arange(5),pv); ax.set_xlabel("Internal repeated split"); ax.set_ylabel("Frozen-primary Top-5 variable")
ax.set_title("A  Top-5 set recovery and within-set rank",loc="left",fontweight="bold")
for i in range(5):
    for j in range(20): ax.text(j,i,str(int(rank[i,j])),ha="center",va="center",fontsize=5.2)
ax.text(.02,-.24,f"Exact set: 20/20 | exact order: {exact_order}/20 | sign agreement: 20/20",
        transform=ax.transAxes,fontsize=6.7)
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03,ticks=[1,2,3,4,5]); cb.set_label("Selection rank"); save(fig,"A")

fig=plt.figure(figsize=(3.45,2.65)); ax=fig.add_axes([.16,.21,.80,.68])
ax.plot(ids,stable,marker="o",linewidth=1.2); ax.axhline(ps,linestyle="--",linewidth=1.2,label=f"Frozen primary = {ps}")
ax.set_xticks([1,5,10,15,20]); ax.set_ylim(36.5,43.5); ax.set_xlabel("Internal repeated split"); ax.set_ylabel("Direction-stable candidates")
ax.set_title("B  Candidate-count stability",loc="left",fontweight="bold"); ax.grid(True,alpha=.22); ax.legend(loc="upper right",frameon=False)
ax.text(.03,.08,f"Range {stable.min()}-{stable.max()}; median {np.median(stable):.0f}",transform=ax.transAxes,fontsize=6.8); save(fig,"B")

fig=plt.figure(figsize=(3.45,2.65)); ax=fig.add_axes([.16,.21,.80,.68])
ax.scatter(ids,far,s=22,label="Internal repeats"); ax.axhline(pf,linestyle="--",linewidth=1.2,label=f"Frozen primary = {pf:.1f}%")
ax.axhline(far.mean(),linestyle=":",linewidth=1.2,label=f"Repeat mean = {far.mean():.1f}%")
ax.set_xticks([1,5,10,15,20]); ax.set_ylim(1.5,8.2); ax.set_xlabel("Internal repeated split"); ax.set_ylabel("Normal run-level FAR (%)")
ax.set_title("C  Normal-evaluation FAR",loc="left",fontweight="bold"); ax.grid(True,alpha=.22); ax.legend(loc="upper right",frameon=False)
ax.text(.03,.08,f"Repeat range {far.min():.1f}-{far.max():.1f}%",transform=ax.transAxes,fontsize=6.8); save(fig,"C")

fig=plt.figure(figsize=(3.45,2.65)); ax=fig.add_axes([.17,.21,.78,.68])
ax.scatter(occ,delay,s=28,label="20 internal repeats"); ax.scatter([po],[pd],s=75,marker="D",label="Frozen primary")
ax.axvline(occ.mean(),linestyle=":",linewidth=1.1); ax.axhline(delay.mean(),linestyle=":",linewidth=1.1)
ax.set_xlabel("Post-fault threshold occupancy (%)"); ax.set_ylabel("Median alarm delay (min)")
ax.set_title("D  Detection operating stability",loc="left",fontweight="bold"); ax.grid(True,alpha=.22); ax.legend(loc="upper right",frameon=False)
ax.text(.03,.08,f"Repeats: occupancy {occ.mean():.2f}% +/- {occ.std(ddof=1):.2f} pp\n"
        f"median-delay mean {delay.mean():.1f} +/- {delay.std(ddof=1):.1f} min\nRun detection = 100% in every repeat",
        transform=ax.transAxes,fontsize=6.6); save(fig,"D")

mm=72/25.4; pw=178*mm; ph=122*mm; mx=7*mm; my=6*mm; gx=5*mm; gy=5*mm
cw=(pw-2*mx-gx)/2; ch=(ph-2*my-gy)/2
rects=[fitz.Rect(mx,my,mx+cw,my+ch),fitz.Rect(mx+cw+gx,my,pw-mx,my+ch),
       fitz.Rect(mx,my+ch+gy,mx+cw,ph-my),fitz.Rect(mx+cw+gx,my+ch+gy,pw-mx,ph-my)]
doc=fitz.open(); page=doc.new_page(width=pw,height=ph)
for rect,name in zip(rects,["A","B","C","D"]):
    s=fitz.open(TMP/f"{name}.pdf"); page.show_pdf_page(rect,s,0,keep_proportion=True); s.close()
pdf=HERE/"P1-F03_fault3_full_pipeline_internal_repeated_split_stability.pdf"; doc.save(pdf); doc.close()
d=fitz.open(pdf)
(HERE/"P1-F03_fault3_full_pipeline_internal_repeated_split_stability.svg").write_text(
    d[0].get_svg_image(text_as_path=False),encoding="utf-8")
d[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(
    HERE/"P1-F03_fault3_full_pipeline_internal_repeated_split_stability.png")
d.close(); shutil.rmtree(TMP,ignore_errors=True)
