"""All 8 MPCC weights over learning AND over the driven way, by sector.

    python3 tools/plot_weights_all8.py

Corrects the earlier 4-weight view: shows every weight, normalized to START so it
is obvious which the tuner moves (and which saturate at the factor-2 box edge).
Reads results/online_smooth_mulocal3.json; writes to results/paper_smooth/:
  weights_all8_time.pdf/png, weights_all8_way.pdf/png, weights_per_sector.pdf/png,
  weights_per_sector.csv
"""
import sys, json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mpcc_tuning import baselines as B
OUT=ROOT/"results/paper_smooth"
WN=['q_c','q_l','q_v','r_d','r_a','r_dv','d_obs','k_v']
CM=plt.cm.tab10(np.linspace(0,1,8))
st=np.exp(np.asarray(B.start("icra_t2_smooth").theta(),float))   # START weights (absolute)
JSON=sys.argv[1] if len(sys.argv)>1 else "online_smooth_mulocal3.json"
d=json.load(open(ROOT/"results"/JSON))
ep=d['episodes']; tr=d['traces']
bestk=max(ep,key=lambda k:max(x['laps'] for x in ep[k])); sdlab=bestk.split('|')[1]
# tag the outputs by the run so different experiments don't overwrite each other
TAG=("_geom3" if "geom3" in JSON else "_f3" if "f3" in JSON
     else "_mulocal3" if "mulocal3" in JSON else "")
factor=float(sys.argv[2]) if len(sys.argv)>2 else 2.0

# ---- FIG A: all 8 over episodes, normalized to START ----
s=ep[bestk]; epi=[x['ep'] for x in s]; TH=np.array([x['theta'] for x in s])
fig,ax=plt.subplots(figsize=(13,6))
for i,wn in enumerate(WN):
    ax.plot(epi,TH[:,i]/st[i],'-o',ms=3,color=CM[i],label=f"{wn} (START {st[i]:.2g})")
ax.axhline(1.0,color='k',lw=0.8,ls='--',alpha=0.6); ax.axhline(factor,color='0.6',ls=':',label=f"box edge x{factor:g}"); ax.axhline(1/factor,color='0.6',ls=':')
ax.set_yscale('log'); ax.set_yticks([1/factor,1,factor]); ax.set_yticklabels([f'{1/factor:.2f}x (floor)','START',f'{factor:g}x (ceil)'])
ax.set_title(f"All 8 weights over learning, relative to START (seed {sdlab}) -- most move to the box edge; r_a stays mid-range")
ax.set_xlabel("episode"); ax.set_ylabel("emitted / START"); ax.legend(fontsize=8,ncol=2,loc="best"); ax.grid(alpha=.3)
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(str(OUT/f"weights_all8_time{TAG}.{e}"),dpi=140)

# ---- FIG B: all 8 over the driven way, own scale, sector bands ----
last=np.array(tr[bestk][-1]); secs=last[:,3].astype(int); x_s=np.arange(len(last))
# sector change points
bounds=np.where(np.diff(secs)!=0)[0]
fig2,axs=plt.subplots(4,2,figsize=(15,11)); axs=axs.ravel()
seccols={sc:plt.cm.Pastel1(i) for i,sc in enumerate(np.unique(secs))}
for i,wn in enumerate(WN):
    a=axs[i]; a.plot(x_s,last[:,4+i],color=CM[i],lw=1.4)
    a.axhline(st[i],color='k',ls='--',lw=0.8,alpha=0.5)
    # shade by sector
    start=0
    for b in list(bounds)+[len(secs)-1]:
        a.axvspan(start,b,color=seccols[secs[start]],alpha=0.25); start=b
    a.set_title(f"{wn}  (START {st[i]:.2g}, dashed)",fontsize=10); a.grid(alpha=.3); a.set_xlabel("sample")
fig2.suptitle(f"All 8 weights over the driven WAY (seed {sdlab}, final lap). Background = sector (0/2/3). Dashed = START.",fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.97])
for e in ("pdf","png"): fig2.savefig(str(OUT/f"weights_all8_way{TAG}.{e}"),dpi=135)

# ---- FIG C + CSV: per-sector mean weight (relative to START) ----
usec=sorted(np.unique(secs)); M=np.zeros((8,len(usec)))
for j,sc in enumerate(usec):
    mask=secs==sc
    for i in range(8): M[i,j]=last[mask,4+i].mean()/st[i]
fig3,ax3=plt.subplots(figsize=(7,6))
im=ax3.imshow(M,cmap="coolwarm",vmin=0.5,vmax=2.0,aspect="auto")
ax3.set_xticks(range(len(usec))); ax3.set_xticklabels([f"sector {s}" for s in usec])
ax3.set_yticks(range(8)); ax3.set_yticklabels(WN)
for i in range(8):
    for j in range(len(usec)): ax3.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=9)
fig3.colorbar(im,label="emitted / START (1=START, 2=ceil, 0.5=floor)")
ax3.set_title(f"Per-sector mean weight vs START (seed {sdlab})\nsame across sectors => NOT sector-specific")
fig3.tight_layout()
for e in ("pdf","png"): fig3.savefig(str(OUT/f"weights_per_sector{TAG}.{e}"),dpi=140)
with open(OUT/f"weights_per_sector{TAG}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["weight","START"]+[f"sector{s}_mean" for s in usec]+[f"sector{s}_rel" for s in usec])
    for i,wn in enumerate(WN):
        means=[last[secs==sc,4+i].mean() for sc in usec]
        w.writerow([wn,round(st[i],4)]+[round(m,4) for m in means]+[round(m/st[i],3) for m in means])
print("wrote weights_all8_time, weights_all8_way, weights_per_sector (+csv); best seed", sdlab, "sectors", usec)
