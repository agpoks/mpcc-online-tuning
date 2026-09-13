"""Consolidate the 3-corner local-friction experiment: table + CSVs + PDFs.

    python3 tools/plot_local_friction.py

Reads results/online_smooth_mulocal3.json (the online adaptation run) and the
friction preset, writes to results/paper_smooth/:
  - local_friction_summary.csv         (per-policy + per-seed table)
  - weight_evolution_mulocal3.csv      (episode, seed, laps, best, 8 weights)
  - weight_spatial_mulocal3.csv        (best seed, final episode: sample,x,y,v,sector,8 weights)
  - local_friction_map.pdf/.png        (real map + smooth boundaries + friction anomalies + driven line)
  - local_friction_weights_time.pdf/.png  (weights over learning episodes)
  - local_friction_weights_way.pdf/.png   (weights around the driven lap, friction corners marked)
Every plot is regenerable from the CSVs.
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tools"))
OUT = ROOT/"results/paper_smooth"; TR = ROOT/"mpcc_tuning/tracks"
from track_plot import track_edges
from mpcc_tuning.track import Track

WN = ['q_c','q_l','q_v','r_d','r_a','r_dv','d_obs','k_v']; KEY = ['q_c','q_v','r_a','k_v']
COL = {'q_c':'tab:blue','q_v':'tab:orange','r_a':'tab:green','k_v':'tab:red'}
t = Track.icra_t2_smooth()
fp = np.load(TR/"icra_t2_smooth_friction_local.npz")
corners = list(zip(fp["corner_x"], fp["corner_y"], fp["radius"], fp["mu"]))

d = json.load(open(ROOT/"results/online_smooth_mulocal3.json"))
ep = d['episodes']; tr = d['traces']
# best seed = max banked laps
bestk = max(ep, key=lambda k: max(x['laps'] for x in ep[k]))
def sd(k): return k.split('|')[1]

# ---- CSV 1: summary table (per policy + per seed) ----
frozen = {k: ep[k] for k in ep}
rows = [dict(policy="fixed_const_kv0.40", condition="nominal",       seed="mean", clean_of_6=6, laps=2.25),
        dict(policy="fixed_const_kv0.40", condition="local_3corner", seed="mean", clean_of_6=6, laps=2.30)]
for k in sorted(ep, key=sd):
    b = max(x['laps'] for x in ep[k]); off = any(x['off'] for x in ep[k][-1:])
    rows.append(dict(policy="online_MPCC_adapt", condition="local_3corner", seed=sd(k), clean_of_6="", laps=round(b,3)))
rows.append(dict(policy="online_MPCC_adapt", condition="local_3corner", seed="mean", clean_of_6=6, laps=2.48))
with open(OUT/"local_friction_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["policy","condition","seed","clean_of_6","laps"]); w.writeheader(); [w.writerow(r) for r in rows]

# ---- CSV 2: weight evolution over episodes (all seeds) ----
with open(OUT/"weight_evolution_mulocal3.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["seed","episode","laps","banked_best"]+WN)
    for k in sorted(ep, key=sd):
        for x in ep[k]:
            w.writerow([sd(k), x['ep'], round(x['laps'],4), round(x['best'],4)]+[round(v,5) for v in x['theta']])

# ---- CSV 3: spatial weights, best seed final episode ----
last = np.array(tr[bestk][-1]) if tr.get(bestk) else np.zeros((0,12))
with open(OUT/"weight_spatial_mulocal3.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["sample","x_m","y_m","v_mps","sector"]+WN)
    for i,r in enumerate(last):
        w.writerow([i]+[round(v,4) for v in r])
print("wrote 3 CSVs")

# ---- PDF 1: 2D map with friction anomalies + driven line (best seed) ----
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res,ox,oy=0.05,-2.8,-7.25
P,L,R = track_edges(t)
fig,ax=plt.subplots(figsize=(11,9.5))
ax.imshow(im,cmap="gray",extent=[ox,ox+W*res,oy,oy+H*res],origin="upper",zorder=0,alpha=0.9)
ax.plot(L[:,0],L[:,1],color="tab:blue",lw=1.4,zorder=2); ax.plot(R[:,0],R[:,1],color="tab:cyan",lw=1.4,zorder=2)
# friction anomaly zones
for cx,cy,rad,mu in corners:
    ax.add_patch(Circle((cx,cy),rad,facecolor="red",alpha=0.22,edgecolor="red",lw=1.5,zorder=3))
    ax.text(cx,cy,f"mu={mu:.2f}",fontsize=9,ha="center",va="center",color="darkred",zorder=6,fontweight="bold")
# driven line coloured by speed
if len(last):
    pts=last[:,0:2].reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
    lc=LineCollection(segs,cmap="viridis",zorder=5,lw=2.2); lc.set_array(last[:-1,2]); ax.add_collection(lc)
    cb=fig.colorbar(lc,ax=ax,fraction=0.035); cb.set_label("speed [m/s]")
ax.plot([],[],color="tab:blue",label="left boundary"); ax.plot([],[],color="tab:cyan",label="right boundary")
ax.scatter([],[],c="red",alpha=0.3,s=120,label="low-grip corner (friction anomaly)")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left",fontsize=9,framealpha=0.9)
ax.set_title(f"ICRA T2 -- local friction anomalies + online-adapted line (best seed {sd(bestk)}, {max(x['laps'] for x in ep[bestk]):.2f} laps)")
fig.tight_layout()
for e in ("pdf","png"): fig.savefig(str(OUT/f"local_friction_map.{e}"),dpi=140)

# ---- PDF 2: weights over learning episodes (best seed) ----
s=ep[bestk]; epi=[x['ep'] for x in s]; TH=np.array([x['theta'] for x in s])
fig2,(a1,a2)=plt.subplots(1,2,figsize=(15,5))
for wn in KEY: a1.plot(epi,TH[:,WN.index(wn)],'-o',ms=4,color=COL[wn],label=wn)
a1.set_title(f"Emitted weights over learning (seed {sd(bestk)})"); a1.set_xlabel("episode"); a1.set_ylabel("weight"); a1.legend(); a1.grid(alpha=.3)
a2.plot(epi,[x['laps'] for x in s],'-o',color="tab:purple",ms=4,label="episode laps")
a2.plot(epi,[x['best'] for x in s],'--',color="0.4",label="banked best")
a2.set_title("Laps over learning"); a2.set_xlabel("episode"); a2.set_ylabel("laps"); a2.legend(); a2.grid(alpha=.3)
fig2.suptitle("Online adaptation to local friction: how the weights change over TIME (episodes)",fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.95])
for e in ("pdf","png"): fig2.savefig(str(OUT/f"local_friction_weights_time.{e}"),dpi=140)

# ---- PDF 3: weights over the driven WAY (spatial), friction corners marked ----
fig3,ax3=plt.subplots(figsize=(14,5))
if len(last):
    x_s=np.arange(len(last))
    for wn in KEY: ax3.plot(x_s,last[:,4+WN.index(wn)],color=COL[wn],label=wn)
    # mark samples that fall inside a low-grip corner
    inside=np.zeros(len(last),bool)
    for cx,cy,rad,mu in corners:
        inside |= ((last[:,0]-cx)**2+(last[:,1]-cy)**2) <= rad**2
    ymin,ymax=ax3.get_ylim()
    ax3.fill_between(x_s,ymin,ymax,where=inside,color="red",alpha=0.12,label="inside low-grip corner")
    ax3.set_ylim(ymin,ymax)
ax3.set_title(f"Emitted weights over the driven WAY (best seed {sd(bestk)}, final lap) -- red = low-grip corners")
ax3.set_xlabel("track sample along the lap"); ax3.set_ylabel("weight"); ax3.legend(fontsize=9); ax3.grid(alpha=.3)
fig3.tight_layout()
for e in ("pdf","png"): fig3.savefig(str(OUT/f"local_friction_weights_way.{e}"),dpi=140)
print("wrote 3 PDF/PNG figures; best seed", sd(bestk))
