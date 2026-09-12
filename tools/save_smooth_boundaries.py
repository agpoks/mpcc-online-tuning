import sys, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
# --- smooth left/right boundaries from the raceline_ref corridor (the MPCC geometry) ---
t=Track.icra_t2_raceline_ref()   # smooth reference + smooth real-wall widths
n=1600; s=np.linspace(0,t.length,n,endpoint=False)
ref=[];L=[];R=[];WL=[];WR=[]
for si in s:
    p=np.array(t.pos(float(si))).ravel(); a=float(t.tangent_angle(float(si)))
    nx,ny=-np.sin(a),np.cos(a); wl,wr=t.width(float(si)); wl=float(wl);wr=float(wr)
    ref.append(p); L.append(p+[nx*wl,ny*wl]); R.append(p-[nx*wr,ny*wr]); WL.append(wl); WR.append(wr)
ref=np.array(ref);L=np.array(L);R=np.array(R);WL=np.array(WL);WR=np.array(WR)
out=TR/"icra_t2_smooth_boundaries.npz"
np.savez(str(out), ref_x=ref[:,0],ref_y=ref[:,1], left_x=L[:,0],left_y=L[:,1],
         right_x=R[:,0],right_y=R[:,1], wl=WL, wr=WR, s=s, length=float(t.length))
def rough(E): return float(np.mean(np.linalg.norm(np.diff(E,2,axis=0),axis=1)))
print(f"saved {out}  L_rough={rough(L):.4f} R_rough={rough(R):.4f}  wl[{WL.min():.2f},{WL.max():.2f}] wr[{WR.min():.2f},{WR.max():.2f}]")

# --- standard plot: REAL MAP + smooth left/right edges + reference ---
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape
res=0.05; ox,oy=-2.8,-7.25
extent=[ox, ox+W*res, oy, oy+H*res]
fig,ax=plt.subplots(figsize=(10,9))
ax.imshow(im, cmap="gray", extent=extent, origin="upper", zorder=0, alpha=0.9)
ax.plot(L[:,0],L[:,1],color="tab:blue",lw=1.8,label="left edge (smooth, real wall)",zorder=3)
ax.plot(R[:,0],R[:,1],color="tab:cyan",lw=1.8,label="right edge (smooth, real wall)",zorder=3)
ax.plot(ref[:,0],ref[:,1],'--',color="tab:orange",lw=1.0,label="MPCC reference (smooth raceline)",zorder=2)
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title(f"ICRA T2 -- real map + SMOOTH MPCC boundaries (edge roughness {rough(L):.4f})")
ax.legend(loc="lower left",fontsize=9,framealpha=0.9)
fig.tight_layout(); f=ROOT/"paper/figures/t2_smooth_boundaries.png"; fig.savefig(str(f),dpi=130); print("saved",f)
