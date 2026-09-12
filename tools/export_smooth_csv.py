import sys, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
t=Track.icra_t2_smooth()
b=np.load(TR/"icra_t2_smooth_boundaries.npz")   # ref,left,right,wl,wr,s (1600 pts)
s=b["s"]; refx,refy=b["ref_x"],b["ref_y"]; lx,ly=b["left_x"],b["left_y"]; rx,ry=b["right_x"],b["right_y"]; wl,wr=b["wl"],b["wr"]
# v_ref (optimiser speed) resampled onto the reference s for the raceline CSV
vsrc=np.asarray(t.v_ref,float); vref=np.interp(np.linspace(0,1,len(s),endpoint=False),
                                               np.linspace(0,1,len(vsrc),endpoint=False), vsrc)
kap=np.array([float(t.curvature(float(si))) for si in s])
# --- CSV 1: the smooth raceline (MPCC reference) ---
rl_csv=TR/"icra_t2_smooth_raceline.csv"
np.savetxt(str(rl_csv), np.column_stack([s,refx,refy,vref,kap]),
           delimiter=",", header="s_m,x_m,y_m,v_ref_mps,kappa_radpm", comments="", fmt="%.6f")
# --- CSV 2: the smooth left/right boundaries ---
bd_csv=TR/"icra_t2_smooth_boundaries.csv"
np.savetxt(str(bd_csv), np.column_stack([s,refx,refy,lx,ly,rx,ry,wl,wr]),
           delimiter=",", header="s_m,ref_x,ref_y,left_x,left_y,right_x,right_y,w_left_m,w_right_m", comments="", fmt="%.6f")
def rough(X,Y): E=np.column_stack([X,Y]); return float(np.mean(np.linalg.norm(np.diff(E,2,axis=0),axis=1)))
print(f"saved {rl_csv.name} ({len(s)} pts) and {bd_csv.name}")
print(f"raceline roughness={rough(refx,refy):.4f}  left={rough(lx,ly):.4f}  right={rough(rx,ry):.4f}")
print(f"v_ref [{vref.min():.2f},{vref.max():.2f}] m/s (carried for reference; MPCC uses curvature ref, not this)")
# --- CONFIRMATION PLOT: real map + smooth raceline + smooth boundaries ---
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res=0.05; ox,oy=-2.8,-7.25
extent=[ox,ox+W*res,oy,oy+H*res]
fig,ax=plt.subplots(figsize=(11,9.5))
ax.imshow(im,cmap="gray",extent=extent,origin="upper",zorder=0)
ax.plot(lx,ly,color="tab:blue",lw=2.0,label="left boundary (smooth, real wall)",zorder=3)
ax.plot(rx,ry,color="tab:cyan",lw=2.0,label="right boundary (smooth, real wall)",zorder=3)
ax.plot(refx,refy,color="red",lw=1.8,label="smooth raceline = MPCC reference",zorder=4)
ax.scatter([refx[0]],[refy[0]],c="lime",s=90,edgecolor="k",zorder=5,label="start")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("CONFIRM: ICRA T2 smooth raceline + smooth boundaries on the real map\n"
             f"raceline rough {rough(refx,refy):.4f}, edges {rough(lx,ly):.4f}/{rough(rx,ry):.4f}")
ax.legend(loc="lower left",fontsize=10,framealpha=0.92)
fig.tight_layout(); f=ROOT/"paper/figures/t2_smooth_confirm.png"; fig.savefig(str(f),dpi=135); print("saved",f)
