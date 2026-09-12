import sys, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
t=Track.icra_t2_smooth()
n=1600; s=np.linspace(0,t.length,n,endpoint=False)
P=[];Nn=[];WL=[];WR=[];KAP=[]
for si in s:
    p=np.array(t.pos(float(si))).ravel(); a=float(t.tangent_angle(float(si)))
    nx,ny=-np.sin(a),np.cos(a); wl,wr=t.width(float(si))
    P.append(p);Nn.append([nx,ny]);WL.append(float(wl));WR.append(float(wr));KAP.append(float(t.curvature(float(si))))
P=np.array(P);Nn=np.array(Nn);WL=np.array(WL);WR=np.array(WR);KAP=np.array(KAP)
# CORRECT convention (verified against the map, side_check FIXED panel):
#   +normal side wall is at distance wr ; -normal side wall at distance wl
# +normal = rotate tangent +90deg = LEFT of travel.
Lx,Ly=(P+Nn*WR[:,None]).T   # LEFT boundary  (+normal, dist wr)
Rx,Ry=(P-Nn*WL[:,None]).T   # RIGHT boundary (-normal, dist wl)
refx,refy=P[:,0],P[:,1]
vsrc=np.asarray(t.v_ref,float); vref=np.interp(np.linspace(0,1,n,endpoint=False),
                                               np.linspace(0,1,len(vsrc),endpoint=False), vsrc)
def rough(X,Y): E=np.column_stack([X,Y]); return float(np.mean(np.linalg.norm(np.diff(E,2,axis=0),axis=1)))
# npz
np.savez(str(TR/"icra_t2_smooth_boundaries.npz"), ref_x=refx,ref_y=refy,
         left_x=Lx,left_y=Ly, right_x=Rx,right_y=Ry, wl=WL,wr=WR, s=s, length=float(t.length))
# CSV: raceline
np.savetxt(str(TR/"icra_t2_smooth_raceline.csv"), np.column_stack([s,refx,refy,vref,KAP]),
           delimiter=",", header="s_m,x_m,y_m,v_ref_mps,kappa_radpm", comments="", fmt="%.6f")
# CSV: boundaries (left = +normal wall at dist wr; right = -normal wall at dist wl)
np.savetxt(str(TR/"icra_t2_smooth_boundaries.csv"),
           np.column_stack([s,refx,refy,Lx,Ly,Rx,Ry,WR,WL]),
           delimiter=",", header="s_m,ref_x,ref_y,left_x,left_y,right_x,right_y,w_left_m,w_right_m",
           comments="", fmt="%.6f")
print(f"re-exported. left_rough={rough(Lx,Ly):.4f} right_rough={rough(Rx,Ry):.4f}")
# plots on the real map
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res=0.05; ox,oy=-2.8,-7.25
ext=[ox,ox+W*res,oy,oy+H*res]
def mapfig(fn,title):
    fig,ax=plt.subplots(figsize=(11,9.5)); ax.imshow(im,cmap="gray",extent=ext,origin="upper",zorder=0)
    ax.plot(Lx,Ly,color="tab:blue",lw=2.0,label="left boundary (real wall)",zorder=3)
    ax.plot(Rx,Ry,color="tab:cyan",lw=2.0,label="right boundary (real wall)",zorder=3)
    ax.plot(refx,refy,color="red",lw=1.6,label="smooth raceline = MPCC reference",zorder=4)
    ax.scatter([refx[0]],[refy[0]],c="lime",s=90,edgecolor="k",zorder=5,label="start")
    ax.set_aspect("equal");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
    ax.set_title(title); ax.legend(loc="lower left",fontsize=10,framealpha=0.92)
    fig.tight_layout(); fig.savefig(str(ROOT/"paper/figures"/fn),dpi=135); print("saved",fn)
mapfig("t2_smooth_confirm.png", f"CONFIRM (fixed L/R): T2 real map + smooth raceline + boundaries\nedges {rough(Lx,Ly):.4f}/{rough(Rx,Ry):.4f}")
mapfig("t2_smooth_boundaries.png","ICRA T2 -- real map + SMOOTH MPCC boundaries (left/right corrected)")
