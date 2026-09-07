import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); SC=Path(sys.argv[1])
from mpcc_tuning.track import Track
# source raceline + speed + real pink edges
base=Track.icra_t2_raceline_mapped(); RL=base.raceline; VREF=np.asarray(base.v_ref)
pk=np.load(SC/"pink_world.npz"); P=np.column_stack([pk["wx"],pk["wy"]])
# resample raceline to uniform arc length, light smooth (it is already smooth: min R 0.93 m)
seg=np.linalg.norm(np.diff(np.vstack([RL,RL[:1]]),axis=0),axis=1)
sraw=np.concatenate([[0],np.cumsum(seg)]); L=sraw[-1]; ds=0.1; ng=int(L/ds)
grid=np.arange(ng)*ds; closed=np.vstack([RL,RL[:1]]); vcl=np.concatenate([VREF,VREF[:1]])
rx=np.interp(grid,sraw,closed[:,0]); ry=np.interp(grid,sraw,closed[:,1]); vref=np.interp(grid,sraw,vcl)
# tiny smooth of the reference itself
rx=gaussian_filter1d(rx,2,mode="wrap"); ry=gaussian_filter1d(ry,2,mode="wrap")
# build a temp Track with the RACELINE as centreline -> spline pos/normal
T=Track(rx,ry,ds=ds); S=T.s; Nc=len(S)
pos=np.array([[float(T.pos(float(s))[0]),float(T.pos(float(s))[1])] for s in S])
phi=np.array([float(T.tangent_angle(float(s))) for s in S])
tang=np.column_stack([np.cos(phi),np.sin(phi)]); nrm=np.column_stack([-np.sin(phi),np.cos(phi)])
# widths = distance to the REAL edges (pink), nearest per side
tree=cKDTree(P); WIN=0.22; RAD=4.5
wl=np.full(Nc,1.0); wr=np.full(Nc,1.0)
for i in range(Nc):
    idx=tree.query_ball_point(pos[i],RAD)
    if not idx: continue
    rel=P[idx]-pos[i]; along=rel@tang[i]; perp=rel@nrm[i]; m=np.abs(along)<WIN
    if not m.any(): continue
    pp=perp[m]; plus=pp[pp>0.08]; minus=pp[pp<-0.08]
    if len(plus):  wr[i]=float(plus.min())
    if len(minus): wl[i]=float((-minus).min())
# smooth boundaries; cap to real walls (<=3 m); floor small
wl=np.clip(gaussian_filter1d(np.minimum(wl,3.0),5,mode="wrap"),0.20,3.0)
wr=np.clip(gaussian_filter1d(np.minimum(wr,3.0),5,mode="wrap"),0.20,3.0)
out=ROOT/"mpcc_tuning/tracks/icra_t2_raceline_ref_corridor.npz"
np.savez(str(out), cx=rx,cy=ry, wl=wl,wr=wr, ds=ds, length=float(L),
         raceline=RL, vref=vref)   # note: cx/cy IS the raceline (reference); vref aligned to it
print(f"raceline-ref: {Nc} pts, length {L:.1f} m, min radius {1/np.max([abs(float(T.curvature(float(s)))) for s in S]):.2f} m")
print(f"widths: wl {wl.min():.2f}-{wl.max():.2f}, wr {wr.min():.2f}-{wr.max():.2f}; half {(0.5*(wl+wr)).min():.2f}-{(0.5*(wl+wr)).max():.2f}")
