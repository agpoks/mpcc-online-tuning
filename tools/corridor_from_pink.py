import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter1d
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); SC=Path(sys.argv[1])
from mpcc_tuning.track import Track
pk=np.load(SC/"pink_world.npz"); P=np.column_stack([pk["wx"],pk["wy"]])
d=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz")
cx,cy=d["cx"],d["cy"]; Nc=len(cx); wl0=d["wl"].copy(); wr0=d["wr"].copy()
t=Track.icra_t2_raceline_mapped()   # SPLINE geometry (MPCC's own normal)
S=t.s
pos=np.array([[float(t.pos(float(s))[0]),float(t.pos(float(s))[1])] for s in S])
phi=np.array([float(t.tangent_angle(float(s))) for s in S])
tang=np.column_stack([np.cos(phi),np.sin(phi)]); nrm=np.column_stack([-np.sin(phi),np.cos(phi)])
tree=cKDTree(P)
WIN=0.22; RAD=4.5
wl=wl0.copy(); wr=wr0.copy()
for i in range(Nc):
    idx=tree.query_ball_point(pos[i], RAD)
    if not idx: continue
    rel=P[idx]-pos[i]; along=rel@tang[i]; perp=rel@nrm[i]
    m=np.abs(along)<WIN
    if not m.any(): continue
    pp=perp[m]
    plus=pp[pp>0.12]; minus=pp[pp<-0.12]
    if len(plus):  wr[i]=float(plus.min())      # nearest pink on +n
    if len(minus): wl[i]=float((-minus).min())  # nearest pink on -n
def med(a,m=5): pad=np.concatenate([a[-(m//2):],a,a[:m//2]]); return np.array([np.median(pad[i:i+m]) for i in range(len(a))])
def repair(w, max_run=8):   # fix ONLY short downward notches (hairpin tip); keep wide sections
    w=med(w,5).astype(float); N=len(w)
    mx=maximum_filter1d(w,size=25,mode="wrap")
    rm=np.array([np.median(np.concatenate([w[-12:],w,w[:12]])[k:k+25]) for k in range(N)])
    bad=(w<0.55*mx) | (w>1.7*rm)   # short downward notch OR upward cross-grab spike
    j=0
    while j<N:
        if bad[j]:
            k=j
            while k<N and bad[k]: k+=1
            if k-j<=max_run:
                lo=w[(j-1)%N]; hi=w[k%N]
                for m2 in range(j,k):
                    f=(m2-j+1)/(k-j+1); w[m2]=lo*(1-f)+hi*f
            j=k
        else: j+=1
    return med(w,5)
wl=np.clip(repair(np.minimum(wl,3.0)),0.30,3.0); wr=np.clip(repair(np.minimum(wr,3.0)),0.30,3.0)
np.savez(str(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz"),
         cx=cx,cy=cy, wl=wl, wr=wr, ds=float(d["ds"]), length=float(d["length"]),
         raceline=d["raceline"], vref=d["vref"])
nfloor=int((wr<=0.31).sum()+(wl<=0.31).sum())
print(f"widths: wl {wl.min():.2f}-{wl.max():.2f}, wr {wr.min():.2f}-{wr.max():.2f}; floor-pts {nfloor}; half {(0.5*(wl+wr)).min():.2f}-{(0.5*(wl+wr)).max():.2f}")
