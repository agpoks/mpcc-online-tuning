import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); SC=Path(sys.argv[1])
pk=np.load(SC/"pink_world.npz"); P=np.column_stack([pk["wx"],pk["wy"]])
d=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz")
cx,cy=d["cx"],d["cy"]; ctr=np.column_stack([cx,cy])
wl0=d["wl"].copy(); wr0=d["wr"].copy()
g=np.gradient(ctr,axis=0); g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
tang=g; nrm=np.column_stack([-g[:,1],g[:,0]])
tree=cKDTree(P)
WIN=0.18; RAD=3.0
wl=wl0.copy(); wr=wr0.copy()
for i in range(len(ctr)):
    idx=tree.query_ball_point(ctr[i], RAD)
    if not idx: continue
    rel=P[idx]-ctr[i]; along=rel@tang[i]; perp=rel@nrm[i]
    m=np.abs(along)<WIN
    if not m.any(): continue
    pp=perp[m]
    plus=pp[pp>0.12]; minus=pp[pp<-0.12]
    if len(plus):  wr[i]=float(plus.min())        # NEAREST pink on +n side
    if len(minus): wl[i]=float((-minus).min())    # NEAREST pink on -n side
# smoothing (median to remove single-sample jitter), floor, and clamp isolated spikes
def sm(a,m=5,floor=0.30):
    pad=np.concatenate([a[-(m//2):],a,a[:m//2]]); a=np.array([np.median(pad[i:i+m]) for i in range(len(a))]); return np.maximum(a,floor)
# isolated-spike clamp vs rolling median (kills any residual cross-track grabs)
def declip(a,win=21,fac=1.4):
    pad=np.concatenate([a[-(win//2):],a,a[:win//2]]); med=np.array([np.median(pad[i:i+win]) for i in range(len(a))])
    return np.where(a>fac*med, med, a)
wl,wr=declip(sm(wl)),declip(sm(wr))
np.savez(str(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz"),
         cx=cx,cy=cy, wl=wl, wr=wr, ds=float(d["ds"]), length=float(d["length"]),
         raceline=d["raceline"], vref=d["vref"])
print(f"widths: wl {wl.min():.2f}-{wl.max():.2f}, wr {wr.min():.2f}-{wr.max():.2f}; half {(0.5*(wl+wr)).min():.2f}-{(0.5*(wl+wr)).max():.2f}")
