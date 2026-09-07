import sys, numpy as np
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); SC=Path(sys.argv[1])
import importlib.util
spec=importlib.util.spec_from_file_location("cl", str(ROOT/"tools/centerline_from_map.py"))
cl=importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
from mpcc_tuning.track import Track
im,res,org=cl.load(str(ROOT/"mpcc_tuning/tracks/icra2026_t2.pgm"),str(ROOT/"mpcc_tuning/tracks/icra2026_t2.yaml"))
H,W=im.shape
pk=np.load(SC/"pink_world.npz"); P=np.column_stack([pk["wx"],pk["wy"]])
pink=np.zeros((H,W),bool)
cc=((P[:,0]-org[0])/res).astype(int); rr=((org[1]+H*res-P[:,1])/res).astype(int)
ok=(rr>=0)&(rr<H)&(cc>=0)&(cc<W); pink[rr[ok],cc[ok]]=True
for _ in range(3): pink=pink|np.roll(pink,1,0)|np.roll(pink,-1,0)|np.roll(pink,1,1)|np.roll(pink,-1,1)
d=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz")
cx,cy=d["cx"],d["cy"]; Nc=len(cx); wl0=d["wl"].copy(); wr0=d["wr"].copy()
t=Track.icra_t2_raceline_mapped()   # SPLINE geometry (the MPCC's own normal, non-flipping)
S=t.s
pos=np.array([[float(t.pos(float(s))[0]),float(t.pos(float(s))[1])] for s in S])
phi=np.array([float(t.tangent_angle(float(s))) for s in S])
nrm=np.column_stack([-np.sin(phi),np.cos(phi)])
def march(p,dvec,maxm=4.0):
    for i in range(1,int(maxm/res)):
        w=p+dvec*(i*res); c=int((w[0]-org[0])/res); r=int((org[1]+H*res-w[1])/res)
        if not(0<=r<H and 0<=c<W): return maxm
        if pink[r,c]: return i*res
    return maxm
wr=np.array([march(pos[i], nrm[i]) for i in range(Nc)])
wl=np.array([march(pos[i],-nrm[i]) for i in range(Nc)])
wr=np.where(wr>=3.99, wr0, wr); wl=np.where(wl>=3.99, wl0, wl)
def med(a,m=5): pad=np.concatenate([a[-(m//2):],a,a[:m//2]]); return np.array([np.median(pad[i:i+m]) for i in range(len(a))])
from scipy.ndimage import maximum_filter1d
def repair(w):
    w=med(w,5).astype(float)
    mx=maximum_filter1d(w, size=25, mode="wrap")      # local envelope
    bad=w < 0.6*mx                                     # downward notch
    if bad.any():
        good=~bad; xi=np.arange(len(w))
        w[bad]=np.interp(xi[bad], xi[good], w[good], period=len(w))
    return med(w,5)
wr=np.maximum(repair(wr),0.30); wl=np.maximum(repair(wl),0.30)
np.savez(str(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz"),
         cx=cx,cy=cy, wl=wl, wr=wr, ds=float(d["ds"]), length=float(d["length"]),
         raceline=d["raceline"], vref=d["vref"])
print(f"widths: wl {wl.min():.2f}-{wl.max():.2f}, wr {wr.min():.2f}-{wr.max():.2f}; half {(0.5*(wl+wr)).min():.2f}-{(0.5*(wl+wr)).max():.2f}")
