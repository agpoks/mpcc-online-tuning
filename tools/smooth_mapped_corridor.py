import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
MODE=sys.argv[1] if len(sys.argv)>1 else "sweep"

d=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz")
cx0,cy0=d["cx"].astype(float),d["cy"].astype(float)
pk=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_pink_edges.npz"); P=np.column_stack([pk["wx"],pk["wy"]])
tree=cKDTree(P)
# resample centreline to uniform arc length ds=0.1
def resample(cx,cy,ds=0.1):
    C=np.column_stack([cx,cy]); seg=np.linalg.norm(np.diff(np.vstack([C,C[:1]]),axis=0),axis=1)
    s=np.concatenate([[0],np.cumsum(seg)]); L=s[-1]; ng=int(round(L/ds)); g=np.arange(ng)*ds
    cl=np.vstack([C,C[:1]])
    return np.interp(g,s,cl[:,0]), np.interp(g,s,cl[:,1]), L
rx0,ry0,L=resample(cx0,cy0)
vref0=np.interp(np.linspace(0,1,len(rx0),endpoint=False), np.linspace(0,1,len(d["vref"]),endpoint=False), d["vref"])

def med(a,m=5): pad=np.concatenate([a[-(m//2):],a,a[:m//2]]); return np.array([np.median(pad[i:i+m]) for i in range(len(a))])
def repair(w,max_run=8):
    w=med(w,5).astype(float); N=len(w); mx=maximum_filter1d(w,size=25,mode="wrap")
    rm=np.array([np.median(np.concatenate([w[-12:],w,w[:12]])[k:k+25]) for k in range(N)])
    bad=(w<0.55*mx)|(w>1.7*rm); j=0
    while j<N:
        if bad[j]:
            k=j
            while k<N and bad[k]: k+=1
            if k-j<=max_run:
                lo=w[(j-1)%N]; hi=w[k%N]
                for m2 in range(j,k): f=(m2-j+1)/(k-j+1); w[m2]=lo*(1-f)+hi*f
            j=k
        else: j+=1
    return med(w,5)

def widths_from(rx,ry):
    T=Track(rx,ry,ds=0.1); S=T.s; Nc=len(S)
    pos=np.array([[float(T.pos(float(s))[0]),float(T.pos(float(s))[1])] for s in S])
    phi=np.array([float(T.tangent_angle(float(s))) for s in S])
    tang=np.column_stack([np.cos(phi),np.sin(phi)]); nrm=np.column_stack([-np.sin(phi),np.cos(phi)])
    wl=np.full(Nc,1.0); wr=np.full(Nc,1.0); WIN=0.22; RAD=4.5
    for i in range(Nc):
        idx=tree.query_ball_point(pos[i],RAD)
        if not idx: continue
        rel=P[idx]-pos[i]; along=rel@tang[i]; perp=rel@nrm[i]; m=np.abs(along)<WIN
        if not m.any(): continue
        pp=perp[m]; plus=pp[pp>0.12]; minus=pp[pp<-0.12]
        if len(plus): wr[i]=float(plus.min())
        if len(minus): wl[i]=float((-minus).min())
    wl=np.clip(repair(np.minimum(wl,3.0)),0.30,3.0); wr=np.clip(repair(np.minimum(wr,3.0)),0.30,3.0)
    return T,S,wl,wr,pos,nrm

def edge_rough(T,S,wl,wr):
    n=1400; s=np.linspace(0,T.length,n,endpoint=False)
    L=[];R=[]
    for si in s:
        p=np.array(T.pos(si)).ravel(); a=float(T.tangent_angle(si)); nx,ny=-np.sin(a),np.cos(a)
        wl_i=float(np.interp(si%T.length,S,wl)); wr_i=float(np.interp(si%T.length,S,wr))
        L.append(p+[nx*wl_i,ny*wl_i]); R.append(p-[nx*wr_i,ny*wr_i])
    L=np.array(L);R=np.array(R)
    jl=np.mean(np.linalg.norm(np.diff(L,2,axis=0),axis=1)); jr=np.mean(np.linalg.norm(np.diff(R,2,axis=0),axis=1))
    return jl,jr

if MODE=="sweep":
    # baseline (sigma 0 = current)
    for sig in (0,2,3,4,6,8):
        if sig==0: rx,ry=rx0.copy(),ry0.copy()
        else: rx,ry=gaussian_filter1d(rx0,sig,mode="wrap"),gaussian_filter1d(ry0,sig,mode="wrap")
        T,S,wl,wr,pos,nrm=widths_from(rx,ry)
        # deviation of smoothed centre from original centre (max lateral shift)
        dev=np.array([np.min(np.hypot(cx0-px,cy0-py)) for px,py in zip(rx,ry)])
        kap=np.array([abs(float(T.curvature(float(s)))) for s in S]); Rmin=1/max(kap.max(),1e-9)
        jl,jr=edge_rough(T,S,wl,wr)
        foldR=np.mean(wr*kap>=1.0); foldL=np.mean(wl*kap>=1.0)
        print(f"sigma {sig}:  edgeRough L={jl:.4f} R={jr:.4f}  Rmin={Rmin:.2f}m  "
              f"centreShift max={dev.max():.3f} mean={dev.mean():.3f}  fold L={foldL:.1%} R={foldR:.1%}  "
              f"w[{wl.min():.2f},{max(wl.max(),wr.max()):.2f}]",flush=True)
else:
    sig=float(MODE)
    rx,ry=gaussian_filter1d(rx0,sig,mode="wrap"),gaussian_filter1d(ry0,sig,mode="wrap")
    T,S,wl,wr,pos,nrm=widths_from(rx,ry)
    out=ROOT/"mpcc_tuning/tracks/icra_t2_mapped_corridor.npz"
    np.savez(str(out), cx=rx,cy=ry, wl=wl,wr=wr, ds=0.1, length=float(L), raceline=d["raceline"], vref=vref0)
    jl,jr=edge_rough(T,S,wl,wr)
    print(f"SAVED smoothed mapped corridor sigma={sig}: {len(rx)} pts, edgeRough L={jl:.4f} R={jr:.4f}, "
          f"w wl[{wl.min():.2f},{wl.max():.2f}] wr[{wr.min():.2f},{wr.max():.2f}]")
