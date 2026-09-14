import sys, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
t=Track.icra_t2_smooth()
# sample geometry + curvature to pick N corners (N=2 default, "3" -> 3-area case)
NCORN=int(sys.argv[1]) if len(sys.argv)>1 else 2
n=len(t.s) if hasattr(t,'s') else 700
s=np.linspace(0,t.length,800,endpoint=False)
kap=np.array([abs(float(t.curvature(float(si)))) for si in s])
# pick the N highest-curvature corner CENTRES, well separated
order=np.argsort(-kap); picks=[]
for i in order:
    if all(abs((s[i]-s[j]+t.length/2)%t.length - t.length/2)>6.0 for j in picks): picks.append(i)
    if len(picks)==NCORN: break
corners=[s[i] for i in picks]
print("corner centres at s =", [round(c,1) for c in corners], "radius", [round(1/max(kap[i],1e-9),2) for i in picks])
# build perturbed widths: at each corner, move the INNER boundary in by 0.15 m over +-2.5 m
d=np.load(TR/"icra_t2_raceline_ref_corridor.npz")
cx,cy,wl,wr=d["cx"].astype(float),d["cy"].astype(float),d["wl"].astype(float),d["wr"].astype(float)
sc=np.linspace(0,t.length,len(wl),endpoint=False)
HALF=2.5
wl2,wr2=wl.copy(),wr.copy()
# the tube moved in by these amounts (one per corner). 2 corners: 20/15 cm;
# 3 corners: 20/18/15 cm.
DELTAS=([0.20,0.18,0.15] if NCORN>=3 else [0.20,0.15])[:NCORN]
for (c,ci),DELTA in zip(zip(corners,picks),DELTAS):
    # sign of curvature at corner -> inner side. +normal(left) inner if turning left.
    ksig=float(t.curvature(float(c)))
    win=np.abs(((sc-c+t.length/2)%t.length)-t.length/2)<HALF
    # inner boundary = the side the car turns toward. left turn (kappa>0): inner=+normal=wr side
    if ksig>0: wr2[win]=np.maximum(wr2[win]-DELTA,0.25)
    else:      wl2[win]=np.maximum(wl2[win]-DELTA,0.25)
TAG="" if NCORN==2 else str(NCORN)
np.savez(str(TR/f"icra_t2_smooth_narrowed{TAG}_corridor.npz"), cx=cx,cy=cy,wl=wl2,wr=wr2,
         ds=float(d["ds"]),length=float(d["length"]),raceline=d["raceline"],vref=d["vref"],
         corners=np.array(corners), delta=np.array(DELTAS), half=HALF)
print("saved narrowed corridor; wl", wl.min(),wl.max(),"->",wl2.min(),wl2.max())
# plot nominal vs narrowed edges on the map
def edges(cxx,cyy,WL,WR):
    tt=Track(cxx,cyy,ds=0.1,w_left=WL,w_right=WR); ss=np.linspace(0,tt.length,1400,endpoint=False); L=[];R=[]
    for si in ss:
        p=np.array(tt.pos(float(si))).ravel(); a=float(tt.tangent_angle(float(si))); nx,ny=-np.sin(a),np.cos(a)
        wl_,wr_=tt.width(float(si)); L.append([p[0]+nx*float(wr_),p[1]+ny*float(wr_)]); R.append([p[0]-nx*float(wl_),p[1]-ny*float(wl_)])
    return np.array(L),np.array(R)
Ln,Rn=edges(cx,cy,wl,wr); Lp,Rp=edges(cx,cy,wl2,wr2)
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res,ox,oy=0.05,-2.8,-7.25
fig,ax=plt.subplots(figsize=(11,9.5)); ax.imshow(im,cmap="gray",extent=[ox,ox+W*res,oy,oy+H*res],origin="upper",zorder=0)
ax.plot(Ln[:,0],Ln[:,1],color="0.5",lw=1.0,zorder=2); ax.plot(Rn[:,0],Rn[:,1],color="0.5",lw=1.0,zorder=2,label="nominal boundary")
_dcm="-".join(str(int(round(x*100))) for x in DELTAS)+" cm"
ax.plot(Lp[:,0],Lp[:,1],color="tab:red",lw=1.8,zorder=3); ax.plot(Rp[:,0],Rp[:,1],color="tab:red",lw=1.8,zorder=3,label=f"narrowed boundary (tube moved in {_dcm})")
for c in corners:
    p=np.array(t.pos(float(c))).ravel(); ax.scatter([p[0]],[p[1]],c="yellow",edgecolor="k",s=120,zorder=5)
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left",fontsize=10,framealpha=0.9)
ax.set_title(f"Use-case: geometry change -- boundary moved into the track at {NCORN} corners (yellow)")
fig.tight_layout()
_ptag="" if NCORN==2 else str(NCORN)
for ext in ("png","pdf"): fig.savefig(str(ROOT/f"results/paper_smooth/perturb_geometry{_ptag}.{ext}"),dpi=140)
print(f"saved perturb_geometry{_ptag}")
