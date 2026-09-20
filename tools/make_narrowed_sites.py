import sys, os, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
TR=ROOT/"mpcc_tuning/tracks"
# Build a narrowed corridor that moves a wall IN at 3 arbitrary sites (s, side) chosen
# where the driven line runs closest to a wall (results/paper_smooth/near_wall_sites.csv),
# each by delta = clearance + BITE so the wall ends BITE metres inside the driven line.
# This is the STRONG (>~40 cm) geometry disturbance: unlike the curvature-picked 20/18/15
# cm case, the move is placed where it actually intrudes on the line, so the fixed constant
# must clip. HALF=1.5 -> a 3 m stretch per site. floor keeps the corridor physically drivable.
BITE=float(os.environ.get("GEOM_BITE","0.12"))
HALF=float(os.environ.get("GEOM_HALF","1.5"))
FLOOR=float(os.environ.get("GEOM_FLOOR","0.10"))
OUTTAG=os.environ.get("GEOM_OUTTAG","3big")
# explicit per-site deltas (metres, in near_wall_sites.csv order) override clearance+BITE
EXPL=[float(x) for x in os.environ["GEOM_DELTAS"].split(",")] if os.environ.get("GEOM_DELTAS") else None
t=Track.icra_t2_smooth()
sites=list(csv:=__import__("csv").DictReader(open(ROOT/"results/paper_smooth/near_wall_sites.csv")))
d=np.load(TR/"icra_t2_raceline_ref_corridor.npz")
cx,cy,wl,wr=d["cx"].astype(float),d["cy"].astype(float),d["wl"].astype(float),d["wr"].astype(float)
sc=np.linspace(0,t.length,len(wl),endpoint=False)
wl2,wr2=wl.copy(),wr.copy()
S=[]; SIDES=[]; DELTAS=[]
for i,row in enumerate(sites):
    s0=float(row["s"]); side=row["side"]; clr=float(row["clearance_m"])
    delta=EXPL[i] if EXPL is not None else clr+BITE
    win=np.abs(((sc-s0+t.length/2)%t.length)-t.length/2) < HALF
    if "wr" in side: wr2[win]=np.maximum(wr2[win]-delta,FLOOR)
    else:            wl2[win]=np.maximum(wl2[win]-delta,FLOOR)
    S.append(s0); SIDES.append(side); DELTAS.append(round(delta,3))
    print(f" site s={s0:5.1f} side={side} clearance={clr*100:4.0f}cm -> move {delta*100:4.0f}cm (floor {FLOOR*100:.0f}cm)")
np.savez(str(TR/f"icra_t2_smooth_narrowed{OUTTAG}_corridor.npz"), cx=cx,cy=cy,wl=wl2,wr=wr2,
         ds=float(d["ds"]),length=float(d["length"]),raceline=d["raceline"],vref=d["vref"],
         sites=np.array(S), sides=np.array(SIDES), delta=np.array(DELTAS), half=HALF, floor=FLOOR)
print(f"saved icra_t2_smooth_narrowed{OUTTAG}_corridor.npz ; wr {wr.min():.2f}-{wr.max():.2f} -> {wr2.min():.2f}; wl {wl.min():.2f} -> {wl2.min():.2f}")
# map: real bg + nominal (grey) + narrowed (red) edges + site markers
def edges(WL,WR):
    tt=Track(cx,cy,ds=0.1,w_left=WL,w_right=WR); ss=np.linspace(0,tt.length,1600,endpoint=False); L=[];R=[]
    for si in ss:
        p=np.asarray(tt.pos(float(si))).ravel(); a=float(tt.tangent_angle(float(si))); nx,ny=-np.sin(a),np.cos(a)
        wl_,wr_=tt.width(float(si)); L.append([p[0]+nx*float(wr_),p[1]+ny*float(wr_)]); R.append([p[0]-nx*float(wl_),p[1]-ny*float(wl_)])
    return np.array(L),np.array(R)
Ln,Rn=edges(wl,wr); Lp,Rp=edges(wl2,wr2)
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res,ox,oy=0.05,-2.8,-7.25
fig,ax=plt.subplots(figsize=(11,9.5)); ax.imshow(im,cmap="gray",extent=[ox,ox+W*res,oy,oy+H*res],origin="upper",zorder=0)
ax.plot(Ln[:,0],Ln[:,1],color="0.55",lw=1.0,zorder=2); ax.plot(Rn[:,0],Rn[:,1],color="0.55",lw=1.0,zorder=2,label="nominal boundary")
ax.plot(Lp[:,0],Lp[:,1],color="tab:red",lw=1.9,zorder=3); ax.plot(Rp[:,0],Rp[:,1],color="tab:red",lw=1.9,zorder=3,label=f"narrowed boundary (moved in {'/'.join(f'{x*100:.0f}' for x in DELTAS)} cm)")
for s0,dl in zip(S,DELTAS):
    p=np.asarray(t.pos(float(s0))).ravel(); ax.scatter([p[0]],[p[1]],c="yellow",edgecolor="k",s=140,zorder=6)
    ax.annotate(f"s={s0:.0f}\nmoved {dl*100:.0f}cm",(p[0],p[1]),textcoords="offset points",xytext=(8,8),
                fontsize=8.5,bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="0.5",alpha=0.9),zorder=7)
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left",fontsize=10,framealpha=0.9)
ax.set_title("Strong 3-area geometry disturbance: wall moved into the DRIVEN LINE at 3 sites")
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(str(ROOT/f"results/paper_smooth/perturb_geometry_{OUTTAG}.{ext}"),dpi=140)
print(f"saved perturb_geometry_{OUTTAG}")
