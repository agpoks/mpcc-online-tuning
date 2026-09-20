import sys, os, numpy as np, csv
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
TR=ROOT/"mpcc_tuning/tracks"
# 2D map for the 3-area GEOMETRY use-case: real occupancy map + nominal boundary
# (grey) + narrowed boundary (red, wall moved in at 3 corners) + the driven line
# coloured by speed, with the clearance to each moved wall annotated. Shows WHY the
# move does not bite: the line stays well inside the removed strip.
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth(); ik=WEIGHT_NAMES.index("k_v")
GTRACK=os.environ.get("GEOM_TRACK","icra_t2_smooth_narrowed3mid_corridor.npz")
OUTTAG=os.environ.get("GEOM_OUTTAG","3mid")
g=np.load(TR/GTRACK)
corners=(g["sites"] if "sites" in g.files else g["corners"]).tolist(); deltas=g["delta"].tolist()
# boundary-aware MPCC: built on the NARROWED track so the plotted line is the actual
# feasible path the controller re-plans THROUGH the pinch (not a nominal-corridor line).
tp=Track(g["cx"],g["cy"],ds=0.1,w_left=g["wl"],w_right=g["wr"])
tp.raceline=g["raceline"]; tp.v_ref=g["vref"]; tp.width_vehicle_adjusted=False; tp.kv_max=0.55
m=AcadosMPCC(tp,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="gm_%d"%os.getpid())
th=np.asarray(st.theta(),float).copy(); th[ik]=np.log(0.55)   # widest-running clean line
P=ScuderiaPlant(tp,model="std",dt=0.05); P.max_steps=st.steps
s5=P.reset(s0=0.0,v0=1.0); m.reset()
XY=[]; V=[]
for _ in range(st.steps):
    u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
    XY.append([float(P._x[0]),float(P._x[1])]); V.append(float(P._x[3]))
    if off or tr: break
XY=np.array(XY); V=np.array(V)
def edges(WL,WR):
    tt=Track(g["cx"],g["cy"],ds=0.1,w_left=WL,w_right=WR); ss=np.linspace(0,tt.length,1400,endpoint=False); L=[];R=[]
    for si in ss:
        p=np.asarray(tt.pos(float(si))).ravel(); a=float(tt.tangent_angle(float(si))); nx,ny=-np.sin(a),np.cos(a)
        wl_,wr_=tt.width(float(si)); L.append([p[0]+nx*float(wr_),p[1]+ny*float(wr_)]); R.append([p[0]-nx*float(wl_),p[1]-ny*float(wl_)])
    return np.array(L),np.array(R)
d0=np.load(TR/"icra_t2_raceline_ref_corridor.npz")
Ln,Rn=edges(d0["wl"],d0["wr"]); Lp,Rp=edges(g["wl"],g["wr"])
im=np.array(Image.open(TR/"icra2026_t2.pgm")); H,W=im.shape; res,ox,oy=0.05,-2.8,-7.25
fig,ax=plt.subplots(figsize=(11,9.5)); ax.imshow(im,cmap="gray",extent=[ox,ox+W*res,oy,oy+H*res],origin="upper",zorder=0)
ax.plot(Ln[:,0],Ln[:,1],color="0.55",lw=1.0,zorder=2); ax.plot(Rn[:,0],Rn[:,1],color="0.55",lw=1.0,zorder=2,label="nominal boundary")
_dcm="/".join(f"{x*100:.0f}" for x in deltas)
ax.plot(Lp[:,0],Lp[:,1],color="tab:red",lw=1.9,zorder=3); ax.plot(Rp[:,0],Rp[:,1],color="tab:red",lw=1.9,zorder=3,label=f"narrowed boundary (moved in {_dcm} cm)")
seg=np.concatenate([XY[:-1,None,:],XY[1:,None,:]],axis=1)
lc=LineCollection(seg,cmap="viridis",zorder=4,lw=2.6); lc.set_array(V[:-1]); ax.add_collection(lc)
cb=fig.colorbar(lc,ax=ax,fraction=0.03,pad=0.02); cb.set_label("speed [m/s]")
for c,dl in zip(corners,deltas):
    p=np.asarray(t.pos(float(c))).ravel(); ax.scatter([p[0]],[p[1]],c="yellow",edgecolor="k",s=140,zorder=6)
    ax.annotate(f"s={float(c):.0f}\nmoved {dl*100:.0f}cm",(p[0],p[1]),textcoords="offset points",xytext=(8,8),
                fontsize=8.5,color="k",bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="0.5",alpha=0.9),zorder=7)
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left",fontsize=10,framealpha=0.9)
ax.set_title("Boundary-aware MPCC re-plans THROUGH the narrowing (wall moved in at 3 sites)")
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(str(ROOT/f"results/paper_smooth/geometry_{OUTTAG}_map.{ext}"),dpi=140)
print(f"saved geometry_{OUTTAG}_map")
