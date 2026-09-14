import sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
# HOW CLOSE does the car get to the moved wall? Drive the k_v=0.55 constant on the
# NOMINAL track, log the signed lateral offset, and at each of the 3 narrowed corners
# report the min clearance to the INNER (moved) nominal wall and how it compares to the
# wall move delta. If min clearance > delta at every corner, the 15-20 cm move can never
# touch the driven line -> nominal and narrowed laps are identical (which they were).
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth(); ik=WEIGHT_NAMES.index("k_v")
m=AcadosMPCC(t,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="gc_%d"%os.getpid())
g=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_smooth_narrowed3_corridor.npz")
corners=g["corners"].tolist(); deltas=g["delta"].tolist(); HALF=float(g["half"])
th=np.asarray(st.theta(),float).copy(); th[ik]=np.log(0.55)
# drive one representative clean seed (seed 0), log lateral + arclength each step
P=ScuderiaPlant(t,model="std",dt=0.05); P.max_steps=st.steps
s5=P.reset(s0=0.0,v0=1.0); m.reset()
S=[]; LAT=[]
for _ in range(st.steps):
    u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
    x,y=float(P._x[0]),float(P._x[1]); sp=t.project(x,y)
    S.append(sp); LAT.append(float(t.lateral(x,y)))
    if off or tr: break
S=np.array(S); LAT=np.array(LAT)
print("corner_s  delta_cm  inner_side  nominal_wall_m  min|lat|_at_corner  clearance_to_wall_cm  bites?")
rows=[]
for c,dl in zip(corners,deltas):
    win=np.abs(((S-c+t.length/2)%t.length)-t.length/2) < HALF
    if not win.any():
        print(f" s={c:5.1f}  no samples in window"); continue
    lat_w=LAT[win]
    ksig=float(t.curvature(float(c)))
    wl,wr=t.width(t.project(*np.asarray(t.pos(float(c))).ravel()))
    # inner side = side the car turns toward. left turn (kappa>0): inner=+normal=wr
    inner_wall=float(wr) if ksig>0 else float(wl)
    side="+n(wr)" if ksig>0 else "-n(wl)"
    # lateral sign: +lat toward +normal. inner clearance = wall - lateral_on_inner_side
    lat_inner = lat_w if ksig>0 else -lat_w   # positive = toward inner wall
    max_reach = float(np.max(lat_inner))       # closest approach to inner wall
    clr = (inner_wall - max_reach)             # gap to nominal inner wall, metres
    bites = "YES" if clr*100 < dl*100 else "no"
    print(f" s={c:5.1f}  {dl*100:4.0f}     {side}       {inner_wall:5.2f}          {max_reach:5.2f}              {clr*100:6.1f}            {bites}")
    rows.append((round(c,1),round(dl*100,0),side,round(inner_wall,2),round(max_reach,2),round(clr*100,1),bites))
import csv
with open(ROOT/"results/paper_smooth/geom_clearance.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["corner_s","delta_cm","inner_side","nominal_wall_m","max_reach_m","clearance_cm","bites"])
    w.writerows(rows)
print("saved results/paper_smooth/geom_clearance.csv")
