import sys, os, numpy as np, csv
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
# WHERE does the driven line run closest to a wall? Drive the widest clean line
# (k_v=0.55) and, at every step, record the clearance to the NEARER corridor wall
# (min over the two sides, minus the 0.12 car margin already used by the off-test).
# Then pick 3 WELL-SEPARATED arc-length sites of minimum clearance -> those are where
# moving a wall in a few cm will actually clip the fixed constant.
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth(); ik=WEIGHT_NAMES.index("k_v")
m=AcadosMPCC(t,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="fnw_%d"%os.getpid())
th=np.asarray(st.theta(),float).copy(); th[ik]=np.log(0.55)
P=ScuderiaPlant(t,model="std",dt=0.05); P.max_steps=st.steps
s5=P.reset(s0=0.0,v0=1.0); m.reset()
S=[]; CLR=[]; SIDE=[]; LAT=[]
for _ in range(st.steps):
    u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
    x,y=float(P._x[0]),float(P._x[1]); sp=t.project(x,y); lat=float(t.lateral(x,y))
    wl,wr=t.width(sp)
    # off-test: off if lat > wr-0.12 (toward +normal/wr) or -lat > wl-0.12 (toward wl)
    clr_wr=float(wr)-lat        # gap to +normal (wr) wall, before the 0.12 margin
    clr_wl=float(wl)+lat        # gap to -normal (wl) wall
    if clr_wr < clr_wl: S.append(sp); CLR.append(clr_wr); SIDE.append("+n(wr)"); LAT.append(lat)
    else:               S.append(sp); CLR.append(clr_wl); SIDE.append("-n(wl)"); LAT.append(lat)
    if off or tr: break
S=np.array(S); CLR=np.array(CLR)
# keep only the first full lap of samples (unique-ish s), sort by s
order=np.argsort(S); Ss=S[order]; Cs=CLR[order]
# find 3 well-separated minima: greedily take global min, exclude +-6 m, repeat
picks=[]; mask=np.ones(len(Ss),bool)
for _ in range(3):
    if not mask.any(): break
    idx=np.argmin(np.where(mask,Cs,np.inf)); si=Ss[idx]; picks.append(idx)
    mask &= np.abs(((Ss-si+t.length/2)%t.length)-t.length/2) > 6.0
print("length=%.1f  min clearance sites (s, clearance_m, side):"%t.length)
rows=[]
for idx in picks:
    si=Ss[idx]; ci=Cs[idx]
    # side + lateral at that s (nearest sample)
    j=int(np.argmin(np.abs(S-si)))
    print("  s=%5.1f  clearance=%.2f m (%2.0f cm)  side=%s  lat=%.2f"%(si,ci,ci*100,SIDE[j],LAT[j]))
    rows.append((round(float(si),1),round(float(ci),3),SIDE[j],round(LAT[j],3)))
with open(ROOT/"results/paper_smooth/near_wall_sites.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["s","clearance_m","side","lateral_m"]); w.writerows(rows)
print("saved results/paper_smooth/near_wall_sites.csv")
