import numpy as np
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import RacelineOpponent
tr=Track.icra_t2_smooth(); st=B.start("icra_t2_smooth"); th0=np.asarray(st.theta(),float)
m=AcadosMPCC(tr,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,
             max_obstacles=3,a_lat_sectors=[A_LAT_RACE]*4,**CORRIDOR_KW,name="multiobs_test")
P=ScuderiaPlant(tr,model="std",dt=0.05); P.reset(s0=0.0,v0=1.4); m.reset()
offs=[0.0,0.18,-0.18]
obs=[RacelineOpponent(tr, s0=(4.0+6.0*i)%tr.length, pace=0.0, offset=offs[i], radius=0.27, a_lat=2.5) for i in range(3)]
for o in obs: o.reset()
mind=[1e9,1e9,1e9]
for step in range(360):
    m.set_obstacles([o.keepout() for o in obs])
    u=m.value(P.state_dyn(),th0)["u0"]; P.step(u)
    ex,ey=float(P._x[0]),float(P._x[1])
    for i,o in enumerate(obs):
        ox,oy,_=o.keepout(); mind[i]=min(mind[i],np.hypot(ex-ox,ey-oy))
print("RESULT min ego-obstacle distance (keepout r=0.27, cleared iff > ~0.25):", flush=True)
for i,d in enumerate(mind): print(f"  obstacle {i} (offset {offs[i]:+.2f}): min dist {d:.2f} m  -> {'CLEARED' if d>0.25 else 'HIT/too close'}", flush=True)
print(f"VERDICT multi-obstacle: {'WORKS (all 3 respected)' if all(d>0.25 for d in mind) else 'FIX INCOMPLETE (1/2 ignored)'}", flush=True)
