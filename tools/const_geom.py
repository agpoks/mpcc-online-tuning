import sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
# FIXED best-constant under the 3-area GEOMETRY change: the corridor wall moved
# in at 3 corners (icra_t2_smooth_narrowed3_corridor.npz). The controller plans on
# the NOMINAL track (m,t) -- it does NOT know the wall moved -- while the PLANT's
# collision boundary is the narrowed corridor. Mirrors tools/const_local_friction.py
# (source), swapping local friction for a moved wall. discrete=True (train integrator).
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth(); ik=WEIGHT_NAMES.index("k_v")
m=AcadosMPCC(t,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="cg3_%d"%os.getpid())
th0=np.asarray(st.theta(),float)
g=np.load(ROOT/"mpcc_tuning/tracks/icra_t2_smooth_narrowed3_corridor.npz")
tp=Track(g["cx"],g["cy"],ds=0.1,w_left=g["wl"],w_right=g["wr"])   # plant boundary = moved wall
def run(kv,narrow):
    th=th0.copy(); th[ik]=np.log(kv); pt=tp if narrow else t; out=[]
    for seed in range(6):
        s0=(seed%4)*t.length/4.0; v0=1.0+0.1*(seed%3)
        P=ScuderiaPlant(pt,model="std",dt=0.05); P.max_steps=st.steps
        s5=P.reset(s0=s0,v0=v0); m.reset(); b=float(s5[4]); off=tr=False
        for _ in range(st.steps):
            u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
            if off or tr: break
        out.append(((float(s5[4])-b)/t.length,bool(off)))
    cl=sum(1 for r in out if not r[1]); mean=np.mean([r[0] for r in out])
    print(" k_v=%.2f narrow=%s: %d/6 clean, mean %.2f, laps %s"%(
        kv,narrow,cl,mean,[round(r[0],2) for r in out]),flush=True)
print("FIXED nominal-tuned best-constant under 3-AREA GEOMETRY change (wall moved 20/18/15 cm), discrete, 6 seeds")
run(0.40,False); run(0.40,True); run(0.55,False); run(0.55,True)
