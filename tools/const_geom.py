import sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
# FIXED best-constant under the 3-site GEOMETRY change, BOUNDARY-AWARE: the MPCC is
# told the moved walls (its corridor constraint uses the narrowed widths), so it
# plans a feasible path AROUND the narrowing. Both the controller and the plant use
# the narrowed track. The question is not "does it crash into the wall" (it knows the
# wall) but "can the FIXED weight mix drive the narrowing without the QP going
# infeasible / losing time" -- vs the online tuner, which adapts the weights.
# discrete=True (train integrator). Mirrors tools/const_local_friction.py (source).
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth(); ik=WEIGHT_NAMES.index("k_v")
th0=np.asarray(st.theta(),float)
GTRACK=os.environ.get("GEOM_TRACK","icra_t2_smooth_narrowed3mid_corridor.npz")
g=np.load(ROOT/"mpcc_tuning/tracks"/GTRACK)
tp=Track(g["cx"],g["cy"],ds=0.1,w_left=g["wl"],w_right=g["wr"])   # narrowed corridor
tp.raceline=g["raceline"]; tp.v_ref=g["vref"]; tp.width_vehicle_adjusted=False; tp.kv_max=0.55
# two solvers: one on the nominal corridor (control), one on the narrowed corridor
m_nom=AcadosMPCC(t, horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="cgN_%d"%os.getpid())
m_nar=AcadosMPCC(tp,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name="cgW_%d"%os.getpid())
def run(kv,narrow):
    th=th0.copy(); th[ik]=np.log(kv); pt=tp if narrow else t; m=m_nar if narrow else m_nom; out=[]
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
print(f"FIXED best-constant, BOUNDARY-AWARE MPCC under 3-site GEOMETRY change (track {GTRACK}), discrete, 6 seeds")
run(0.40,False); run(0.40,True); run(0.55,False); run(0.55,True)
