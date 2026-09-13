import sys, os, csv, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import jax.numpy as jnp, jax
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.ltc import features
from drive_policy import load_policy
OUT=ROOT/"results/paper_smooth"; st=B.start("icra_t2_smooth"); DT=0.05; STEPS=2500
TR=ROOT/"mpcc_tuning/tracks"
def nominal(): return Track.icra_t2_smooth()
def narrowed():
    d=np.load(TR/"icra_t2_smooth_narrowed_corridor.npz")
    t=Track(d["cx"],d["cy"],ds=0.1,w_left=d["wl"],w_right=d["wr"]); t.raceline=d["raceline"]; t.v_ref=d["vref"]
    t.width_vehicle_adjusted=False; t.kv_max=0.55; return t
def const_theta(kv):
    th=np.asarray(st.theta(),float).copy(); th[WEIGHT_NAMES.index("k_v")]=np.log(kv); return th
def drive(m,t,policy=None,th=None,s0=0.0,v0=1.0,mu=1.0):
    P=ScuderiaPlant(t,model="std",dt=DT); P.max_steps=STEPS
    if mu!=1.0:
        P.env._mu_scale=lambda x, _mu=mu: jnp.full((x.shape[0],), _mu)
        P._step_env=jax.jit(P.env.step_env)
    s5=P.reset(s0=s0,v0=v0); m.reset()
    if policy is not None: policy.reset()
    base=float(s5[4]); off=tr=False; top=0.0
    cur=(np.asarray(policy.step(features(t,s5)),float) if policy is not None else th)
    for k in range(STEPS):
        u=m.value(P.state_dyn(),cur)["u0"]; s5,r,off,tr=P.step(u); top=max(top,float(s5[3]))
        cur=(np.asarray(policy.step(features(t,s5)),float) if policy is not None else th)
        if off or tr: break
    return (float(s5[4])-base)/t.length, bool(off), top
def eval_pol(name, spec, cond, m, t, mu=1.0):
    laps=[];offs=[];tops=[]
    for sd in (0,1,2):
        s0=(sd%4)*t.length/4.0; v0=1.0+0.1*(sd%3)
        if "th" in spec: l,o,tp=drive(m,t,th=spec["th"],s0=s0,v0=v0,mu=mu)
        else:
            pol,_,_=load_policy(str(ROOT/spec["npz"]),"icra_t2_smooth",seed=spec["seed"])
            l,o,tp=drive(m,t,policy=pol,s0=s0,v0=v0,mu=mu)
        laps.append(l);offs.append(o);tops.append(tp)
    cl=sum(1 for o in offs if not o)
    print(f"  {name:16s} {cond:16s} clean {cl}/3  laps {[round(x,2) for x in laps]} mean {np.mean(laps):.2f} top {max(tops):.2f}",flush=True)
    return dict(policy=name, condition=cond, clean_of_3=cl, mean_laps=round(float(np.mean(laps)),3),
                laps=";".join(f"{x:.2f}" for x in laps), top_speed=round(max(tops),3))
POLS=[("best_constant",dict(th=const_theta(0.55))),
      ("online_MPCC",  dict(npz="results/best_policy_icra_t2_smooth_2_progress_mpcc_val_f3.npz",seed=2))]
rows=[]
tn=nominal(); mn=AcadosMPCC(tn,horizon=st.horizon,dt=DT,vehicle="dynamic",q_vref=st.q_vref,name=f"rob_nom_{os.getpid()}")
print("NOMINAL"); [rows.append(eval_pol(n,s,"nominal",mn,tn)) for n,s in POLS]
print("LOW FRICTION (mu=0.80)"); [rows.append(eval_pol(n,s,"friction_0.80",mn,tn,mu=0.80)) for n,s in POLS]
tw=narrowed(); mw=AcadosMPCC(tw,horizon=st.horizon,dt=DT,vehicle="dynamic",q_vref=st.q_vref,name=f"rob_narrow_{os.getpid()}")
print("GEOMETRY (boundary moved in 0.15 m @ 2 corners)"); [rows.append(eval_pol(n,s,"geometry_narrowed",mw,tw)) for n,s in POLS]
keys=["policy","condition","clean_of_3","mean_laps","laps","top_speed"]
with open(OUT/"robustness.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print("saved",OUT/"robustness.csv")
