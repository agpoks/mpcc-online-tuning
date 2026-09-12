import sys, os, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.ltc import features
from drive_policy import load_policy
OUT=ROOT/"results/paper_smooth"; OUT.mkdir(parents=True, exist_ok=True)
TRK="icra_t2_smooth"; st=B.start(TRK); DT=0.05; STEPS=3000
t=Track.getattr=None
t=getattr(Track,TRK)()
m=AcadosMPCC(t,horizon=st.horizon,dt=DT,vehicle="dynamic",q_vref=st.q_vref,name=f"eval_{os.getpid()}")

def const_theta(kv):
    th=np.asarray(st.theta(),float).copy(); th[WEIGHT_NAMES.index("k_v")]=np.log(kv); return th

def drive(policy=None, th_const=None, s0=0.0, v0=1.0, steps=STEPS):
    P=ScuderiaPlant(t,model="std",dt=DT); P.max_steps=steps
    s5=P.reset(s0=s0,v0=v0); m.reset()
    if policy is not None: policy.reset()
    base=float(s5[4]); off=tr=False; rows=[]; lap_steps=[0]; nextlap=1
    def emit():
        return np.asarray(policy.step(features(t,s5)),float) if policy is not None else th_const
    th=emit()
    for k in range(steps):
        u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
        d=(float(s5[4])-base)
        rows.append([k*DT,float(s5[0]),float(s5[1]),float(s5[3]),d]+np.exp(th).tolist())
        while d>=nextlap*t.length:            # lap completed
            lap_steps.append(k); nextlap+=1
        th=emit()
        if off or tr: break
    rows=np.array(rows)
    laps=(rows[-1,4])/t.length
    top=float(rows[:,3].max())
    lap_times=np.diff(lap_steps)*DT if len(lap_steps)>1 else np.array([])
    fastest=float(lap_times.min()) if lap_times.size else float('nan')
    return dict(rows=rows, laps=float(laps), top_speed=top, fastest_lap=fastest,
                off=bool(off), n_laps=int(len(lap_steps)-1), lap_times=lap_times.tolist())

def clean3(policy=None, th_const=None):
    ok=0
    for seed in (0,1,2):
        s0=(seed%4)*t.length/4.0; v0=1.0+0.1*(seed%3)
        r=drive(policy=policy,th_const=th_const,s0=s0,v0=v0,steps=2500)
        if not r["off"]: ok+=1
    return ok

# policy registry (only the ones ready now)
POLS=[]
POLS.append(("START",            dict(th=const_theta(0.45))))
POLS.append(("best_constant",    dict(th=const_theta(0.55))))
POLS.append(("online_MPCC_f2",   dict(npz="results/best_policy_icra_t2_smooth_2_progress_mpcc_val.npz", seed=2)))
POLS.append(("online_MPCC_f3",   dict(npz="results/best_policy_icra_t2_smooth_2_progress_mpcc_val_f3.npz", seed=2)))

summary=[]
for name,spec in POLS:
    if "th" in spec:
        res=drive(th_const=spec["th"]); cl=clean3(th_const=spec["th"])
    else:
        pol,banked,meta=load_policy(str(ROOT/spec["npz"]),TRK,seed=spec["seed"])
        sd=spec["seed"]; s0=(sd%4)*t.length/4.0; v0=1.0+0.1*(sd%3)   # network's own banked start
        res=drive(policy=pol, s0=s0, v0=v0); cl=clean3(policy=pol)
    # save trajectory CSV
    hdr="t_s,x_m,y_m,v_mps,dist_m,"+",".join(WEIGHT_NAMES)
    np.savetxt(str(OUT/f"traj_{name}.csv"), res["rows"], delimiter=",", header=hdr, comments="", fmt="%.5f")
    summary.append(dict(policy=name, top_speed_mps=round(res["top_speed"],3), laps=round(res["laps"],3),
                        n_full_laps=res["n_laps"], fastest_lap_s=round(res["fastest_lap"],3) if res["fastest_lap"]==res["fastest_lap"] else "",
                        clean_of_3=cl, off_in_run=res["off"]))
    print(f"{name:18s} top {res['top_speed']:.2f} m/s  laps {res['laps']:.2f}  fastest {res['fastest_lap']:.2f}s  clean {cl}/3", flush=True)
# summary CSV
import csv
keys=["policy","top_speed_mps","laps","n_full_laps","fastest_lap_s","clean_of_3","off_in_run"]
with open(OUT/"summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in summary]
print("saved", OUT/"summary.csv")
