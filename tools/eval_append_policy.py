import sys, os, csv, numpy as np
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
OUT=ROOT/"results/paper_smooth"; TRK="icra_t2_smooth"; st=B.start(TRK); DT=0.05; STEPS=3000
t=getattr(Track,TRK)()
m=AcadosMPCC(t,horizon=st.horizon,dt=DT,vehicle="dynamic",q_vref=st.q_vref,name=f"evalA_{os.getpid()}")
def drive(policy,s0,v0,steps=STEPS):
    P=ScuderiaPlant(t,model="std",dt=DT); P.max_steps=steps
    s5=P.reset(s0=s0,v0=v0); m.reset(); policy.reset()
    base=float(s5[4]); off=tr=False; rows=[]; lap_steps=[0]; nextlap=1
    th=np.asarray(policy.step(features(t,s5)),float)
    for k in range(steps):
        u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u); d=float(s5[4])-base
        rows.append([k*DT,float(s5[0]),float(s5[1]),float(s5[3]),d]+np.exp(th).tolist())
        while d>=nextlap*t.length: lap_steps.append(k); nextlap+=1
        th=np.asarray(policy.step(features(t,s5)),float)
        if off or tr: break
    rows=np.array(rows); lt=np.diff(lap_steps)*DT if len(lap_steps)>1 else np.array([])
    return rows, rows[-1,4]/t.length, float(rows[:,3].max()), (float(lt.min()) if lt.size else float('nan')), bool(off), int(len(lap_steps)-1)
def clean3(policy):
    ok=0
    for sd in (0,1,2):
        r=drive(policy,(sd%4)*t.length/4.0,1.0+0.1*(sd%3),2500)
        if not r[4]: ok+=1
    return ok
name=sys.argv[1]; npz=sys.argv[2]; seed=int(sys.argv[3])
pol,banked,meta=load_policy(str(ROOT/npz),TRK,seed=seed)
s0=(seed%4)*t.length/4.0; v0=1.0+0.1*(seed%3)
rows,laps,top,fast,off,nlap=drive(pol,s0,v0); cl=clean3(pol)
np.savetxt(str(OUT/f"traj_{name}.csv"), rows, delimiter=",",
           header="t_s,x_m,y_m,v_mps,dist_m,"+",".join(WEIGHT_NAMES), comments="", fmt="%.5f")
row=dict(policy=name, top_speed_mps=round(top,3), laps=round(laps,3), n_full_laps=nlap,
         fastest_lap_s=round(fast,3) if fast==fast else "", clean_of_3=cl, off_in_run=off)
# append to summary.csv
sp=OUT/"summary.csv"; existing=list(csv.DictReader(open(sp))) if sp.exists() else []
existing=[r for r in existing if r["policy"]!=name]+[ {k:str(v) for k,v in row.items()} ]
keys=["policy","top_speed_mps","laps","n_full_laps","fastest_lap_s","clean_of_3","off_in_run"]
with open(sp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in existing]
print(f"{name}: top {top:.2f} laps {laps:.2f} fastest {fast:.2f}s clean {cl}/3 -> appended")
