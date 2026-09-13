import sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"tools"))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.ltc import features
from drive_policy import load_policy
from track_plot import draw_track
st=B.start("icra_t2_smooth"); t=Track.icra_t2_smooth()
m=AcadosMPCC(t,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,name=f"diag_{os.getpid()}")
pol,_,_=load_policy(str(ROOT/"results/best_policy_icra_t2_smooth_2_progress_mpcc_val_f3.npz"),"icra_t2_smooth",seed=2)
def drive(mu,seed=0):
    s0=(seed%4)*t.length/4.0; v0=1.0+0.1*(seed%3)
    P=ScuderiaPlant(t,model="std",dt=0.05,mu_scale=mu); P.max_steps=2500
    s5=P.reset(s0=s0,v0=v0); m.reset(); pol.reset(); b=float(s5[4]); off=tr=False; rows=[]
    th=np.asarray(pol.step(features(t,s5)),float)
    for k in range(2500):
        u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
        rows.append([float(s5[4])-b, float(s5[0]),float(s5[1]),float(s5[3]), int(t.sector(t.wrap(float(s5[4]))))])
        th=np.asarray(pol.step(features(t,s5)),float)
        if off or tr: break
    return np.array(rows), bool(off)
D1,o1=drive(1.0); D8,o8=drive(0.80)
print(f"mu=1.0: {D1[-1,0]/t.length:.2f} laps off={o1}   mu=0.80: {D8[-1,0]/t.length:.2f} laps off={o8}")
fig,(ax,axv)=plt.subplots(1,2,figsize=(17,7))
draw_track(ax,t,edge_lw=1.3)
ax.plot(D1[:,1],D1[:,2],color="tab:green",lw=1.6,zorder=4,label=f"mu=1.0 (nominal): {D1[-1,0]/t.length:.2f} laps")
ax.plot(D8[:,1],D8[:,2],color="tab:red",lw=1.6,zorder=4,label=f"mu=0.80: {D8[-1,0]/t.length:.2f} laps"+(" OFF" if o8 else ""))
if o8: ax.scatter([D8[-1,1]],[D8[-1,2]],marker='x',s=200,c='red',lw=3,zorder=6,label="crash point")
ax.set_title("Nominal (frozen) online-MPCC net: mu=1.0 vs mu=0.80, seed 0"); ax.legend(loc="lower left",fontsize=9)
# speed vs progress within first lap
def firstlap(D): return D[D[:,0]<=t.length]
L1,L8=firstlap(D1),firstlap(D8)
axv.plot(L1[:,0],L1[:,3],color="tab:green",lw=1.5,label="mu=1.0 speed")
axv.plot(L8[:,0],L8[:,3],color="tab:red",lw=1.5,label="mu=0.80 speed")
if o8: axv.axvline(D8[-1,0],ls=':',c='red',label="mu=0.80 leaves track")
axv.set_xlabel("progress along lap [m]"); axv.set_ylabel("speed [m/s]"); axv.set_title("Speed profile: the frozen net carries the SAME plan into less grip"); axv.legend(fontsize=9); axv.grid(alpha=.3)
fig.suptitle("Diagnosis: the frozen nominal net has no friction sense -- it drives its mu=1.0 plan under mu=0.80 and over-speeds",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
for ext in ("png","pdf"): fig.savefig(str(ROOT/f"results/paper_smooth/diag_friction_crash.{ext}"),dpi=135)
print("saved diag_friction_crash")
