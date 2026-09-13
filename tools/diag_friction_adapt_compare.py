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
m=AcadosMPCC(t,horizon=st.horizon,dt=0.05,vehicle="dynamic",q_vref=st.q_vref,discrete=True,name=f"da_{os.getpid()}")
def drive(npz,seed,mu):
    pol,_,_=load_policy(str(ROOT/npz),"icra_t2_smooth",seed=seed)
    s0=(0%4)*t.length/4.0; v0=1.0  # seed-0 start (where nominal crashes)
    P=ScuderiaPlant(t,model="std",dt=0.05,mu_scale=mu); P.max_steps=2500
    s5=P.reset(s0=s0,v0=v0); m.reset(); pol.reset(); b=float(s5[4]); off=tr=False; rows=[]
    th=np.asarray(pol.step(features(t,s5)),float)
    for k in range(2500):
        u=m.value(P.state_dyn(),th)["u0"]; s5,r,off,tr=P.step(u)
        rows.append([float(s5[4])-b,float(s5[0]),float(s5[1]),float(s5[3])])
        th=np.asarray(pol.step(features(t,s5)),float)
        if off or tr: break
    return np.array(rows), bool(off)
Dn,on=drive("results/best_policy_icra_t2_smooth_2_progress_mpcc_val_f3.npz",2,0.80)   # nominal net @ mu0.8
Da,oa=drive("results/best_policy_icra_t2_smooth_0_progress_mpcc_val_mu080.npz",0,0.80) # adapted net @ mu0.8
print(f"nominal@mu0.8: {Dn[-1,0]/t.length:.2f} laps off={on}  adapted@mu0.8: {Da[-1,0]/t.length:.2f} laps off={oa}")
fig,(ax,axv)=plt.subplots(1,2,figsize=(17,7)); draw_track(ax,t,edge_lw=1.3)
ax.plot(Dn[:,1],Dn[:,2],color="tab:red",lw=1.6,zorder=4,label=f"nominal net @ mu=0.80: {Dn[-1,0]/t.length:.2f} laps"+(" OFF" if on else ""))
ax.plot(Da[:,1],Da[:,2],color="tab:blue",lw=1.6,zorder=4,label=f"mu=0.80-ADAPTED net @ mu=0.80: {Da[-1,0]/t.length:.2f} laps"+(" OFF" if oa else ""))
if on: ax.scatter([Dn[-1,1]],[Dn[-1,2]],marker='x',s=200,c='red',lw=3,zorder=6,label="nominal crash point")
ax.set_title("Same reduced grip (mu=0.80), seed 0: nominal crashes, adapted stays on"); ax.legend(loc="lower left",fontsize=9)
def fl(D): return D[D[:,0]<=t.length]
Ln,La=fl(Dn),fl(Da)
axv.plot(Ln[:,0],Ln[:,3],color="tab:red",lw=1.5,label="nominal net speed")
axv.plot(La[:,0],La[:,3],color="tab:blue",lw=1.5,label="mu=0.80-adapted speed")
if on: axv.axvline(Dn[-1,0],ls=':',c='red',label="nominal leaves track")
axv.set_xlabel("progress along lap [m]"); axv.set_ylabel("speed [m/s]"); axv.set_title("The adapted net LEARNED to carry less speed (stays within the reduced grip)"); axv.legend(fontsize=9); axv.grid(alpha=.3)
fig.suptitle("Online adaptation to mu=0.80: the tuner learns to drive slower where the nominal net over-speeds and crashes",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
for ext in ("png","pdf"): fig.savefig(str(ROOT/f"results/paper_smooth/friction_adapt_compare.{ext}"),dpi=135)
print("saved friction_adapt_compare")
