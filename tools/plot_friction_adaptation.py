import sys, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); OUT=ROOT/"results/paper_smooth"
WN=['q_c','q_l','q_v','r_d','r_a','r_dv','d_obs','k_v']
def best(fn):
    d=json.load(open(ROOT/"results"/fn)); ep=d['episodes']
    k=max(ep, key=lambda kk: max(x['laps'] for x in ep[kk]))
    return d, ep[k], k, d['traces'][k]
dN,sN,kN,trN=best("online_smooth_mpcc.json")          # mu=1.0
dM,sM,kM,trM=best("online_smooth_mu080_adapt.json")   # mu=0.80
def col(series,w): return np.array([x['theta'][WN.index(w)] for x in series])
def lap(series): return [x['laps'] for x in series]; 
epN=[x['ep'] for x in sN]; epM=[x['ep'] for x in sM]
fig,axs=plt.subplots(2,2,figsize=(15,9))
# k_v over learning
ax=axs[0,0]
ax.plot(epN,col(sN,'k_v'),'-o',color="tab:green",ms=4,label="mu=1.0 (nominal)")
ax.plot(epM,col(sM,'k_v'),'-o',color="tab:red",ms=4,label="mu=0.80 (adapting)")
ax.set_title("Grip claim k_v over learning (best seed) -- both explore, no single-weight signature"); ax.set_xlabel("episode"); ax.set_ylabel("k_v (emitted)"); ax.legend(); ax.grid(alpha=.3)
# r_a over learning
ax=axs[0,1]
ax.plot(epN,col(sN,'r_a'),'-o',color="tab:green",ms=4,label="mu=1.0")
ax.plot(epM,col(sM,'r_a'),'-o',color="tab:red",ms=4,label="mu=0.80")
ax.set_title("Accel damping r_a over learning"); ax.set_xlabel("episode"); ax.set_ylabel("r_a"); ax.legend(); ax.grid(alpha=.3)
# laps over learning (recovery)
ax=axs[1,0]
ax.plot(epN,lap(sN),'-o',color="tab:green",ms=4,label="mu=1.0 laps")
ax.plot(epM,lap(sM),'-o',color="tab:red",ms=4,label="mu=0.80 laps")
ax.plot(epM,[x['best'] for x in sM],'--',color="0.4",label="mu=0.80 banked best")
ax.set_title("Laps over learning (recovery from early over-speed crashes)"); ax.set_xlabel("episode"); ax.set_ylabel("laps"); ax.legend(); ax.grid(alpha=.3)
# spatial: emitted k_v around the lap (final ep), mu=0.80 vs mu=1.0, by sector
ax=axs[1,1]
for tr,c,lab in [(trN,"tab:green","mu=1.0"),(trM,"tab:red","mu=0.80")]:
    if tr and len(tr):
        last=np.array(tr[-1])
        if last.ndim==2 and last.shape[1]>=12:
            ax.plot(np.arange(len(last)),last[:,4+WN.index('k_v')],color=c,label=lab+" k_v(s)")
            secs=last[:,3]
ax.set_title("Emitted k_v around the lap (final episode, best seed)"); ax.set_xlabel("track sample"); ax.set_ylabel("k_v"); ax.legend(); ax.grid(alpha=.3)
fig.suptitle("Online MPCC-critic ADAPTS to mu=0.80: all seeds recover to CLEAN laps 2.15/2.29/2.83 (frozen nominal was 2/3, crashed 1.48)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
for ext in ("png","pdf"): fig.savefig(str(OUT/f"friction_adaptation.{ext}"),dpi=140); 
print("saved friction_adaptation")
# print the numeric adaptation summary
print("mu=1.0 final k_v %.3f | mu=0.80 final k_v %.3f" % (col(sN,'k_v')[-1], col(sM,'k_v')[-1]))
