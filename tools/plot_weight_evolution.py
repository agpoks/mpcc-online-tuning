import sys, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); OUT=ROOT/"results/paper_smooth"
WN=['q_c','q_l','q_v','r_d','r_a','r_dv','d_obs','k_v']; KEYW=['q_c','q_v','r_a','k_v']
COL={'q_c':'tab:blue','q_v':'tab:orange','r_a':'tab:green','k_v':'tab:red'}
METH=[("MPCC critic","online_smooth_mpcc.json"),("RETURN critic","online_smooth_return.json")]
def best_key(ep):
    bk=None;bb=None
    for k,s in ep.items():
        m=max(x['laps'] for x in s)
        if bb is None or m>bb: bb=m;bk=k
    return bk
fig,axs=plt.subplots(2,2,figsize=(15,9))
for col,(name,fn) in enumerate(METH):
    p=ROOT/"results"/fn
    if not p.exists(): continue
    d=json.load(open(p)); ep=d['episodes']; k=best_key(ep); s=ep[k]; sd=k.split('|')[1]
    epi=[x['ep'] for x in s]; laps=[x['laps'] for x in s]; bestv=[x['best'] for x in s]
    TH=np.array([x['theta'] for x in s])
    aw=axs[0][col]
    for w in KEYW: aw.plot(epi,TH[:,WN.index(w)],marker='o',ms=4,color=COL[w],label=w)
    aw.set_title(f"{name}: emitted weights over learning (seed {sd})"); aw.set_xlabel("episode"); aw.set_ylabel("weight value"); aw.legend(fontsize=9); aw.grid(alpha=.3)
    al=axs[1][col]
    al.plot(epi,laps,'-o',color='tab:purple',ms=4,label="episode laps")
    al.plot(epi,bestv,'--',color='0.4',label="banked best")
    al.set_title(f"{name}: laps over learning (seed {sd})"); al.set_xlabel("episode"); al.set_ylabel("laps"); al.legend(fontsize=9); al.grid(alpha=.3)
fig.suptitle("Online weight learning on icra_t2_smooth: how the cost weights change over episodes",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(str(OUT/"weight_evolution.png"),dpi=140); fig.savefig(str(OUT/"weight_evolution.pdf")); print("saved weight_evolution",[len(axs[0][0].lines),len(axs[1][0].lines)])
# spatial
fig2,axs2=plt.subplots(1,2,figsize=(15,5))
for col,(name,fn) in enumerate(METH):
    p=ROOT/"results"/fn
    if not p.exists(): continue
    d=json.load(open(p)); ep=d['episodes']; k=best_key(ep); tr=d['traces'][k]
    if not tr: continue
    last=np.array(tr[-1]); 
    if last.ndim!=2 or last.shape[1]<12: continue
    ax=axs2[col]
    for w in KEYW: ax.plot(np.arange(len(last)),last[:,4+WN.index(w)],color=COL[w],label=w)
    ax.set_title(f"{name}: weights along the lap (final ep, seed {k.split('|')[1]})"); ax.set_xlabel("track sample"); ax.set_ylabel("weight"); ax.legend(fontsize=9); ax.grid(alpha=.3)
fig2.suptitle("Situation-dependence: emitted weights around the lap",fontsize=12); fig2.tight_layout(rect=[0,0,1,0.95])
fig2.savefig(str(OUT/"weight_spatial.png"),dpi=140); fig2.savefig(str(OUT/"weight_spatial.pdf")); print("saved weight_spatial")
