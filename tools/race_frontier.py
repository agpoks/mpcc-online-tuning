import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
def load(f): return json.load(open(ROOT/'results/race'/f))
def off(js,arm): rs=[r for x in js['runs'] if x['arm']==arm for r in x['rows']]; return np.mean([r['off'] for r in rs])
def eqp(js,arm): rs=[r for x in js['runs'] if x['arm']==arm for r in x['rows']]; return np.mean([r['passes'] for r in rs if r['kind']=='equal'] or [0])
PACE={('MPCC','ltc'):1.45,('MPCC','mlp'):1.49,('RETURN','ltc'):2.0,('RETURN','mlp'):1.94}
def pace(tag,js,arm):
    if (tag,arm) in PACE: return PACE[(tag,arm)]
    ms=[r['mean_speed'] for x in js['runs'] if x['arm']==arm for r in x['rows'] if r.get('mean_speed') is not None]
    return float(np.mean(ms))
V=[('MPCC','race_phase1_fast.json','tab:blue'),('RETURN','race_phase1_return.json','tab:red'),
   ('STAY-IN heavy','race_phase1_stayin.json','tab:orange'),('STAY-IN light','race_phase1_stayin_light.json','tab:green')]
fig,ax=plt.subplots(1,2,figsize=(15,6))
for tag,f,c in V:
    js=load(f); short=tag.split()[-1] if ' ' in tag else tag
    for arm,mk in (('ltc','o'),('mlp','s')):
        x=off(js,arm)*100; y=pace(tag,js,arm)
        ax[0].scatter(x,y,s=180,c=c,marker=mk,edgecolor='k',zorder=3)
        ax[0].annotate(short+"\n"+arm,(x,y),(x+1.5,y),fontsize=7,va='center')
ax[0].axhline(1.57,ls='--',c='.5',lw=1); ax[0].text(62,1.59,'equal opp 1.57',fontsize=8,ha='right')
ax[0].axhline(1.93,ls='--',c='.5',lw=1); ax[0].text(62,1.95,'faster opp 1.93',fontsize=8,ha='right')
ax[0].set_xlabel('off-track % of episodes  (more stable <-)'); ax[0].set_ylabel('mean on-track speed m/s  (-> faster)')
ax[0].set_title('The pace-vs-stability FRONTIER\n(want top-LEFT: fast AND clean -- nothing reaches it)')
ax[0].scatter([],[],c='k',marker='o',label='ltc'); ax[0].scatter([],[],c='k',marker='s',label='mlp'); ax[0].legend(loc='lower right')
ax[0].annotate('empty corner:\nfast + clean\n= enrichment target',(8,2.0),(20,1.97),fontsize=9,color='green',
               arrowprops=dict(arrowstyle='->',color='green'))
ax[0].grid(alpha=.3)
x=np.arange(len(V)); w=0.35
for j,arm in enumerate(('ltc','mlp')):
    ax[1].bar(x+(j-.5)*w,[eqp(load(f),arm) for _,f,_ in V],w,label=arm,edgecolor='k')
ax[1].set_xticks(x); ax[1].set_xticklabels([v[0] for v in V],rotation=15); ax[1].set_ylabel('equal overtakes/ep')
ax[1].set_title('Overtakes of EQUAL opponents'); ax[1].legend(); ax[1].grid(axis='y',alpha=.3)
fig.suptitle('Stay-in penalty is a DIAL along a frontier, not a fix: heavy=cleaner-slower, light=fast-wide',fontsize=13)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_frontier.{e}',dpi=140) for e in ('pdf','png')]
print('OK saved race_frontier.png')
