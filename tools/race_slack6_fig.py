import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
def load(f): return json.load(open(ROOT/'results/race'/f))
def rows(js,a): return [r for x in js['runs'] if x['arm']==a for r in x['rows']]
def off(js,a): return np.mean([r['off'] for r in rows(js,a)])
def eqp(js,a): return np.mean([r['passes'] for r in rows(js,a) if r['kind']=='equal'] or [0])
def pace(js,a): 
    ms=[r['mean_speed'] for r in rows(js,a) if r.get('mean_speed') is not None]; return float(np.mean(ms))
E=load('race_phase1_enrich.json'); S=load('race_phase1_slack6.json')
fig,ax=plt.subplots(1,3,figsize=(15,5))
mets=[('off-track fraction',off,'lower = more stable'),('mean pace m/s',pace,'held ~1.75, > equal opp 1.57'),
      ('equal overtakes/ep',eqp,'the cost: dropped')]
x=np.arange(2); w=0.35
for i,(title,fn,sub) in enumerate(mets):
    for j,arm in enumerate(('ltc','mlp')):
        ax[i].bar(x+(j-.5)*w,[fn(E,arm),fn(S,arm)],w,label=arm,edgecolor='k',
                  color=('tab:purple' if arm=='ltc' else 'tab:gray'))
    ax[i].set_xticks(x); ax[i].set_xticklabels(['ENRICH\n(slack 1)','SLACK6']); ax[i].set_title(title+'\n'+sub)
    ax[i].legend(); ax[i].grid(axis='y',alpha=.3)
ax[1].axhline(1.57,ls='--',c='.5',lw=1)
fig.suptitle('corridor_slack_scale 1 -> 6: off-track HALVED, pace held, but equal-overtakes dropped '
             '(the uniform stiffening costs passing -> motivates the two-layer)',fontsize=11)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_slack6.{e}',dpi=140) for e in ('png','pdf')]
print('off  ltc %.0f%%->%.0f%%  mlp %.0f%%->%.0f%%'%(100*off(E,'ltc'),100*off(S,'ltc'),100*off(E,'mlp'),100*off(S,'mlp')))
print('equal ltc %.2f->%.2f  mlp %.2f->%.2f'%(eqp(E,'ltc'),eqp(S,'ltc'),eqp(E,'mlp'),eqp(S,'mlp')))
print('saved race_slack6.png')
