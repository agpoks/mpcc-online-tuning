import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
def load(f): return json.load(open(ROOT/'results/race'/f))
def rows(js,a): return [r for x in js['runs'] if x['arm']==a for r in x['rows']]
def off(js,a): return np.mean([r['off'] for r in rows(js,a)])
def con(js,a): return np.mean([r['contact'] for r in rows(js,a)])
def eqp(js,a): return np.mean([r['passes'] for r in rows(js,a) if r['kind']=='equal'] or [0])
def pace(js,a):
    ms=[r['mean_speed'] for r in rows(js,a) if r.get('mean_speed') is not None]; return float(np.mean(ms)) if ms else np.nan
V=[('ENRICH\n(soft)','race_phase1_enrich.json'),('SLACK6\n(stiff)','race_phase1_slack6.json'),('TWO-LAYER','race_phase1_twolayer.json')]
fig,ax=plt.subplots(1,4,figsize=(17,4.6))
mets=[('off-track',off,'lower=stable'),('contact (crash)',con,'lower=safe'),('equal overtakes',eqp,'higher=races'),('pace m/s',pace,'')]
x=np.arange(len(V)); w=0.35
for i,(t,fn,sub) in enumerate(mets):
    for j,(arm,c) in enumerate((('ltc','tab:purple'),('mlp','tab:gray'))):
        ax[i].bar(x+(j-.5)*w,[fn(load(f),arm) for _,f in V],w,label=arm,color=c,edgecolor='k')
    ax[i].set_xticks(x); ax[i].set_xticklabels([v[0] for v in V],fontsize=8); ax[i].set_title(t+('\n'+sub if sub else '')); ax[i].legend(fontsize=8); ax[i].grid(axis='y',alpha=.3)
ax[3].axhline(1.57,ls='--',c='.5',lw=1); ax[3].text(2.4,1.59,'equal opp',fontsize=7,ha='right')
fig.suptitle('The corridor trilemma: enforcing the corridor (slack6 / two-layer) cuts off-track but CRATERS equal-overtaking '
             '-- much of it was off-track-enabled. two-layer did not beat slack6.',fontsize=10)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_threeway_final.{e}',dpi=140) for e in ('png','pdf')]
print('done')
