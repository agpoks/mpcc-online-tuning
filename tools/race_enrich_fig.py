import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
def load(f): return json.load(open(ROOT/'results/race'/f))
def rows(js,arm): return [r for x in js['runs'] if x['arm']==arm for r in x['rows']]
def off(js,arm): return np.mean([r['off'] for r in rows(js,arm)])
def contact(js,arm): return np.mean([r['contact'] for r in rows(js,arm)])
def eqp(js,arm): return np.mean([r['passes'] for r in rows(js,arm) if r['kind']=='equal'] or [0])
PACE={('MPCC','ltc'):1.45,('MPCC','mlp'):1.49,('RETURN','ltc'):2.0,('RETURN','mlp'):1.94}
def pace(tag,js,arm):
    if (tag,arm) in PACE: return PACE[(tag,arm)]
    ms=[r['mean_speed'] for r in rows(js,arm) if r.get('mean_speed') is not None]; return float(np.mean(ms))
V=[('MPCC','race_phase1_fast.json','tab:blue'),('RETURN','race_phase1_return.json','tab:red'),
   ('heavy','race_phase1_stayin.json','tab:orange'),('light','race_phase1_stayin_light.json','tab:green'),
   ('ENRICH','race_phase1_enrich.json','tab:purple')]
fig,ax=plt.subplots(1,2,figsize=(15,6))
for tag,f,c in V:
    js=load(f)
    for arm,mk in (('ltc','o'),('mlp','s')):
        x=off(js,arm)*100; y=pace(tag,js,arm)
        ax[0].scatter(x,y,s=190,c=c,marker=mk,edgecolor='k',zorder=3)
        ax[0].annotate(tag+" "+arm,(x,y),(x+1.2,y),fontsize=7,va='center')
ax[0].axhline(1.57,ls='--',c='.5',lw=1); ax[0].text(60,1.59,'equal opp',fontsize=8,ha='right')
ax[0].axhline(1.93,ls='--',c='.5',lw=1); ax[0].text(60,1.95,'faster opp',fontsize=8,ha='right')
ax[0].set_xlabel('off-track %  (more stable <-)'); ax[0].set_ylabel('mean speed m/s  (-> faster)')
ax[0].set_title('Frontier: ENRICH (purple) still ~48-57% off -- not the fast+clean corner')
ax[0].scatter([],[],c='k',marker='o',label='ltc'); ax[0].scatter([],[],c='k',marker='s',label='mlp'); ax[0].legend(loc='lower right'); ax[0].grid(alpha=.3)
# memory test: ltc vs mlp within ENRICH
js=load('race_phase1_enrich.json')
mets=[('equal\novertakes',eqp),('off-track',off),('contact',contact)]
x=np.arange(len(mets)); w=0.35
for j,(arm,col) in enumerate((('ltc','tab:purple'),('mlp','tab:gray'))):
    ax[1].bar(x+(j-.5)*w,[m[1](js,arm) for m in mets],w,label=arm,color=col,edgecolor='k')
ax[1].set_xticks(x); ax[1].set_xticklabels([m[0] for m in mets]); ax[1].legend()
ax[1].set_title('Memory test (ENRICH): ltc BEATS mlp\nmore equal passes, less contact, less off-track'); ax[1].grid(axis='y',alpha=.3)
fig.suptitle('Observation enrichment: memory (ltc) now wins the race -- but off-track (stability) is still unsolved',fontsize=12)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_enrich.{e}',dpi=140) for e in ('pdf','png')]
print('saved race_enrich.png')
