import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning import baselines as B
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
ik=WEIGHT_NAMES.index('k_v'); kv0=float(np.exp(B.start('icra_t2_smooth').theta()[ik]))
def load(f): return json.load(open(ROOT/'results/race'/f))
def eqp(js,a): rs=[r for x in js['runs'] if x['arm']==a for r in x['rows']]; return np.mean([r['passes'] for r in rs if r['kind']=='equal'] or [0])
r2=load('race_phase1_reward2.json'); r3=load('race_phase1_reward3.json')
# CORRECT k_v: theta_mean is already LINEAR (loop stores np.exp(theta)); do NOT exp again
print("LTC equal per seed (reward3) -- CORRECT k_v (linear):")
for run in sorted([x for x in r3['runs'] if x['arm']=='ltc'], key=lambda r:r['seed']):
    rs=[r for r in run['rows'] if r['kind']=='equal']
    p=np.mean([r['passes'] for r in rs] or [0]); kv=np.mean([r['theta_mean'][ik] for r in rs if 'theta_mean' in r])
    print(f"  seed {run['seed']}: equal passes {p:.2f}  k_v {kv:.2f} ({'>baseline' if kv>kv0 else 'at/below'})")
fig,ax=plt.subplots(1,2,figsize=(13,5))
x=np.arange(2); w=0.35
for j,arm in enumerate(('ltc','mlp')):
    ax[0].bar(x+(j-.5)*w,[eqp(r2,arm),eqp(r3,arm)],w,label=arm,edgecolor='k',color=('tab:purple' if arm=='ltc' else 'tab:gray'))
ax[0].set_xticks(x); ax[0].set_xticklabels(['REWARD2\n(faster 1.35)','REWARD3\n(faster 1.05\n+hold-pos)']); ax[0].set_ylabel('equal overtakes/ep')
ax[0].set_title('EQUAL overtaking: ltc recovered 0.55->0.82'); ax[0].legend(); ax[0].grid(axis='y',alpha=.3)
seeds=sorted([x['seed'] for x in r3['runs'] if x['arm']=='ltc'])
vals=[]
for s in seeds:
    run=[x for x in r3['runs'] if x['arm']=='ltc' and x['seed']==s][0]
    vals.append(np.mean([r['passes'] for r in run['rows'] if r['kind']=='equal'] or [0]))
ax[1].bar([f'seed {s}' for s in seeds],vals,color='tab:purple',edgecolor='k')
ax[1].axhline(np.mean(vals),ls='--',color='k',label=f'mean {np.mean(vals):.2f}')
ax[1].set_ylabel('ltc equal overtakes/ep'); ax[1].set_title('ltc equal is BIMODAL across seeds\n(seed 0 nails it, seeds 1-2 weak)'); ax[1].legend(); ax[1].grid(axis='y',alpha=.3)
fig.suptitle('Rebalance (faster 1.05 + hold-position): ltc equal recovered on average (0.55->0.82), but bimodal across seeds',fontsize=11)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_reward3.{e}',dpi=140) for e in ('png','pdf')]
print('saved race_reward3.png')
