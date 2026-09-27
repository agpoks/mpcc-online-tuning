import json,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
def load(f): return json.load(open(ROOT/'results/race'/f))
def rows(js,a): return [r for x in js['runs'] if x['arm']==a for r in x['rows']]
def off(js,a): return np.mean([r['off'] for r in rows(js,a)])
def bykind(js,a,k): return np.mean([r['passes'] for r in rows(js,a) if r['kind']==k] or [0])
V=[('dbound+match','race_phase1_dbound.json'),('reward2+wall60','race_phase1_reward2.json')]
fig,ax=plt.subplots(1,3,figsize=(16,5))
kinds=['slower','equal','faster']; x=np.arange(3); w=0.35
# passes by class, mlp
for j,(tag,f) in enumerate(V):
    ax[0].bar(x+(j-.5)*w,[bykind(load(f),'mlp',k) for k in kinds],w,label=tag,edgecolor='k')
ax[0].set_xticks(x); ax[0].set_xticklabels(kinds); ax[0].set_title('mlp: overtakes by class\n(faster jumps 0.40->0.80)'); ax[0].set_ylabel('passes/ep'); ax[0].legend(fontsize=8); ax[0].grid(axis='y',alpha=.3)
for j,(tag,f) in enumerate(V):
    ax[1].bar(x+(j-.5)*w,[bykind(load(f),'ltc',k) for k in kinds],w,label=tag,edgecolor='k')
ax[1].set_xticks(x); ax[1].set_xticklabels(kinds); ax[1].set_title('ltc: overtakes by class\n(faster 0.20->0.60; equal dips)'); ax[1].legend(fontsize=8); ax[1].grid(axis='y',alpha=.3)
# off-track
names=['dbound\nltc','dbound\nmlp','reward2\nltc','reward2\nmlp']
offs=[off(load(V[0][1]),'ltc'),off(load(V[0][1]),'mlp'),off(load(V[1][1]),'ltc'),off(load(V[1][1]),'mlp')]
ax[2].bar(range(4),offs,color=['tab:blue','tab:cyan','tab:green','tab:olive'],edgecolor='k')
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(names,fontsize=8); ax[2].set_title('off-track (wall 30->60)\nmlp 57%->29%'); ax[2].set_ylabel('fraction'); ax[2].grid(axis='y',alpha=.3)
fig.suptitle('Continuous-risk reward + commitment bridge + wall-60: faster-overtaking UP, off-track DOWN, contact ~0',fontsize=11)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_reward2.{e}',dpi=140) for e in ('png','pdf')]
print('saved race_reward2.png')
