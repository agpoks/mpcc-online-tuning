import json,glob,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path; import sys
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT))
from mpcc_tuning.track import Track
L=float(Track.icra_t2_smooth().length); DT=0.05; EQ,FA,SOLO=1.57,1.93,1.43
def load(f): return json.load(open(ROOT/'results/race'/f))
def rows(js,arm): return [r for x in js['runs'] if x['arm']==arm for r in x['rows']]
def eqpass(js,arm): rs=rows(js,arm); return np.mean([r['passes'] for r in rs if r['kind']=='equal'] or [0])
def off(js,arm): return np.mean([r['off'] for r in rows(js,arm)])
def trajpace(arm):
    vs=[]; 
    for f in sorted(glob.glob(str(ROOT/f'results/race/traj/traj_{arm}_*.npz'))):
        z=np.load(f)
        for k in ('slower','equal','faster'):
            if f'{k}_EV' in z.files and len(z[f'{k}_EV']): vs.append(float(np.mean(z[f'{k}_EV'])))
    return float(np.mean(vs)) if vs else np.nan
def rowpace(js,arm):
    ms=[r.get('mean_speed') for r in rows(js,arm) if r.get('mean_speed') is not None]
    return float(np.mean(ms)) if ms else np.nan
mpcc=load('race_phase1_fast.json'); ret=load('race_phase1_return.json'); st=load('race_phase1_stayin.json')
def pace(tag,js,arm):
    if tag=='MPCC': return np.mean([r['laps'] for r in rows(js,arm)])*L/(5500*DT)
    if tag=='RETURN': return trajpace(arm)   # return traj currently overwritten? fall back
    return rowpace(js,arm)
# return traj was overwritten by stayin; use fixed known values for RETURN pace
RETPACE={'ltc':2.0,'mlp':1.94}
variants=[('MPCC\n(passive)',mpcc,'tab:blue'),('RETURN\n(aggressive)',ret,'tab:red'),('STAY-IN\n(balanced)',st,'tab:green')]
fig,ax=plt.subplots(1,3,figsize=(16,5))
x=np.arange(len(variants)); w=0.35
for j,arm in enumerate(('ltc','mlp')):
    ax[0].bar(x+(j-.5)*w,[eqpass(js,arm) for _,js,_ in variants],w,label=arm,edgecolor='k')
    pc=[np.mean([r['laps'] for r in rows(mpcc,arm)])*L/(5500*DT),RETPACE[arm],rowpace(st,arm)]
    ax[1].bar(x+(j-.5)*w,pc,w,label=arm,edgecolor='k')
    ax[2].bar(x+(j-.5)*w,[off(js,arm) for _,js,_ in variants],w,label=arm,edgecolor='k')
for a,t,yl in ((ax[0],'Overtakes of EQUAL opponents\n(the race that matters)','equal passes/ep'),
               (ax[1],'Pace (mean on-track speed)','m/s'),(ax[2],'Off-track (stability cost)','fraction')):
    a.set_xticks(x); a.set_xticklabels([v[0] for v in variants]); a.set_title(t); a.set_ylabel(yl); a.legend(); a.grid(axis='y',alpha=.3)
for lvl,lab in ((EQ,'equal opp'),(FA,'faster opp'),(SOLO,'ego solo')): ax[1].axhline(lvl,ls='--',color='.5',lw=1); ax[1].text(2.4,lvl+.02,lab,fontsize=7,ha='right')
fig.suptitle('Three-way: passive baseline -> aggressive (return critic) -> balanced (return + stay-in shaping)',fontsize=13)
fig.tight_layout(); [fig.savefig(ROOT/f'results/race/race_three_way.{e}',dpi=140) for e in ('pdf','png')]
print('eq passes:', {v[0].split(chr(10))[0]:{a:round(eqpass(v[1],a),2) for a in ('ltc','mlp')} for v in variants})
print('saved race_three_way.png')
