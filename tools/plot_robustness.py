import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=Path("/home/poxx/github/mpcc-online-tuning/results/paper_smooth")
rows=list(csv.DictReader(open(OUT/"robustness.csv")))
conds=["nominal","friction_0.80","geometry_narrowed"]; pols=["best_constant","online_MPCC"]
def get(p,c,k): 
    for r in rows:
        if r["policy"]==p and r["condition"]==c: return float(r[k])
    return 0
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5.2)); x=np.arange(len(conds)); w=0.36
cols={"best_constant":"black","online_MPCC":"tab:red"}
for i,p in enumerate(pols):
    a1.bar(x+(i-0.5)*w,[get(p,c,"clean_of_3") for c in conds],w,color=cols[p],label=p)
    a2.bar(x+(i-0.5)*w,[get(p,c,"mean_laps") for c in conds],w,color=cols[p],label=p)
a1.set_xticks(x); a1.set_xticklabels(conds,rotation=15); a1.set_ylabel("clean starts / 3"); a1.set_ylim(0,3.3); a1.set_title("Robustness: clean starts (higher=safer)"); a1.legend(); a1.grid(axis='y',alpha=.3)
a1.axhline(3,ls=':',c='0.6')
a2.set_xticks(x); a2.set_xticklabels(conds,rotation=15); a2.set_ylabel("mean laps"); a2.set_title("Robustness: mean laps"); a2.legend(); a2.grid(axis='y',alpha=.3)
fig.suptitle("Use-case: fixed best-constant vs online-MPCC under a friction drop (mu=0.80)\n"
             "and a geometry change (boundary moved 15-20 cm into 2 corners)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
for ext in ("png","pdf"): fig.savefig(str(OUT/f"robustness_bars.{ext}"),dpi=140)
print("saved robustness_bars")
