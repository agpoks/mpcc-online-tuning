import sys, json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning")
JS=ROOT/"results/race"/(sys.argv[1] if len(sys.argv)>1 else "race_phase1_s4.json")
OUT=ROOT/"results/race"; TAG=JS.stem.replace("race_phase1","")
d=json.load(open(JS)); ARMS=["const","fixed","ltc","mlp"]; KINDS=["static","slower","equal","faster"]
def rows(arm): return [r for x in d["runs"] if x["arm"]==arm for r in x["rows"]]
# 1) per-arm summary with composite = passes + clean - 2*contact (safety-weighted)
with open(OUT/f"race_summary{TAG}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["arm","laps","passes","clean_frac","contact_frac","composite"])
    for arm in ARMS:
        R=rows(arm)
        if not R: continue
        laps=np.mean([x["laps"] for x in R]); ps=np.mean([x["passes"] for x in R])
        cl=np.mean([x["clean"] for x in R]); ct=np.mean([x["contact"] for x in R])
        comp=ps+cl-2*ct
        w.writerow([arm,round(laps,3),round(ps,3),round(cl,3),round(ct,3),round(comp,3)])
# 2) per-arm x kind
with open(OUT/f"race_by_kind{TAG}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["arm","kind","passes","contact_frac","n"])
    for arm in ARMS:
        R=rows(arm)
        for k in KINDS:
            rk=[x for x in R if x["kind"]==k]
            if rk: w.writerow([arm,k,round(np.mean([y["passes"] for y in rk]),3),
                               round(np.mean([y["contact"] for y in rk]),3),len(rk)])
# 3) per-sector suitability (all arms) + learned-only (ltc+mlp)
def sectors(arms):
    att=np.zeros(4); pas=np.zeros(4); ct=np.zeros(4)
    for x in d["runs"]:
        if x["arm"] not in arms: continue
        for r in x["rows"]:
            att+=np.array(r["sec_attempt"]); pas+=np.array(r["sec_pass"]); ct+=np.array(r["sec_contact"])
    return att,pas,ct
with open(OUT/f"race_sectors{TAG}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","sector","attempts","passes","contacts","contact_per_pass"])
    for grp,arms in [("all",ARMS),("learned",["ltc","mlp"]),("rule",["fixed"])]:
        att,pas,ct=sectors(arms)
        for s in range(4):
            w.writerow([grp,s,int(att[s]),int(pas[s]),int(ct[s]),
                        round(ct[s]/max(pas[s],1),3)])
# FIGURE: per-sector passes vs contacts (all arms) = overtake suitability
att,pas,ct=sectors(ARMS)
fig,ax=plt.subplots(figsize=(7,4.2)); x=np.arange(4); wd=0.38
ax.bar(x-wd/2,pas,wd,label="passes",color="tab:green")
ax.bar(x+wd/2,ct,wd,label="contacts",color="tab:red")
for s in range(4):
    ax.annotate(f"{int(ct[s])}/{int(pas[s])}",(s,max(pas[s],ct[s])+0.5),ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["S0","S1","S2","S3"]); ax.set_xlabel("sector")
ax.set_ylabel("count (all arms, all seeds)")
ax.set_title("Per-sector overtake suitability: passes vs contacts (S3 = crash-prone)")
ax.legend(); fig.tight_layout()
for e in ("pdf","png"): fig.savefig(OUT/f"race_sector_suitability{TAG}.{e}",dpi=140)
# FIGURE: per-kind passes + contact by arm = learned behaviour discrimination
fig2,ax2=plt.subplots(figsize=(8,4.2)); xk=np.arange(4); w2=0.2
for i,arm in enumerate(ARMS):
    R=rows(arm); pv=[np.mean([y["passes"] for y in R if y["kind"]==k] or [0]) for k in KINDS]
    ax2.bar(xk+(i-1.5)*w2,pv,w2,label=arm)
ax2.set_xticks(xk); ax2.set_xticklabels(KINDS); ax2.set_ylabel("mean passes / episode")
ax2.set_title("Behaviour by opponent pace: learned arms decline equal/faster (rule attempts them)")
ax2.legend(fontsize=8); fig2.tight_layout()
for e in ("pdf","png"): fig2.savefig(OUT/f"race_behaviour{TAG}.{e}",dpi=140)
print("wrote race_summary/by_kind/sectors CSVs + race_sector_suitability + race_behaviour figures, TAG="+repr(TAG))
