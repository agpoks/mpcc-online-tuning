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

# 4) WEIGHTS BY OPPONENT TYPE (learned arms) -- how the 8 weights the tuner emits
#    change with the opponent it faces. Needs theta_mean in the rows (new runs).
sys.path.insert(0,str(ROOT))
try:
    from mpcc_tuning import baselines as B
    from mpcc_tuning.mpcc import WEIGHT_NAMES
    start=np.exp(np.asarray(B.start("icra_t2_smooth").theta(),float))
    have_theta=any("theta_mean" in r for x in d["runs"] for r in x["rows"])
    if have_theta:
        with open(OUT/f"race_weights_by_kind{TAG}.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["arm","kind"]+list(WEIGHT_NAMES)+[n+"_relSTART" for n in WEIGHT_NAMES])
            for arm in ["ltc","mlp"]:
                for k in KINDS:
                    R=[r for x in d["runs"] if x["arm"]==arm for r in x["rows"]
                       if r["kind"]==k and "theta_mean" in r]
                    if not R: continue
                    wm=np.mean([r["theta_mean"] for r in R],axis=0)
                    w.writerow([arm,k]+[round(float(v),4) for v in wm]
                               +[round(float(v),3) for v in wm/start])
        for arm in ["ltc","mlp"]:
            M=[]
            for k in KINDS:
                R=[r for x in d["runs"] if x["arm"]==arm for r in x["rows"]
                   if r["kind"]==k and "theta_mean" in r]
                M.append(np.mean([r["theta_mean"] for r in R],axis=0)/start if R else np.ones(8))
            M=np.array(M)                      # 4 kinds x N weights, relative to START
            fig,ax=plt.subplots(figsize=(8,3.8))
            im=ax.imshow(np.log2(M).T,cmap="RdBu_r",vmin=-1.2,vmax=1.2,aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels(KINDS)
            ax.set_yticks(range(len(WEIGHT_NAMES))); ax.set_yticklabels(WEIGHT_NAMES)
            for i in range(len(WEIGHT_NAMES)):
                for j in range(4): ax.text(j,i,f"{M[j,i]:.2f}",ha="center",va="center",fontsize=7)
            ax.set_title(f"{arm}: emitted weights by opponent type (x START)")
            fig.colorbar(im,ax=ax,fraction=0.03,label="log2(weight / START)")
            fig.tight_layout()
            for e in ("pdf","png"): fig.savefig(OUT/f"race_weights_by_kind_{arm}{TAG}.{e}",dpi=140)
        print("  + race_weights_by_kind CSV + heatmaps (ltc,mlp)")
    else:
        print("  (no theta_mean in rows -- re-run race_mode.py to get weights-by-kind)")
except Exception as e:
    print("  weights-by-kind skipped:",e)

# 5) WEIGHT LEARNING CURVE (over episodes) + WEIGHTS BY SECTOR (from sec_theta)
try:
    from mpcc_tuning import baselines as B
    from mpcc_tuning.mpcc import WEIGHT_NAMES
    start=np.exp(np.asarray(B.start("icra_t2_smooth").theta(),float))
    if any("theta_mean" in r for x in d["runs"] for r in x["rows"]):
        for arm in ["ltc","mlp"]:
            runs=[x for x in d["runs"] if x["arm"]==arm]
            if not runs: continue
            neps=max(len(x["rows"]) for x in runs); curve=np.full((neps,8),np.nan)
            for e in range(neps):
                vals=[x["rows"][e]["theta_mean"] for x in runs
                      if e<len(x["rows"]) and "theta_mean" in x["rows"][e]]
                if vals: curve[e]=np.mean(vals,axis=0)/start
            fig,ax=plt.subplots(figsize=(8,4.2))
            for i,n in enumerate(WEIGHT_NAMES): ax.plot(range(neps),curve[:,i],marker="o",ms=3,label=n)
            ax.axhline(1.0,color="k",lw=0.7,ls="--"); ax.set_yscale("log")
            ax.set_xlabel("episode"); ax.set_ylabel("emitted weight / START")
            ax.set_title(f"{arm}: weight learning curve over episodes (how weights change over time)")
            ax.legend(fontsize=7,ncol=4); fig.tight_layout()
            for e2 in ("pdf","png"): fig.savefig(OUT/f"race_weight_curve_{arm}{TAG}.{e2}",dpi=140)
    if any("sec_theta" in r for x in d["runs"] for r in x["rows"]):
        with open(OUT/f"race_weights_by_sector{TAG}.csv","w",newline="") as f:
            wtr=csv.writer(f); wtr.writerow(["arm","sector"]+list(WEIGHT_NAMES)+[n+"_relSTART" for n in WEIGHT_NAMES])
            for arm in ["ltc","mlp"]:
                runs=[x for x in d["runs"] if x["arm"]==arm]
                if not runs: continue
                M=np.zeros((4,8)); cnt=np.zeros(4)
                for x in runs:
                    for r in x["rows"][len(x["rows"])//2:]:      # last-half episodes = converged
                        st_=np.array(r.get("sec_theta",np.zeros((4,8))))
                        for s in range(4):
                            if st_[s].any(): M[s]+=st_[s]; cnt[s]+=1
                seen=[s for s in range(4) if cnt[s]>0]     # only sectors the car visits
                for s in range(4):
                    if cnt[s]: M[s]/=cnt[s]
                    else: continue                          # no samples -> not an "all-zero" row
                    wtr.writerow([arm,s]+[round(float(v),4) for v in M[s]]
                                 +[round(float(v),3) for v in (M[s]/start)])
                rel=np.full((4,8),np.nan)                   # empty sectors stay blank (NaN)
                for s in seen: rel[s]=M[s]/start
                fig,ax=plt.subplots(figsize=(8,3.8))
                im=ax.imshow(np.log2(rel).T,cmap="RdBu_r",vmin=-1.2,vmax=1.2,aspect="auto")
                im.cmap.set_bad("0.85")                      # NaN sectors shown grey (no data)
                ax.set_xticks(range(4)); ax.set_xticklabels([f"S{i}" for i in range(4)])
                ax.set_yticks(range(len(WEIGHT_NAMES))); ax.set_yticklabels(WEIGHT_NAMES)
                for i in range(len(WEIGHT_NAMES)):
                    for j in range(4):
                        ax.text(j,i,("--" if np.isnan(rel[j,i]) else f"{rel[j,i]:.2f}"),ha="center",va="center",fontsize=7)
                ax.set_title(f"{arm}: emitted weights BY SECTOR (x START)")
                fig.colorbar(im,ax=ax,fraction=0.03,label="log2(weight / START)")
                fig.tight_layout()
                for e2 in ("pdf","png"): fig.savefig(OUT/f"race_weights_by_sector_{arm}{TAG}.{e2}",dpi=140)
        print("  + weight learning curves + weights-by-sector (CSV+heatmaps)")
except Exception as e:
    print("  learning-curve/per-sector skipped:",e)
print("wrote race_summary/by_kind/sectors CSVs + race_sector_suitability + race_behaviour figures, TAG="+repr(TAG))
