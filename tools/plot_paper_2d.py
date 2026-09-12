import sys, csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0,str(ROOT/"tools"))
from track_plot import draw_track
from mpcc_tuning.track import Track
OUT=ROOT/"results/paper_smooth"; t=Track.icra_t2_smooth()
# policy display order + colors + labels (pull fastest lap from summary)
summ={r["policy"]:r for r in csv.DictReader(open(OUT/"summary.csv"))}
SHOW=[("START","tab:gray"),("best_constant","black"),("grid_fitted","tab:green"),
      ("online_RETURN","tab:blue"),("online_MPCC_f3","tab:red")]
def load_traj(name):
    p=OUT/f"traj_{name}.csv"
    if not p.exists(): return None
    d=np.genfromtxt(p,delimiter=",",names=True); return np.column_stack([d["x_m"],d["y_m"],d["v_mps"],d["dist_m"]])
fig,ax=plt.subplots(figsize=(11,9.5)); draw_track(ax,t,edge_lw=1.6)
for name,c in SHOW:
    D=load_traj(name)
    if D is None: continue
    # show one representative lap (first full lap) for a clean line
    one=D[D[:,3]<=t.length]
    s=summ.get(name,{})
    lab=f"{name}: {s.get('fastest_lap_s','?')}s/lap, top {s.get('top_speed_mps','?')} m/s"
    ax.plot(one[:,0],one[:,1],color=c,lw=2.0,zorder=4,label=lab)
ax.set_title("ICRA T2 time-trial: policy racing lines on the real map + smooth boundaries",fontsize=12)
ax.legend(loc="lower left",fontsize=9,framealpha=0.92)
fig.tight_layout()
for ext in ("png","pdf"):
    f=OUT/f"paper_2d_lines.{ext}"; fig.savefig(str(f),dpi=150); print("saved",f)
