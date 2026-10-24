#!/usr/bin/env python3
"""Real-world-deployment figure for the RA-L paper (DATA-ONLY, reads committed deploy_labtest data).
Single-column, two panels, in the paper house style (Times/Nimbus + STIX, viridis speed, dashed
raceline, open-circle start, 1 m scale bar) so it matches Fig 2/3/6:

  (a) the DEPLOYED seed-2 net driving the SLAM-mapped figure-eight lab map (Track 2) FROZEN -- the
      occupancy grid in grey, the raceline-referenced corridor edges, the driven line coloured by
      speed. Evidence that the net transfers to the deployment map (D1 interim).
  (b) the keep-best online refinement that PRODUCED that net on the deployment map: per-episode laps,
      the banked incumbent ratcheting up monotonically, every off-track (x) and slower (.) attempt
      reverted. Safe online adaptation on the deployment map (D2 interim).

Reads only:  compare_nets.npz (traj_labtest-s2),  qual_labtest.json (episodes),
             mpcc_tuning/tracks/lab_test_ev8_corridor.npz,  lab_test_ev8.pgm/.yaml.
No acados. Reproduce:  python3 results/deploy_labtest/paper_deploy_labmap.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/poxx/github/mpcc-online-tuning")
OUT = Path("/home/poxx/github/R-AL-MPCC_online_learning/figures/deploy_labmap.pdf")
CN = np.load(ROOT / "results/deploy_labtest/compare_nets.npz", allow_pickle=True)
COR = np.load(ROOT / "mpcc_tuning/tracks/lab_test_ev8_corridor.npz")
QUAL = json.load(open(ROOT / "results/deploy_labtest/qual_labtest.json"))

# Times-compatible serif (Nimbus Roman) + STIX math so this matches the paper's Times type and the
# other result figures -- not matplotlib's default DejaVu Serif.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})
VMIN, VMAX = 1.0, 2.0
SEED = 2                                   # the deploy-default net


def load_map():
    """SLAM occupancy grid (P5 .pgm) -> array in [0,1] + world extent from the .yaml."""
    import re
    from PIL import Image
    y = (ROOT / "lab_test_ev8.yaml").read_text()
    res = float(re.search(r"resolution:\s*([\d.]+)", y).group(1))
    ox, oy = [float(v) for v in re.search(r"origin:\s*\[([^\]]+)\]", y).group(1).split(",")[:2]]
    img = np.asarray(Image.open(ROOT / "lab_test_ev8.pgm")).astype(float) / 255.0
    H, W = img.shape
    extent = [ox, ox + W * res, oy, oy + H * res]        # pgm row 0 is the TOP -> origin="upper"
    return img, extent


def corridor_edges():
    cx, cy, wl, wr = COR["cx"], COR["cy"], COR["wl"], COR["wr"]
    t = np.gradient(np.c_[cx, cy], axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.c_[-t[:, 1], t[:, 0]]                         # left normal
    left = np.c_[cx, cy] + n * wl[:, None]
    right = np.c_[cx, cy] - n * wr[:, None]
    return left, right


fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.0), constrained_layout=True,
                        gridspec_kw={"height_ratios": [1.25, 1.0]})

# ---- (a) deployed net on the lab map, frozen, speed-coloured ----
a = axs[0]
img, extent = load_map()
a.imshow(img, cmap="gray", extent=extent, origin="upper", vmin=-0.35, vmax=1, zorder=0, alpha=0.9)
left, right = corridor_edges()
a.plot(left[:, 0], left[:, 1], color="0.55", lw=0.7, zorder=1)
a.plot(right[:, 0], right[:, 1], color="0.55", lw=0.7, zorder=1)
a.plot(COR["raceline"][:, 0], COR["raceline"][:, 1], color="0.35", lw=0.8,
       ls=(0, (4, 3)), zorder=1.5)                        # raceline
tr = CN["traj_labtest-s2"]                                # (T,3): x,y,v -- deploy net, frozen, clean
sc = a.scatter(tr[:, 0], tr[:, 1], c=np.clip(tr[:, 2], VMIN, VMAX), cmap="viridis",
               s=2.4, vmin=VMIN, vmax=VMAX, zorder=3)
a.plot(tr[0, 0], tr[0, 1], marker="o", mfc="none", mec="k", mew=1.2, ms=7, zorder=5)   # start
# crop to the track with a small margin; 1 m scale bar in the bottom margin
xlo, xhi = COR["cx"].min() - 0.4, COR["cx"].max() + 0.4
ylo, yhi = COR["cy"].min() - 0.4, COR["cy"].max() + 0.4
a.plot([xlo + 0.15, xlo + 1.15], [ylo + 0.18, ylo + 0.18], color="k", lw=1.8, zorder=6)
a.text(xlo + 0.65, ylo + 0.10, "1 m", ha="center", va="top", fontsize=7, zorder=6)
a.set_xlim(xlo, xhi); a.set_ylim(ylo, yhi); a.set_aspect("equal")
a.set_xticks([]); a.set_yticks([])
for s in a.spines.values():
    s.set_visible(False)
a.set_title("(a) deployed net on the lab map (frozen)", fontsize=9, loc="left")
cb = fig.colorbar(sc, ax=a, shrink=0.82, aspect=18, pad=0.015)
cb.set_label("speed [m/s]", fontsize=8); cb.ax.tick_params(labelsize=7)

# ---- (b) keep-best ratchet that produced the deployed net ----
b = axs[1]
ep = QUAL["episodes"][f"icra_t2_smooth|{SEED}|tuner"]
n = np.array([e["ep"] for e in ep]); laps = np.array([e["laps"] for e in ep])
best = np.array([e["best"] if e["best"] is not None else 0.0 for e in ep])
COL = "#1f77b4"
b.step(n, best, where="post", color=COL, lw=2.0, zorder=2)                      # banked incumbent
for e in ep:                                                                    # per-episode attempts
    if e["off"]:
        b.plot(e["ep"], e["laps"], "x", color="#c0392b", ms=7, mew=1.6, zorder=4)
    elif e["action"] == "banked":
        b.plot(e["ep"], e["laps"], "o", mfc=COL, mec="k", mew=0.6, ms=7, zorder=4)
    else:
        b.plot(e["ep"], e["laps"], ".", color="0.55", ms=9, zorder=3)
b.set_xlabel("online-learning episode"); b.set_ylabel("laps (per episode)")
b.set_xlim(-0.5, n.max() + 0.5); b.set_ylim(0, max(laps.max(), best.max()) + 0.4)
b.grid(alpha=0.25); b.tick_params(labelsize=8)
b.set_title("(b) keep-best online refinement on the lab map", fontsize=9, loc="left")
from matplotlib.lines import Line2D
b.legend(handles=[Line2D([], [], color=COL, lw=2.0),
                  Line2D([], [], marker="o", color="w", mfc=COL, mec="k", ms=7),
                  Line2D([], [], marker="x", color="#c0392b", ms=7, mew=1.6, ls=""),
                  Line2D([], [], marker=".", color="0.55", ms=9, ls="")],
         labels=["banked net", "banked", "off-track (rev.)", "slower (rev.)"],
         loc="lower right", fontsize=6.5, ncol=2, frameon=True, framealpha=0.9,
         handlelength=1.3, columnspacing=1.0, handletextpad=0.4)

fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("wrote", OUT)
ev = [e["laps"] for e in QUAL["evals"][f"icra_t2_smooth|{SEED}|tuner"]]
print(f"  seed{SEED}: banked {best.max():.2f} laps (episodic) -> frozen eval {np.mean(ev):.2f} clean laps; "
      f"deploy drive {tr.shape[0]} steps, v {tr[:,2].min():.2f}-{tr[:,2].max():.2f} m/s")
