"""Weight-learning GIF: the car driving a race (left: track + speed-coloured recent trail + opponent)
alongside a LIVE weight panel (right: k_v / r_a / q_c / q_v as log2(w / START), updating each frame), so
you can WATCH the online tuner adapt the weights through the race. Animated from the saved caches
(no acados re-drive). Default = the ONLINE (explore+learn) run, which is where the weights move.

    python3 tools/race_eval_laps.py --seed 0          # online caches (states_online_*)
    python3 tools/paper_race_weight_gif.py --seed 0 --kinds equal faster
Writes results/race/gif/weights_<kind>_s<seed>.gif .
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning import baselines as B

OUT = ROOT / "results/race/gif"; OUT.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / "results/race/paper"
L = Track.icra_t2_smooth().length
SHOW = [("k_v", "#2a8f3a"), ("r_a", "#c23b3b"), ("q_c", "#1f4e8c"), ("q_v", "#d08020")]


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -L / 2] += L; dS[dS > L / 2] -= L
    return np.concatenate([[0.0], np.cumsum(dS)]) / L


def make(seed, kind, frozen=False, stride=12, fps=18, trail=90):
    pre = "states_10lapfrozen" if frozen else "states_online"
    f = PAPER / f"{pre}_ltc_{seed}_{kind}.npz"
    if not f.exists():
        print(f"({kind}: no {pre} cache)"); return
    z = np.load(f); EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
    TH = z["THETA"]; lap = _laps(z["S"]); PASS = np.asarray(z["PASS"], float)
    start = np.exp(np.asarray(B.start("icra_t2_smooth").theta(), float))
    idx = [WEIGHT_NAMES.index(n) for n, _ in SHOW]; cols = [c for _, c in SHOW]
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")

    fig, (axt, axw) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    axt.plot(b["left_x"], b["left_y"], color="0.6", lw=1); axt.plot(b["right_x"], b["right_y"], color="0.6", lw=1)
    axt.plot(b["ref_x"], b["ref_y"], color="0.4", lw=.7, ls=(0, (4, 3)), alpha=.6)
    axt.set_aspect("equal"); axt.set_xticks([]); axt.set_yticks([])
    trail_sc = axt.scatter([], [], c=[], cmap="viridis", vmin=1.2, vmax=2.8, s=10, zorder=4)
    ego_m = axt.scatter([], [], s=90, marker="o", color="#1f4e8c", edgecolor="k", zorder=6, label="ego")
    opp_m = axt.scatter([], [], s=80, marker="s", color="#d08020", edgecolor="k", zorder=6, label="opponent")
    axt.legend(loc="upper left", fontsize=8); cb = fig.colorbar(trail_sc, ax=axt, shrink=.8); cb.set_label("ego speed [m/s]")
    bars = axw.bar(range(len(SHOW)), [0] * len(SHOW), color=cols)
    axw.axhline(0, color="0.5", lw=.8); axw.set_xticks(range(len(SHOW))); axw.set_xticklabels([n for n, _ in SHOW])
    axw.set_ylim(-2.2, 2.2); axw.set_ylabel("log2(weight / START)"); axw.set_title("emitted weights (online)", fontsize=10)
    ttl = fig.suptitle("", fontsize=11)
    xpad = (EX.max() - EX.min()) * .05 + .5; ypad = (EY.max() - EY.min()) * .05 + .5
    axt.set_xlim(EX.min() - xpad, EX.max() + xpad); axt.set_ylim(EY.min() - ypad, EY.max() + ypad)
    frames = list(range(0, len(EX), stride))

    def update(i):
        lo = max(0, i - trail)
        pts = np.column_stack([EX[lo:i + 1], EY[lo:i + 1]])
        trail_sc.set_offsets(pts if len(pts) else np.empty((0, 2))); trail_sc.set_array(V[lo:i + 1])
        ego_m.set_offsets([[EX[i], EY[i]]]); opp_m.set_offsets([[OX[i], OY[i]]])
        for bar, j in zip(bars, idx):
            bar.set_height(float(np.log2(max(TH[i, j] / start[j], 1e-3))))
        ttl.set_text(f"vs {kind}  —  lap {lap[i]:.1f},  v {V[i]:.2f} m/s,  gap {z['GAP'][i]:+.1f} m,  passes {int(PASS[i])}")
        return [trail_sc, ego_m, opp_m, *bars, ttl]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = OUT / f"weights_{kind}_s{seed}.gif"
    anim.save(str(out), writer=PillowWriter(fps=fps)); plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=["equal", "faster"])
    ap.add_argument("--frozen", action="store_true")
    a = ap.parse_args()
    for k in a.kinds:
        make(a.seed, k, frozen=a.frozen)
