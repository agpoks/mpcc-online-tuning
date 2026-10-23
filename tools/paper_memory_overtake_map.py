#!/usr/bin/env python3
"""Why/where memory helps: a 2D matched-encounter map on the EQUAL-pace occluded opponent.
Same initial conditions (seed), deployed FROZEN nets. Two panels (LTC | MLP), each showing the ego
path over a fixed early window coloured SEEN (blue) vs BLIND (grey) under the 270-deg occluding sensor,
the opponent's position at the end of the window, and the overtake point (star). The contrast: by the
time LTC has tracked the occluded car through the blind stretches and passed, MLP is still trailing it.
HONEST SCOPE: this is ONE representative encounter drawn from the deployed-frozen sweep
(equal_frozen_sweep.npz); the quantitative claim is the aggregate bar chart (tools/paper_memory_bars.py:
deployed pass 0.89 LTC vs 0.67 MLP, n=9). This map shows the MECHANISM behind that pass gap: LTC tracks
the occluded car through the blind stretch and completes the pass, while the memoryless net loses it
(higher blind%) and stalls behind. DATA-ONLY.
Reproduce:  python3 tools/paper_memory_overtake_map.py --net 3 --start 3 --window 50
            python3 tools/paper_memory_overtake_map.py --seed 3 --window 60   # old static/equal pair
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/race_phase2"
BND = ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz"
FIG = Path("/home/poxx/github/R-AL-MPCC_online_learning/figures/memory_overtake_map.pdf")
DT = 0.05
SEEN, BLIND = "#1f4e8c", "#b8b8b8"
# Times-compatible serif (Nimbus Roman) + STIX math to match the paper and the TikZ figures.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})


def draw_track_bg(ax, b):
    ax.set_facecolor("0.93")
    ax.fill(np.r_[b["left_x"], b["right_x"][::-1]], np.r_[b["left_y"], b["right_y"][::-1]],
            facecolor="white", edgecolor="none", zorder=0.5)
    ax.plot(b["left_x"], b["left_y"], color="0.6", lw=0.8, zorder=1)
    ax.plot(b["right_x"], b["right_y"], color="0.6", lw=0.8, zorder=1)
    if "ref_x" in b.files:                      # the raceline = the equal-pace opponent's line
        ax.plot(b["ref_x"], b["ref_y"], color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=1.5)
    xs = np.r_[b["left_x"], b["right_x"]]; ys = np.r_[b["left_y"], b["right_y"]]
    xlo, ylo = xs.min(), ys.min()
    ax.plot([xlo, xlo + 1.0], [ylo - 0.4, ylo - 0.4], color="k", lw=2.2, zorder=6)
    ax.text(xlo + 0.5, ylo - 0.3, "1 m", ha="center", va="bottom", fontsize=7, zorder=6)
    ax.set_xlim(xs.min() - 0.6, xs.max() + 0.6); ax.set_ylim(ys.min() - 0.7, ys.max() + 0.6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def seen_blind_line(ax, X, Y, det, lw=2.4):
    pts = np.array([X, Y]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = np.where(det[:-1], SEEN, BLIND)
    lc = LineCollection(segs, colors=cols, linewidths=lw, zorder=4); ax.add_collection(lc)


def passtick(G):
    cr = np.where((G[:-1] > 0) & (G[1:] <= 0) & (np.abs(G[1:]) < 10))[0]
    return int(cr[0]) if len(cr) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3, help="old static/equal pair: reads sensor_equal_{arm}_{seed}.npz")
    ap.add_argument("--net", type=int, default=None, help="deployed-sweep net seed; with --start reads sensor_eqsweep_{arm}_{start}_n{net}.npz")
    ap.add_argument("--start", type=int, default=None, help="deployed-sweep start index")
    ap.add_argument("--window", type=float, default=60.0, help="seconds of the encounter to draw")
    a = ap.parse_args()
    use_sweep = a.net is not None and a.start is not None
    srcfile = (lambda arm: OUT / f"sensor_eqsweep_{arm}_{a.start}_n{a.net}.npz") if use_sweep \
        else (lambda arm: OUT / f"sensor_equal_{arm}_{a.seed}.npz")
    srctag = f"net {a.net} start {a.start} (deployed frozen)" if use_sweep else f"seed {a.seed}"
    b = np.load(BND); W = int(a.window / DT)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.9), constrained_layout=True)
    for ax, arm, lab in ((axs[0], "ltc", "LTC (recurrent)"), (axs[1], "mlp", "MLP (memoryless)")):
        d = np.load(srcfile(arm), allow_pickle=True)
        EX, EY = np.asarray(d["EX"], float), np.asarray(d["EY"], float)
        OX, OY = np.asarray(d["OX"], float), np.asarray(d["OY"], float)
        det = np.asarray(d["DET"], float).astype(bool); G = np.asarray(d["GAP"], float)
        n = min(W, len(EX)); sl = slice(0, n)
        draw_track_bg(ax, b)
        seen_blind_line(ax, EX[sl], EY[sl], det[sl])
        ax.plot(EX[0], EY[0], marker="o", mfc="none", mec="k", mew=1.4, ms=8, zorder=6)  # start
        ax.plot(EX[n - 1], EY[n - 1], marker="o", mfc=SEEN, mec="k", mew=0.8, ms=9, zorder=7)  # ego now
        ax.plot(OX[n - 1], OY[n - 1], marker="s", mfc="0.4", mec="k", mew=0.8, ms=9, zorder=7)  # opp now
        pt = passtick(G)
        passed = pt is not None and pt < n
        if passed:
            ax.scatter(EX[pt], EY[pt], marker="*", s=230, facecolor="gold", edgecolor="k", lw=0.8, zorder=8)
        allpt = passtick(G)
        tag = (f"passed at {allpt * DT:.0f} s" if passed else
               (f"still trailing (passes at {allpt * DT:.0f} s)" if allpt is not None else "no pass"))
        ax.set_title(f"{lab}\n{tag}", fontsize=9)
    # shared legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=SEEN, lw=2.4, label="opponent in view"),
           Line2D([0], [0], color=BLIND, lw=2.4, label="opponent occluded (blind)"),
           Line2D([0], [0], marker="o", mfc="none", mec="k", ls="", label="start"),
           Line2D([0], [0], marker="s", mfc="0.4", mec="k", ls="", label="opponent"),
           Line2D([0], [0], marker="*", mfc="gold", mec="k", ls="", label="overtake")]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight"); fig.savefig(FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", FIG, f"(window {a.window:.0f}s, {srctag})")
    for arm in ("ltc", "mlp"):
        d = np.load(srcfile(arm), allow_pickle=True)
        G = np.asarray(d["GAP"], float); pt = passtick(G)
        print(f"  {arm}: pass@{None if pt is None else round(pt*DT,1)}s  blind%={100*(1-np.asarray(d['DET'],float).mean()):.0f}"
              f"  clean={(not bool(d['off'][0])) and (not bool(d['contact'][0]))}")


if __name__ == "__main__":
    main()
