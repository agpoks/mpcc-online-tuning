"""The car driving an ICRA circuit while its own cost weights are learned.

    python3 scripts/anim_learning.py --track icra_t2_raceline

Two panels moving together: the car on the track, and every learned parameter
on a log axis with a marker at the current tick. The point is to see the two at
once -- which weights move, and *where* on the track they move -- because a
static plot of either separately cannot show that.

The sector the policy is being told about is shaded behind the trace, since the
sector one-hot is one of the network's inputs rather than an annotation for the
reader.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,  # noqa: E402
                             PolicyTuner, WeightPolicy, features)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights, WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static" / "anim"
INK, BLUE, RED, GREEN, AMBER = "#22303f", "#2f6fb2", "#c0392b", "#1e8449", "#d68910"
CAT = (BLUE, RED, GREEN, AMBER)
GROUPS = (("path", ("q_c", "q_l", "q_v")),
          ("input", ("r_d", "r_a", "r_dv")),
          ("constraint", ("d_obs", "k_v")))


def roll(track_name, episodes, steps, seed, q_c, q_l, r_d, opponents):
    track = getattr(Track, track_name)()
    from examples.tune_online import Plant
    th0 = MPCCWeights(q_c=q_c, q_l=q_l, q_v=2.0, r_d=r_d).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=80, max_obstacles=1 if opponents else 0)
    pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, THETA_LO,
                       THETA_HI, seed=seed, gauge_fix=True)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0, seed=seed,
                     trust_region=0.01, theta_prior=0.0)
    rec = []
    for ep in range(episodes):
        opp = (Opponent(track, s0=6.0, speed=(0.0, 1.0, 2.6, 3.4)[ep % 4],
                        radius=0.24) if opponents else None)
        P = Plant(track, dt=0.05, max_steps=steps,
                  opponents=[opp] if opp else [])
        s5 = P.reset(); m.reset(); tu.reset()
        if opp:
            m.set_obstacles(P.keepouts())
        tr = ObstacleTracker(dt=0.05)
        if opp:
            tr.update(opp.pose()[:2])
        f = features(track, s5, [opp] if opp else [],
                     opp_speed_est=tr.speed if opp else None)
        th, u = tu.act(f, s5)
        for _ in range(steps):
            s5n, r, off, done = P.step(u)
            if opp:
                m.set_obstacles(P.keepouts()); tr.update(opp.pose()[:2])
            rec.append((ep, float(s5n[0]), float(s5n[1]), float(s5n[3]),
                        int(track.sector(track.wrap(float(s5n[4])))),
                        *np.exp(th),
                        *(opp.pose()[:2] if opp else (np.nan, np.nan))))
            f = features(track, s5n, [opp] if opp else [],
                         opp_speed_est=tr.speed if opp else None)
            out = tu.learn(r, s5n, f, off)
            if out[0] is None:
                break
            th, u = out; s5 = s5n
            if off or done:
                break
        print(f"    ep {ep}: {len(rec)} ticks", flush=True)
    return track, np.array(rec)


def render(track, R, path, title, fps=25, stride=2):
    R = R[::stride]
    n = len(R)
    ep, xs, ys, v, sec = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4].astype(int)
    W = R[:, 5:5 + len(WEIGHT_NAMES)]
    ox, oy = R[:, -2], R[:, -1]
    idx = {nm: i for i, nm in enumerate(WEIGHT_NAMES)}

    fig = plt.figure(figsize=(12.6, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.32, wspace=0.16)
    axt = fig.add_subplot(gs[:, 0])
    axw = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # One thick ribbon coloured by sector, not centreline-plus-two-edges.
    #
    # T1's half-width runs 0.35-1.56 m, so edges offset from the centreline
    # self-intersect at the hairpins and the track came out looking like a
    # maze. The sector colouring also earns its place here: it is the one-hot
    # the policy receives, so the viewer sees what the network is told.
    from matplotlib.collections import LineCollection
    pts = track.center.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    sec_at = np.array([int(track.sector(track.wrap(v))) for v in track.s])
    for j in range(4):
        msk = sec_at[:-1] == j
        if msk.any():
            axt.add_collection(LineCollection(segs[msk], colors=[CAT[j]],
                                              linewidth=7.0, alpha=0.30,
                                              capstyle="round", zorder=0))
    axt.legend(handles=[plt.Line2D([], [], color=CAT[j], lw=6, alpha=0.45,
                                   label=Track.SECTOR_NAMES[j])
                        for j in range(4) if (sec_at == j).any()],
               fontsize=7.5, frameon=False, loc="lower left", ncol=2)
    axt.set_aspect("equal"); axt.axis("off")
    (trail,) = axt.plot([], [], "-", color=INK, lw=2.0, alpha=0.9)
    (car,) = axt.plot([], [], "o", ms=9, color=RED, mec="white", mew=1.3, zorder=5)
    (opp_m,) = axt.plot([], [], "o", ms=9, color=INK, mec="white", mew=1.3, zorder=5)
    hud = axt.text(0.02, 0.98, "", transform=axt.transAxes, va="top", fontsize=10,
                   color=INK, family="monospace")

    lines = {}
    t = np.arange(n)
    for ax, (lab, group) in zip(axw, GROUPS):
        for c, nm in enumerate(group):
            y = W[:, idx[nm]]
            ax.plot(t, y, "-", color=CAT[c], lw=1.0, alpha=0.30)
            (ln,) = ax.plot([], [], "-", color=CAT[c], lw=1.8)
            (dot,) = ax.plot([], [], "o", ms=5, color=CAT[c])
            lines[nm] = (ln, dot, y)
            ax.annotate(f" {nm}", (n - 1, y[-1]), color=CAT[c], fontsize=8,
                        va="center", annotation_clip=False)
        for b in np.flatnonzero(np.diff(ep)) + 1:
            ax.axvline(b, color="0.9", lw=0.7, zorder=0)
        ax.set_yscale("log"); ax.set_ylabel(lab, fontsize=9)
        ax.set_xlim(0, n - 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axw[-1].set_xlabel("control tick", fontsize=9)

    def frame(k):
        a = max(0, k - 400)
        trail.set_data(xs[a:k + 1], ys[a:k + 1])
        car.set_data([xs[k]], [ys[k]])
        if np.isfinite(ox[k]):
            opp_m.set_data([ox[k]], [oy[k]])
        hud.set_text(f"episode {int(ep[k]):>2d}\nv  {v[k]:5.2f} m/s\n"
                     f"{Track.SECTOR_NAMES[sec[k]]}")
        for nm, (ln, dot, y) in lines.items():
            ln.set_data(t[:k + 1], y[:k + 1])
            dot.set_data([t[k]], [y[k]])
        return [trail, car, opp_m, hud, *[x for v_ in lines.values() for x in v_[:2]]]

    fig.suptitle(title, fontsize=10.5, color=INK, y=0.98)
    an = FuncAnimation(fig, frame, frames=n, interval=1000 / fps, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    an.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {path}  ({n} frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q-c", type=float, default=0.1)
    ap.add_argument("--q-l", type=float, default=50.0)
    ap.add_argument("--r-d", type=float, default=0.1)
    ap.add_argument("--opponents", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    track, R = roll(a.track, a.episodes, a.steps, a.seed, a.q_c, a.q_l, a.r_d,
                    a.opponents)
    if not len(R):
        print("  no ticks recorded"); return
    out = Path(a.out) if a.out else OUT / f"learning_{a.track}.gif"
    render(track, R, out,
           f"{a.track}: the weights being learned while the car drives")


if __name__ == "__main__":
    main()
