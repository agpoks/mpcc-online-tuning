"""All eight weights, against every situation the policy is meant to distinguish.

    python3 experiments/weight_matrix.py --seeds 3

The policy is 18 inputs -> LTC(12) -> **8 weights**: each weight has its own row
of the readout and can respond differently to the same situation. Every report
so far collapsed that to one number -- the ratio q_v/q_c -- which cannot show
whether ``r_a`` behaves differently from ``d_obs``, or whether the corridor
width moves anything at all.

This trains the policy, freezes it, and sweeps the *situation* over the grid it
is supposed to distinguish -- four named sectors crossed with four opponent
classes, at two corridor widths -- reporting **every weight in every cell**.

Read it as: a row that is flat across the grid is a weight the policy does not
condition on, and a grid that is flat everywhere is a policy that is not a
policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, OPPONENT_CLASSES,  # noqa: E402
                             THETA_HI, THETA_LO, PolicyTuner, WeightPolicy,
                             features)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights, WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

TH0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()


def train(seed, n_ep, steps, theta_explore, trace=None):
    """Train online. If ``trace`` is a list, append every tick's emitted theta.

    The frozen end-state grid says what the policy *became*; the trace says how
    it got there, tick by tick, while the car was driving. Both are needed: a
    flat grid with a moving trace is a policy that learned and then stopped
    conditioning, which is a different failure from one that never moved.
    """
    from examples.tune_online import Plant
    track = Track.oval()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), TH0, THETA_LO,
                       THETA_HI, seed=seed)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0, seed=seed,
                     theta_explore=theta_explore)
    for ep in range(n_ep):
        kind = (seed + ep) % 4
        opp = Opponent(track, s0=3.0, speed=(0.0, 1.0, 2.6, 3.4)[kind], radius=0.24)
        P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
        s5 = P.reset(); m.reset(); m.set_obstacles(P.keepouts()); tu.reset()
        tr = ObstacleTracker(dt=0.05); tr.update(opp.pose()[:2])
        th, u = tu.act(features(track, s5, [opp], opp_speed_est=tr.speed), s5)
        for _ in range(steps):
            s5n, r, off, done = P.step(u)
            m.set_obstacles(P.keepouts()); tr.update(opp.pose()[:2])
            fnow = features(track, s5n, [opp], opp_speed_est=tr.speed)
            out = tu.learn(r, s5n, fnow, off)
            if out[0] is None:
                break
            th, u = out; s5 = s5n
            if trace is not None:
                trace.append((ep, float(s5n[4]), float(s5n[3]),
                              int(np.argmax(fnow[9:13])), kind, *np.exp(th)))
            if off or done:
                break
    return pol


def emit(pol, sector, opp_class, width, settle=14):
    f = np.zeros(N_FEATURES)
    f[3], f[8] = 0.5, 0.4            # mid speed, an opponent within reach
    f[13] = width
    f[9 + sector] = 1.0
    f[14 + opp_class] = 1.0
    pol.reset()
    for _ in range(settle):
        th = pol.step(f)
    return np.exp(th)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--theta-explore", type=float, default=0.10)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "weight_matrix.json"))
    a = ap.parse_args(argv)

    grids, traces = [], []
    for seed in range(a.seeds):
        print(f"  training seed {seed} (theta_explore={a.theta_explore}) ...",
              flush=True)
        tr_ = []
        pol = train(seed, a.episodes, a.steps, a.theta_explore, trace=tr_)
        traces.append(np.array(tr_))
        g = np.zeros((4, 4, 2, len(WEIGHT_NAMES)))
        for si in range(4):
            for oi in range(4):
                for wi, w in enumerate((0.25, 0.85)):
                    g[si, oi, wi] = emit(pol, si, oi, w)
        grids.append(g)
    G = np.array(grids)                      # (seed, sector, opp, width, weight)

    mean = G.mean(0)
    print(f"\n  weights by SECTOR (mean over opponent class, width and seed)")
    print("  %-12s" % "sector" + "".join(f"{n:>9}" for n in WEIGHT_NAMES))
    for si, nm in enumerate(Track.SECTOR_NAMES):
        row = mean[si].mean((0, 1))
        print(f"  {nm:<12}" + "".join(f"{v:9.3f}" for v in row))

    print(f"\n  weights by OPPONENT CLASS (mean over sector, width and seed)")
    print("  %-12s" % "opponent" + "".join(f"{n:>9}" for n in WEIGHT_NAMES))
    for oi, nm in enumerate(OPPONENT_CLASSES):
        row = mean[:, oi].mean((0, 1))
        print(f"  {nm:<12}" + "".join(f"{v:9.3f}" for v in row))

    print(f"\n  weights by CORRIDOR WIDTH")
    print("  %-12s" % "width" + "".join(f"{n:>9}" for n in WEIGHT_NAMES))
    for wi, nm in (("narrow", 0), ("wide", 1))[:0] or [("narrow", 0), ("wide", 1)]:
        row = mean[:, :, nm].mean((0, 1))
        print(f"  {wi:<12}" + "".join(f"{v:9.3f}" for v in row))

    print(f"\n  {'weight':<8}{'sector':>10}{'opponent':>11}{'width':>9}   verdict")
    summary = {}
    for k, n in enumerate(WEIGHT_NAMES):
        base = max(float(mean[..., k].mean()), 1e-9)
        s_sp = float(mean[..., k].mean((1, 2)).ptp()) / base
        o_sp = float(mean[..., k].mean((0, 2)).ptp()) / base
        w_sp = float(mean[..., k].mean((0, 1)).ptp()) / base
        used = max(s_sp, o_sp, w_sp) > 0.05
        summary[n] = dict(sector=s_sp, opponent=o_sp, width=w_sp, used=bool(used))
        print(f"  {n:<8}{s_sp:9.1%}{o_sp:10.1%}{w_sp:8.1%}   "
              f"{'conditions' if used else 'FLAT -- not conditioned on'}")
    # The online trace: how far each weight moved while driving, per episode.
    T = traces[0]
    print(f"\n  ONLINE TRACE, seed 0 -- each weight's range within an episode")
    print("  %-4s%-8s" % ("ep", "ticks") + "".join(f"{n:>9}" for n in WEIGHT_NAMES))
    for ep in range(int(T[:, 0].max()) + 1):
        m_ = T[:, 0] == ep
        if not m_.any():
            continue
        w = T[m_][:, 5:]
        print(f"  {ep:<4d}{int(m_.sum()):<8d}"
              + "".join(f"{v:9.3f}" for v in (w.max(0) - w.min(0))))
    print("  (0.000 means the weight did not move at all while the car drove)")

    Path(a.out).write_text(json.dumps(
        dict(summary=summary, mean=mean.tolist(),
             weights=list(WEIGHT_NAMES), theta_explore=a.theta_explore,
             trace_cols=["ep", "s", "v", "sector", "opp_class"] + list(WEIGHT_NAMES),
             trace=traces[0].tolist()),
        indent=2) + "\n")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
