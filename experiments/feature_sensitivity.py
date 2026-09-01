"""Does the trained policy actually *use* the features, or merely receive them?

    python3 experiments/feature_sensitivity.py --seeds 3

Having the sector in the input vector is not the same as learning to condition
on it, and it is easy to claim the first while meaning the second. This
measures the second.

Method: train the policy online, then **freeze it** and sweep one feature group
at a time, holding the rest at a neutral baseline, and record the theta it
emits. If the emitted weights do not move when the sector changes, the policy
is ignoring the sector however prominently the sector appears in its input.

Reported per group as the spread of the emitted ``q_v/q_c`` -- the ratio,
because that is the axis with a measured meaning (the behaviour boundary at 1),
and a spread of zero means the group is decorative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, OPPONENT_CLASSES,  # noqa: E402
                             THETA_HI, THETA_LO, PolicyTuner, WeightPolicy,
                             features)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

GROUPS = {"sector": (9, 13), "opponent class": (14, 18),
          "curvature preview": (0, 3), "gap": (5, 9), "track width": (13, 14)}


def train(seed, n_ep, steps, gauge_fix=False):
    from examples.tune_online import Plant
    track = Track.oval()
    th0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, THETA_LO,
                       THETA_HI, seed=seed, gauge_fix=gauge_fix)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0, seed=seed,
                     trust_region=0.01, theta_prior=0.5)
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
            out = tu.learn(r, s5n, features(track, s5n, [opp], opp_speed_est=tr.speed), off)
            if out[0] is None:
                break
            th, u = out
            s5 = s5n
            if off or done:
                break
    return pol


def ratio_for(pol, feat, settle=12):
    """Emit theta for a held-constant input, after the cell settles."""
    pol.reset()
    for _ in range(settle):
        th = pol.step(feat)
    return float(np.exp(th[2]) / np.exp(th[0]))


def _sweep_one(seed, episodes, steps, base, gauge_fix=False):
    """Train one seed and sweep every feature group. Module level so it pickles."""
    pol = train(seed, episodes, steps, gauge_fix)
    per_group = {}
    for g, (lo, hi) in GROUPS.items():
        rs = []
        if hi - lo > 1 and g in ("sector", "opponent class"):
            for k in range(lo, hi):                  # sweep the one-hot
                f = base.copy(); f[lo:hi] = 0.0; f[k] = 1.0
                rs.append(ratio_for(pol, f))
        else:
            for v in (0.0, 0.5, 1.0):                # sweep the magnitude
                f = base.copy(); f[lo:hi] = v
                rs.append(ratio_for(pol, f))
        per_group[g] = rs
    return per_group


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--gauge-fix", action="store_true",
                    help="hold the mean log cost weight fixed, removing the "
                         "direction along which V = -J* can be raised without "
                         "driving any differently")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "feature_sensitivity.json"))
    a = ap.parse_args(argv)

    base = np.zeros(N_FEATURES)
    base[3], base[8], base[13] = 0.5, 0.5, 0.5
    base[9], base[15] = 1.0, 1.0            # straight, slower opponent

    out = {g: [] for g in GROUPS}
    # Seeds are independent; training them one after another on one core is
    # hours of wall clock for no reason.
    with ProcessPoolExecutor(max_workers=min(a.jobs, a.seeds)) as ex:
        futs = [ex.submit(_sweep_one, seed, a.episodes, a.steps, base, a.gauge_fix)
                for seed in range(a.seeds)]
        for seed, fut in enumerate(futs):
            per_group = fut.result()
            for g, rs in per_group.items():
                out[g].append(rs)
            print(f"  seed {seed} done", flush=True)
            for g, rs in per_group.items():
                print(f"    {g:<18} ratios {np.round(rs, 3)}", flush=True)

    print(f"\n  {'feature group':<18}{'spread of q_v/q_c':>20}{'relative':>11}   verdict")
    summary = {}
    for g, runs in out.items():
        R = np.array(runs)
        per_seed_rel = (R.max(1) - R.min(1)) / np.maximum(R.mean(1), 1e-9)
        spread = float(np.mean(R.max(1) - R.min(1)))
        rel = float(np.mean(per_seed_rel))
        se = float(np.std(per_seed_rel, ddof=1) / np.sqrt(len(per_seed_rel)))
        used = rel - se > 0.05
        summary[g] = dict(spread=spread, relative=rel, relative_se=se,
                          per_seed=per_seed_rel.tolist(), used=bool(used))
        print(f"  {g:<18}{spread:>20.4f}{rel:>9.1%} +-{se:<6.1%} "
              f"{'USED' if used else 'ignored -- decorative'}")
    print("\n  A group whose spread is ~0 is in the input and not in the policy.")
    Path(a.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
