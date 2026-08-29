"""LTC-based online tuning: does the recurrence earn its place?

    python3 experiments/ltc_behaviour.py --seeds 5

TODO item 2d, run rather than specified. Four arms on the oval with one slower
opponent, everything identical except what emits theta:

``global``    one theta tuned by TD(lambda) -- the existing method, no situation
``fixed``     a hand-written lookup on the same features (the scripted control)
``mlp``       a memoryless net, trained by the identical rule
``ltc``       a liquid time-constant cell, trained by the identical rule

**The gate, fixed before running: the LTC is claimed only if it beats both
``fixed`` and ``mlp``.** Otherwise it is decoration, and this file says so.

The prediction stated against ourselves in ``docs/source/behaviour_policy.md``
is that ``fixed`` will be strong, because the behaviour boundary measured in
``overtake_or_follow.py`` is the ratio q_v/q_c -- a *difference of log weights*,
and therefore linear in theta. The recurrence can only win on the temporal axis:
a closing rate is a derivative of a range and is not in any single frame.

**The opponent's speed is deliberately withheld from the features.** If the
policy were given it there would be no hidden state, no temporal pattern, and no
role for recurrence -- the gate above would then be measuring nothing. That
choice is what creates the problem the network is credited with solving, so it
is made explicitly here rather than inherited.

Reported: distance covered, passes, and crashes. Return alone is the wrong
headline -- a policy that never passes anybody and never crashes scores
respectably, because progress alone pays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (LTCCell, MLPCell, N_FEATURES, THETA_HI,  # noqa: E402
                             THETA_LO, PolicyTuner, WeightPolicy, features,
                             fixed_schedule)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

ARMS = ("global", "fixed", "mlp", "ltc")


def signed_gap(track, s_ego, s_opp):
    d = (s_opp - s_ego) % track.length
    return d - track.length if d > track.length / 2 else d


def run(arm, seed=0, n_ep=20, steps=300, n_hidden=12):
    from examples.tune_online import Plant

    track = Track.oval()
    theta0 = MPCCWeights(q_c=10.0, q_l=200.0, q_v=0.5, r_d=1.0).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)

    tuner = pol = None
    if arm in ("mlp", "ltc"):
        cell = (LTCCell if arm == "ltc" else MLPCell)(N_FEATURES, n_hidden, seed=seed)
        pol = WeightPolicy(cell, theta0, THETA_LO, THETA_HI, seed=seed)
        tuner = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0,
                            seed=seed)
    elif arm == "global":
        from mpcc_tuning.learner import QLambdaTuner
        gtheta = theta0.copy()
        tuner = QLambdaTuner(m, len(theta0), gamma=0.98, lam=0.9, alpha=2e-3,
                             explore=0.05, delta_clip=1.0, seed=seed)

    rows = []
    for ep in range(n_ep):
        opp = Opponent(track, s0=3.0, speed=1.0 + 0.2 * ((seed + ep) % 3),
                       radius=0.24)
        P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
        s5 = P.reset()
        m.reset()
        m.set_obstacles(P.keepouts())
        if pol is not None:
            tuner.reset()
        feat = features(track, s5, [opp])

        if arm in ("mlp", "ltc"):
            theta, u = tuner.act(feat, s5)
        elif arm == "global":
            tuner.reset()
            u = tuner.start(gtheta, s5)
            theta = gtheta
        else:
            theta = fixed_schedule(feat, theta0)
            u = m.value(s5, theta)["u0"]

        cov, off, passes, seen = 0.0, False, 0, False
        for _ in range(steps):
            s5n, r, off, tr = P.step(u)
            cov += r
            m.set_obstacles(P.keepouts())
            fn = features(track, s5n, [opp])
            g = signed_gap(track, track.project(s5n[0], s5n[1]), opp.s)
            if g < 0 and abs(g) < track.length / 4 and not seen:
                passes, seen = passes + 1, True
            elif g > 0.5:
                seen = False

            if arm in ("mlp", "ltc"):
                out = tuner.learn(r, s5n, fn, off)
                if out[0] is None:
                    break
                theta, u = out
            elif arm == "global":
                gtheta, u = tuner.step(gtheta, s5, r, s5n, off)
                theta = gtheta
            else:
                theta = fixed_schedule(fn, theta0)
                u = m.value(s5n, theta)["u0"]
            s5 = s5n
            if off or tr:
                break
        rows.append(dict(ep=ep, covered=float(cov), off=bool(off),
                         failure=P.failure, passes=int(passes)))
    last = rows[-8:]
    return dict(arm=arm, seed=seed,
                covered=float(np.mean([r["covered"] for r in last])),
                passes=float(np.mean([r["passes"] for r in last])),
                crashes=float(np.mean([r["off"] for r in last])),
                rows=rows)


def one(job):
    arm, seed, n_ep, steps = job
    t0 = time.perf_counter()
    out = run(arm, seed=seed, n_ep=n_ep, steps=steps)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "ltc.json"))
    a = ap.parse_args(argv)

    jobs = [(arm, s, a.episodes, a.steps) for s in range(a.seeds) for arm in ARMS]
    n_proc = a.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  {len(jobs)} runs, {a.episodes} episodes, {n_proc} processes\n", flush=True)

    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(n_proc) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o)
            print(f"  done  {o['arm']:<7} seed {o['seed']}  {o['covered']:6.1f} m"
                  f"  {o['passes']:.2f} passes  {o['crashes']:.0%} crash"
                  f"  ({o['wall_s']:.0f} s)", flush=True)

    print(f"\n  {'arm':<8}{'covered':>9}{'sd':>7}{'passes':>9}{'crashes':>9}")
    S = {}
    for arm in ARMS:
        r = [x for x in res if x["arm"] == arm]
        c = np.array([x["covered"] for x in r])
        S[arm] = dict(covered=float(c.mean()), sd=float(c.std(ddof=1)),
                      passes=float(np.mean([x["passes"] for x in r])),
                      crashes=float(np.mean([x["crashes"] for x in r])), n=len(r))
        print(f"  {arm:<8}{c.mean():9.1f}{c.std(ddof=1):7.1f}"
              f"{S[arm]['passes']:9.2f}{S[arm]['crashes']:9.0%}")

    print("\n  the gate: LTC is claimed only if it beats BOTH fixed and mlp")
    ok = True
    for base in ("fixed", "mlp"):
        d = S["ltc"]["covered"] - S[base]["covered"]
        se = np.hypot(S["ltc"]["sd"], S[base]["sd"]) / np.sqrt(S["ltc"]["n"])
        sep = abs(d) / max(se, 1e-9)
        verdict = "beats" if (d > 0 and sep > 2) else "does NOT beat"
        ok &= (d > 0 and sep > 2)
        print(f"    ltc - {base:<6} = {d:+6.1f} m  {sep:4.1f} SE   -> {verdict}")
    print(f"\n  VERDICT: {'the recurrence earns its place' if ok else 'NOT CLAIMED -- decoration'}")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(summary=S, runs=res), indent=2) + "\n")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
