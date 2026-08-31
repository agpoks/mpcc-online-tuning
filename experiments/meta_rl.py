"""Does meta-RL feedback make the policy situation-dependent?

    python3 experiments/meta_rl.py --seeds 3

The last untested component of RTRRL (Lemmel & Grosu, arXiv:2311.04830). Their
method has three separable parts: TD(lambda) with eligibility traces, online
autodiff (RTRL), and a **meta-RL architecture** that feeds the previous action
and reward back in as network inputs. We had the first two -- the second as
RFLO, measured equivalent to exact RTRL at cosine 0.9997 -- and never the third.

Why this and not another output-layer fix. The anti-saturation term
(:attr:`PolicyTuner.entropy`, our adaptation of their entropy bonus) does move
theta off its bound: 0.006 -> 1.733. But the four sector values it lands on are
1.733 / 1.610 / 1.877 / 1.622 -- still one number -- and distance covered halves,
4.9 -> 2.3. Freeing the readout does not make the policy condition on anything,
so the readout was never the binding constraint. Meta-RL is the remaining
candidate that acts upstream of it.

Measured as the spread of the REALISED q_v/q_c per sector while driving, not
from a frozen held-input probe. A policy with feedback is by construction a
function of its history, so a static probe understates it; what matters is
whether the weights it actually emits differ by sector.

Reports distance covered alongside, because a term that buys spread by driving
worse has not helped -- that is the trade the entropy term lost.
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

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, N_META, THETA_HI,  # noqa: E402
                             THETA_LO, PolicyTuner, WeightPolicy, features,
                             meta_features)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def run(seed, meta, n_ep, steps, entropy=0.0, track_name="circuit"):
    """Train one policy; return (per-sector mean ratio, distance covered)."""
    from examples.tune_online import Plant
    # The circuit, not the oval: an oval has only two distinct named sectors,
    # so a "spread across four sectors" measured there is across two by
    # construction. The circuit has all four.
    track = getattr(Track, track_name)()
    th0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    n_in = N_FEATURES + (N_META if meta else 0)
    pol = WeightPolicy(LTCCell(n_in, 12, seed=seed), th0, THETA_LO, THETA_HI,
                       seed=seed)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0, seed=seed,
                     trust_region=0.01, theta_prior=0.5, entropy=entropy)

    # Reward scale for the tanh squash, tracked online: a fixed scale either
    # saturates the input or wastes its range, and we do not know the magnitude
    # in advance.
    r_rms, covered, per_sec = 1.0, [], {k: [] for k in range(4)}

    def obs(s5, opp, tr, theta=None, r=0.0):
        f = features(track, s5, [opp], opp_speed_est=tr.speed)
        if not meta:
            return f
        return np.concatenate([f, meta_features(theta, r, tu._last_td,
                                                THETA_LO, THETA_HI, r_rms)])

    for ep in range(n_ep):
        kind = (seed + ep) % 4
        opp = Opponent(track, s0=6.0, speed=(0.0, 1.0, 2.6, 3.4)[kind], radius=0.24)
        P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
        s5 = P.reset(); m.reset(); m.set_obstacles(P.keepouts()); tu.reset()
        tr = ObstacleTracker(dt=0.05); tr.update(opp.pose()[:2])
        s_start = float(s5[4])
        th, u = tu.act(obs(s5, opp, tr), s5)
        for _ in range(steps):
            s5n, r, off, done = P.step(u)
            r_rms = 0.99 * r_rms + 0.01 * max(abs(float(r)), 1e-6)
            m.set_obstacles(P.keepouts()); tr.update(opp.pose()[:2])
            # Record what the policy emitted, and where it emitted it.
            per_sec[int(track.sector(track.wrap(float(s5[4]))))].append(
                float(np.exp(th[2]) / np.exp(th[0])))
            out = tu.learn(r, s5n, obs(s5n, opp, tr, th, r), off)
            if out[0] is None:
                break
            th, u = out
            s5 = s5n
            if off or done:
                break
        covered.append(float(s5[4]) - s_start)
    ratios = np.array([np.mean(per_sec[k]) if per_sec[k] else np.nan
                       for k in range(4)])
    return ratios, float(np.mean(covered[-4:]))


def _one(task, episodes, steps, track_name):
    """One (condition, seed) run. Module level so ProcessPoolExecutor can pickle it."""
    _name, meta, ent, seed = task
    return run(seed, meta, episodes, steps, ent, track_name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--track", default="circuit")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "meta_rl.json"))
    a = ap.parse_args(argv)

    # The (condition, seed) runs are independent, so run them in parallel. Done
    # sequentially this is 12 runs one after another on a single core, which on
    # a loaded machine is hours; the work is not CPU-starved, it is serialised.
    CONDITIONS = (("features only", False, 0.0),
                  ("+ anti-saturation", False, 0.5),
                  ("+ meta-RL feedback", True, 0.0),
                  ("+ meta-RL + anti-sat", True, 0.5))
    tasks = [(name, meta, ent, seed)
             for (name, meta, ent) in CONDITIONS for seed in range(a.seeds)]
    print(f"  {len(tasks)} runs over {a.jobs} workers", flush=True)

    out = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_one, t, a.episodes, a.steps, a.track): t for t in tasks}
        for fut in futs:
            pass
        for fut, t in futs.items():
            name, _m, _e, seed = t
            r, c = fut.result()
            out.setdefault(name, ([], []))
            out[name][0].append(r); out[name][1].append(c)
            print(f"    done {name:<24} seed {seed}  covered {c:6.1f}", flush=True)

    print()
    print(f"  {'condition':<24}{'ratio across 4 sectors':>32}"
          f"{'spread +- SE':>16}{'covered +- SE':>16}")
    res = {}
    for name, _m, _e in CONDITIONS:
        Rs, C = out[name]
        A = np.array(Rs)                      # (seeds, 4)
        # Spread PER SEED, then averaged -- not the spread of the seed-averaged
        # ratios. Averaging first washes out per-seed structure whenever seeds
        # favour different sectors, which understates a policy that conditions
        # on the sector differently from run to run, and it leaves no error bar
        # at all. With n=3 the error bar is most of the story.
        per_seed = ((np.nanmax(A, axis=1) - np.nanmin(A, axis=1))
                    / np.maximum(np.nanmean(A, axis=1), 1e-9))
        sp, sp_se = float(np.nanmean(per_seed)), float(
            np.nanstd(per_seed, ddof=1) / np.sqrt(len(per_seed)))
        cov, cov_se = float(np.mean(C)), float(np.std(C, ddof=1) / np.sqrt(len(C)))
        res[name] = dict(ratios=np.nanmean(A, axis=0).tolist(),
                         ratios_per_seed=A.tolist(),
                         spread=sp, spread_se=sp_se,
                         spread_per_seed=per_seed.tolist(),
                         covered=cov, covered_se=cov_se, covered_per_seed=C)
        print(f"  {name:<24}{str(np.round(np.nanmean(A, axis=0), 2)):>32}"
              f"{sp:>10.1%} +-{sp_se:>4.1%}{cov:>10.1f} +-{cov_se:>4.1f}")
    print()
    print("  n=3. A difference smaller than the two SEs together is not a"
          " difference.")

    print("\n  spread is across SECTORS: 0% means one weight vector everywhere,")
    print("  which is the failure this whole line of work is chasing.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
