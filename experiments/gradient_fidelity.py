"""Is the influence approximation the reason the policy fails? Measured: no.

    python3 experiments/gradient_fidelity.py

The LTC policy runs away -- it drives ``q_v`` onto its ceiling and ``q_c``
towards zero, and the return collapses. Two explanations are available and they
call for opposite fixes:

1. the **gradient is wrong**, because the influence of the recurrent state is
   carried in the RFLO approximation (each neuron's own influence, the coupling
   *between* neurons dropped) rather than exactly;
2. the **objective is wrong**, and no gradient however exact would help.

There is already strong circumstantial evidence for (2): the *global* tuner uses
the envelope-theorem gradient, which is exact to cosine 0.99999 against finite
differences, and shows the identical runaway. A failure that reproduces under an
exact gradient is not caused by an approximate one.

But that is an argument, so this measures it, with the same protocol paper 1
used for the memoryless-versus-RTI question: replay one recorded feature
sequence, compute the policy gradient both ways along it, and compare
**direction** rather than magnitude -- for a method whose step size is a tuned
hyperparameter the direction is what has to survive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,  # noqa: E402
                             WeightPolicy, features)
from mpcc_tuning.mpcc import MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def replay(track, steps, seed):
    """A recorded feature sequence, so both estimators see identical input."""
    from examples.tune_online import Plant
    from mpcc_tuning.mpcc import MPCC
    from mpcc_tuning.model import KinematicBicycle

    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    opp = Opponent(track, s0=3.0, speed=1.0, radius=0.24)
    P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
    s5 = P.reset(); m.reset()
    th = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    feats, grads = [], []
    for _ in range(steps):
        m.set_obstacles(P.keepouts())
        feats.append(features(track, s5, [opp]))
        out = m.value(s5, th)
        grads.append(m.grad_theta(out, s5, th))
        s5, _r, off, tr = P.step(out["u0"])
        if off or tr:
            break
    return np.array(feats), np.array(grads)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "gradient_fidelity.json"))
    a = ap.parse_args(argv)

    track = Track.oval()
    feats, gQ = replay(track, a.steps, 0)
    print(f"  replayed {len(feats)} ticks\n")
    print(f"  {'seed':>4}  {'cos(RFLO, exact)':>18}  {'|RFLO|/|exact|':>15}")
    cs, rs = [], []
    for seed in range(a.seeds):
        th0 = MPCCWeights(q_c=10.0, q_l=200.0, q_v=0.5, r_d=1.0).to_log()
        pols = {}
        for kind in ("rflo", "exact"):
            cell = LTCCell(N_FEATURES, a.hidden, seed=seed)
            pols[kind] = WeightPolicy(cell, th0, THETA_LO, THETA_HI, seed=seed,
                                      influence=kind)
        acc = {k: [] for k in pols}
        for t, f in enumerate(feats):
            for k, pol in pols.items():
                pol.step(f)
                dG, dc = pol.grads(gQ[t])
                acc[k].append(np.concatenate([dG.ravel(), dc.ravel()]))
        A = np.array(acc["rflo"]).sum(0)
        B = np.array(acc["exact"]).sum(0)
        c = float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))
        r = float(np.linalg.norm(A) / (np.linalg.norm(B) + 1e-12))
        cs.append(c); rs.append(r)
        print(f"  {seed:>4}  {c:>18.5f}  {r:>15.4f}")
    cs, rs = np.array(cs), np.array(rs)
    print(f"\n  cosine  mean {cs.mean():.5f}  min {cs.min():.5f}")
    print(f"  |RFLO|/|exact|  mean {rs.mean():.4f}")
    verdict = ("the approximation is not the problem: RFLO and exact RTRL point "
               "the same way" if cs.min() > 0.99 else
               "the approximation changes the direction and may matter")
    print(f"\n  VERDICT: {verdict}")
    Path(a.out).write_text(json.dumps(
        dict(cosine_mean=float(cs.mean()), cosine_min=float(cs.min()),
             ratio_mean=float(rs.mean()), n_ticks=len(feats),
             seeds=a.seeds), indent=2) + "\n")


if __name__ == "__main__":
    main()
