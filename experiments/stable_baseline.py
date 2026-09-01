"""SUPERSEDED by experiments/oval_acceptance.py. Kept for its grid, not its verdict.

Two reasons it should not be used to decide anything.

Its selection rule ranks ``(survived, distance)``, so **a car that does not move
wins**: it reported ICRA T1's best setting as "COMPLETED" at 0.01 laps, because a
stationary car never leaves the track and therefore never fails. Any acceptance
rule has to require a MINIMUM DISTANCE before it calls anything a pass, which
``oval_acceptance.py`` does and this does not.

And every number it produced predates the soft-constraint repair, so it was
ranking weight settings on a controller whose optimisation problem was
infeasible on a third of ticks. Its verdicts -- no stable baseline on the oval,
T1 or T2 -- are all false: one fixed setting now drives all three.

Original description follows.

A stable working parameterisation per track, before any adaptation.

    python3 experiments/stable_baseline.py --jobs 6

The project's first principle, and the one it has not been following: start
from a baseline that *works* -- not the fastest, the one that finishes -- and
adapt from there by track, sector, opponent and surface.

What is anchored on instead is ``q_c=1.0, q_l=200, q_v=2.0, r_d=1.0``, used in
seven places, which covers 0.5--0.6 m on ICRA Track 1. The policy has therefore
been learning deviations from an operating point that crashes on the tracks the
work is aimed at, with its output spans measured from that same point.

This searches a small grid per track and reports the vector that survives
longest, judged in one metric and one episode length so the numbers are
comparable -- two earlier scripts disagreed by a factor of 2.5 on the same
configuration because one accumulated the step reward and the other took the
arc-length delta.

**Laps completed is the criterion, not distance.** A baseline that covers more
ground and then leaves the track is not a baseline; it is a faster crash.
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

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: (track, horizon) -- the horizon is part of the baseline, not something the
#: weight policy can adapt: 0.6 s of lookahead cannot see through a 0.7 m-radius
#: hairpin, which is 2.2 m of arc against 1.8 m of plan.
TRACKS = (("circuit", 12), ("oval", 12), ("icra_t1_raceline", 40),
          ("icra_t2_raceline", 40), ("icra2025", 50))

GRID = [(qc, ql, rd)
        for qc in (0.1, 0.3, 1.0)
        for ql in (10.0, 50.0, 200.0)
        for rd in (0.1, 1.0)]


def one(job):
    nm, horizon, qc, ql, rd, steps = job
    from examples.tune_online import Plant

    t = getattr(Track, nm)()
    th = MPCCWeights(q_c=qc, q_l=ql, q_v=2.0, r_d=rd).to_log()
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=80)
    P = Plant(t, dt=0.05, max_steps=steps)
    s5 = P.reset()
    m.reset()
    off = trunc = False
    s0 = float(s5[4])
    nok = k = 0
    for _ in range(steps):
        o = m.value(s5, th)
        nok += int(bool(o["ok"]))
        k += 1
        s5, _r, off, trunc = P.step(o["u0"])
        if off or trunc:
            break
    # ONE metric: arc length travelled, in laps. Not the step reward, which a
    # sibling script used for the same quantity and disagreed by 2.5x.
    covered = float(s5[4]) - s0
    return (nm, qc, ql, rd), covered, covered / t.length, bool(off), \
        100.0 * nok / max(k, 1), t.length


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "stable_baseline.json"))
    a = ap.parse_args(argv)

    jobs = [(nm, h, qc, ql, rd, a.steps)
            for nm, h in TRACKS for qc, ql, rd in GRID]
    print(f"  {len(jobs)} runs over {a.jobs} workers, {a.steps} steps each",
          flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for key, cov, laps, off, ok, lap in ex.map(one, jobs):
            res[key] = dict(covered=cov, laps=laps, off=off, solve_ok=ok,
                            lap=lap)
            print(f"    {key[0]:<18s} q_c={key[1]:<4.1f} q_l={key[2]:<6.1f} "
                  f"r_d={key[3]:<4.1f}  {laps:5.2f} laps "
                  f"{'OFF' if off else 'ok '} solve {ok:3.0f}%", flush=True)

    print()
    print(f"  {'track':<20}{'q_c':>6}{'q_l':>8}{'r_d':>6}{'laps':>8}"
          f"{'solve':>8}   outcome")
    best = {}
    for nm, h in TRACKS:
        cand = [(k, v) for k, v in res.items() if k[0] == nm]
        # survived first, then distance -- a faster crash is not a baseline
        k, v = max(cand, key=lambda kv: (not kv[1]["off"], kv[1]["laps"]))
        best[nm] = dict(horizon=h, q_c=k[1], q_l=k[2], r_d=k[3], **v)
        print(f"  {nm:<20}{k[1]:>6.1f}{k[2]:>8.1f}{k[3]:>6.1f}"
              f"{v['laps']:>8.2f}{v['solve_ok']:>7.0f}%   "
              f"{'left the track' if v['off'] else 'COMPLETED'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"best": {k: v for k, v in best.items()},
         "grid": [list(g) for g in GRID],
         "all": {"|".join(map(str, k)): v for k, v in res.items()}},
        indent=2) + "\n")
    print()
    print("  A track with no COMPLETED row has no stable baseline yet, and")
    print("  nothing should be learned on it until it does.")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
