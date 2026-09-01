"""Can a FIXED parameter setting drive the oval for several laps?

    python3 experiments/oval_acceptance.py --laps 3 --jobs 6

The acceptance test the project needs and has never had. The oval is the
easiest track here -- constant 0.75 m half-width, 2.46 m minimum radius,
nothing tight -- so a weight vector that cannot complete several laps of it is
not a baseline, and nothing built on top of it means anything.

The order matters and has been the wrong way round:

1. find a fixed parameterisation that drives the oval for multiple laps;
2. **only then** switch the online tuner on and ask what it adds, under
   different sectors, opponents, corridors and speeds.

Every adaptation result so far was measured before step 1 was passed. The
baseline search found no oval setting that completes even one lap out of
eighteen tried, at 62-77% solve success -- which is the other thing this
checks, because an action taken from a solve that did not converge is not the
controller's decision and no weight setting can repair it.

Reports laps completed, solve success, and IPOPT's own return status, over a
grid of weights crossed with solver iteration budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

GRID = [(qc, ql, rd)
        for qc in (0.3, 1.0, 3.0)
        for ql in (10.0, 50.0, 200.0)
        for rd in (0.1, 1.0)]


def one(job):
    qc, ql, rd, max_iter, horizon, laps, track_name = job
    from examples.tune_online import Plant

    t = getattr(Track, track_name)()
    steps = int(laps * t.length / (0.05 * 2.0)) + 400   # generous
    th = MPCCWeights(q_c=qc, q_l=ql, q_v=2.0, r_d=rd).to_log()
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=max_iter)
    P = Plant(t, dt=0.05, max_steps=steps)
    s5 = P.reset()
    m.reset()
    off = False
    s0 = float(s5[4])
    nok = k = 0
    status = Counter()
    for _ in range(steps):
        o = m.value(s5, th)
        nok += int(bool(o["ok"]))
        k += 1
        status[str(m.solver.stats().get("return_status", "?"))] += 1
        s5, _r, off, tr = P.step(o["u0"])
        if off or tr:
            break
    covered = float(s5[4]) - s0
    return (job, covered, covered / t.length, bool(off),
            100.0 * nok / max(k, 1), status.most_common(1), steps)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--laps", type=float, default=3.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--iters", type=int, nargs="*", default=[80, 300])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results"
                    / f"acceptance_{a.track}.json")

    jobs = [(qc, ql, rd, mi, a.horizon, a.laps, a.track)
            for qc, ql, rd in GRID for mi in a.iters]
    t = getattr(Track, a.track)()
    print(f"  {a.track}: {t.length:.1f} m lap, target {a.laps:g} laps, "
          f"{len(jobs)} runs over {a.jobs} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            job, cov, lp, off, ok, st, steps = r
            print(f"    q_c={job[0]:<4.1f} q_l={job[1]:<6.1f} "
                  f"r_d={job[2]:<4.1f} iter={job[3]:<4d} "
                  f"{lp:5.2f} laps {'OFF' if off else 'ok '} "
                  f"solve {ok:3.0f}%  {st[0][0] if st else '?'}", flush=True)

    good = [r for r in rows if not r[3] and r[2] >= a.laps - 0.05]
    print()
    if good:
        best = max(good, key=lambda r: (r[4], r[2]))
        job = best[0]
        print(f"  PASSES: q_c={job[0]}, q_l={job[1]}, r_d={job[2]}, "
              f"max_iter={job[3]} -> {best[2]:.2f} laps, "
              f"solve {best[4]:.0f}%")
        print("  This is a valid parameterisation. The online tuner may now be")
        print("  switched on and asked what it adds.")
    else:
        best = max(rows, key=lambda r: (not r[3], r[2]))
        job = best[0]
        print(f"  NO SETTING COMPLETES {a.laps:g} LAPS. Best: q_c={job[0]}, "
              f"q_l={job[1]}, r_d={job[2]}, max_iter={job[3]} -> "
              f"{best[2]:.2f} laps, solve {best[4]:.0f}%")
        print("  Nothing should be learned on this track until one does.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"track": a.track, "laps_target": a.laps, "horizon": a.horizon,
         "rows": [{"q_c": r[0][0], "q_l": r[0][1], "r_d": r[0][2],
                   "max_iter": r[0][3], "laps": r[2], "off": r[3],
                   "solve_ok": r[4], "status": r[5][0][0] if r[5] else "?"}
                  for r in rows]}, indent=2) + "\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
