"""Is "overtake" or "stay behind" a choice of cost weights?

    python3 experiments/overtake_or_follow.py

This is the first question that can be asked at all now that
``mpcc_tuning/mpcc.py`` has a keep-out constraint. Before it, an opponent was
invisible to the controller and the two behaviours were the *same* problem, so
no policy over theta could distinguish them -- which is why the obstacle
constraint was a precondition for the behaviour-policy work rather than a
detail of it.

The setup is the smallest one that makes the choice real: the oval, an opponent
3 m ahead driving the centreline at 1.0 m/s against an ego car capable of about
3.9, and a track just wide enough to pass on. Only ``q_c`` -- how hard the MPCC
is held to the reference line -- and ``q_v`` -- how much progress is worth --
are varied. Everything else is default.

The keep-out makes "stay behind" a *feasible, finite-cost* option rather than a
crash, so the solver is genuinely choosing, and that is the point.
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

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def signed_gap(track, s_ego, s_opp):
    """Arc length from ego to opponent, in ``(-L/2, L/2]``. Positive = ahead."""
    d = (s_opp - s_ego) % track.length
    return d - track.length if d > track.length / 2 else d


def one(job):
    q_c, q_v, steps, opp_speed, s0 = job
    from examples.tune_online import Plant

    track = Track.oval()
    t0 = time.perf_counter()
    opp = Opponent(track, s0=s0, speed=opp_speed, radius=0.24)
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_obstacles=1)
    P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
    s5 = P.reset()
    m.reset()
    theta = MPCCWeights(q_c=q_c, q_v=q_v).to_log()

    cov, off, gap_min, lat_max, pass_step = 0.0, False, 1e9, 0.0, None
    for i in range(steps):
        m.set_obstacles(P.keepouts())
        u = m.value(s5, theta)["u0"]
        s5, r, off, tr = P.step(u)
        cov += r
        ox, oy, rr = opp.keepout()
        gap_min = min(gap_min, float(np.hypot(s5[0] - ox, s5[1] - oy) - rr))
        lat_max = max(lat_max, abs(float(track.lateral(s5[0], s5[1]))))
        if pass_step is None:
            g = signed_gap(track, track.project(s5[0], s5[1]), opp.s)
            # A pass: the opponent stops being ahead. Guarded by the ego
            # actually being alongside, so a whole lap gained does not read as
            # an overtake.
            if g < 0 and abs(g) < track.length / 4:
                pass_step = i
        if off or tr:
            break
    return dict(q_c=q_c, q_v=q_v, covered=float(cov), steps=i + 1, off=bool(off),
                failure=P.failure, gap_min=float(gap_min), lat_max=float(lat_max),
                pass_step=pass_step, wall_s=round(time.perf_counter() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--opp-speed", type=float, default=1.0)
    ap.add_argument("--opp-s0", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "overtake_or_follow.json"))
    args = ap.parse_args(argv)

    grid = [(qc, qv, args.steps, args.opp_speed, args.opp_s0)
            for qv in (0.5, 2.0, 10.0) for qc in (10.0, 3.0, 1.0, 0.3, 0.1)]
    n_proc = args.jobs or min(len(grid), os.cpu_count() or 1)
    print(f"  opponent {args.opp_s0} m ahead at {args.opp_speed} m/s, "
          f"{args.steps} steps, {len(grid)} runs on {n_proc} processes\n", flush=True)

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(n_proc) as pool:
        res = list(pool.imap(one, grid))

    print("  %-6s %-6s %9s %8s %9s %8s  %s"
          % ("q_v", "q_c", "covered", "gap-r", "max|lat|", "steps", "outcome"))
    for r in res:
        if r["pass_step"] is not None:
            outcome = f"PASSED at step {r['pass_step']}"
        elif r["off"]:
            outcome = f"FAILED ({r['failure']})"
        else:
            outcome = "followed"
        print("  %-6.2f %-6.2f %9.2f %8.3f %9.3f %8d  %s"
              % (r["q_v"], r["q_c"], r["covered"], r["gap_min"], r["lat_max"],
                 r["steps"], outcome))

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  wrote {p}")
    return res


if __name__ == "__main__":
    main()
