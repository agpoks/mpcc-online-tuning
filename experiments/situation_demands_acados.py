"""Is there anything for an adaptive policy to WIN, on the stack that ships?

    PYTHONPATH=/path/to/scuderia_gym_jax python3 \
        experiments/situation_demands_acados.py

The blocking result in TODO 2z was measured on the KINEMATIC bicycle with
IPOPT: the best weight vector per situation scored 77.9 m, exactly what the
best single constant scored. If nothing is gained by varying weights, a policy
that collapses to a constant is not broken -- it is right, and online tuning is
unnecessary.

That measurement should not be assumed to carry to this stack, for a physical
reason. **A kinematic bicycle has no grip limit.** Lateral acceleration is
whatever the geometry asks for, so a corner and a straight impose the same
demands and no weight vector can do better in one than the other. The dynamic
model has tyres: a friction ellipse, combined slip, and a peak lateral force
that a corner reaches and a straight does not. Whether that makes the sectors
genuinely want different weights is an empirical question, and this asks it.

There is already one piece of evidence that situations differ on this stack,
found by hand rather than by search: **the oval's best k_v is 0.50 and ICRA
T2's is 0.85.** Two tracks, opposite preferences, both verified clean. That is
the same claim as sector-dependence, one level up.

## What is measured

For each cell of (track x sector x entry speed), every weight vector in the
grid is driven from inside that sector, and the best is kept. The prize is

    best-per-cell mean  -  best-single-constant mean

which is an UPPER BOUND on what any learner could win, since this search sees
the answer. If it is zero, no learner can beat a constant and the honest
result is to say so.

One build per track: the weights are runtime parameters, so a single generated
solver drives every cell of the grid.
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

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

OUT = ROOT / "results"
SECTOR_NAME = {0: "straight", 1: "long curve", 2: "90-degree", 3: "hairpin"}


def _sector_starts(track, n_probe=1500):
    """One start arc-length per named sector, at the middle of its longest run."""
    s = np.linspace(0.0, track.length, n_probe, endpoint=False)
    sec = np.array([int(track.sector(track.wrap(x))) for x in s])
    out = {}
    for k in sorted(set(sec.tolist())):
        m = sec == k
        # longest contiguous run of this sector, so the car spends the episode
        # in the sector under test rather than driving straight out of it
        idx = np.flatnonzero(m)
        if not idx.size:
            continue
        splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        run = max(splits, key=len)
        out[k] = float(s[run[len(run) // 2]])
    return out


def one_track(job):
    """Every grid cell for one track, on a single generated solver."""
    track_name, grid, entry_speeds, steps = job
    import time

    import casadi as ca
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.mpcc import MPCCWeights
    from mpcc_tuning.plant_scuderia import ScuderiaPlant

    track = getattr(Track, track_name)()
    st = B.start(track_name)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, name=f"sit_{track_name}")
    starts = _sector_starts(track)
    base = dict(st.weights)

    rows = []
    for sec, s0 in sorted(starts.items()):
        for v0 in entry_speeds:
            for (q_v, k_v) in grid:
                w = dict(base); w["q_v"] = q_v; w["k_v"] = k_v
                th = MPCCWeights(**w).to_log()
                P = ScuderiaPlant(track, model="std", dt=0.05)
                P.max_steps = steps
                P.reset(s0=s0, v0=v0)
                m.reset()
                covered, off = 0.0, False
                t0 = time.perf_counter()
                for _ in range(steps):
                    out = m.value(P.state_dyn(), th)
                    s5n, r, off, done = P.step(out["u0"])
                    covered += float(r) + (5.0 if off else 0.0)
                    if off or done:
                        break
                rows.append(dict(track=track_name, sector=int(sec),
                                 sector_name=SECTOR_NAME.get(int(sec),
                                                             f"s{sec}"),
                                 v0=float(v0), q_v=float(q_v), k_v=float(k_v),
                                 covered=float(covered), off=bool(off),
                                 secs=time.perf_counter() - t0))
            print("    %-18s sector %d (%s) v0=%.1f: %d vectors done"
                  % (track_name, sec, SECTOR_NAME.get(int(sec), "?"), v0,
                     len(grid)), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=list(B.TRACKS))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)

    # q_v carries behaviour, k_v carries how much grip the plan claims. Those
    # are the two the tracks are already known to disagree about.
    grid = [(qv, kv) for qv in (0.2, 0.5, 1.0, 2.0)
            for kv in (0.35, 0.50, 0.70, 0.85)]
    entry = (1.0, 2.0)
    jobs = [(t, grid, entry, a.steps) for t in a.tracks]

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for f in as_completed([ex.submit(one_track, j) for j in jobs]):
            rows.extend(f.result())

    print()
    print("  Best weight vector per cell, against the best SINGLE constant")
    print("  over the whole grid. The difference is the prize an adaptive")
    print("  policy competes for -- and an upper bound, since this search")
    print("  sees the answer.")
    print()
    summary = {}
    for t in a.tracks:
        R = [r for r in rows if r["track"] == t]
        if not R:
            continue
        cells = sorted({(r["sector"], r["v0"]) for r in R})
        # best per cell
        per_cell, best_vec = {}, {}
        for c in cells:
            sub = [r for r in R if (r["sector"], r["v0"]) == c]
            b = max(sub, key=lambda r: r["covered"])
            per_cell[c] = b["covered"]
            best_vec[c] = (b["q_v"], b["k_v"])
        # best single constant across all cells
        const = {}
        for (qv, kv) in {(r["q_v"], r["k_v"]) for r in R}:
            vals = []
            for c in cells:
                sub = [r for r in R if (r["sector"], r["v0"]) == c
                       and r["q_v"] == qv and r["k_v"] == kv]
                if sub:
                    vals.append(sub[0]["covered"])
            if len(vals) == len(cells):
                const[(qv, kv)] = float(np.mean(vals))
        bc, bcv = max(const.items(), key=lambda kv: kv[1])
        adaptive = float(np.mean([per_cell[c] for c in cells]))
        summary[t] = dict(adaptive=adaptive, constant=bcv,
                          best_constant=list(bc),
                          prize=adaptive - bcv,
                          per_cell={f"{c[0]}|{c[1]}": per_cell[c]
                                    for c in cells},
                          best_vec={f"{c[0]}|{c[1]}": list(best_vec[c])
                                    for c in cells})
        print(f"  {t}")
        print("    %-14s %6s %10s %14s" % ("sector", "v0", "best m", "best (q_v,k_v)"))
        for c in cells:
            print("    %-14s %6.1f %10.2f %14s"
                  % (SECTOR_NAME.get(c[0], f"s{c[0]}"), c[1], per_cell[c],
                     "%.2f, %.2f" % best_vec[c]))
        print("    best single constant q_v=%.2f k_v=%.2f -> %.2f m mean"
              % (bc[0], bc[1], bcv))
        print("    best PER SITUATION                     -> %.2f m mean"
              % adaptive)
        print("    prize for adapting: %+.2f m (%.1f%%)"
              % (adaptive - bcv, 100 * (adaptive - bcv) / max(bcv, 1e-9)))
        print()
    print("  A prize of zero means no learner can beat a constant here, and")
    print("  the honest conclusion is that the benchmark cannot reward")
    print("  adaptation -- not that the learner failed.")

    OUT.mkdir(exist_ok=True)
    p = OUT / "situation_demands_acados.json"
    p.write_text(json.dumps(dict(summary=summary, rows=rows,
                                 config=vars(a)), indent=1))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
