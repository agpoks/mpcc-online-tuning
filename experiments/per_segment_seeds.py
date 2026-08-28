"""Does the per-segment result survive a change of seed?

    python3 experiments/per_segment_seeds.py --seeds 6

``per_segment_weights.py`` reports 78.0 m and 0/8 off-track against a global
theta's 7.6 m and 8/8. That is **one seed**, and the failure it claims to fix
-- the tuner walking theta out of the good region -- is a stochastic one: the
only randomness in the loop is the actuator exploration noise, and whether a
run collapses at episode 5 or never is exactly the kind of thing one draw can
get wrong in either direction.

So this runs both modes over N seeds and reports the spread, not the mean
alone. The number that matters is not "per-segment is better on average" -- it
is **how many seeds each mode ends the run off-track on**, because the global
mode's failure is not a small performance loss, it is a collapse.

Each run is independent (see the seed striding in ``per_segment_weights.run``)
and the runs are farmed out across processes, one BLAS thread each.
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

from experiments.per_segment_weights import run  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


def one(job):
    """One (mode, seed) run. Top level so it is picklable."""
    mode, seed, n_ep, steps = job
    t0 = time.perf_counter()
    rows = run(Track.oval(), mode, n_ep=n_ep, seed=seed, steps=steps)
    cov = np.array([r["covered"] for r in rows])
    off = np.array([r["off"] for r in rows], dtype=bool)
    return dict(
        mode=mode, seed=seed,
        last8_mean=float(cov[-8:].mean()),
        last8_off=int(off[-8:].sum()),
        best=float(cov.max()), best_ep=int(cov.argmax()),
        first_off=(int(np.argmax(off)) if off.any() else None),
        total_off=int(off.sum()),
        covered=cov.round(3).tolist(),
        off_flags=off.tolist(),
        theta=rows[-1]["theta"],
        wall_s=round(time.perf_counter() - t0, 1),
    )


def summarise(res, mode):
    r = [x for x in res if x["mode"] == mode]
    r.sort(key=lambda x: x["seed"])
    m = np.array([x["last8_mean"] for x in r])
    o = np.array([x["last8_off"] for x in r])
    return r, m, o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=26)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=0, help="0 = one per run")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "per_segment_seeds.json"))
    args = ap.parse_args(argv)

    jobs = [(mode, s, args.episodes, args.steps)
            for s in range(args.seeds) for mode in ("global", "per_segment")]
    n_proc = args.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  {len(jobs)} runs, {args.episodes} episodes each, {n_proc} processes\n",
          flush=True)

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(n_proc) as pool:
        res = []
        for out in pool.imap_unordered(one, jobs):
            res.append(out)
            print(f"  done  {out['mode']:<12} seed {out['seed']}  "
                  f"last8 {out['last8_mean']:6.1f} m  {out['last8_off']}/8 off  "
                  f"({out['wall_s']:.0f} s)", flush=True)

    print()
    summary = {}
    for mode in ("global", "per_segment"):
        r, m, o = summarise(res, mode)
        print(f"  === {mode} ===")
        print(f"  {'seed':>4}  {'last8':>8}  {'off':>5}  {'first off':>9}  {'best':>7}")
        for x in r:
            fo = "never" if x["first_off"] is None else f"ep {x['first_off']}"
            print(f"  {x['seed']:>4}  {x['last8_mean']:8.1f}  {x['last8_off']:>3}/8"
                  f"  {fo:>9}  {x['best']:7.1f}")
        # Report the spread, and the count of seeds that ended collapsed --
        # a mean over a bimodal "78 m or 7 m" distribution describes nothing.
        collapsed = int((o >= 4).sum())
        print(f"  mean {m.mean():.1f} m  sd {m.std(ddof=1):.1f}  "
              f"min {m.min():.1f}  max {m.max():.1f}   "
              f"seeds ending collapsed (>=4/8 off): {collapsed}/{len(r)}\n")
        summary[mode] = dict(
            last8_mean=float(m.mean()), last8_sd=float(m.std(ddof=1)),
            last8_min=float(m.min()), last8_max=float(m.max()),
            per_seed=[dict(seed=x["seed"], last8_mean=x["last8_mean"],
                           last8_off=x["last8_off"], first_off=x["first_off"],
                           best=x["best"]) for x in r],
            seeds_collapsed=collapsed, n_seeds=len(r))

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(summary=summary, runs=res), indent=2) + "\n")
    print(f"  wrote {p}")
    return res


if __name__ == "__main__":
    main()
