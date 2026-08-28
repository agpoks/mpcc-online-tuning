"""Do *named* sectors beat curvature bins, or just cost more parameters?

    python3 experiments/per_sector_weights.py --seeds 6

``per_segment_weights.py`` schedules theta on three **quantile bins of pointwise
curvature**. That is not a sector schedule, and the difference is not cosmetic:
pointwise curvature cannot separate a 90-degree corner from a 180-degree one,
because ``kappa = 1/R`` for an arc of radius ``R`` however far it sweeps. So the
published six-seed result is a *three-bin curvature* schedule and should be
described as one.

This runs the comparison the naming implies, on ``Track.circuit()`` -- the only
track here that contains all four sector types:

``global``      one theta for the lap (the control)
``curvature3``  three quantile bins of |kappa|  (what item 1 actually measured)
``sector4``     straight / long curve / 90-degree / 180-degree, classified by
                the corner's **total heading change**

The question is not whether ``sector4`` wins. It is whether it wins by **enough
to pay for twice the parameters of** ``curvature3``, on a task where the tuner's
whole problem is that it has too much freedom. A null result here is a real
result and should be reported as one.
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

from mpcc_tuning.learner import QLambdaTuner  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

MODES = {"global": 1, "curvature3": 3, "sector4": 4}


def labeller(track, mode):
    """s -> bin index. Read from the path *ahead*, so no detector is needed."""
    if mode == "global":
        return lambda s: 0
    if mode == "curvature3":
        return lambda s: track.segment(track.wrap(s))
    return lambda s: track.sector(float(track.wrap(s)))


def run(track, mode, n_ep=26, seed=0, steps=400, preview=1.0):
    from examples.tune_online import Plant

    n_bin = MODES[mode]
    lab = labeller(track, mode)
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    th = np.tile(MPCCWeights().to_log(), (n_bin, 1))
    # One tuner per bin: the eligibility trace belongs to the weights it
    # accumulated for, and sharing it across bins credits a hairpin's gradient
    # to the straight's weights. Seeds strided so a sweep gets independent runs.
    tus = [QLambdaTuner(m, th.shape[1], gamma=0.98, lam=0.9, alpha=2e-3,
                        explore=0.05, delta_clip=1.0, seed=seed * 1000 + i)
           for i in range(n_bin)]
    rows = []
    for ep in range(n_ep):
        P = Plant(track, dt=0.05)
        P.max_steps = steps
        s5 = P.reset()
        m.reset()
        for t_ in tus:
            t_.reset()
        b = lab(s5[4] + preview)
        u = tus[b].start(th[b], s5)
        cov, off = 0.0, False
        for _ in range(steps):
            s5n, r, off, tr = P.step(u)
            cov += r
            th[b], _u = tus[b].step(th[b], s5, r, s5n, off)
            nb = lab(s5n[4] + preview)
            if nb != b:
                b = nb
                tus[b].reset()          # the trace does not cross a boundary
                u = tus[b].start(th[b], s5n)
            else:
                u = _u
            s5 = s5n
            if off or tr:
                break
        rows.append(dict(ep=ep, covered=float(cov), off=bool(off)))
    cov = np.array([r["covered"] for r in rows])
    offs = np.array([r["off"] for r in rows], dtype=bool)
    return dict(mode=mode, seed=seed, n_bin=n_bin,
                last8_mean=float(cov[-8:].mean()), last8_off=int(offs[-8:].sum()),
                best=float(cov.max()), first_off=(int(np.argmax(offs)) if offs.any() else None),
                covered=cov.round(3).tolist(), theta=np.exp(th).tolist())


def one(job):
    mode, seed, n_ep, steps = job
    t0 = time.perf_counter()
    out = run(Track.circuit(), mode, n_ep=n_ep, seed=seed, steps=steps)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=26)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"
                                         / "per_sector.json"))
    a = ap.parse_args(argv)

    jobs = [(m, s, a.episodes, a.steps) for s in range(a.seeds) for m in MODES]
    n_proc = a.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  {len(jobs)} runs on Track.circuit(), {a.episodes} episodes, "
          f"{n_proc} processes\n", flush=True)

    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(n_proc) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o)
            print(f"  done  {o['mode']:<11} seed {o['seed']}  last8 {o['last8_mean']:6.1f} m"
                  f"  {o['last8_off']}/8 off  ({o['wall_s']:.0f} s)", flush=True)

    print()
    summary = {}
    for mode in MODES:
        r = sorted([x for x in res if x["mode"] == mode], key=lambda x: x["seed"])
        m8 = np.array([x["last8_mean"] for x in r])
        o8 = np.array([x["last8_off"] for x in r])
        summary[mode] = dict(n_bin=MODES[mode], last8_mean=float(m8.mean()),
                             last8_sd=float(m8.std(ddof=1)), last8_min=float(m8.min()),
                             last8_max=float(m8.max()),
                             seeds_collapsed=int((o8 >= 4).sum()), n_seeds=len(r))
        print(f"  {mode:<11} ({MODES[mode]} theta)  {m8.mean():6.1f} m  sd {m8.std(ddof=1):4.1f}"
              f"  range {m8.min():.1f}-{m8.max():.1f}"
              f"   collapsed {int((o8 >= 4).sum())}/{len(r)}")
    # The comparison that decides it: sector4 must beat curvature3 by more than
    # the seed spread, or the extra parameters are not paid for.
    a4, a3 = summary["sector4"], summary["curvature3"]
    diff = a4["last8_mean"] - a3["last8_mean"]
    se = np.hypot(a4["last8_sd"], a3["last8_sd"]) / np.sqrt(a4["n_seeds"])
    print(f"\n  sector4 - curvature3 = {diff:+.1f} m,  {abs(diff)/max(se,1e-9):.1f} SE"
          f"  -> {'separated' if abs(diff) > 2*se else 'NOT separated'}")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(summary=summary, runs=res), indent=2) + "\n")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
