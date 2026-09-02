"""Step 2: does the tuner improve on a baseline that already drives?

    PYTHONPATH=/path/to/scuderia_gym_jax \
        python3 experiments/tuner_from_baseline.py --plant std --jobs 4

The question in the project's stated order, asked for the first time under
conditions where it means something:

1. a fixed parameterisation that simply drives -- found, on a plant with
   tyres: ``q_c=0.3, q_v=1.0, r_delta=5`` completes 5.97 laps of the oval on
   the STD model at 100% solve;
2. **switch the tuner on from there and see whether it goes faster.**

Everything before this measured a learner deviating from an operating point
that crashed, on a plant with no tyres. Both are fixed here.

Runs every track, records the per-episode trace so the learning curve can be
drawn against the track it was driven on, and reports the honest comparison:
distance per episode with the tuner against the same fixed baseline, same
seeds, same plant.
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

from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,  # noqa: E402
                             PolicyTuner, WeightPolicy, features)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights, WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: The baseline that PASSES on the tyre model. r_delta = 5 against the
#: bicycle's 0.1: fifty times more steering-rate damping, because a car that
#: can slide has to be steered gently. The bicycle grid never reached here.
BASE = dict(q_c=0.3, q_l=50.0, q_v=1.0, r_d=5.0)

TRACKS = (("oval", 12), ("circuit", 12), ("icra_t1_raceline", 40),
          ("icra_t2_raceline", 40), ("icra2025", 50))


def _plant(track, plant, steps):
    if plant == "bicycle":
        from examples.tune_online import Plant
        return Plant(track, dt=0.05, max_steps=steps)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(track, model=plant, dt=0.05)
    P.max_steps = steps
    return P


def one(job):
    track_name, horizon, plant, seed, episodes, steps, learn = job
    t = getattr(Track, track_name)()
    th0 = MPCCWeights(**BASE).to_log()
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300)

    tu = pol = None
    if learn:
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, THETA_LO,
                           THETA_HI, seed=seed)
        tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0,
                         seed=seed, trust_region=0.01)

    per_ep, trace = [], []
    for ep in range(episodes):
        P = _plant(t, plant, steps)
        s5 = P.reset()
        m.reset()
        s0 = float(s5[4])
        off = False
        if learn:
            tu.reset()
            th, u = tu.act(features(t, s5), s5)
        else:
            th = th0
        for _ in range(steps):
            if not learn:
                u = m.value(s5, th)["u0"]
            s5n, r, off, tr = P.step(u)
            trace.append((ep, float(s5n[0]), float(s5n[1]),
                          int(t.sector(t.wrap(float(s5n[4])))), *np.exp(th)))
            if learn:
                out = tu.learn(r, s5n, features(t, s5n), off)
                if out[0] is None:
                    break
                th, u = out
            s5 = s5n
            if off or tr:
                break
        per_ep.append(dict(ep=ep, laps=(float(s5[4]) - s0) / t.length,
                           off=bool(off)))
    return (track_name, plant, seed, learn), per_ep, trace


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default="std")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tracks", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results"
                    / f"tuner_from_baseline_{a.plant}.json")

    picked = [(n, h) for n, h in TRACKS
              if a.tracks is None or n in a.tracks]
    jobs = [(n, h, a.plant, s, a.episodes, a.steps, learn)
            for n, h in picked for s in range(a.seeds) for learn in (False, True)]
    print(f"  {a.plant} plant, {len(picked)} tracks, {len(jobs)} runs over "
          f"{a.jobs} workers", flush=True)
    print(f"  baseline {BASE}", flush=True)

    res, traces = {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for key, per_ep, trace in ex.map(one, jobs):
            res[key] = per_ep
            if key[3] and key[2] == 0:
                traces[key[0]] = trace
            lp = np.mean([e["laps"] for e in per_ep[-3:]])
            print(f"    {key[0]:<18s} seed {key[2]} "
                  f"{'tuner ' if key[3] else 'fixed '} last-3 laps {lp:5.2f}",
                  flush=True)

    print()
    print("%-20s %14s %14s %10s" % ("track", "fixed", "tuner", "change"))
    summary = {}
    for n, _h in picked:
        fx = [e["laps"] for k, v in res.items() if k[0] == n and not k[3]
              for e in v[-3:]]
        tn = [e["laps"] for k, v in res.items() if k[0] == n and k[3]
              for e in v[-3:]]
        if not fx or not tn:
            continue
        f_m, t_m = float(np.mean(fx)), float(np.mean(tn))
        f_s = float(np.std(fx) / max(np.sqrt(len(fx)), 1))
        t_s = float(np.std(tn) / max(np.sqrt(len(tn)), 1))
        summary[n] = dict(fixed=f_m, fixed_se=f_s, tuner=t_m, tuner_se=t_s,
                          change_pct=100.0 * (t_m - f_m) / max(f_m, 1e-9))
        print("%-20s %7.2f +-%4.2f %7.2f +-%4.2f %9.1f%%"
              % (n, f_m, f_s, t_m, t_s, summary[n]["change_pct"]))
    print()
    print("  Laps over the last three episodes. A difference smaller than the")
    print("  two standard errors together is not a difference.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        dict(plant=a.plant, baseline=BASE, weights=list(WEIGHT_NAMES),
             summary=summary,
             per_episode={f"{k[0]}|{k[2]}|{'tuner' if k[3] else 'fixed'}": v
                          for k, v in res.items()},
             traces={k: v for k, v in traces.items()}), indent=2) + "\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
