"""Step 4: opponents, sectors, corridor widths and race strategy, on tyres.

    PYTHONPATH=/path/to/scuderia_gym_jax \
        python3 experiments/race_matrix.py --jobs 5

The full situation grid, on a plant that can slide, from a baseline that
drives. Four axes, all of which the network already receives as input:

* **track** -- all five, which is also how corridor width varies without
  changing anything else: 0.75 m on the oval and circuit, 0.85 on ICRA T1,
  0.67 on T2, 1.10 on 2025. T1 and T2 are the controlled pair, the same
  geometry at two widths (curvature cross-correlation 0.874).
* **opponent** -- none, slower, equal, faster, and *defending*: an opponent
  that holds the racing line rather than driving a fixed offset, which is the
  only one that makes a pass a decision rather than a formality.
* **sector** -- recorded per tick, so the weights emitted can be split by the
  named sector the car was in rather than averaged over a lap.
* **strategy** -- fixed baseline against the online tuner.

Everything is reported per cell as it finishes, because the failure mode of
this repo's experiments has not been wrong numbers, it has been an hour of
silence hiding a run that was never going to produce any.
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

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights, WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: The baseline that passes the acceptance gate on the STD tyre model.
BASE = dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=5.0)

TRACKS = (("oval", 12), ("circuit", 12), ("icra_t1_raceline", 40),
          ("icra_t2_raceline", 40), ("icra2025", 50))

#: Opponent speed as a fraction of the ego's own measured pace, plus a
#: DEFENDING opponent, which is the case a pass has to be earned against.
OPPONENTS = ("none", "slower", "equal", "faster", "defending")


def one(job):
    (track_name, horizon, plant, opp_kind, learn, seed, episodes, steps,
     pace) = job
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.opponents import Opponent

    t = getattr(Track, track_name)()
    th0 = MPCCWeights(**BASE).to_log()
    n_obs = 0 if opp_kind == "none" else 1
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300, max_obstacles=n_obs)

    tu = None
    if learn:
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, THETA_LO,
                           THETA_HI, seed=seed)
        tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0,
                         seed=seed, trust_region=0.01)

    ratio = {"slower": 0.55, "equal": 0.90, "faster": 1.15,
             "defending": 0.80}.get(opp_kind)
    per_ep, by_sector = [], {k: [] for k in range(4)}
    for ep in range(episodes):
        from mpcc_tuning.plant_scuderia import ScuderiaPlant
        opps = []
        if ratio is not None:
            opps = [Opponent(t, s0=2.5, speed=ratio * pace, radius=0.24)]
        P = ScuderiaPlant(t, model=plant, dt=0.05)
        P.max_steps = steps
        s5 = P.reset()
        m.reset()
        if opps:
            m.set_obstacles([o.keepout() for o in opps])
        s0 = float(s5[4])
        off = False
        th = th0
        if learn:
            tu.reset()
            th, u = tu.act(features(t, s5, opps), s5)
        for _ in range(steps):
            if not learn:
                u = m.value(s5, th)["u0"]
            for o in opps:
                o.step(0.05) if hasattr(o, "step") else None
            s5n, r, off, tr = P.step(u)
            if opps:
                m.set_obstacles([o.keepout() for o in opps])
            k = int(t.sector(t.wrap(float(s5n[4]))))
            by_sector[k].append(float(np.exp(th[2]) / np.exp(th[0])))
            if learn:
                out = tu.learn(r, s5n, features(t, s5n, opps), off)
                if out[0] is None:
                    break
                th, u = out
            s5 = s5n
            if off or tr:
                break
        per_ep.append(dict(ep=ep, laps=(float(s5[4]) - s0) / t.length,
                           off=bool(off)))
    ratios = {k: (float(np.mean(v)) if v else None)
              for k, v in by_sector.items()}
    return (track_name, opp_kind, learn, seed), per_ep, ratios


def precheck(track_name, horizon, plant, min_laps=2.0):
    """Can the BASELINE finish a couple of laps here? If not, stop.

    Fail fast, on one track, before spending anything on the other four. The
    grid below is 50 cells; if the fixed baseline cannot complete two laps of
    the first track then every one of those cells is measuring a car that
    crashes, and the answer is to find out why -- not to confirm it four more
    times.

    Today this repo ran a five-track sweep, a behaviour matrix and a
    multi-opponent grid to completion before anyone checked whether the
    controller could drive at all. It could not: a third of its solves were
    infeasible.
    """
    t = getattr(Track, track_name)()
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    steps = int(min_laps * t.length / (0.05 * 1.2)) + 300
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300)
    P = ScuderiaPlant(t, model=plant, dt=0.05)
    P.max_steps = steps
    s5 = P.reset()
    m.reset()
    th = MPCCWeights(**BASE).to_log()
    s0 = float(s5[4])
    off = False
    nok = k = 0
    for _ in range(steps):
        o = m.value(s5, th)
        nok += int(bool(o["ok"]))
        k += 1
        s5, _r, off, tr = P.step(o["u0"])
        if off or tr:
            break
    laps = (float(s5[4]) - s0) / t.length
    return laps, bool(off), 100.0 * nok / max(k, 1)


def measure_pace(track_name, horizon, plant, steps=400):
    t = getattr(Track, track_name)()
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300)
    P = ScuderiaPlant(t, model=plant, dt=0.05)
    P.max_steps = steps
    s5 = P.reset(); m.reset(); vs = []
    th = MPCCWeights(**BASE).to_log()
    for _ in range(steps):
        o = m.value(s5, th)
        s5, _r, off, tr = P.step(o["u0"])
        vs.append(float(s5[3]))
        if off or tr:
            break
    return float(np.mean(vs)) if vs else 2.0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default="std")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--tracks", nargs="*", default=None)
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--min-laps", type=float, default=2.0,
                    help="the baseline must complete this many laps of the "
                         "FIRST track or the whole grid is abandoned")
    ap.add_argument("--no-precheck", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results" / f"race_matrix_{a.plant}.json")

    picked = [(n, h) for n, h in TRACKS if a.tracks is None or n in a.tracks]
    opps = a.opponents or list(OPPONENTS)

    print(f"  {a.plant} plant, baseline {BASE}", flush=True)

    # Fail fast on ONE track before paying for the rest.
    n0, h0 = picked[0]
    lp, off, ok = ((99.0, False, 100.0) if a.no_precheck
                   else precheck(n0, h0, a.plant, a.min_laps))
    print(f"  precheck on {n0}: {lp:.2f} laps, {'OFF' if off else 'clean'}, "
          f"solve {ok:.0f}%", flush=True)
    if off or lp < a.min_laps:
        print(f"\n  ABORTED. The fixed baseline cannot complete "
              f"{a.min_laps:g} laps of {n0}, so every cell of this grid would "
              f"measure a car that crashes.", flush=True)
        print("  Find the cause before running the other tracks. A solve rate "
              "below ~95% usually means the OCP is infeasible on some ticks, "
              "not that the weights are wrong.", flush=True)
        return 1
    print(f"  {'track':<20}{'lap':>8}{'half-width':>12}{'ego pace':>10}",
          flush=True)
    pace = {}
    for n, h in picked:
        t = getattr(Track, n)()
        w = float(np.median([float(t.width(x)[0])
                             for x in np.linspace(0, t.length, 200)]))
        pace[n] = measure_pace(n, h, a.plant)
        print(f"  {n:<20}{t.length:>7.1f}m{w:>11.2f}m{pace[n]:>9.2f} m/s",
              flush=True)

    jobs = [(n, h, a.plant, o, learn, s, a.episodes, a.steps, pace[n])
            for n, h in picked for o in opps
            for learn in (False, True) for s in range(a.seeds)]
    print(f"\n  {len(jobs)} cells over {a.jobs} workers, reported as they "
          f"finish\n", flush=True)

    res, ratios = {}, {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            key, per_ep, rt = f.result()
            res[key] = per_ep
            ratios[key] = rt
            done += 1
            lp = float(np.mean([e["laps"] for e in per_ep[-3:]]))
            crash = float(np.mean([e["off"] for e in per_ep]))
            print(f"  [{done:>3d}/{len(jobs)}] {key[0]:<18s} {key[1]:<10s} "
                  f"{'tuner' if key[2] else 'fixed'}  {lp:5.2f} laps  "
                  f"crash {100*crash:3.0f}%", flush=True)

    print()
    print(f"  {'track':<20}{'opponent':<11}{'fixed':>9}{'tuner':>9}{'change':>9}")
    summary = {}
    for n, _h in picked:
        for o in opps:
            fx = [e["laps"] for k, v in res.items()
                  if k[0] == n and k[1] == o and not k[2] for e in v[-3:]]
            tn = [e["laps"] for k, v in res.items()
                  if k[0] == n and k[1] == o and k[2] for e in v[-3:]]
            if not fx or not tn:
                continue
            f_m, t_m = float(np.mean(fx)), float(np.mean(tn))
            ch = 100.0 * (t_m - f_m) / max(f_m, 1e-9)
            summary[f"{n}|{o}"] = dict(fixed=f_m, tuner=t_m, change_pct=ch)
            print(f"  {n:<20}{o:<11}{f_m:>9.2f}{t_m:>9.2f}{ch:>8.1f}%")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        dict(plant=a.plant, baseline=BASE, weights=list(WEIGHT_NAMES),
             pace=pace, summary=summary,
             sector_ratio={f"{k[0]}|{k[1]}|{'tuner' if k[2] else 'fixed'}": v
                           for k, v in ratios.items()},
             per_episode={f"{k[0]}|{k[1]}|{'tuner' if k[2] else 'fixed'}|{k[3]}": v
                          for k, v in res.items()}), indent=2) + "\n")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
