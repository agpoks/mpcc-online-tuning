"""Do the weights actually express the behaviours we say they do?

    python3 experiments/behaviour_modes.py --seeds 3

The claim behind item 2 is that theta is a *behaviour* parameterisation: that
"stay behind and follow", "overtake when it is safe", "try to overtake anyway"
and "how aggressive is this driver" are all reachable by moving cost weights,
with the MPCC still enforcing the constraints. This measures that claim on a
**published circuit nobody here designed** -- the Red Bull Ring at F1TENTH's
1:10 scaling -- rather than on a track built to suit it.

Why ICRA Track 2 and not ``circuit``: a track can punish a bad weight only
where it is grip-limited, and T2 is the hardest available by that measure --
its slowest corner is 2.32 m/s, 29% of the 8 m/s cap, against the synthetic
circuit's 3.94 m/s, or 49%.

This previously read "why Spielberg", on the stronger claim that the synthetic
circuit could not discriminate *at all* because every weight setting landed
within 1 m of every other. That was true under a 4 m/s speed cap, and it
stopped being true when the cap was corrected to 8 m/s on the friction-ellipse
analysis: the circuit discriminates perfectly well now. The benchmark had been
chosen to compensate for a limit that was itself wrong, which is worth
recording -- and an F1 circuit at 1:10 was never the kind of track these cars
race on.

Three postures crossed with three aggression levels, one slower opponent.
Reported: distance, passes, and crashes -- **never distance alone**, because a
policy that never passes anybody and never crashes scores respectably on
progress and has not done the task.
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

from mpcc_tuning.ltc import (AGGRESSION, POSTURES, features,  # noqa: E402
                             posture_theta)
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def signed_gap(track, s_ego, s_opp):
    d = (s_opp - s_ego) % track.length
    return d - track.length if d > track.length / 2 else d


def one(job):
    posture, aggr, seed, steps, scale, kind, ego_pace, track_name = job
    from examples.tune_online import Plant

    t0 = time.perf_counter()
    # ICRA 2026 Track 2, not the Red Bull Ring: the competition circuit these
    # cars actually race, and the harder test by the criterion that matters --
    # its slowest corner is 29% of the speed cap against the synthetic
    # circuit's 49%, so there is more of the lap on which a weight setting can
    # be punished. (The Spielberg argument was that our synthetic circuit could
    # not discriminate at all; that was an artefact of a 4 m/s cap and stopped
    # being true when the cap was corrected to 8.)
    track = getattr(Track, track_name)(scale=scale) \
        if track_name != "circuit" else Track.circuit()
    theta0 = MPCCWeights(q_l=200.0, r_d=1.0).to_log()
    # horizon 40, not 12. The competition circuits have 0.7 m-radius hairpins,
    # which are 2.2 m of arc, and 0.6 s of lookahead at 3 m/s reaches 1.8 m --
    # the plan ends inside the corner. Measured on T1: 22.8 m covered at
    # horizon 12 against 125.3 m at horizon 40, same weights.
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=40, dt=0.05,
             max_iter=60, max_obstacles=1)
    # A stopped car is not a slow car, and "stay behind" is only a behaviour
    # against something that is going somewhere.
    # Scaled to the EGO'S OWN PACE on this track, not a fixed number.
    #
    # It was 1.2-1.8 m/s, chosen when these ran on a fast open circuit. On ICRA
    # T2 the ego averages about 1.5 m/s -- the corridor is tight and the car is
    # slow through it -- so the opponent was as fast as the ego, never caught,
    # and no pass was ever available. Measured consequence: all THREE postures
    # returned bit-identical numbers (30.9 / 31.9 / 26.9 m) with passes = 0.00
    # in all 54 runs. "Stay behind" and "always try" cannot differ when there
    # is nothing to try. `pass_is_available`'s own docstring records the same
    # failure being found and fixed once already, for the same reason.
    speed = 0.0 if kind == "static" else ego_pace * (0.55 + 0.10 * (seed % 3))
    # 2.0 m ahead, not 6.0.
    #
    # The postures differ only when `close` is true, and close is
    # gap_m = tanh(d/3) < 0.6, i.e. d < 2.08 m. From 6 m ahead, with the ego at
    # 2.4 m/s against an opponent at 2.0-2.7, the closing rate is at most
    # 0.4 m/s and it takes 15 s of a 20 s episode just to reach engagement
    # range. Measured consequence: all nine posture-by-aggression cells came
    # back identical (48.4 m, 0 passes, 0 posture switches) even on the wide
    # circuit -- not because the postures do nothing but because the situation
    # that separates them never arose.
    opp = Opponent(track, s0=2.0, speed=speed, radius=0.24)
    tracker = ObstacleTracker(dt=0.05)
    P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
    s5 = P.reset()
    m.reset()

    cov, off, passes, seen, gapmin, sw = 0.0, False, 0, False, 1e9, 0
    prev = None
    for i in range(steps):
        m.set_obstacles(P.keepouts())
        feat = features(track, s5, [opp])
        # Classified from observed positions, not read off the opponent.
        tracker.update(opp.pose()[:2])
        theta = posture_theta(posture, feat, aggr, theta0,
                              track=track, s=float(s5[4]), v=float(s5[3]),
                              is_dynamic=tracker.is_dynamic)
        # Count how often the posture flips, which is the chattering the
        # "decide once and hold" argument is about.
        cur = float(theta[2] - theta[0])
        if prev is not None and abs(cur - prev) > 1e-9:
            sw += 1
        prev = cur
        u = m.value(s5, theta)["u0"]
        s5, r, off, tr = P.step(u)
        cov += r
        ox, oy, rr = opp.keepout()
        gapmin = min(gapmin, float(np.hypot(s5[0] - ox, s5[1] - oy) - rr))
        g = signed_gap(track, track.project(s5[0], s5[1]), opp.s)
        if g < 0 and abs(g) < track.length / 4 and not seen:
            passes, seen = passes + 1, True
        elif g > 0.5:
            seen = False
        if off or tr:
            break
    return dict(posture=posture, aggression=aggr, kind=kind, seed=seed,
                covered=float(cov), dyn_est=bool(tracker.is_dynamic),
                passes=int(passes), off=bool(off), failure=P.failure,
                steps=i + 1, gap_min=float(gapmin), switches=int(sw),
                wall_s=round(time.perf_counter() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results"
                    / f"behaviour_modes_{a.track}.json")

    # Measure the ego's solo pace on this track before choosing opponent
    # speeds. A number fixed in advance is how the opponent ended up as fast as
    # the ego, with no pass available and every posture identical.
    from examples.tune_online import Plant
    _track = (Track.circuit() if a.track == "circuit"
              else getattr(Track, a.track)(scale=a.scale))
    _m = MPCC(_track, model=KinematicBicycle(dt=0.05), horizon=40, dt=0.05,
              max_iter=80)
    _P = Plant(_track, dt=0.05, max_steps=a.steps)
    _s5 = _P.reset(); _m.reset(); _v = []
    _th = MPCCWeights(q_c=0.3, q_l=50.0, q_v=2.0, r_d=0.1).to_log()
    for _ in range(a.steps):
        _o = _m.value(_s5, _th)
        _s5, _r, _off, _tr = _P.step(_o["u0"])
        _v.append(float(_s5[3]))
        if _off or _tr:
            break
    ego_pace = float(np.mean(_v)) if _v else 1.5

    jobs = [(p, g, s, a.steps, a.scale, k, ego_pace, a.track) for s in range(a.seeds)
            for k in ("dynamic", "static") for p in POSTURES for g in AGGRESSION]
    n_proc = a.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  ego solo pace {ego_pace:.2f} m/s -- opponents at "
          f"{0.55 * ego_pace:.2f}-{0.75 * ego_pace:.2f} m/s so a pass exists",
          flush=True)
    _w = float(np.median([_track.width(x)[0] + _track.width(x)[1]
                          for x in np.linspace(0, _track.length, 200)]))
    print(f"  {a.track}: median corridor {_w:.2f} m wide, "
          f"car 0.24 m, opponent 0.48 m across", flush=True)
    print(f"  ({a.track}, scale {a.scale}), {len(jobs)} runs, "
          f"{n_proc} processes\n",
          flush=True)

    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(n_proc) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o)
    print(f"  {'obstacle':<9}{'posture':<20}{'aggression':<12}{'covered':>9}{'sd':>6}"
          f"{'passes':>8}{'crash':>7}{'switches':>10}")
    S = {}
    for k in ("dynamic", "static"):
      for p in POSTURES:
        for g in AGGRESSION:
            r = [x for x in res if x["posture"] == p and x["aggression"] == g
                 and x["kind"] == k]
            c = np.array([x["covered"] for x in r])
            S[f"{k}/{p}/{g}"] = d = dict(
                covered=float(c.mean()), sd=float(c.std(ddof=1)) if len(r) > 1 else 0.0,
                passes=float(np.mean([x["passes"] for x in r])),
                crashes=float(np.mean([x["off"] for x in r])),
                switches=float(np.mean([x["switches"] for x in r])),
                gap_min=float(np.min([x["gap_min"] for x in r])))
            print(f"  {k:<9}{p:<20}{g:<12}{d['covered']:9.1f}{d['sd']:6.1f}"
                  f"{d['passes']:8.2f}{d['crashes']:7.0%}{d['switches']:10.1f}")
        print()
    est = {k: np.mean([x["dyn_est"] for x in res if x["kind"] == k])
           for k in ("dynamic", "static")}
    print(f"  tracker classified dynamic: {est['dynamic']:.0%} of dynamic runs, "
          f"{est['static']:.0%} of static runs")
    print(f"  closest approach to the opponent over every run: "
          f"{min(x['gap_min'] for x in res):+.3f} m outside its radius")
    collisions = sum(x["failure"] == "collision" for x in res)
    print(f"  collisions: {collisions}/{len(res)}   (the keep-out is a soft constraint)")
    q = Path(a.out)
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps(dict(summary=S, runs=res), indent=2) + "\n")
    print(f"  wrote {q}")


if __name__ == "__main__":
    main()
