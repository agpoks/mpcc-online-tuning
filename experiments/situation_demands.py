"""What each racing situation actually *demands* of the cost weights.

    python3 experiments/situation_demands.py --track icra_t2_raceline --jobs 6

Every other experiment here asks what the learner *emits*. This one asks what
the situation *wants*, by direct search: for each cell of a situation grid,
drive a fixed weight vector and measure it, then report the vector that wins.

That is the premise the whole project rests on and it has never been tested
directly. If one weight vector wins every cell, a constant is the right answer,
online adaptation is unnecessary, and the policy that "collapses to a constant"
was correct all along. If different cells want different vectors, the spread
between them is the prize an adaptive policy is competing for -- and an upper
bound on what any learner could win, since this search sees the answer.

The grid is the one a racing driver would name:

* **sector** -- straight, long curve, 90-degree, hairpin, from the corner's
  total heading change (curvature at a point cannot separate a 90 from a 180)
* **opponent** -- none, slower, equal, faster, by the speed ratio the ego can
  actually estimate, not by an oracle
* **corridor** -- narrow or wide, split at the track's own median half-width,
  which is why ICRA Tracks 1 and 2 matter: the same geometry at two widths

Reported per cell as the best ``q_v/q_c`` -- the ratio, because that is the
axis with a measured meaning (the behaviour boundary sits at 1) -- together
with the distance it achieved and how much better it is than the single best
constant over the whole grid. That last number is the one that decides whether
any of this is worth doing.
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

from mpcc_tuning.ltc import OPPONENT_CLASSES  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.opponents import Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: (q_c, q_v) pairs spanning the behaviour boundary at q_v/q_c = 1.
GRID = [(1.0, 0.2), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (1.0, 5.0),
        (0.3, 2.0), (0.3, 5.0), (3.0, 2.0), (3.0, 5.0)]

#: Opponent speeds, as a fraction of the ego's MEASURED pace, per class.
#:
#: Measured, not assumed. The first version of this fixed the reference at
#: 2.5 m/s while the ego actually drove at about 4, so "equal" was 60% of the
#: ego's pace and "faster" barely matched it: no opponent could block, and the
#: four opponent columns came back within 1 m of each other -- 74-80 m whether
#: the opponent was absent, slower, equal or faster. An opponent that cannot
#: block cannot make behaviour matter, which is most of why `opponent class`
#: reads as decorative in every sensitivity table here.
OPP_SPEED = {"none": None, "slower": 0.70, "equal": 0.95, "faster": 1.15}


def _sector_starts(track, n_probe=1500):
    """One start arc-length per named sector, at the deepest point of each."""
    s = np.linspace(0.0, track.length, n_probe, endpoint=False)
    sec = np.array([int(track.sector(track.wrap(x))) for x in s])
    out = {}
    for k in range(4):
        m = sec == k
        if not m.any():
            continue
        # The middle of the longest contiguous run of this sector, so the car
        # spends the episode in it rather than crossing straight out.
        idx = np.flatnonzero(m)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
        run = max(splits, key=len)
        out[k] = float(s[run[len(run) // 2]])
    return out


def one(job):
    (track_name, horizon, q_c, q_v, sector, opp_cls, wide, s0, steps,
     base_v, v0) = job
    track = getattr(Track, track_name)()
    from examples.tune_online import Plant

    opp = None
    if OPP_SPEED[opp_cls] is not None:
        opp = Opponent(track, s0=s0 + 2.5, speed=OPP_SPEED[opp_cls] * base_v,
                       radius=0.24)
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=80, max_obstacles=1 if opp else 0)
    P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp] if opp else [])
    P.reset()
    # Start the ego inside the sector under test, not at the track origin.
    #
    # Plant.reset takes an s0 and ignores it -- `center[...] if False else
    # center[0]` -- so it always starts at the origin. Writing a modified state5
    # into a local variable does nothing either: the plant integrates its own
    # x and s, so the controller would be told one position while the car was
    # at another. That produced the giveaway in the first run of this
    # experiment: cells reading -4.7 m covered, the car apparently driving
    # backwards, and every sector's four opponent cells identical because they
    # were all in fact the same opening stretch.
    pos = np.asarray(track.pos(s0)).ravel()
    # Our OWN SPEED is a situation too, and one the network already receives
    # (feature 3, v/v_max). Entering a corner slow and entering it fast are
    # different problems and may want different weights; the grid never asked.
    P.x = np.array([pos[0], pos[1], float(track.tangent_angle(s0)), float(v0)])
    P.s = float(s0)
    P.trace = [P.x.copy()]
    s5 = P.state5()
    m.reset()
    if opp:
        m.set_obstacles(P.keepouts())
    th = MPCCWeights(q_c=q_c, q_l=50.0, q_v=q_v, r_d=0.1).to_log()
    covered, off, passes = 0.0, False, 0
    behind0 = None
    contact = 0
    for _ in range(steps):
        out = m.value(s5, th)
        s5n, r, off, done = P.step(out["u0"])
        if opp:
            m.set_obstacles(P.keepouts())
            gap = float(track.wrap(s5n[4] - opp.s))
            behind = gap > track.length / 2
            if behind0 is None:
                behind0 = behind
            elif behind0 and not behind:
                passes += 1
                behind0 = behind
        covered += float(r)
        s5 = s5n
        if off or done:
            if getattr(P, "failure", None) == "collision":
                contact = 1
            break
    return (sector, opp_cls, wide, v0, q_c, q_v), covered, bool(off), passes, contact


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--speeds", type=float, nargs="*", default=[1.0, 3.0],
                    help="entry speeds to test, m/s -- our own speed is a "
                         "situation the network already sees (feature 3)")
    ap.add_argument("--base-speed", type=float, default=0.0,
                    help="0 = measure the ego's solo pace and scale "
                         "the opponents to it, which is the only way "
                         "an opponent can be made to block")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    track = getattr(Track, a.track)()
    starts = _sector_starts(track)

    base_v = a.base_speed
    if base_v <= 0.0:
        # Drive the ego alone and take its own pace as the reference, so
        # "equal" means equal to this car on this track rather than to a
        # number chosen in advance.
        from examples.tune_online import Plant
        m0 = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=a.horizon,
                  dt=0.05, max_iter=80)
        P0 = Plant(track, dt=0.05, max_steps=a.steps)
        s5 = P0.reset(); m0.reset(); vs = []
        th0 = MPCCWeights(q_c=1.0, q_l=50.0, q_v=2.0, r_d=0.1).to_log()
        for _ in range(a.steps):
            o = m0.value(s5, th0)
            s5, _r, off, tr = P0.step(o["u0"])
            vs.append(float(s5[3]))
            if off or tr:
                break
        base_v = float(np.mean(vs)) if vs else 2.5
        print(f"  ego solo pace {base_v:.2f} m/s -- opponents scaled to it "
              f"(slower {0.70 * base_v:.2f}, equal {0.95 * base_v:.2f}, "
              f"faster {1.15 * base_v:.2f})", flush=True)
    med = float(np.median([track.width(x)[0] + track.width(x)[1]
                           for x in np.linspace(0, track.length, 400)]))
    wide_of = {}
    for k, s0 in starts.items():
        w = track.width(s0)[0] + track.width(s0)[1]
        wide_of[k] = bool(w >= med)

    jobs = []
    for k, s0 in starts.items():
        for opp_cls in OPPONENT_CLASSES if False else ("none", "slower",
                                                       "equal", "faster"):
            for v0 in a.speeds:
                for q_c, q_v in GRID:
                    jobs.append((a.track, a.horizon, q_c, q_v, k, opp_cls,
                                 wide_of[k], s0, a.steps, base_v, v0))
    print(f"  {a.track}: {len(starts)} sectors present, {len(jobs)} runs "
          f"over {a.jobs} workers", flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        # as_completed, not map. map yields in SUBMISSION order, so one
        # slow first job hides every result behind it -- three runs today
        # showed nothing for an hour while their workers were finishing.
        for _f in as_completed(futs):
            key, cov, off, passes, contact = _f.result()
            res[key] = dict(covered=cov, off=off, passes=passes, contact=contact)

    # Best (q_c, q_v) per situation, and the best single constant over all.
    cells, per_cell_best = {}, {}
    for (k, opp_cls, wide, v0, q_c, q_v), v in res.items():
        cells.setdefault((k, opp_cls, wide, v0), {})[(q_c, q_v)] = v
    def score(v):
        """Metres, minus what a race actually charges for.

        Distance alone makes "attack always" optimal by construction: q_v
        weights progress and the score IS progress, so the search runs to the
        top of the grid in every cell and per-situation tuning wins nothing.
        Measured that way the headroom was +0.0%. A racing score has to price
        the things a driver trades progress against.
        """
        return (v["covered"]
                - 25.0 * float(v.get("contact", 0))
                - 10.0 * float(v["off"])
                + 5.0 * float(v.get("passes", 0)))

    for cell, d in cells.items():
        best = max(d.items(), key=lambda kv: score(kv[1]))
        per_cell_best[cell] = dict(q_c=best[0][0], q_v=best[0][1],
                                   ratio=best[0][1] / best[0][0],
                                   covered=best[1]["covered"],
                                   off=best[1]["off"], passes=best[1]["passes"])
    totals = {}
    for w in GRID:
        totals[w] = float(np.mean([score(cells[c][w]) for c in cells]))
    best_const = max(totals, key=totals.get)

    adaptive = float(np.mean([score(cells[c][(per_cell_best[c]["q_c"],
                                              per_cell_best[c]["q_v"])])
                              for c in cells]))
    constant = totals[best_const]

    print()
    print(f"  {'sector':<12}{'opponent':<9}{'corridor':<9}{'entry v':>8}"
          f"{'best q_v/q_c':>14}{'covered':>10}")
    for (k, opp_cls, wide, v0), b in sorted(per_cell_best.items()):
        print(f"  {Track.SECTOR_NAMES[k]:<12}{opp_cls:<9}"
              f"{'wide' if wide else 'narrow':<9}{v0:>7.1f}"
              f"{b['ratio']:>14.2f}{b['covered']:>9.1f}m"
              f"{'  OFF' if b['off'] else ''}")
    print()
    print(f"  best single constant over the whole grid: "
          f"q_v/q_c = {best_const[1] / best_const[0]:.2f}, "
          f"{constant:.1f} m mean")
    print(f"  best weight PER SITUATION:                "
          f"{adaptive:.1f} m mean")
    gain = 100.0 * (adaptive - constant) / max(abs(constant), 1e-9)
    print(f"  headroom for an adaptive policy:          {gain:+.1f}%")
    print()
    print("  This is an UPPER BOUND: the search sees each cell's answer, which")
    print("  no online learner does. If it is near zero, a constant is correct")
    print("  and the collapse was never the problem.")

    out = Path(a.out) if a.out else (ROOT / "benchmarks" / "results"
                                     / f"situation_demands_{a.track}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(track=a.track, horizon=a.horizon, steps=a.steps,
             grid=[list(g) for g in GRID],
             cells={f"{k}|{o}|{int(w)}|{v0}": v
                    for (k, o, w, v0), v in per_cell_best.items()},
             raw={f"{k}|{o}|{int(w)}|{v0}|{qc}|{qv}": v
                  for (k, o, w, v0, qc, qv), v in res.items()},
             best_constant=list(best_const), constant_mean=constant,
             adaptive_mean=adaptive, headroom_pct=gain), indent=2) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
