"""One weight set for a whole lap is a modelling choice. Is it a bad one?

    python experiments/per_segment_weights.py

A straight and a hairpin want opposite things: the straight wants progress
weighted heavily and steering barely penalised, the hairpin wants the car held
on the line and the speed down. A single theta has to average them.

This compares a global theta against one theta per curvature segment, with the
segment read from the track ahead -- so the schedule is *observed*, not
inferred, and needs no network to detect it.

There is a second, less obvious reason to expect this to help. With a global
theta, a bad update made in a hairpin corrupts the weights used on the
straight. Per-segment weights isolate that damage, which is the same mechanism
that made discarding updates work in ``event_triggered_tuning.py``.

## Result

On the oval, 26 episodes, same seed as every other experiment here:

    mode           first off-track   last 8 episodes
    global              ep 5         7.6 m,  8/8 off
    per-segment         never       78.0 m,  0/8 off

**The collapse disappears.** This is the best result on this problem so far,
against everything else tried:

    baseline (global theta)                7.6 m,  8/8 off
    event-triggered, discard              38.5 m,  6/8 off
    behind a predictive safety filter     68.1 m,  0/8 off
    per-segment weights                   78.0 m,  0/8 off

and unlike the safety filter it needs nothing bolted on -- 78.0 m is also
higher than the best single episode a global theta ever reached before
collapsing (79.1 m at episode 4, immediately before driving into a wall). The
tuner stops overshooting the good region because a bad update in a corner no
longer moves the straight's weights.

The learned weights are *not* the intuitive ones. The straight ends at
q_v = 29.5 with q_c = 0.38, and the tightest segment at q_v = 71.2 -- a higher
progress weight in the corner, not a lower one. Reading a mechanism into that
would be over-interpretation of one seed.

## Where it does not work, and this is not a scheduling failure

On the ``mixed`` track **both** modes leave the track in episode 0 and never
recover (3.0 m and 2.9 m). That track has a 1.76 m minimum radius against the
oval's 2.46 m, and the default weights do not survive a single lap of it
regardless of how they are scheduled. It is the same failure as the
``scuderia`` plant in ``docs/source/plant.md``: an initialisation outside the
feasible region, not something tuning can fix from where it starts. Do not read
the mixed rows as a comparison between the two modes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tune_online import Plant  # noqa: E402
from mpcc_tuning.learner import QLambdaTuner  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


def run(track, mode, n_ep=26, seed=0, steps=400, preview=1.0):
    """``mode`` is 'global' or 'per_segment'."""
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    n_seg = 3 if mode == "per_segment" else 1
    th = np.tile(MPCCWeights().to_log(), (n_seg, 1))
    # One tuner per segment: the eligibility trace belongs to the weights it
    # accumulated for, and sharing it across segments would credit a hairpin's
    # gradient to the straight's weights.
    tus = [QLambdaTuner(m, th.shape[1], gamma=0.98, lam=0.9, alpha=2e-3,
                        explore=0.05, delta_clip=1.0, seed=seed + i)
           for i in range(n_seg)]
    rows = []
    for ep in range(n_ep):
        P = Plant(track, dt=0.05)
        P.max_steps = steps
        s5 = P.reset()
        m.reset()
        for t_ in tus:
            t_.reset()
        seg = 0 if n_seg == 1 else track.segment(track.wrap(s5[4] + preview))
        u = tus[seg].start(th[seg], s5)
        cov, off = 0.0, False
        for _ in range(steps):
            s5n, r, off, tr = P.step(u)
            cov += r
            th[seg], _u = tus[seg].step(th[seg], s5, r, s5n, off)
            # The segment for the NEXT tick, read from the path ahead.
            nseg = 0 if n_seg == 1 else track.segment(track.wrap(s5n[4] + preview))
            if nseg != seg:
                # Entering a different segment: its own tuner and its own trace
                # take over. The trace does not cross the boundary.
                seg = nseg
                tus[seg].reset()
                u = tus[seg].start(th[seg], s5n)
            else:
                u = _u
            s5 = s5n
            if off or tr:
                break
        rows.append(dict(ep=ep, covered=float(cov), off=bool(off),
                         theta=np.exp(th).tolist()))
    return rows


def main():
    out = {}
    for tname, track in (("mixed", Track.mixed()), ("oval", Track.oval())):
        print(f"\n=== {tname} track ===")
        for mode in ("global", "per_segment"):
            rows = run(track, mode)
            last = rows[-8:]
            mean = float(np.mean([r["covered"] for r in last]))
            offs = sum(r["off"] for r in last)
            first_off = next((r["ep"] for r in rows if r["off"]), None)
            out.setdefault(tname, {})[mode] = dict(
                last8_mean=mean, last8_off=offs, first_off=first_off,
                theta=rows[-1]["theta"])
            print(f"  {mode:<12} first off-track: "
                  f"{'ep ' + str(first_off) if first_off is not None else 'never':<8} "
                  f" last 8: {mean:6.1f} m, {offs}/8 off")
        w = out[tname]["per_segment"]["theta"]
        if len(w) == 3:
            print("  final weights per segment:")
            for i, nm in enumerate(("straight  ", "long curve", "hairpin   ")):
                print(f"    {nm}  " + "  ".join(
                    f"{n}={v:8.3f}" for n, v in zip(NAMES, w[i])))
    p = ROOT / "benchmarks" / "results" / "per_segment.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
