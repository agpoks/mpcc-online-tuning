"""Event-triggered tuning: move theta only when the car is in trouble.

    python experiments/event_triggered_tuning.py

An integrate-and-fire potential is driven by *measurable badness* -- distance
off the centreline, and whether the tyre limit is saturating -- and the tuner
acts only when it fires. No network is involved: this is one leaky accumulator
and a threshold, on quantities the plant already reports.

The motivation is a measurement, not an analogy. The tuner drifts hardest when
nothing is wrong: most of the damage happens in episode 0, where q_v moves by a
factor of 222 while the car is driving perfectly well.

## Two ways to do it, and they are not close

``defer``    accumulate the proposed moves and apply the sum when the event
             fires. Feels gentler. **It is worse**: the total movement of theta
             is unchanged -- the same per-tick moves, summed -- so all that
             changes is granularity, and coarser steps are less stable. It
             collapses two episodes EARLIER than updating every tick.

``discard``  throw the proposed move away unless something is wrong. A biased
             estimator: it learns only from trouble. It is also the only one
             that bounds anything, because a move made during good driving
             never lands at all.

Measured, 26 episodes, bicycle plant, identical seed:

    mode        collapses at   last 8 episodes    updates/ep
    every tick     ep 5        7.6 m,  8/8 off       400
    defer          ep 3        7.4 m,  8/8 off       ~55
    discard        ep 18      38.5 m,  6/8 off       ~60

Discarding delays the collapse by 13 episodes and gives 5x the final distance.
It does not prevent it: by episode 24 the car is off the track again. The drift
is slowed, not stopped, and that is the honest headline.

The mechanism is visible in the weights. At the end of episode 0, updating every
tick has already taken q_v from 0.05 to 11.1 and q_c from 10 to 2.5; discarding
leaves q_v at 1.7 and q_c at 6.8. Suppressing the updates made while the car is
driving well is exactly what the difference consists of.
"""
import sys, numpy as np
sys.path.insert(0, "/home/poxx/github/mpcc-online-tuning")
from mpcc_tuning.track import Track
from mpcc_tuning.model import KinematicBicycle, A_LAT_MAX, WHEELBASE
from mpcc_tuning.mpcc import MPCC, MPCCWeights
from mpcc_tuning.learner import QLambdaTuner
from examples.tune_online import Plant

t = Track.oval()
NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


def badness(s5, u, half):
    """A scalar 'something is wrong': off the line, or at the tyre limit."""
    lat = abs(float(t.lateral(s5[0], s5[1]))) / half
    v, delta = float(s5[3]), float(u[0])
    if v > 1e-3:
        want = abs(v / WHEELBASE * np.tan(delta))
        slip = max(0.0, want - A_LAT_MAX / v) / max(A_LAT_MAX / v, 1e-6)
    else:
        slip = 0.0
    return lat + slip


def run(mode, n_ep=26, thr=8.0, seed=0):
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    th = MPCCWeights().to_log()
    tu = QLambdaTuner(m, len(th), gamma=0.98, lam=0.9, alpha=2e-3,
                      explore=0.05, delta_clip=1.0, seed=seed)
    half = t.half_width
    rows = []
    for ep in range(n_ep):
        P = Plant(t, dt=0.05); P.max_steps = 400
        s5 = P.reset(); m.reset(); tu.reset()
        u = tu.start(th, s5)
        cov, off, pot, fires, held = 0.0, False, 0.0, 0, th.copy()
        for _ in range(P.max_steps):
            s5n, r, off, tr = P.step(u); cov += r
            th_new, u = tu.step(th, s5, r, s5n, off)
            if mode == "always":
                th = th_new
                fires += 1
            elif mode == "defer":
                # accumulate the proposed move, apply it only on an event.
                # Total movement of theta is unchanged -- only the granularity.
                held = held + (th_new - th)
                pot += badness(s5n, u, half)
                if pot >= thr or off:
                    th = np.clip(held, tu.lo, tu.hi)
                    held = th.copy(); pot = 0.0; fires += 1
            else:
                # DISCARD: throw the proposed move away unless something is
                # wrong. This is "only tune when in trouble" literally -- a
                # biased estimator, and the version the idea actually describes.
                pot += badness(s5n, u, half)
                if pot >= thr or off:
                    th = np.clip(th_new, tu.lo, tu.hi)
                    pot = 0.0; fires += 1
            s5 = s5n
            if off or tr:
                break
        rows.append((ep, cov, off, np.exp(th).copy(), fires))
    return rows


for mode in ("defer", "discard"):
    rows = run(mode)
    tag = {"always": "every tick", "defer": "deferred (accumulate)",
           "discard": "discard unless in trouble"}[mode]
    print(f"=== {tag} ===", flush=True)
    for ep, cov, off, w, fires in rows:
        if ep < 7 or ep % 6 == 0 or ep == len(rows) - 1:
            print("  ep %2d  covered %6.1f m %-4s  updates %3d  q_c=%7.3f q_v=%7.3f"
                  % (ep, cov, "OFF" if off else "", fires, w[0], w[2]), flush=True)
    last = rows[-8:]
    print("  last 8: mean %6.1f m, off-track %d/8\n"
          % (np.mean([r[1] for r in last]), sum(r[2] for r in last)), flush=True)
