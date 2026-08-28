"""Do the cost weights actually behave like a behaviour dial?

    python experiments/weights_as_behaviour.py

The premise behind "learn a policy over MPCC weights" is that theta *means*
something behavioural -- aggressive or conservative, use the whole track or hug
the line. If sweeping theta gives an uninterpretable jumble, the idea has
nothing to stand on. Measured, oval, 300 steps each:

    q_v      covered   mean v   max|d|   outcome
    0.02       5.9 m   0.40      0.018   crawls
    0.10      13.4 m   0.89      0.006
    0.50      46.4 m   3.13      0.308
    2.00      55.4 m   3.88      0.345
    10.0      56.5 m   3.92      0.349   saturated
    40.0      57.2 m   3.92      0.385   saturated

**q_v is a real behaviour dial.** It spans crawling to flat out, monotonically,
and then saturates above about 2 because the car hits its speed cap. So the
useful range is roughly [0.02, 2] and everything above is the same behaviour --
which is worth knowing before handing the range to a learner, because the tuner
in ``tune_online.py`` drives q_v to 30 and beyond, well inside the dead zone.

    q_c      covered   mean v   max|d|   outcome
    0.1       55.9 m   3.88      0.372
    1.0       55.7 m   3.88      0.352
    10.0      55.4 m   3.88      0.345
    100.0     16.4 m   3.70      0.631   OFF-TRACK at step 122

**q_c is not.** Holding the line ten times harder barely changes the line
(0.372 -> 0.345), and holding it a hundred times harder **drives off the
track**. A very stiff contouring penalty makes the solver fight the yaw-rate
cap rather than track anything.

## Why this matters for a learned weight policy

The attractive safety argument for putting a network on the weights rather than
on the steering is that the MPCC still enforces the constraints, so the network
can only choose among *feasible* behaviours. The q_c row shows that argument is
**false as usually stated**: there exist weight settings that leave the track.
The network's output has to be bounded to a region where the weights are
behaviour knobs and not conditioning knobs, and that region has to be
established by measurement -- like this -- rather than assumed.
"""
import sys, numpy as np
sys.path.insert(0, "/home/poxx/github/mpcc-online-tuning")
from mpcc_tuning.track import Track
from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.mpcc import MPCC, MPCCWeights
from examples.tune_online import Plant

t = Track.oval()
m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)

def drive(th, steps=300):
    P = Plant(t, dt=0.05); P.max_steps = steps
    s5 = P.reset(); m.reset()
    vs, lats, cov, off = [], [], 0.0, False
    for i in range(steps):
        u = m.value(s5, th)["u0"]
        s5, r, off, tr = P.step(u); cov += r
        vs.append(float(s5[3])); lats.append(abs(float(t.lateral(s5[0], s5[1]))))
        if off or tr: break
    return dict(cov=cov, off=off, steps=i+1,
                v=float(np.mean(vs)), vmax=float(np.max(vs)),
                lat=float(np.mean(lats)), latmax=float(np.max(lats)))

print("sweeping q_v -- the progress weight, with everything else at default\n")
print("  %-8s %8s %7s %7s %8s %8s  %s" % ("q_v","covered","mean v","max v","mean|d|","max|d|","outcome"))
for qv in (0.02, 0.1, 0.5, 2.0, 10.0, 40.0):
    w = MPCCWeights(q_v=qv); r = drive(w.to_log())
    print("  %-8.2f %8.1f %7.2f %7.2f %8.3f %8.3f  %s"
          % (qv, r["cov"], r["v"], r["vmax"], r["lat"], r["latmax"],
             "OFF at %d" % r["steps"] if r["off"] else "survived"), flush=True)

print("\nsweeping q_c -- how hard it is held to the reference line\n")
print("  %-8s %8s %7s %8s %8s  %s" % ("q_c","covered","mean v","mean|d|","max|d|","outcome"))
for qc in (0.1, 1.0, 10.0, 100.0):
    w = MPCCWeights(q_c=qc, q_v=2.0); r = drive(w.to_log())
    print("  %-8.2f %8.1f %7.2f %8.3f %8.3f  %s"
          % (qc, r["cov"], r["v"], r["lat"], r["latmax"],
             "OFF at %d" % r["steps"] if r["off"] else "survived"), flush=True)
