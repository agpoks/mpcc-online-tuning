"""Generate the GIFs the docs embed, into ``docs/source/_static/anim/``.

    python scripts/make_animations.py             # all of them
    python scripts/make_animations.py --only horizon

Committed, like the figures: a docs build cannot run IPOPT. The scuderia clip
needs ``scuderia_gym_jax`` on the path and is skipped with a message if it is
not importable, so this still runs on a machine that only has this repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.learner import QLambdaTuner  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402
from mpcc_tuning.viz import (animate_behaviour, animate_run,  # noqa: E402
                             animate_tuning)

OUT = ROOT / "docs" / "source" / "_static" / "anim"


def _mpcc(track, dt=0.05, horizon=20, max_iter=60):
    return MPCC(track, model=KinematicBicycle(dt=dt), horizon=horizon,
                dt=dt, max_iter=max_iter)


def anim_horizon():
    """The controller running well, with its prediction drawn every tick."""
    from examples.tune_online import Plant
    track = Track.oval()
    plant = Plant(track, dt=0.05)
    plant.max_steps = 400
    return animate_run(_mpcc(track), plant, MPCCWeights().to_log(),
                       OUT / "mpcc_horizon.gif", steps=300, max_frames=170,
                       title="MPCC on the bicycle plant — orange is the horizon "
                             "the solver commits to, re-solved every 50 ms")


def anim_scuderia():
    """The same controller, the same weights, real fitted tyres.

    The interesting frame is the one where the prediction sails through a
    corner the car cannot take. Nothing about the controller changed; the plant
    grew slip angles.
    """
    try:
        from mpcc_tuning.plant_scuderia import ScuderiaPlant
        track = Track.oval()
        plant = ScuderiaPlant(track, model="st", dt=0.05)
    except ImportError as exc:
        print(f"    skipped: {exc}".splitlines()[0])
        return None
    plant.max_steps = 400
    return animate_run(_mpcc(track), plant, MPCCWeights().to_log(),
                       OUT / "mpcc_scuderia.gif", steps=300, max_frames=120, fps=12,
                       title="the same controller and the same weights, on "
                             "scuderia_gym_jax's fitted tyres — it does not survive "
                             "the first corner")


def _drive_behaviour(track, theta, opp_speed, steps, label):
    """One run against one opponent, recording everything the animation needs."""
    import numpy as np
    from mpcc_tuning.mpcc import MPCC
    from mpcc_tuning.model import KinematicBicycle
    from mpcc_tuning.opponents import Opponent
    from examples.tune_online import Plant

    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    opp = Opponent(track, s0=3.0, speed=opp_speed, radius=0.24)
    P = Plant(track, dt=0.05, max_steps=steps, opponents=[opp])
    s5 = P.reset(); m.reset()
    xy, op, cov_t, cov, off = [], [], [], 0.0, False
    for _ in range(steps):
        m.set_obstacles(P.keepouts())
        u = m.value(s5, theta)["u0"]
        s5, r, off, tr = P.step(u); cov += r
        xy.append((s5[0], s5[1])); op.append(tuple(opp.pose()[:2])); cov_t.append(cov)
        if off or tr:
            break
    import numpy as np
    ratio = np.full(len(xy), float(np.exp(theta[2]) / np.exp(theta[0])))
    return dict(label=label, xy=np.array(xy), opp=np.array(op), r=0.24,
                ratio=ratio, covered=cov, covered_t=np.array(cov_t), off=bool(off))


def anim_behaviour(steps=220):
    """Follow versus overtake, same controller, same opponent, side by side."""
    import numpy as np
    from mpcc_tuning.ltc import behaviour_theta
    from mpcc_tuning.mpcc import MPCCWeights

    track = Track.oval()
    t0 = MPCCWeights(q_l=200.0, r_d=1.0).to_log()
    runs = [_drive_behaviour(track, behaviour_theta("follow", "neutral", t0),
                             1.0, steps, "follow  ($q_v/q_c < 1$)"),
            _drive_behaviour(track, behaviour_theta("overtake", "aggressive", t0),
                             1.0, steps, "overtake  ($q_v/q_c > 1$)")]
    for r in runs:
        print(f"    {r['label']}: {r['covered']:.1f} m, off={r['off']}", flush=True)
    return animate_behaviour(
        track, runs, OUT / "mpcc_behaviour.gif",
        title="Two cost weights decide whether the car passes or sits behind")


def anim_static_vs_dynamic(steps=220):
    """The same posture against a moving car and a stopped one.

    Against a stopped obstacle "stay behind" is not caution, it is stopping --
    which the animation shows as a car that simply parks and never recovers.
    """
    import numpy as np
    from mpcc_tuning.ltc import behaviour_theta
    from mpcc_tuning.mpcc import MPCCWeights

    track = Track.oval()
    t0 = MPCCWeights(q_l=200.0, r_d=1.0).to_log()
    th = behaviour_theta("follow", "neutral", t0)
    runs = [_drive_behaviour(track, th, 1.0, steps, "moving car: following works"),
            _drive_behaviour(track, th, 0.0, steps, "stopped car: the same weights park")]
    for r in runs:
        print(f"    {r['label']}: {r['covered']:.1f} m", flush=True)
    return animate_behaviour(
        track, runs, OUT / "mpcc_static_vs_dynamic.gif",
        title="A stopped car is not a slow car: following it means stopping")


def anim_tuning(n_ep=26):
    """A full tuning session: the rise, then the collapse, with the weights.

    These settings are ``examples/tune_online.py``'s argparse defaults, not
    :class:`QLambdaTuner`'s constructor defaults, and the difference is not
    cosmetic. ``explore`` defaults to **0.0** on the class and 0.05 in the
    script: with no exploration the weights barely move, every episode covers
    40.9 m, and the animation shows a flat line where the results page reports
    a rise and a collapse. ``alpha`` (2e-3 against 1e-3) and ``horizon``
    (12 against 20) differ too.
    """
    from examples.tune_online import Plant
    track = Track.oval()
    mpcc = _mpcc(track, horizon=12)
    theta = MPCCWeights().to_log()
    tuner = QLambdaTuner(mpcc, len(theta), gamma=0.98, lam=0.9, alpha=2e-3,
                         explore=0.05, delta_clip=1.0, seed=0)
    episodes = []
    for ep in range(n_ep):
        plant = Plant(track, dt=0.05)
        plant.max_steps = 400
        s5 = plant.reset()
        mpcc.reset(); tuner.reset()
        u = tuner.start(theta, s5)
        traj, covered, off = [s5[:2].copy()], 0.0, False
        for _ in range(plant.max_steps):
            s5n, r, off, tr = plant.step(u)
            covered += r
            traj.append(s5n[:2].copy())
            theta, u = tuner.step(theta, s5, r, s5n, off)
            s5 = s5n
            if off or tr:
                break
        episodes.append(dict(traj=np.array(traj), theta=theta.copy(),
                             covered=covered, off=bool(off)))
        w = np.exp(theta)
        print(f"    ep {ep:3d}  covered {covered:6.1f} m"
              f"{'  OFF' if off else '     '}   "
              + "  ".join(f"{n}={v:8.3f}" for n, v in zip(
                  ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv"), w)), flush=True)
    return animate_tuning(episodes, OUT / "mpcc_tuning.gif", track=track,
                          title="online tuning on the bicycle plant: it improves the "
                                "controller, then destroys it")


ANIMS = {
    "behaviour": anim_behaviour,
    "static_dynamic": anim_static_vs_dynamic,"horizon": anim_horizon, "scuderia": anim_scuderia, "tuning": anim_tuning}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(ANIMS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for name in (a.only or sorted(ANIMS)):
        print(f"  {name} ...", flush=True)
        p = ANIMS[name]()
        if p is None:
            continue
        kb = p.stat().st_size / 1024
        total += kb
        print(f"    wrote {p.name}  {kb:.0f} KB")
    print(f"  total {total / 1024:.1f} MB")
