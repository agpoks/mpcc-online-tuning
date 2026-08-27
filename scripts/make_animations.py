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
from mpcc_tuning.viz import animate_run, animate_tuning  # noqa: E402

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


ANIMS = {"horizon": anim_horizon, "scuderia": anim_scuderia, "tuning": anim_tuning}

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
