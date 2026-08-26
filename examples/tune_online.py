"""Start the MPCC with deliberately bad cost weights, and let it fix itself while driving.

    python examples/tune_online.py
    python examples/tune_online.py --episodes 30 --alpha 3e-3 --grip 0.7

Nothing is pre-trained, there is no dataset, and the tuner sees one scalar per
control tick. The MPCC is the policy and the critic at the same time; all that
is learned is six numbers.
"""

# %% [markdown]
# # Tuning an MPCC while it drives
#
# The MPCC starts with deliberately bad cost weights — far too much lag
# penalty, almost no reward for progress, so it crawls — and has to find better
# ones from driving. Nothing is pre-trained, there is no dataset, and the tuner
# sees one scalar per control tick. All that is learned is six numbers.
#
# The controller is unchanged: same solver, same constraints, same guarantees.
# Only its cost weights move.

# %%
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.learner import QLambdaTuner
from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.mpcc import MPCC, MPCCWeights
from mpcc_tuning.track import Track


# %% [markdown]
# ## The plant
#
# The car and the track — the thing the controller is wrong about. It has a
# tyre grip limit (a cap on yaw rate at `A_LAT_MAX·grip/v`) that the MPCC does
# not model, and the reward is the **real** objective: metres of track covered,
# with leaving it treated as the failure it is. That is deliberately not the
# MPCC's internal cost. If they were the same quantity there would be nothing
# to learn.

# %%
class Plant:
    """The car and the track: the thing the controller is wrong about."""

    def __init__(self, track, grip: float = 1.0, dt: float = 0.05, max_steps: int = 600):
        self.track, self.dt, self.max_steps = track, dt, max_steps
        self.model = KinematicBicycle(dt=dt, grip=grip)
        self.margin = track.half_width - 0.12

    def reset(self, s0: float = 0.0):
        p = self.track.center[int(s0 / self.ds_of(s0)) % len(self.track.center)] \
            if False else self.track.center[0]
        nxt = self.track.center[1]
        psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
        self.x = np.array([p[0], p[1], psi, 1.0])
        self.s = 0.0
        self.t = 0
        self.trace = [self.x.copy()]
        return self.state5()

    def ds_of(self, _s):  # pragma: no cover - kept for the placeholder above
        return self.track.ds

    def state5(self) -> np.ndarray:
        return np.array([*self.x, self.s])

    def step(self, u):
        """Apply ``[delta, a, v_s]``; the plant ignores ``v_s`` -- it is the MPCC's own bookkeeping."""
        prev = self.track.project(self.x[0], self.x[1])
        self.x = self.model.step(self.x, np.asarray(u, float)[:2])
        now = self.track.project(self.x[0], self.x[1])
        d = (now - prev) % self.track.length
        progress = d - self.track.length if d > self.track.length / 2 else d
        self.s += float(np.asarray(u, float)[2]) * self.dt
        self.t += 1
        self.trace.append(self.x.copy())
        lateral = self.track.lateral(self.x[0], self.x[1])
        off = abs(lateral) > self.margin
        # The real objective, and deliberately not the MPCC's cost: metres of
        # track covered, with leaving it treated as the failure it is.
        reward = progress - (5.0 if off else 0.0)
        return self.state5(), reward, off, self.t >= self.max_steps


# %% [markdown]
# ## The loop
#
# Per tick: solve, apply, observe one reward, update six weights. No buffer, no
# batch, no episode boundary to wait for.

# %%
def run(args):
    track = Track.oval(half_width=args.half_width)
    plant = Plant(track, grip=args.grip, dt=args.dt, max_steps=args.steps)
    mpcc = MPCC(track, model=KinematicBicycle(dt=args.mpc_dt), horizon=args.horizon,
                dt=args.mpc_dt, max_iter=args.max_iter)
    theta = MPCCWeights(**{k: v for k, v in
                           (("q_c", args.q_c), ("q_l", args.q_l), ("q_v", args.q_v),
                            ("r_d", args.r_d), ("r_a", args.r_a))}).to_log()
    tuner = QLambdaTuner(mpcc, len(theta), gamma=args.gamma, lam=args.lam,
                         alpha=args.alpha, explore=args.explore, seed=args.seed,
                         delta_clip=args.delta_clip)

    print(f"  start:  {MPCCWeights.from_log(theta)}")
    history = []
    for ep in range(args.episodes):
        s5 = plant.reset()
        mpcc.reset()
        tuner.reset()
        u = tuner.start(theta, s5)
        total, t0 = 0.0, time.perf_counter()
        for _ in range(args.steps):
            s5n, r, off, done = plant.step(u)
            total += r
            if not args.frozen:
                theta, u = tuner.step(theta, s5, r, s5n, off)
            else:
                u = mpcc.value(s5n, theta)["u0"]
            s5 = s5n
            if off or done:
                break
        dt_ms = (time.perf_counter() - t0) / max(plant.t, 1) * 1e3
        history.append(dict(ep=ep, covered=total, steps=plant.t, off=off,
                            theta=theta.copy(), ms=dt_ms))
        print(f"  ep {ep:3d}  covered {total:7.2f} m  steps {plant.t:4d}"
              f"{'  OFF-TRACK' if off else '':<12}  {dt_ms:5.0f} ms/tick"
              f"   {MPCCWeights.from_log(theta)}")
    return history, track


# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--steps", type=int, default=400, help="plant steps per episode")
    ap.add_argument("--dt", type=float, default=0.05, help="plant/control period")
    ap.add_argument("--mpc-dt", type=float, default=0.15, help="MPCC shooting interval")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--grip", type=float, default=1.0, help="plant grip; the MPCC never models it")
    ap.add_argument("--half-width", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=2e-3)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--frozen", action="store_true", help="do not tune -- the control condition")
    ap.add_argument("--explore", type=float, default=0.05,
                    help="actuator exploration, as a fraction of full scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-clip", type=float, default=8.0,
                    help="clip the TD error. Too small and the crash signal is crushed: "
                         "leaving the track costs -5 plus the whole discounted future, "
                         "and clipping that to 1 teaches the tuner that crashes are cheap")
    # Deliberately bad starting weights: far too much lag penalty and far too
    # little reward for progress, which makes the MPCC crawl.
    ap.add_argument("--q-c", type=float, default=10.0)
    ap.add_argument("--q-l", type=float, default=200.0)
    ap.add_argument("--q-v", type=float, default=0.05)
    ap.add_argument("--r-d", type=float, default=1.0)
    ap.add_argument("--r-a", type=float, default=0.01)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    history, track = run(args)
    covered = np.array([h["covered"] for h in history])
    offs = np.array([h["off"] for h in history])
    k = max(len(covered) // 3, 1)
    print(f"\n  first {k} episodes: {covered[:k].mean():7.2f} m   off-track {offs[:k].mean():.0%}")
    print(f"  last  {k} episodes: {covered[-k:].mean():7.2f} m   off-track {offs[-k:].mean():.0%}")
    best = int(np.argmax(covered))
    print(f"  best episode {best}: {covered[best]:7.2f} m\n"
          f"      {MPCCWeights.from_log(history[best]['theta'])}")
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot(history, args.plot)
    return history


# %% [markdown]
# ## What to look for
#
# **The weights should move the way you would move them by hand** — `q_v` up,
# `q_l` down — and coverage should rise with them. That part works.
#
# **Then watch for the collapse.** Once performance saturates the TD error stays
# slightly positive, so `q_v` keeps climbing and `q_c` keeps falling long after
# either helps, until the MPCC rides the constraint boundary and the unmodelled
# tyre limit puts it off the track. There is no stopping criterion. That is the
# open problem, and it is the honest result of this spike.

# %%
def plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpcc_tuning.mpcc import WEIGHT_NAMES

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot([h["covered"] for h in history], "-o", ms=3)
    ax1.set_ylabel("metres covered")
    ax1.set_title("MPCC cost weights tuned online, one TD error per control tick")
    ax1.grid(alpha=0.3)
    th = np.array([h["theta"] for h in history])
    for i, name in enumerate(WEIGHT_NAMES):
        ax2.plot(np.exp(th[:, i]), label=name)
    ax2.set_yscale("log")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("weight")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
