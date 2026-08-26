"""The outer loop: TD(lambda) Q-learning on the MPCC's cost weights.

Gros & Zanon's framework, with eligibility traces added -- the same traces
``rtrrl-playground`` uses for its recurrent agent, for the same reason: the
consequence of a cost weight being wrong shows up several ticks after the tick
it was wrong on, and a trace is how one scalar reaches back that far at one
number per parameter.

## The algorithm, in five lines

Per control tick, with ``theta`` the log cost weights:

    a          = argmin of the MPCC at s          -- this is pi_theta(s)
    Q(s, a)    = the MPCC's optimal value with u_0 pinned to a
    delta      = r + gamma V(s') - Q(s, a)        -- one scalar
    e         <- gamma lambda e + dQ/dtheta
    theta     <- theta + alpha delta e

The reward ``r`` is the **real** objective -- distance covered without leaving
the track -- and is deliberately not the MPCC's internal cost. If they were the
same quantity there would be nothing to learn.

## Why this is real-time

``dQ/dtheta`` is the only derivative needed, and by the envelope theorem it is
the partial derivative of the Lagrangian at the primal-dual solution: no
implicit function theorem, no differentiating through the solver, no adjoint
sweep. It falls out of a solve that was happening anyway. That is the whole
argument for why an MPC's parameters can be tuned at control rate when a neural
policy's cannot.

Deterministic policy gradient would need ``d(u*)/dtheta`` instead, which *does*
require differentiating the KKT conditions -- affordable, but a different and
larger piece of machinery. Q-learning is the cheap door, so it is the one this
spike goes through.

## What it does not do

No convergence guarantee. Q-learning with a nonlinear function approximator
(which an MPC certainly is) has none, and the traces do not add one. This is an
empirical procedure with a good structural prior, not a proof.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.model import ACCEL_MAX, STEER_MAX


class QLambdaTuner:
    """Tune ``theta`` from one TD error per control tick."""

    def __init__(self, mpcc, n_theta: int, gamma: float = 0.98, lam: float = 0.9,
                 alpha: float = 1e-3, clip: float = 1.0, delta_clip: float = 1.0,
                 theta_bounds=(-6.0, 6.0), grad_scale: float | None = None,
                 explore: float = 0.0, seed: int = 0):
        self.mpcc, self.n_theta = mpcc, int(n_theta)
        self.gamma, self.lam, self.alpha, self.clip = gamma, lam, alpha, clip
        # Clip the TD error, not just the gradient. Leaving the track pays -5
        # while an ordinary tick pays about +0.07, so the terminal transition
        # produces a delta two orders of magnitude larger than any other -- and
        # it arrives multiplied by a trace that has been accumulating for ten
        # steps. Unclipped, one crash moves every weight by more than the
        # preceding episode did, and the run never recovers. This was not a
        # hypothetical: it is what the first working version did, on episode 2.
        self.delta_clip = float(delta_clip)
        self.lo, self.hi = theta_bounds
        # The MPCC's optimal value has no natural scale -- it is a sum of
        # weighted squared errors and can be anything -- so dQ/dtheta has no
        # natural scale either, and a fixed alpha is meaningless across
        # problems. Normalising by a running RMS of the gradient makes alpha a
        # step in *relative* terms, which is the only version of it that
        # transfers.
        self.grad_scale = grad_scale
        self._rms = None
        # Exploration, in the actuators. Q-learning needs the applied action to
        # sometimes differ from the argmin, or Q is only ever evaluated where it
        # already equals V and there is nothing to compare. It costs a second
        # NLP solve on the ticks where it fires, which is why it is off by
        # default and small when on.
        self.explore = float(explore)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _explore(self, u):
        if self.explore <= 0:
            return u
        u = np.asarray(u, float).copy()
        u[:2] += self.rng.normal(0.0, self.explore, 2) * np.array([STEER_MAX, ACCEL_MAX])
        u[0] = float(np.clip(u[0], -STEER_MAX, STEER_MAX))
        u[1] = float(np.clip(u[1], -ACCEL_MAX, ACCEL_MAX))
        return u

    def reset(self) -> None:
        self.e = np.zeros(self.n_theta)
        self.prev = None
        self.stats: dict = {}

    def step(self, theta: np.ndarray, state5: np.ndarray, reward: float,
             next_state5: np.ndarray, terminated: bool):
        """One update. Returns ``(theta, action)`` for the next tick."""
        # Q(s, a) at the action we actually took, and its gradient.
        s, a, gradQ, q = self.prev if self.prev is not None else (None, None, None, None)
        v_next = 0.0
        nxt = self.mpcc.value(next_state5, theta)
        if not terminated:
            v_next = nxt["value"]

        if self.prev is not None:
            # The MPCC *minimises* cost; the RL layer *maximises* return. So the
            # value function is the negated optimal cost, V = -J*, and so is its
            # gradient: grad_theta Q = -grad_theta J*.
            #
            # Both negations are needed and it is easy to apply one and not the
            # other. The symptom is unmistakable once you know it: the tuner
            # drives every weight confidently in the wrong direction and the
            # performance decays smoothly, which reads exactly like a learning
            # rate that is too high.
            delta = reward + self.gamma * (-v_next) - (-q)
            delta = float(np.clip(delta, -self.delta_clip, self.delta_clip))
            g = -np.clip(gradQ, -1e6, 1e6)
            if self.grad_scale is None:
                self._rms = (np.abs(g) if self._rms is None
                             else 0.99 * self._rms + 0.01 * np.abs(g))
                scale = np.maximum(self._rms, 1e-8)
            else:
                scale = self.grad_scale
            gn = np.clip(g / scale, -self.clip, self.clip)
            self.e = self.gamma * self.lam * self.e + gn
            theta = np.clip(theta + self.alpha * delta * self.e, self.lo, self.hi)
            self.stats = {"delta": float(delta), "q": float(-q),
                          "grad_norm": float(np.linalg.norm(gn))}

        action = self._explore(nxt["u0"])
        qout = self.mpcc.action_value(next_state5, theta, action, v_out=nxt)
        self.prev = (next_state5, action, self.mpcc.grad_theta(qout, next_state5, theta),
                     qout["value"])
        return theta, action

    def start(self, theta: np.ndarray, state5: np.ndarray):
        """Begin an episode: clear the trace, solve once, return the first action."""
        self.e[:] = 0.0
        out = self.mpcc.value(state5, theta)
        action = self._explore(out["u0"])
        q = self.mpcc.action_value(state5, theta, action, v_out=out)
        self.prev = (state5, action, self.mpcc.grad_theta(q, state5, theta), q["value"])
        return action
