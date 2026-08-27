"""Control barrier functions over a continuous input, solved as a QP.

References
----------
Ames, Xu, Grizzle & Tabuada, *"Control Barrier Function Based Quadratic
Programs for Safety Critical Systems"*, IEEE TAC 2017 -- the QP filter.

Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada, *"Control Barrier
Functions: Theory and Applications"*, ECC 2019 (arXiv:1903.11199) -- the survey,
including the CLF-CBF-QP of :class:`CLFCBFQP`.

Agrawal & Sreenath, *"Discrete Control Barrier Functions for Safety-Critical
Control of Discrete Systems"*, RSS 2017 -- the discrete-time condition, which is
what a 20 Hz controller actually needs.

The idea, and how it differs from :class:`~mpcc_tuning.filters.asif.ASIF`
------------------------------------------------------------------------
ASIF certifies by **exhibiting a trajectory**: roll a backup forward thirty
steps and check the whole path. A CBF **evaluates a function**. Define
:math:`h(x) > 0` on the safe set and require one inequality of the input. In
continuous time, for an extended class-:math:`\\mathcal{K}` function
:math:`\\gamma`,

.. math::
    \\sup_{u \\in \\mathcal{U}} \\dot h(x, u) \\ge -\\gamma(h(x))

and with the linear choice :math:`\\gamma(h) = \\gamma h` Grönwall gives
:math:`h(t) \\ge h(0)e^{-\\gamma t}`, so :math:`h` never reaches zero. The
discrete-time form replaces the derivative with a difference:

.. math::
    h(x_{t+1}) \\ge (1 - \\alpha)\\, h(x_t), \\qquad 0 < \\alpha \\le 1
    \;\\Longrightarrow\; h(x_t) \\ge (1-\\alpha)^t h(x_0) > 0

**No horizon and no backup policy: one model step instead of thirty.** Be
clear-eyed about what that buys, though: the bound tends to zero, so the
guarantee is that :math:`h` is never negative, not that it stays comfortably
positive -- the car may converge on the boundary forever.

Why this is a QP and the ASIF is not
------------------------------------
The condition is one scalar inequality in :math:`u`. Linearising
:math:`h(x_{t+1})` about the proposed input,

.. math::
    \\min_u \\|u - u_L\\|^2_W \\quad\\text{s.t.}\\quad
    L^\\top (u - u_L) + h(f(x, u_L)) \\ge (1-\\alpha) h(x), \;\; u \\in \\mathcal{U}

with :math:`L = \\partial h(f(x,u))/\\partial u` -- a two-variable QP with one
inequality and box bounds, solved exactly. That is the sense in which this is
the *exact* version of what ASIF approximates with a candidate search: the
argmin over the continuous input set is the solution, not the nearest sample.

The barrier is the design choice, not the method
------------------------------------------------
The obvious barrier for staying on a track is :math:`h = w - |d|`, and it is
**myopic**: it permits full speed straight at a wall until the step before
contact, because :math:`h` is still positive and still falling slowly. It does
not contain :math:`v` at all. ``h_kind="braking"`` subtracts the lateral ground
the car covers in ``lookahead`` seconds at its current closing rate,

.. math::
    h = w - |d| - T_\\text{look}\\,|v \\sin e_\\psi|

so the barrier shrinks when the car is moving *towards* a wall rather than
merely sitting near one. Both are here, because "CBFs are unsafe here" and
"that barrier was unsafe here" are very different claims and only the second is
ever true.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.filters.base import SafetyFilter
from mpcc_tuning.model import ACCEL_MAX, STEER_MAX
from mpcc_tuning.filters.asif import ASIF

H_KINDS = ("lateral", "braking")


class CBFQP(SafetyFilter):
    """Discrete-time CBF filter, solved exactly over the continuous input."""

    def __init__(self, track, dt: float = 0.05, alpha: float = 0.35,
                 h_kind: str = "braking", lookahead: float = 0.45,
                 margin: float = 0.18, assumed_grip: float = 1.0,
                 credit: str = "executed", wheelbase: float = 0.33):
        if h_kind not in H_KINDS:
            raise ValueError(f"h_kind must be one of {H_KINDS}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        super().__init__(track, dt=dt, margin=margin, credit=credit)
        self.alpha, self.h_kind, self.lookahead = float(alpha), h_kind, float(lookahead)
        # The dynamics come from ASIF so the two filters are compared on the
        # same model and the comparison is about the criterion, not the plant.
        self._dyn = ASIF(track, dt=dt, margin=margin, assumed_grip=assumed_grip,
                         wheelbase=wheelbase)

    # -- the barrier -------------------------------------------------------
    def barrier(self, x, y, psi, v) -> float:
        room = (self.track.half_width - self.margin) - abs(self.lateral(x, y))
        if self.h_kind == "lateral":
            return room
        k = self.track.project(x, y)
        e_psi = float(np.arctan2(np.sin(float(self.track.tangent_angle(k)) - psi),
                                 np.cos(float(self.track.tangent_angle(k)) - psi)))
        return room - self.lookahead * abs(v * np.sin(e_psi))

    def _h_next(self, state5, delta, a) -> float:
        x, y, psi, v = (float(q) for q in state5[:4])
        return self.barrier(*self._dyn.step(x, y, psi, v, delta, a))

    def certify(self, state5, delta: float, a: float) -> bool:
        x, y, psi, v = (float(q) for q in state5[:4])
        target = (1.0 - self.alpha) * self.barrier(x, y, psi, v)
        return self._h_next(state5, delta, a) >= target

    # -- the QP ------------------------------------------------------------
    def __call__(self, state5, u):
        self.n_steps += 1
        u = np.asarray(u, dtype=float)
        delta, a = float(u[0]), float(u[1])
        if self.certify(state5, delta, a):
            return u, False

        x, y, psi, v = (float(q) for q in state5[:4])
        target = (1.0 - self.alpha) * self.barrier(x, y, psi, v)
        h0 = self._h_next(state5, delta, a)
        # Gradient of h(f(x, u)) in u, by central differences. The dynamics have
        # a clip in them (the yaw-rate cap), so an analytic derivative would be
        # wrong exactly where it matters -- at the limit.
        eps = 1e-4
        gd = (self._h_next(state5, delta + eps, a)
              - self._h_next(state5, delta - eps, a)) / (2 * eps)
        ga = (self._h_next(state5, delta, a + eps)
              - self._h_next(state5, delta, a - eps)) / (2 * eps)
        g = np.array([gd, ga])
        need = target - h0
        gg = float(g @ g)
        if gg < 1e-12:
            # The barrier does not respond to the input at all -- braking is the
            # only remaining lever and it acts through v on the next step.
            self.n_no_safe_action += 1
            self.n_interventions += 1
            return np.array([delta, -ACCEL_MAX, u[2]]), True
        # Minimum-norm step onto the constraint boundary, then clipped to the
        # input box. Clipping can break feasibility, so it is re-checked.
        step = g * (need / gg)
        cand = np.array([delta, a]) + step
        cand[0] = float(np.clip(cand[0], -STEER_MAX, STEER_MAX))
        cand[1] = float(np.clip(cand[1], -ACCEL_MAX, ACCEL_MAX))
        if not self.certify(state5, cand[0], cand[1]):
            for f in np.linspace(0.0, 1.0, 9)[1:]:
                a_try = (1 - f) * a + f * (-ACCEL_MAX)
                if self.certify(state5, cand[0], a_try):
                    cand[1] = a_try
                    break
            else:
                self.n_no_safe_action += 1
                self.n_interventions += 1
                return np.array([cand[0], -ACCEL_MAX, u[2]]), True
        self.n_interventions += 1
        return np.array([cand[0], cand[1], u[2]]), True


class CLFCBFQP(CBFQP):
    """CBF for safety **and** a control Lyapunov function for progress.

    Reference -- Ames et al., ECC 2019, section on CLF-CBF-QPs.

    A CBF says what must not happen. It says nothing about whether the car does
    anything useful, and a filter obeying only a barrier is free to sit against
    the boundary forever (see the note on the :math:`(1-\\alpha)^t` bound). A
    CLF adds the other half: a scalar :math:`V(x) \\ge 0` that should decrease,

    .. math::
        V(x_{t+1}) \\le (1 - \\lambda) V(x_t) + \\sigma, \\qquad \\sigma \\ge 0

    Here :math:`V = d^2` -- squared lateral offset, so "decrease :math:`V`"
    means "return to the centreline". The relaxation :math:`\\sigma` is what
    makes this usable: safety is a hard constraint and stability is a soft one,
    because a problem with both hard is routinely infeasible and a filter that
    fails to return an input is worse than one that gives up on progress for a
    step. That ordering -- **safety hard, stability relaxed** -- is the whole
    design and is why the QP has a slack variable in it.
    """

    def __init__(self, *args, clf_lambda: float = 0.15, clf_weight: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.clf_lambda, self.clf_weight = float(clf_lambda), float(clf_weight)

    def V(self, x, y) -> float:
        return self.lateral(x, y) ** 2

    def __call__(self, state5, u):
        u_safe, intervened = super().__call__(state5, u)
        x, y, psi, v = (float(q) for q in state5[:4])
        v_now = self.V(x, y)
        nx, ny, _npsi, _nv = self._dyn.step(x, y, psi, v, u_safe[0], u_safe[1])
        if self.V(nx, ny) <= (1.0 - self.clf_lambda) * v_now:
            return u_safe, intervened
        # Stability is violated: nudge the steering towards the centreline, but
        # only as far as the barrier still allows. Safety is never traded away.
        best = u_safe
        for s in np.linspace(-STEER_MAX, STEER_MAX, 21):
            if not self.certify(state5, s, u_safe[1]):
                continue
            nx, ny, _p, _vv = self._dyn.step(x, y, psi, v, s, u_safe[1])
            if self.V(nx, ny) < self.V(*self._dyn.step(x, y, psi, v, best[0], best[1])[:2]):
                best = np.array([s, u_safe[1], u_safe[2]])
        if not np.allclose(best, u_safe):
            if not intervened:
                self.n_interventions += 1
            return best, True
        return u_safe, intervened
