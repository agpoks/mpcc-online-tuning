"""ASIF: certify by exhibiting the manoeuvre that would save you.

References
----------
Gurriet, Singletary, Reher, Ciarletta, Feron & Ames, *"Towards a Framework for
Realizable Safety Critical Control through Active Set Invariance"*, ICCPS 2018
-- the **active set invariance filter**, which is the published name for the
construction here: integrate a backup controller forward and enforce
invariance of the tube it sweeps out.

Wabersich & Zeilinger, *"A predictive safety filter for learning-based control
of constrained nonlinear dynamical systems"*, Automatica 2021
(arXiv:1812.05506) -- the MPC-shaped statement of the same idea, where the
backup is an optimisation variable rather than a fixed policy.

The idea
--------
For a proposed input :math:`u`, apply it for one step and then run a fixed
backup policy :math:`\\pi_b` for :math:`N` steps:

.. math::
    x_1 = f(x_0, u), \\qquad x_{k+1} = f(x_k, \\pi_b(x_k))

Accept :math:`u` if and only if

.. math::
    x_k \\in \\mathcal{X} \;\; \\forall k \\le N
    \\quad\\text{and}\\quad x_N \\in \\mathcal{X}_\\text{safe}

with :math:`\\mathcal{X}` the corridor and :math:`\\mathcal{X}_\\text{safe}` the
terminal set -- here "stopped, and inside the corridor".

Why the terminal set is not optional
------------------------------------
It is what makes this an induction rather than a lookahead. If :math:`x_1` is
certified then at the next step :math:`u = \\pi_b(x_1)` is *also* certified: it
continues the same backup, which stays legal and reaches
:math:`\\mathcal{X}_\\text{safe}` with a step to spare, and
:math:`\\mathcal{X}_\\text{safe}` is invariant under :math:`\\pi_b` -- a stopped
car under braking stays stopped -- so the extra step costs nothing. The filter
is therefore never empty. Drop the terminal condition and an :math:`N`-step
lookahead will approve full throttle at a wall :math:`N+1` steps away, again at
:math:`N`, and again, until every action is too late; it was feasible at every
step and crashed anyway.

The backup here is full braking with the steering turned back towards the
centreline. It does not have to be quick or comfortable, only to reach the
terminal set from anywhere the filter is willing to certify.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.filters.base import SafetyFilter
from mpcc_tuning.model import (
    A_LAT_MAX, ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, WHEELBASE,
)


class ASIF(SafetyFilter):
    """Active set invariance filter over the MPCC's continuous input.

    ``rtrrl-playground``'s version filters nine discrete actions, so
    "minimally modify subject to a backup existing" is enumerate-and-check and
    the argmin is exact. The MPCC emits continuous :math:`(\\delta, a)`, and
    this searches a structured candidate set ordered by distance from the
    proposal -- so the result is the nearest *sampled* safe input.
    :class:`~mpcc_tuning.filters.cbf_qp.CBFQP` is the version where that
    argmin is exact.

    Deceleration is tried before steering, deliberately. A filter that swerves
    can lose the car; one that brakes gives up progress, which is the currency
    the tuner is trading in and therefore an intervention it can learn from.
    """

    def __init__(self, track, dt: float = 0.05, horizon: int = 30,
                 wheelbase: float = WHEELBASE, margin: float = 0.18,
                 stop_speed: float = 0.25, credit: str = "executed",
                 assumed_grip: float = 1.0, n_decel: int = 7, n_steer: int = 9):
        super().__init__(track, dt=dt, margin=margin, credit=credit)
        self.horizon, self.wheelbase = int(horizon), float(wheelbase)
        self.assumed_grip = float(assumed_grip)
        self.stop_speed = float(stop_speed)
        self._decel = np.linspace(0.0, 1.0, int(n_decel))
        self._steer = np.linspace(-1.0, 1.0, int(n_steer))

    # -- the model ---------------------------------------------------------
    def step(self, x, y, psi, v, delta, a, h=None, grip=None):
        """One step of the **plant's** bicycle, including the yaw-rate cap.

        Two things here are load-bearing and both were bugs first.

        The cap. The MPCC's own prediction model has no lateral-acceleration
        limit, and a filter built on it certifies every input from every state
        -- a car that can turn on a dime can always save itself. That is a
        filter switched off while still reporting a 0% intervention rate.
        ``A_LAT_MAX * grip / v`` is the same quantity as ``ay_max`` in an
        acados MPCC's path constraints.

        The integration order. Position advances with the heading from *before*
        the update, then the heading advances. Doing it the other way looks
        equivalent and costs 1.4 cm per step whenever the steering is non-zero,
        which over a 30-step backup is comparable to the entire margin.
        """
        h = self.dt if h is None else h
        grip = self.assumed_grip if grip is None else grip
        delta = min(max(delta, -STEER_MAX), STEER_MAX)
        a = min(max(a, -ACCEL_MAX), ACCEL_MAX)
        v = min(max(v + (a - DRAG * v) * h, 0.0), SPEED_MAX)
        psi_dot = v / self.wheelbase * np.tan(delta)
        if v > 1e-3:
            lim = A_LAT_MAX * grip / v
            psi_dot = min(max(psi_dot, -lim), lim)
        return x + v * np.cos(psi) * h, y + v * np.sin(psi) * h, psi + psi_dot * h, v

    def backup(self, x, y, psi, v) -> float:
        """``pi_b``: steer back to the centreline. Braking is applied separately."""
        k = self.track.project(x, y)
        tgt = float(self.track.tangent_angle(k))
        d = self.lateral(x, y)
        err = np.arctan2(np.sin(tgt - psi), np.cos(tgt - psi))
        return float(np.clip(err - 1.2 * d, -STEER_MAX, STEER_MAX))

    # -- the certificate ---------------------------------------------------
    def certify(self, state5, delta: float, a: float, grip=None) -> bool:
        x, y, psi, v = (float(q) for q in state5[:4])
        x, y, psi, v = self.step(x, y, psi, v, delta, a, grip=grip)
        if not self.inside(x, y):
            return False
        for _ in range(self.horizon):
            if v <= self.stop_speed:
                return True                       # reached X_safe, legally
            steer = self.backup(x, y, psi, v)
            x, y, psi, v = self.step(x, y, psi, v, steer, -ACCEL_MAX, grip=grip)
            if not self.inside(x, y):
                return False
        return v <= self.stop_speed

    # -- the filter --------------------------------------------------------
    def __call__(self, state5, u):
        """``u = [delta, a, v_s]``. ``v_s`` is MPCC bookkeeping and passes through."""
        self.n_steps += 1
        u = np.asarray(u, dtype=float)
        delta, a = float(u[0]), float(u[1])
        if self.certify(state5, delta, a):
            return u, False
        for f in self._decel[1:]:
            a_try = (1.0 - f) * a + f * (-ACCEL_MAX)
            if self.certify(state5, delta, a_try):
                self.n_interventions += 1
                return np.array([delta, a_try, u[2]]), True
        for s in sorted(self._steer, key=lambda t: abs(t * STEER_MAX - delta)):
            d_try = float(s * STEER_MAX)
            if self.certify(state5, d_try, -ACCEL_MAX):
                self.n_interventions += 1
                return np.array([d_try, -ACCEL_MAX, u[2]]), True
        # Nothing certifiable. With a correct model this state is unreachable;
        # it means the certificate that let us in here was wrong. Brake, count it.
        self.n_no_safe_action += 1
        self.n_interventions += 1
        return np.array([delta, -ACCEL_MAX, u[2]]), True


class TubeASIF(ASIF):
    """ASIF that certifies against an *interval of models*, not one model.

    Reference -- Wabersich & Zeilinger, *"Linear model predictive safety
    certification for learning-based control"*, CDC 2018, and the robust
    predictive safety filter in the Automatica 2021 paper.

    The problem it solves
    ---------------------
    Every guarantee an :class:`ASIF` makes is a statement about the model it
    predicts with. Pick ``assumed_grip`` above the truth and the certificate is
    void -- ``rtrrl-playground``'s sweep measures the cliff precisely: at the
    worst case (0.6) it never crashes, at the *mean* (1.0) it already lets 7%
    through, and at 1.2 it crashes 71% of episodes **while intervening less
    than the correct filter did**.

    "Assume the mean" is therefore not a defensible choice, and "assume the
    worst case" is only defensible if you know it. This class asks instead for
    a *set* :math:`[g_\\mathrm{lo}, g_\\mathrm{hi}]` the true grip lies in, and
    accepts an input only if the backup keeps the car legal for **every** model
    in the set:

    .. math::
        \\forall g \\in [g_\\mathrm{lo}, g_\\mathrm{hi}] : \;
        x_k(g) \\in \\mathcal{X} \;\\forall k, \\quad x_N(g) \\in \\mathcal{X}_\\text{safe}

    Grip enters the dynamics monotonically -- less grip means a tighter yaw-rate
    cap means a wider swept path -- so the worst case for staying on the track
    is :math:`g_\\mathrm{lo}`, and one rollout at the interval's lower end is
    enough. ``n_samples`` rolls out at several points instead, which costs more
    and is what you want if that monotonicity is not obvious in your model. It
    is checked rather than assumed in ``tests/test_filters.py``.
    """

    def __init__(self, *args, grip_interval=(0.6, 1.4), n_samples: int = 1, **kw):
        kw.setdefault("assumed_grip", float(grip_interval[0]))
        super().__init__(*args, **kw)
        self.grip_interval = (float(grip_interval[0]), float(grip_interval[1]))
        self.n_samples = max(1, int(n_samples))

    @property
    def _grips(self):
        lo, hi = self.grip_interval
        return [lo] if self.n_samples == 1 else list(np.linspace(lo, hi, self.n_samples))

    def certify(self, state5, delta: float, a: float, grip=None) -> bool:
        if grip is not None:
            return ASIF.certify(self, state5, delta, a, grip=grip)
        return all(ASIF.certify(self, state5, delta, a, grip=g) for g in self._grips)
