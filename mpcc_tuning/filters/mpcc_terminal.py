"""The controller's own OCP, with a terminal set. The filter for free.

Reference -- Wabersich & Zeilinger, Automatica 2021. This is the predictive
safety filter in its original form: an optimisation problem whose *feasibility*
is the certificate.

The idea
--------
Every other filter here bolts a second model, a second horizon and a second set
of constraints onto a controller that already has all three. An MPCC is already
a constrained trajectory-optimisation problem over a prediction horizon. Add one
thing -- a terminal constraint saying the predicted trajectory can come to a
**stop inside the corridor** -- and solving it *is* the safety certificate:

.. math::
    \\min_{u_{0:N-1}} \; \\|u_0 - u_L\\|^2
    \\quad\\text{s.t.}\\quad
    x_{k+1} = f(x_k, u_k), \;\;
    |e_c(x_k)| \\le w, \;\;
    u_k \\in \\mathcal{U}, \;\;
    v_N \\le v_\\text{stop}

If this is feasible, a stop exists and :math:`u_0` is safe. The objective is
*minimum modification*, so the returned input is the closest one to what the
controller asked for -- which is the exact continuous argmin, not a sampled
approximation, and not a linearisation.

Why this is attractive, and where the tension is
------------------------------------------------
Attractive: the model, the horizon, the corridor and the solver all already
exist and are already tuned. On an acados controller the safety OCP is the same
generated solver with one extra terminal constraint and a different cost, so it
inherits the real-time guarantees.

The tension is **soft constraints**. Production MPCC formulations soften the
path constraints with slacks so the QP always has a solution -- that is
correct for a controller, because returning a slightly-infeasible plan beats
returning nothing. But a filter's whole output is the *feasibility bit*, and a
problem that is always feasible carries no information. So for the safety OCP
the corridor and the terminal constraint have to be **hard**, with slack left
only on comfort constraints, and the failure of the solve is the signal rather
than a fault.

The second tension is real-time: this costs an extra NLP solve per tick, on top
of the controller's. A :class:`~mpcc_tuning.filters.asif.ASIF` rollout is
microseconds. Whether that trade is affordable depends entirely on the solver,
which is why both are here.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from mpcc_tuning.filters.base import SafetyFilter
from mpcc_tuning.model import ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, WHEELBASE, A_LAT_MAX


class MPCCSafetyFilter(SafetyFilter):
    """Minimum-modification safety filter posed as an NLP with a terminal set."""

    def __init__(self, track, dt: float = 0.05, horizon: int = 25,
                 margin: float = 0.18, stop_speed: float = 0.25,
                 assumed_grip: float = 1.0, credit: str = "executed",
                 max_iter: int = 60, wheelbase: float = WHEELBASE):
        super().__init__(track, dt=dt, margin=margin, credit=credit)
        self.N, self.stop_speed = int(horizon), float(stop_speed)
        self.assumed_grip, self.wheelbase = float(assumed_grip), float(wheelbase)
        self._build(max_iter)
        self._w0 = None

    def _f(self, x, u):
        """Bicycle with the yaw-rate cap, symbolically.

        The cap is a ``min``/``max`` pair rather than a clip, because CasADi
        needs it differentiable-ish for IPOPT; ``fmin``/``fmax`` are the
        standard smooth-enough encoding and are what an acados ``ay_max``
        path constraint would express as an inequality instead.
        """
        px, py, psi, v = x[0], x[1], x[2], x[3]
        delta, a = u[0], u[1]
        v2 = ca.fmin(ca.fmax(v + (a - DRAG * v) * self.dt, 0.0), SPEED_MAX)
        psi_dot = v2 / self.wheelbase * ca.tan(delta)
        lim = A_LAT_MAX * self.assumed_grip / ca.fmax(v2, 1e-3)
        psi_dot = ca.fmin(ca.fmax(psi_dot, -lim), lim)
        return ca.vertcat(px + v2 * ca.cos(psi) * self.dt,
                          py + v2 * ca.sin(psi) * self.dt,
                          psi + psi_dot * self.dt, v2)

    def _build(self, max_iter):
        N = self.N
        # ``s`` is a state with its own input, exactly as in the MPCC. The first
        # version propagated it open-loop as ``s += v*dt``, which is only right
        # if the car tracks the path perfectly -- precisely the assumption a
        # safety filter may not make. The reference then drifted away from the
        # car, ``e_c`` was computed against the wrong point, and the corridor
        # constraint became infeasible: 363 of 400 solves failed and the filter
        # overrode every input.
        X = ca.MX.sym("X", 5, N + 1)        # [x, y, psi, v, s]
        U = ca.MX.sym("U", 3, N)            # [delta, a, v_s]
        x0 = ca.MX.sym("x0", 5)
        uL = ca.MX.sym("uL", 2)
        p = ca.vertcat(x0, uL)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Minimum modification of the FIRST input only. The rest of the horizon
        # is free -- it is a witness that a stop exists, not a plan to execute.
        # Minimum modification of the first two inputs only; v_s is bookkeeping.
        J = ca.sumsqr(U[0:2, 0] - uL)
        g, lbg, ubg = [X[:, 0] - x0], [0.0] * 5, [0.0] * 5
        for k in range(N):
            nxt = ca.vertcat(self._f(X[0:4, k], U[0:2, k]),
                             X[4, k] + U[2, k] * self.dt)
            g.append(X[:, k + 1] - nxt)
            lbg += [0.0] * 5
            ubg += [0.0] * 5
            e_c, e_l = self.track.errors(X[0, k], X[1, k], X[4, k])
            g.append(e_c)                       # hard corridor -- see the docstring
            lbg.append(-(self.track.half_width - self.margin))
            ubg.append(+(self.track.half_width - self.margin))
            # Keep the reference point near the car -- without this the solver
            # can satisfy the corridor by running s away to somewhere the car
            # is not. The bound has to be generous: at +/-0.5 m it was itself
            # the binding constraint and made 53 of 100 solves infeasible, at
            # +/-2.0 m only 8, and the distance covered doubled. A lag bound
            # tight enough to be tidy is tight enough to be the reason the
            # filter refuses.
            g.append(e_l)
            lbg.append(-2.0)
            ubg.append(2.0)
        # The terminal set: stopped. This is the whole difference between a
        # safety filter and an N-step lookahead.
        g.append(X[3, N])
        lbg.append(0.0)
        ubg.append(self.stop_speed)

        self._lbg, self._ubg = np.array(lbg), np.array(ubg)
        self._nx = 5 * (N + 1)
        lbw = np.concatenate([np.tile([-1e6, -1e6, -1e6, 0.0, -1e6], N + 1),
                              np.tile([-STEER_MAX, -ACCEL_MAX, 0.0], N)])
        ubw = np.concatenate([np.tile([1e6, 1e6, 1e6, SPEED_MAX, 1e6], N + 1),
                              np.tile([STEER_MAX, ACCEL_MAX, SPEED_MAX], N)])
        self._lbw, self._ubw = lbw, ubw
        nlp = {"x": w, "p": p, "f": J, "g": ca.vertcat(*g)}
        self.solver = ca.nlpsol("safety", "ipopt", nlp, {
            "ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter, "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        })

    def _guess(self, state5):
        """A braking trajectory that actually *moves*.

        The first version decayed the speed while leaving x, y and psi at their
        initial values, which is not a trajectory -- it violates the dynamics
        constraints at every node, and IPOPT has to discover the whole path
        from a point that is nowhere near feasible. On this problem that meant
        363 of 400 solves reported infeasible and the filter overrode 100% of
        inputs while the car crawled. Integrating the braking manoeuvre for the
        guess costs a few dozen floating-point operations and is the difference
        between a filter and a handbrake.
        """
        st = np.asarray(state5, float)
        x, y, psi, v, arc = (float(q) for q in st[:5])
        X = np.zeros((5, self.N + 1))
        X[:, 0] = (x, y, psi, v, arc)
        for k in range(self.N):
            # Steer towards the centreline while braking, i.e. the same backup
            # manoeuvre the ASIF filter uses -- it is the answer most of the time.
            kk = self.track.project(x, y)
            tgt = float(self.track.tangent_angle(kk))
            d = self.lateral(x, y)
            err = np.arctan2(np.sin(tgt - psi), np.cos(tgt - psi))
            delta = float(np.clip(err - 1.2 * d, -STEER_MAX, STEER_MAX))
            v = min(max(v + (-ACCEL_MAX - DRAG * v) * self.dt, 0.0), SPEED_MAX)
            psi_dot = v / self.wheelbase * np.tan(delta)
            lim = A_LAT_MAX * self.assumed_grip / max(v, 1e-3)
            psi_dot = min(max(psi_dot, -lim), lim)
            x, y, psi = (x + v * np.cos(psi) * self.dt,
                         y + v * np.sin(psi) * self.dt, psi + psi_dot * self.dt)
            arc += v * self.dt
            X[:, k + 1] = (x, y, psi, v, arc)
        U = np.vstack([np.zeros(self.N), np.full(self.N, -ACCEL_MAX),
                       np.maximum(X[3, :self.N], 0.0)])
        return np.concatenate([X.ravel(order="F"), U.ravel(order="F")])

    def _solve(self, state5, uL):
        w0 = self._w0 if self._w0 is not None else self._guess(state5)
        p = np.concatenate([np.asarray(state5, float)[:5],
                            np.asarray(uL, float)[:2]])
        sol = self.solver(x0=w0, p=p, lbx=self._lbw, ubx=self._ubw,
                          lbg=self._lbg, ubg=self._ubg)
        ok = self.solver.stats().get("success", False)
        w = np.array(sol["x"]).ravel()
        if ok:
            self._w0 = w
        return ok, w[self._nx:self._nx + 2]

    def certify(self, state5, delta: float, a: float) -> bool:
        """Feasibility with ``u_0`` pinned. The certificate *is* the solve."""
        lb, ub = self._lbw.copy(), self._ubw.copy()
        i = self._nx
        lb[i:i + 2] = ub[i:i + 2] = [delta, a]
        w0 = self._guess(state5)
        sol = self.solver(x0=w0, p=np.concatenate(
            [np.asarray(state5, float)[:5], [delta, a]]),
            lbx=lb, ubx=ub, lbg=self._lbg, ubg=self._ubg)
        return bool(self.solver.stats().get("success", False))

    def __call__(self, state5, u):
        self.n_steps += 1
        u = np.asarray(u, dtype=float)
        ok, u0 = self._solve(state5, u[:2])
        if not ok:
            self.n_no_safe_action += 1
            self.n_interventions += 1
            return np.array([u[0], -ACCEL_MAX, u[2]]), True
        # 1e-3, not 1e-4: an interior-point solve does not reproduce the input
        # to the last decimal even when it is accepting it unchanged, and
        # counting that as an intervention inflates the one number the whole
        # comparison turns on.
        if np.allclose(u0, u[:2], atol=1e-3):
            return u, False
        self.n_interventions += 1
        return np.array([u0[0], u0[1], u[2]]), True
