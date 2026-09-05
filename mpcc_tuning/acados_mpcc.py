"""An acados solver behind :class:`mpcc_tuning.mpcc.MPCC`'s interface.

The learner in ``ltc.py`` asks its solver for three things: ``value(s, theta)``,
``action_value(s, theta, a)`` and ``grad_theta(out, s, theta)``. Only the
CasADi/IPOPT problem implements them, so every tuning result in this repo was
measured on a solver that needs 126 ms per tick and cannot run on the car.

This wraps the acados solver in the same three methods, so the tuner runs
unchanged on the backend that ships -- 9-11 ms/tick, and the configuration
measured at 7.12 laps of the oval against IPOPT's 5.12.

## What is genuinely different, and stays different

``Q(s, a)`` with the first control PINNED is what the CasADi problem provides
by tightening two bounds. acados can do the same through ``lbu``/``ubu`` on
stage 0, and does here -- but pinning the input to an SQP solver that takes
ONE bounded set of iterations is not the same object as pinning it to a
problem solved to convergence. Where the tuner only needs ``Q(s, pi(s))``,
which equals ``V(s)`` by definition, no second solve happens at all and the
difference does not arise. That is the common path.

## The warm start is the controller's, not the learner's

acados carries its solution between calls, which is the whole point of RTI.
A value query that re-solves from a different ``theta`` therefore *disturbs*
the controller's warm start. The controller's own solve happens first and its
result is cached; anything the learner asks afterwards is done on a copy of
the iterate and restored, so the learner cannot degrade the driving.
"""

from __future__ import annotations

import numpy as np


class AcadosMPCC:
    """acados behind the MPCC interface the tuner expects.

    ``track``, ``horizon`` and ``dt`` mirror :class:`mpcc_tuning.mpcc.MPCC` so
    the two are interchangeable at the call site.
    """

    def __init__(self, track, horizon=25, dt=0.05, vehicle="dynamic",
                 variant="fqp_soft_funnel", max_obstacles=0,
                 export_dir=None, name=None, solver_fallback=True,
                 fallback_brake=1.0, **build_kw):
        from acados_template import AcadosOcpSolver
        from mpcc_tuning.acados_grad import AcadosEnvelopeGradient
        from mpcc_tuning.acados_ocp import build_ocp
        from mpcc_tuning.acados_variants import BY_NAME, sanitize_name
        from mpcc_tuning.model import DynamicBicycle

        self.track, self.N, self.dt = track, int(horizon), float(dt)
        self.vehicle = vehicle
        v = BY_NAME[variant]
        nm = sanitize_name(name or f"tune_{variant}_{horizon}")
        kw = dict(v.build_kwargs()); kw.update(build_kw)
        kw.setdefault("spline_mode", "spline")
        self.ocp = build_ocp(track, horizon=self.N, dt=self.dt, vehicle=vehicle,
                             max_obstacles=max_obstacles, name=nm, **kw)
        v.apply(self.ocp)
        import os
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        self.ocp.code_export_directory = str(
            export_dir or (root / "c_generated" / f"{nm}_{os.getpid()}"))
        self.sv = AcadosOcpSolver(
            self.ocp, json_file=self.ocp.code_export_directory + ".json",
            generate=True, build=True)
        # theta as p_global unlocks acados' own dV*/dtheta, which includes
        # the constraint-multiplier term the envelope form has to assume away.
        # That term is not optional once d_obs is active: d_obs sits in the
        # keep-out row, so during overtaking -- the experiment this project
        # leans on -- the hand-rolled gradient is knowingly incomplete.
        self.theta_global = bool(build_kw.get("theta_global", False))
        self._grad = AcadosEnvelopeGradient(self.ocp, self.N)
        self.max_obstacles = int(max_obstacles)
        self._obs = None
        self._nx = self.ocp.model.x.shape[0]
        self._n_theta = 8
        self._model = DynamicBicycle(dt=self.dt)
        # SOLVER-FAILURE FALLBACK.
        #
        # On a failed solve (status 1 = NaN, 4 = QP failure) acados returns its
        # last INFEASIBLE iterate. Applying it thrashes the actuators -- 39
        # ticks of it steered a car off the track in the hairpin (see TODO
        # 2i/2g). A controller must never act on a failed solve. So on failure
        # we hold the last FEASIBLE steering and command a GENTLE brake, to
        # shed the speed that made the corner infeasible. Measured on the
        # aggressive online network, cold standing start: 1.01 -> 2.81 laps,
        # peak speed 4.06 -> 2.03 m/s. A HARD brake is worse (a violent decel
        # unsettles the drift car), hence a modest fallback_brake.
        #
        # FOR THE REAL CAR: this guard is CONTROL-LOOP logic, not part of the
        # generated OCP. The acados C solver returns the status; the deployed
        # control loop (ROS node / embedded main) must replicate this check --
        # never send the actuators a control from a solve whose status is not
        # 0 or 2. This Python path is the reference implementation.
        self.solver_fallback = bool(solver_fallback)
        self.fallback_brake = float(fallback_brake)
        self._last_good_u = None
        self._seeded = False
        # the solver's own bounds, so a pinned action can be released again
        self._lbu = np.array(self.ocp.constraints.lbu, float)
        self._ubu = np.array(self.ocp.constraints.ubu, float)

    # -- parameters --------------------------------------------------------
    def _p(self, theta):
        """The PER-STAGE parameter vector.

        With ``theta_global`` the weights are not in it at all -- they go to
        ``set_p_global_and_precompute_dependencies`` once -- and this carries
        only the obstacle slots.
        """
        th = np.asarray(theta, float).ravel()
        n_p = self.ocp.model.p.shape[0]
        if self.theta_global:
            pad = np.zeros(n_p)
            if n_p and self._obs is not None and self.max_obstacles:
                ox, oy, r = self._obs[0]
                pad[:3] = [ox, oy, r]
            return pad
        if n_p == th.size:
            return th
        pad = np.zeros(n_p - th.size)
        if self._obs is not None and self.max_obstacles:
            ox, oy, r = self._obs[0]
            # inactive-slot convention: r_raw = -d_obs so r_eff is exactly zero
            pad[:3] = [ox, oy, r]
        return np.concatenate([th, pad])

    def _set_params(self, theta, p):
        """Push theta and the per-stage parameters into the solver."""
        if self.theta_global:
            self.sv.set_p_global_and_precompute_dependencies(
                np.asarray(theta, float).ravel())
        if p.size:
            for k in range(self.N + 1):
                self.sv.set(k, "p", p)
        elif not self.theta_global:
            for k in range(self.N + 1):
                self.sv.set(k, "p", p)

    def set_obstacles(self, obs):
        self._obs = list(obs) if obs else None

    def reset(self):
        """Forget the previous episode entirely.

        ``_seed`` re-fills the primal (x, u) on the first solve, but acados
        also carries the multipliers, the slacks and the funnel globalization
        memory, and none of those were being cleared. Measured: a FIXED theta
        on a deterministic plant gave different laps in different episodes --
        one seed dropped from ~1.8 to 0.4 laps in episode 2 and recovered --
        which is noise from the solver, not the car. ``solver.reset()`` zeroes
        all internal state; the primal is then re-seeded as before.
        """
        try:
            self.sv.reset()
        except Exception:      # older acados without reset(): seed only
            pass
        self._seeded = False
        self._last_good_u = None

    def _seed(self, state, theta):
        """Roll the model forward for the first solve.

        acados starts every stage at x = 0, which for this model is vx = 0 --
        the worst-conditioned point, where the slip angles are atan(vy/0).
        """
        import casadi as ca
        x0 = np.asarray(state, float).ravel()
        if x0.size < self._nx:
            x0 = np.concatenate([x0, np.zeros(self._nx - x0.size)])
        d = self._model
        xs, us = ca.MX.sym("x", 4 + d.n_dyn), ca.MX.sym("u", 2)
        stp = ca.Function("s", [xs, us], [d.step_sym(xs, us, self.dt)])
        xk = np.concatenate([x0[:4], x0[5:]]); ss = float(x0[4])
        for k in range(self.N + 1):
            self.sv.set(k, "x", np.concatenate([xk[:4], [ss], xk[4:]]))
            if k < self.N:
                self.sv.set(k, "u", np.array([0.0, 0.0, max(xk[3], 0.5)]))
                xk = np.asarray(stp(xk, [0.0, 0.0])).ravel()
                ss += max(xk[3], 0.5) * self.dt
        self._seeded = True

    def _shift(self):
        """RTI shift, terminal node extrapolated rather than left stale."""
        import casadi as ca
        d = self._model
        for j in range(self.N):
            self.sv.set(j, "x", self.sv.get(j + 1, "x"))
            if j < self.N - 1:
                self.sv.set(j, "u", self.sv.get(j + 1, "u"))
        xN1 = self.sv.get(self.N - 1, "x"); uN1 = self.sv.get(self.N - 1, "u")
        xs, us = ca.MX.sym("x", 4 + d.n_dyn), ca.MX.sym("u", 2)
        stp = ca.Function("s", [xs, us], [d.step_sym(xs, us, self.dt)])
        xd = np.asarray(stp(np.concatenate([xN1[:4], xN1[5:]]), uN1[:2])).ravel()
        self.sv.set(self.N, "x",
                    np.concatenate([xd[:4], [xN1[4] + uN1[2] * self.dt], xd[4:]]))

    # -- the interface the tuner uses -------------------------------------
    def value(self, state, theta):
        """``V(s)``: solve, return the value and the action to apply."""
        x0 = np.asarray(state, float).ravel()
        if x0.size < self._nx:
            x0 = np.concatenate([x0, np.zeros(self._nx - x0.size)])
        if not self._seeded:
            self._seed(x0, theta)
        else:
            self._shift()
        p = self._p(theta)
        self._set_params(theta, p)
        self.sv.set(0, "lbx", x0); self.sv.set(0, "ubx", x0)
        status = self.sv.solve()
        u0 = np.asarray(self.sv.get(0, "u"), float)
        # usable, not converged: MAX_ITER and MIN_STEP return the best iterate
        # and a bounded-iteration controller reports them constantly.
        ok = bool(np.isfinite(u0).all()) and status not in (1, 4)
        cost = float(self.sv.get_cost())
        used_fallback = False
        if self.solver_fallback:
            if ok:
                self._last_good_u = u0.copy()
            elif self._last_good_u is not None:
                # do NOT apply the failed iterate; hold last feasible steering,
                # brake gently to recover feasibility (see __init__)
                u0 = self._last_good_u.copy()
                u0[1] = min(float(u0[1]), -self.fallback_brake)
                used_fallback = True
        return dict(u0=u0, value=cost, ok=ok, status=int(status),
                    fallback=used_fallback, _p=p)

    def action_value(self, state, theta, action, v_out=None):
        """``Q(s, a)``. Identical to ``V(s)`` when ``a`` is the policy action."""
        if v_out is not None:
            a = np.asarray(action, float).ravel()[:2]
            if np.allclose(a, np.asarray(v_out["u0"], float).ravel()[:2],
                           atol=1e-9):
                return v_out          # Q(s, pi(s)) = V(s), by definition
        x0 = np.asarray(state, float).ravel()
        if x0.size < self._nx:
            x0 = np.concatenate([x0, np.zeros(self._nx - x0.size)])
        lbu, ubu = self._lbu.copy(), self._ubu.copy()
        a = np.asarray(action, float).ravel()
        lbu[:2] = ubu[:2] = a[:2]
        self.sv.constraints_set(0, "lbu", lbu)
        self.sv.constraints_set(0, "ubu", ubu)
        p = self._p(theta)
        self._set_params(theta, p)
        self.sv.set(0, "lbx", x0); self.sv.set(0, "ubx", x0)
        status = self.sv.solve()
        out = dict(u0=np.asarray(self.sv.get(0, "u"), float),
                   value=float(self.sv.get_cost()),
                   ok=status not in (1, 4), status=int(status), _p=p)
        self.sv.constraints_set(0, "lbu", self._lbu)
        self.sv.constraints_set(0, "ubu", self._ubu)
        return out

    def grad_theta(self, out, state, theta) -> np.ndarray:
        """``dJ*/dtheta`` at the returned solution.

        Native when theta is a ``p_global`` parameter -- acados differentiates
        its own optimal value and the multiplier term comes along -- and the
        envelope form otherwise, which is exact only while theta stays out of
        the constraints.
        """
        p = out.get("_p") if isinstance(out, dict) else None
        if p is None:
            p = self._p(theta)
        if self.theta_global:
            return np.asarray(
                self.sv.eval_and_get_optimal_value_gradient("p_global"),
                float).ravel()[:self._n_theta]
        return self._grad(self.sv, lambda k: p)

    def gradient_is_exact(self) -> tuple[bool, str]:
        if self.theta_global:
            return True, ("acados' own dV*/dtheta via "
                          "eval_and_get_optimal_value_gradient('p_global'), "
                          "which carries the constraint-multiplier term")
        return self._grad.trustworthy()
