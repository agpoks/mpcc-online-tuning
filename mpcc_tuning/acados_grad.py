"""The envelope gradient for the acados backend.

Online tuning needs ``dJ*/dtheta`` every tick. ``MPCC.grad_theta`` gets it from
IPOPT by reading ``lam_g`` out of the solution, and there is no acados
equivalent -- which is why the tuner could only ever run on the reference
solver and not on the one that ships.

It does not need one. The envelope theorem says that at the optimum

    dJ*/dtheta  =  dL/dtheta

and when ``theta`` appears **only in the cost** the multipliers drop out
entirely: the gradient is the partial derivative of the objective, evaluated at
the solution. That is a CasADi function of the trajectory the solver already
returned, so it can be built once and called on acados' output without touching
acados' internals.

## When this is valid, and when it is not

``theta`` is in the cost alone for the dynamic model as configured: the grip row
(where ``k_v`` lives) auto-disables, the terminal speed row with it, and the
keep-out row (where ``d_obs`` lives) exists only when ``max_obstacles > 0``.

**With obstacles the assumption breaks.** ``d_obs`` enters ``g``, and the
honest gradient needs the ``lambda^T dg/dtheta`` term. acados exposes the
multipliers as ``lam``, so :func:`grad_theta` takes them when available and
reports which components are trustworthy rather than silently returning a
number that is wrong for one of eight weights.

This mirrors a defect already recorded for the CasADi backend: ``k_v`` in the
grip row made ``dJ*/dtheta`` wrong exactly when that row was active, and the
paper still claims "theta stays out of g". Getting it right here rather than
inheriting the same error.
"""

from __future__ import annotations

import numpy as np


class AcadosEnvelopeGradient:
    """``dJ*/dtheta`` from an acados solution, by the envelope theorem.

    Built once per solver; called every tick. The cost expression comes from
    the same :func:`mpcc_tuning.acados_ocp.build_ocp` that generated the
    solver, so the thing differentiated is the thing minimised -- if those two
    ever diverge the gradient is exact for the wrong problem.
    """

    def __init__(self, ocp, horizon: int, n_theta: int = 8):
        import casadi as ca

        self.N = int(horizon)
        self.n_theta = int(n_theta)
        x, u = ocp.model.x, ocp.model.u
        p = ocp.model.p
        if p is None:
            p = ca.SX.sym("p_none", 0)
        # theta lives in p_global when the OCP was built for acados' own
        # sensitivities, and model.p is then EMPTY -- slicing it for theta
        # raised "Slice (0, 8) out of bounds with length 0" and aborted the
        # process. Both layouts have to work: this class is the cross-check
        # for the native gradient, so it must build on the same OCP.
        pg = getattr(ocp.model, "p_global", None)
        if pg is not None and pg.numel():
            theta = pg[:n_theta]
        else:
            pg = ca.SX.sym("p_global_none", 0)
            theta = p[:n_theta]
        self.theta_global = bool(pg.numel())

        stage = ocp.model.cost_expr_ext_cost
        term = ocp.model.cost_expr_ext_cost_e
        # d(stage cost)/d(theta) and the same for the terminal cost. Summed over
        # the horizon at the solution, this IS dJ*/dtheta when theta is only in
        # the cost.
        self._dstage = ca.Function("dJdth", [x, u, p, pg],
                                   [ca.jacobian(stage, theta).T])
        self._dterm = ca.Function("dJedth", [x, p, pg],
                                  [ca.jacobian(term, theta).T])

        # Does theta reach the constraints? If so the envelope formula needs the
        # multiplier term and this class cannot be trusted alone.
        h = getattr(ocp.model, "con_h_expr", None)
        self.theta_in_g = False
        if h is not None and h.numel():
            self.theta_in_g = bool(ca.depends_on(h, theta))
            if self.theta_in_g:
                self._dh = ca.Function("dhdth", [x, u, p, pg],
                                       [ca.jacobian(h, theta)])

    def __call__(self, solver, p_of_stage, p_global=None) -> np.ndarray:
        """Sum ``dJ/dtheta`` over the solved horizon.

        ``p_of_stage`` is a callable ``k -> parameter vector``, because the
        parameters differ per stage when the corridor is passed as half-spaces.
        ``p_global`` carries theta when the OCP declares it global; it is
        ignored for the per-stage layout.
        """
        pg = np.zeros(0) if p_global is None else np.asarray(p_global, float)
        if self.theta_global and pg.size != self.n_theta:
            raise ValueError(
                "theta is a p_global parameter on this OCP, so p_global must "
                f"carry the {self.n_theta} weights; got size {pg.size}")
        g = np.zeros(self.n_theta)
        for k in range(self.N):
            xk = solver.get(k, "x")
            uk = solver.get(k, "u")
            g += np.asarray(self._dstage(xk, uk, p_of_stage(k), pg)).ravel()
        xN = solver.get(self.N, "x")
        g += np.asarray(self._dterm(xN, p_of_stage(self.N), pg)).ravel()
        return g

    def trustworthy(self) -> tuple[bool, str]:
        """Whether the envelope form alone is valid for this OCP."""
        if self.theta_in_g:
            return False, ("theta appears in the constraints (d_obs in the "
                           "keep-out row), so dJ*/dtheta needs the "
                           "lambda^T dg/dtheta term this class omits")
        return True, "theta is in the cost only; the envelope form is exact"


def check_against_finite_differences(solver, ocp, grad, p_of_stage, theta,
                                     set_theta, eps=1e-4):
    """Compare the analytic gradient with finite differences of the cost.

    Worth running once per configuration and NOT trusting blindly: on the
    CasADi backend the finite-difference side turned out to be the unreliable
    one -- |fd| grew as eps shrank, which is solver tolerance divided by eps
    rather than a derivative. A disagreement here means "investigate", not
    "the analytic gradient is wrong".
    """
    an = grad(solver, p_of_stage)
    fd = np.empty_like(an)
    for i in range(len(an)):
        out = []
        for sgn in (+1.0, -1.0):
            th = np.array(theta, float)
            th[i] += sgn * eps
            set_theta(th)
            solver.solve()
            out.append(solver.get_cost())
        fd[i] = (out[0] - out[1]) / (2 * eps)
    set_theta(np.array(theta, float))
    solver.solve()
    denom = np.linalg.norm(an) * np.linalg.norm(fd) + 1e-12
    return dict(analytic=an, finite=fd,
                cosine=float(an @ fd / denom),
                rel=float(np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-12)))
