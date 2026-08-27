"""A real real-time iteration: one QP per control tick, warm-started.

Why this file exists
--------------------
The obvious way to fake RTI with an interior-point solver is to cap it at one
iteration. **That does not work, and it produces a convincing wrong answer.**
Measured on this problem, IPOPT with ``max_iter=1``:

* reports success on **0 of 41** solves, and
* moves the iterate by 8.19 on average -- an order of magnitude *further* than
  the converged step (0.748), because an interior-point method's first step
  from a warm start is large and not yet meaningful.

The resulting ``u_0`` is not an approximate solution, it is a failed one, and
its sensitivity to the cost weights is noise. An earlier version of
``experiments/rti_influence.py`` measured exactly that noise and reported it as
evidence that the memoryless gradient "points elsewhere". It does not; the
experiment was measuring a broken solve.

Real-time iteration (Diehl et al.) is an **SQP** scheme: linearise once at the
current iterate and solve **one full QP**, warm-started from the previous tick.
That is a well-defined approximate solution with a convergence theory, and it
is what ``acados``'s ``SQP_RTI`` does. This class implements it, so the question
"does the warm start carry information the envelope theorem misses" can be
asked of the thing the question is actually about.

The scheme
----------
At iterate :math:`w`, with constraints :math:`g(w) = 0` (dynamics) and bounds,
solve

.. math::
    \\min_{\\Delta w} \; \\tfrac12 \\Delta w^\\top B \\Delta w
        + \\nabla_w f^\\top \\Delta w
    \\quad\\text{s.t.}\\quad
    \\mathrm{lbg} \\le g(w) + \\nabla_w g\\, \\Delta w \\le \\mathrm{ubg},
    \;\; \\mathrm{lbw} \\le w + \\Delta w \\le \\mathrm{ubw}

and set :math:`w \\leftarrow w + \\alpha\\,\\Delta w`. ``B`` is the
Gauss-Newton Hessian of the cost, regularised -- the standard choice for a
least-squares-shaped MPCC objective, and positive definite by construction, so
the QP is convex and one solve is cheap and reliable.
"""

from __future__ import annotations

import casadi as ca
import numpy as np


class RTISolver:
    """One SQP step per call, warm-started. Wraps an existing :class:`MPCC`."""

    def __init__(self, mpcc, qp: str = "qrqp", reg: float = 1e-6,
                 step: float = 1.0):
        self.m, self.reg, self.step = mpcc, float(reg), float(step)
        w, p = mpcc._w_sym, mpcc._p_sym
        f, g = mpcc._f_sym, mpcc._g_sym
        self._nw = w.shape[0]
        self._ng = g.shape[0]
        H = ca.hessian(f, w)[0] + self.reg * ca.DM.eye(self._nw)
        self._fn = ca.Function("lin", [w, p],
                               [ca.gradient(f, w), ca.jacobian(g, w), g, H],
                               ["w", "p"], ["gf", "Jg", "gval", "H"])
        self.qp = ca.conic("rti", qp,
                           {"h": H.sparsity(), "a": ca.jacobian(g, w).sparsity()},
                           {"print_time": False, "error_on_fail": False,
                            "print_iter": False, "print_header": False}
                           | ({"printLevel": "none"} if qp == "qpoases" else {}))
        self.w = None
        self.n_calls = 0

    def reset(self) -> None:
        self.w = None

    def solve(self, state5, theta):
        """One QP step from the carried iterate. Returns the same dict as MPCC."""
        m = self.m
        p = np.concatenate([np.asarray(state5, float), np.asarray(theta, float)])
        if self.w is None:
            self.w = m._initial_guess(state5)
        r = self._fn(w=self.w, p=p)
        gf = np.array(r["gf"]).ravel()
        gval = np.array(r["gval"]).ravel()
        sol = self.qp(h=r["H"], g=gf, a=r["Jg"],
                      lba=m._lbg - gval, uba=m._ubg - gval,
                      lbx=m._lbw - self.w, ubx=m._ubw - self.w)
        dw = np.array(sol["x"]).ravel()
        if not np.all(np.isfinite(dw)):
            dw = np.zeros_like(dw)
        self.w = self.w + self.step * dw
        self.n_calls += 1
        return dict(w=self.w.copy(), lam_g=np.array(sol["lam_a"]).ravel(),
                    value=float("nan"),
                    u0=self.w[m._nx:m._nx + 3], ok=bool(np.all(np.isfinite(dw))),
                    dw=float(np.linalg.norm(dw)))
