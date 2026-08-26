"""The MPCC, parametrised so its cost weights can be learned.

One NLP, built once, solved twice per control tick:

``V(s)``
    the ordinary MPCC solve. Its optimal value is the state-value estimate and
    its first control is the action actually applied.
``Q(s, a)``
    the *same* NLP with the first control pinned to ``a``. Not a second
    problem -- just tighter bounds on two decision variables -- which is what
    makes evaluating a state-action value affordable at control rate.

That correspondence is the whole trick of MPC-as-function-approximator
(Gros & Zanon): a single optimal-control problem supplies the policy, the value
function and the action-value function at once, and the RL layer never has to
learn a critic network, because the MPC already is one.

## The cost

Standard MPCC. With ``e_c`` the contouring error, ``e_l`` the lag error, and
``v_s`` the rate of the progress variable::

    J = sum_k  q_c e_c^2 + q_l e_l^2 - q_v v_s dt + r_d delta^2 + r_a a^2
               + r_dv (v_s - v)^2

The learnable parameters are the **logs** of the five weights, so they stay
positive under an unconstrained gradient step and so a step is multiplicative
-- which is what you want for a quantity spanning orders of magnitude.

Note what ``-q_v v_s dt`` is: the MPC's *internal* incentive to make progress.
It is not the reward. The reward the RL layer sees is the real objective
(distance covered without leaving the track), and the point of the exercise is
that the weights which best serve that objective are not knowable in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from mpcc_tuning.model import ACCEL_MAX, SPEED_MAX, STEER_MAX, KinematicBicycle

WEIGHT_NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


@dataclass
class MPCCWeights:
    """The tunable weights. ``to_log``/``from_log`` are what the learner moves."""

    q_c: float = 10.0     # contouring error
    q_l: float = 10.0     # lag error
    q_v: float = 1.0      # reward for progress
    r_d: float = 0.1      # steering effort
    r_a: float = 0.01     # acceleration effort
    r_dv: float = 0.1     # keep the progress rate near the actual speed

    def to_log(self) -> np.ndarray:
        return np.log(np.array([getattr(self, n) for n in WEIGHT_NAMES], dtype=float))

    @staticmethod
    def from_log(theta: np.ndarray) -> "MPCCWeights":
        return MPCCWeights(**dict(zip(WEIGHT_NAMES, np.exp(np.asarray(theta, float)))))

    def __str__(self) -> str:
        return "  ".join(f"{n}={getattr(self, n):8.3f}" for n in WEIGHT_NAMES)


class MPCC:
    """Model predictive contouring control with learnable cost weights."""

    def __init__(self, track, model: KinematicBicycle | None = None, horizon: int = 20,
                 dt: float = 0.1, car_half_width: float = 0.12, max_iter: int = 60):
        self.track = track
        self.model = model or KinematicBicycle(dt=dt)
        self.N, self.dt = int(horizon), float(dt)
        self.margin = track.half_width - car_half_width
        self.n_theta = len(WEIGHT_NAMES)
        self._build(max_iter)
        self._w0 = None   # warm start

    # -- construction ------------------------------------------------------
    def _build(self, max_iter: int) -> None:
        N = self.N
        X = ca.MX.sym("X", 5, N + 1)     # [x, y, psi, v, s]
        U = ca.MX.sym("U", 3, N)         # [delta, a, v_s]
        x0 = ca.MX.sym("x0", 5)
        theta = ca.MX.sym("theta", self.n_theta)
        p = ca.vertcat(x0, theta)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        q_c, q_l, q_v, r_d, r_a, r_dv = (ca.exp(theta[i]) for i in range(self.n_theta))

        g, lbg, ubg = [ca.reshape(X[:, 0] - x0, -1, 1)], [0.0] * 5, [0.0] * 5
        J = 0
        for k in range(N):
            e_c, e_l = self.track.errors(X[0, k], X[1, k], X[4, k])
            J += (q_c * e_c ** 2 + q_l * e_l ** 2
                  - q_v * U[2, k] * self.dt
                  + r_d * U[0, k] ** 2 + r_a * U[1, k] ** 2
                  + r_dv * (U[2, k] - X[3, k]) ** 2)
            nxt = ca.vertcat(self.model.step_sym(X[0:4, k], U[0:2, k], self.dt),
                             X[4, k] + U[2, k] * self.dt)
            g.append(X[:, k + 1] - nxt)
            lbg += [0.0] * 5
            ubg += [0.0] * 5
            # Stay on the track. As an inequality on the contouring error, which
            # is the natural coordinate here -- the MPCC already computes it.
            g.append(e_c)
            lbg.append(-self.margin)
            ubg.append(self.margin)
        e_cN, e_lN = self.track.errors(X[0, N], X[1, N], X[4, N])
        J += q_c * e_cN ** 2 + q_l * e_lN ** 2

        self._lbg, self._ubg = np.array(lbg), np.array(ubg)
        self._nx = 5 * (N + 1)
        self._nw = self._nx + 3 * N
        lbw = np.concatenate([np.tile([-ca.inf, -ca.inf, -ca.inf, 0.0, -ca.inf], N + 1),
                              np.tile([-STEER_MAX, -ACCEL_MAX, 0.0], N)])
        ubw = np.concatenate([np.tile([ca.inf, ca.inf, ca.inf, SPEED_MAX, ca.inf], N + 1),
                              np.tile([STEER_MAX, ACCEL_MAX, SPEED_MAX], N)])
        self._lbw, self._ubw = np.array(lbw, float), np.array(ubw, float)

        gg = ca.vertcat(*g)
        nlp = {"x": w, "p": p, "f": J, "g": gg}
        self.solver = ca.nlpsol("mpcc", "ipopt", nlp, {
            "print_time": False,
            "ipopt": {"print_level": 0, "sb": "yes", "max_iter": max_iter,
                      "tol": 1e-4, "acceptable_tol": 1e-3, "warm_start_init_point": "yes"},
        })

        # The gradient the learner needs, by the envelope theorem: at the
        # solution, d(optimal value)/d(theta) is the *partial* derivative of the
        # Lagrangian, with the primal and dual variables held fixed. No implicit
        # function theorem, no differentiating through the solver, no adjoint
        # pass -- one evaluation of a function that was built once. That is why
        # this is affordable at control rate, and it is the single most
        # important line in the repo.
        lam_g = ca.MX.sym("lam_g", gg.shape[0])
        lagrangian = J + ca.dot(lam_g, gg)
        self.dQ_dtheta = ca.Function("dQ", [w, lam_g, p],
                                     [ca.gradient(lagrangian, theta)])

    # -- solving -----------------------------------------------------------
    def _solve(self, state5: np.ndarray, theta: np.ndarray, fix_u0=None):
        lbw, ubw = self._lbw.copy(), self._ubw.copy()
        if fix_u0 is not None:
            i = self._nx
            lbw[i:i + 2] = ubw[i:i + 2] = np.asarray(fix_u0, float)[:2]
        w0 = self._w0 if self._w0 is not None else self._initial_guess(state5)
        sol = self.solver(x0=w0, p=np.concatenate([state5, theta]),
                          lbx=lbw, ubx=ubw, lbg=self._lbg, ubg=self._ubg)
        ok = self.solver.stats().get("success", False)
        w = np.array(sol["x"]).ravel()
        return dict(w=w, lam_g=np.array(sol["lam_g"]).ravel(),
                    value=float(sol["f"]), u0=w[self._nx:self._nx + 3], ok=ok)

    def _initial_guess(self, state5: np.ndarray) -> np.ndarray:
        X = np.tile(np.asarray(state5, float)[:, None], (1, self.N + 1))
        X[4] += np.arange(self.N + 1) * state5[3] * self.dt
        U = np.tile(np.array([0.0, 0.0, max(state5[3], 0.5)])[:, None], (1, self.N))
        return np.concatenate([X.ravel(order="F"), U.ravel(order="F")])

    def value(self, state5, theta):
        """``V(s)``: solve, and return the optimal value and the action to apply."""
        out = self._solve(state5, theta)
        self._w0 = out["w"]                       # warm start the next tick
        return out

    def action_value(self, state5, theta, action, v_out=None):
        """``Q(s, a)``: the same NLP with the first control pinned to ``a``.

        If ``a`` is the action the unconstrained solve already chose, the two
        problems have the same solution -- ``Q(s, pi(s)) = V(s)`` is the
        definition of the policy -- so pass that solve in as ``v_out`` and no
        second NLP is run at all. This matters: a second IPOPT call per control
        tick is most of the compute budget, and it is only genuinely needed when
        the applied action was perturbed away from the argmin for exploration.
        """
        if v_out is not None and np.allclose(v_out["u0"][:2], np.asarray(action, float)[:2],
                                             atol=1e-9):
            return v_out
        return self._solve(state5, theta, fix_u0=action)

    def grad_theta(self, out, state5, theta) -> np.ndarray:
        """``dQ/dtheta`` at a solved problem, via the envelope theorem."""
        return np.array(self.dQ_dtheta(out["w"], out["lam_g"],
                                       np.concatenate([state5, theta]))).ravel()

    def reset(self) -> None:
        self._w0 = None
