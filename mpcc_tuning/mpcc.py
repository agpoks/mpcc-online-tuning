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

## Obstacles

``max_obstacles`` circular keep-outs, passed as runtime parameters, softened
with explicit slacks. This exists because *overtaking cannot be expressed by a
weight* until the controller can see an opponent at all: with no obstacle in
the OCP, "go around" and "stay behind" are the same problem and no policy over
theta can distinguish them.

The formulation is copied from
``MPCC_planner_acados/scripts/generate_acados_solver.py`` (its ``max_obstacles``
block) and adapted from an acados soft path constraint to this CasADi NLP:

* there, ``h = dist2 - r_eff**2 >= 0`` softened by ``idxsh``/``Zl``/``zl``;
* here, the same ``h``, with the slack as an explicit decision variable
  ``h + s >= 0``, ``s >= 0``, penalised ``Z s**2 + z s`` -- which is what
  acados' ``Zl``/``zl`` are, written out.

Two details are carried over deliberately. **Inactive obstacles are passed with
``r_raw = -obs_margin``** so that ``r_eff = r_raw + obs_margin`` is exactly
zero and the keep-out vanishes without a non-smooth ``max()`` appearing in the
NLP. And the slack is in units of **squared** distance, because the constraint
is, which is worth knowing when reading ``Z`` -- it is a penalty per m^2 of
overlap area, not per metre of intrusion. Keeping both quirks identical to the
template is what makes the eventual acados port a swap rather than a rewrite.

The keep-out is *not* applied at stage 0: that state is pinned to the measured
``x0``, so the constraint there is a statement about the past. It would be
unsatisfiable exactly when the car is already touching an opponent, which is
the one moment the solver most needs to still return something.

``max_obstacles=0`` (the default) changes nothing: no parameters, no slacks, no
extra constraint rows, and an unchanged decision vector, so every existing
result and ``mpcc_tuning/rti.py`` are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from mpcc_tuning.model import ACCEL_MAX, SPEED_MAX, STEER_MAX, KinematicBicycle

#: The learnable parameters. The last is a **distance, not a cost**, and that
#: distinction was measured rather than assumed.
#:
#: The obvious candidate was the slack penalty on the keep-out -- price
#: intrusion, and a cheap price should mean a willingness to squeeze past. It
#: does nothing: the slack is zero unless the constraint is violated, the
#: solver simply does not violate it, and sweeping that penalty over two orders
#: of magnitude moved the closest approach by 0.000 m. What *does* move it is
#: the keep-out's own margin -- 0.15 to 0.45 m changed clearance from 0.570 to
#: 0.765 m and progress from 1.185 to 0.491.
#:
#: So ``d_obs`` is the berth the driver insists on, in metres, and it is the
#: parameter that expresses **how badly the pass is wanted**: small squeezes
#: through a gap, large waits for a clean one. A behaviour policy that cannot
#: vary it is deciding overtake-versus-follow entirely through q_v and q_c
#: while the term that actually prices proximity stays frozen.
WEIGHT_NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v")


@dataclass
class MPCCWeights:
    """The tunable weights. ``to_log``/``from_log`` are what the learner moves."""

    q_c: float = 10.0     # contouring error -- how hard the line is held
    q_l: float = 10.0     # lag error
    q_v: float = 1.0      # reward for progress -- aggression
    r_d: float = 0.1      # steering effort -- how sharply it is willing to turn
    r_a: float = 0.01     # acceleration effort -- how hard it steps on the gas
    r_dv: float = 0.1     # keep the progress rate near the actual speed
    d_obs: float = 0.15   # keep-out margin [m] -- how wide a berth an opponent gets
    k_v: float = 0.85     # fraction of the grip-limited corner speed it will use

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
                 dt: float = 0.1, car_half_width: float = 0.12, max_iter: int = 60,
                 max_obstacles: int = 0, obs_margin: float = 0.15,
                 obs_slack: tuple = (200.0, 5.0), obs_shape: str = "circle",
                 car_half_length: float = 0.285, terminal_speed: bool = True,
                 terminal_grip: float = 0.85, speed_from_grip: bool = True,
                 cbf: bool = False, cbf_alpha: float = 0.35,
                 cbf_margin: float = 0.18, cbf_lookahead: float = 0.45,
                 cbf_penalty: float = 1e3, grip_penalty: float = 1e3,
                 assumed_grip: float = 1.0):
        self.track = track
        self.model = model or KinematicBicycle(dt=dt)
        self.N, self.dt = int(horizon), float(dt)
        self.margin = track.half_width - car_half_width
        self.car_half_width = float(car_half_width)
        self.n_theta = len(WEIGHT_NAMES)
        # A fixed budget of keep-outs, as in the acados template: the OCP is
        # built once, so the number of obstacles is structural and only their
        # positions and radii are runtime parameters.
        self.max_obstacles = int(max_obstacles)
        self.obs_margin = float(obs_margin)
        # (quadratic, linear) slack penalty -- acados' Zl and zl. The template
        # uses 200 / 5 for the obstacle rows; kept, so the two agree.
        self.obs_Z, self.obs_z = (float(v) for v in obs_slack)
        # A car is about 0.57 x 0.30 m -- nearly twice as long as it is wide --
        # so a circular keep-out is conservative along the flanks and optimistic
        # at the corners. That is exactly backwards for overtaking, which
        # happens side by side. An ellipse aligned with the opponent's heading
        # costs one rotation and is still smooth.
        if obs_shape not in ("circle", "ellipse"):
            raise ValueError("obs_shape must be 'circle' or 'ellipse'")
        self.obs_shape = obs_shape
        self.car_half_length = float(car_half_length)
        self.car_half_width = float(car_half_width)
        #: (x, y, r) per obstacle for a circle; (x, y, r, psi) for an ellipse.
        self.obs_stride = 3 if obs_shape == "circle" else 4
        self._obstacles = np.zeros((0, self.obs_stride))
        # Assume slightly LESS grip than the plant has for the terminal
        # promise; a safety condition that assumes the best case is not one.
        self.terminal_speed = bool(terminal_speed)
        self.terminal_grip = float(terminal_grip)
        self.speed_from_grip = bool(speed_from_grip)
        # A discrete control barrier function as a CONSTRAINT of the OCP rather
        # than an override applied after it.
        #
        # The reason is specific to learning the weights. Anything expressed as
        # a cost term is negotiable: the policy can learn a small weight for it
        # and trade it away, which is item 5's failure -- a proxy optimised past
        # the point where the proxy is valid -- applied to safety. A constraint
        # cannot be learned away.
        #
        # It also removes a mismatch. With an external filter the learner
        # differentiates the UNFILTERED problem while the plant executes the
        # FILTERED action, so the value being learned is not the value being
        # executed. Inside the OCP they are the same object.
        #
        # Deliberately NOT parameterised by theta. theta entering g is where the
        # envelope gradient became unreliable: k_v enters the grip row and its
        # analytic gradient goes to zero while finite differences read -4.5 once
        # that row is ACTIVE (tests/test_obstacles.py, the xfail). A learnable
        # barrier margin would add another such term in exactly that regime, so
        # the margin is a constant and dJ*/dtheta stays free.
        #
        # Soft, with an explicit slack, for the same reason the keep-out is: a
        # hard barrier can make the OCP infeasible, and an infeasible solve
        # yields no action at all, which is worse than an override.
        self.cbf = bool(cbf)
        self.cbf_alpha = float(cbf_alpha)
        self.cbf_margin = float(cbf_margin)
        self.cbf_lookahead = float(cbf_lookahead)
        self.cbf_penalty = float(cbf_penalty)
        self.grip_penalty = float(grip_penalty)
        if not 0.0 < self.cbf_alpha <= 1.0:
            raise ValueError("cbf_alpha must be in (0, 1]")
        self.assumed_grip = float(assumed_grip)
        self._build(max_iter)
        self._w0 = None   # warm start

    # -- construction ------------------------------------------------------
    def _build(self, max_iter: int) -> None:
        N = self.N
        car_half_width = self.car_half_width
        # Some corridors already have the vehicle taken out of them.
        #
        # A raceline optimiser reports w_left/w_right as the room remaining for
        # the car's CENTRE, not the geometric distance to the wall -- they
        # bottom out at exactly -0.000 at every apex, which is the signature of
        # its own w >= 0 constraint. Subtracting car_half_width from those again
        # double-counts, and on ICRA T2 it makes 24% of the lap infeasible: the
        # car cannot stay inside its own corridor, covers -4.9 m and leaves the
        # track on every run of every configuration.
        #
        # Map-derived corridors are the other case: those widths are distances
        # to the wall and the vehicle must still be subtracted.
        chw = 0.0 if getattr(self.track, "width_vehicle_adjusted", False) \
            else car_half_width
        terminal_speed, terminal_grip = self.terminal_speed, self.terminal_grip
        speed_from_grip, assumed_grip = self.speed_from_grip, self.assumed_grip
        from mpcc_tuning.model import A_LAT_MAX as a_lat_max
        M = self.max_obstacles
        X = ca.MX.sym("X", 5, N + 1)     # [x, y, psi, v, s]
        U = ca.MX.sym("U", 3, N)         # [delta, a, v_s]
        x0 = ca.MX.sym("x0", 5)
        theta = ca.MX.sym("theta", self.n_theta)
        # Obstacles as runtime parameters, [ox, oy, r_raw] each, and one slack
        # per obstacle per shooting node from k=1 (stage 0 is pinned to x0).
        obs = ca.MX.sym("obs", self.obs_stride * M)
        S = ca.MX.sym("S", M, N)
        Sc = ca.MX.sym("Sc", N)          # barrier slacks, one per stage
        Sg = ca.MX.sym("Sg", N)          # grip slacks, one per stage
        St = ca.MX.sym("St", 1)          # terminal-speed slack
        p = ca.vertcat(x0, theta, obs) if M else ca.vertcat(x0, theta)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        if M:
            w = ca.vertcat(w, ca.reshape(S, -1, 1))
        if self.cbf:
            w = ca.vertcat(w, Sc)
        if speed_from_grip:
            w = ca.vertcat(w, Sg)
        if terminal_speed:
            w = ca.vertcat(w, St)

        (q_c, q_l, q_v, r_d, r_a, r_dv,
         d_obs, k_v) = (ca.exp(theta[i]) for i in range(self.n_theta))

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
            if self.cbf:
                # h(x_{k+1}) >= (1 - alpha) h(x_k), the discrete barrier
                # condition, which by induction gives h_k >= (1-a)^k h_0 > 0.
                # Same h as filters/cbf_qp.py's "braking" barrier so the
                # in-solver and post-hoc versions are the SAME criterion and any
                # difference between them is about where it is enforced.
                g.append(self._barrier_sym(X[:, k + 1])
                         - (1.0 - self.cbf_alpha) * self._barrier_sym(X[:, k])
                         + Sc[k])
                lbg.append(0.0)
                ubg.append(ca.inf)
                J += self.cbf_penalty * Sc[k] ** 2
            # Stay on the track, as an inequality on the contouring error --
            # the natural coordinate here, since the MPCC already computes it.
            #
            # On a variable-width circuit this is TWO rows, because the corridor
            # is not symmetric about the centreline and its bounds are functions
            # of s. A constant bound cannot represent a track that varies from
            # 0.69 m to 3.13 m round a lap, and a controller given one number
            # for the whole track cannot be asked whether its weights should
            # depend on the width.
            # GRIP-LIMITED SPEED, with a learnable utilisation factor.
            #
            # A flat speed cap is an assumption; the physical limit is
            # v <= sqrt(a_lat_max * mu / |kappa|), which varies round the lap.
            # k_v is the fraction of that the controller believes it can use --
            # exactly what a driver calibrates on an unfamiliar surface. If the
            # plant's true grip is mu and the controller assumes mu_hat, the
            # correct value is sqrt(mu / mu_hat), so unlike every other weight
            # here this one has a KNOWN RIGHT ANSWER and learning it can be
            # checked rather than merely reported.
            #
            # Note this puts theta into g for the first time. formulation.md
            # says the lambda term "is the whole point" the moment that
            # happens, and it is implemented; tests/test_gradient.py checks it.
            if speed_from_grip:
                # SOFT, with an explicit slack. Hard, this is the single
                # largest source of solver failure in the repo: measured on the
                # oval, 34% of ticks returned Infeasible_Problem_Detected, and
                # turning the row off drops that to 0%. The reason is
                # structural rather than numerical -- raising max_iter from 80
                # to 300 changed nothing. The constraint bounds the speed the
                # car may carry into a corner, so the moment it is already
                # travelling faster than the upcoming curvature allows, no
                # feasible trajectory exists and IPOPT is right to say so. An
                # MPC cannot have hard state constraints that the CURRENT state
                # already violates.
                #
                # It cannot simply be dropped either: with it the car covers
                # 1.84 laps of the oval and without it 0.47, because it is what
                # keeps the entry speed sane. Softening keeps the physics and
                # returns a best-effort trajectory when the car is already over
                # the limit, which is what the obstacle keep-out has always
                # done.
                kap_k = self.track.curvature_sym(X[4, k])
                g.append(a_lat_max * assumed_grip
                         - X[3, k] ** 2 * ca.fabs(kap_k) / (k_v ** 2 + 1e-9)
                         + Sg[k])
                J += self.grip_penalty * Sg[k] ** 2
                lbg.append(0.0)
                ubg.append(ca.inf)

            if getattr(self.track, "variable_width", False):
                wl, wr = self.track.width(X[4, k])
                g.append(wl - chw - e_c)                 # room to the left
                lbg.append(0.0); ubg.append(ca.inf)
                g.append(e_c + wr - chw)                 # room to the right
                lbg.append(0.0); ubg.append(ca.inf)
            else:
                g.append(e_c)
                lbg.append(-self.margin)
                ubg.append(self.margin)
            # Circular keep-outs on the *next* state, so k=1..N are covered and
            # the pinned stage 0 is not. Copied from
            # MPCC_planner_acados/scripts/generate_acados_solver.py:
            #   r_eff = r_raw + obs_margin;  dist2 - r_eff**2 >= 0
            # with inactive obstacles passed as r_raw = -obs_margin so r_eff is
            # exactly zero -- no max() and so no kink in the NLP.
            for j in range(M):
                st = self.obs_stride
                ox, oy, r_raw = obs[st * j], obs[st * j + 1], obs[st * j + 2]
                # The margin is LEARNABLE. An inactive slot is still switched
                # off arithmetically, but now by passing r_raw = -d_obs from
                # _obs_param, which reads the same theta.
                r_eff = r_raw + d_obs
                dx, dy = X[0, k + 1] - ox, X[1, k + 1] - oy
                s_kj = S[j, k]
                if self.obs_shape == "circle":
                    h_j = dx ** 2 + dy ** 2 - r_eff ** 2
                else:
                    # In the opponent's frame: u along its heading, w across.
                    # (u/a)^2 + (w/b)^2 >= 1, with the ego's own body added to
                    # both semi-axes so the constraint is between two rectangles
                    # rather than between a point and one.
                    psi_o = obs[st * j + 3]
                    cu, su = ca.cos(psi_o), ca.sin(psi_o)
                    # NOT `w`: that is the decision vector, and shadowing it
                    # here makes nlp={"x": w} a scalar expression, which CasADi
                    # reports as "argument 0(x) is not symbolic" from inside
                    # nlpsol rather than at the assignment.
                    u_ax = cu * dx + su * dy
                    v_ax = -su * dx + cu * dy
                    a = r_eff + self.car_half_length
                    b = r_eff * (self.car_half_width / max(self.car_half_length, 1e-9)) \
                        + self.car_half_width
                    # Scaled by a*b so the row has the units of the circular one
                    # (squared distance) and the slack penalty Z is comparable.
                    h_j = ((u_ax / a) ** 2 + (v_ax / b) ** 2 - 1.0) * (a * b)
                g.append(h_j + s_kj)
                lbg.append(0.0)
                ubg.append(ca.inf)
                # acados' Zl (quadratic) and zl (linear) on the same row,
                # written out because a plain NLP has no idxsh. The quadratic
                # term is now LEARNABLE -- q_obs is the price of getting close
                # to an opponent, and a policy that cannot move it cannot
                # express how badly it wants the pass. The linear term stays
                # fixed so the constraint keeps a non-zero activation cost even
                # if the learner drives q_obs down.
                J += self.obs_Z * s_kj ** 2 + self.obs_z * s_kj
        e_cN, e_lN = self.track.errors(X[0, N], X[1, N], X[4, N])
        J += q_c * e_cN ** 2 + q_l * e_lN ** 2

        # TERMINAL SAFETY CONSTRAINT. With the speed cap raised to a physical
        # value, nothing else stops the horizon ending at a speed the car
        # cannot hold through the corner it is entering -- the cap was doing
        # that job by accident, and doing it everywhere rather than where it
        # was needed.
        #
        # The condition is the cornering limit at the terminal station:
        #     v_N^2 |kappa(s_N)| <= a_lat_max * grip
        # so the last predicted state is one the vehicle can actually hold. It
        # is a *terminal* constraint rather than a stage one because the stages
        # are already bounded by the corridor; what the horizon lacks is a
        # promise about what happens after it ends.
        if terminal_speed:
            kap = self.track.curvature_sym(X[4, N])
            g.append(a_lat_max * terminal_grip - X[3, N] ** 2 * ca.fabs(kap)
                     + St[0])
            J += self.grip_penalty * St[0] ** 2
            lbg.append(0.0)
            ubg.append(ca.inf)

        self._lbg, self._ubg = np.array(lbg), np.array(ubg)
        self._nx = 5 * (N + 1)
        self._nw = self._nx + 3 * N
        # Slacks live at the end of w, after the states and controls, so _nx and
        # the u0 slice keep their meaning and rti.py needs no change.
        self._ns = (M * N + (N if self.cbf else 0)
                    + (N if self.speed_from_grip else 0)
                    + (1 if self.terminal_speed else 0))
        self._nw += self._ns
        lbw = np.concatenate([np.tile([-ca.inf, -ca.inf, -ca.inf, 0.0, -ca.inf], N + 1),
                              np.tile([-STEER_MAX, -ACCEL_MAX, 0.0], N),
                              np.zeros(self._ns)])
        ubw = np.concatenate([np.tile([ca.inf, ca.inf, ca.inf, SPEED_MAX, ca.inf], N + 1),
                              np.tile([STEER_MAX, ACCEL_MAX, SPEED_MAX], N),
                              np.full(self._ns, ca.inf)])
        self._lbw, self._ubw = np.array(lbw, float), np.array(ubw, float)

        gg = ca.vertcat(*g)
        # Kept so an SQP/RTI step can be built from the same problem rather
        # than a second transcription of it -- see mpcc_tuning/rti.py. Two
        # transcriptions that drift apart would make the comparison between
        # solvers meaningless.
        self._w_sym, self._p_sym, self._f_sym, self._g_sym = w, p, J, gg
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

    # -- the barrier -------------------------------------------------------
    def _barrier_sym(self, Xk):
        """``h`` for one stage, as a CasADi expression.

        The same quantity as :meth:`mpcc_tuning.filters.cbf_qp.CBFQP.barrier`
        with ``h_kind="braking"``:

            h = (w - margin) - |d| - T_look * |v sin(e_psi)|

        The lookahead term is what stops it being myopic. Without it, ``h`` does
        not contain the speed at all, so it permits full speed straight at a
        wall until the step before contact -- still positive, still falling
        slowly. Subtracting the lateral ground covered in ``T_look`` seconds at
        the current closing rate makes the barrier shrink when the car is moving
        *towards* a wall rather than merely sitting near one.

        Uses ``width(s)`` rather than the constant half-width so it means the
        same thing on a corridor that varies round the lap, and the narrower
        side is the binding one.
        """
        x, y, psi, v, s = Xk[0], Xk[1], Xk[2], Xk[3], Xk[4]
        e_c, _ = self.track.errors(x, y, s)
        wl, wr = self.track.width(s)
        # |z| smoothed as sqrt(z^2 + eps^2). Both absolute values here are on
        # DECISION VARIABLES, unlike the grip row's |kappa|, which is a property
        # of the track. A kink in a constraint that the solver is choosing
        # across is not a detail: with the hard fabs, solves succeeded 16% of
        # the time against pathological weights -- the barrier was not making
        # the car safe, it was making the problem intractable, and an infeasible
        # solve is no safety at all. eps is 1 cm and 1 cm/s, far below anything
        # the barrier is meant to resolve.
        eps = 1e-2
        abs_e = ca.sqrt(e_c ** 2 + eps ** 2)
        room = ca.fmin(wl, wr) - self.cbf_margin - abs_e
        phi = self.track.tangent_angle(s)
        e_psi = ca.atan2(ca.sin(phi - psi), ca.cos(phi - psi))
        closing = v * ca.sin(e_psi)
        return room - self.cbf_lookahead * ca.sqrt(closing ** 2 + eps ** 2)

    # -- obstacles ---------------------------------------------------------
    def set_obstacles(self, obstacles) -> None:
        """Set the keep-outs for subsequent solves: an iterable of ``(x, y, r)``.

        Held on the controller rather than threaded through every call, because
        they are a property of *the world at this tick* and not of the value
        being asked for -- and because the learner calls ``value``,
        ``action_value`` and ``grad_theta`` with the same world and should not
        have to carry it. That is also how the acados version passes them: as
        stage parameters set once per tick.
        """
        obs = np.asarray(list(obstacles), float).reshape(-1, self.obs_stride)
        if len(obs) > self.max_obstacles:
            raise ValueError(
                f"{len(obs)} obstacles but the OCP was built for "
                f"{self.max_obstacles}; pass max_obstacles= at construction")
        self._obstacles = obs

    def _obs_param(self, d_obs: float | None = None) -> np.ndarray:
        """The obstacle parameter block, unused slots switched off.

        An unused slot is ``r_raw = -obs_margin``, so ``r_eff`` is exactly 0 and
        ``dist2 >= 0`` holds everywhere -- the constraint is present but inert.
        Same trick as the acados template, and for the same reason: it avoids a
        ``max(0, .)`` in the NLP.
        """
        m = self.obs_margin if d_obs is None else float(d_obs)
        off = ([0.0, 0.0, -m] if self.obs_shape == "circle"
               else [0.0, 0.0, -m, 0.0])
        q = np.tile(off, (self.max_obstacles, 1))
        if len(self._obstacles):
            q[:len(self._obstacles)] = self._obstacles
        return q.ravel()

    def _p(self, state5: np.ndarray, theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, float)
        parts = [np.asarray(state5, float), th]
        if self.max_obstacles:
            parts.append(self._obs_param(float(np.exp(th[6])) if len(th) > 6 else None))
        return np.concatenate(parts)

    # -- solving -----------------------------------------------------------
    def _solve(self, state5: np.ndarray, theta: np.ndarray, fix_u0=None):
        lbw, ubw = self._lbw.copy(), self._ubw.copy()
        if fix_u0 is not None:
            i = self._nx
            lbw[i:i + 2] = ubw[i:i + 2] = np.asarray(fix_u0, float)[:2]
        w0 = self._w0 if self._w0 is not None else self._initial_guess(state5)
        sol = self.solver(x0=w0, p=self._p(state5, theta),
                          lbx=lbw, ubx=ubw, lbg=self._lbg, ubg=self._ubg)
        ok = self.solver.stats().get("success", False)
        w = np.array(sol["x"]).ravel()
        return dict(w=w, lam_g=np.array(sol["lam_g"]).ravel(),
                    value=float(sol["f"]), u0=w[self._nx:self._nx + 3], ok=ok)

    def _initial_guess(self, state5: np.ndarray) -> np.ndarray:
        X = np.tile(np.asarray(state5, float)[:, None], (1, self.N + 1))
        X[4] += np.arange(self.N + 1) * state5[3] * self.dt
        U = np.tile(np.array([0.0, 0.0, max(state5[3], 0.5)])[:, None], (1, self.N))
        return np.concatenate([X.ravel(order="F"), U.ravel(order="F"),
                               np.zeros(self._ns)])

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
                                       self._p(state5, theta))).ravel()

    def reset(self) -> None:
        self._w0 = None
