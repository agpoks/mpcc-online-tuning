"""The MPCC as an acados OCP, with the cost weights as runtime parameters.

``mpcc_tuning/rti.py`` already meets 20 Hz with CasADi's ``qrqp`` (1.9 ms mean,
3.4 ms worst at N=12). acados is faster, generates C, and is what the car will
run -- and it is what ``docs/source/influence_through_a_solver.md`` owes a
check: that repo's headline result was measured against a hand-rolled SQP, not
against a real ``SQP_RTI`` step.

Adapted from ``MPCC_planner_acados/scripts/generate_acados_solver.py``
-- **copied in, not imported**, per the standing rule. Two things in that
template are deliberately *not* copied, and both are formulation rather than
plumbing.

## 1. The progress reward is not a least-squares residual

``-q_v v_s dt`` is **linear** in the decision variables. The template encodes it
inside a ``NONLINEAR_LS`` cost as the residual ``-sqrt(w_progress) * p_prog``,
and a least-squares cost squares its residuals: that contributes
``+w_progress * p_prog**2``, a quadratic **penalty** on progress, not a linear
reward. With ``yref = 0`` and ``W = I`` as the generator sets them, it pushes
progress *down*. (It would behave as intended only if ``yref`` were driven to a
large negative value at runtime, which the generator does not do.)

So this uses ``cost_type = "EXTERNAL"`` and writes the cost out exactly. The
consequence to plan for is that EXTERNAL rules out a Gauss-Newton Hessian, so
the solver needs an exact one -- which is why ``hessian_approx`` differs from
the template's.

**This matters beyond tidiness.** The envelope theorem gives
``dJ*/dtheta = dL/dtheta`` at the solution. If the cost the solver minimises is
not the cost we differentiate, the gradient is exact for the wrong problem.

## 2. The path is a spline here, a stage parameter there

This repo carries the centreline as a periodic B-spline in the progress
variable, evaluated **inside** the NLP, so the solver chooses its own reference
point -- that is what makes it contouring control rather than trajectory
tracking. The template instead passes ``ref_x``, ``ref_y``, ``t_angle`` as
per-stage parameters sampled outside the solver.

That difference is not cosmetic either: ``s`` stops being coupled to the path
within a solve, and that coupling is the mechanism the envelope gradient runs
through. ``spline_mode`` selects which formulation is built, so the two can be
*measured* against each other rather than argued about.

## What is copied verbatim, because it is right

The circular keep-out and its slack form: ``r_eff = r_raw + obs_margin``,
``dist2 - r_eff**2 >= 0``, inactive slots passed as ``r_raw = -obs_margin`` so
``r_eff`` is exactly zero without a ``max()``; softened through ``idxsh`` with
``Zl``/``zl``. ``mpcc_tuning/mpcc.py`` already carries the same formulation
written out as explicit slacks, so this direction is a straight swap.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.model import (ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX,
                               WHEELBASE, DynamicBicycle, _smax, _sabs)
from mpcc_tuning.mpcc import WEIGHT_NAMES


def build_ocp(track, horizon: int = 12, dt: float = 0.15,
              a_lat_grip: float = 6.0 * 1.0,
              soft_corridor: bool = True,
              lin_corridor: bool = False,
              car_half_width: float = 0.12, max_obstacles: int = 0,
              obs_margin: float = 0.15, spline_mode: str = "parameter",
              obs_shape: str = "circle", car_half_length: float = 0.285,
              name: str = "mpcc_tuning", vehicle: str = "kinematic",
              q_friction: float = 50.0, q_slip: float = 50.0,
              vy_soft: float = 0.5, friction_peak: float = 24.29,
              friction_peak_long: float = 23.186):
    """The MPCC as an :class:`AcadosOcp`.

    ``spline_mode``
        ``"parameter"`` passes the reference point per stage, as the template
        does. ``"spline"`` code-generates the B-spline into the model so ``s``
        keeps its differentiable coupling to the path. The second is what this
        repo's CasADi problem does; the first is what generates cleanly.
    """
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    ocp = AcadosOcp()
    model = AcadosModel()
    # Sanitised HERE, at the point the name reaches CasADi, rather than trusting
    # every caller: an invalid identifier fails the build with an SXFunction
    # error that never mentions naming, and it takes a whole sweep with it.
    from mpcc_tuning.acados_variants import sanitize_name
    model.name = sanitize_name(name, fallback="mpcc")

    dyn = vehicle == "dynamic"
    px, py, psi = ca.SX.sym("px"), ca.SX.sym("py"), ca.SX.sym("psi")
    v, s = ca.SX.sym("v"), ca.SX.sym("s")
    delta, a, v_s = ca.SX.sym("delta"), ca.SX.sym("a"), ca.SX.sym("v_s")
    u = ca.vertcat(delta, a, v_s)
    vy = r_yaw = None
    if dyn:
        # Same layout as mpcc_tuning/mpcc.py: [x, y, psi, vx, s, vy, r], with
        # s at index 4, so every expression below is shared between the two
        # backends unchanged.
        # n_dyn extra states, appended after s: vy, r, and delta when the
        # model carries the steering servo. Reading it off the model rather
        # than hardcoding 7 keeps the two backends from diverging the moment
        # a state is added on one side.
        _m = DynamicBicycle(dt=dt)
        n_dyn = _m.n_dyn
        vy, r_yaw = ca.SX.sym("vy"), ca.SX.sym("r_yaw")
        extra = [vy, r_yaw]
        if n_dyn > 2:
            extra.append(ca.SX.sym("delta_state"))
        x = ca.vertcat(px, py, psi, v, s, *extra)
        xdot = ca.SX.sym("xdot", 5 + n_dyn)
        # ONE definition of the tyre dynamics, imported rather than retyped.
        # A second copy here is how the acados solver and the CasADi solver
        # silently stop being the same controller.
        fd = _m.f_sym(ca.vertcat(x[0:4], x[5:]), u[0:2])
        f = ca.vertcat(fd[0:4], v_s, fd[4:])
    else:
        x = ca.vertcat(px, py, psi, v, s)
        xdot = ca.SX.sym("xdot", 5)
        # Same kinematic bicycle as mpcc_tuning/model.py, and the same progress
        # state. acados integrates it itself (ERK), where the CasADi problem
        # does RK4 by hand -- a difference to keep in mind when comparing.
        f = ca.vertcat(v * ca.cos(psi), v * ca.sin(psi),
                       v / WHEELBASE * ca.tan(delta), a - DRAG * v, v_s)
    model.x, model.u, model.xdot = x, u, xdot
    model.f_expl_expr = f
    model.f_impl_expr = xdot - f

    n_th = len(WEIGHT_NAMES)
    theta = ca.SX.sym("theta", n_th)          # the learnable log weights
    if spline_mode == "parameter":
        ref = ca.SX.sym("ref", 3)             # ref_x, ref_y, path heading
        ref_x, ref_y, phi = ref[0], ref[1], ref[2]
        p_list = [theta, ref]
    else:
        p_list = [theta]
        pos = track.pos(s)
        ref_x, ref_y = pos[0], pos[1]
        phi = track.tangent_angle(s)
    stride = 3 if obs_shape == "circle" else 4
    obs = ca.SX.sym("obs", stride * max_obstacles) if max_obstacles else None
    if obs is not None:
        p_list.append(obs)
    _p_pre = list(p_list)

    e_c = ca.sin(phi) * (px - ref_x) - ca.cos(phi) * (py - ref_y)
    e_l = -ca.cos(phi) * (px - ref_x) - ca.sin(phi) * (py - ref_y)
    # All EIGHT, not six. theta grew d_obs and k_v and this line was never
    # updated, so build_ocp raised "too many values to unpack (expected 6)"
    # and the acados backend has not built since. Unpack by NAME off
    # WEIGHT_NAMES, so the next weight added is a KeyError rather than silence.
    _w = {n: ca.exp(theta[i]) for i, n in enumerate(WEIGHT_NAMES)}
    k_v = _w["k_v"]
    q_c, q_l, q_v = _w["q_c"], _w["q_l"], _w["q_v"]
    r_d, r_a, r_dv = _w["r_d"], _w["r_a"], _w["r_dv"]
    d_obs = _w["d_obs"]

    # EXTERNAL, so the linear progress term survives -- see the module note.
    stage = (q_c * e_c ** 2 + q_l * e_l ** 2 - q_v * v_s * dt
             + r_d * delta ** 2 + r_a * a ** 2 + r_dv * (v_s - v) ** 2)
    if dyn and (q_friction > 0.0 or q_slip > 0.0):
        # The same two cost terms mpcc_tuning/mpcc.py adds, so the exported
        # controller optimises the objective that was actually tuned. Both use
        # fabs; with hessian_approx EXACT that is fine, and it is one reason
        # EXACT is not negotiable here -- see the acados section in TODO.md.
        Frx, _Ffy, Fry = _m.tyre_sym(ca.vertcat(x[0:4], x[5:]), u[0:2])
        comb = (Frx / friction_peak_long) ** 2 + (Fry / friction_peak) ** 2
        exc = _smax(comb - 1.0)
        vy_exc = _smax(_sabs(vy) - vy_soft)
        stage = stage + q_friction * exc ** 2 + q_slip * vy_exc ** 2
    model.cost_expr_ext_cost = stage
    model.cost_expr_ext_cost_e = q_c * e_c ** 2 + q_l * e_l ** 2

    # Corridor as a LINEAR half-space pair, not a nonlinear row.
    #
    #   track geometry -> local convex corridor -> linear half-spaces -> HPIPM
    #
    # e_c = sin(phi)(px - ref_x) - cos(phi)(py - ref_y) is nonlinear in the
    # decision variables when phi and ref come from the spline AT THE STATE s,
    # so HPIPM only ever sees a linearisation of it and can leave the corridor
    # between linearisation points. Passing the normal and the reference point
    # as per-stage PARAMETERS makes the same quantity affine in (px, py):
    # exactly representable in the QP, and convex by construction.
    #
    # The cost keeps the spline, so s stays coupled to the path and the
    # envelope gradient still runs through it. Only the CONSTRAINT is
    # linearised -- which is the part HPIPM has to satisfy exactly.
    if lin_corridor:
        cor = ca.SX.sym("cor", 4)              # nx, ny, ref_x, ref_y
        p_list.append(cor)
        e_c_lin = cor[0] * (px - cor[2]) + cor[1] * (py - cor[3])
        h = [e_c_lin]
    else:
        h = [e_c]                              # corridor, as in the NLP
    # The grip row, which was in mpcc_tuning/mpcc.py and NOT here. Porting the
    # cost across without the constraints is why the acados controller drove
    # differently from the CasADi one for the same model: measured on the oval,
    # IPOPT kinematic peaks at 4.20 m/s with this row and completes 7.16 laps;
    # acados without it peaked at 4.69 and left the track inside a third of a
    # lap. The row is what keeps the car off its own limit.
    #
    # Only in spline mode, where curvature is a function of s inside the NLP.
    # In parameter mode kappa would have to be sampled outside and frozen,
    # which is the same decoupling that broke the reference point.
    # Same virtual speed coupling as mpcc_tuning/mpcc.py, and as the working
    # controller this OCP was adapted from. Without it the progress variable
    # outruns the car and the reference is sampled where the car is not.
    h.append(v + 0.5 - v_s)

    kap = None
    if spline_mode == "spline" and not dyn:
        kap = ca.fabs(track.curvature_sym(track.wrap(s)))
        h.append(a_lat_grip - v ** 2 * kap / (k_v ** 2 + 1e-9))
    for j in range(max_obstacles):
        ox, oy, r_raw = obs[stride * j], obs[stride * j + 1], obs[stride * j + 2]
        r_eff = r_raw + d_obs                  # inactive slot: r_raw = -d_obs
        dx, dy = px - ox, py - oy
        if obs_shape == "circle":
            h.append(dx ** 2 + dy ** 2 - r_eff ** 2)
        else:
            # Same ellipse as mpcc_tuning/mpcc.py: in the opponent's frame,
            # (u/a)^2 + (v/b)^2 >= 1, scaled by a*b so the row keeps the units
            # of the circular one and the slack weights stay comparable.
            psi_o = obs[stride * j + 3]
            cu, su = ca.cos(psi_o), ca.sin(psi_o)
            u_ax = cu * dx + su * dy
            v_ax = -su * dx + cu * dy
            a_e = r_eff + car_half_length
            b_e = r_eff * (car_half_width / max(car_half_length, 1e-9)) + car_half_width
            h.append(((u_ax / a_e) ** 2 + (v_ax / b_e) ** 2 - 1.0) * (a_e * b_e))
    model.p = ca.vertcat(*p_list)
    model.con_h_expr = ca.vertcat(*h)
    # TERMINAL corridor (and grip), which acados did not have at all.
    #
    # mpcc_tuning/mpcc.py constrains the corridor on X[:,k+1] for k in
    # range(N), so stages 1..N are covered INCLUDING the last one. acados sets
    # con_h_expr for stages 1..N-1 only; with no con_h_expr_e the terminal
    # state is free, so the solver may plan a horizon that ENDS outside the
    # track -- and at N=12, dt=0.05 that is only 0.6 s of lookahead, so it then
    # drives there. This is the remaining reason acados reached 2.16 laps where
    # the same problem in IPOPT reached 5.12.
    #
    # Only the rows that depend on x alone: the v_s coupling needs a control
    # and there is no control at the terminal node.
    h_e = [e_c_lin if lin_corridor else e_c]
    if spline_mode == "spline" and not dyn:
        h_e.append(a_lat_grip - v ** 2 * kap / (k_v ** 2 + 1e-9))
    model.con_h_expr_e = ca.vertcat(*h_e)
    nh_e = len(h_e)
    nh = len(h)

    ocp.model = model
    # Both spellings: acados moved this from ocp.dims.N to
    # solver_options.N_horizon, and setting only the new one fails on older
    # installs with an unhelpful "unsupported operand /: float and NoneType".
    # Only set N_horizon if this acados already has it. Assigning to a plain
    # Python object always succeeds, so a try/except does not detect the old
    # layout -- it silently *creates* the field, and code generation then dies
    # with "field 'N_horizon' is not in layout but in OCP description".
    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = horizon
    ocp.dims.N = horizon
    ocp.solver_options.tf = dt * horizon
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    margin = track.half_width - car_half_width
    ocp.constraints.x0 = np.zeros(x.shape[0])
    if not dyn:
        ocp.constraints.idxbx = np.array([3], dtype=np.int64)  # speed
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([SPEED_MAX])
    # else: no hard velocity bound. "Euler dynamics + hard velocity bounds
    # cause infeasibility near the limits" -- MPCC_controller_ipopt. The same
    # applies to acados' ERK, and the envelope is clamped in the plant.
    ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)
    ocp.constraints.lbu = np.array([-STEER_MAX, -ACCEL_MAX, 0.0])
    ocp.constraints.ubu = np.array([STEER_MAX, ACCEL_MAX, SPEED_MAX])

    ocp.constraints.lh = np.array([-margin] + [0.0] * (nh - 1))
    ocp.constraints.uh = np.array([margin] + [1e8] * (nh - 1))
    # Soft, so "stay behind" stays a finite-cost option rather than an
    # infeasible solve -- the same argument as in mpcc_tuning/mpcc.py.
    # WHICH rows are soft matters, and this softened all of them.
    #
    # mpcc_tuning/mpcc.py holds the corridor HARD -- `wl - chw - e_c >= 0` with
    # lbg = 0 -- and softens only the keep-out and grip rows. Here `idxsh =
    # arange(nh)` put row 0, the corridor, in the soft set too, so the acados
    # car could buy its way off the track: leaving costs Zl and the progress
    # reward pays more. That is why it left the corridor inside a quarter lap
    # while the CasADi controller on the same cost drove 7.16 laps.
    #
    # MEASURED: holding the corridor hard here is WORSE, not better. Solve
    # rate collapsed to 27% (kinematic) and 2% (dynamic) and the car stopped
    # turning at all. The same hard row is fine in the CasADi problem, which
    # says the two are not equivalent in a way still not understood -- so the
    # default stays soft, and the flag exists to re-test rather than to argue.
    soft = (np.arange(nh, dtype=np.int64) if soft_corridor
            else np.arange(1, nh, dtype=np.int64))
    ocp.constraints.idxsh = soft
    ocp.constraints.lsh = np.zeros(len(soft))
    ocp.constraints.ush = np.zeros(len(soft))
    Z = np.full(nh, 200.0); z = np.full(nh, 5.0)
    Z[0], z[0] = 500.0, 10.0                   # corridor held harder
    Z, z = Z[soft], z[soft]
    ocp.cost.Zl = Z.copy(); ocp.cost.Zu = Z.copy()
    ocp.cost.zl = z.copy(); ocp.cost.zu = z.copy()
    # Same rows at the terminal node, softened the same way.
    ocp.constraints.lh_e = np.array([-margin] + [0.0] * (nh_e - 1))
    ocp.constraints.uh_e = np.array([margin] + [1e8] * (nh_e - 1))
    ocp.constraints.idxsh_e = np.arange(nh_e, dtype=np.int64)
    ocp.constraints.lsh_e = np.zeros(nh_e)
    ocp.constraints.ush_e = np.zeros(nh_e)
    Ze = np.full(nh_e, 200.0); ze = np.full(nh_e, 5.0)
    Ze[0], ze[0] = 500.0, 10.0
    ocp.cost.Zl_e = Ze.copy(); ocp.cost.Zu_e = Ze.copy()
    ocp.cost.zl_e = ze.copy(); ocp.cost.zu_e = ze.copy()

    n_p = (n_th + (3 if spline_mode == "parameter" else 0)
           + (4 if lin_corridor else 0) + stride * max_obstacles)
    ocp.parameter_values = np.zeros(n_p)
    if max_obstacles:
        off = ([0.0, 0.0, -obs_margin] if obs_shape == "circle"
               else [0.0, 0.0, -obs_margin, 0.0])
        ocp.parameter_values[-stride * max_obstacles:] = np.tile(off, max_obstacles)

    # cost_scaling = 1 everywhere, NOT the newer acados default.
    #
    # From acados v0.5.x, cost_scaling defaults to [*time_steps, 1.0]: every
    # STAGE cost is multiplied by dt while the TERMINAL cost keeps 1.0, making
    # the terminal 1/dt = 20x stronger relative to the stages. The terminal
    # cost here is pure tracking (q_c e_c^2 + q_l e_l^2) with no progress
    # reward, so under that scaling the solver stops caring about progress and
    # the car crawls. Measured, same code, same weights:
    #
    #     acados v0.1.9 (no scaling)      3.40 laps, 77% solve, peak 4.11 m/s
    #     acados v0.5.3 (new default)     0.10 laps, 30% solve, peak 1.48 m/s
    #     acados v0.5.3, scaling = ones   1.44 laps, 83% solve, peak 4.15 m/s
    #
    # Pinning it keeps the objective the same object across acados versions,
    # which is the only way a cross-version comparison means anything.
    if hasattr(ocp.solver_options, "cost_scaling"):
        ocp.solver_options.cost_scaling = np.ones(horizon + 1)
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    # 2 steps is enough for the kinematic model and NOT for the dynamic one.
    # ERK4 with num_steps = 2 at dt = 0.05 integrates with h = 25 ms, and this
    # car's yaw mode has a 17.8 ms time constant -- measured, r_dot = 77.5
    # rad/s^2 from rest toward an equilibrium of 1.378. The CasADi backend
    # showed the same thing directly: h = 25 ms reaches r = 1.007 where the
    # plant reaches 1.390, a 28% error, and h = 12.5 ms reaches 1.366.
    #
    # Under-resolved dynamics give wrong sensitivities, and wrong sensitivities
    # are what HPIPM reported as QP status 3 (NaN) on 147 solves.
    ocp.solver_options.sim_method_num_steps = 4 if dyn else 2
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # EXACT, not GAUSS_NEWTON: the template's Gauss-Newton choice goes with its
    # NONLINEAR_LS cost, and an EXTERNAL cost has no residual to build one from.
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.print_level = 0
    return ocp


def pack_params(theta, track=None, s_nodes=None, obstacles=(), max_obstacles=0,
                obs_margin: float = 0.15, spline_mode: str = "parameter",
                obs_shape: str = "circle"):
    """Per-stage parameter vectors: ``[theta | ref (3) | obstacles (3M)]``.

    In ``parameter`` mode the reference point is sampled at the *predicted*
    progress of each node, which is where the approximation enters: the solver
    can no longer move its own reference within a solve.
    """
    theta = np.asarray(theta, float)
    off = ([0.0, 0.0, -obs_margin] if obs_shape == "circle"
           else [0.0, 0.0, -obs_margin, 0.0])
    obs = np.tile(off, (max_obstacles, 1))
    for i, o in enumerate(list(obstacles)[:max_obstacles]):
        obs[i] = o
    obs = obs.ravel()
    out = []
    for s in (s_nodes if s_nodes is not None else [0.0]):
        parts = [theta]
        if spline_mode == "parameter":
            p = np.array(track.pos(float(s))).ravel()
            parts.append(np.array([p[0], p[1], float(track.tangent_angle(float(s)))]))
        if max_obstacles:
            parts.append(obs)
        out.append(np.concatenate(parts))
    return np.array(out)
