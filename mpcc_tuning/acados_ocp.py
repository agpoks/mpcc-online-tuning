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
                               WHEELBASE)
from mpcc_tuning.mpcc import WEIGHT_NAMES


def build_ocp(track, horizon: int = 12, dt: float = 0.15,
              car_half_width: float = 0.12, max_obstacles: int = 0,
              obs_margin: float = 0.15, spline_mode: str = "parameter",
              obs_shape: str = "circle", car_half_length: float = 0.285,
              name: str = "mpcc_tuning"):
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
    model.name = name

    px, py, psi = ca.SX.sym("px"), ca.SX.sym("py"), ca.SX.sym("psi")
    v, s = ca.SX.sym("v"), ca.SX.sym("s")
    x = ca.vertcat(px, py, psi, v, s)
    delta, a, v_s = ca.SX.sym("delta"), ca.SX.sym("a"), ca.SX.sym("v_s")
    u = ca.vertcat(delta, a, v_s)
    xdot = ca.SX.sym("xdot", 5)

    # Same kinematic bicycle as mpcc_tuning/model.py, and the same progress
    # state. acados integrates it itself (ERK), where the CasADi problem does
    # RK4 by hand -- a difference to keep in mind when comparing solutions.
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
    model.p = ca.vertcat(*p_list)

    e_c = ca.sin(phi) * (px - ref_x) - ca.cos(phi) * (py - ref_y)
    e_l = -ca.cos(phi) * (px - ref_x) - ca.sin(phi) * (py - ref_y)
    q_c, q_l, q_v, r_d, r_a, r_dv = (ca.exp(theta[i]) for i in range(n_th))

    # EXTERNAL, so the linear progress term survives -- see the module note.
    stage = (q_c * e_c ** 2 + q_l * e_l ** 2 - q_v * v_s * dt
             + r_d * delta ** 2 + r_a * a ** 2 + r_dv * (v_s - v) ** 2)
    model.cost_expr_ext_cost = stage
    model.cost_expr_ext_cost_e = q_c * e_c ** 2 + q_l * e_l ** 2

    h = [e_c]                                  # corridor, as in the NLP
    for j in range(max_obstacles):
        ox, oy, r_raw = obs[stride * j], obs[stride * j + 1], obs[stride * j + 2]
        r_eff = r_raw + obs_margin             # inactive slot: r_raw = -margin
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
    model.con_h_expr = ca.vertcat(*h)
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
    ocp.constraints.x0 = np.zeros(5)
    ocp.constraints.idxbx = np.array([3], dtype=np.int64)      # speed
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([SPEED_MAX])
    ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)
    ocp.constraints.lbu = np.array([-STEER_MAX, -ACCEL_MAX, 0.0])
    ocp.constraints.ubu = np.array([STEER_MAX, ACCEL_MAX, SPEED_MAX])

    ocp.constraints.lh = np.array([-margin] + [0.0] * (nh - 1))
    ocp.constraints.uh = np.array([margin] + [1e8] * (nh - 1))
    # Soft, so "stay behind" stays a finite-cost option rather than an
    # infeasible solve -- the same argument as in mpcc_tuning/mpcc.py.
    ocp.constraints.idxsh = np.arange(nh, dtype=np.int64)
    ocp.constraints.lsh = np.zeros(nh)
    ocp.constraints.ush = np.zeros(nh)
    Z = np.full(nh, 200.0); z = np.full(nh, 5.0)
    Z[0], z[0] = 500.0, 10.0                   # corridor held harder
    ocp.cost.Zl = Z.copy(); ocp.cost.Zu = Z.copy()
    ocp.cost.zl = z.copy(); ocp.cost.zu = z.copy()

    n_p = n_th + (3 if spline_mode == "parameter" else 0) + stride * max_obstacles
    ocp.parameter_values = np.zeros(n_p)
    if max_obstacles:
        off = ([0.0, 0.0, -obs_margin] if obs_shape == "circle"
               else [0.0, 0.0, -obs_margin, 0.0])
        ocp.parameter_values[-stride * max_obstacles:] = np.tile(off, max_obstacles)

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 2
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
