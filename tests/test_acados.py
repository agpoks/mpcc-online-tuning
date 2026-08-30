"""The acados backend, skipped when acados is not installed.

These are dimension and formulation checks, not performance ones: the timing
lives in ``benchmarks/solve_time.py`` where it can be reported as mean *and*
worst case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

acados = pytest.importorskip("acados_template", reason="acados not installed")

from mpcc_tuning.acados_ocp import build_ocp, pack_params  # noqa: E402
from mpcc_tuning.mpcc import WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


@pytest.fixture(scope="module")
def track():
    return Track.oval()


def test_ocp_dimensions(track):
    ocp = build_ocp(track, horizon=12, dt=0.15, max_obstacles=2)
    assert ocp.model.x.shape[0] == 5          # [x, y, psi, v, s]
    assert ocp.model.u.shape[0] == 3          # [delta, a, v_s]
    # theta (6) + reference point (3) + obstacles (3 each)
    assert ocp.model.p.shape[0] == len(WEIGHT_NAMES) + 3 + 3 * 2
    assert ocp.model.con_h_expr.shape[0] == 1 + 2      # corridor + keep-outs


def test_cost_is_external_not_least_squares(track):
    """The template's NLS encoding of the progress reward is wrong, so we differ.

    ``-q_v v_s dt`` is linear. Squared as a least-squares residual it becomes a
    quadratic *penalty* on progress. This asserts the formulation that avoids
    that, and the Hessian choice it forces.
    """
    ocp = build_ocp(track, horizon=12, dt=0.15)
    assert ocp.cost.cost_type == "EXTERNAL"
    assert ocp.cost.cost_type_e == "EXTERNAL"
    # EXTERNAL has no residual to build a Gauss-Newton Hessian from.
    assert ocp.solver_options.hessian_approx == "EXACT"


def test_progress_term_rewards_progress(track):
    """Differentiate the cost and check the sign, rather than trusting the text."""
    import casadi as ca

    ocp = build_ocp(track, horizon=12, dt=0.15)
    v_s = ocp.model.u[2]
    g = ca.Function("g", [ocp.model.x, ocp.model.u, ocp.model.p],
                    [ca.gradient(ocp.model.cost_expr_ext_cost, v_s)])
    p = np.zeros(ocp.model.p.shape[0])          # theta = 0 -> every weight 1
    d = float(g(np.zeros(5), np.zeros(3), p))
    assert d < 0, f"d(cost)/d(v_s) = {d:+.3f}; progress must lower the cost"


def test_inactive_obstacle_slot_is_exactly_inert(track):
    """``r_raw = -obs_margin`` must give exactly zero effective radius."""
    ocp = build_ocp(track, horizon=12, dt=0.15, max_obstacles=2, obs_margin=0.15)
    tail = ocp.parameter_values[-6:].reshape(2, 3)
    assert np.all(tail[:, 2] + 0.15 == 0.0)


def test_solves_and_respects_the_corridor(track, tmp_path):
    """One SQP_RTI step: it must succeed and return finite, in-bounds inputs."""
    from acados_template import AcadosOcpSolver
    from mpcc_tuning.model import ACCEL_MAX, STEER_MAX

    N, DT = 12, 0.15
    ocp = build_ocp(track, horizon=N, dt=DT, max_obstacles=1)
    ocp.code_export_directory = str(tmp_path / "gen")
    sol = AcadosOcpSolver(ocp, json_file=str(tmp_path / "gen" / "ocp.json"))

    p0, nxt = track.center[0], track.center[1]
    psi = float(np.arctan2(nxt[1] - p0[1], nxt[0] - p0[0]))
    x0 = np.array([p0[0], p0[1], psi, 1.5, 0.0])
    theta = np.log(np.array([1.0, 10.0, 2.0, 1.0, 0.01, 0.1]))
    s_nodes = x0[4] + np.arange(N + 1) * x0[3] * DT
    P = pack_params(theta, track=track, s_nodes=s_nodes, max_obstacles=1)
    for k in range(N + 1):
        sol.set(k, "p", P[k])
    sol.set(0, "lbx", x0)
    sol.set(0, "ubx", x0)
    assert sol.solve() == 0
    u = sol.get(0, "u")
    assert np.isfinite(u).all()
    assert abs(u[0]) <= STEER_MAX + 1e-6
    assert abs(u[1]) <= ACCEL_MAX + 1e-6
    assert u[2] >= -1e-6
