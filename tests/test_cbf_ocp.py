"""The barrier as a constraint of the OCP, not an override applied after it.

Why in the solver at all, given a filter already exists: when the cost weights
are what is being *learned*, anything expressed as a cost term is negotiable.
The policy can learn a small weight for it and trade it away -- which is the
failure this repo documents for a proxy optimised past the point where the
proxy is valid, applied to safety. A constraint cannot be learned away.

It also removes a mismatch: with an external filter the learner differentiates
the unfiltered problem while the plant executes the filtered action, so the
value being learned is not the value being executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


@pytest.fixture(scope="module")
def track():
    return Track.oval()


def start_state(track, v=1.5):
    p, nxt = track.center[0], track.center[1]
    psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
    return np.array([p[0], p[1], psi, v, 0.0])


def test_off_by_default_and_costs_nothing(track):
    """Every stored result predates this. The default build must be unchanged."""
    base = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05)
    assert base.cbf is False
    on = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, cbf=True)
    assert len(on._lbg) - len(base._lbg) == on.N, "one barrier row per stage"
    assert on._nw - base._nw == on.N, "one slack per stage, and nothing else"
    assert on._ns == on.N and base._ns == 0


def test_alpha_is_validated(track):
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="cbf_alpha"):
            MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
                 cbf=True, cbf_alpha=bad)


def test_the_barrier_is_the_filters_barrier(track):
    """Same criterion as filters/cbf_qp.py, or the comparison means nothing.

    If the in-solver barrier drifts from the post-hoc one, "the constraint made
    the filter idle" stops being a statement about where safety is enforced and
    becomes one about two different definitions of safe.
    """
    import casadi as ca
    from mpcc_tuning.filters.cbf_qp import CBFQP

    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             cbf=True)
    f = CBFQP(track, alpha=m.cbf_alpha, margin=m.cbf_margin,
              lookahead=m.cbf_lookahead, h_kind="braking")
    Xs = ca.MX.sym("X", 5)
    h_sym = ca.Function("h", [Xs], [m._barrier_sym(Xs)])

    rng = np.random.default_rng(0)
    for _ in range(12):
        s = float(rng.uniform(0.0, track.length))
        p = np.array(track.pos(s)).ravel()
        d = float(rng.uniform(-0.4, 0.4))
        phi = float(track.tangent_angle(s))
        x = p[0] + d * np.sin(phi)
        y = p[1] - d * np.cos(phi)
        psi = phi + float(rng.uniform(-0.3, 0.3))
        v = float(rng.uniform(0.5, 4.0))
        mine = float(h_sym(ca.DM([x, y, psi, v, s])))
        theirs = f.barrier(x, y, psi, v)
        assert abs(mine - theirs) < 2e-2, (
            f"in-solver barrier {mine:.4f} vs filter {theirs:.4f}")


def test_the_envelope_gradient_survives_the_barrier(track):
    """theta must stay out of g, or the gradient stops being free.

    The barrier margin is deliberately a constant. k_v enters the grip row and
    its analytic gradient collapses to zero once that row is active while
    finite differences read -4.5 (tests/test_obstacles.py, the xfail); a
    learnable barrier margin would add another such term in that same regime.
    """
    m = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
             max_iter=300, cbf=True)
    state = start_state(track)
    theta = MPCCWeights().to_log()
    m.reset()
    sol = m.value(state, theta)
    assert sol["ok"]
    analytic = m.grad_theta(sol, state, theta)
    eps = 1e-4
    n = len(theta)
    fd = np.array([(m.action_value(state, theta + eps * np.eye(n)[i], sol["u0"])["value"]
                    - m.action_value(state, theta - eps * np.eye(n)[i], sol["u0"])["value"])
                   / (2 * eps) for i in range(n)])
    cos = float(analytic @ fd / (np.linalg.norm(analytic) * np.linalg.norm(fd) + 1e-12))
    assert cos > 0.999, f"cosine {cos:.4f} with the barrier present"
