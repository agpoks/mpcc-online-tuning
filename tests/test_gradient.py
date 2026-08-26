"""The tests that decide whether the spike's premise holds.

Everything else here is engineering. These two are the claim: that the gradient
the tuner needs exists in closed form, and that it is the right one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.mpcc import MPCC, WEIGHT_NAMES, MPCCWeights
from mpcc_tuning.track import Track


@pytest.fixture(scope="module")
def setup():
    track = Track.oval()
    mpcc = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15)
    p = track.center[0]
    nxt = track.center[1]
    psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
    state = np.array([p[0], p[1], psi, 1.5, 0.0])
    return track, mpcc, state


def test_track_spline_is_finite_and_unit_speed(setup):
    """The path must be differentiable everywhere the solver can reach.

    Including well past the finish line: the progress variable is unbounded and
    runs off the end of the knots within one horizon if the wrap is missing,
    which shows up as NaN derivatives and an IPOPT failure with no useful
    message.
    """
    import casadi as ca

    track, _mpcc, _state = setup
    s = ca.MX.sym("s")
    jac = ca.Function("J", [s], [ca.jacobian(track.pos(s), s)])
    for sv in np.linspace(-3.0, track.length + 10.0, 300):
        d = np.array(jac(sv)).ravel()
        assert np.isfinite(d).all(), f"non-finite path derivative at s={sv}"
        assert abs(np.linalg.norm(d) - 1.0) < 0.05, f"not unit-speed at s={sv}"


def test_q_equals_v_at_the_policy_action(setup):
    """``Q(s, pi(s)) = V(s)`` -- the definition of the policy, and a solver check."""
    _track, mpcc, state = setup
    theta = MPCCWeights().to_log()
    v = mpcc.value(state, theta)
    assert v["ok"]
    q = mpcc.action_value(state, theta, v["u0"], v_out=None)
    assert q["ok"]
    assert abs(q["value"] - v["value"]) < 1e-4 * max(abs(v["value"]), 1.0)


def test_envelope_gradient_matches_finite_differences(setup):
    """The claim the whole approach rests on.

    ``d(optimal value)/d(theta)`` is the partial derivative of the Lagrangian at
    the primal-dual solution -- no differentiating through the solver. If this
    test fails, the tuner is following something that is not a gradient.
    """
    _track, mpcc, state = setup
    theta = MPCCWeights().to_log()
    sol = mpcc.value(state, theta)
    assert sol["ok"]
    analytic = mpcc.grad_theta(sol, state, theta)

    eps = 1e-4
    fd = np.empty(len(WEIGHT_NAMES))
    for i in range(len(WEIGHT_NAMES)):
        step = eps * np.eye(len(WEIGHT_NAMES))[i]
        up = mpcc.action_value(state, theta + step, sol["u0"])
        dn = mpcc.action_value(state, theta - step, sol["u0"])
        fd[i] = (up["value"] - dn["value"]) / (2 * eps)

    cos = float(analytic @ fd / (np.linalg.norm(analytic) * np.linalg.norm(fd) + 1e-12))
    rel = float(np.linalg.norm(analytic - fd) / (np.linalg.norm(fd) + 1e-12))
    assert cos > 0.999, f"cosine {cos:.4f}\nanalytic {analytic}\nfinite  {fd}"
    assert rel < 5e-2, f"relative error {rel:.4f}"


def test_weights_round_trip_through_logs():
    w = MPCCWeights(q_c=3.0, q_l=17.0, q_v=0.25, r_d=0.4, r_a=0.05, r_dv=2.0)
    back = MPCCWeights.from_log(w.to_log())
    for name in WEIGHT_NAMES:
        assert abs(getattr(back, name) - getattr(w, name)) < 1e-9


def test_tuner_moves_progress_weight_up_when_progress_pays(setup):
    """A directional sanity check on the sign convention.

    The MPCC minimises cost while the RL layer maximises return, so both the
    value *and its gradient* have to be negated. Applying one negation and not
    the other drives every weight the wrong way while looking exactly like a
    learning rate that is too high, so it is worth a test rather than a comment.
    """
    from mpcc_tuning.learner import QLambdaTuner

    track, mpcc, state = setup
    theta = MPCCWeights(q_v=0.05, q_l=200.0).to_log()
    tuner = QLambdaTuner(mpcc, len(theta), alpha=5e-3)
    model = KinematicBicycle(dt=0.05, grip=1.0)
    q_v_start = np.exp(theta[2])

    s5 = state.copy()
    u = tuner.start(theta, s5)
    for _ in range(40):
        x = model.step(s5[:4], np.asarray(u, float)[:2])
        prev = track.project(s5[0], s5[1])
        now = track.project(x[0], x[1])
        d = (now - prev) % track.length
        reward = d - track.length if d > track.length / 2 else d
        s5n = np.array([*x, s5[4] + float(u[2]) * 0.05])
        theta, u = tuner.step(theta, s5, reward, s5n, False)
        s5 = s5n
    assert np.exp(theta[2]) > q_v_start, (
        f"progress weight went {q_v_start:.4f} -> {np.exp(theta[2]):.4f} while being "
        "paid for progress -- check the sign convention in learner.step")
