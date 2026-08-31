"""The obstacle keep-out, and the thing it must not break.

The keep-out exists so that "overtake" and "stay behind" are different
problems. The tests that matter are therefore in two groups: that the
constraint is actually enforced, and that adding it did not quietly invalidate
the envelope-theorem gradient the whole repo rests on.
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
from mpcc_tuning.opponents import Opponent
from mpcc_tuning.track import Track


@pytest.fixture(scope="module")
def track():
    return Track.oval()


@pytest.fixture(scope="module")
def mpcc_obs(track):
    return MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
                max_obstacles=2)


def start_state(track, v=1.5):
    p, nxt = track.center[0], track.center[1]
    psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
    return np.array([p[0], p[1], psi, v, 0.0])


def test_zero_obstacles_changes_nothing(track):
    """``max_obstacles=0`` must be the problem that existed before.

    Every measured result in the repo was produced without this feature. If the
    default build gains a decision variable or a constraint row, none of them
    are comparable any more and ``rti.py``'s w-layout assumptions move.
    """
    base = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15)
    assert base._ns == 0
    assert base._nw == base._nx + 3 * base.N
    # dynamics + corridor + the grip-limited speed row per node. The grip
    # rows are a SEPARATE feature from obstacles and are on by default: with
    # SPEED_MAX raised from 4 to 8 m/s on the g-g analysis, they are what makes
    # 8 m/s a physical speed rather than a number the planner may aim at a
    # hairpin. The two changes belong together and neither is comparable with
    # benchmarks stored before them.
    # The grip rows are per STAGE (N of them, inside the stage loop); the
    # terminal speed row is the single extra one at node N.
    n_grip = base.N if base.speed_from_grip else 0
    n_term = 1 if base.terminal_speed else 0
    assert len(base._lbg) == 5 * (base.N + 1) + base.N + n_grip + n_term

    # The invariant this test is named for: obstacles add NOTHING when there
    # are none. Measured against the same build with room for one.
    one = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
               max_obstacles=1)
    assert one._ns == one.N and base._ns == 0
    assert len(one._lbg) - len(base._lbg) == one.N, \
        "an obstacle slot should cost exactly one keep-out row per stage"
    assert one._nw - base._nw == one.N, \
        "and exactly one slack per stage -- nothing else moves in the w-layout"
    assert base._p_sym.shape[0] == 5 + len(WEIGHT_NAMES)


def test_slack_and_parameter_dimensions(track, mpcc_obs):
    m = mpcc_obs
    assert m._ns == m.max_obstacles * m.N
    assert m._nw == m._nx + 3 * m.N + m._ns
    assert m._p_sym.shape[0] == 5 + len(WEIGHT_NAMES) + 3 * m.max_obstacles
    # The u0 slice must still mean u0 -- slacks go on the end for this reason.
    assert m._nx == 5 * (m.N + 1)


def test_inactive_slots_are_exactly_inert(mpcc_obs):
    """The template's trick: ``r_raw = -obs_margin`` makes ``r_eff`` exactly 0.

    Not approximately zero. If it were, the keep-out would be a small live
    constraint at the origin, which is on the oval's centreline.
    """
    mpcc_obs.set_obstacles([])
    q = mpcc_obs._obs_param().reshape(-1, 3)
    assert q.shape == (mpcc_obs.max_obstacles, 3)
    assert np.all(q[:, 2] + mpcc_obs.obs_margin == 0.0)


def test_too_many_obstacles_is_an_error(mpcc_obs):
    with pytest.raises(ValueError):
        mpcc_obs.set_obstacles([(0, 0, 0.2)] * (mpcc_obs.max_obstacles + 1))


def test_predicted_trajectory_avoids_the_keepout(track, mpcc_obs):
    """The constraint has to bind on the *plan*, not only on the outcome."""
    state = start_state(track)
    theta = MPCCWeights(q_c=0.3, q_v=2.0).to_log()
    ox, oy = np.array(track.pos(3.0)).ravel()
    r = 0.25
    mpcc_obs.reset()
    mpcc_obs.set_obstacles([(ox, oy, r)])
    sol = mpcc_obs.value(state, theta)
    assert sol["ok"]
    X = sol["w"][:mpcc_obs._nx].reshape(5, mpcc_obs.N + 1, order="F")
    # Stage 0 is pinned to x0 and deliberately not constrained; check 1..N.
    d = np.hypot(X[0, 1:] - ox, X[1, 1:] - oy)
    r_eff = r + mpcc_obs.obs_margin
    assert d.min() > r_eff - 1e-4, f"plan enters the keep-out: min d {d.min():.4f} vs {r_eff:.4f}"


def test_envelope_gradient_still_exact_with_an_obstacle(track):
    """The load-bearing one.

    Adding slacks put new decision variables in ``w`` and new rows in ``g``.
    The envelope theorem still applies -- theta enters neither -- but "still
    applies in principle" is exactly the kind of claim this repo checks against
    finite differences rather than asserts.
    """
    state = start_state(track)
    theta = MPCCWeights(q_c=1.0, q_v=2.0).to_log()
    # The keep-out must be INACTIVE here. At s=3.0 the plan sits exactly on the
    # boundary (min distance 0.400 m = r + obs_margin) and the check stops
    # meaning anything -- see the xfail below for the measurement.
    ox, oy = np.array(track.pos(12.0)).ravel()
    # A FRESH solver, not the module-scoped fixture: that one is warm-started
    # from whichever test ran before it, and a gradient check should not depend
    # on test ordering.
    mpcc_obs = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
                    max_obstacles=1, max_iter=300)
    mpcc_obs.reset()
    mpcc_obs.set_obstacles([(ox, oy, 0.25)])
    sol = mpcc_obs.value(state, theta)
    assert sol["ok"]
    analytic = mpcc_obs.grad_theta(sol, state, theta)

    eps = 1e-4
    fd = np.empty(len(WEIGHT_NAMES))
    for i in range(len(WEIGHT_NAMES)):
        step = eps * np.eye(len(WEIGHT_NAMES))[i]
        up = mpcc_obs.action_value(state, theta + step, sol["u0"])
        dn = mpcc_obs.action_value(state, theta - step, sol["u0"])
        fd[i] = (up["value"] - dn["value"]) / (2 * eps)

    cos = float(analytic @ fd / (np.linalg.norm(analytic) * np.linalg.norm(fd) + 1e-12))
    rel = float(np.linalg.norm(analytic - fd) / (np.linalg.norm(fd) + 1e-12))
    assert cos > 0.999, f"cosine {cos:.4f}\nanalytic {analytic}\nfinite  {fd}"
    assert rel < 5e-2, f"relative error {rel:.4f}"


def test_rti_solver_handles_the_obstacle_parameters(track, mpcc_obs):
    """``rti.py`` used to pack the parameter vector itself.

    With obstacles the vector grew a keep-out block, and a second place that
    knows the layout is a second place to get it wrong -- silently, because a
    short parameter vector is a CasADi error but a *mispacked* one is not.
    """
    from mpcc_tuning.rti import RTISolver

    state = start_state(track)
    theta = MPCCWeights(q_c=1.0, q_v=2.0).to_log()
    ox, oy = np.array(track.pos(3.0)).ravel()
    mpcc_obs.set_obstacles([(ox, oy, 0.25)])
    rti = RTISolver(mpcc_obs)
    out = rti.solve(state, theta)
    assert out["ok"]
    assert len(out["w"]) == mpcc_obs._nw
    assert np.isfinite(out["u0"]).all()


# -- opponents -------------------------------------------------------------

def test_opponent_offset_uses_laterals_sign_convention(track):
    """``Track.errors`` and ``Track.lateral`` disagree in sign.

    Off-track is judged by ``lateral``, so an opponent placed at ``offset=+0.3``
    must read as ``lateral=+0.3``. Getting this backwards puts the opponent on
    the wrong side of the track and nothing else complains.
    """
    for off in (0.0, 0.3, -0.3):
        x, y, _ = Opponent(track, s0=4.0, offset=off).pose()
        assert abs(track.lateral(x, y) - off) < 1e-3


def test_opponent_moves_at_its_own_speed(track):
    o = Opponent(track, s0=0.0, speed=2.0)
    for _ in range(10):
        o.step(0.05)
    assert abs(o.s - 1.0) < 1e-9


def test_plant_ends_the_episode_on_a_collision(track):
    """A collision must be terminal and must be distinguishable from a spin-off."""
    from examples.tune_online import Plant

    # An opponent parked on the start line: the ego car begins at v=1 m/s
    # pointing straight at it and cannot stop in time.
    opp = Opponent(track, s0=0.6, speed=0.0, radius=0.24)
    P = Plant(track, dt=0.05, max_steps=60, opponents=[opp])
    P.reset()
    assert P.failure is None
    off = False
    for _ in range(60):
        _s, r, off, tr = P.step(np.array([0.0, 1.0, 1.0]))
        if off or tr:
            break
    assert off and P.failure == "collision", f"failure={P.failure}"
    assert r < -4.0, "a collision should pay the same -5 as leaving the track"


def test_plant_without_opponents_is_unchanged(track):
    from examples.tune_online import Plant

    P = Plant(track, dt=0.05, max_steps=20)
    P.reset()
    assert P.keepouts() == []
    for _ in range(20):
        _s, _r, off, _tr = P.step(np.array([0.0, 0.5, 1.0]))
    assert not off and P.failure is None


# -- elliptical keep-out ---------------------------------------------------

def test_ellipse_semi_axes_beat_a_circle_where_it_matters(track):
    """A car is twice as long as it is wide, so a circle is wrong both ways.

    Alongside -- which is where overtaking happens -- a circle inscribing the
    car's length is far too conservative; nose-to-tail it is optimistic. The
    ellipse is the point of the exercise, so the asymmetry is asserted.
    """
    m = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
             max_obstacles=1, obs_shape="ellipse")
    r = 0.20
    a = r + m.obs_margin + m.car_half_length
    b = (r + m.obs_margin) * (m.car_half_width / m.car_half_length) + m.car_half_width
    assert a > 2.0 * b, f"semi-axes {a:.3f} along vs {b:.3f} across"
    circle = r + m.obs_margin
    assert b < circle < a, "the circle should sit between the two semi-axes"


def test_ellipse_orientation_is_used(track):
    """Rotating the opponent must move the constraint, or psi is being ignored."""
    import numpy as np
    m = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
             max_obstacles=1, obs_shape="ellipse")
    # Close, and big enough that the constraint actually binds. At 1.8 m with
    # r=0.2 the ellipse is inactive in every orientation, and the plans then
    # match for a trivial reason -- which is what the first version of this
    # test measured.
    ox, oy = np.array(track.pos(1.6)).ravel()
    state = start_state(track)
    theta = MPCCWeights(q_c=0.3, q_v=2.0).to_log()
    outs = []
    for psi_o in (0.0, np.pi / 2):
        m.reset()
        m.set_obstacles([(ox, oy, 0.55, psi_o)])
        sol = m.value(state, theta)
        assert sol["ok"]
        outs.append(sol["w"][:m._nx].reshape(5, m.N + 1, order="F")[:2].copy())
    assert np.abs(outs[0] - outs[1]).max() > 1e-4, \
        "the plan is identical for a 90-degree rotation: psi is not entering"


def test_circle_mode_is_unchanged_by_the_ellipse_work(track):
    """The default must still be the circle, with its three-number obstacles."""
    m = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
             max_obstacles=2)
    assert m.obs_shape == "circle" and m.obs_stride == 3
    assert m._p_sym.shape[0] == 5 + len(WEIGHT_NAMES) + 3 * 2


@pytest.mark.xfail(reason="open: the envelope gradient is not established while "
                          "the keep-out is ACTIVE, and the finite-difference "
                          "reference is not trustworthy there either",
                   strict=False)
def test_envelope_gradient_with_an_ACTIVE_keepout(track):
    """Recorded as a known limitation, not quietly dropped.

    Measured on the oval, obstacle r=0.25, sweeping how close it sits:

        obstacle    min dist    cos       rel      k_v analytic   k_v finite
        s=12.0      4.22 m      1.00000   0.0007      -1.5724       -1.5708
        s= 9.0      2.42 m      0.98977   0.1481      -0.9865       -2.9245
        s= 3.0      0.400 m     0.94597   0.3608      -0.0000       -4.5121
        s= 6.0      0.400 m    -0.90327   1.0033      -0.0000     +989.9226

    0.400 m is exactly r + obs_margin: the constraint is on its boundary in the
    last two rows. The claim holds cleanly while the keep-out is slack and
    degrades as it engages.

    What is NOT established is which side is wrong. A +989 finite difference is
    not the derivative of anything, so the reference is failing too -- a central
    difference straddles a kink in theta when the active set changes across the
    step. Whether the analytic gradient is also wrong there (degeneracy: the
    multiplier and dg/dk_v both vanish, so k_v drops out of the Lagrangian
    exactly when the constraint that contains it starts to bind) needs a
    one-sided check against a solve at a fixed active set, which is not written.

    Consequence to state plainly rather than work around: k_v -- the grip
    utilisation, the one weight with a known right answer sqrt(mu/mu_hat) --
    receives no reliable learning signal in exactly the situation the weight
    exists for, which is a close opponent in a corner.

    A second regime is unreliable and is NOT the same bug. Holding the keep-out
    inactive at s=12 and varying only the number of obstacle SLOTS:

        slots   unused   cos        rel
          1        0     1.000000   0.0007
          2        1     0.997815   0.0661
          3        2     1.000000   0.0002
          4        3     0.018385   1.0000

    Non-monotonic, so it is not the unused slots leaking theta into g -- the
    d_obs component reads 0.0 in the analytic AND the finite difference at every
    row, which is consistent. A cosine of 0.018 between two estimates of the
    same vector, with one of them a difference of solver outputs, is the
    signature of a solve that reports ok while sitting somewhere that is not a
    converged KKT point. That is the failure mode paper 1 documents, and
    diagnosing it needs the KKT residual rather than another gradient check.
    """
    import numpy as np
    m = MPCC(track, model=KinematicBicycle(dt=0.15), horizon=12, dt=0.15,
             max_obstacles=1, max_iter=300)
    state = start_state(track)
    theta = MPCCWeights(q_c=1.0, q_v=2.0).to_log()
    ox, oy = np.array(track.pos(3.0)).ravel()
    m.reset(); m.set_obstacles([(ox, oy, 0.25)])
    sol = m.value(state, theta)
    assert sol["ok"]
    X = np.array(sol["w"])[:m._nx].reshape(5, m.N + 1, order="F")
    assert np.hypot(X[0, 1:] - ox, X[1, 1:] - oy).min() < 0.41, "keep-out not active"
    analytic = m.grad_theta(sol, state, theta)
    eps = 1e-4
    fd = np.array([(m.action_value(state, theta + eps * np.eye(8)[i], sol["u0"])["value"]
                    - m.action_value(state, theta - eps * np.eye(8)[i], sol["u0"])["value"])
                   / (2 * eps) for i in range(8)])
    rel = float(np.linalg.norm(analytic - fd) / (np.linalg.norm(fd) + 1e-12))
    assert rel < 5e-2, f"relative error {rel:.4f} with the keep-out active"
