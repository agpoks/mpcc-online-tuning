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
    assert len(base._lbg) == 5 * (base.N + 1) + base.N   # dynamics + corridor
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


def test_envelope_gradient_still_exact_with_an_obstacle(track, mpcc_obs):
    """The load-bearing one.

    Adding slacks put new decision variables in ``w`` and new rows in ``g``.
    The envelope theorem still applies -- theta enters neither -- but "still
    applies in principle" is exactly the kind of claim this repo checks against
    finite differences rather than asserts.
    """
    state = start_state(track)
    theta = MPCCWeights(q_c=1.0, q_v=2.0).to_log()
    ox, oy = np.array(track.pos(3.0)).ravel()
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
