"""The behaviour policy: features, cell, policy, and the learning rule.

These test that the machinery is *correct*. Whether a trained policy actually
*uses* a feature is a different question and a measurement, not a unit test --
see ``experiments/feature_sensitivity.py``. Conflating the two is easy: having
the sector in the input vector is not the same as learning to condition on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.ltc import (AGGRESSION, LTCCell, MLPCell, N_FEATURES,  # noqa: E402
                             OPPONENT_CLASSES, POSTURES, THETA_HI, THETA_LO,
                             WeightPolicy, behaviour_theta, classify_opponent,
                             features, posture_theta)
from mpcc_tuning.mpcc import MPCCWeights, WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.opponents import ObstacleTracker, Opponent  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

TH0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()


@pytest.fixture(scope="module")
def circuit():
    return Track.circuit()


# -- features --------------------------------------------------------------

def test_feature_vector_has_the_advertised_length(circuit):
    o = Opponent(circuit, s0=5.0, speed=1.0)
    f = features(circuit, np.array([0.0, 0.0, 0.0, 2.0, 1.0]), [o], opp_speed_est=1.0)
    assert len(f) == N_FEATURES == 18
    assert np.isfinite(f).all()


def test_sector_one_hot_tracks_the_lap(circuit):
    """Features 9-12 must be a one-hot that changes with the sector ahead."""
    o = Opponent(circuit, s0=5.0, speed=1.0)
    seen = set()
    for s in np.linspace(0.0, circuit.length, 60, endpoint=False):
        p = np.array(circuit.pos(float(s))).ravel()
        f = features(circuit, np.array([p[0], p[1], 0.0, 2.0, s]), [o],
                     opp_speed_est=1.0)
        oh = f[9:13]
        assert oh.sum() == 1.0, f"sector one-hot is not one-hot at s={s:.1f}: {oh}"
        seen.add(int(np.argmax(oh)))
    assert len(seen) >= 3, f"only {len(seen)} sector types seen round the lap"


def test_no_opponent_is_a_state_not_a_missing_value(circuit):
    """Absent opponent must saturate the gap features, not read as 'right here'."""
    s5 = np.array([0.0, 0.0, 0.0, 2.0, 1.0])
    f = features(circuit, s5, [], opp_speed_est=None)
    assert f[5] == 1.0 and f[8] == 1.0, "gap features must saturate with no opponent"
    assert f[14:18].sum() == 0.0, "no opponent means no class is set"


def test_opponent_classes_are_graded_by_relative_speed():
    for v_opp, want in ((0.0, "static"), (1.0, "slower"), (2.0, "equal"), (3.2, "faster")):
        got = OPPONENT_CLASSES[classify_opponent(2.0, v_opp)]
        assert got == want, f"ego 2.0 vs {v_opp} -> {got}, expected {want}"


def test_tracker_estimates_motion_without_being_told(circuit):
    """The class must come from observation, not from Opponent.speed."""
    for speed, dynamic in ((1.2, True), (0.0, False)):
        o, tr = Opponent(circuit, s0=6.0, speed=speed), ObstacleTracker(dt=0.05)
        for _ in range(12):
            tr.update(o.pose()[:2])
            o.step(0.05)
        assert bool(tr.is_dynamic) == dynamic, f"speed {speed} -> is_dynamic {tr.is_dynamic}"


# -- behaviours ------------------------------------------------------------

def test_aggression_never_crosses_the_behaviour_boundary():
    """Crossing q_v/q_c = 1 is a change of behaviour, not of intensity."""
    for name in ("follow", "overtake"):
        for aggr in AGGRESSION:
            w = np.exp(behaviour_theta(name, aggr, TH0))
            ratio = w[2] / w[0]
            if name == "overtake":
                assert ratio > 1.0, f"overtake/{aggr} at ratio {ratio:.2f} would follow"
            else:
                assert ratio < 1.0, f"follow/{aggr} at ratio {ratio:.2f} would pass"


def test_every_behaviour_cell_is_distinct():
    """Aggression must do something at every level, including under the ceiling."""
    seen = {round(float(np.exp(behaviour_theta(n, a, TH0))[2]
                        / np.exp(behaviour_theta(n, a, TH0))[0]), 4)
            for n in ("follow", "overtake") for a in AGGRESSION}
    assert len(seen) == 6, f"only {len(seen)} distinct ratios among 6 cells"


def test_stay_behind_goes_around_a_static_obstacle():
    """Behind something that is not going anywhere, following is stopping."""
    f = np.zeros(N_FEATURES)
    f[8] = 0.3                                   # close
    f[7] = 0.6                                   # pass available
    for dynamic, should_pass in ((True, False), (False, True)):
        w = np.exp(posture_theta("stay_behind", f, "neutral", TH0,
                                 is_dynamic=dynamic))
        passes = bool((w[2] / w[0]) > 1.0)   # bool(): np.False_ is not False
        assert passes == should_pass, \
            f"dynamic={dynamic}: ratio {w[2]/w[0]:.2f}"


def test_fixed_schedule_never_attacks_a_faster_car():
    """You are being caught, not catching."""
    from mpcc_tuning.ltc import fixed_schedule
    for ci, name in enumerate(OPPONENT_CLASSES):
        f = np.zeros(N_FEATURES)
        f[8], f[7], f[14 + ci] = 0.3, 0.6, 1.0
        w = np.exp(fixed_schedule(f, TH0))
        passes = bool((w[2] / w[0]) > 1.0)
        assert passes == (name != "faster"), f"{name}: ratio {w[2]/w[0]:.2f}"


# -- the policy and its gradient -------------------------------------------

def test_policy_output_stays_strictly_inside_the_box():
    """A tanh squash, not a clip: a clip has no gradient at the bound."""
    cell = LTCCell(N_FEATURES, 12, seed=0)
    pol = WeightPolicy(cell, TH0, THETA_LO, THETA_HI, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        th = pol.step(rng.normal(0.0, 3.0, N_FEATURES))   # deliberately extreme
        assert (th > THETA_LO).all() and (th < THETA_HI).all()


def test_gradient_survives_at_the_bound():
    """The bug this replaced: a hard clip zeroed dtheta/dphi at the ceiling."""
    cell = LTCCell(N_FEATURES, 12, seed=0)
    pol = WeightPolicy(cell, THETA_HI - 1e-6, THETA_LO, THETA_HI, seed=0)
    for _ in range(10):
        pol.step(np.full(N_FEATURES, 5.0))            # drive it to the ceiling
    dG, dcell = pol.grads(np.ones(len(WEIGHT_NAMES)))
    assert np.abs(dG).sum() > 0.0, "no gradient for G at the bound"
    assert np.abs(dcell).sum() > 0.0, "no gradient for the cell at the bound"


def test_ltc_leak_is_below_one_by_construction():
    """The influence series converges without an arbitrary cap."""
    cell = LTCCell(N_FEATURES, 16, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        _h, _imm, leak = cell.step(rng.normal(0.0, 2.0, N_FEATURES))
        assert (leak < 1.0).all(), f"leak reached {leak.max()}"


def test_exact_rtrl_and_rflo_agree_in_direction():
    """If they disagreed, the approximation would be the suspect for the runaway."""
    feats = np.random.default_rng(3).normal(0.0, 0.6, (60, N_FEATURES))
    acc = {}
    for kind in ("rflo", "exact"):
        pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=2), TH0, THETA_LO,
                           THETA_HI, seed=2, influence=kind)
        tot = 0.0
        for f in feats:
            pol.step(f)
            dG, dc = pol.grads(np.ones(len(WEIGHT_NAMES)))
            tot = tot + np.concatenate([dG.ravel(), dc.ravel()])
        acc[kind] = tot
    a, b = acc["rflo"], acc["exact"]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"cosine {cos:.5f}"


def test_mlp_carries_no_influence():
    """The memoryless control must have zero leak, or it is not memoryless."""
    cell = MLPCell(N_FEATURES, 12, seed=0)
    _h, _imm, leak = cell.step(np.zeros(N_FEATURES))
    assert (leak == 0.0).all()


def test_theta0_must_be_strictly_inside_the_box():
    """An anchor on a bound has zero gradient on that side -- refuse to build.

    This shipped: q_l and q_v were anchored exactly on their ceilings, so the
    two weights that carry behaviour and safety could only ever be revised
    downward, and the policy gradient through them was identically zero
    whenever the pre-activation was non-negative. The failure looks like "the
    policy learns a constant", which is a long way from its cause.
    """
    cell = LTCCell(N_FEATURES, 6, seed=0)
    bad = TH0.copy()
    with pytest.raises(ValueError, match="strictly interior"):
        WeightPolicy(cell, bad, THETA_LO, bad.copy())      # hi == theta0


def test_every_shipped_weight_has_room_to_move_both_ways():
    """The real bounds, against the real offline anchor."""
    th0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    assert (THETA_HI - th0 > 1e-6).all(), "a weight cannot be increased"
    assert (th0 - THETA_LO > 1e-6).all(), "a weight cannot be decreased"
