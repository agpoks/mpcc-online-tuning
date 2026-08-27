"""The predictive safety filter, and the two bugs that made it useless."""

from __future__ import annotations

import numpy as np
import pytest

from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.safety import PredictiveSafetyFilter
from mpcc_tuning.track import Track


@pytest.fixture
def track():
    return Track.oval()


def test_the_filters_model_matches_the_plant_exactly(track):
    """The filter predicts with the plant's bicycle, not the MPCC's.

    An earlier version updated the heading before integrating position, which
    the plant does the other way round. It looks equivalent, costs 1.4 cm per
    step whenever the steering is non-zero, and over a 30-step backup that is a
    systematic error comparable to the whole margin -- it certified braking
    manoeuvres that then left the track.
    """
    km = KinematicBicycle(dt=0.05, grip=1.0)
    f = PredictiveSafetyFilter(track)
    x = np.array([1.0, 0.5, 0.3, 3.5])
    worst = 0.0
    for d in np.linspace(-0.4, 0.4, 9):
        for a in np.linspace(-4.0, 4.0, 5):
            p = km.step(x, np.array([d, a]))
            m = f._step(x[0], x[1], x[2], x[3], d, a, 0.05)
            worst = max(worst, max(abs(p[i] - m[i]) for i in range(4)))
    assert worst == 0.0, f"filter model disagrees with the plant by {worst:.2e}"


def test_the_filter_is_more_conservative_than_the_plants_own_limit(track):
    """Its corridor must be strictly inside the plant's off-track threshold.

    At the original margin of 0.10 the filter's corridor was *wider* than the
    plant's (0.65 against 0.63), so it certified an action, the action put the
    car outside, and the filter first refused on the step the car was already
    off. A filter less conservative than the thing it protects is not a filter.
    """
    f = PredictiveSafetyFilter(track)
    plant_limit = track.half_width - 0.12
    assert track.half_width - f.margin < plant_limit


def test_the_model_has_a_lateral_acceleration_limit(track):
    """Without it the filter certifies everything and reports 0% intervention.

    The MPCC's internal bicycle has no yaw-rate cap, so a car predicted with it
    can turn on a dime and can therefore always save itself. A filter built on
    that model is switched off while still looking like it is working.
    """
    f = PredictiveSafetyFilter(track, assumed_grip=1.0)
    # Hard steer at speed must be rate-limited relative to the same at low speed.
    _x1, _y1, psi_fast, _v1 = f._step(0.0, 0.0, 0.0, 4.0, 0.4, 0.0, 0.05)
    _x2, _y2, psi_slow, _v2 = f._step(0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.05)
    yaw_fast, yaw_slow = psi_fast / 0.05, psi_slow / 0.05
    assert yaw_fast < 4.0 / 0.33 * np.tan(0.4), "no cap applied at 4 m/s"
    assert yaw_fast < yaw_slow * 2.0, "cap should bite harder as speed rises"


def test_it_saves_a_controller_that_would_otherwise_crash(track):
    """The whole point: weights the tuner collapsed to, with and without."""
    from examples.tune_online import Plant
    m = MPCC_ = __import__("mpcc_tuning.mpcc", fromlist=["MPCC"]).MPCC(
        track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    theta = np.log(np.array([0.41, 1.40, 22.0, 0.019, 0.006, 0.019]))

    def go(use_filter):
        f = PredictiveSafetyFilter(track)
        plant = Plant(track, dt=0.05)
        plant.max_steps = 300
        s5 = plant.reset()
        m.reset()
        off = False
        for _ in range(plant.max_steps):
            u = m.value(s5, theta)["u0"]
            if use_filter:
                u, _ = f(s5, u)
            s5, _r, off, tr = plant.step(u)
            if off or tr:
                break
        return off, f

    off_bare, _ = go(False)
    off_filt, filt = go(True)
    assert off_bare, "the unfiltered controller was supposed to crash"
    assert not off_filt, "the filter failed to keep the car on the track"
    assert filt.n_no_safe_action == 0, "recursive feasibility was violated"


def test_it_is_invisible_to_a_controller_that_was_not_going_to_crash(track):
    """A filter that intervenes on a good controller is a controller."""
    from examples.tune_online import Plant
    m = __import__("mpcc_tuning.mpcc", fromlist=["MPCC"]).MPCC(
        track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    from mpcc_tuning.mpcc import MPCCWeights
    theta = MPCCWeights().to_log()
    f = PredictiveSafetyFilter(track)
    plant = Plant(track, dt=0.05)
    plant.max_steps = 150
    s5 = plant.reset()
    m.reset()
    for _ in range(plant.max_steps):
        u = m.value(s5, theta)["u0"]
        u, _ = f(s5, u)
        s5, _r, off, tr = plant.step(u)
        if off or tr:
            break
    assert f.intervention_rate == 0.0, (
        f"filter overrode {f.intervention_rate:.1%} of a safe controller's actions")
