"""All seven safety filters, against the properties that make one a filter."""

from __future__ import annotations

import numpy as np
import pytest

from mpcc_tuning.filters import FILTERS, ASIF, TubeASIF, AdaptiveTubeASIF, CBFQP
from mpcc_tuning.model import A_LAT_MAX, KinematicBicycle
from mpcc_tuning.track import Track

# The NLP filter builds an IPOPT solver per instance, which is slow enough that
# it is opted into rather than run in every parametrised case.
FAST = [k for k in FILTERS if k != "mpcc_terminal"]


@pytest.fixture(scope="module")
def track():
    return Track.oval()


def _state(track, s=0.0, v=3.0, off=0.0):
    p = np.array([float(track.pos(s)[0]), float(track.pos(s)[1])])
    psi = float(track.tangent_angle(s))
    n = np.array([np.sin(psi), -np.cos(psi)])
    q = p + off * n
    return np.array([q[0], q[1], psi, v, s])


# -- the model every filter shares -----------------------------------------
def test_the_filter_model_matches_the_plant_exactly(track):
    """An earlier version updated the heading before integrating position,
    which the plant does the other way round. It looks equivalent, costs 1.4 cm
    per step with non-zero steering, and over a 30-step backup that is a
    systematic error comparable to the whole margin."""
    km = KinematicBicycle(dt=0.05, grip=1.0)
    f = ASIF(track)
    x = np.array([1.0, 0.5, 0.3, 3.5])
    worst = 0.0
    for d in np.linspace(-0.4, 0.4, 9):
        for a in np.linspace(-4.0, 4.0, 5):
            p = km.step(x, np.array([d, a]))
            m = f.step(x[0], x[1], x[2], x[3], d, a)
            worst = max(worst, max(abs(p[i] - m[i]) for i in range(4)))
    assert worst == 0.0, f"disagrees with the plant by {worst:.2e}"


def test_the_model_has_a_yaw_rate_cap(track):
    """Without it every input is certified from every state and the filter is
    switched off while reporting a 0% intervention rate."""
    f = ASIF(track, assumed_grip=1.0)
    _x, _y, psi_fast, v_next = f.step(0.0, 0.0, 0.0, 4.0, 0.4, 0.0)
    uncapped = 4.0 / f.wheelbase * np.tan(0.4) * f.dt
    assert psi_fast < uncapped, "no cap applied at 4 m/s"
    # The cap uses the speed *after* the update, which drag has already reduced
    # -- 3.97, not 4.0. Asserting against the pre-update speed is off by the
    # drag term and was, on first writing, mistaken for a bug in the model.
    assert psi_fast == pytest.approx(A_LAT_MAX * 1.0 / v_next * f.dt, rel=1e-9)


@pytest.mark.parametrize("name", FAST)
def test_corridor_is_stricter_than_the_plants(track, name):
    """A filter less conservative than the thing it protects is not a filter."""
    f = FILTERS[name](track)
    assert track.half_width - f.margin < track.half_width - 0.12


# -- the two properties that define a filter -------------------------------
#: The pointwise filters are *structurally* more conservative: a one-step
#: condition cannot tell that a plan exists, only that the next state is
#: acceptable, so it refuses inputs a rollout certifies. That is a property of
#: the method, not a bug, and it is measured in
#: ``test_pointwise_filters_are_measurably_more_conservative`` rather than
#: hidden by loosening the bound for everyone.
POINTWISE = {"cbf", "clf_cbf"}


@pytest.mark.parametrize("name", [n for n in FAST if n not in POINTWISE])
def test_invisible_on_a_controller_that_would_not_crash(track, name):
    """Any intervention on a safe controller is pure cost."""
    from examples.tune_online import Plant
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    f = FILTERS[name](track)
    plant = Plant(track, dt=0.05)
    plant.max_steps = 120
    s5 = plant.reset()
    m.reset()
    for _ in range(plant.max_steps):
        u = m.value(s5, MPCCWeights().to_log())["u0"]
        u, _ = f(s5, u)
        s5, _r, off, tr = plant.step(u)
        if hasattr(f, "observe"):
            f.observe(s5)
        if off or tr:
            break
    assert not off, f"{name} let a safe controller off the track"
    assert f.intervention_rate <= 0.02, (
        f"{name} overrode {f.intervention_rate:.1%} of a safe controller")


@pytest.mark.parametrize("name", FAST)
def test_saves_a_controller_that_would_crash(track, name):
    from examples.tune_online import Plant
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    theta = MPCCWeights(q_c=0.41, q_l=1.40, q_v=22.0, r_d=0.019, r_a=0.006,
                        r_dv=0.019).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)

    def go(filt):
        plant = Plant(track, dt=0.05)
        plant.max_steps = 200
        s5 = plant.reset()
        m.reset()
        off = False
        for _ in range(plant.max_steps):
            u = m.value(s5, theta)["u0"]
            if filt is not None:
                u, _ = filt(s5, u)
            s5, _r, off, tr = plant.step(u)
            if filt is not None and hasattr(filt, "observe"):
                filt.observe(s5)
            if off or tr:
                break
        return off

    assert go(None), "the unfiltered controller was supposed to crash"
    f = FILTERS[name](track)
    assert not go(f), f"{name} failed to keep the car on the track"


# -- per-filter properties --------------------------------------------------
@pytest.mark.parametrize("name", sorted(POINTWISE))
def test_pointwise_filters_are_measurably_more_conservative(track, name):
    """The CBF *does* override a controller that was never going to crash.

    Measured at ~4% over 120 steps and ~8% over 400. This is the price of not
    having a horizon, and it is asserted rather than excused: if it ever drops
    to zero the barrier has become permissive and the safety claim needs
    re-checking, and if it climbs the filter is turning into the controller.
    """
    from examples.tune_online import Plant
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    f = FILTERS[name](track)
    plant = Plant(track, dt=0.05)
    plant.max_steps = 120
    s5 = plant.reset()
    m.reset()
    for _ in range(plant.max_steps):
        u = m.value(s5, MPCCWeights().to_log())["u0"]
        u, _ = f(s5, u)
        s5, _r, off, tr = plant.step(u)
        if off or tr:
            break
    assert not off, f"{name} let a safe controller off the track"
    assert 0.005 < f.intervention_rate < 0.15, (
        f"{name} overrode {f.intervention_rate:.1%}; expected a few percent")


def test_tube_worst_case_is_the_lower_grip(track):
    """The tube's monotonicity claim: less grip is the binding case.

    TubeASIF certifies at the interval's lower end only, which is sound *if*
    grip enters monotonically. This checks it rather than assuming it: anything
    the low-grip model certifies, the high-grip model must certify too.
    """
    f = TubeASIF(track, grip_interval=(0.6, 1.4))
    checked = 0
    for s in (0.0, 4.0, 7.5, 9.0):
        for v in (2.0, 3.0, 4.0):
            for off in (-0.4, 0.0, 0.4):
                st = _state(track, s, v, off)
                for d in np.linspace(-0.4, 0.4, 5):
                    for a in (-4.0, 0.0, 4.0):
                        lo = ASIF.certify(f, st, d, a, grip=0.6)
                        hi = ASIF.certify(f, st, d, a, grip=1.4)
                        if lo:
                            assert hi, "low grip certified what high grip refused"
                            checked += 1
    assert checked > 50, "the sweep did not exercise enough certified cases"


def test_adaptive_starts_pessimistic_and_learns(track):
    f = AdaptiveTubeASIF(track, grip_interval=(0.6, 1.4), min_samples=20)
    assert f.grip_lcb == 0.6, "should start at the prior's worst case"
    true_g = 1.1
    psi = 0.0
    for _ in range(120):
        psi += A_LAT_MAX * true_g / 2.0 * f.dt
        f.observe(np.array([0.0, 0.0, psi, 2.0, 0.0]))
    assert f.grip_lcb > 0.9, f"did not learn: lcb {f.grip_lcb:.3f}"
    assert f.grip_lcb <= 1.4, "must stay inside the prior interval"


def test_cbf_barrier_kinds_differ_in_the_way_claimed(track):
    """The naive barrier does not depend on speed; the braking one does."""
    lat = CBFQP(track, h_kind="lateral")
    brk = CBFQP(track, h_kind="braking")
    st = _state(track, s=4.0, v=1.0, off=0.3)
    x, y, psi = st[0], st[1], st[2] + 0.3
    assert lat.barrier(x, y, psi, 1.0) == pytest.approx(lat.barrier(x, y, psi, 4.0))
    assert brk.barrier(x, y, psi, 4.0) < brk.barrier(x, y, psi, 1.0)


def test_viability_kernel_is_a_fixed_point(track):
    """One more sweep must not remove any further states."""
    from mpcc_tuning.filters.reachability import ViabilityFilter
    f = ViabilityFilter(track, n_d=21, n_e=21, n_v=11, iters=60)
    before = f.kernel.sum()
    assert before > 0, "the kernel is empty"
    f2 = ViabilityFilter(track, n_d=21, n_e=21, n_v=11, iters=120)
    assert f2.kernel.sum() == before, "not converged at 60 iterations"


def test_intervention_rate_is_not_a_safety_metric(track):
    """The property that makes a broken filter dangerous, asserted.

    An optimistic filter intervenes LESS than a correct one -- it certifies
    what it should refuse. Anyone reading intervention rate as safety gets the
    sign backwards, so it is pinned here.
    """
    from examples.tune_online import Plant
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    theta = MPCCWeights(q_c=0.41, q_l=1.40, q_v=22.0, r_d=0.019, r_a=0.006,
                        r_dv=0.019).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    rates = {}
    for grip in (0.6, 2.5):
        f = ASIF(track, assumed_grip=grip)
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
        rates[grip] = f.intervention_rate
    assert rates[2.5] < rates[0.6], (
        f"the optimistic filter intervened MORE ({rates[2.5]:.1%} vs "
        f"{rates[0.6]:.1%}) -- the premise of the docs is wrong")
