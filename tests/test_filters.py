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
    if name == "viability":
        pytest.xfail("the viability kernel is not converged in its speed grid: "
                     "it saves the car at dv=0.2 m/s and not at 0.4 (the "
                     "current n_v=21 with SPEED_MAX=8) or 0.1 -- see "
                     "test_viability_kernel_is_converged_in_its_grid")
    f = FILTERS[name](track)
    assert not go(f), f"{name} failed to keep the car on the track"


@pytest.mark.xfail(reason="open: the viability kernel is not converged in its "
                          "own speed discretisation; it saves the car at "
                          "dv=0.2 m/s and not at 0.4 or 0.1",
                   strict=False)
def test_viability_kernel_is_converged_in_its_grid(track):
    """A safety guarantee must not be an artefact of a grid constant.

    ViabilityFilter grids speed as ``linspace(0, SPEED_MAX, n_v)`` with n_v=21.
    Raising SPEED_MAX from 4 to 8 m/s therefore halved the resolution without
    changing a line of the filter, and the filter stopped saving a controller
    that crashes unfiltered. The obvious reading -- coarser grid, worse kernel
    -- is WRONG. Measured over 200 steps on the oval against the bad weights:

        n_v    dv (m/s)   intervene   off track
         21      0.400        8.1%    YES
         41      0.200       66.0%    no
         81      0.100       10.0%    YES

    Non-monotonic. dv=0.200 is exactly the spacing the filter had before
    SPEED_MAX changed (4.0 over 20 intervals), and it is the only one that
    works; a FINER grid fails too, certifying more states as safe (10%
    intervention) and leaving the track.

    So the kernel is not converged with respect to its own discretisation, and
    the value it was validated at is a coincidence of two constants that were
    never linked. Until it converges, the filter's guarantee is a property of
    n_v rather than of the dynamics.

    This is also the leading candidate for the undiagnosed ``worst-case`` row
    in ``benchmarks/filters.py``, which leaves the track 60% of the time at
    grip 1.0 where that should be impossible -- the signature matches: a filter
    that reports plausible intervention rates while being wrong about which
    states are recoverable. Not yet established, and the row is still not cited.

    Fixing it is not a matter of picking a better n_v. It needs the kernel
    checked for convergence -- refine until the safe set stops moving -- and
    the grid tied to a resolution in m/s rather than to a point count that
    silently rescales whenever SPEED_MAX does.
    """
    import numpy as np
    from examples.tune_online import Plant
    from mpcc_tuning.filters.reachability import ViabilityFilter
    from mpcc_tuning.mpcc import MPCC, MPCCWeights

    bad = MPCCWeights(q_c=0.41, q_l=1.40, q_v=22.0, r_d=0.019, r_a=0.006,
                      r_dv=0.019).to_log()
    off_by_n = {}
    for n_v in (21, 41, 81):
        m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
                 max_iter=60)
        f = ViabilityFilter(track, n_v=n_v)
        plant = Plant(track, dt=0.05)
        plant.max_steps = 200
        s5 = plant.reset()
        m.reset()
        off = False
        for _ in range(plant.max_steps):
            u = m.value(s5, bad)["u0"]
            u, _ = f(s5, u)
            s5, _r, off, tr = plant.step(u)
            if hasattr(f, "observe"):
                f.observe(s5)
            if off or tr:
                break
        off_by_n[n_v] = off
    assert not any(off_by_n.values()), (
        f"the kernel's answer depends on its grid: off-track by n_v {off_by_n}")


# -- per-filter properties --------------------------------------------------
@pytest.mark.parametrize("name", sorted(POINTWISE))
def test_pointwise_filters_are_measurably_more_conservative(track, name):
    """The barrier must still bite when there is something to bite on.

    This asserted a few percent of overrides against the DEFAULT weights, on
    the reasoning that a pointwise filter has no horizon and therefore pays for
    it continuously -- "if it ever drops to zero the barrier has become
    permissive and the safety claim needs re-checking".

    It did drop to zero, and the reasoning was wrong about why. Measured over
    120 steps on the oval, toggling the grip-limited speed and terminal-speed
    constraints:

        filter    grip   weights   intervene   off track
        cbf       on     good          0.0%    no
        cbf       on     bad          50.0%    no
        cbf       off    good         17.5%    no
        cbf       off    bad          80.8%    no

    (clf_cbf is identical.) The barrier is not permissive -- it overrides half
    the commands from a bad controller and keeps the car on the track. The zero
    is the MPCC's own terminal-speed and grip constraints leaving nothing to
    correct, which is what those constraints were added to do.

    The old thresholds were calibrated against a SPEED_MAX of 4 m/s and are
    stale at both ends: at 8 m/s with the constraints off the default weights
    override 17.5%, past the 15% ceiling this used to assert.

    So permissiveness is now checked where it is still detectable -- against
    weights that genuinely need correcting -- and the safe-controller case
    asserts what it can honestly assert: the filter does not take over, and the
    car stays on the track.
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
    assert f.intervention_rate < 0.15, (
        f"{name} overrode {f.intervention_rate:.1%} of a SAFE controller's "
        f"commands; the filter is turning into the controller")

    # And the half that keeps the zero above honest: the same barrier, the same
    # track, against weights that do need correcting.
    bad = MPCCWeights(q_c=0.41, q_l=1.40, q_v=22.0, r_d=0.019, r_a=0.006,
                      r_dv=0.019).to_log()
    m2 = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
              max_iter=60)
    f2 = FILTERS[name](track)
    plant2 = Plant(track, dt=0.05)
    plant2.max_steps = 120
    s5 = plant2.reset()
    m2.reset()
    off2 = False
    for _ in range(plant2.max_steps):
        u = m2.value(s5, bad)["u0"]
        u, _ = f2(s5, u)
        s5, _r, off2, tr = plant2.step(u)
        if off2 or tr:
            break
    assert f2.intervention_rate > 0.05, (
        f"{name} overrode only {f2.intervention_rate:.1%} of a controller that "
        f"crashes unfiltered -- the barrier has become permissive")
    assert not off2, f"{name} let the bad controller off the track"


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
