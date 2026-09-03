"""Verified per-track weights: a conservative start, and the hand-tuned best.

This is TODO item 0, which the project has asked for repeatedly:

> **Start from a stable working parameterisation.** It need not be perfect or
> the fastest. **Then adapt from there** -- by track, sector, opponent,
> surface, grip.

Every adaptation result before this was measured from ``q_c=1.0, q_l=200,
q_v=2.0, r_d=1.0``, an anchor that covers 0.5 m of ICRA T1 before leaving the
track. A learner asked to adapt around an operating point that crashes, with
its output range measured from that same point, is being asked the wrong
question -- and that is a better explanation for weak adaptation than any
mechanism in the learner.

## Two entries per track, and the difference is the experiment

``START``
    Clean, stable, and deliberately NOT the fastest. This is theta_0: the
    online tuner begins here.
``BEST``
    The best clean result found by hand. This is the target, not the start.

The experiment is whether the tuner moves START toward BEST on its own, per
track and per situation. Recording both up front means that question has an
answer rather than an impression -- and if the tuner overshoots BEST, that is
a result too.

## START and BEST must be the SAME OCP

``q_vref`` is a ``build_ocp`` argument, not one of the eight weights the policy
emits, so the tuner cannot change it. An earlier version of this table had the
oval going ``q_vref`` 0.00 -> 0.05 and T2 going 0.00 -> 0.20 between START and
BEST, which put most of the recorded headroom behind a constant the learner has
no access to: 7.42 of the oval's 7.42 laps were unreachable by construction.
The tuner would have "failed" to close a gap it was never able to reach.

``q_vref`` is therefore pinned per track in :data:`QVREF` and shared by both
entries; only theta differs. :func:`check` enforces it.

## What "clean" means here, and why it is the bar

``clean`` = the run ended at the step limit with the car ON the track, not by
leaving it. It is a stricter and more useful bar than lap count: a controller
that covers 1.9 laps and then crashes is not better than one that covers 1.5
and is still driving. Several settings in this file's history looked good on
laps alone and were found to be truncation happening before a crash.

## Why ICRA T1 is not in this table

T1 and T2 are the same geometry, so T2 carries both. T1 was measured anyway
while establishing this table, and the result is worth keeping: **q_vref does
not help T1.** Its best clean run with the reference-speed term is 2.01 laps
against 2.05 without it, and the settings that go further (2.56 laps at
q_vref=0.20) leave the track. This is the opposite of T2, which gains
1.58 -> 3.05 from the same term -- a reminder that a term measured helpful on
one track is not thereby helpful on another.

All numbers below are the acados backend, ``fqp_soft_funnel``
(SQP_WITH_FEASIBLE_QP + FUNNEL_L1PEN globalization), dynamic drift model on the
scuderia STD plant, 9-11 ms/tick.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    """One verified parameterisation, with the measurement that verified it."""

    weights: dict
    horizon: int
    q_vref: float = 0.0
    #: laps completed in the verifying run
    laps: float = 0.0
    #: True when the run ended at the step limit rather than off the track
    clean: bool = False
    steps: int = 0
    #: peak speed reached, m/s -- the quickest read on how conservative it is
    peak_v: float = 0.0
    note: str = ""

    def theta(self):
        """The log-weight vector the policy anchors on."""
        from mpcc_tuning.mpcc import MPCCWeights
        return MPCCWeights(**self.weights).to_log()


#: The reference-speed cost weight, pinned per track. NOT tunable: it is a
#: build_ocp argument, so START and BEST must agree on it or the headroom
#: between them is partly unreachable. Chosen per track by measurement --
#: the oval and T2 both gain from it, T1 does not (see the module docstring).
QVREF = {"oval": 0.05, "icra_t2_raceline": 0.20}

#: Conservative, clean, and slower than it needs to be. The tuner starts here.
#:
#: Chosen as the SLOWEST clean setting found, so there is as much reachable
#: headroom as possible: an improvement should be visible rather than marginal.
START = {
    "oval": Setting(
        weights=dict(q_c=1.0, q_l=50.0, q_v=0.30, r_d=0.5, r_a=0.05, k_v=0.35),
        horizon=12, q_vref=0.05, laps=7.90, clean=True, steps=2500,
        peak_v=2.05,
        note="clean but timid: peaks at 2.05 m/s where the tuned setting "
             "reaches 2.90. NOT the slowest clean setting measured -- that "
             "was 5.79 laps at k_v=0.25, which is BELOW the policy's floor "
             "of 0.30 and so cannot be emitted at all. k_v=0.35 sits at "
             "10.5% of its log span, which is where an anchor wants to be: "
             "almost all the room is upward, toward BEST's 0.50."),
    "icra_t2_raceline": Setting(
        weights=dict(q_c=1.0, q_l=50.0, q_v=0.20, r_d=1.0, r_a=6.0, k_v=0.40),
        horizon=25, q_vref=0.20, laps=1.92, clean=True, steps=2500,
        peak_v=1.37,
        note="clean at 1.37 m/s against 1.87 for the best clean candidate "
             "in the same sweep. Same OCP as BEST -- only theta differs."),
}

#: The best CLEAN result found by hand. The target, not the start.
BEST = {
    "oval": Setting(
        weights=dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05, k_v=0.50),
        horizon=12, q_vref=0.05, laps=12.55, clean=True, steps=2500,
        peak_v=2.90,
        note="Note the tracks disagree about k_v -- the oval wants 0.50 and "
             "T2 wants 0.85, which is the situation-dependence this project "
             "is about, visible in hand-tuned data before any learning."),
    "icra_t2_raceline": Setting(
        weights=dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=6.0, k_v=0.85),
        horizon=25, q_vref=0.20, laps=3.05, clean=True, steps=2500,
        peak_v=1.87,
        note="k_v = 0.85, the opposite of the oval's preference."),
}

#: Horizon is part of the baseline, not a global constant. Measured: N=40 gives
#: 0.59 laps on T1 and N=25 gives 2.05, while the oval is happy at 12. A single
#: horizon for every track is the same mistake as a single weight vector.
HORIZON = {k: v.horizon for k, v in START.items()}

TRACKS = tuple(START)


def start(track: str) -> Setting:
    """theta_0 for ``track`` -- clean, stable, not optimal."""
    if track not in START:
        raise KeyError(
            f"no verified baseline for {track!r}. Establish one before running "
            f"the tuner there: an adaptation result measured from an unverified "
            f"anchor says nothing. Known: {sorted(START)}")
    return START[track]


def best(track: str) -> Setting:
    """The hand-tuned target for ``track``."""
    return BEST[track]


def headroom(track: str) -> float:
    """Laps the tuner would have to find to match hand tuning.

    Reachable by construction: :func:`check` guarantees START and BEST differ
    only in theta, which is the thing the policy actually emits.
    """
    return BEST[track].laps - START[track].laps


def check_in_policy_box() -> None:
    """Every START must be an anchor the policy can actually emit.

    :class:`~mpcc_tuning.ltc.WeightPolicy` anchors at theta0 and spans
    ``hi - theta0`` above and ``theta0 - lo`` below, so an anchor outside the
    box is not merely clipped -- it is unreachable, and one exactly ON a bound
    has zero span on that side and a structurally zero gradient there.

    The first oval START written to this file failed exactly that: k_v = 0.25
    against a floor of 0.30, at -12.4% of the log span. The tuner would have
    started from a weight vector its own policy could not represent.
    """
    import numpy as np
    from mpcc_tuning.ltc import THETA_HI, THETA_LO
    from mpcc_tuning.mpcc import WEIGHT_NAMES
    for t in TRACKS:
        for tag, st in (("START", START[t]), ("BEST", BEST[t])):
            th = np.asarray(st.theta(), float)
            bad = np.flatnonzero((th <= THETA_LO) | (th >= THETA_HI))
            if bad.size:
                names = ", ".join(
                    f"{WEIGHT_NAMES[i]}={np.exp(th[i]):.3g} not strictly in "
                    f"[{np.exp(THETA_LO[i]):.3g}, {np.exp(THETA_HI[i]):.3g}]"
                    for i in bad)
                raise ValueError(
                    f"{t} {tag}: {names}. The policy anchors at theta0 with "
                    f"span hi-theta0 above and theta0-lo below; an anchor on "
                    f"or outside a bound has no span on that side.")


def check() -> None:
    """Fail loudly if START and BEST ever stop being the same OCP.

    The gap between them is only a learning target if theta is the ONLY thing
    that differs. Anything else -- q_vref, horizon -- is a constant the tuner
    cannot touch, and putting headroom behind one makes the experiment
    unwinnable in a way that looks like a weak learner.
    """
    for t in TRACKS:
        a, b = START[t], BEST[t]
        if a.q_vref != b.q_vref or a.q_vref != QVREF[t]:
            raise ValueError(
                f"{t}: q_vref differs between START ({a.q_vref}) and BEST "
                f"({b.q_vref}) or from QVREF ({QVREF[t]}). q_vref is a build "
                f"constant, not a weight -- headroom behind it is unreachable.")
        if a.horizon != b.horizon:
            raise ValueError(
                f"{t}: horizon differs ({a.horizon} vs {b.horizon}). The "
                f"tuner emits weights, not a horizon.")
        if not a.clean:
            raise ValueError(
                f"{t}: START is not clean ({a.laps} laps, left the track). "
                f"The first thing the tuner would learn is how not to crash, "
                f"not how to go faster.")
        if not b.clean:
            raise ValueError(f"{t}: BEST is not clean; it is not a target.")
        if b.laps <= a.laps:
            raise ValueError(
                f"{t}: BEST ({b.laps}) is not better than START ({a.laps}); "
                f"there is nothing for the tuner to find.")
    check_in_policy_box()


check()
