"""What every safety filter here has in common.

A safety filter is a map from (state, proposed input) to (applied input), which
leaves the input alone unless applying it would forfeit the ability to stay
safe. That is the whole interface, and it is deliberately narrow: nothing in it
knows what produced the proposed input, so the same filter wraps a tuned MPCC,
an untuned one, or a random number generator, and the guarantee does not depend
on which.

The filters differ in *what they check*, and that is the only axis on which they
should be compared:

===================  ==========================================================
:class:`ASIF`        exhibits a trajectory -- roll a backup controller forward
                     and require the whole path legal, ending in a set you can
                     stay in
:class:`TubeASIF`    the same, but over an interval of models, so the
                     certificate holds for every plant in the set
:class:`CBFQP`       evaluates a function -- one inequality on the input,
                     solved exactly as a QP
:class:`CLFCBFQP`    the same plus a stability constraint
:class:`ViabilityFilter`  set membership in a precomputed viability kernel
:class:`AdaptiveTubeASIF`  a tube whose width is estimated online
:class:`MPCCSafetyFilter`  the controller's own OCP, with a terminal set
===================  ==========================================================

Common vocabulary, used by all of them:

``h(x)``
    the safety margin, positive inside the safe set. Here it is the distance
    from the corridor edge, so ``h > 0`` means "on the track with room".
``pi_b(x)``
    the *backup* controller. It does not have to be good, only safe.
``X_safe``
    the terminal set: states you can remain in forever. Here, "stopped and
    inside the corridor".
``margin``
    how much narrower the filter's corridor is than the plant's. This absorbs
    model error and **must be non-zero**: a filter whose corridor matches the
    plant's exactly first refuses on the step the car is already off.
"""

from __future__ import annotations

import numpy as np


class SafetyFilter:
    """Base class: statistics, the interface, and the invariants worth stating."""

    def __init__(self, track, dt: float = 0.05, margin: float = 0.18,
                 credit: str = "executed"):
        if credit not in ("executed", "proposed"):
            raise ValueError("credit must be 'executed' or 'proposed'")
        self.track, self.dt, self.margin = track, float(dt), float(margin)
        self.credit = credit
        self.reset_stats()

    # -- statistics --------------------------------------------------------
    def reset_stats(self) -> None:
        self.n_steps = 0
        self.n_interventions = 0
        self.n_no_safe_action = 0

    @property
    def intervention_rate(self) -> float:
        """Fraction of steps the filter changed the input.

        **This is a cost metric, not a safety metric.** A filter with a broken
        model intervenes *less*, not more -- it certifies things it should
        refuse. Every bug found while building these showed up as a
        reassuringly low intervention rate. Read it as "what the filter cost
        the controller", and read ``n_no_safe_action`` for whether it was in
        trouble.
        """
        return self.n_interventions / max(self.n_steps, 1)

    # -- geometry ----------------------------------------------------------
    def lateral(self, x, y) -> float:
        return float(self.track.lateral(x, y))

    def h(self, x, y) -> float:
        """Safety margin: how much corridor is left, in metres."""
        return (self.track.half_width - self.margin) - abs(self.lateral(x, y))

    def inside(self, x, y) -> bool:
        return self.h(x, y) >= 0.0

    # -- interface ---------------------------------------------------------
    def __call__(self, state5, u):
        """Return ``(u_to_apply, intervened)``."""
        raise NotImplementedError

    def certify(self, state5, delta: float, a: float) -> bool:
        """Is ``(delta, a)`` safe from ``state5``, by this filter's criterion?"""
        raise NotImplementedError
