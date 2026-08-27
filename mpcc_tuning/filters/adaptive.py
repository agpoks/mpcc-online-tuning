"""A tube whose width is learned online, instead of guessed.

References
----------
Wabersich & Zeilinger, *"Probabilistic model predictive safety certification
for learning-based control"*, IEEE TAC 2021 (arXiv:1906.10417) -- the
probabilistic safety filter this is a cheap stand-in for.

Berkenkamp, Turchetta, Schoellig & Krause, *"Safe Model-based Reinforcement
Learning with Stability Guarantees"*, NeurIPS 2017 -- learning the model while
keeping a guarantee.

Hewing, Kabzan & Zeilinger, *"Cautious Model Predictive Control using Gaussian
Process Regression"*, IEEE TCST 2020 -- the GP-residual formulation, on a race
car.

The problem
-----------
:class:`~mpcc_tuning.filters.asif.TubeASIF` fixes the "which grip do I assume"
question by refusing to answer it: give it an interval and it certifies against
the whole interval. That is sound and it is *pessimistic for the entire run* --
if the true grip is 1.3 and the interval is [0.6, 1.4], the filter drives as if
it were 0.6 forever, and the intervention rate reflects a car that does not
exist.

The fix is to shrink the interval as evidence arrives. Grip is observable: it
enters through the yaw-rate cap, so whenever the cap binds, the achieved yaw
rate reveals it directly,

.. math::
    \\dot\\psi_\\text{obs} = \\frac{A_{\\text{lat,max}}\\, g}{v}
    \\quad\\Longrightarrow\\quad
    \\hat g = \\frac{v\\,\\dot\\psi_\\text{obs}}{A_{\\text{lat,max}}}

This class runs a recursive mean and variance over those observations and
certifies against :math:`\\hat g - \\kappa\\hat\\sigma`, a lower confidence
bound, clipped to the prior interval. Early on there is no evidence and the
bound sits at the prior's worst case; as evidence accumulates the tube narrows
towards the truth.

What this is not
----------------
It is **not a GP**, and the difference matters. A GP over the model residual
gives a state-dependent posterior with calibrated uncertainty everywhere,
including states never visited; this gives one scalar with a
frequentist-flavoured bound, valid only where data has been collected. It is
the right shape of idea at a hundredth of the machinery, and it inherits the
central caveat of the whole family: **the guarantee is now only as good as the
confidence bound**, and a bound that is too tight is exactly the optimistic
filter that crashes while intervening less.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.filters.asif import TubeASIF
from mpcc_tuning.model import A_LAT_MAX


class AdaptiveTubeASIF(TubeASIF):
    """A tube filter that estimates the grip it is being conservative about."""

    def __init__(self, *args, kappa: float = 2.0, min_samples: int = 20,
                 forget: float = 0.995, **kw):
        super().__init__(*args, **kw)
        self.kappa, self.min_samples, self.forget = float(kappa), int(min_samples), float(forget)
        self._n, self._mean, self._m2 = 0.0, 0.0, 0.0
        self._prev = None

    # -- estimation --------------------------------------------------------
    def observe(self, state5) -> None:
        """Feed one transition. Call it every tick, after stepping the plant.

        Only transitions where the cap plausibly bound carry information about
        grip; the rest are discarded rather than averaged in, which would pull
        the estimate towards whatever the car happened to be doing.
        """
        s = np.asarray(state5, float)
        if self._prev is not None:
            v = float(self._prev[3])
            dpsi = float(np.arctan2(np.sin(s[2] - self._prev[2]),
                                    np.cos(s[2] - self._prev[2]))) / self.dt
            if v > 1.0 and abs(dpsi) > 1e-3:
                g = abs(dpsi) * v / A_LAT_MAX
                if 0.05 < g < 3.0:
                    self._n = self._n * self.forget + 1.0
                    d = g - self._mean
                    self._mean += d / self._n
                    self._m2 = self._m2 * self.forget + d * (g - self._mean)
        self._prev = s.copy()

    @property
    def grip_lcb(self) -> float:
        """Lower confidence bound on grip, clipped to the prior interval."""
        lo, hi = self.grip_interval
        if self._n < self.min_samples:
            return lo                      # no evidence: the prior's worst case
        sd = float(np.sqrt(max(self._m2 / max(self._n, 1.0), 0.0)))
        return float(np.clip(self._mean - self.kappa * sd, lo, hi))

    @property
    def _grips(self):
        return [self.grip_lcb]
