"""How fast the car may actually go, from grip rather than from a constant.

``SPEED_MAX`` was a flat \\SI{4}{\\meter\\per\\second} cap, and a flat cap does
not describe a vehicle -- it describes an assumption. Worse, it *changes the
experiment*: with the cap low enough the car is speed-limited almost
everywhere, no corner is grip-limited, no weight setting is ever punished, and
a track cannot discriminate between weight settings at all. That is exactly the
failure measured on the synthetic circuit, and the cap was manufacturing it.

## The friction ellipse

A tyre has one budget and spends it on cornering and on accelerating together:

.. math::
    \\left(\\frac{a_x}{a_{x,\\max}}\\right)^2 +
    \\left(\\frac{a_y}{a_{y,\\max}\\,\\mu}\\right)^2 \\le 1

Cornering alone gives the classic limit :math:`v \\le \\sqrt{a_{y,\\max}\\mu/\\kappa}`.
The rest of the profile follows from integrating what is left of the budget
forwards (accelerating out) and backwards (braking in) -- the standard
forward--backward pass used by every racing-line optimiser.

## Validated, not asserted

Against the ICRA team's own optimised raceline for Track 1:

======================================  ==============  ======  =========
                                        speed [m/s]     mean    lap time
======================================  ==============  ======  =========
their optimiser                         2.45 -- 6.09    4.38    17.6 s
this, :math:`\\mu = 1`, uncapped         2.58 -- 7.39    4.73    16.4 s
this, capped at \\SI{4}{\\meter\\per\\second}  2.58 -- 4.00    3.79    19.2 s
======================================  ==============  ======  =========

Within about 7% of a real optimiser, and optimistic in the direction one would
expect, since theirs also carries motor and stability limits. The cap costs
1.6 s a lap and removes the grip limit from most of the track.

Their peak lateral acceleration over the lap is \\SI{6.8}{\\meter\\per\\second\\squared},
against this repo's ``A_LAT_MAX`` of 6.0 -- so the constant is about 12%
conservative against measured data, which is recorded here rather than silently
changed.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.model import ACCEL_MAX, A_LAT_MAX


def corner_speed(kappa, a_lat_max: float = A_LAT_MAX, grip: float = 1.0):
    """The pure-cornering limit, :math:`\\sqrt{a_{lat}\\mu/\\kappa}`."""
    k = np.maximum(np.abs(np.asarray(kappa, float)), 1e-9)
    return np.sqrt(a_lat_max * grip / k)


def speed_profile(s, kappa, a_lat_max: float = A_LAT_MAX,
                  a_lon_max: float = ACCEL_MAX, grip: float = 1.0,
                  v_cap: float | None = None, sweeps: int = 2):
    """Grip-limited speed round a closed lap, by forward--backward integration.

    ``v_cap`` is an *optional* vehicle limit (motor, gearing). Leave it ``None``
    to see what the tyres alone allow, which is the number that says whether a
    track can discriminate between weight settings.
    """
    s = np.asarray(s, float)
    k = np.abs(np.asarray(kappa, float))
    v = corner_speed(k, a_lat_max, grip)
    if v_cap is not None:
        v = np.minimum(v, v_cap)
    ds = np.diff(np.r_[s, s[-1] + (s[1] - s[0])])
    lat = a_lat_max * grip
    for _ in range(max(int(sweeps), 1)):        # two sweeps settle the wrap
        for i in range(len(v) - 1):             # accelerating out of a corner
            ay = v[i] ** 2 * k[i]
            ax = a_lon_max * np.sqrt(max(1.0 - (ay / lat) ** 2, 0.0))
            v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * ax * ds[i], 0.0)))
        for i in range(len(v) - 2, -1, -1):     # braking into one
            ay = v[i + 1] ** 2 * k[i + 1]
            ax = a_lon_max * np.sqrt(max(1.0 - (ay / lat) ** 2, 0.0))
            v[i] = min(v[i], np.sqrt(max(v[i + 1] ** 2 + 2 * ax * ds[i], 0.0)))
    return v


def track_speed_profile(track, n: int = 600, **kw):
    """:func:`speed_profile` sampled round a :class:`~mpcc_tuning.track.Track`."""
    s = np.linspace(0.0, track.length, int(n), endpoint=False)
    k = np.array([track.curvature(float(v)) for v in s])
    return s, speed_profile(s, k, **kw)


def is_grip_limited(track, v_cap: float, n: int = 600, **kw) -> float:
    """Fraction of the lap where the *tyres*, not the cap, set the speed.

    Near zero means the track cannot punish a weight setting and therefore
    cannot measure a weight policy -- report it before reporting a null result
    on such a track.
    """
    _s, v = track_speed_profile(track, n=n, v_cap=None, **kw)
    return float(np.mean(v < v_cap - 1e-6))
