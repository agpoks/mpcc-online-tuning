"""Learn the real per-sector cornering grip online, as a ceiling on k_v.

Motivation (TODO 2h/2i). The MPCC-critic policy over-claims grip: it drifts to
k_v = 0.69 and plans ~4 m/s into a hairpin that tops out at 2.78 m/s, the tyres
saturate (v^2*kappa > (D_F+D_R)/MASS = 10.79 m/s^2), the QP goes infeasible and
the car slides off. A FIXED k_v cap of 0.50 fixes it but throws away what we
actually want to learn: the real grip, which differs between corners and between
the test track and the race.

Adding a look-ahead friction CONSTRAINT to the OCP was measured to destabilise
this acados setup (TODO 2i: erratic low-speed crashes, breaks clean policies).
So the grip limit is enforced NOT in the solver but as a ceiling on the grip
claim the policy emits -- no OCP change, no solver strain. k_v feeds the
curvature-limited reference-speed profile, so a lower k_v already brakes BEFORE
the corner; clamping k_v by the sector AHEAD makes that look-ahead per-corner.

Learning the ceiling naively (creep up while clean, drop on a slip) LIMIT-CYCLES:
the k_v -> corner-speed map has a cliff (~0.50 clean, ~0.69 crash), so a per-tick
creep overshoots the cliff every clean lap, crashes, resets, and repeats
(measured, TODO 2j). Two design choices fix it:

1. **Per sector PASS, not per tick.** One update when the car leaves a sector,
   from the worst lateral demand seen in that pass -- so a long clean sector does
   not ratchet the ceiling up dozens of times in one lap.
2. **A crash ratchet (congestion-control style).** A crash means the active
   ceiling was too high for that sector: drop the sector's safe-frontier CAP
   below it and never creep back above the cap. Each crash lowers the cap toward
   the largest safe value, so it converges instead of oscillating.

No cross-repo dependency; k_v index from mpcc_tuning.mpcc.WEIGHT_NAMES.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.mpcc import WEIGHT_NAMES

#: physical lateral grip ceiling, (D_F + D_R) / MASS from mpcc_tuning.model
A_LAT_PHYS = (21.5829 + 24.2897) / 4.251     # = 10.79 m/s^2


class GripCeiling:
    """Per-sector-type learned ceiling on the grip claim k_v.

    Use ``clamp`` every tick and ``observe`` every tick; ``observe`` internally
    detects sector changes and commits one learning update per completed pass.

    Parameters
    ----------
    n_sectors : number of sector TYPES (feature one-hot width, 4 on T2).
    kv0       : initial ceiling, conservative (approach the limit from safety).
    kv_min/kv_max : bounds on the learned ceiling.
    up        : additive increase per CLEAN sector pass.
    down      : decrease per pass that neared the limit.
    down_crash: how far below the crashing ceiling the safe cap is dropped.
    lookahead : metres ahead to read the sector, so the clamp brakes for the
                corner being ENTERED.
    frac      : fraction of A_LAT_PHYS treated as "near the limit".
    """

    def __init__(self, n_sectors, kv0=0.50, kv_min=0.40, kv_max=0.95,
                 up=0.03, down=0.06, down_crash=0.10, lookahead=1.5, frac=0.80):
        self.n = int(n_sectors)
        self.kv = np.full(self.n, float(kv0))
        self.cap = np.full(self.n, float(kv_max))     # safe frontier per sector
        self.kv_min, self.kv_max = float(kv_min), float(kv_max)
        self.up, self.down, self.down_crash = float(up), float(down), float(down_crash)
        self.lookahead = float(lookahead)
        self.a_near = float(frac) * A_LAT_PHYS
        self.ik = WEIGHT_NAMES.index("k_v")
        self._cur = None          # sector currently being traversed
        self._amax = 0.0          # worst lateral demand seen in this pass

    # -- use ---------------------------------------------------------------
    def sector_ahead(self, track, s):
        return int(track.sector(track.wrap(float(s) + self.lookahead))) % self.n

    def clamp(self, theta, track, s):
        """Clamp emitted k_v (log space) to the ceiling of the sector ahead."""
        sec = self.sector_ahead(track, s)
        th = np.asarray(theta, float).copy()
        th[self.ik] = min(float(th[self.ik]), float(np.log(self.kv[sec])))
        return th, sec

    # -- learn -------------------------------------------------------------
    def _commit(self, sec, off):
        """One learning update for a completed sector pass."""
        if off:                                        # the ceiling was too high
            self.cap[sec] = max(self.kv_min, self.kv[sec] - self.down_crash)
            self.kv[sec] = self.cap[sec]
        elif self._amax >= self.a_near:                # neared the limit: back off
            self.kv[sec] = max(self.kv_min, self.kv[sec] - self.down)
        elif self._amax < 0.7 * self.a_near:           # grip to spare: creep up
            self.kv[sec] = min(self.cap[sec], self.kv[sec] + self.up)

    def observe(self, track, s, v, off=False, solver_fail=False):
        """Feed one tick; commits a pass update on sector change or a crash."""
        sec = int(track.sector(track.wrap(float(s)))) % self.n
        a_lat = float(v) ** 2 * abs(float(track.curvature(track.wrap(float(s)))))
        if solver_fail:
            a_lat = max(a_lat, self.a_near)            # a failed solve counts as at-limit
        if self._cur is None:
            self._cur = sec
        if off or sec != self._cur:                    # pass ended (or crashed)
            self._amax = max(self._amax, a_lat)
            self._commit(self._cur, off)
            self._cur = sec
            self._amax = 0.0
        else:
            self._amax = max(self._amax, a_lat)
        return a_lat

    def reset_pass(self):
        self._cur = None
        self._amax = 0.0

    def snapshot(self):
        return self.kv.copy()
