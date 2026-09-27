"""Other cars on the track, for the MPCC's keep-out constraint to see.

Deliberately dumb. An opponent here drives the centreline at a constant speed
with a fixed lateral offset -- it does not react, does not defend, and does not
have a controller. That is the right first opponent for the question this repo
is asking, which is not "can the MPCC race" but **"is overtake-vs-follow
expressible as a choice of cost weights"**. A reactive opponent would make the
outcome depend on two policies at once and make that question harder to read,
not easier.

The output that matters is :meth:`Opponent.keepout`, an ``(x, y, r)`` circle
fed straight to :meth:`mpcc_tuning.mpcc.MPCC.set_obstacles`. The radius is the
sum of the two cars' half-widths -- the MPCC predicts a point mass, so the
opponent's circle has to carry the ego car's body as well, exactly as the
``obs_margin`` in the acados template does.

## The sign convention, which is not obvious

The track exposes two lateral measures with **opposite signs**:
``Track.errors`` returns a contouring error ``e_c``, and ``Track.lateral``
returns ``-e_c``. Off-track is judged by ``lateral``, so ``offset`` here is in
``lateral``'s convention: ``offset = +0.3`` means an opponent sitting where the
plant would report ``lateral = +0.3``. ``tests/test_opponents.py`` asserts it,
because getting this backwards puts the opponent on the wrong side of the track
and everything still runs.
"""

from __future__ import annotations

import numpy as np


class ObstacleTracker:
    """Estimate whether an obstacle is moving, from its positions alone.

    This exists because **"stay behind" is only a behaviour if the obstacle is
    dynamic.** Against a static one -- a cone, a stopped car, debris -- staying
    behind means stopping forever, and the only options are to go around or to
    park. So the posture has to be conditioned on a classification, and that
    classification is not given.

    **It cannot be made from one frame.** A stopped car and a slow car are
    identical in a single observation; only their positions *over time* differ.
    That makes static-versus-dynamic a genuinely temporal decision, and unlike a
    closing rate it gates the entire behaviour choice rather than tuning it.

    On the vehicle this is what ``datmo`` (detection and tracking of moving
    objects) provides, with the same caveats: it is an *estimate*, it is late,
    and it is sometimes wrong. Here it is an exponentially-weighted speed over
    observed positions -- the crudest thing that works, and honest about being
    an estimate rather than the true ``Opponent.speed``.
    """

    def __init__(self, dt: float, tau: float = 0.4, moving_thresh: float = 0.25):
        self.dt, self.a = float(dt), float(np.exp(-float(dt) / max(tau, 1e-6)))
        self.moving_thresh = float(moving_thresh)
        self.reset()

    def reset(self) -> None:
        self._prev = None
        self.speed = 0.0
        self.n = 0

    def update(self, xy) -> float:
        """One observation. Returns the current speed estimate."""
        q = np.asarray(xy, float)[:2]
        if self._prev is not None:
            v = float(np.linalg.norm(q - self._prev) / self.dt)
            self.speed = self.a * self.speed + (1.0 - self.a) * v
            self.n += 1
        self._prev = q
        return self.speed

    @property
    def is_dynamic(self) -> bool:
        """``True`` once the evidence says it is moving.

        Deliberately requires a few observations: with ``n`` small the estimate
        is one finite difference, and a single noisy frame would flip the whole
        behaviour. That warm-up is the cost of the decision being temporal, and
        it is not an implementation detail -- it is what a one-frame policy
        cannot pay.
        """
        return self.n >= 3 and self.speed > self.moving_thresh


class Opponent:
    """A car driving the centreline at constant speed, at a fixed offset."""

    def __init__(self, track, s0: float = 3.0, speed: float = 1.0,
                 offset: float = 0.0, radius: float = 0.24):
        self.track = track
        self.s0, self.speed, self.offset = float(s0), float(speed), float(offset)
        # Half-width of ego plus half-width of opponent: the MPCC predicts a
        # point, so the whole of both bodies lives in this radius.
        self.radius = float(radius)
        self.reset()

    def reset(self) -> None:
        self.s = self.s0

    def step(self, dt: float) -> None:
        self.s = (self.s + self.speed * float(dt)) % self.track.length

    def pose(self) -> np.ndarray:
        """``[x, y, psi]`` of the opponent right now."""
        s = self.s % self.track.length
        p = np.array(self.track.pos(s)).ravel()
        psi = float(self.track.tangent_angle(s))
        # lateral()'s normal, not errors()'s -- see the module docstring.
        n = np.array([-np.sin(psi), np.cos(psi)])
        return np.array([p[0] + self.offset * n[0], p[1] + self.offset * n[1], psi])

    def keepout(self) -> tuple:
        """``(x, y, r)`` for :meth:`MPCC.set_obstacles`."""
        x, y, _ = self.pose()
        return (float(x), float(y), self.radius)


class RacelineOpponent:
    """A FAIR opponent: follows the raceline at a GRIP-LIMITED speed (it slows for
    corners like a real car, same lateral-accel limit), and reactively steps to the
    open side to OVERTAKE when it is faster and blocked behind the ego -- rather than
    the dumb :class:`Opponent`, which drives the centreline at constant speed through
    hairpins a real car could not hold and simply rams a slower ego.

    ``pace`` is the target straight-line speed (m/s); in corners the speed is capped at
    ``sqrt(a_lat * mu / |kappa|)`` so cornering respects grip. ``a_lat`` is the
    opponent's lateral-accel budget (2.5 m/s^2 by default -- a touch above the ego's
    conservative START line so a "faster" opponent really is faster on the straights
    without being a physics-free ghost). ``step(dt, ego)`` takes the ego pose+speed so
    the reactive overtake can see it.
    """

    def __init__(self, track, s0: float = 3.0, pace: float = 1.2, offset: float = 0.0,
                 radius: float = 0.24, a_lat: float = 2.5, mu: float = 1.0):
        self.track = track
        self.s0, self.pace, self.offset0 = float(s0), float(pace), float(offset)
        self.radius, self.a_lat, self.mu = float(radius), float(a_lat), float(mu)
        self.reset()

    def reset(self) -> None:
        self.s = self.s0
        self.offset = self.offset0
        self.speed = self.pace

    def _vlimit(self, s: float) -> float:
        kap = abs(float(self.track.curvature(self.track.wrap(s))))
        return float(min(self.pace, (self.a_lat * self.mu / max(kap, 1e-3)) ** 0.5))

    def step(self, dt: float, ego=None) -> None:
        v = self._vlimit(self.s)
        # MUTUAL avoidance, done REALISTICALLY: when the ego is close the opponent eases to
        # ONE side by a modest CLEARANCE (not slamming into the wall), with hysteresis so it
        # does not flap between sides, a limited lateral RATE, and -- crucially -- a GRIP COST:
        # moving sideways uses tyre grip that is then unavailable to go forward, so a lateral
        # manoeuvre SLOWS the car (a real car sliding across the track loses speed).
        #
        # SAFETY INVARIANT (a racing opponent may not crash into us on purpose): whenever the ego
        # is alongside (arc-length overlap), the opponent's lateral step is CLAMPED so it can only
        # move AWAY from the ego's side or hold -- never toward it. This makes non-ramming a hard
        # guarantee, not just a property of the yield heuristic: even if the recentre pull toward
        # offset0 or a mis-picked side would close the lateral gap, the clamp forbids it. A contact
        # can then only come from the EGO closing on a yielding car, never from the opponent
        # steering in. tests/test_obstacles.py asserts the invariant.
        CLEAR = 0.45                                  # how far to ease aside [m] (not to the wall)
        LAT_RATE = 0.8                                # max lateral speed [m/s] (grip-limited)
        DEAD = 0.05                                   # ego-lateral deadband for "which side is it on"
        tgt = self.offset0
        ego_lat = None; alongside = False
        if ego is not None:
            ex, ey, _ = ego
            es = self.track.project(float(ex), float(ey))
            d = (es - self.s) % self.track.length
            gap = d - self.track.length if d > self.track.length / 2 else d
            alongside = abs(gap) < 3.5
            if alongside:
                ego_lat = float(self.track.lateral(float(ex), float(ey)))
                wl, wr = self.track.width(self.track.wrap(self.s))
                side = -1.0 if ego_lat >= DEAD else (1.0 if ego_lat <= -DEAD else np.sign(self.offset) or 1.0)
                room = (float(wr) if side > 0 else float(wl)) - self.radius - 0.05
                tgt = side * min(CLEAR, max(room, 0.0))
        step_lat = float(np.clip(tgt - self.offset, -LAT_RATE * dt, LAT_RATE * dt))
        # invariant: alongside the ego, the opponent may not steer INTO it. If the ego is on the
        # +lateral side it may only move - (away) or hold; on the -side, only + or hold.
        if alongside and ego_lat is not None:
            if ego_lat > DEAD:
                step_lat = min(step_lat, 0.0)
            elif ego_lat < -DEAD:
                step_lat = max(step_lat, 0.0)
        self.offset += step_lat
        lat_speed = abs(step_lat) / max(dt, 1e-3)
        self.speed = float(max(0.0, v * (1.0 - 0.4 * lat_speed / max(v, 0.3))))   # grip cost of moving sideways
        self.s = (self.s + self.speed * float(dt)) % self.track.length

    def pose(self) -> np.ndarray:
        s = self.s % self.track.length
        p = np.array(self.track.pos(s)).ravel()
        psi = float(self.track.tangent_angle(s))
        n = np.array([-np.sin(psi), np.cos(psi)])
        return np.array([p[0] + self.offset * n[0], p[1] + self.offset * n[1], psi])

    def keepout(self) -> tuple:
        x, y, _ = self.pose()
        return (float(x), float(y), self.radius)
