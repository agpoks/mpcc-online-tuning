"""A closed centreline, exposed to CasADi as a differentiable function of arc length.

MPCC needs the reference path as a *smooth function of a progress variable*,
because the progress variable is a decision variable: the optimiser moves along
the path as part of the solve, so the path has to be differentiable with
respect to it. That rules out the nearest-sample lookup you would use in a
simulator, and rules it out for a real reason, not a stylistic one.

Here that function is a periodic B-spline through the centreline samples, built
with ``casadi.interpolant`` so it can appear directly in the NLP and be
differentiated by the same machinery that differentiates everything else.
"""

from __future__ import annotations

import casadi as ca
import numpy as np


class Track:
    """Closed centreline with a constant half-width, as CasADi splines."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, half_width: float = 0.75,
                 ds: float = 0.1, pad: int = 8):
        pts = np.stack([np.asarray(xs, float), np.asarray(ys, float)], axis=1)
        # Resample to uniform arc length first. The progress variable in MPCC
        # *is* arc length, so a spline parametrised by anything else quietly
        # makes "v_s" not a speed.
        seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s_raw[-1])
        n = max(int(self.length / ds), 32)
        self.ds = self.length / n
        grid = np.arange(n) * self.ds
        closed = np.vstack([pts, pts[:1]])
        self.center = np.stack([np.interp(grid, s_raw, closed[:, 0]),
                                np.interp(grid, s_raw, closed[:, 1])], axis=1)
        self.s = grid
        self.half_width = float(half_width)

        # Extend the data by `pad` samples of the *wrapped* path on each side,
        # so the spline is smooth across the start/finish line instead of
        # flattening there. Evaluation always happens on [0, length) after
        # wrapping, so the padding is only ever used by the B-spline's support
        # and never extrapolated into.
        idx = np.concatenate([np.arange(-pad, 0), np.arange(n), np.arange(n, n + pad)])
        s_ext = idx * self.ds
        p_ext = self.center[idx % n]
        self._x = ca.interpolant("cx", "bspline", [s_ext.tolist()], p_ext[:, 0].tolist())
        self._y = ca.interpolant("cy", "bspline", [s_ext.tolist()], p_ext[:, 1].tolist())

    def wrap(self, s):
        """Arc length into ``[0, length)``, differentiably.

        ``floor`` has zero derivative almost everywhere, so ``d(wrap(s))/ds = 1``
        wherever it matters and the solver sees a path that simply continues
        past the finish line. Without this the progress variable runs off the
        end of the spline's knots within one horizon, the derivatives come back
        NaN, and IPOPT reports failure with no useful message.
        """
        return s - self.length * ca.floor(s / self.length)

    # -- CasADi-side ------------------------------------------------------
    def pos(self, s):
        """Reference position at arc length ``s`` (symbolic or numeric)."""
        sw = self.wrap(s)
        return ca.vertcat(self._x(sw), self._y(sw))

    def tangent_angle(self, s):
        """Heading of the reference path at ``s``, by differentiating the spline."""
        eps = 1e-3
        d = (self.pos(s + eps) - self.pos(s - eps)) / (2 * eps)
        return ca.atan2(d[1], d[0])

    def errors(self, x, y, s):
        """Contouring and lag error, the two quantities MPCC is built around.

        With ``phi`` the path heading at ``s``, the vector from the reference
        point to the car splits into a component *across* the path (contouring
        error, what you want small) and one *along* it (lag error, which exists
        only because ``s`` is an optimisation variable and may run ahead of or
        behind the car's true projection). Penalising lag is what keeps ``s``
        honest; without it the optimiser wins by racing the progress variable
        forward and leaving the car behind.
        """
        p = self.pos(s)
        phi = self.tangent_angle(s)
        dx, dy = x - p[0], y - p[1]
        e_c = ca.sin(phi) * dx - ca.cos(phi) * dy
        e_l = -ca.cos(phi) * dx - ca.sin(phi) * dy
        return e_c, e_l

    # -- numpy-side -------------------------------------------------------
    def project(self, x: float, y: float) -> float:
        """Nearest arc length, by sampling. Used to initialise ``s``, not inside the NLP."""
        d2 = (self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2
        return float(self.s[int(np.argmin(d2))])

    def lateral(self, x: float, y: float) -> float:
        k = int(np.argmin((self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2))
        nxt = (k + 1) % len(self.center)
        t = self.center[nxt] - self.center[k]
        t = t / (np.linalg.norm(t) + 1e-12)
        r = np.array([x, y]) - self.center[k]
        return float(-r[0] * t[1] + r[1] * t[0])

    def curvature(self, s) -> float:
        """Signed curvature of the centreline at arc length ``s``, 1/m.

        Needed because a single set of cost weights for a whole lap is a
        modelling choice, not a fact: a straight wants progress weighted
        heavily and steering barely penalised, a hairpin wants the opposite.
        Curvature is what tells them apart, and the MPCC already has the path,
        so it costs a finite difference rather than an estimator.
        """
        # A wider stencil than the grid: where two arcs meet, the geometric
        # curvature steps discontinuously and a tight finite difference reports
        # a radius the car cannot physically take (0.26 m against a 0.78 m
        # minimum turn radius). The car does not experience that step, and a
        # scheduler keyed on it would switch segment for one tick at a junction.
        h = max(6.0 * self.ds, 0.4)
        a = float(self.tangent_angle(self.wrap(s - h)))
        b = float(self.tangent_angle(self.wrap(s + h)))
        d = np.arctan2(np.sin(b - a), np.cos(b - a))
        return float(d / (2.0 * h))

    def segment_edges(self, n: int = 3, samples: int = 400):
        """Curvature thresholds splitting *this* track into ``n`` equal parts.

        Fixed radius thresholds do not transfer. Edges calibrated for a large
        circuit put 92% of a 5 m test track into one bin and leave another
        empty, which makes a per-segment scheduler look useless for a reason
        that has nothing to do with scheduling.

        Quantiles of the track's own curvature give a balanced split on any
        geometry, and match what "straight, long curve, hairpin" means in
        practice: relative to the lap you are driving. Cached -- the track does
        not change.
        """
        if getattr(self, "_seg_edges", None) is None or len(self._seg_edges) != n - 1:
            ss = np.linspace(0.0, self.length, samples, endpoint=False)
            k = np.abs([self.curvature(v) for v in ss])
            self._seg_edges = np.quantile(k, np.linspace(0, 1, n + 1)[1:-1])
        return self._seg_edges

    def segment(self, s, n: int = 3) -> int:
        """0 straight, 1 long curve, 2 hairpin -- by this track's own quantiles."""
        return int(np.searchsorted(self.segment_edges(n), abs(self.curvature(s))))

    # -- named sectors ----------------------------------------------------
    def corners(self, kappa_frac: float = 0.10, samples: int = 900):
        """Corners as maximal runs of high curvature, each with its *total* turn.

        This exists because :meth:`segment` **cannot** express the distinction
        the name "90-degree corner" implies, and no retuning of its quantiles
        will fix that. Curvature is :math:`1/R` for an arc of radius ``R``
        *however far it sweeps*, so a 90-degree corner and a 180-degree hairpin
        of the same radius have identical curvature at every point inside them.
        Measured on a track carrying both at ``R = 1.5`` m: mean ``|kappa|``
        0.651 against 0.662, landing in the same bins.

        What separates them is the **integral** of curvature through the corner
        -- the total heading change -- so a corner has to be detected as an
        extended object first and classified second. That is what this does.

        Returns a list of ``(s_start, s_end, dpsi, kappa_peak)``, ``dpsi`` signed
        in radians.

        The entry threshold is a *fraction of this track's own* peak curvature
        rather than an absolute radius, for the reason
        :meth:`segment_edges` already documents: fixed radius thresholds do not
        transfer between a 5 m test track and a circuit, and calibrating them
        for one makes the other degenerate.

        The fraction has to be **small**, and 0.25 was measured to be too large.
        A lap containing one tight hairpin sets the peak, and at 0.25 every
        gentler corner falls below the threshold and is reported as straight --
        while the corners that *are* detected have their entry and exit ramps
        clipped, so their total turn reads low. On the four-sector circuit that
        turned a designed -90 degrees into a measured -59 and lost both
        sweepers entirely. At 0.10 the tails are kept and the totals land within
        a few degrees of the design.
        """
        ss = np.linspace(0.0, self.length, samples, endpoint=False)
        ds = self.length / samples
        k = np.array([self.curvature(v) for v in ss])
        thr = kappa_frac * np.abs(k).max()
        on = np.abs(k) > thr
        if not on.any():
            return []
        # Rotate so index 0 starts outside a corner, otherwise a corner
        # straddling the start/finish line is reported as two.
        shift = int(np.argmin(on))
        on_r, k_r = np.roll(on, -shift), np.roll(k, -shift)
        out, i = [], 0
        while i < samples:
            if not on_r[i]:
                i += 1
                continue
            j = i
            while j < samples and on_r[j]:
                j += 1
            dpsi = float(np.sum(k_r[i:j]) * ds)
            out.append((float(ss[(i + shift) % samples]),
                        float(ss[(j - 1 + shift) % samples]),
                        dpsi, float(np.abs(k_r[i:j]).max())))
            i = j
        return out

    #: Total-turn thresholds, in degrees, separating the three corner classes.
    SECTOR_EDGES_DEG = (60.0, 135.0)

    #: 0 straight, 1 long curve, 2 ninety, 3 one-eighty.
    SECTOR_NAMES = ("straight", "long curve", "90-deg", "180-deg")

    def sector(self, s, kappa_frac: float = 0.10) -> int:
        """Named sector at arc length ``s``: 0 straight, 1 long, 2 ninety, 3 one-eighty.

        Unlike :meth:`segment`, **the whole corner carries one label** -- the
        classification is a property of the corner as an object, not of the
        point. That is the behaviour a weight schedule wants: the label must not
        flicker part-way through a corner as the pointwise curvature wanders
        across a bin edge.
        """
        if getattr(self, "_sector_cache", None) is None:
            lo, hi = (np.deg2rad(d) for d in self.SECTOR_EDGES_DEG)
            cls = []
            for s0, s1, dpsi, _kp in self.corners(kappa_frac):
                a = abs(dpsi)
                cls.append((s0, s1, 1 if a < lo else (2 if a < hi else 3)))
            self._sector_cache = cls
        sw = float(s) % self.length
        for s0, s1, c in self._sector_cache:
            inside = (s0 <= sw <= s1) if s0 <= s1 else (sw >= s0 or sw <= s1)
            if inside:
                return c
        return 0

    @staticmethod
    def circuit(half_width: float = 0.75, ds: float = 0.1) -> "Track":
        """A closed lap containing all four sector types, by construction.

        The oval has straights and 180s only; ``mixed`` is a smooth harmonic
        with no 90s. Neither can test a four-way schedule, so this builds one
        explicitly: straights, a chicane of two 90-degree corners, two
        180-degree hairpins, and a pair of long sweepers.

        Two constraints make this less free than it looks, and getting either
        wrong produces a track that *looks* fine and is not.

        **The turns must sum to exactly 360 degrees.** A closed loop's heading
        returns to where it started. A first version of this summed to 480, the
        constructor force-closed the gap, and the seam produced curvature that
        the corner detector duly reported as two corners that were never
        designed. That budget is why the corner types come in balancing pairs.

        **Every radius must be one the car can take.** The geometric minimum
        turn radius is ``WHEELBASE / tan(STEER_MAX)`` = 0.78 m, and ``mixed``'s
        1.76 m is already tight enough that the default weights do not survive
        it. The tightest corner here is 2.5 m, matching the oval's 2.46 m, so
        that a four-sector *scheduling* result is not confounded by the
        initialisation failure documented in ``docs/source/results.md``.

        Built by integrating a heading profile rather than by stitching arcs to
        lines: stitched joins are only C0 unless every tangent is matched by
        hand, and one mismatched tangent gives a curvature spike reporting a
        radius the car cannot take. Integrating ``psi(s)`` makes the tangent
        continuous by construction.
        """
        # (radius, signed turn) for corners; radius None for a straight of the
        # given nominal length. Turns sum to -2*pi exactly -- see below.
        # A paperclip with a chicane and a sweeper pair. The shape is forced by
        # the arithmetic: a 180 cannot appear in a closed lap without something
        # turning back the other way, so the corner types have to be chosen in
        # *balancing pairs* rather than picked off a wish list. Two hairpins
        # give +360 on their own, and the 90s and the sweepers are therefore
        # each a matched pair that nets to zero -- a chicane and an S.
        P = np.pi
        # Found by searching all 720 orderings of the six corners against every
        # constraint at once (closure, positive straights, same-sign corners
        # separated, no self-intersection); hand-picking an order does not work,
        # and two earlier hand-picked ones failed in different ways.
        plan = [(None, 2.0), (4.2, -P / 3),            # straight, sweeper
                (None, 2.0), (2.6, P),                 # straight, 180 hairpin
                (None, 2.0), (4.2, P / 3),             # straight, sweeper back
                (None, 2.0), (2.6, -P / 2),            # straight, 90
                (None, 2.0), (2.6, P),                 # straight, 180 hairpin
                (None, 2.0), (2.6, P / 2),             # straight, 90
                (None, 2.0)]
        turn = sum(t for r, t in plan if r is not None)
        # Exactly one full turn. Not zero: a closed curve's total turning is a
        # multiple of 360, and the *balanced* set (+180,-180,+90,-90,+60,-60)
        # sums to zero, which closes as a self-intersecting figure-eight. That
        # is a perfectly good closed curve and a useless racetrack, so the two
        # hairpins have to turn the same way and the 90s and sweepers pair off
        # around them.
        assert abs(turn - 2 * np.pi) < 1e-9, f"turns sum to {np.degrees(turn):.1f} deg"

        # Closing the *heading* is not closing the *curve*. Turns summing to
        # 360 makes the tangent come back; the position comes back only if the
        # displacement integrates to zero too, which is two further conditions.
        # Getting it wrong is silent: the constructor bridges the gap with a
        # chord, and the chord's ends read as a pair of sharp corners that were
        # never designed (measured: two 150-degree corners at R = 0.44 m).
        #
        # The displacement is *linear* in the straight lengths -- each straight
        # contributes L_i (cos psi_i, sin psi_i), and psi_i is fixed by the
        # corners before it, not by any L -- so this is a 2 x n_straight system.
        psi0, C, cols, nominal = 0.0, np.zeros(2), [], []
        for r, t in plan:
            if r is None:
                cols.append([np.cos(psi0), np.sin(psi0)])
                nominal.append(t)
            else:
                # SIGNED radius. The arc displacement formula is written in
                # terms of R = 1/kappa, which carries the turn's sign; the
                # unsigned r mirrors every right-hand corner. That bug closed a
                # model of a *different* track -- residual 2e-15 while the real
                # geometry missed by 2.17 m.
                rs = r * np.sign(t)
                C += rs * np.array([np.sin(psi0 + t) - np.sin(psi0),
                                    np.cos(psi0) - np.cos(psi0 + t)])
                psi0 += t
        A = np.array(cols).T                       # 2 x n_straight
        L = np.array(nominal, float)
        # Straights must stay long enough to separate two same-sign corners:
        # Track.curvature uses a 0.6 m stencil, so a shorter straight never lets
        # the curvature fall back to zero and the detector reports one corner
        # where there are two (measured: a 180 and a 60 merged into +239.5 deg).
        # Alternating projection -- reproject onto the closure constraint, clip
        # to the floor, repeat -- which needs numpy alone.
        pinv, lmin = np.linalg.pinv(A), 1.4
        for _ in range(500):
            L = np.maximum(L + pinv @ (-C - A @ L), lmin)
        L = L + pinv @ (-C - A @ L)
        assert (L > lmin - 1e-6).all() and np.linalg.norm(A @ L + C) < 1e-9, \
            f"circuit layout does not close: {L}"

        # Compose the exact arcs and lines. An earlier version integrated a
        # discretised curvature array instead, and the per-piece rounding of
        # sample counts accumulated into a **2.18 m** closure error -- which the
        # constructor then bridged with a chord, and the chord read as two
        # spurious 150-degree corners at R = 0.44 m. Composing exact geometry
        # closes to machine precision and removes the failure entirely.
        #
        # Joining an arc to a line at a shared tangent is C1 by construction --
        # it is a fillet, not the hand-stitched C0 join ``mixed`` warns about --
        # and the periodic B-spline the constructor fits smooths the curvature
        # step, which is a feature a real circuit has anyway.
        pos, psi0, i, xs, ys = np.zeros(2), 0.0, 0, [], []
        for r, t in plan:
            if r is None:
                m = max(int(round(L[i] / ds)), 2)
                for j in range(m):
                    q = pos + (L[i] * j / m) * np.array([np.cos(psi0), np.sin(psi0)])
                    xs.append(q[0]); ys.append(q[1])
                pos = pos + L[i] * np.array([np.cos(psi0), np.sin(psi0)])
                i += 1
            else:
                sgn = np.sign(t)
                # Centre is a signed radius to the left of the current heading.
                c = pos + sgn * r * np.array([-np.sin(psi0), np.cos(psi0)])
                a0 = np.arctan2(*(pos - c)[::-1])
                m = max(int(round(abs(t) * r / ds)), 2)
                for j in range(m):
                    ang = a0 + t * j / m
                    xs.append(c[0] + r * np.cos(ang)); ys.append(c[1] + r * np.sin(ang))
                pos = c + r * np.array([np.cos(a0 + t), np.sin(a0 + t)])
                psi0 += t
        x, y = np.array(xs), np.array(ys)
        assert np.hypot(*(pos - np.zeros(2))) < 1e-9, \
            f"circuit does not close: {np.hypot(*pos):.4f} m"
        return Track(x, y, half_width, ds=ds)

    @staticmethod
    def mixed(scale: float = 5.2, lobe: float = 0.45, harmonic: int = 3,
              half_width: float = 0.75, ds: float = 0.2) -> "Track":
        """A lap with straights, long curves and a hairpin, smooth by construction.

        The oval has only two segment types and cannot test whether per-segment
        weights help. The obvious fix -- stitching arcs and straights together
        -- is worse than it looks: the joins are only C0 unless every tangent is
        matched by hand, and a single 180-degree tangent flip at one junction
        produces a curvature spike reporting a radius of 0.77 m against a
        geometric minimum turn radius of 0.78 m. The scheduler then switches
        segment for one tick at a place the car never actually goes.

        A closed harmonic radius, :math:`R(u) = s(1 + \ell\cos(nu))`, is
        periodic and smooth by construction, so there are no joins to get wrong,
        and ``lobe`` sets how tight the tightest corner is. ``lobe=0.30`` gives
        a minimum radius near 1.2 m, comfortably inside the car's 0.78 m limit.
        """
        # lobe=0.45, harmonic=3: minimum radius 1.76 m, comfortably inside the
        # car's 0.78 m limit, with curvature spread over more than a decade.
        u = np.linspace(0.0, 2.0 * np.pi, 1400, endpoint=False)
        r = scale * (1.0 + lobe * np.cos(harmonic * u))
        return Track(r * np.cos(u), r * np.sin(u), half_width, ds=ds)

    @staticmethod
    def oval(length: float = 16.0, width: float = 5.0, half_width: float = 0.75,
             ds: float = 0.2) -> "Track":
        """The same rounded rectangle as ``rtrrl-playground``'s ``lanekeep``."""
        r = width / 2.0
        straight = max(length - width, 1e-3) / 2.0
        n_s, n_c = max(int(straight / ds), 2), max(int(np.pi * r / ds), 4)
        xs, ys = [], []
        for i in range(n_s):
            xs.append(-straight / 2 + straight * i / n_s); ys.append(-r)
        for i in range(n_c):
            a = -np.pi / 2 + np.pi * i / n_c
            xs.append(straight / 2 + r * np.cos(a)); ys.append(r * np.sin(a))
        for i in range(n_s):
            xs.append(straight / 2 - straight * i / n_s); ys.append(r)
        for i in range(n_c):
            a = np.pi / 2 + np.pi * i / n_c
            xs.append(-straight / 2 + r * np.cos(a)); ys.append(r * np.sin(a))
        return Track(np.array(xs), np.array(ys), half_width)
