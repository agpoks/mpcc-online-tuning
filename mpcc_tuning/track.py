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

from pathlib import Path

import casadi as ca
import numpy as np


class Track:
    """Closed centreline with a constant half-width, as CasADi splines."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, half_width: float = 0.75,
                 ds: float = 0.1, pad: int = 8, w_left=None, w_right=None):
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

        # Variable corridor width, as two more splines in s. A real circuit is
        # not a constant-width ribbon -- the ICRA raceline data varies from
        # 0.69 m to 3.13 m round one lap, a factor of 4.5 -- and a controller
        # given one number for the whole track cannot be asked whether its
        # weights should depend on the width. self.half_width remains the
        # scalar fallback and the conservative summary of the lap.
        self.variable_width = w_left is not None and w_right is not None
        if self.variable_width:
            wl = np.interp(grid, np.linspace(0.0, self.length, len(w_left),
                                             endpoint=False), np.asarray(w_left, float))
            wr = np.interp(grid, np.linspace(0.0, self.length, len(w_right),
                                             endpoint=False), np.asarray(w_right, float))
            self.w_left_samples, self.w_right_samples = wl, wr
            self._wl = ca.interpolant("wl", "bspline", [s_ext.tolist()],
                                      wl[idx % n].tolist())
            self._wr = ca.interpolant("wr", "bspline", [s_ext.tolist()],
                                      wr[idx % n].tolist())
            self.half_width = float(np.percentile(np.minimum(wl, wr), 10.0))

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

    def width(self, s):
        """``(w_left, w_right)`` at arc length ``s``, symbolic or numeric.

        Falls back to the constant half-width when the track has no width data,
        so every caller can use this unconditionally.
        """
        if not getattr(self, "variable_width", False):
            return self.half_width, self.half_width
        sw = self.wrap(s)
        return self._wl(sw), self._wr(sw)

    def curvature_sym(self, s):
        """Curvature as a CasADi expression, for use inside the NLP.

        :meth:`curvature` evaluates numerically with a wide stencil; this is the
        same quantity built symbolically so a constraint can depend on it.
        """
        eps = 0.15
        a = self.tangent_angle(self.wrap(s - eps))
        b = self.tangent_angle(self.wrap(s + eps))
        d = ca.atan2(ca.sin(b - a), ca.cos(b - a))
        return d / (2.0 * eps)

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
    def from_centerline(path, scale: float = 1.0, half_width: float | None = None,
                        ds: float = 0.1, stride: int = 1) -> "Track":
        """A track from an ``f1tenth_racetracks`` centreline CSV.

        Format: ``# x_m, y_m, w_tr_right_m, w_tr_left_m``. This repo's
        :class:`Track` carries a *constant* half-width, so a variable-width
        circuit is reduced to its narrowest point unless ``half_width`` is given
        --- narrowest rather than mean, because the corridor constraint has to
        hold everywhere and a mean would put the car outside the real track at
        the tightest part.

        ``scale`` multiplies the geometry. It scales the lap length and every
        radius together, so it trades "how long is a lap" against "can the car
        take the tightest corner" and those cannot be set independently.
        """
        raw = np.genfromtxt(str(path), delimiter=",", comments="#")
        raw = raw[::max(int(stride), 1)]
        xy = raw[:, :2] * float(scale)
        if half_width is None:
            # A low PERCENTILE, not the strict minimum. An extracted centreline
            # has pinch points -- a single sample where the corridor momentarily
            # measures 0.10 m on a track whose median is 1.00 m -- and taking
            # the minimum hands the controller a corridor narrower than the car
            # (half-width 0.12 m), which is unsatisfiable everywhere. The
            # percentile is still conservative: it is narrower than 90% of the
            # lap. Pass half_width explicitly to override.
            half_width = (float(np.percentile(raw[:, 2:4], 10.0)) * float(scale)
                          if raw.shape[1] >= 4 else 0.75)
        return Track(xy[:, 0], xy[:, 1], half_width, ds=ds)

    @staticmethod
    def spielberg(scale: float = 1.0, half_width: float | None = None,
                  ds: float = 0.15) -> "Track":
        """The Red Bull Ring at F1TENTH's 1:10 scaling. A track nobody here designed.

        Every other track in this module is synthetic and built by hand, and a
        synthetic track can be built to suit the hypothesis being tested --- the
        per-sector result in ``docs/source/results.md`` is precisely a case of a
        conclusion moving between two tracks that were both designed here. A
        published circuit is the control for that.

        See ``mpcc_tuning/tracks/PROVENANCE.md``.
        """
        here = Path(__file__).resolve().parent / "tracks" / "Spielberg_centerline.csv"
        return Track.from_centerline(here, scale=scale, half_width=half_width, ds=ds)

    @staticmethod
    def icra2025(scale: float = 2.0, half_width: float | None = None,
                 ds: float = 0.1) -> "Track":
        """The ICRA 2025 competition track, from the team's own occupancy grid.

        A \SI{106.9}{\meter} serpentine over a 26.5 x 12.3 m footprint with a
        0.70 m median half-width -- a real circuit, driven by the real car this
        work is aimed at, and neither designed here nor scaled from a full-size
        one.

        The centreline was extracted from the ROS map beside it by
        ``tools/centerline_from_map.py``: 100% of it lies inside the corridor
        and it closes to 0.15 m. That extraction is the thing
        ``docs/source/plant.md`` had listed as missing.

        **``scale`` defaults to 2, and that is not cosmetic.** At 1:1 the two
        hairpins (at s = 18.9 m and s = 67.4 m -- real geometry, not seam
        artefacts) have a 0.59 m radius against a car whose geometric minimum
        turn radius is 0.78 m, so the map is not drivable as recorded.

        **And it needs ``r_d = 10``.** Swept open-loop at scale 2, 400 steps:

            q_v   r_d    covered   steps   outcome
            0.3   1.0     17.2 m     400   survived
            0.3  10.0      0.9 m     400   survived
            1.0   1.0     12.9 m     113   off
            1.0  10.0     17.0 m     400   survived
            2.0   1.0     12.9 m     101   off
            2.0  10.0     17.1 m     400   survived

        Every run at ``r_d = 10`` completes; the ``r_d = 1`` runs crash except
        at the slowest progress weight. That is the **same lever, and the same
        sentence**, as ``docs/source/plant.md`` reports for the fitted-tyre
        plant -- a steering-rate penalty produces a command the vehicle can
        actually follow, whether what it cannot follow is an unmodelled tyre or
        a hairpin tighter than its turning circle.

        Note the curvature column is not to be trusted here: the reported
        minimum radius goes 0.59, 0.76, 1.04 m for scales 1, 2, 3 where
        geometry says 0.59, 1.18, 1.77. ``Track.curvature``'s stencil is a
        fixed 0.6 m while the sample spacing grows with scale, so larger scales
        resolve more pixel wiggle and cancel the geometric gain. The drive test
        above is the trustworthy measurement.
        """
        here = Path(__file__).resolve().parent / "tracks" / "icra2025_centerline.csv"
        return Track.from_centerline(here, scale=scale, half_width=half_width, ds=ds)

    @staticmethod
    def icra_t1_raceline(scale: float = 1.0, ds: float = 0.1) -> "Track":
        """ICRA 2026 Track 1 from the team's optimised raceline, **variable width**.

        The real experiment, as against the synthetic tracks above which exist
        to demonstrate one thing at a time. \SI{71.7}{\meter}, corridor
        \SI{0.69}{}--\SI{3.13}{\meter} (a factor of 4.5 round one lap),
        minimum radius \SI{0.93}{\meter}, and the optimiser's own speed
        profile from 2.4 to \SI{6.1}{\meter\per\second}.

        Every other track here is a constant-width ribbon, and on one of those
        the question "should the weights depend on the corridor width" cannot
        be asked at all. See ``mpcc_tuning/tracks/PROVENANCE.md``.
        """
        return Track._raceline("icra_t1_raceline.csv", scale=scale, ds=ds,
                               map_stem="icra2026_t1")

    @staticmethod
    def _map_widths(centre, stem, max_m: float = 3.0):
        """Perpendicular half-width along ``centre``, from the occupancy grid.

        Returns the symmetric usable half-width -- min(left, right) at each
        point, since the corridor the controller may use is bounded by whichever
        wall is nearer. ``None`` if the map is not present, so a missing grid
        degrades to the raceline's own margins rather than failing.
        """
        import importlib.util
        here = Path(__file__).resolve().parent / "tracks"
        pgm, yml = here / f"{stem}.pgm", here / f"{stem}.yaml"
        if not pgm.exists() or not yml.exists():
            return None
        spec = importlib.util.spec_from_file_location(
            "cl", str(Path(__file__).resolve().parents[1] / "tools"
                      / "centerline_from_map.py"))
        cl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cl)
        im, res, org = cl.load(str(pgm), str(yml))
        H, W = im.shape
        occ = cl.connect_cone_rows(im <= 50)

        def blocked(x, y):
            c = int((x - org[0]) / res)
            r = int((org[1] + H * res - y) / res)
            if not (0 <= r < H and 0 <= c < W):
                return True
            return bool(occ[r, c])

        g = np.gradient(centre, axis=0)
        n = np.stack([g[:, 1], -g[:, 0]], axis=1) / np.linalg.norm(
            g, axis=1)[:, None]
        out = np.empty(len(centre))
        for i, (pt, nv) in enumerate(zip(centre, n)):
            side = []
            for sgn in (+1, -1):
                d = 0.0
                while d < max_m and not blocked(pt[0] + sgn * d * nv[0],
                                                pt[1] + sgn * d * nv[1]):
                    d += res
                side.append(d)
            out[i] = min(side)
        return out

    @staticmethod
    def icra_t2_raceline(scale: float = 1.0, ds: float = 0.1,
                         widen: float = 1.35) -> "Track":
        """ICRA 2026 Track 2 from the team's optimised raceline, variable width.

        \SI{73.8}{\meter}, corridor \SI{0.59}{}--\SI{3.76}{\meter} -- a factor
        of 6.4 round one lap, wider still than Track 1's 4.5 -- and the
        optimiser's own speed profile peaking at \SI{8.80}{\meter\per\second}.

        That peak is **above** this repo's ``SPEED_MAX`` of 8.0, so the cap is
        currently below what the team's own optimiser asks for on this track.
        Track 2 has no occupancy grid in the archive, but it needs none: the
        raceline carries ``x, y, w_left_m, w_right_m``, which is a corridor.

        Newest of 41 T2 runs in the archive. See ``tracks/PROVENANCE.md``.

        ``widen`` recovers the room the occupancy grid would show if T2 had
        one. On Track 1, where BOTH the raceline margins and the team's map are
        available, the map is wider by a measured 1.58x at the tightest point
        and 1.17x at the median -- the optimiser leaves its own safety margin
        inside a corridor that is really there. Track 2 has no grid, so the
        same correction is applied by proportion rather than by raycast.

        This is an approximation and is flagged as one: it assumes the
        optimiser was equally conservative on both tracks, which is plausible
        (same team, same tool, same week) but not measured. Set ``widen=1.0``
        for the raw margins.
        """
        return Track._raceline("icra_t2_raceline.csv", scale=scale, ds=ds,
                               widen=widen)

    @staticmethod
    def _raceline(fname: str, scale: float = 1.0, ds: float = 0.1,
                  smooth_m: float = 0.6, map_stem: str | None = None,
                  widen: float = 1.0) -> "Track":
        """Build a variable-width Track from one of the vendored raceline CSVs.

        Semicolon separated, the optimiser's own column names, with the
        corridor carried alongside the line as ``w_left_m``/``w_right_m``.
        """
        here = Path(__file__).resolve().parent / "tracks" / fname
        rows = [ln for ln in open(here) if not ln.startswith("#") and ln.strip()]
        hdr = rows[0].strip().split(";")
        col = {n: i for i, n in enumerate(hdr)}
        d = np.array([[float(v) for v in r.split(";")] for r in rows[1:]])
        xy = d[:, [col["x"], col["y"]]] * float(scale)
        wl = d[:, col["w_left_m"]] * float(scale)
        wr = d[:, col["w_right_m"]] * float(scale)

        # Drive the corridor CENTRE, not the raceline.
        #
        # The optimiser's line is the fastest way round, so it touches the
        # boundary at every apex: w_left or w_right is 0 there (and a shade
        # negative at a few points, from its own tolerance). Used directly as an
        # MPCC reference that is unusable -- the corridor rows pin the
        # contouring error to zero exactly where the car most needs room, and
        # the solver fails: measured at 1% of solves succeeding, -4.9 m covered,
        # and every run leaving the track.
        #
        # The raceline plus its margins is still a full description of the
        # corridor, so recover the centre from it: shift by (w_left - w_right)/2
        # along the normal and keep (w_left + w_right)/2 as the half-width. The
        # car then has symmetric room, which is what an MPCC is for, and the
        # optimiser's line is kept on the side as `raceline` for comparison.
        tang = np.gradient(xy, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
        nrm = np.column_stack([-tang[:, 1], tang[:, 0]])      # left normal
        half = 0.5 * (wl + wr)
        centre = xy + ((0.5 * (wl - wr))[:, None]) * nrm

        # Smooth the reconstructed centre, and smooth it in METRES.
        #
        # The offset (w_left - w_right)/2 carries the optimiser's own
        # point-to-point noise, and differentiating a noisy offset twice is what
        # curvature does. Unsmoothed, the reconstructed centre reached
        # |kappa| = 2.59 on T2 -- a 0.39 m radius, where the raceline's own
        # tightest is 0.93 m -- and the grip-limited speed constraint then
        # brakes the car to a standstill: measured at 0.095 m/s after 25 steps.
        # Curvature that is not in the track is still curvature to the solver.
        #
        # Boxcar over a window fixed in metres, wrapped, after the pattern in
        # MPCC's sibling project event-driven-rtrl (bridges/mapimport.py,
        # _smooth); copied and adapted rather than imported, since this repo
        # must not depend on that one.
        step = float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        w = max(int(round(smooth_m / max(step, 1e-9))) | 1, 3)
        k = np.ones(w) / w
        pad = np.vstack([centre[-(w // 2):], centre, centre[:w // 2]])
        centre = np.stack([np.convolve(pad[:, 0], k, "valid"),
                           np.convolve(pad[:, 1], k, "valid")], axis=1)
        vehicle_adjusted = True
        if map_stem is not None:
            # Measure the corridor from the OCCUPANCY GRID instead.
            #
            # The optimiser's w_left/w_right are conservative -- the room it
            # left for the car's centre, with its own safety margin already
            # taken out. Raycast perpendicular from the same centreline in the
            # team's own map and the corridor is wider: 0.90 m median
            # half-width against 0.72 m, and 0.55 m at the tightest point
            # against 0.35 m, which is 57% more room exactly where the car
            # needs it. The map is the track; the raceline margins are one
            # optimiser's opinion about how much of it to use.
            got = Track._map_widths(centre, map_stem)
            if got is not None:
                half = got
                vehicle_adjusted = False      # these ARE distances to the wall
        if widen != 1.0:
            # Scale the corridor toward what an occupancy grid would show. See
            # icra_t2_raceline: measured 1.58x tightest / 1.17x median on the
            # one track where both are available.
            half = half * float(widen)
        t = Track(centre[:, 0], centre[:, 1], ds=ds,
                  w_left=half, w_right=half)
        t.width_vehicle_adjusted = vehicle_adjusted
        t.raceline = xy
        # The optimiser's reference speed, for comparison rather than for use.
        t.v_ref = d[:, col["vx_mps"]]
        return t

    @staticmethod
    def icra2026_t1(scale: float = 1.0, half_width: float | None = None,
                    ds: float = 0.1) -> "Track":
        """ICRA 2026 Track 1, outer loop, from the competition occupancy grid.

        69.2 m, 1.00 m median half-width, extracted at 100% inside the corridor
        and closing to 0.07 m.

        **The corridor branches**, so it has no unique centreline -- an outer
        ring plus an inner section, and "the" centreline of a branching
        corridor is a choice of route rather than a property of the geometry.
        Three methods failed on exactly that before the framing changed: a loop
        is named by the *hole it encircles*, so choosing the hole chooses the
        loop. See ``centerline_around`` in ``tools/centerline_from_map.py``.
        The inner loop is ``hole_rank=1`` (16.2 m) if it is ever wanted.
        """
        here = Path(__file__).resolve().parent / "tracks" / "icra2026_t1_centerline.csv"
        return Track.from_centerline(here, scale=scale, half_width=half_width, ds=ds)

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
