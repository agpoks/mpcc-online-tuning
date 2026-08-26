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
