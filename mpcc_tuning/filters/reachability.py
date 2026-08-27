"""Viability kernel: decide safety by table lookup, computed once offline.

References
----------
Bansal, Chen, Herbert & Tomlin, *"Hamilton-Jacobi Reachability: A Brief
Overview and Recent Advances"*, CDC 2017 -- the HJ formulation.

Mitchell, Bayen & Tomlin, *"A time-dependent Hamilton-Jacobi formulation of
reachable sets for continuous dynamic games"*, IEEE TAC 2005 -- the level-set
method the kernel below is a discrete stand-in for.

Aubin, *Viability Theory*, 1991 -- the viability kernel itself.

The idea
--------
Everything else in this package decides safety *online*, by predicting. This
decides it *offline*, once, and then looks the answer up.

The **viability kernel** is the set of states from which some input sequence
keeps the system inside the constraints forever:

.. math::
    \\mathrm{Viab}(\\mathcal{X}) = \\{x_0 : \\exists\\, u(\\cdot),\;
        x_k \\in \\mathcal{X} \;\; \\forall k \\ge 0\\}

It is computed by the fixed-point iteration

.. math::
    V_0 = \\mathcal{X}, \\qquad
    V_{i+1} = \\{x \\in V_i : \\exists u \\in \\mathcal{U},\; f(x,u) \\in V_i\\}

which shrinks monotonically and converges. The filter is then one membership
test per candidate input -- no rollout, no horizon, no backup policy. That is
the appeal: **the online cost is a table lookup**, and the answer is exact
rather than a sufficient condition.

Why it is tractable here
------------------------
The full state is :math:`(x, y, \\psi, v)`, and gridding that finely enough is
expensive. But the track is a corridor, and what matters for staying inside it
is not *where* the car is on the lap but where it is *across* the lap. In
path-relative coordinates the state collapses to

.. math::
    (d,\; e_\\psi,\; v) \\quad\\text{-- lateral offset, heading error, speed}

which is three dimensions and grids comfortably. The price is that the kernel
is computed for a **single curvature**, so it is exact on a constant-radius
corner and conservative-or-wrong elsewhere; ``curvature`` selects which. Using
the tightest curvature on the track makes it conservative everywhere, which is
the setting that is honest to ship.

This is a discrete dynamic-programming stand-in for a proper HJ solve, not a
substitute for one -- there is no level-set function and no numerical
Hamiltonian, so accuracy is set by the grid rather than by a PDE scheme.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.filters.base import SafetyFilter
from mpcc_tuning.model import ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, WHEELBASE, A_LAT_MAX


class ViabilityFilter(SafetyFilter):
    """Safety by membership in a precomputed kernel over ``(d, e_psi, v)``."""

    def __init__(self, track, dt: float = 0.05, margin: float = 0.18,
                 curvature: float | None = None, assumed_grip: float = 1.0,
                 n_d: int = 41, n_e: int = 41, n_v: int = 21, n_u: int = 9,
                 iters: int = 60, credit: str = "executed",
                 wheelbase: float = WHEELBASE):
        super().__init__(track, dt=dt, margin=margin, credit=credit)
        self.assumed_grip, self.wheelbase = float(assumed_grip), float(wheelbase)
        if curvature is None:
            # The tightest corner on the track: the kernel is then conservative
            # everywhere else, which is the only defensible single-curvature choice.
            curvature = float(np.max(np.abs(getattr(track, "curvature",
                                                    np.array([1 / 2.5])))))
        self.curvature = float(curvature)
        self.half = track.half_width - self.margin
        self.d_grid = np.linspace(-self.half, self.half, n_d)
        self.e_grid = np.linspace(-0.9, 0.9, n_e)
        self.v_grid = np.linspace(0.0, SPEED_MAX, n_v)
        self.u_grid = [(dd, aa) for dd in np.linspace(-STEER_MAX, STEER_MAX, n_u)
                       for aa in (-ACCEL_MAX, 0.0, ACCEL_MAX)]
        self.kernel = self._solve(iters)

    # -- path-relative dynamics -------------------------------------------
    def _f(self, d, e, v, delta, a):
        v2 = np.clip(v + (a - DRAG * v) * self.dt, 0.0, SPEED_MAX)
        psi_dot = v2 / self.wheelbase * np.tan(delta)
        lim = np.where(v2 > 1e-3, A_LAT_MAX * self.assumed_grip / np.maximum(v2, 1e-3),
                       np.inf)
        psi_dot = np.clip(psi_dot, -lim, lim)
        # Progress along the path, and the curvature-induced rotation of the frame.
        s_dot = v2 * np.cos(e) / np.maximum(1.0 - self.curvature * d, 1e-3)
        return (d + v2 * np.sin(e) * self.dt,
                e + (psi_dot - self.curvature * s_dot) * self.dt,
                v2)

    def _solve(self, iters):
        """The fixed-point iteration. Shrinks monotonically, so it converges."""
        D, E, V = np.meshgrid(self.d_grid, self.e_grid, self.v_grid, indexing="ij")
        alive = np.abs(D) <= self.half
        for _ in range(iters):
            nxt = np.zeros_like(alive)
            for delta, a in self.u_grid:
                d2, e2, v2 = self._f(D, E, V, delta, a)
                # A state survives this sweep if *some* input keeps it in the
                # current set -- so this is a union over inputs, intersected
                # with the set below.
                nxt |= self._lookup(alive, d2, e2, v2)
            new = alive & nxt
            if new.sum() == alive.sum():
                break
            alive = new
        return alive

    @staticmethod
    def _idx(grid, q):
        """Index of the nearest grid point. The grids are uniform, so this is
        arithmetic rather than a search -- and it is *nearest*, not the
        insertion point. ``searchsorted`` returns the latter, which shifts
        every lookup by up to a cell and, because the shift is one-sided, makes
        the kernel systematically permissive: the filter then under-intervenes
        and fails to catch a controller it should have caught."""
        lo, hi, n = grid[0], grid[-1], len(grid)
        t = (q - lo) / (hi - lo) * (n - 1)
        return np.clip(np.rint(t).astype(int), 0, n - 1)

    def _lookup(self, table, d, e, v):
        """Membership, with out-of-range treated as unsafe rather than clipped."""
        ok = table[self._idx(self.d_grid, d),
                   self._idx(self.e_grid, e),
                   self._idx(self.v_grid, v)]
        # Clipping an out-of-range query to the edge of the grid would report
        # the edge cell's verdict for a state that is not in the grid at all.
        inside = ((np.abs(d) <= self.half)
                  & (e >= self.e_grid[0]) & (e <= self.e_grid[-1])
                  & (v >= self.v_grid[0]) & (v <= self.v_grid[-1]))
        return ok & inside

    # -- interface ---------------------------------------------------------
    def _relative(self, state5):
        x, y, psi, v = (float(q) for q in state5[:4])
        k = self.track.project(x, y)
        tgt = float(self.track.tangent_angle(k))
        d = self.lateral(x, y)
        e = float(np.arctan2(np.sin(psi - tgt), np.cos(psi - tgt)))
        return d, e, v

    def certify(self, state5, delta: float, a: float) -> bool:
        d, e, v = self._relative(state5)
        d2, e2, v2 = self._f(np.array(d), np.array(e), np.array(v), delta, a)
        return bool(self._lookup(self.kernel, d2, e2, v2))

    def __call__(self, state5, u):
        self.n_steps += 1
        u = np.asarray(u, dtype=float)
        if self.certify(state5, u[0], u[1]):
            return u, False
        best = None
        for delta, a in sorted(self.u_grid,
                               key=lambda p: abs(p[0] - u[0]) + 0.1 * abs(p[1] - u[1])):
            if self.certify(state5, delta, a):
                best = np.array([delta, a, u[2]])
                break
        self.n_interventions += 1
        if best is None:
            self.n_no_safe_action += 1
            return np.array([u[0], -ACCEL_MAX, u[2]]), True
        return best, True
