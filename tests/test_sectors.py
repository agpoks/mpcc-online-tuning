"""Named sectors: straight, long curve, 90-degree, 180-degree.

The motivating fact is in :func:`test_curvature_alone_cannot_separate_90_from_180`
and it is the reason this module exists at all: the pointwise-curvature
segmenter that produced the per-segment result **cannot** express the sector
names, and no retuning of its bin edges will change that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.track import Track  # noqa: E402


@pytest.fixture(scope="module")
def circuit():
    return Track.circuit()


def test_curvature_alone_cannot_separate_90_from_180(circuit):
    """The premise. A corner's curvature says nothing about how far it sweeps.

    ``kappa = 1/R`` for an arc of radius ``R`` however long it is, so the
    circuit's 90-degree and 180-degree corners -- built at the *same* radius on
    purpose -- are indistinguishable to anything reading curvature at a point.
    :meth:`Track.segment` reads curvature at a point.
    """
    c = circuit.corners()
    ninety = [x for x in c if 60 <= abs(np.degrees(x[2])) < 135]
    one_eighty = [x for x in c if abs(np.degrees(x[2])) >= 135]
    assert ninety and one_eighty
    # Same radius by construction, therefore same curvature ...
    k90 = np.mean([x[3] for x in ninety])
    k180 = np.mean([x[3] for x in one_eighty])
    assert abs(k90 - k180) < 0.02 * k180, f"peak kappa {k90:.3f} vs {k180:.3f}"
    # ... so the pointwise segmenter puts them in the same bin ...
    mid90 = (ninety[0][0] + ninety[0][1]) / 2
    mid180 = (one_eighty[0][0] + one_eighty[0][1]) / 2
    assert circuit.segment(mid90) == circuit.segment(mid180)
    # ... and the total-turn classifier does not.
    assert circuit.sector(mid90) != circuit.sector(mid180)


def test_circuit_has_all_four_sector_types(circuit):
    ss = np.linspace(0.0, circuit.length, 600, endpoint=False)
    present = {circuit.sector(v) for v in ss}
    assert present == {0, 1, 2, 3}, f"missing sectors: {set(range(4)) - present}"


def test_circuit_detects_exactly_the_designed_corners(circuit):
    """Six corners, the designed totals, and a lap that turns once."""
    c = circuit.corners()
    assert len(c) == 6, [round(np.degrees(x[2]), 1) for x in c]
    got = sorted(round(np.degrees(x[2])) for x in c)
    want = sorted([-60, 180, 60, -90, 180, 90])
    assert all(abs(a - b) <= 2 for a, b in zip(got, want)), f"{got} vs {want}"
    total = np.degrees(sum(x[2] for x in c))
    assert abs(total - 360.0) < 3.0, f"total turn {total:.1f} deg"


def test_circuit_closes_and_is_drivable(circuit):
    """Closure, and a radius the car can physically take.

    ``WHEELBASE / tan(STEER_MAX)`` is 0.78 m; ``mixed``'s 1.76 m is already
    tight enough that the default weights do not survive it, so the circuit is
    held at the oval's radius to keep a *scheduling* result from being
    confounded by the initialisation failure documented in ``results.md``.
    """
    from mpcc_tuning.model import STEER_MAX, WHEELBASE

    r_min = 1.0 / max(x[3] for x in circuit.corners())
    assert r_min > 2.4, f"tightest corner {r_min:.2f} m"
    assert r_min > WHEELBASE / np.tan(STEER_MAX)
    # Consecutive centreline samples are one ds apart, including across the
    # start/finish line -- which is what "closed" means for this representation.
    gap = np.linalg.norm(circuit.center[0] - circuit.center[-1])
    assert abs(gap - circuit.ds) < 0.25 * circuit.ds, f"seam gap {gap:.4f} m"


def test_sector_label_is_constant_through_a_corner(circuit):
    """One label per corner, not per point.

    A schedule keyed on a label that flickers part-way through a corner would
    switch weight set mid-manoeuvre, which is the failure the wide curvature
    stencil in ``Track.curvature`` was introduced to avoid in the first place.
    """
    for s0, s1, dpsi, _k in circuit.corners():
        span = (s1 - s0) % circuit.length
        inner = [circuit.sector((s0 + 0.15 * span + 0.7 * span * f) % circuit.length)
                 for f in np.linspace(0.0, 1.0, 25)]
        assert len(set(inner)) == 1, (
            f"corner at s={s0:.2f} ({np.degrees(dpsi):.0f} deg) changes label: "
            f"{sorted(set(inner))}")


def test_oval_has_no_ninety_degree_corners():
    """A negative control, and the reason ``circuit`` had to be built.

    The oval is straights and two 180s. If it reported a 90-degree sector the
    classifier would be inventing structure.
    """
    oval = Track.oval()
    ss = np.linspace(0.0, oval.length, 400, endpoint=False)
    assert 2 not in {oval.sector(v) for v in ss}


def test_existing_segment_api_is_untouched():
    """The published per-segment result uses ``segment``; it must not move."""
    oval = Track.oval()
    assert oval.segment(0.0) in (0, 1, 2)
    assert len(oval.segment_edges(3)) == 2
