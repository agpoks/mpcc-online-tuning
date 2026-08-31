"""A row of cones must bound the corridor, not sit inside it.

The extractor fills holes in the occupancy grid, which it needs to do -- an
unfilled cone makes the skeleton loop around each one. But a *row* of cones is
a boundary whose pixels are mostly the free space between the cones, so filling
those holes turned a wall into driveable track and the extracted centreline was
free to cut straight through a cone line.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PGM = ROOT / "mpcc_tuning" / "tracks" / "icra2026_t1.pgm"
YML = ROOT / "mpcc_tuning" / "tracks" / "icra2026_t1.yaml"


def _cl():
    spec = importlib.util.spec_from_file_location(
        "cl", str(ROOT / "tools" / "centerline_from_map.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cl():
    return _cl()


def test_welding_only_ever_adds_wall(cl):
    """The weld may close gaps; it must never open one."""
    im, _res, _org = cl.load(str(PGM), str(YML))
    occ = im <= 50
    welded = cl.connect_cone_rows(occ)
    assert np.all(welded | ~occ), "welding removed occupied pixels"
    assert welded.sum() > occ.sum(), "welding closed nothing at all"


def test_a_cone_row_becomes_continuous_wall(cl):
    """Neighbouring cones join; the map is not simply dilated everywhere."""
    import scipy.ndimage as ndi
    im, _res, _org = cl.load(str(PGM), str(YML))
    occ = im <= 50
    welded = cl.connect_cone_rows(occ)
    n_before = ndi.label(occ)[1]
    n_after = ndi.label(welded)[1]
    assert n_after < n_before, "no blobs merged, so no row was welded"


def test_extracted_centreline_does_not_cross_a_cone_row(cl):
    """The point of the exercise: the car cannot drive through the wall.

    Measured against the car's half-width -- clearing the wall by less than
    that is still a collision, however clean the line looks.
    """
    import scipy.ndimage as ndi
    im, res, org = cl.load(str(PGM), str(YML))
    H, W = im.shape
    xy, _w, _L, _ok, _gap, _a = cl.centerline_around(
        str(PGM), str(YML), hole_rank=0)
    weld = cl.connect_cone_rows(im <= 50) & ~(im <= 50)
    d = ndi.distance_transform_edt(~weld) * res
    r = np.clip(((org[1] + H * res - xy[:, 1]) / res).astype(int), 0, H - 1)
    c = np.clip(((xy[:, 0] - org[0]) / res).astype(int), 0, W - 1)
    assert weld[r, c].sum() == 0, "centreline passes through a welded cone row"
    assert d[r, c].min() > 0.12, "centreline clears the wall by less than a car"


def test_welding_does_not_break_the_lap(cl):
    """Closing gaps must not sever the corridor into an open path."""
    xy, w, L, ok, gap, _a = cl.centerline_around(str(PGM), str(YML), hole_rank=0)
    assert 60.0 < L < 80.0, f"lap length {L:.1f} m is not the known ~69 m"
    assert ok > 0.99, f"only {100*ok:.0f}% of the centreline is in the corridor"
    assert gap < 0.15, f"loop fails to close ({gap:.2f} m)"
    assert np.median(w) > 0.3
