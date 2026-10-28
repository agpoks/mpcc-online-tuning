"""Presentation GIF helpers: load the shipped track geometry + cached race logs,
render in the project house style (viridis speed, white corridor on grey, tab orange
opponents, open-circle start). Data-only: reads results/race{,_multicar,_phase2} caches,
no acados. Used by tools/present_make.py.
"""
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repo root = parent of this tools/ directory, so mpcc_tuning imports resolve
sys.path.insert(0, _ROOT)
from mpcc_tuning.track import Track  # noqa: E402

_BOUNDARIES = os.path.join(_ROOT, "mpcc_tuning", "tracks", "icra_t2_smooth_boundaries.npz")


def walls(track=None, ds=0.1):
    """Reference + left/right corridor wall polylines, as (ref, left, right).

    Prefers the saved, paper-validated corridor edges (raceline-referenced, matching
    the real map, tools/save_smooth_boundaries.py). Reconstructing them from
    pos + normal*width is error-prone: the corridor is asymmetric (wl != wr), so the
    two widths must land on the correct sides of the normal, which the saved file
    already encodes. Fallback reconstruction below keeps that same convention
    (wr on the +normal side, wl on the -normal side).
    """
    if os.path.exists(_BOUNDARIES):
        z = np.load(_BOUNDARIES)
        c = np.column_stack([z["ref_x"], z["ref_y"]])
        left = np.column_stack([z["left_x"], z["left_y"]])
        right = np.column_stack([z["right_x"], z["right_y"]])
        return c, left, right
    # fallback (no saved edges): reconstruct along the track
    if track is None:
        track = Track.icra_t2_smooth()
    L = float(track.length); ss = np.arange(0, L, ds)
    c = np.array([np.asarray(track.pos(float(s))).ravel() for s in ss])
    ang = np.array([float(track.tangent_angle(float(s))) for s in ss])
    nrm = np.stack([-np.sin(ang), np.cos(ang)], 1)
    w = [track.width(float(s)) for s in ss]
    wl = np.array([float(np.asarray(x[0]).ravel()[0]) for x in w])
    wr = np.array([float(np.asarray(x[1]).ravel()[0]) for x in w])
    left = c + nrm * wr[:, None]; right = c - nrm * wl[:, None]
    return c, left, right
