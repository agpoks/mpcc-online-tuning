"""Centreline of a closed track corridor, from a ROS occupancy grid.

**STATUS: NOT FINISHED. Do not use the output as a track.** It reproduces
roughly 80% of a lap correctly and cuts across the infield on the rest; the
diagnosis and what is needed are at the bottom of this docstring.

Centreline of a closed track corridor, as an equidistance contour.

Skeletonization is the wrong tool for this. Every cone punches a hole in the
free space, so a skeleton loops around each one, and with several large holes
the pruned skeleton is a multi-cycle graph that no greedy walk can order.

A racing corridor is an annulus: the centreline is the locus equidistant from
the *infield* and the *outer wall*. That is the zero level set of
``d_infield - d_outer``, which is a closed curve, comes out already ordered
from a contour tracer, and is indifferent to cones -- they perturb neither
distance field enough to matter, and they never create a spurious branch.
"""
import numpy as np
from scipy import ndimage as ndi


def centerline(pgm, yaml_path, close_px=2, min_hole_px=400, n_out=600):
    import re
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = dict(re.findall(r"^(\w+):\s*(.+)$", open(yaml_path).read(), re.M))
    res = float(meta["resolution"])
    ox, oy, _ = [float(v) for v in meta["origin"].strip("[] ").split(",")]
    im = np.array(Image.open(pgm))
    H = im.shape[0]

    free = im >= 250
    free = ndi.binary_closing(free, np.ones((close_px * 2 + 1,) * 2))
    free = ndi.binary_opening(free, np.ones((3, 3)))
    lab, k = ndi.label(free)
    free = lab == (1 + int(np.argmax(ndi.sum(free, lab, range(1, k + 1)))))

    filled = ndi.binary_fill_holes(free)
    holes = filled & ~free
    hl, hk = ndi.label(holes)
    areas = np.array([(hl == i).sum() for i in range(1, hk + 1)])
    if not len(areas):
        raise RuntimeError("corridor has no infield: not a closed loop")
    infield = hl == (1 + int(np.argmax(areas)))          # the lap's island

    # Outer wall: everything outside the corridor. Cones and the secondary
    # holes are deliberately NOT included -- the centreline should pass them.
    outer = ~filled

    d_in = ndi.distance_transform_edt(~infield)
    d_out = ndi.distance_transform_edt(~outer)
    phi = d_in - d_out

    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(phi, levels=[0.0])
    segs = [q.vertices for q in (cs.get_paths() if hasattr(cs, "get_paths")
                                 else cs.collections[0].get_paths())]
    plt.close(fig)
    # The zero level set of d_in - d_out exists everywhere, including inside the
    # infield and outside the outer wall, and those branches are longer than the
    # lap. Keep only pieces that lie in the corridor, then take the longest.
    corridor = filled & ~infield
    best, bl = None, -1.0
    for v in segs:
        rr = np.clip(v[:, 1].astype(int), 0, H - 1)
        cc = np.clip(v[:, 0].astype(int), 0, im.shape[1] - 1)
        if corridor[rr, cc].mean() < 0.97:
            continue
        L = float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())
        if L > bl:
            best, bl = v, L
    if best is None:
        raise RuntimeError("no contour lies inside the corridor")
    col, row = best[:, 0], best[:, 1]
    inside = free[np.clip(row.astype(int), 0, H - 1),
                  np.clip(col.astype(int), 0, im.shape[1] - 1)]
    keep = inside.mean()
    x = ox + (col + 0.5) * res
    y = oy + (H - row - 0.5) * res
    xy = np.stack([x, y], 1)

    # Resample to uniform arc length.
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n_out, endpoint=False)
    xy = np.stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])], 1)
    half = ndi.distance_transform_edt(free)
    rr = np.clip(((oy + (H - 0) * res - xy[:, 1]) / res).astype(int), 0, H - 1)
    cc = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, im.shape[1] - 1)
    w = half[rr, cc] * res
    return xy, w, float(d[-1]), keep
