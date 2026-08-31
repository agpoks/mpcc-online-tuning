import numpy as np, re
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image

def load(pgm, yaml_path=None, res=0.05, origin=(0.0, 0.0)):
    if yaml_path:
        m = dict(re.findall(r"^(\w+):\s*(.+)$", open(yaml_path).read(), re.M))
        res = float(m["resolution"])
        origin = tuple(float(v) for v in m["origin"].strip("[] ").split(",")[:2])
    return np.array(Image.open(pgm)), res, origin

def connect_cone_rows(occ, bridge_px=21, min_cone_px=8, max_cone_px=200):
    """Join a row of cones into one continuous wall.

    A line of cones IS a track boundary. Filling them -- which is what a naive
    hole-fill does, and what this tool did -- turns that boundary into free
    space, so the extracted corridor lets the car drive straight through a wall
    and the centreline can be routed on the wrong side of it. That is a
    correctness bug in the track, not a cosmetic one in the figure.

    Cones on these maps are 8-200 px blobs spaced a median of 15.7 px apart
    (0.78 m at 0.05 m/px), 19.4 px at the 90th percentile. A morphological
    closing wide enough to bridge that spacing links each row into a barrier,
    while leaving isolated obstacles and the outer walls as they are.
    """
    lab, k = ndi.label(occ)
    if k == 0:
        return occ
    sizes = np.array(ndi.sum(occ, lab, range(1, k + 1)))
    cone = np.isin(lab, [i + 1 for i, sz in enumerate(sizes)
                         if min_cone_px <= sz <= max_cone_px])
    # Close only the cone-sized blobs, so bridging a cone row cannot also weld
    # two genuine walls together across a gap the car is meant to drive through.
    b = int(bridge_px) | 1
    joined = ndi.binary_closing(cone, np.ones((b, b)))
    # Keep only what the closing added *between* cones: a closed blob that
    # touches no cone is an artefact of the kernel, not a wall.
    joined &= ndi.binary_dilation(cone, np.ones((b, b)))
    return occ | joined


def corridor(im, close_px=2, min_hole_px=300, connect_cones=True,
             bridge_px=21):
    if connect_cones:
        occ = connect_cone_rows(im <= 50, bridge_px=bridge_px)
        im = np.where(occ, 0, im)
    free = im >= 250
    free = ndi.binary_closing(free, np.ones((close_px*2+1,)*2))
    free = ndi.binary_opening(free, np.ones((3, 3)))
    lab, k = ndi.label(free)
    free = lab == (1 + int(np.argmax(ndi.sum(free, lab, range(1, k+1)))))
    filled = ndi.binary_fill_holes(free)
    hl, hk = ndi.label(filled & ~free)
    keep = np.zeros_like(free)
    for i in range(1, hk+1):
        m = hl == i
        if m.sum() >= min_hole_px:
            keep |= m
    return (filled & ~keep), free

def cycle(sk):
    """Longest simple cycle: prune spurs, then walk with degree bookkeeping."""
    S = sk.copy()
    def deg(S):
        return sum(np.roll(np.roll(S, dy, 0), dx, 1).astype(np.uint8)
                   for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx) != (0,0)) * S
    for _ in range(5000):
        d = deg(S)
        ends = S & (d == 1)
        if not ends.any():
            break
        S &= ~ends
    lab, k = ndi.label(S, structure=np.ones((3,3)))
    if k == 0:
        return None
    S = lab == (1 + int(np.argmax(ndi.sum(S, lab, range(1, k+1)))))
    pts = np.argwhere(S)
    idx = {tuple(p): i for i, p in enumerate(pts)}
    # walk preferring the neighbour that keeps us going (largest turn continuity)
    cur, prev, order, seen = 0, None, [0], {0}
    while True:
        y, x = pts[cur]
        cands = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == dx == 0: continue
                j = idx.get((y+dy, x+dx))
                if j is not None and j not in seen:
                    cands.append(j)
        if not cands: break
        if prev is None:
            nxt = cands[0]
        else:
            v0 = pts[cur] - pts[prev]
            nxt = max(cands, key=lambda j: float(np.dot(pts[j]-pts[cur], v0)))
        order.append(nxt); seen.add(nxt); prev, cur = cur, nxt
    return pts[order]

def centerline(pgm, yaml_path=None, res=0.05, origin=(0,0), min_width_px=5,
               n_out=800, smooth_m=0.35, connect_cones=True, bridge_px=21):
    im, res, origin = load(pgm, yaml_path, res, origin)
    H, W = im.shape
    corr, free = corridor(im, connect_cones=connect_cones, bridge_px=bridge_px)
    dist = ndi.distance_transform_edt(corr)
    core = corr & (dist >= min_width_px/2)
    lab, k = ndi.label(core)
    core = lab == (1 + int(np.argmax(ndi.sum(core, lab, range(1, k+1)))))
    sk = skeletonize(core)
    pts = cycle(sk)
    if pts is None: raise RuntimeError("no cycle")
    x = origin[0] + (pts[:,1]+0.5)*res
    y = origin[1] + (H - pts[:,0]-0.5)*res
    xy = np.stack([x,y],1)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    t = np.linspace(0, d[-1], n_out, endpoint=False)
    xy = np.stack([np.interp(t,d,xy[:,0]), np.interp(t,d,xy[:,1])],1)
    # Smooth on the closed curve before anything reads curvature from it. A
    # skeleton is pixel-quantised, and Track.curvature's stencil reads those
    # steps as real corners: unsmoothed, this map reports a 0.58 m minimum
    # radius against a car that cannot turn inside 0.78 m, and scaling the
    # track 2x moves it only to 0.76 m -- sub-linear, which is the signature of
    # noise rather than geometry. The window is a few centimetres, far shorter
    # than any real corner here.
    if smooth_m > 0:
        w_ = max(int(smooth_m / max(d[-1] / n_out, 1e-9)) | 1, 3)
        k_ = np.ones(w_) / w_
        pad = np.r_[xy[-w_:], xy, xy[:w_]]
        xy = np.stack([np.convolve(pad[:, i], k_, "same")[w_:-w_] for i in (0, 1)], 1)
    hw = ndi.distance_transform_edt(free)*res
    rr = np.clip(((origin[1]+H*res - xy[:,1])/res).astype(int),0,H-1)
    cc = np.clip(((xy[:,0]-origin[0])/res).astype(int),0,W-1)
    return xy, hw[rr,cc], float(d[-1]), float(free[rr,cc].mean()), np.linalg.norm(xy[0]-xy[-1])


def centerline_around(pgm, yaml_path=None, res=0.05, origin=(0, 0), hole_rank=0,
                      close_px=2, n_out=800, smooth_m=0.35, connect_cones=True,
                      bridge_px=21):
    """Centreline of ONE loop, chosen by which hole it goes around.

    A branching corridor -- an outer ring plus an inner section, as on the ICRA
    2026 maps -- has no unique centreline, and every attempt to find "the"
    centreline of one fails for exactly that reason. But it has a well-defined
    centreline *per loop*, and a loop is named by the hole it encircles.

    So pick the hole and the loop follows. ``hole_rank=0`` is the largest hole
    (the outer lap), ``1`` the next one in, and so on. The locus equidistant
    from that hole and from everything else is a closed curve around it, and a
    contour tracer returns it already ordered -- no graph to walk, so no
    junction to terminate at.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    im, res, origin = load(pgm, yaml_path, res, origin)
    H, W = im.shape
    corr, free = corridor(im, close_px=close_px, connect_cones=connect_cones,
                          bridge_px=bridge_px)
    filled = ndi.binary_fill_holes(free)
    hl, hk = ndi.label(filled & ~free)
    if hk == 0:
        raise RuntimeError("no hole: the corridor is not a closed loop")
    areas = np.array([(hl == i).sum() for i in range(1, hk + 1)])
    order = np.argsort(areas)[::-1]
    if hole_rank >= len(order):
        raise IndexError("only %d holes; asked for rank %d" % (len(order), hole_rank))
    target = hl == (order[hole_rank] + 1)

    # Everything the loop must stay clear of: the outer wall and every OTHER
    # hole. Excluding the target is what makes the contour encircle it.
    others = (~filled) | ((filled & ~free) & ~target)
    phi = ndi.distance_transform_edt(~target) - ndi.distance_transform_edt(~others)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(phi, levels=[0.0])
    paths = [q.vertices for q in (cs.get_paths() if hasattr(cs, "get_paths")
                                  else cs.collections[0].get_paths())]
    plt.close(fig)
    best, bl = None, -1.0
    for v in paths:
        rr = np.clip(v[:, 1].astype(int), 0, H - 1)
        cc = np.clip(v[:, 0].astype(int), 0, W - 1)
        if free[rr, cc].mean() < 0.97:          # must lie in the drivable corridor
            continue
        L = float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())
        if L > bl:
            best, bl = v, L
    if best is None:
        raise RuntimeError("no contour for hole rank %d lies in the corridor" % hole_rank)

    x = origin[0] + (best[:, 0] + 0.5) * res
    y = origin[1] + (H - best[:, 1] - 0.5) * res
    xy = np.stack([x, y], 1)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    t = np.linspace(0.0, d[-1], n_out, endpoint=False)
    xy = np.stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])], 1)
    if smooth_m > 0:
        w_ = max(int(smooth_m / max(d[-1] / n_out, 1e-9)) | 1, 3)
        k_ = np.ones(w_) / w_
        pad = np.r_[xy[-w_:], xy, xy[:w_]]
        xy = np.stack([np.convolve(pad[:, i], k_, "same")[w_:-w_] for i in (0, 1)], 1)
    hw = ndi.distance_transform_edt(free) * res
    rr = np.clip(((origin[1] + H * res - xy[:, 1]) / res).astype(int), 0, H - 1)
    cc = np.clip(((xy[:, 0] - origin[0]) / res).astype(int), 0, W - 1)
    return (xy, hw[rr, cc], float(d[-1]), float(free[rr, cc].mean()),
            float(np.linalg.norm(xy[0] - xy[-1])), int(areas[order[hole_rank]]))
