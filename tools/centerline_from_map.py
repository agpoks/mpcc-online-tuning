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

def corridor(im, close_px=2, min_hole_px=300):
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
               n_out=800, smooth_m=0.35):
    im, res, origin = load(pgm, yaml_path, res, origin)
    H, W = im.shape
    corr, free = corridor(im)
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
