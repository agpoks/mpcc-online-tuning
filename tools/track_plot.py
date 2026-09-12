"""Shared background for T2 figures: the REAL occupancy map + the smooth
left/right boundaries the MPCC enforces. Import and call draw_track(ax, t).
"""
from pathlib import Path
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_TR = _ROOT / "mpcc_tuning" / "tracks"

def track_edges(t, n=1600):
    """Left/right boundary world coords from the track's own spline + width.
    Correct convention (verified vs the map): +normal wall at dist wr, -normal at wl."""
    s = np.linspace(0, t.length, n, endpoint=False)
    P=[];L=[];R=[]
    for si in s:
        p=np.array(t.pos(float(si))).ravel(); a=float(t.tangent_angle(float(si)))
        nx,ny=-np.sin(a),np.cos(a); wl,wr=t.width(float(si))
        P.append(p); L.append([p[0]+nx*float(wr),p[1]+ny*float(wr)]); R.append([p[0]-nx*float(wl),p[1]-ny*float(wl)])
    return np.array(P),np.array(L),np.array(R)

def draw_track(ax, t, show_map=True, show_ref=True, edge_lw=1.6, ref=None):
    """Draw the real map + smooth left/right boundaries (+ optional reference)."""
    if show_map:
        im=np.array(Image.open(_TR/"icra2026_t2.pgm")); H,W=im.shape
        res,ox,oy=0.05,-2.8,-7.25   # icra2026_t2.yaml
        ax.imshow(im,cmap="gray",extent=[ox,ox+W*res,oy,oy+H*res],origin="upper",zorder=0,alpha=0.9)
    P,L,R=track_edges(t)
    ax.plot(L[:,0],L[:,1],color="tab:blue",lw=edge_lw,zorder=2,label="left boundary")
    ax.plot(R[:,0],R[:,1],color="tab:cyan",lw=edge_lw,zorder=2,label="right boundary")
    if show_ref:
        rl=ref if ref is not None else getattr(t,"raceline",None)
        if rl is not None: ax.plot(rl[:,0],rl[:,1],':',color="0.35",lw=1.0,zorder=1,label="reference")
    ax.set_aspect("equal"); ax.axis("off")
    return P,L,R
