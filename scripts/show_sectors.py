"""Print and draw the automatically detected sectors of any loaded track.

    python3 scripts/show_sectors.py --track icra_t2_raceline
    python3 scripts/show_sectors.py --track oval --png

Nothing is hand-set per track: corners are detected at a fraction of the
track's own peak curvature and classified by total turn (straight, long
curve, 90-deg, 180-deg); sub-metre straights are absorbed. This is what the
weight policy is told, as a soft membership ramping over about two metres.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track

SEC_COL = ("#4C6EF5", "#0CA678", "#E8590C", "#AE3EC9")

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--png", action="store_true", help="also write paper/figures/sectors_<track>.png")
    a = ap.parse_args(argv)
    t = getattr(Track, a.track)()
    tab = t.sectors()
    print(f"  {a.track}: {t.length:.1f} m lap, {len(tab)} sectors (detected automatically)")
    print("  %8s %8s %7s  %s" % ("s_start", "s_end", "len m", "sector"))
    for s0, s1, L, k, nm in tab:
        print("  %8.2f %8.2f %7.2f  %s" % (s0, s1, L, nm))
    lens = np.array([r[2] for r in tab])
    print("  run length: min %.2f  median %.2f  max %.2f m" % (lens.min(), np.median(lens), lens.max()))
    if a.png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        c = t.center; sec = np.array([t.sector(float(v)) for v in t.s])
        fig, ax = plt.subplots(figsize=(7, 6.5)); fig.patch.set_facecolor("white")
        g = np.gradient(c, axis=0); g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        n = np.column_stack([-g[:, 1], g[:, 0]])
        wl = np.array([float(t.width(v)[0]) for v in t.s]); wr = np.array([float(t.width(v)[1]) for v in t.s])
        for e in (c + n * wr[:, None], c - n * wl[:, None]):
            ax.plot(e[:, 0], e[:, 1], "-", color="#212529", lw=1.0)
        pts = c.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=[SEC_COL[k] for k in sec[:-1]], linewidth=7, alpha=0.55))
        for k, nm in enumerate(t.SECTOR_NAMES):
            ax.plot([], [], "-", lw=6, alpha=0.6, color=SEC_COL[k], label=nm)
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.set_title(f"{a.track}: {len(tab)} sectors, detected automatically", loc="left", fontsize=11, fontweight="bold")
        out = ROOT / "paper" / "figures" / f"sectors_{a.track}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white"); print("  wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
