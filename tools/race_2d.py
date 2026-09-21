"""Static 2D PAPER figure of a race-mode episode: ego & opponent paths + snapshots.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/race_2d.py --arm ltc --kind slower

Unlike the GIF, this is a single publication figure: the ego path (coloured by speed)
and the opponent path (red dashed) over the whole drive on the real map + corridor, with
both cars drawn at several snapshot times (thin link = their separation at that instant),
the completed-pass points ringed, and the minimum body-to-body gap annotated. Reuses the
same drive() as the GIF so the two always agree. --arm ltc/mlp replays the trained net.
"""
import sys, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from tools.race_gif import drive
from mpcc_tuning.track import Track


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ltc"); ap.add_argument("--kind", default="slower")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ego-pace", type=float, default=1.35)
    ap.add_argument("--snapshots", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or str(ROOT / "results/race/paper" / f"{a.arm}_{a.kind}_2d.pdf")
    d = drive(a.arm, a.kind, a.seed, a.steps, a.ego_pace)
    EX, EY, EV = d["EX"], d["EY"], d["EV"]; OX, OY = d["OX"], d["OY"]
    passes = np.array(d["PASS"]); track = d["track"]; TR = ROOT / "mpcc_tuning/tracks"
    body_gap = np.hypot(EX - OX, EY - OY) - 0.24        # 0.24 = sum of half-widths

    im = np.array(Image.open(TR / "icra2026_t2.pgm")); H, W = im.shape; res, ox, oy = 0.05, -2.8, -7.25
    d0 = np.load(TR / "icra_t2_raceline_ref_corridor.npz")
    tt = Track(d0["cx"], d0["cy"], ds=0.1, w_left=d0["wl"], w_right=d0["wr"])
    ss = np.linspace(0, tt.length, 1200, endpoint=False); L = []; Rr = []
    for si in ss:
        p = np.asarray(tt.pos(float(si))).ravel(); ang = float(tt.tangent_angle(float(si)))
        nx, ny = -np.sin(ang), np.cos(ang); wl, wr = tt.width(float(si))
        L.append([p[0] + nx * wr, p[1] + ny * wr]); Rr.append([p[0] - nx * wl, p[1] - ny * wl])
    L = np.array(L); Rr = np.array(Rr)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.imshow(im, cmap="gray", extent=[ox, ox + W * res, oy, oy + H * res], origin="upper", zorder=0)
    ax.plot(L[:, 0], L[:, 1], color="0.5", lw=0.8, zorder=1); ax.plot(Rr[:, 0], Rr[:, 1], color="0.5", lw=0.8, zorder=1)
    ax.plot(OX, OY, color="tab:red", lw=1.6, ls="--", zorder=2, label="opponent path")
    pts = np.column_stack([EX, EY]); seg = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    lc = LineCollection(seg, cmap="viridis", lw=2.6, zorder=3); lc.set_array(EV[:-1]); ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02); cb.set_label("ego speed [m/s]")
    # snapshots of both cars + their separation link
    for j in np.linspace(0, len(EX) - 1, a.snapshots).astype(int):
        ax.plot([EX[j], OX[j]], [EY[j], OY[j]], color="k", lw=0.7, alpha=0.5, zorder=4)
        ax.scatter([EX[j]], [EY[j]], c="white", edgecolor="k", s=70, zorder=5)
        ax.scatter([OX[j]], [OY[j]], c="tab:red", edgecolor="k", marker="s", s=70, zorder=5)
    # completed-pass points ringed
    for pi in np.where(np.diff(passes) > 0)[0]:
        ax.scatter([EX[pi]], [EY[pi]], facecolor="none", edgecolor="lime", s=280, lw=2.4, zorder=6,
                   label="overtake" if pi == np.where(np.diff(passes) > 0)[0][0] else None)
    ax.scatter([], [], c="white", edgecolor="k", s=70, label="ego")
    ax.scatter([], [], c="tab:red", edgecolor="k", marker="s", s=70, label="opponent")
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title(f"{a.arm} vs {a.kind}: ego (speed-coloured) & opponent (red) paths — "
                 f"{int(passes[-1])} pass(es), min body gap {body_gap.min():.2f} m")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(Path(out).with_suffix("." + ext)), dpi=140)
    print(f"saved {out}  (passes={int(passes[-1])}, min body gap {body_gap.min():.2f} m)")


if __name__ == "__main__":
    main()
