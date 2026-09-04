"""What the controller is allowed to use, against what the track offers.

The corridor was a SCALAR `track.half_width - car_half_width` -- one number
for the whole lap. On ICRA T2 that is 0.403 m while the real half-width runs
0.40-2.54 m, so the car drove a constant tunnel down a course that opens to
5.07 m, and the optimiser's own racing line (p95 0.758 m, max 0.969 m off the
reference) was geometrically OUTSIDE it. No amount of learning reaches a line
the constraints forbid.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track  # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
C_OLD, C_NEW, C_RL = "#E8590C", "#0CA678", "#4C6EF5"
CHW, SAFE = 0.12, 0.08


def main(track_name="icra_t2_raceline"):
    t = getattr(Track, track_name)()
    c = t.center
    g = np.gradient(c, axis=0)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-g[:, 1], g[:, 0]])
    wl = np.array([float(t.width(v)[0]) for v in t.s])
    wr = np.array([float(t.width(v)[1]) for v in t.s])
    old = float(t.half_width) - CHW
    new_l = np.maximum(wl - CHW - SAFE, 0.0)
    new_r = np.maximum(wr - CHW - SAFE, 0.0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("white")

    # -- left: the track, both corridors, the racing line -------------------
    for e in (c + n * wl[:, None], c - n * wr[:, None]):
        ax.plot(e[:, 0], e[:, 1], "-", color=INK, lw=1.4, zorder=4)
    xs = np.concatenate([c + n * new_l[:, None], (c - n * new_r[:, None])[::-1]])
    ax.fill(xs[:, 0], xs[:, 1], color=C_NEW, alpha=0.22, lw=0, zorder=2)
    for e in (c + n * old, c - n * old):
        ax.plot(e[:, 0], e[:, 1], "-", color=C_OLD, lw=1.5, zorder=3)
    rl = np.asarray(getattr(t, "raceline"))
    ax.plot(rl[:, 0], rl[:, 1], "-", color=C_RL, lw=1.7, zorder=5)
    ax.plot([], [], "-", color=INK, lw=1.4, label="track boundary")
    ax.fill([], [], color=C_NEW, alpha=0.22, label="corridor now: width(s)")
    ax.plot([], [], "-", color=C_OLD, lw=1.5, label=f"corridor before: {old:.2f} m constant")
    ax.plot([], [], "-", color=C_RL, lw=1.7, label="optimiser's racing line")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    ax.set_title(f"{track_name} — the corridor the solver is given",
                 fontsize=12, fontweight="bold", loc="left", color=INK)

    # -- right: the same thing as numbers, along the lap --------------------
    ax2.fill_between(t.s, 0, wl, color=INK, alpha=0.10, lw=0,
                     label="track half-width")
    ax2.plot(t.s, new_l, "-", color=C_NEW, lw=1.8, label="corridor now")
    ax2.axhline(old, color=C_OLD, lw=1.6, label=f"corridor before ({old:.2f} m)")
    off = np.abs([t.lateral(float(p[0]), float(p[1])) for p in rl])
    s_rl = np.array([t.project(float(p[0]), float(p[1])) for p in rl])
    o = np.argsort(s_rl)
    ax2.plot(s_rl[o], off[o], "-", color=C_RL, lw=1.2, alpha=0.85,
             label="racing line needs")
    ax2.set_xlabel("arc length s (m)", fontsize=9.5, color=MUT)
    ax2.set_ylabel("lateral room (m)", fontsize=9.5, color=MUT)
    ax2.set_xlim(0, t.length); ax2.set_ylim(0, min(2.8, wl.max() * 1.05))
    ax2.grid(True, color=GRID, lw=0.7); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(colors=MUT, labelsize=8.5)
    ax2.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    frac = np.mean(off > old) * 100
    ax2.set_title("The blue line above the orange line is racing the old "
                  f"corridor forbade\n({frac:.0f}% of the lap)",
                  fontsize=11, fontweight="bold", loc="left", color=INK)

    name = "corridor_width.png" if track_name == "icra_t2_raceline" else f"corridor_width_{track_name}.png"
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    print("  wrote", OUT / name)
    print("  racing line outside the OLD corridor on %.0f%% of the lap" % frac)
    print("  old corridor %.3f m constant; new %.3f-%.3f m" %
          (old, new_l.min(), new_l.max()))


if __name__ == "__main__":
    main(*sys.argv[1:])
