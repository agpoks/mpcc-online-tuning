"""NEW 2D-track paper figures from the SAVED 10-lap caches (no acados re-drive): the RACELINE, the
ego trajectory and the opponent trajectory as TIME-FADING trails (older = transparent), with overtake
points marked. One figure per opponent class plus a 2x2 comparison. Emits PDF+PNG and, via
--tikz, pgfplots .tex (+ .dat) so it is fully reproducible in both formats.

    python3 tools/race_eval_laps.py --seed 0 --frozen        # caches (band applied)
    python3 tools/paper_race_track2d.py --seed 0 --tikz       # these figures (PDF + TikZ)
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track

OUT = ROOT / "results/race/paper"; TZ = OUT / "tikz"
KINDS = ["static", "slower", "equal", "faster"]
L = Track.icra_t2_smooth().length
VLO, VHI = 1.2, 2.8
EGO_CMAP, OPP_RGB = "viridis", (0.82, 0.45, 0.10)   # opponent = orange


def _b():
    return np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")


def _fading_speed(ax, X, Y, V, lw=1.6, amin=0.06):
    """Ego trail: speed-coloured, alpha fading from amin (start) to 1 (end)."""
    pts = np.array([X, Y]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(VLO, VHI); rgba = cm.get_cmap(EGO_CMAP)(norm(np.asarray(V)[:-1]))
    rgba[:, 3] = np.linspace(amin, 1.0, len(rgba))
    lc = LineCollection(segs, colors=rgba, linewidths=lw, zorder=4); ax.add_collection(lc); return norm


def _fading_solid(ax, X, Y, rgb, lw=1.4, amin=0.05, z=3):
    """Opponent trail: single colour, alpha fading older->transparent."""
    pts = np.array([X, Y]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    rgba = np.tile(np.array([*rgb, 1.0]), (len(segs), 1)); rgba[:, 3] = np.linspace(amin, 0.9, len(rgba))
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=lw, zorder=z))


def _panel(ax, b, z, kind, raceline=True):
    ax.plot(b["left_x"], b["left_y"], color="0.55", lw=0.9, zorder=1); ax.plot(b["right_x"], b["right_y"], color="0.55", lw=0.9, zorder=1)
    if raceline:
        ax.plot(b["ref_x"], b["ref_y"], color="0.35", lw=0.8, ls=(0, (4, 3)), alpha=.7, zorder=2, label="raceline")
    EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
    norm = _fading_speed(ax, EX, EY, V)
    if kind != "static":
        _fading_solid(ax, OX, OY, OPP_RGB)
    pov = np.where(np.diff(np.asarray(z["PASS"], float)) > 0)[0]
    if len(pov):
        ax.scatter(EX[pov], EY[pov], marker="*", s=130, color="crimson", edgecolor="k", lw=.4, zorder=7)
    if bool(z["off"][0]):
        ax.scatter([EX[-1]], [EY[-1]], marker="X", s=90, color="red", zorder=8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return norm, len(pov)


def per_class(seed, tag, raceline):
    b = _b()
    for kind in KINDS:
        f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
        if not f.exists():
            print(f"({kind}: no cache)"); continue
        z = np.load(f)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        norm, npass = _panel(ax, b, z, kind, raceline)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=EGO_CMAP), ax=ax, shrink=.8); cb.set_label("ego speed [m/s]")
        ax.plot([], [], color=OPP_RGB, lw=2, label="opponent (fading)"); ax.scatter([], [], marker="*", color="crimson", label=f"{npass} overtakes")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"vs {kind}: raceline + ego trail (speed) + opponent trail — fading", fontsize=10)
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(OUT / f"fig_track2d_{kind}.{e}", dpi=150)
        plt.close(fig); print(f"wrote {OUT}/fig_track2d_{kind}.pdf (+.png)")


def combined(seed, tag, raceline):
    b = _b(); fig, axs = plt.subplots(2, 2, figsize=(11, 8.6)); norm = None
    for ax, kind in zip(axs.ravel(), KINDS):
        f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
        if not f.exists():
            ax.set_axis_off(); continue
        z = np.load(f); norm, npass = _panel(ax, b, z, kind, raceline)
        ax.set_title(f"{kind}  ({npass} overtakes{', OFF' if bool(z['off'][0]) else ''})", fontsize=10)
    if norm is not None:
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=EGO_CMAP), ax=axs, shrink=.6, location="right", pad=.02); cb.set_label("ego speed [m/s]")
    fig.suptitle(f"Raceline + ego trail (speed) + opponent trail (fading) + overtakes (★) — band {tag.strip('_')}", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_track2d_all.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_track2d_all.pdf (+.png)")


def emit_tikz(seed, tag):
    """pgfplots .tex per class: raceline + edges + ego path (speed-coloured) + opponent + overtakes.
    Trajectories go to .dat; the ego path is mesh/scatter-coloured by speed (pgfplots can't alpha-fade
    per point, so TikZ shows the speed-coloured path -- same data, vector output)."""
    b = _b(); TZ.mkdir(parents=True, exist_ok=True)
    def dat(name, cols, arrs):
        with open(TZ / name, "w") as fh:
            fh.write(" ".join(cols) + "\n")
            for row in zip(*arrs):
                fh.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    # static track geometry (shared)
    dat("track2d_left.dat", ["x", "y"], [b["left_x"], b["left_y"]])
    dat("track2d_right.dat", ["x", "y"], [b["right_x"], b["right_y"]])
    dat("track2d_ref.dat", ["x", "y"], [b["ref_x"], b["ref_y"]])
    for kind in KINDS:
        f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
        if not f.exists():
            continue
        z = np.load(f); EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
        ds = max(1, len(EX) // 1500)                       # downsample for a light .tex
        dat(f"track2d_ego_{kind}.dat", ["x", "y", "v"], [EX[::ds], EY[::ds], V[::ds]])
        if kind != "static":
            dat(f"track2d_opp_{kind}.dat", ["x", "y"], [OX[::ds], OY[::ds]])
        pov = np.where(np.diff(np.asarray(z["PASS"], float)) > 0)[0]
        ov = " ".join(f"({EX[i]:.3f},{EY[i]:.3f})" for i in pov)
        opp_line = (rf"\addplot[orange!80!black, thick, opacity=0.6] table[x=x,y=y] "
                    rf"{{tikz/track2d_opp_{kind}.dat}};" if kind != "static" else "")
        ovl = (rf"\addplot[only marks, mark=star, mark size=3pt, red] coordinates {{{ov}}};" if ov else "")
        body = rf"""% Auto-generated by tools/paper_race_track2d.py --tikz  (vs {kind}, band {tag.strip('_')})
\begin{{tikzpicture}}
\begin{{axis}}[width=8cm, axis equal image, hide axis, enlargelimits=0.02,
    colormap/viridis, point meta min={VLO}, point meta max={VHI},
    colorbar, colorbar style={{ylabel=ego speed [m/s]}}]
  \addplot[gray, thin] table[x=x,y=y] {{tikz/track2d_left.dat}};
  \addplot[gray, thin] table[x=x,y=y] {{tikz/track2d_right.dat}};
  \addplot[black!55, dashed, thin] table[x=x,y=y] {{tikz/track2d_ref.dat}};  % raceline
  \addplot[mesh, point meta=explicit, line width=1.1pt] table[x=x,y=y,meta=v] {{tikz/track2d_ego_{kind}.dat}};  % ego, speed
  {opp_line}
  {ovl}
\end{{axis}}
\end{{tikzpicture}}
"""
        (TZ / f"fig_track2d_{kind}.tex").write_text(body)
        print(f"wrote {TZ}/fig_track2d_{kind}.tex (+ .dat)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true"); ap.add_argument("--tikz", action="store_true")
    ap.add_argument("--no-raceline", action="store_true"); a = ap.parse_args()
    tag = "_10lap" if a.online else "_10lapfrozen"; rl = not a.no_raceline
    per_class(a.seed, tag, rl); combined(a.seed, tag, rl)
    if a.tikz:
        emit_tikz(a.seed, tag)
