"""NEW paper figures, all from the SAVED 10-lap state caches (no acados re-drive):
  fig_track_opponents.{pdf,png}        -- full track, ego trajectory (speed-coloured) + opponent path
                                          + overtake points, one panel per opponent class (comparison).
  fig_overtake_zoom_<kind>.{pdf,png}   -- ZOOM on an overtake: a small 2D patch of the track with the
                                          ego and opponent drawn as cars at before/at/after the pass,
                                          their paths, PLUS the ego/opponent speeds and the gap across
                                          the manoeuvre (the "special behaviour").

Reads states{_10lapfrozen|_10lap}_ltc_<seed>_<kind>.npz written by tools/race_eval_laps.py, so it is
fully regeneratable without re-driving:
    python3 tools/race_eval_laps.py --seed 0 --frozen       # caches (band applied)
    python3 tools/paper_race_overtake_zoom.py --seed 0       # these figures
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track

OUT = ROOT / "results/race/paper"
KINDS = ["static", "slower", "equal", "faster"]
L = Track.icra_t2_smooth().length
CARL, CARW = 0.30, 0.16          # car glyph (m)
EGO_C, OPP_C = "#1f4e8c", "#d08020"


def _b():
    return np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")


def _heading(X, Y):
    return np.arctan2(np.gradient(np.asarray(Y, float)), np.gradient(np.asarray(X, float)))


def _car(ax, x, y, psi, color, z=6):
    t = Affine2D().rotate(float(psi)).translate(float(x), float(y)) + ax.transData
    ax.add_patch(Rectangle((-CARL / 2, -CARW / 2), CARL, CARW, transform=t,
                           facecolor=color, edgecolor="k", lw=.6, alpha=.92, zorder=z))


def _overtakes(PASS):
    return np.where(np.diff(np.asarray(PASS, float)) > 0)[0]


def fig_track_opponents(seed, tag):
    b = _b()
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6)); sc = None
    for ax, kind in zip(axs.ravel(), KINDS):
        f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
        if not f.exists():
            ax.text(.5, .5, f"(no {kind} cache)", ha="center", va="center", color="0.5"); ax.set_axis_off(); continue
        z = np.load(f); EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
        ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
        sc = ax.scatter(EX, EY, c=V, cmap="viridis", s=3, zorder=4, vmin=1.2, vmax=2.8)
        if kind != "static":
            ax.plot(OX, OY, color="0.35", lw=.6, alpha=.45, zorder=3)
        pov = _overtakes(z["PASS"])
        if len(pov):
            ax.scatter(EX[pov], EY[pov], marker="*", s=130, color="crimson", edgecolor="k", lw=.4, zorder=7)
        off = bool(z["off"][0])
        if off:
            ax.scatter([EX[-1]], [EY[-1]], marker="X", s=90, color="red", zorder=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{kind}  —  v_mean {V.mean():.2f}, {len(pov)} overtakes{' (OFF)' if off else ''}", fontsize=10)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, shrink=.6, location="right", pad=.02); cb.set_label("ego speed [m/s]")
    fig.suptitle(f"Track + ego trajectory (speed) + opponent path + overtakes (★), per class — band {tag.strip('_')}", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_track_opponents.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_track_opponents.pdf (+.png)")


def fig_overtake_zoom(seed, tag, kind, W=45):
    f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
    if not f.exists():
        print(f"({kind}: no cache)"); return
    z = np.load(f); EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
    OV = np.asarray(z["OV"]) if "OV" in z.files else np.zeros_like(V)
    G = z["GAP"]; OPSI = np.asarray(z["OPSI"]) if "OPSI" in z.files else _heading(OX, OY)
    pov = _overtakes(z["PASS"])
    if not len(pov):
        print(f"({kind}: no overtakes to zoom)"); return
    k0 = int(pov[0]); w = slice(max(0, k0 - W), min(len(V), k0 + W)); he = _heading(EX, EY)
    b = _b()
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    axm.plot(b["left_x"], b["left_y"], color="0.6", lw=1); axm.plot(b["right_x"], b["right_y"], color="0.6", lw=1)
    sc = axm.scatter(EX[w], EY[w], c=V[w], cmap="viridis", s=16, zorder=4); axm.plot(EX[w], EY[w], "k-", lw=.5, alpha=.35)
    axm.plot(OX[w], OY[w], color=OPP_C, lw=1.3, ls="--", alpha=.8, zorder=3, label="opponent path")
    n = w.stop - w.start
    for frac, lab in [(0.18, "before"), (0.5, "at pass"), (0.85, "after")]:
        i = int(w.start + n * frac)
        _car(axm, EX[i], EY[i], he[i], EGO_C); _car(axm, OX[i], OY[i], OPSI[i], OPP_C)
        axm.annotate(lab, (EX[i], EY[i]), textcoords="offset points", xytext=(4, 7), fontsize=7.5)
    axm.scatter([EX[k0]], [EY[k0]], marker="*", s=220, color="crimson", edgecolor="k", zorder=9, label="overtake")
    axm.scatter([], [], marker="s", color=EGO_C, label="ego"); axm.scatter([], [], marker="s", color=OPP_C, label="opponent")
    xs = np.r_[EX[w], OX[w]]; ys = np.r_[EY[w], OY[w]]; pad = 0.6
    axm.set_xlim(xs.min() - pad, xs.max() + pad); axm.set_ylim(ys.min() - pad, ys.max() + pad)
    axm.set_aspect("equal"); axm.legend(loc="best", fontsize=8); plt.colorbar(sc, ax=axm, label="ego speed [m/s]", shrink=.8)
    axm.set_title(f"{kind}: overtake zoom (ego blue, opponent orange)", fontsize=10)
    t = (np.arange(len(V))[w] - k0) * 0.05
    axs.axvline(0, color="crimson", ls="--", lw=1)
    axs.plot(t, V[w], color=EGO_C, lw=1.8, label="ego speed"); axs.plot(t, OV[w], color=OPP_C, lw=1.5, label="opp speed")
    axb = axs.twinx(); axb.plot(t, G[w], color="#2a8f3a", lw=1.4, ls=":", label="gap"); axb.axhline(0, color="0.6", lw=.6)
    axs.set_xlabel("time relative to overtake [s]"); axs.set_ylabel("speed [m/s]"); axb.set_ylabel("signed gap [m]", color="#2a8f3a")
    axs.set_title(f"{kind}: states across the overtake", fontsize=10)
    h1, l1 = axs.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels(); axs.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_overtake_zoom_{kind}.{e}", dpi=150)
    plt.close(fig); print(f"wrote {OUT}/fig_overtake_zoom_{kind}.pdf (+.png)")


TZ = OUT / "tikz"


def emit_tikz(seed, tag, kind, W=45):
    """Simplified pgfplots of the overtake zoom: track edges + ego path (speed) + opponent path +
    before/at/after position markers + overtake star, and a states panel (speeds + gap). The rotated
    car glyphs stay PDF-only; TikZ shows paths + markers. Data -> .dat (vector, light)."""
    f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
    if not f.exists():
        return
    z = np.load(f); EX, EY, V = z["EX"], z["EY"], z["V"]; OX, OY = z["OX"], z["OY"]
    OV = np.asarray(z["OV"]) if "OV" in z.files else np.zeros_like(V); G = z["GAP"]
    pov = _overtakes(z["PASS"])
    if not len(pov):
        return
    k0 = int(pov[0]); lo = max(0, k0 - W); hi = min(len(V), k0 + W); w = slice(lo, hi)
    TZ.mkdir(parents=True, exist_ok=True); b = _b()

    def dat(name, cols, arrs):
        with open(TZ / name, "w") as fh:
            fh.write(" ".join(cols) + "\n")
            for row in zip(*arrs):
                fh.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    dat("ozoom_left.dat", ["x", "y"], [b["left_x"], b["left_y"]]); dat("ozoom_right.dat", ["x", "y"], [b["right_x"], b["right_y"]])
    dat(f"ozoom_{kind}_ego.dat", ["x", "y", "v"], [EX[w], EY[w], V[w]])
    dat(f"ozoom_{kind}_opp.dat", ["x", "y"], [OX[w], OY[w]])
    t = (np.arange(len(V))[w] - k0) * 0.05
    dat(f"ozoom_{kind}_st.dat", ["t", "v", "ov", "gap"], [t, V[w], OV[w], G[w]])
    xs = np.r_[EX[w], OX[w]]; ys = np.r_[EY[w], OY[w]]; pad = 0.6
    marks = []
    for frac in (0.18, 0.5, 0.85):
        i = int(lo + (hi - lo) * frac)
        marks.append((EX[i], EY[i], OX[i], OY[i]))
    egom = " ".join(f"({m[0]:.3f},{m[1]:.3f})" for m in marks)
    oppm = " ".join(f"({m[2]:.3f},{m[3]:.3f})" for m in marks)
    body = rf"""% Auto-generated by tools/paper_race_overtake_zoom.py --tikz ({kind}, band {tag.strip('_')})
% requires: \usepgfplotslibrary{{groupplots}}
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.6cm}}]
\nextgroupplot[width=7cm, axis equal image, hide axis, title={{{kind}: overtake zoom}},
    xmin={xs.min()-pad:.2f}, xmax={xs.max()+pad:.2f}, ymin={ys.min()-pad:.2f}, ymax={ys.max()+pad:.2f},
    colormap/viridis, point meta min=1.2, point meta max=2.8]
  \addplot[gray, thin] table[x=x,y=y] {{tikz/ozoom_left.dat}};
  \addplot[gray, thin] table[x=x,y=y] {{tikz/ozoom_right.dat}};
  \addplot[mesh, point meta=explicit, line width=1.2pt] table[x=x,y=y,meta=v] {{tikz/ozoom_{kind}_ego.dat}};
  \addplot[orange!80!black, thick, dashed] table[x=x,y=y] {{tikz/ozoom_{kind}_opp.dat}};
  \addplot[only marks, mark=square*, blue, mark size=2.5pt] coordinates {{{egom}}};
  \addplot[only marks, mark=square*, orange, mark size=2.5pt] coordinates {{{oppm}}};
  \addplot[only marks, mark=star, red, mark size=4pt] coordinates {{({EX[k0]:.3f},{EY[k0]:.3f})}};
\nextgroupplot[width=7cm, height=5cm, xlabel=time rel. overtake [s], ylabel=speed [m/s], grid=both,
    legend pos=south east]
  \addplot[blue, thick] table[x=t,y=v] {{tikz/ozoom_{kind}_st.dat}};
  \addplot[orange!80!black, thick] table[x=t,y=ov] {{tikz/ozoom_{kind}_st.dat}};
  \addplot[green!50!black, dotted, thick] table[x=t,y=gap] {{tikz/ozoom_{kind}_st.dat}};
  \draw[red, dashed] (axis cs:0,\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:0,\pgfkeysvalueof{{/pgfplots/ymax}});
  \legend{{ego speed, opp speed, gap [m]}}
\end{{groupplot}}
\end{{tikzpicture}}
"""
    (TZ / f"fig_overtake_zoom_{kind}.tex").write_text(body); print(f"wrote {TZ}/fig_overtake_zoom_{kind}.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--online", action="store_true", help="use online caches instead of frozen")
    ap.add_argument("--tikz", action="store_true", help="also emit pgfplots .tex + .dat")
    a = ap.parse_args()
    tag = "_10lap" if a.online else "_10lapfrozen"
    fig_track_opponents(a.seed, tag)
    for k in ["slower", "equal", "faster"]:
        fig_overtake_zoom(a.seed, tag, k)
        if a.tikz:
            emit_tikz(a.seed, tag, k)
