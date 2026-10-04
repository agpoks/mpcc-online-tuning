"""WHERE on the track the online tuner LEARNS/adapts the most. Colours the ego trajectory by the
"weight-adaptation intensity" -- the smoothed per-tick change of the emitted weight vector,
||d(log theta)/dt|| over all 9 weights -- so hot sections = where the policy is most actively changing
its weights in response to the situation (corners, overtakes, catch-ups). From the ONLINE
(explore+learn) caches, so it reflects actual online learning, not a frozen replay.

Also prints/plots a per-SECTOR summary (which of the 4 track sectors sees the most adaptation).
PDF+PNG and pgfplots TikZ. Reproducible:
    python3 tools/paper_race_learning.py --seed 0          # produces states_online_* (explore+learn)
    python3 tools/paper_race_weight_learning_map.py --seed 0 --tikz
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
CMAP = "inferno"


def _b():
    return np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")


def adapt_intensity(TH, win=25):
    """How far the weights are pushed from their race-average, per tick -- the situation-driven
    adaptation. Smooth each weight first (kills the zero-mean exploration jitter that a raw per-tick
    change is dominated by), z-score per weight, then the L2 deviation of the smoothed weight vector
    from its mean. Hot = the policy sets its weights most differently here (strongest adaptation)."""
    logw = np.log(np.clip(np.asarray(TH, float), 1e-6, None))
    k = np.ones(int(win)) / int(win)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, logw)
    z = (sm - sm.mean(0)) / (sm.std(0) + 1e-6)
    return np.linalg.norm(z, axis=1)


def _load(seed, kind, tag):
    f = OUT / f"states{tag}_ltc_{seed}_{kind}.npz"
    return np.load(f) if f.exists() else None


_L = Track.icra_t2_smooth().length


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -_L / 2] += _L; dS[dS > _L / 2] -= _L
    return np.concatenate([[0.0], np.cumsum(dS)]) / _L


def _dist_to_final(TH, lap, win=25):
    """Per-tick distance (z-scored, L2) of the smoothed weight vector from its FINAL-LAP average ->
    high while the tuner is still learning, 0 once it has settled. The settling signal."""
    logw = np.log(np.clip(np.asarray(TH, float), 1e-6, None))
    k = np.ones(int(win)) / int(win)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, logw)
    sd = sm.std(0) + 1e-6
    final = sm[lap >= max(lap.max() - 1.0, 0.0)].mean(0)      # last-lap mean = the converged weights
    return np.linalg.norm((sm - final) / sd, axis=1)


def settling(seed, tag, tikz=False):
    """Lap-faceted maps + a per-lap curve showing the weight adaptation SETTLING toward the final laps."""
    b = _b(); track = Track.icra_t2_smooth(); curves = {}
    for kind in KINDS:
        z = _load(seed, kind, tag)
        if z is None:
            continue
        EX, EY = z["EX"], z["EY"]; lap = _laps(z["S"]); dd = _dist_to_final(z["THETA"], lap)
        nlap = max(1, int(np.floor(lap.max()))); laps = list(range(1, min(nlap, 4) + 1)) or [1]
        vmax = float(np.percentile(dd, 98)) or 1.0
        fig, axs = plt.subplots(1, len(laps), figsize=(4.4 * len(laps), 4.6)); axs = np.atleast_1d(axs)
        per = []
        for ax, Lp in zip(axs, laps):
            m = (lap >= Lp - 1) & (lap <= Lp)
            ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
            if m.sum() > 1:
                ax.scatter(EX[m], EY[m], c=dd[m], cmap=CMAP, vmin=0, vmax=vmax, s=7, zorder=4)
            mv = float(dd[m].mean()) if m.any() else np.nan; per.append(mv)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"lap {Lp}  (dist {mv:.2f})", fontsize=10)
        curves[kind] = per
        fig.suptitle(f"vs {kind}: weight distance-to-FINAL over laps (bright=still learning, dark=settled)", fontsize=10.5)
        for e in ("pdf", "png"):
            fig.savefig(OUT / f"fig_weight_settling_{kind}.{e}", dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"wrote {OUT}/fig_weight_settling_{kind}.pdf (+.png)")
    if curves:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        col = {"static": "0.5", "slower": "#2a8f3a", "equal": "#1f4e8c", "faster": "#c23b3b"}
        for kind, per in curves.items():
            ax.plot(range(1, len(per) + 1), per, "-o", color=col.get(kind, "k"), label=kind)
        ax.set_xlabel("lap"); ax.set_ylabel("mean weight distance to final"); ax.grid(alpha=.3); ax.legend(fontsize=9)
        ax.set_title("Online weight adaptation SETTLING over the race (↓ = converging)", fontsize=10.5)
        for e in ("pdf", "png"):
            fig.savefig(OUT / f"fig_weight_settling_curve.{e}", dpi=150)
        plt.close(fig); print(f"wrote {OUT}/fig_weight_settling_curve.pdf (+.png)")
        print("mean distance-to-final per lap:")
        for kind, per in curves.items():
            print(f"  {kind:7s}: " + " -> ".join(f"{v:.2f}" for v in per))
        if tikz:
            TZ.mkdir(parents=True, exist_ok=True); plots = []
            for kind, per in curves.items():
                with open(TZ / f"settle_{kind}.dat", "w") as fh:
                    fh.write("lap dist\n"); [fh.write(f"{i + 1} {v:.4f}\n") for i, v in enumerate(per)]
                plots.append(rf"\addplot+[mark=*] table[x=lap,y=dist] {{tikz/settle_{kind}.dat}};" + f" \\addlegendentry{{{kind}}}")
            body = ("% Auto-generated by tools/paper_race_weight_learning_map.py --tikz\n"
                    "\\begin{tikzpicture}\n"
                    "\\begin{axis}[xlabel=lap, ylabel={weight distance to final}, width=9cm, height=6cm,\n"
                    "    grid=both, legend pos=north east, xtick=data, title={Online weight adaptation settling}]\n"
                    + "\n".join(plots) + "\n\\end{axis}\n\\end{tikzpicture}\n")
            (TZ / "fig_weight_settling_curve.tex").write_text(body); print(f"wrote {TZ}/fig_weight_settling_curve.tex")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frozen", action="store_true", help="use frozen caches (default: online, where it learns)")
    ap.add_argument("--tikz", action="store_true")
    a = ap.parse_args()
    tag = "_10lapfrozen" if a.frozen else "_online"
    b = _b(); track = Track.icra_t2_smooth()
    # shared colour scale across classes
    inten = {}; vmax = 1e-9
    for k in KINDS:
        z = _load(a.seed, k, tag)
        if z is None: continue
        inten[k] = adapt_intensity(z["THETA"]); vmax = max(vmax, np.percentile(inten[k], 98))
    present = list(inten)
    if not present:
        print(f"(no {tag} caches for seed {a.seed})"); return

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.8)); sc = None; sect = {}
    for ax, kind in zip(axs.ravel(), KINDS):
        z = _load(a.seed, kind, tag)
        if z is None:
            ax.set_axis_off(); continue
        EX, EY = z["EX"], z["EY"]; I = inten[kind]
        ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
        pts = np.array([EX, EY]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=plt.Normalize(0, vmax), linewidths=2.2, zorder=4)
        lc.set_array(I[:-1]); sc = ax.add_collection(lc)
        pov = np.where(np.diff(np.asarray(z["PASS"], float)) > 0)[0]
        if len(pov):
            ax.scatter(EX[pov], EY[pov], marker="*", s=90, facecolor="none", edgecolor="cyan", lw=1.2, zorder=6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # per-sector mean adaptation
        secs = np.array([int(track.sector(track.wrap(track.project(float(x), float(y))))) % 4 for x, y in zip(EX, EY)])
        sm = [float(I[secs == s].mean()) if (secs == s).any() else 0.0 for s in range(4)]
        sect[kind] = sm
        ax.set_title(f"{kind}  —  most adaptation in sector {int(np.argmax(sm))}", fontsize=10)
    if sc is not None:
        cb = fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=CMAP), ax=axs, shrink=.6, location="right", pad=.02)
        cb.set_label("weight-adaptation intensity  ||d(log w)/dt||  (hot = learns most)")
    fig.suptitle(f"Where the online tuner adapts the weights most over the race (ltc {'frozen' if a.frozen else 'online'}, seed {a.seed}); ★ = overtake", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_weight_learning_map.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_weight_learning_map.pdf (+.png)")
    print("per-sector mean adaptation (0..3):")
    for k in present:
        print(f"  {k:7s}: " + "  ".join(f"s{s} {sect[k][s]:.3f}" for s in range(4)) + f"   -> peak sector {int(np.argmax(sect[k]))}")

    settling(a.seed, tag, a.tikz)  # lap-faceted settling maps + the convergence curve (+curve TikZ)

    if a.tikz:
        TZ.mkdir(parents=True, exist_ok=True)
        with open(TZ / "wlearn_left.dat", "w") as fh:
            fh.write("x y\n"); [fh.write(f"{x:.4f} {y:.4f}\n") for x, y in zip(b["left_x"], b["left_y"])]
        with open(TZ / "wlearn_right.dat", "w") as fh:
            fh.write("x y\n"); [fh.write(f"{x:.4f} {y:.4f}\n") for x, y in zip(b["right_x"], b["right_y"])]
        cells = []
        for kind in present:
            z = _load(a.seed, kind, tag); EX, EY = z["EX"], z["EY"]; I = inten[kind]
            ds = max(1, len(EX) // 1500)
            with open(TZ / f"wlearn_{kind}.dat", "w") as fh:
                fh.write("x y a\n"); [fh.write(f"{EX[i]:.4f} {EY[i]:.4f} {I[i]:.4f}\n") for i in range(0, len(EX), ds)]
            cells.append(rf"""\nextgroupplot[title={{{kind}}}]
  \addplot[gray, thin] table[x=x,y=y] {{tikz/wlearn_left.dat}};
  \addplot[gray, thin] table[x=x,y=y] {{tikz/wlearn_right.dat}};
  \addplot[mesh, point meta=explicit, line width=1.3pt] table[x=x,y=y,meta=a] {{tikz/wlearn_{kind}.dat}};""")
        body = rf"""% Auto-generated by tools/paper_race_weight_learning_map.py --tikz
% requires: \usepgfplotslibrary{{groupplots}}
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 2, horizontal sep=0.5cm, vertical sep=1cm}},
    width=7cm, axis equal image, hide axis, colormap/hot, point meta min=0, point meta max={vmax:.3f},
    colorbar, colorbar style={{ylabel=weight-adaptation intensity}}]
{chr(10).join(cells)}
\end{{groupplot}}
\end{{tikzpicture}}
"""
        (TZ / "fig_weight_learning_map.tex").write_text(body); print(f"wrote {TZ}/fig_weight_learning_map.tex")


if __name__ == "__main__":
    main()
