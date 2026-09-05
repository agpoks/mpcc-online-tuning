"""A side-by-side GIF: the same policy, standing start vs a warmed start.

    python3 scripts/anim_scenario.py <net.npz> [--box-from <fitted.npz>] [--seed 2]

Left panel drives the network from a COLD standing start; right panel drives it
after a warm-up lap under the safe baseline (the network watching, so its memory
warms). Both on the same track, sectors shaded, the car and its speed shown, and
a live speed trace underneath. The point to see: the cold car builds too much
speed into the hairpin and leaves the track where the solver's QP fails, while
the warmed car settles into a moderate regime and stays on. Where the cold car
crashes is marked.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.ltc import LTCCell, N_FEATURES, WeightPolicy, features  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

ANIM = ROOT / "docs" / "source" / "_static" / "anim"
INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
SEC_COL = ("#4C6EF5", "#0CA678", "#E8590C", "#AE3EC9")
RED = "#E03131"


def _pol(z, box, th0, seed):
    p = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, box[0], box[1], seed=seed)
    p.G[...] = z["G"]; p.cell.p[...] = z["cell_p"]
    return p


def roll_cold(m, pol, t, s0, v0, steps, features):
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = steps
    s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
    rec = []; th = pol.step(features(t, s5)); off = tr = False
    for _ in range(steps):
        out = m.value(P.state_dyn(), th); u = out["u0"]
        s5, r, off, tr = P.step(u)
        rec.append((float(s5[0]), float(s5[1]), float(s5[3]), int(out["status"])))
        th = pol.step(features(t, s5))
        if off or tr:
            break
    return np.array(rec), bool(off)


def roll_warm(m, pol, t, s0, v0, steps, warm_theta, warm_steps, features):
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = warm_steps + steps
    s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
    for _ in range(warm_steps):
        pol.step(features(t, s5)); u = m.value(P.state_dyn(), warm_theta)["u0"]
        s5, r, off, tr = P.step(u)
    rec = []; th = pol.step(features(t, s5)); off = tr = False
    for _ in range(steps):
        out = m.value(P.state_dyn(), th); u = out["u0"]
        s5, r, off, tr = P.step(u)
        rec.append((float(s5[0]), float(s5[1]), float(s5[3]), int(out["status"])))
        th = pol.step(features(t, s5))
        if off or tr:
            break
    return np.array(rec), bool(off)


def _track(ax, t):
    c = t.center
    g = np.gradient(c, axis=0); g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-g[:, 1], g[:, 0]])
    wl = np.array([float(t.width(v)[0]) for v in t.s]); wr = np.array([float(t.width(v)[1]) for v in t.s])
    for e in (c + n * wr[:, None], c - n * wl[:, None]):
        ax.plot(e[:, 0], e[:, 1], "-", color=INK, lw=1.0, zorder=2)
    sec = np.array([int(t.sector(float(v))) for v in t.s])
    pts = c.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors=[SEC_COL[k] for k in sec[:-1]], linewidth=6, alpha=0.30, zorder=1))
    ax.set_aspect("equal"); ax.axis("off")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--box-from", default=None)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--warm-steps", type=int, default=1100)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from mpcc_tuning.acados_mpcc import AcadosMPCC
    t = getattr(Track, a.track)(); st = B.start(a.track); th0 = np.asarray(st.theta(), float)
    z = np.load(a.npz)
    if "lo" in z and "hi" in z:
        box = (z["lo"], z["hi"])
    else:
        b = np.load(a.box_from or str(ROOT / "results" / f"fitted_policy_{a.track}_kv0.50.npz")); box = (b["lo"], b["hi"])
    s0 = (a.seed % 4) * t.length / 4.0; v0 = 1.0 + 0.1 * (a.seed % 3)
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, name=f"anim_sc_{os.getpid()}")
    cold, coff = roll_cold(m, _pol(z, box, th0, a.seed), t, s0, v0, st.steps, features)
    warm, woff = roll_warm(m, _pol(z, box, th0, a.seed), t, s0, v0, st.steps, th0, a.warm_steps, features)
    print(f"  cold {len(cold)} ticks {'CRASH' if coff else 'ok'}; warm {len(warm)} ticks {'CRASH' if woff else 'ok'}", flush=True)

    C, W = cold[::a.stride], warm[::a.stride]
    n = max(len(C), len(W))
    fig = plt.figure(figsize=(12.6, 6.6), dpi=64); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.1)
    axc, axw = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axsc, axsw = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for ax, R, off, title, col in ((axc, C, coff, "STANDING start (cold)", RED),
                                   (axw, W, woff, "after a WARM-UP lap", "#0CA678")):
        _track(ax, t)
        ax.set_title(title, fontsize=12, fontweight="bold", color=col, loc="left")
        if off and len(R):
            ax.plot(R[-1, 0], R[-1, 1], "x", color=RED, ms=16, mew=3, zorder=6)
    (trc,) = axc.plot([], [], "-", color=RED, lw=1.6, alpha=0.9)
    (carc,) = axc.plot([], [], "o", ms=9, color=RED, mec="white", mew=1.3, zorder=5)
    (trw,) = axw.plot([], [], "-", color="#0CA678", lw=1.6, alpha=0.9)
    (carw,) = axw.plot([], [], "o", ms=9, color="#0CA678", mec="white", mew=1.3, zorder=5)
    for ax, R, col in ((axsc, C, RED), (axsw, W, "#0CA678")):
        ax.plot(np.arange(len(R)), R[:, 2], "-", color=col, lw=1, alpha=0.3)
        ax.axhline(4.0, color=RED, lw=0.8, ls=":"); ax.set_ylim(0, 4.5)
        ax.set_ylabel("v (m/s)", color=MUT, fontsize=8); ax.tick_params(colors=MUT, labelsize=7)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    (spc,) = axsc.plot([], [], "-", color=RED, lw=1.8)
    (spw,) = axsw.plot([], [], "-", color="#0CA678", lw=1.8)
    lab = fig.text(0.5, 0.92, "", ha="center", fontsize=9, color=INK)

    def frame(k):
        kc, kw = min(k, len(C) - 1), min(k, len(W) - 1)
        trc.set_data(C[:kc + 1, 0], C[:kc + 1, 1]); carc.set_data([C[kc, 0]], [C[kc, 1]])
        trw.set_data(W[:kw + 1, 0], W[:kw + 1, 1]); carw.set_data([W[kw, 0]], [W[kw, 1]])
        spc.set_data(np.arange(kc + 1), C[:kc + 1, 2]); spw.set_data(np.arange(kw + 1), W[:kw + 1, 2])
        lab.set_text(f"cold {C[kc,2]:.1f} m/s   |   warmed {W[kw,2]:.1f} m/s     (dotted line = 4 m/s, where the hairpin QP fails)")
        return [trc, carc, trw, carw, spc, spw, lab]

    fig.suptitle(f"{a.track}: the same policy, cold vs after a warm-up lap", fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left", y=1.0)
    an = FuncAnimation(fig, frame, frames=n, interval=50, blit=False)
    ANIM.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else ANIM / "scenario_cold_vs_warm.gif"
    an.save(str(out), writer=PillowWriter(fps=20)); plt.close(fig)
    print(f"  wrote {out}  ({n} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
