"""Learn for a few laps, bank the network, freeze it -- and watch the weights
keep changing around the track anyway.

    PYTHONPATH=... python3 scripts/anim_learn_then_freeze.py --seed 0

The question this answers is the one the project is built on: after learning
stops, is the policy a *function of the situation* or just a constant it
settled on? A frozen constant would draw flat lines on the right-hand panels.
A frozen function draws lines that rise into corners and fall on straights,
lap after lap, with no learning signal at all.

Two phases, one seed, one continuous recording:

1. **learning** -- the tuner drives ``--learn-laps`` laps from
   ``baselines.START`` with the fitted critic, theta-exploration and the
   per-metre clock, inside the adaptation box. At the end the network is a
   keep-best candidate: it is driven one frozen episode (not drawn) and banked
   at THAT score.
2. **frozen** -- the banked network is restored, learning and exploration
   are switched off, and it drives ``--frozen-laps`` laps. Whatever the
   weights do now is the network's mapping from features to theta, nothing
   else.

Outputs: ``paper/figures/learn_then_freeze.png`` (the whole recording plus a
panel of the frozen weights against arc length, sector-shaded) and
``docs/source/_static/anim/learn_then_freeze.gif``.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.ltc import (LTCCell, N_FEATURES, PolicyTuner,  # noqa: E402
                             WeightPolicy, features)
from mpcc_tuning.mpcc import WEIGHT_NAMES  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

FIG = ROOT / "paper" / "figures"
ANIM = ROOT / "docs" / "source" / "_static" / "anim"
INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
C_LEARN, C_FROZEN = "#4C6EF5", "#AE3EC9"
SEC_COL = ("#4C6EF5", "#0CA678", "#E8590C", "#AE3EC9")     # validated 4-hue set
SEC_NAME = ("straight", "long curve", "90-deg", "180-deg")
GROUPS = (("path", (0, 1, 2)), ("input", (3, 4, 5)), ("constraint", (6, 7)))
GCOL = ("#4C6EF5", "#0CA678", "#E8590C")


def roll(track_name, seed, learn_laps, frozen_laps, critic, steps_per_lap):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from experiments.online_from_baseline import run_frozen

    t = getattr(Track, track_name)()
    st = B.start(track_name)
    th0 = np.asarray(st.theta(), float)
    lo, hi = B.adaptation_box(track_name, 2.0)
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, name=f"ltf_{track_name}_{seed}")
    pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, lo, hi, seed=seed)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.0, delta_clip=1.0, seed=seed,
                     trust_region=0.01, clock="progress", critic=critic,
                     theta_explore=0.1 if critic == "fitted" else 0.0)
    s0 = (seed % 4) * t.length / 4.0
    v0 = 1.0 + 0.1 * (seed % 3)
    rec = []                                     # phase, x, y, v, s, sector, theta[8]

    # ---------------- phase 1: learning ----------------
    P = ScuderiaPlant(t, model="std", dt=0.05)
    P.max_steps = learn_laps * steps_per_lap
    s5 = P.reset(s0=s0, v0=v0); m.reset(); tu.reset()
    base = float(s5[4]); off = False
    th, u = tu.act(features(t, s5), P.state_dyn())
    for k in range(P.max_steps):
        s5n, r, off, done = P.step(u)
        rec.append((0, float(s5n[0]), float(s5n[1]), float(s5n[3]),
                    float(t.wrap(float(s5n[4]))),
                    int(t.sector(t.wrap(float(s5n[4])))), *np.exp(th)))
        progress = float(r) + (5.0 if off else 0.0)
        out = tu.learn(-0.05 - (5.0 if off else 0.0), P.state_dyn(),
                       features(t, s5n), off, ds=progress)
        if out[0] is None:
            break
        th, u = out
        if off or done:
            break
    learn_laps_done = (float(s5n[4]) - base) / t.length
    print(f"  learning: {learn_laps_done:.2f} laps{' then OFF' if off else ''}",
          flush=True)

    # ---------------- bank, with validation ----------------
    action = tu.end_episode(learn_laps_done, crashed=bool(off), validate=True)
    val = None
    if action == "validate":
        with tu.frozen():
            val_laps, val_off = run_frozen(m, pol, t, s0, v0, frozen_laps * steps_per_lap,
                                           features)
        val = (val_laps, val_off)
        tu.confirm_candidate(-1.0 if val_off else val_laps)
        print(f"  validation of the candidate, frozen: {val_laps:.2f} laps"
              f"{' OFF' if val_off else ''} -> "
              f"{'banked' if tu.best_score is not None else 'rejected'}", flush=True)
    if tu.best_score is None:
        print("  nothing banked (the learning episode crashed); freezing the "
              "network as it stands", flush=True)
        banked_from = "final network (nothing validated)"
    else:
        _, G, cp = tu._best
        pol.G[...] = G; pol.cell.p[...] = cp
        banked_from = f"validated banked network ({tu.best_score:.2f} laps frozen)"

    # ---------------- phase 2: frozen ----------------
    with tu.frozen():
        P = ScuderiaPlant(t, model="std", dt=0.05)
        P.max_steps = frozen_laps * steps_per_lap
        s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
        base = float(s5[4]); off = False
        th = pol.step(features(t, s5))
        for k in range(P.max_steps):
            u = m.value(P.state_dyn(), th)["u0"]
            s5, r, off, done = P.step(u)
            rec.append((1, float(s5[0]), float(s5[1]), float(s5[3]),
                        float(t.wrap(float(s5[4]))),
                        int(t.sector(t.wrap(float(s5[4])))), *np.exp(th)))
            th = pol.step(features(t, s5))       # the network keeps running
            if off or done:
                break
    frozen_laps_done = (float(s5[4]) - base) / t.length
    print(f"  frozen:   {frozen_laps_done:.2f} laps{' then OFF' if off else ''}",
          flush=True)
    return t, np.array(rec), dict(learn=learn_laps_done, frozen=frozen_laps_done,
                                  banked_from=banked_from, validation=val)


def _track_panel(ax, t, R=None, trail_upto=None):
    c = t.center
    g = np.gradient(c, axis=0); g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-g[:, 1], g[:, 0]])
    wl = np.array([float(t.width(v)[0]) for v in t.s]); wr = np.array([float(t.width(v)[1]) for v in t.s])
    for e in (c + n * wl[:, None], c - n * wr[:, None]):
        ax.plot(e[:, 0], e[:, 1], "-", color=INK, lw=1.0, alpha=0.8, zorder=2)
    sec = np.array([int(t.sector(float(v))) for v in t.s])
    from matplotlib.collections import LineCollection
    pts = c.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors=[SEC_COL[k] for k in sec[:-1]],
                                     linewidth=7.0, alpha=0.30, zorder=1))
    for k, nm in enumerate(SEC_NAME):
        ax.plot([], [], "-", color=SEC_COL[k], lw=6, alpha=0.4, label=nm)
    ax.set_aspect("equal"); ax.axis("off")
    if R is not None:
        upto = len(R) if trail_upto is None else trail_upto
        for ph, col in ((0, C_LEARN), (1, C_FROZEN)):
            sel = R[:upto][R[:upto, 0] == ph]
            if len(sel):
                ax.plot(sel[:, 1], sel[:, 2], "-", color=col, lw=1.4, alpha=0.9, zorder=3)


def _sector_spans(ax, sec_ticks, alpha=0.10):
    start = 0
    for i in range(1, len(sec_ticks) + 1):
        if i == len(sec_ticks) or sec_ticks[i] != sec_ticks[start]:
            ax.axvspan(start, i, color=SEC_COL[int(sec_ticks[start])], alpha=alpha, lw=0)
            start = i


def make_png(t, R, info, path, track_name):
    W = R[:, 6:6 + len(WEIGHT_NAMES)]
    phase, sec = R[:, 0].astype(int), R[:, 5].astype(int)
    switch = int(np.argmax(phase == 1)) if (phase == 1).any() else len(R)
    fig = plt.figure(figsize=(15, 8.2)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.25], hspace=0.45, wspace=0.12)
    axt = fig.add_subplot(gs[:, 0]); axw = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    _track_panel(axt, t, R)
    axt.plot([], [], "-", color=C_LEARN, lw=2, label=f"learning, {info['learn']:.1f} laps")
    axt.plot([], [], "-", color=C_FROZEN, lw=2, label=f"frozen network, {info['frozen']:.1f} laps")
    axt.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    axt.set_title(f"{track_name}: learn, bank, freeze -- one seed", loc="left",
                  fontsize=12, fontweight="bold", color=INK)
    x = np.arange(len(R))
    for ax, (gname, idx) in zip(axw[:3], GROUPS):
        _sector_spans(ax, sec)
        for c, i in zip(GCOL, idx):
            ax.plot(x, W[:, i], "-", color=c, lw=1.1)
            ax.annotate(WEIGHT_NAMES[i], (x[-1], W[-1, i]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=c, va="center")
        ax.axvline(switch, color=INK, lw=1.2, ls="--")
        ax.set_yscale("log"); ax.set_ylabel(gname, color=MUT, fontsize=9)
        ax.tick_params(colors=MUT, labelsize=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_xlim(0, len(R) * 1.06)
    axw[0].text(switch, axw[0].get_ylim()[1], "  learning off, banked network in",
                fontsize=8.5, color=INK, va="top")
    axw[0].set_title("weights the network emits, every tick (sector shaded behind)",
                     loc="left", fontsize=10, fontweight="bold", color=INK)
    axw[2].set_xlabel("control tick", color=MUT, fontsize=9)
    # ---- the point: frozen phase, theta against arc length, one lap overlaid
    ax = axw[3]
    fr = R[phase == 1]
    if len(fr):
        s_ = fr[:, 4]; Wf = fr[:, 6:6 + len(WEIGHT_NAMES)]
        spread = (np.log(Wf).max(0) - np.log(Wf).min(0))
        top = np.argsort(-spread)[:3]
        s_grid = np.linspace(0, t.length, 400)
        sec_grid = np.array([int(t.sector(float(v))) for v in s_grid])
        start = 0
        for i in range(1, len(sec_grid) + 1):
            if i == len(sec_grid) or sec_grid[i] != sec_grid[start]:
                ax.axvspan(s_grid[start], s_grid[min(i, len(s_grid) - 1)],
                           color=SEC_COL[sec_grid[start]], alpha=0.12, lw=0)
                start = i
        for j, i in enumerate(top):
            o = np.argsort(s_)
            ax.plot(s_[o], Wf[o, i] / np.exp(np.log(Wf[:, i]).mean()), ".", ms=2.2,
                    color=GCOL[j], label=f"{WEIGHT_NAMES[i]} (x{np.exp(spread[i]):.2f} range)")
        ax.set_yscale("log"); ax.set_xlim(0, t.length)
        ax.set_xlabel("arc length s (m)  --  frozen laps only", color=MUT, fontsize=9)
        ax.set_ylabel("theta / mean", color=MUT, fontsize=9)
        ax.set_title("FROZEN: the three weights that vary most, against position on the track",
                     loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
        ax.tick_params(colors=MUT, labelsize=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, f"frozen phase drives the {info['banked_from']}. Flat lines in the "
             f"bottom panel would mean the network learned a constant; structure means it "
             f"learned a function of the situation.", fontsize=8.5, color=MUT)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  wrote", path)


def make_gif(t, R, info, path, track_name, stride=10, fps=20, dpi=64):
    R = R[::stride]
    W = R[:, 6:6 + len(WEIGHT_NAMES)]; phase = R[:, 0].astype(int); sec = R[:, 5].astype(int)
    switch = int(np.argmax(phase == 1)) if (phase == 1).any() else len(R)
    n = len(R)
    fig = plt.figure(figsize=(12.6, 6.4), dpi=dpi); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1.0], hspace=0.32, wspace=0.16)
    axt = fig.add_subplot(gs[:, 0]); axw = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    _track_panel(axt, t)
    (trail_l,) = axt.plot([], [], "-", color=C_LEARN, lw=1.6, alpha=0.9)
    (trail_f,) = axt.plot([], [], "-", color=C_FROZEN, lw=1.6, alpha=0.9)
    (car,) = axt.plot([], [], "o", ms=9, color="#E03131", mec="white", mew=1.3, zorder=5)
    label = axt.text(0.02, 0.98, "", transform=axt.transAxes, fontsize=10, va="top",
                     family="monospace", color=INK)
    x = np.arange(n); lines = []
    for ax, (gname, idx) in zip(axw, GROUPS):
        _sector_spans(ax, sec, alpha=0.08)
        for c, i in zip(GCOL, idx):
            ax.plot(x, W[:, i], "-", color=c, lw=0.9, alpha=0.25)
            (ln,) = ax.plot([], [], "-", color=c, lw=1.8); (dot,) = ax.plot([], [], "o", ms=5, color=c)
            ax.annotate(WEIGHT_NAMES[i], (x[-1], W[-1, i]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=c, va="center", annotation_clip=False)
            lines.append((ln, dot, i))
        ax.axvline(switch, color=INK, lw=1.0, ls="--")
        ax.set_yscale("log"); ax.set_ylabel(gname, color=MUT, fontsize=9)
        ax.set_xlim(0, n * 1.06); ax.tick_params(colors=MUT, labelsize=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axw[2].set_xlabel("control tick", color=MUT, fontsize=9)
    fig.suptitle(f"{track_name}: learning for {info['learn']:.1f} laps, then the banked network "
                 f"frozen for {info['frozen']:.1f} laps", fontsize=11, color=INK)
    lap_len = t.length

    def frame(k):
        L = R[:k + 1][R[:k + 1, 0] == 0]; F = R[:k + 1][R[:k + 1, 0] == 1]
        trail_l.set_data(L[:, 1], L[:, 2]); trail_f.set_data(F[:, 1], F[:, 2])
        car.set_data([R[k, 1]], [R[k, 2]])
        ph = "LEARNING   (theta-exploration on)" if phase[k] == 0 else "FROZEN     banked network, no learning"
        label.set_text(f"{ph}\nv {R[k, 3]:4.2f} m/s   {SEC_NAME[sec[k]]}")
        label.set_color(C_LEARN if phase[k] == 0 else C_FROZEN)
        for ln, dot, i in lines:
            ln.set_data(x[:k + 1], W[:k + 1, i]); dot.set_data([x[k]], [W[k, i]])
        return [trail_l, trail_f, car, label] + [a for ln, dot, _ in lines for a in (ln, dot)]

    an = FuncAnimation(fig, frame, frames=n, interval=1000 / fps, blit=False)
    an.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig); print(f"  wrote {path}  ({n} frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn-laps", type=int, default=3)
    ap.add_argument("--frozen-laps", type=int, default=2)
    ap.add_argument("--steps-per-lap", type=int, default=1100, help="T2 at ~1.5 m/s")
    ap.add_argument("--critic", choices=("fitted", "mpcc"), default="fitted")
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args(argv)
    t, R, info = roll(a.track, a.seed, a.learn_laps, a.frozen_laps, a.critic, a.steps_per_lap)
    if not len(R):
        print("  no ticks recorded"); return 1
    FIG.mkdir(parents=True, exist_ok=True); ANIM.mkdir(parents=True, exist_ok=True)
    np.save(str(ROOT / "results" / f"learn_then_freeze_{a.track}_{a.seed}.npy"), R)
    make_png(t, R, info, FIG / "learn_then_freeze.png", a.track)
    make_gif(t, R, info, ANIM / "learn_then_freeze.gif", a.track, stride=a.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
