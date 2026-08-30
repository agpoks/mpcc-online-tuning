"""Every figure the docs use, into ``docs/source/_static/plots/``.

    python scripts/make_figures.py                 # all of them
    python scripts/make_figures.py --only geometry

``geometry`` is a drawing and costs nothing. The other three run the real
thing: the gradient check solves the NLP a few hundred times, the tuning curve
is an actual online run, and the plant comparison needs ``scuderia_gym_jax`` on
the path. Figures are committed, because a docs build cannot run IPOPT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static" / "plots"
INK, BLUE, RED, GREEN = "#222222", "#1f4e9c", "#c1272d", "#2a9d5c"
AMBER, SURF = "#e8a33d", "#fcfcfb"
# Categorical hues in FIXED order, never cycled. Validated against the
# colourblind checks: worst adjacent pair is green/red at dE 8.4 under
# deuteranopia, which is legal only with a secondary encoding, and amber sits at
# 2.1:1 against the surface. Both are why every series below is *direct
# labelled* rather than identified by colour alone.
CAT = (BLUE, RED, GREEN, AMBER)


def fig_geometry():
    """What the two errors in the cost function actually are.

    Every other page assumes you can see this picture. ``e_c`` is across the
    path and is the thing you want small; ``e_l`` is *along* it and exists only
    because the reference point is chosen by the optimiser rather than by
    projection. Without a penalty on ``e_l`` the solver's cheapest move is to
    race ``s`` forward and collect the progress reward while leaving the car
    behind -- which is the failure the lag term exists to prevent.
    """
    t = Track.oval()
    s_ref = 6.6
    p = np.array([float(t.pos(s_ref)[0]), float(t.pos(s_ref)[1])])
    phi = float(t.tangent_angle(s_ref))
    tang = np.array([np.cos(phi), np.sin(phi)])
    norm = np.array([np.sin(phi), -np.cos(phi)])
    car = p + 0.42 * norm + 0.72 * tang        # off the line and ahead of it

    fig, ax = plt.subplots(figsize=(8.2, 4.15))
    ax.plot(t.center[:, 0], t.center[:, 1], color=INK, lw=1.0, ls="--", alpha=0.6)
    for sgn in (+1, -1):
        off = t.center + sgn * t.half_width * np.stack(
            [np.gradient(t.center[:, 1]), -np.gradient(t.center[:, 0])], axis=1
            ) / np.linalg.norm(np.gradient(t.center, axis=0), axis=1)[:, None]
        ax.plot(off[:, 0], off[:, 1], color=INK, lw=1.2, alpha=0.85)

    foot = car - np.dot(car - p, norm) * norm
    ax.plot([p[0], foot[0]], [p[1], foot[1]], color=GREEN, lw=2.2, zorder=4)
    ax.plot([foot[0], car[0]], [foot[1], car[1]], color=RED, lw=2.2, zorder=4)
    ax.plot(*p, "o", ms=9, color=INK, zorder=5)
    ax.plot(*car, "*", ms=19, color=BLUE, zorder=5)
    ax.arrow(*p, *(1.15 * tang), head_width=0.13, color=INK, alpha=0.8,
             length_includes_head=True, zorder=3)

    mid_l = (p + foot) / 2
    mid_c = (foot + car) / 2
    ax.annotate("$e_l$  lag error\n(along the path)", mid_l, (mid_l[0] - 1.5, mid_l[1] - 0.85),
                color=GREEN, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.annotate("$e_c$  contouring error\n(across the path)", mid_c,
                (mid_c[0] + 1.75, mid_c[1] + 0.30), color=RED, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.annotate("reference point $p(s)$\n$s$ is a decision variable, not\na projection of the car", p,
                (p[0] - 1.9, p[1] + 1.05), fontsize=8.5, ha="center", color=INK,
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    ax.annotate("the car", car, (car[0] + 1.15, car[1] - 0.72), fontsize=9,
                color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    ax.set_title("model predictive contouring control: the two errors in the cost\n"
                 r"$J=\sum_k q_c\,e_c^2 + q_l\,e_l^2 - q_v\,v_s\,\Delta t"
                 r" + r_d\,\delta^2 + r_a\,a^2$", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    c = (p + car) / 2
    ax.set_xlim(c[0] - 3.5, c[0] + 3.5); ax.set_ylim(c[1] - 1.75, c[1] + 1.75)
    fig.tight_layout()
    fig.savefig(OUT / "mpcc_geometry.png", dpi=160)
    print("  wrote mpcc_geometry.png")


def fig_gradient_check():
    """The envelope-theorem gradient against finite differences.

    The claim is that the derivative of the optimal value with respect to the
    cost weights is just the partial derivative of the Lagrangian at the
    solution -- no differentiating *through* the solver. If that is right, the
    points lie on the diagonal, and they cost one gradient evaluation rather
    than 2 x 6 extra solves.
    """
    t = Track.oval()
    # A high iteration cap because these are cold solves: for the finite
    # differences to mean anything, all three problems have to be solved to
    # convergence from the same starting point, and the online loop's budget
    # (which leans on warm starts) is not available here.
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=20, dt=0.05, max_iter=3000)
    theta = MPCCWeights().to_log()

    # States the controller actually visits, rather than points placed on the
    # centreline by hand -- half of those are far enough off the path that a
    # cold solve does not converge, and checking a gradient at a state the
    # solver failed on would be checking nothing.
    from examples.tune_online import Plant
    plant = Plant(t, dt=0.05)
    s5 = plant.reset()
    states = []
    for k in range(90):
        u = m.value(s5, theta)["u0"]
        s5, _r, off, _tr = plant.step(u)
        if off:
            break
        if k % 10 == 9:
            states.append(s5.copy())

    exact, fd = [], []
    n_skipped = 0
    for s5 in states:
        # Every solve starts from the same initial guess. Warm-starting would
        # make the two perturbed solves land in slightly different places for
        # reasons that have nothing to do with theta, and the difference would
        # show up in the finite difference as noise.
        m.reset()
        out = m._solve(s5, theta)
        if not out["ok"]:
            n_skipped += 1
            continue
        g = m.grad_theta(out, s5, theta)
        eps = 1e-4
        for j in range(len(theta)):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += eps; tm[j] -= eps
            m.reset(); a = m._solve(s5, tp)
            m.reset(); b = m._solve(s5, tm)
            if not (a["ok"] and b["ok"]):
                n_skipped += 1
                continue
            exact.append(g[j])
            fd.append((a["value"] - b["value"]) / (2 * eps))
    exact, fd = np.array(exact), np.array(fd)
    cos = float(exact @ fd / (np.linalg.norm(exact) * np.linalg.norm(fd)))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = max(np.abs(exact).max(), np.abs(fd).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1.0, ls="--", alpha=0.6)
    ax.plot(fd, exact, "o", ms=5.5, color=BLUE, alpha=0.75)
    ax.set_xlabel("finite differences   (12 extra solves per point)")
    ax.set_ylabel("envelope theorem   (free)")
    ax.set_title(f"the gradient through the solver is free and exact\n"
                 f"cosine = {cos:.5f} over {len(exact)} components, "
                 f"at states the controller visits", fontsize=10)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "gradient_check.png", dpi=160)
    print(f"  wrote gradient_check.png  (cosine {cos:.6f}, n={len(exact)}, "
          f"{n_skipped} non-converged probes skipped)")


def fig_rti():
    """When is the memoryless gradient the one you think it is?

    Two panels, both from experiments/rti_influence.py. Left: how far the
    sensitivity computed through the warm-started loop is from the one the
    envelope theorem predicts, as a function of solver effort. Right: how fast
    a perturbation of the warm start decays, which is what sets how far back
    the influence has to be carried.
    """
    import json
    runs = []
    for plant, label in (("bicycle", "kinematic bicycle"),
                         ("scuderia", "fitted tyres (ST)")):
        f = ROOT / "benchmarks" / "results" / f"rti_{plant}.json"
        if f.exists():
            runs.append((label, json.loads(f.read_text())))
    if not runs:
        print("  skipped rti: run experiments/rti_influence.py first")
        return

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    marks = ["o", "s"]
    for (label, d), mk in zip(runs, marks):
        sw = d["sweep"]
        ks = sorted(int(k) for k in sw)
        ax.plot(ks, [sw[str(k)]["cos"] for k in ks], mk + "-", color=INK if mk == "o" else "0.45",
                linewidth=1.6, markersize=5, label=label)
    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(0.0, color=RED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("solver iterations per control tick")
    ax.set_ylabel("cosine to the memoryless (envelope) gradient")
    # Mark the SQP-RTI result, which is the paper's actual finding: one full
    # QP agrees with the memoryless gradient exactly in direction. The
    # max_iter=1 point on the same axis is a FAILED interior-point solve and is
    # labelled as such, because it is the number that misleads.
    for (label, d), mk in zip(runs, marks):
        if d.get("cos_sqp") is not None:
            ax.plot([1.0], [d["cos_sqp"]], "*", markersize=14,
                    color=GREEN, zorder=6,
                    label=None if mk == "s" else "genuine SQP-RTI (one full QP)")
    ax.annotate("capped IPOPT:\nthe solve has failed", xy=(1.0, -0.30),
                xytext=(2.0, -0.48), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_title("a genuine RTI agrees exactly in direction;\n"
                 "a capped interior-point solve does not converge",
                 fontsize=10)
    ax.set_ylim(-0.6, 1.15)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="lower right")

    for (label, d), mk in zip(runs, marks):
        dec = d["decay"]
        t = np.arange(len(dec)) * 0.05
        bx.semilogy(t, np.maximum(dec, 1e-12), "-", linewidth=1.6,
                    color=INK if mk == "o" else "0.45",
                    label=f"{label}   $\\rho \\approx$ {d['rho']:.2f}")
    bx.set_xlabel("time since the perturbation [s]")
    bx.set_ylabel("relative size of the perturbation")
    bx.set_title("the warm start is memory, and it decays\n"
                 "geometrically -- over tens of ticks", fontsize=10)
    bx.grid(alpha=0.25, linewidth=0.6, which="both")
    bx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rti_influence.png", dpi=170)
    print("  wrote rti_influence.png")


def _load(name):
    import json
    return json.load(open(ROOT / "benchmarks" / "results" / name))


def fig_tracks():
    """The three tracks, with the named sectors drawn on them.

    A map, not a chart: the job is identity over geometry, so the sectors are
    categorical colour on the centreline and everything else is recessive. This
    figure is also the argument -- the 90-degree and 180-degree corners on the
    circuit are drawn at the *same radius*, so the reader can see that curvature
    at a point cannot tell them apart.
    """
    tracks = [("oval", Track.oval()), ("mixed", Track.mixed()),
              ("circuit", Track.circuit())]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, (name, tr) in zip(axes, tracks):
        n = len(tr.center)
        sec = np.array([tr.sector(v) for v in tr.s])
        # Corridor first, as ground rather than as a series.
        ax.plot(tr.center[:, 0], tr.center[:, 1], "-", color="0.86",
                linewidth=13, solid_capstyle="round", zorder=1)
        for k in range(4):
            m = sec == k
            if not m.any():
                continue
            xy = np.where(m[:, None], tr.center, np.nan)
            ax.plot(xy[:, 0], xy[:, 1], "-", color=CAT[k], linewidth=3.0,
                    solid_capstyle="butt", zorder=2)
        # Direct labels on the corners, which is the secondary encoding the
        # palette check requires -- and it is what names a corner anyway.
        for s0, s1, dpsi, kp in tr.corners():
            span = (s1 - s0) % tr.length
            mid = (s0 + 0.5 * span) % tr.length
            q = np.array(tr.pos(mid)).ravel()
            ax.annotate(f"{abs(np.degrees(dpsi)):.0f}$\\degree$\n{1/max(kp,1e-9):.1f} m",
                        xy=q, fontsize=6.5, ha="center", va="center", color=INK,
                        bbox=dict(boxstyle="round,pad=0.18", fc=SURF, ec="0.8", lw=0.5),
                        zorder=4)
        ax.set_aspect("equal")
        ax.axis("off")
    # One scale for all three panels. Auto-scaling each makes the 26.7 m oval
    # look the size of the 47.2 m circuit, which misleads on exactly the
    # comparison this figure exists to support.
    half = max(max(np.ptp(tr.center[:, k]) for k in (0, 1))
               for _, tr in tracks) / 2 * 1.15
    for ax, (name, tr) in zip(axes, tracks):
        cx, cy = tr.center[:, 0].mean(), tr.center[:, 1].mean()
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f"{name}   {tr.length:.1f} m", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], color=CAT[k], lw=3.0, label=Track.SECTOR_NAMES[k])
               for k in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Only the circuit has all four sector types -- and its 90 and 180 "
                 "corners share a radius,\nwhich is why pointwise curvature cannot "
                 "separate them", fontsize=9.5, color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "tracks.png", dpi=170, bbox_inches="tight")
    print("  wrote tracks.png")


def fig_reversal():
    """The result that reverses between tracks.

    Change over time with identity, so: lines over episodes, one small multiple
    per track, mean over six seeds with the full seed range as a band. Shared y
    axis, because the whole point is that the two panels are the same
    measurement and land in different places.
    """
    oval = _load("per_segment_seeds.json")["runs"]
    circ = _load("per_sector.json")["runs"]
    panels = [
        ("oval, 26.7 m -- mostly straight",
         [("global", "global", BLUE), ("per_segment", "3 curvature bins", RED)], oval),
        ("circuit, 47.2 m -- 82% corners",
         [("global", "global", BLUE), ("curvature3", "3 curvature bins", RED),
          ("sector4", "4 named sectors", GREEN)], circ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
    for n, (ax, (title, series, runs)) in enumerate(zip(axes, panels)):
        # The right panel's three arms live inside 1 m of each other, so at the
        # shared axis the ordering between them is invisible -- and the shared
        # axis is the whole point, because the headline is that one panel
        # collapses and the other does not. An inset resolves the detail
        # without a second y-scale on the same axes.
        ins = ax.inset_axes([0.30, 0.10, 0.66, 0.34]) if n == 1 else None
        for key, label, col in series:
            cov = np.array([r["covered"] for r in runs if r["mode"] == key])
            if not len(cov):
                continue
            ep = np.arange(cov.shape[1])
            ax.fill_between(ep, cov.min(0), cov.max(0), color=col, alpha=0.16,
                            linewidth=0)
            ax.plot(ep, cov.mean(0), "-", color=col, linewidth=2.0, zorder=3,
                    label=label)
            if ins is None:
                ax.annotate(f" {label}", xy=(ep[-1], cov.mean(0)[-1]), fontsize=8,
                            color=col, va="center", ha="left", zorder=4)
            else:
                ins.plot(ep, cov.mean(0), "-", color=col, linewidth=1.6)
                ins.annotate(f" {label}", xy=(ep[-1], cov.mean(0)[-1]), fontsize=6.5,
                             color=col, va="center", ha="left")
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_xlim(0, 32)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ins is not None:
            ins.set_xlim(0, 34)
            ins.set_ylim(78.4, 80.3)
            ins.set_yticks([79, 80])
            ins.tick_params(labelsize=6.5, length=2)
            ins.grid(alpha=0.2, linewidth=0.5)
            ins.set_title("detail: last 8 episodes differ by <1 m", fontsize=6.5,
                          color="0.35", pad=2)
            for sp in ("top", "right"):
                ins.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc="center left",
                      bbox_to_anchor=(0.02, 0.72))
    axes[0].set_ylabel("distance covered [m]")
    fig.suptitle("The same comparison reverses between tracks: scheduling rescues a "
                 "collapse on the oval,\nand costs a little where there is no collapse "
                 "to rescue.  Mean of 6 seeds; band is the seed range.",
                 fontsize=9.5, color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "reversal.png", dpi=170, bbox_inches="tight")
    print("  wrote reversal.png")


def fig_overtake():
    """Outcome over a 2-D weight sweep: three states, so a labelled state grid.

    Not a heat map -- the quantity is categorical (followed / passed / passed
    then left the track), and a sequential ramp would imply an ordering between
    "followed" and "crashed" that does not exist.
    """
    rows = _load("overtake_or_follow.json")
    qv = sorted({r["q_v"] for r in rows})
    qc = sorted({r["q_c"] for r in rows}, reverse=True)
    state = {"follow": GREEN, "pass": BLUE, "off": RED}
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for r in rows:
        i, j = qc.index(r["q_c"]), qv.index(r["q_v"])
        passed = r["pass_step"] is not None
        k = "off" if r["off"] else ("pass" if passed else "follow")
        ax.add_patch(plt.Rectangle((j - .46, i - .46), .92, .92, facecolor=state[k],
                                   alpha=0.20, edgecolor=state[k], linewidth=1.4))
        ax.text(j, i, f"{r['covered']:.0f} m", ha="center", va="center",
                fontsize=8.5, color=INK, zorder=3)
    # The boundary the measurement found: q_v/q_c = 1, a straight line in logs.
    b = [(j, i) for i, c in enumerate(qc) for j, v in enumerate(qv) if v / c > 1]
    ax.plot([j - 0.5 for j, _ in b][:1] * 2, [-0.5, len(qc) - 0.5], alpha=0)
    xs, ys = [], []
    for i, c in enumerate(qc):
        first = next((j for j, v in enumerate(qv) if v / c > 1), len(qv))
        xs += [first - 0.5, first - 0.5]
        ys += [i - 0.5, i + 0.5]
    ax.plot(xs, ys, "-", color=INK, linewidth=1.8, zorder=4)
    # The boundary label goes in the title, not on the plot: at 15 cells every
    # in-plot position collides with a value, and the values are the evidence.
    ax.set_xticks(range(len(qv)), [f"{v:g}" for v in qv])
    ax.set_yticks(range(len(qc)), [f"{c:g}" for c in qc])
    ax.set_xlabel("$q_v$  (progress weight)")
    ax.set_ylabel("$q_c$  (contouring weight)")
    ax.set_xlim(-0.5, len(qv) - 0.5)
    ax.set_ylim(-0.5, len(qc) - 0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=c, markeredgecolor=c, alpha=0.6, label=l)
               for l, c in (("followed", GREEN), ("passed", BLUE),
                            ("passed, then off-track", RED))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("Behaviour is the ratio $q_v/q_c$; safety is $q_v$ alone\n"
                 "black line: $q_v/q_c = 1$   ·   cells show distance covered",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "overtake_grid.png", dpi=170, bbox_inches="tight")
    print("  wrote overtake_grid.png")


def fig_spielberg():
    """The published circuit, and why it is the one that can measure behaviour.

    The point is the speed limit each corner imposes, so the centreline is
    coloured by ``sqrt(a_lat_max / kappa)`` -- a sequential ramp, because the
    quantity is a magnitude with an order. The synthetic circuit is shown beside
    it at the same scale and on the same ramp, where it is almost uniformly at
    the cap and therefore cannot discriminate between weight settings.
    """
    from matplotlib.collections import LineCollection
    from mpcc_tuning.model import A_LAT_MAX, SPEED_MAX

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    tracks = [("circuit (synthetic)", Track.circuit()),
              ("Spielberg (F1TENTH 1:10)", Track.spielberg())]
    for ax, (name, tr) in zip(axes, tracks):
        k = np.abs([tr.curvature(v) for v in tr.s])
        vmax = np.minimum(np.sqrt(A_LAT_MAX / np.maximum(k, 1e-6)), SPEED_MAX)
        pts = tr.center.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(1.5, SPEED_MAX),
                            linewidth=4.0, capstyle="round")
        lc.set_array(vmax[:-1])
        ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")
        # What matters is the *slowest corner the car actually reaches*, not
        # the fraction of the lap at the cap. Measured, that fraction points
        # the wrong way: Spielberg is at the cap for 97% of its lap and the
        # synthetic circuit for 58%, yet it is Spielberg that discriminates
        # between weight settings. A track punishes a bad weight only where it
        # is grip-limited at all, and the circuit's *hardest* corner still
        # allows 3.94 m/s against a 4.0 cap.
        driven = 80.0                       # ~400 steps at the speeds reached
        m = tr.s <= driven
        ax.plot(*tr.center[m].T, "-", color=INK, linewidth=0.9, alpha=0.55,
                zorder=4)
        ax.set_title(f"{name}\n{tr.length:.0f} m · "
                     f"{driven / tr.length:.2f} laps in an episode · "
                     f"slowest corner driven {vmax[m].min():.2f} m/s",
                     fontsize=9, color=INK)
    cb = fig.colorbar(lc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("grip-limited corner speed [m/s]   (vehicle cap 4.0)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("A track measures a weight policy only where it is grip-limited. "
                 "The synthetic circuit's\n*hardest* corner still allows 3.94 m/s "
                 "against a 4.0 cap, so nothing is ever punished; the thin line "
                 "marks the section actually driven.",
                 fontsize=9.5, color=INK, y=1.03)
    fig.savefig(OUT / "spielberg.png", dpi=170, bbox_inches="tight")
    print("  wrote spielberg.png")


def fig_behaviour():
    """Behaviour as two axes: what it achieves, and what it decides."""
    d = _load("behaviour_modes.json")["summary"]
    post = ["stay_behind", "overtake_when_safe", "always_try"]
    aggr = ["cautious", "neutral", "aggressive"]
    kinds = sorted({k.split("/")[0] for k in d}) if "/" in next(iter(d)) else [None]
    kinds = [k for k in ("dynamic", "static") if k in kinds] or [None]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 4.0),
                             squeeze=False)
    x = np.arange(len(aggr))
    for ax, kind in zip(axes[0], kinds):
        for i, p_ in enumerate(post):
            key = (lambda g: f"{kind}/{p_}/{g}") if kind else (lambda g: f"{p_}/{g}")
            cov = [d[key(g)]["covered"] for g in aggr]
            sd = [d[key(g)]["sd"] for g in aggr]
            ax.errorbar(x + (i - 1) * 0.13, cov, yerr=sd, marker="o", markersize=7,
                        linewidth=2.0, capsize=3, color=CAT[i],
                        label=p_.replace("_", " "))
            for j, g in enumerate(aggr):
                n = d[key(g)]["passes"]
                if n > 0:
                    ax.annotate(f"{n:.2f}", xy=(x[j] + (i - 1) * 0.13, cov[j]),
                                xytext=(0, 9), textcoords="offset points",
                                fontsize=7, ha="center", color=CAT[i])
        ax.set_xticks(x, aggr)
        ax.set_xlabel("aggression")
        ax.grid(alpha=0.22, linewidth=0.6, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{kind or 'opponent'} obstacle", fontsize=10, color=INK)
    axes[0][0].set_ylabel("distance covered [m]")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Behaviour from two cost weights. Labels are passes per episode; "
                 "bars are the seed range.", fontsize=9.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "behaviour.png", dpi=170, bbox_inches="tight")
    print("  wrote behaviour.png")


def fig_ltc_gate():
    """The gate: four arms, distance against crash rate.

    Two measures that must be read together -- an arm that passes more by
    crashing more has not done better -- so they are the two axes rather than
    two bars, and the pass count is the label.
    """
    d = _load("ltc.json")["summary"]
    order = [("global", "one theta"), ("fixed", "fixed schedule"),
             ("mlp", "per-tick MLP"), ("ltc", "LTC (recurrent)")]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, (k, label) in enumerate(order):
        v = d[k]
        ax.errorbar(100 * v["crashes"], v["covered"], yerr=v["sd"], marker="o",
                    markersize=11, capsize=4, linewidth=2.0, color=CAT[i], zorder=3)
        ax.annotate(f"  {label}\n  {v['passes']:.2f} passes",
                    xy=(100 * v["crashes"], v["covered"]), fontsize=8.5,
                    color=CAT[i], va="center", ha="left")
    ax.set_xlabel("crashes [% of episodes]   ->  worse")
    ax.set_ylabel("distance covered [m]   ->  better")
    ax.set_xlim(10, 95)
    ax.grid(alpha=0.22, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("The gate: the hand-written schedule wins.\n"
                 "Both learned arms pass more and crash twice as often; the LTC's "
                 "error bar is five times the schedule's.",
                 fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ltc_gate.png", dpi=170, bbox_inches="tight")
    print("  wrote ltc_gate.png")


FIGS = {"geometry": fig_geometry, "gradient_check": fig_gradient_check,
        "rti": fig_rti, "tracks": fig_tracks, "reversal": fig_reversal,
        "overtake": fig_overtake, "spielberg": fig_spielberg,
        "behaviour": fig_behaviour, "ltc_gate": fig_ltc_gate}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (a.only or sorted(FIGS)):
        FIGS[name]()
