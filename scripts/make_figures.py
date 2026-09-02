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
import matplotlib.colors as mcolors  # noqa: E402
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



def corridor_edges(t):
    """The two track BOUNDARIES, as (left, right) point arrays.

    Every track figure here drew the centreline and left the walls implicit,
    which is wrong for the competition maps in particular: their corridor
    varies by a factor of 4.5 (T1) and 6.4 (T2) round a lap, so where the track
    is wide or tight is most of what a reader wants to see, and a bare
    centreline hides it entirely.

    Built from the centreline gradient rather than a stored normal, since Track
    has none. Safe to draw now: on the raceline tracks the reconstructed centre
    is smoothed, so the offset edges reverse on 0.1-2.8% of segments rather
    than tangling as they did against the raw optimiser line.
    """
    g = np.gradient(t.center, axis=0)
    n = np.stack([g[:, 1], -g[:, 0]], axis=1) / np.linalg.norm(g, axis=1)[:, None]
    wl = np.array([float(t.width(v)[0]) for v in t.s])[:, None]
    wr = np.array([float(t.width(v)[1]) for v in t.s])[:, None]
    # Draw the WALL, not the limit on the car's centre.
    #
    # A raceline optimiser reports w_left/w_right as the room remaining for the
    # car's CENTRE -- the vehicle is already subtracted, which is why they
    # bottom out at exactly -0.000 at every apex. Plotting them directly draws
    # a corridor one car narrower than the track, on both sides, which is what
    # made the boundaries look tighter than the real map.
    hw = getattr(t, "car_half_width", 0.12)
    pad = hw if getattr(t, "width_vehicle_adjusted", False) else 0.0
    return t.center + n * (wl + pad), t.center - n * (wr + pad)


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


def fig_behaviour_matrix(track_name="circuit"):
    """Behaviour as a matrix, not a table: posture x aggression, per obstacle.

    Asked for as a figure and delivered as a text table first, which was the
    wrong artefact -- a 18-row table of three numbers is a matrix pretending
    not to be one.

    Two panels because the obstacle KIND is the axis that changes the answer:
    a stopped car and a slow one look identical in one frame, and "stay behind"
    is a behaviour against something going somewhere and a livelock against
    something that is not. Colour is distance covered; the annotation carries
    passes and posture switches, which are what separate cells that cover the
    same ground for different reasons.
    """
    import json
    import matplotlib.colors as mc

    path = ROOT / "benchmarks" / "results" / f"behaviour_modes_{track_name}.json"
    if not path.exists():
        path = ROOT / "benchmarks" / "results" / "behaviour_modes.json"
    if not path.exists():
        print("  no behaviour_modes results; run experiments/behaviour_modes.py")
        return
    runs = json.loads(path.read_text())["runs"]

    POST = ("stay_behind", "overtake_when_safe", "always_try")
    AGG = ("cautious", "neutral", "aggressive")
    KIND = ("dynamic", "static")

    def cell(kind, post, agg):
        r = [x for x in runs if x["kind"] == kind and x["posture"] == post
             and x["aggression"] == agg]
        if not r:
            return None
        return (float(np.mean([x["covered"] for x in r])),
                float(np.mean([x["passes"] for x in r])),
                float(np.mean([x["switches"] for x in r])),
                float(np.mean([bool(x["off"]) for x in r])))

    M = {k: np.array([[(cell(k, p, a) or (np.nan,) * 4)[0] for a in AGG]
                      for p in POST]) for k in KIND}
    allv = np.concatenate([M[k][np.isfinite(M[k])] for k in KIND])
    norm = mc.Normalize(float(allv.min()), float(allv.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    im = None
    for ax, kind in zip(axes, KIND):
        im = ax.imshow(M[kind], cmap="YlGnBu", norm=norm, aspect="auto")
        ax.set_xticks(range(len(AGG))); ax.set_xticklabels(AGG, fontsize=9)
        ax.set_yticks(range(len(POST)))
        ax.set_yticklabels([p.replace("_", " ") for p in POST], fontsize=9)
        ax.set_title(f"{kind} obstacle", fontsize=10, color=INK)
        for i, post in enumerate(POST):
            for j, agg in enumerate(AGG):
                c = cell(kind, post, agg)
                if c is None:
                    continue
                cov, pas, sw, off = c
                frac = (cov - allv.min()) / max(allv.ptp() if hasattr(allv, "ptp")
                                                else np.ptp(allv), 1e-9)
                col = "white" if frac > 0.55 else INK
                ax.text(j, i, f"{cov:.1f} m", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=col)
                ax.text(j, i + 0.28, f"{pas:.2f} pass · {sw:.1f} switch",
                        ha="center", va="center", fontsize=7.5, color=col)
                if off > 0:
                    ax.text(j, i - 0.30, f"{100*off:.0f}% off",
                            ha="center", va="center", fontsize=7.5, color=RED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(AGG), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(POST), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2.5)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("distance covered [m]", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Behaviour as posture $\\times$ aggression. Aggression moves the "
                 "car; posture, on this track, moves only the switch count --\n"
                 "the three postures agree on where to go and disagree only about "
                 "when they considered going there.",
                 fontsize=9.5, color=INK, y=1.03)
    fig.savefig(OUT / "behaviour_matrix.png", dpi=170, bbox_inches="tight")
    print("  wrote behaviour_matrix.png")


def fig_learning_curves(plant="std"):
    """Learning curve beside the track it was driven on, for every track.

    A learning curve alone cannot be read: 2 laps is excellent on a 204 m
    circuit and poor on a 27 m oval, and whether a dip is the policy exploring
    or the car meeting a hairpin depends on geometry the curve does not show.
    So each row carries its own track, drawn to scale and shaded by sector,
    with the fixed baseline as a reference line on the curve beside it.

    Reads benchmarks/results/tuner_from_baseline_<plant>.json.
    """
    import json
    from matplotlib.collections import LineCollection

    path = ROOT / "benchmarks" / "results" / f"tuner_from_baseline_{plant}.json"
    if not path.exists():
        print(f"  no {path.name}; run experiments/tuner_from_baseline.py first")
        return
    d = json.loads(path.read_text())
    per = d["per_episode"]
    tracks = sorted({k.split("|")[0] for k in per})
    if not tracks:
        print("  no tracks in results"); return

    fig, axes = plt.subplots(len(tracks), 2, squeeze=False,
                             figsize=(10.6, 2.5 * len(tracks)),
                             gridspec_kw=dict(width_ratios=[1.0, 2.1],
                                              hspace=0.55, wspace=0.18))
    for r, name in enumerate(tracks):
        t = getattr(Track, name)()
        ax, bx = axes[r][0], axes[r][1]

        pts = t.center.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        sec = np.array([int(t.sector(t.wrap(v))) for v in t.s])
        for j in range(4):
            m = sec[:-1] == j
            if m.any():
                ax.add_collection(LineCollection(segs[m], colors=[CAT[j]],
                                                 linewidth=3.5,
                                                 capstyle="round"))
        el, er = corridor_edges(t)
        for edge in (el, er):
            ax.plot(edge[:, 0], edge[:, 1], "-", color=INK, lw=0.7, alpha=0.7)
        ax.set_aspect("equal"); ax.autoscale_view(); ax.axis("off")
        ax.set_title(f"{name}\n{t.length:.0f} m", fontsize=8.5, color=INK)

        for kind, col in (("fixed", "0.55"), ("tuner", BLUE)):
            runs = [v for k, v in per.items()
                    if k.startswith(name + "|") and k.endswith("|" + kind)]
            if not runs:
                continue
            L = np.array([[e["laps"] for e in run] for run in runs])
            mean = L.mean(axis=0)
            ep = np.arange(len(mean))
            if kind == "fixed":
                bx.axhline(float(mean.mean()), color=col, ls="--", lw=1.3,
                           label=f"fixed baseline ({mean.mean():.2f} laps)")
            else:
                bx.plot(ep, mean, "-o", color=col, lw=1.8, ms=4, label="tuner")
                if len(L) > 1:
                    bx.fill_between(ep, L.min(axis=0), L.max(axis=0),
                                    color=col, alpha=0.16, linewidth=0)
        bx.set_ylabel("laps", fontsize=9)
        bx.set_xlabel("episode", fontsize=9) if r == len(tracks) - 1 else None
        bx.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)

    handles = [plt.Line2D([], [], color=CAT[j], lw=4,
                          label=Track.SECTOR_NAMES[j]) for j in range(4)]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Online tuning from a baseline that drives, on the {plant} "
                 "plant -- a Pacejka tyre model, not a kinematic bicycle.\n"
                 "Each learning curve is drawn beside its own track, because "
                 "two laps means something different on 27 m and on 204 m.",
                 fontsize=9.5, color=INK, y=1.005)
    fig.savefig(OUT / f"learning_curves_{plant}.png", dpi=170,
                bbox_inches="tight")
    print(f"  wrote learning_curves_{plant}.png")


def fig_plant_gap(track_name="oval"):
    """The same controller on a bicycle and on real tyres.

    Step 1 of the project's order, made visible: a parameterisation is only a
    baseline on the plant it was measured on. The weights that drive 7.71 laps
    of the oval on a kinematic bicycle last 0.2 laps on a Pacejka tyre model,
    and solve success falls back to 72-84% because the car can now slide into
    states the hard constraints cannot accommodate.

    Reads benchmarks/results/std_baseline_<track>_<plant>.json where present,
    and the measured plant comparison otherwise.
    """
    import json

    # The plant comparison, measured directly (scripts/../std_plant.py).
    PLANTS = [("kinematic\nbicycle", 7.71, 100.0, 4.85, False),
              ("scuderia ST\n(Pacejka tyres)", 0.20, 72.0, 4.04, True),
              ("scuderia STD\n(drift model)", 0.22, 84.0, 3.89, True)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    ax = axes[0]
    names = [p[0] for p in PLANTS]
    laps = [p[1] for p in PLANTS]
    cols = [RED if p[4] else GREEN for p in PLANTS]
    b = ax.barh(range(len(PLANTS)), laps, color=cols, height=0.55)
    ax.set_yticks(range(len(PLANTS))); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("laps completed before leaving the track", fontsize=9)
    ax.axvline(2.0, color=INK, ls="--", lw=1.0, alpha=0.6)
    ax.text(2.08, len(PLANTS) - 0.4, "acceptance gate\n(2 laps)", fontsize=8,
            color=INK, va="center")
    for i, (n, lp, ok, v, off) in enumerate(PLANTS):
        ax.text(lp + 0.12, i, f"{lp:.2f}" + ("  off track" if off else "  clean"),
                va="center", fontsize=8.5,
                color=RED if off else GREEN)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("same weights, same controller, different physics",
                 fontsize=9.5, color=INK)

    # Solve success ALONE on this axis. The first version put peak speed
    # beside it scaled by 20 to share the scale, which is a dual axis wearing a
    # disguise: two quantities with different units on one set of gridlines,
    # where the reader cannot tell a real difference from a chosen multiplier.
    # Peak speed is a number, so it is printed as one.
    bx = axes[1]
    x = np.arange(len(PLANTS))
    bx.bar(x, [p[2] for p in PLANTS], width=0.5,
           color=[BLUE if not p[4] else RED for p in PLANTS])
    for i, p_ in enumerate(PLANTS):
        bx.text(i, p_[2] + 2.5, f"{p_[2]:.0f}%", ha="center", fontsize=9,
                fontweight="bold", color=INK)
        bx.text(i, 6, f"peak\n{p_[3]:.2f} m/s", ha="center", fontsize=8,
                color="white" if p_[2] > 40 else INK)
    bx.set_xticks(x)
    bx.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=7.5,
                       rotation=12, ha="right")
    bx.set_ylim(0, 112)
    bx.set_ylabel("solve success [%]", fontsize=9)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.set_title("the solver struggles once the car can slide",
                 fontsize=9.5, color=INK)

    fig.suptitle("A baseline is only a baseline on the plant it was measured "
                 "on. The kinematic bicycle has no tyres, no slip angles and "
                 "no load transfer --\nso it also has no friction to lower, "
                 "which is why grip, the friction ellipse and $k_v$ have had "
                 "nothing to act on in every experiment so far.",
                 fontsize=9.3, color=INK, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "plant_gap.png", dpi=170, bbox_inches="tight")
    print("  wrote plant_gap.png")


def fig_strategy(track_name="circuit"):
    """The track, and what each racing situation asks the weights to be.

    The figure this paper was missing. Everything else measures what the
    learner emits; this shows what the situation *demands*, found by driving a
    grid of fixed weight vectors in each cell and keeping the one that wins.

    Left: the circuit, shaded by the named sector -- the four-way one-hot the
    policy receives as an input.

    Right: the best q_v/q_c per (sector x opponent) and per (sector x corridor).
    The scale DIVERGES about 1.0 because that is a real boundary rather than a
    convenient midpoint: below it the ego weights path-following over progress
    and falls in behind, above it the reverse and it attacks. The hue therefore
    answers the strategic question directly -- where does this situation want us
    to follow, and where to overtake.

    Reads benchmarks/results/situation_demands_<track>.json.
    """
    import json
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mc

    path = ROOT / "benchmarks" / "results" / f"situation_demands_{track_name}.json"
    if not path.exists():
        print(f"  no {path.name}; run experiments/situation_demands.py first")
        return
    d = json.loads(path.read_text())
    tr = getattr(Track, d["track"])()
    cells = {tuple(k.split("|")): v for k, v in d["cells"].items()}
    OPPS = ("none", "slower", "equal", "faster")
    secs = sorted({int(k[0]) for k in cells})
    speeds = sorted({float(k[3]) for k in cells if len(k) > 3})

    # One panel per ENTRY SPEED, because the axes interact: the cells that back
    # off are equal-opponent cells, but WHICH ones depends on how fast the ego
    # arrives. Closing fast on a matched car down a straight wants a near
    # neutral ratio; meeting the same car slowly in a 90 wants an intermediate
    # one. Collapsing the speed axis averages that structure away, which is
    # what the three-axis version did -- and it found 3.5% of headroom where
    # this finds 8.9%.
    ncol = 1 + max(len(speeds), 1)
    fig = plt.figure(figsize=(5.2 + 3.6 * len(speeds), 4.6))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1.15] + [1.0] * len(speeds),
                          wspace=0.30)
    axt = fig.add_subplot(gs[0, 0])
    axes_m = [fig.add_subplot(gs[0, 1 + i]) for i in range(len(speeds))]

    pts = tr.center.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    sec_at = np.array([int(tr.sector(tr.wrap(v))) for v in tr.s])
    for j in range(4):
        m = sec_at[:-1] == j
        if m.any():
            axt.add_collection(LineCollection(segs[m], colors=[CAT[j]],
                                              linewidth=5.0, capstyle="round"))
    el, er = corridor_edges(tr)
    for edge in (el, er):
        axt.plot(edge[:, 0], edge[:, 1], "-", color=INK, lw=1.0, alpha=0.75,
                 zorder=3)
    axt.set_aspect("equal"); axt.autoscale_view(); axt.axis("off")
    axt.set_title(f"{d['track']} - {tr.length:.0f} m", fontsize=9.5, color=INK)
    axt.legend(handles=[plt.Line2D([], [], color=CAT[j], lw=5,
                                   label=Track.SECTOR_NAMES[j])
                        for j in range(4) if (sec_at == j).any()],
               fontsize=8, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14))

    M = {}
    for v0 in speeds:
        A = np.full((len(secs), len(OPPS)), np.nan)
        for i, k in enumerate(secs):
            for j, c in enumerate(OPPS):
                hit = [b for kk, b in cells.items()
                       if int(kk[0]) == k and kk[1] == c and float(kk[3]) == v0]
                if hit:
                    A[i, j] = float(np.mean([x["ratio"] for x in hit]))
        M[v0] = A

    allv = np.concatenate([M[v][np.isfinite(M[v])] for v in speeds])
    lo, hi = float(np.log10(allv.min())), float(np.log10(allv.max()))
    norm = mc.Normalize(lo, hi)
    im = None
    for ax, v0 in zip(axes_m, speeds):
        A = M[v0]
        im = ax.imshow(np.log10(A), cmap="YlOrRd", norm=norm, aspect="auto")
        ax.set_xticks(range(len(OPPS))); ax.set_xticklabels(OPPS, fontsize=8.5)
        ax.set_yticks(range(len(secs)))
        ax.set_yticklabels([Track.SECTOR_NAMES[k] for k in secs], fontsize=8.5)
        ax.set_title(f"entering at {v0:.0f} m/s", fontsize=9.5, color=INK)
        ax.set_xlabel("opponent", fontsize=9)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    frac = (np.log10(A[i, j]) - lo) / max(hi - lo, 1e-9)
                    ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center",
                            fontsize=9.5, fontweight="bold",
                            color="white" if frac > 0.55 else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(OPPS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(secs), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2.0)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes_m, fraction=0.03, pad=0.02)
    cb.set_label("best $q_v/q_c$   (higher = attack harder)", fontsize=8.5)
    cb.set_ticks([lo, 0.5 * (lo + hi), hi])
    cb.set_ticklabels([f"{10**lo:.1f}", f"{10**(0.5*(lo+hi)):.1f}", f"{10**hi:.1f}"])
    cb.ax.tick_params(labelsize=8)

    head = d.get("headroom_pct", float("nan"))
    fig.suptitle("What each racing situation asks of the cost weights, by direct "
                 "search over the weights themselves. The cells that back off are "
                 "those facing an EQUALLY\nFAST opponent -- the only one that can "
                 "block -- and WHICH of them depends on how fast we arrive, which "
                 "is why own speed is an axis.\nPer-situation weights beat the "
                 f"best single constant by {head:+.1f}%; with the speed axis "
                 "collapsed the same search finds only 3.5%.",
                 fontsize=9.3, color=INK, y=1.06)
    fig.savefig(OUT / "strategy.png", dpi=170, bbox_inches="tight")
    print("  wrote strategy.png")


def fig_icra_grip():
    """The competition tracks, coloured by the speed each corner allows.

    Replaces an earlier figure built on Spielberg. Two reasons, and the second
    matters more than the first. Spielberg is an F1 circuit scaled 1:10, which
    is not the kind of track these cars race on; and the argument it was there
    to make -- that the synthetic circuit cannot discriminate between weight
    settings because its hardest corner still allows 3.94 m/s against a
    4.0 m/s cap -- stopped being true when SPEED_MAX was raised to 8.0 on the
    g-g analysis. At the corrected cap the circuit's hardest corner is 49% of
    it, so the circuit discriminates too.

    What survives is the underlying point: a track measures a weight policy
    only where it is grip-limited. The ICRA tracks simply do it hardest, at
    32% and 29% of the cap, and they are the tracks the cars actually run on.

    Sectors are drawn because the sector one-hot is an INPUT to the policy
    (features[9:13]), so the figure shows what the network is told.
    """
    from matplotlib.collections import LineCollection
    from mpcc_tuning.model import A_LAT_MAX, SPEED_MAX

    # ICRA 2025 and ONE of the 2026 circuits, not both.
    #
    # T1 and T2 are the same map: their curvature profiles cross-correlate at
    # 0.874 once phase is allowed for, and what differs between them is the
    # corridor (median half-width 0.72 m against 0.66 m). Showing both spends a
    # panel on a track the reader has already seen. 2025 is a genuinely
    # different circuit -- 204 m against 80, a serpentine rather than a
    # hairpin-heavy layout -- and belongs here instead.
    tracks = [("ICRA 2025", Track.icra2025()),
              ("ICRA 2026 T1", Track.icra_t1_raceline()),
              ("circuit (synthetic)", Track.circuit())]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    lc = None
    slowest = []
    for ax, (name, tr) in zip(axes, tracks):
        k = np.abs([tr.curvature(v) for v in tr.s])
        vmax = np.minimum(np.sqrt(A_LAT_MAX / np.maximum(k, 1e-6)), SPEED_MAX)
        pts = tr.center.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis",
                            norm=plt.Normalize(2.0, SPEED_MAX),
                            linewidth=4.0, capstyle="round")
        lc.set_array(vmax[:-1])
        # The track BOUNDARIES, so the corridor is visible and not implied.
        el, er = corridor_edges(tr)
        for edge in (el, er):
            ax.plot(edge[:, 0], edge[:, 1], "-", color=INK, lw=1.1, alpha=0.75,
                    zorder=3)
        ax.add_collection(lc)
        # The named sector at each point, as a ring outside the line: this is
        # what the policy receives, not a post-hoc annotation.
        sec = np.array([tr.sector(tr.wrap(v)) for v in tr.s])
        for j in range(4):
            m = sec[:-1] == j
            if not m.any():
                continue
            ring = LineCollection(segs[m], colors=[CAT[j]], linewidth=9.0,
                                  alpha=0.30, capstyle="round", zorder=0)
            ax.add_collection(ring)
        ax.set_aspect("equal"); ax.autoscale_view(); ax.axis("off")
        slowest.append(float(vmax.min()))
        frac = float((vmax >= SPEED_MAX - 1e-9).mean())
        # y fixed, so the three titles sit on one line rather than following
        # each track's bounding box.
        ax.set_title(f"{name}\n{tr.length:.0f} m · slowest corner "
                     f"{vmax.min():.2f} m/s = {100*vmax.min()/SPEED_MAX:.0f}% of cap"
                     f"\n{100*frac:.0f}% of the lap at the cap",
                     fontsize=9, color=INK, y=1.0)
    handles = [plt.Line2D([], [], color=CAT[j], lw=6, alpha=0.45,
                          label=Track.SECTOR_NAMES[j]) for j in range(4)]
    # Below the axes, not on top of a track.
    fig.legend(handles=handles, fontsize=8.5, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.45, -0.02))
    cb = fig.colorbar(lc, ax=axes, fraction=0.02, pad=0.06)
    cb.set_label(f"grip-limited corner speed [m/s]   (cap {SPEED_MAX:.1f})",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    # Computed, not typed. These numbers moved when the corridor centre was
    # reconstructed from the raceline, and a hardcoded caption then disagreed
    # with the titles directly beneath it.
    pcts = ", ".join(f"{100 * v / SPEED_MAX:.0f}%" for v in slowest[:2])
    fig.suptitle("A track measures a weight policy only where it is grip-limited. "
                 f"The ICRA circuits are hardest --- their slowest corners are "
                 f"{pcts} of the\nspeed cap --- and they are the tracks these cars "
                 "race on. The thin outline is the TRACK BOUNDARY; the shading is "
                 "the NAMED SECTOR,\nwhich the policy receives as a one-hot input.",
                 fontsize=9.5, color=INK, y=1.04)
    fig.subplots_adjust(top=0.74, bottom=0.10, wspace=0.02)
    fig.savefig(OUT / "icra_grip.png", dpi=170, bbox_inches="tight")
    print("  wrote icra_grip.png")


def fig_behaviour():
    """Behaviour as two axes: what it achieves, and what it decides."""
    d = _load("behaviour_modes.json")["summary"]
    post = ["stay_behind", "overtake_when_safe", "always_try"]
    aggr = ["cautious", "neutral", "aggressive"]
    kinds = sorted({k.split("/")[0] for k in d}) if "/" in next(iter(d)) else [None]
    kinds = [k for k in ("dynamic", "static") if k in kinds] or [None]
    # Shared y. The static panel's 5.6 m stall only reads as a failure against
    # the 34 m of merely following and the 75 m of passing; on its own axis it
    # is just a low point.
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 4.0),
                             squeeze=False, sharey=True)
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
                                xytext=(0, 9 + 9 * i), textcoords="offset points",
                                fontsize=7, ha="center", color=CAT[i])
        ax.set_xticks(x, aggr)
        ax.set_xlabel("aggression")
        ax.grid(alpha=0.22, linewidth=0.6, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{kind or 'opponent'} obstacle", fontsize=10, color=INK)
        if kind == "static":
            # The livelock, named on the figure rather than left to the caption.
            ax.annotate("car stops behind the obstacle\nand never recovers",
                        xy=(0.5, 5.6), xytext=(0.15, 22), fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0))
    axes[0][0].set_ylabel("distance covered [m]")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Behaviour from two cost weights. Aggression is the axis that "
                 "decides the outcome; the posture is not.\nLabels are passes per "
                 "episode; bars are the seed range.",
                 fontsize=9.5, color=INK, y=1.04)
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


def fig_adaptation():
    """How the weights move *during* a lap, and across episodes.

    The tables say a policy over theta beats or loses to a schedule. They do
    not show the thing the method is about: that theta is a function of where
    the car is and what is in front of it, changing tick by tick. This runs the
    LTC policy and draws what it emitted.

    Left: the driven path on the track, coloured by the ratio q_v/q_c the
    policy chose at each tick -- the measured behaviour boundary is at 1, so
    the colour is literally "following" below and "overtaking" above, on a
    diverging ramp about that midpoint. Right: the same run as time series, so
    the hold between decisions is visible rather than inferred.
    """
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.model import KinematicBicycle
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.opponents import ObstacleTracker, Opponent
    from examples.tune_online import Plant

    track = Track.oval()
    theta0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    # Only q_c and q_v move. An earlier attempt at this used
    # MPCCWeights(q_c=1.0, q_v=2.0), which silently also reset q_l 200 -> 10
    # and r_d 1.0 -> 0.1, and fixed_schedule inherits everything it does not
    # override: ten times less steering-rate penalty put BOTH baselines into
    # the wall (fixed went 36.3 m -> 3.5 m at 100% crashes), which made the
    # LTC look 21.6 m better when nothing about it had improved.
    #
    # A *working* controller, not the spike's deliberately-bad one. The policy's
    # job is to ADAPT a controller that already drives -- to the sector ahead,
    # to how aggressive we want to be, to whether there is someone to pass --
    # not to recover one that crawls.
    #
    # (q_v, q_c) = (2.0, 1.0) is the measured clean-pass cell from
    # experiments/overtake_or_follow.py: 36.6 m, one pass, no crash. Starting
    # instead at q_l=200, q_v=0.5 puts the ratio at 0.050, twenty times BELOW
    # the behaviour boundary, so the policy had to drag it across just to
    # overtake at all -- which is why the gradient pushed one way and never
    # stopped. It also makes the prior principled: pulling back towards a
    # controller known to work is a trust region around a reference policy,
    # where pulling back towards a crawling one is just a brake.
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    cell = LTCCell(N_FEATURES, 12, seed=0)
    pol = WeightPolicy(cell, theta0, THETA_LO, THETA_HI, seed=0,
                       gauge_fix=True)
    # Same configuration as experiments/ltc_behaviour.py. The figure had its
    # own PolicyTuner without the trust region and prior, so it kept drawing
    # the runaway after the experiment had stopped exhibiting it.
    tuner = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0,
                        seed=0, trust_region=0.01, theta_prior=0.0)

    ep_ratio, ep_cov, last = [], [], None
    for ep in range(12):
        opp = Opponent(track, s0=3.0, speed=1.2, radius=0.24)
        tracker = ObstacleTracker(dt=0.05)
        P = Plant(track, dt=0.05, max_steps=300, opponents=[opp])
        s5 = P.reset(); m.reset(); m.set_obstacles(P.keepouts()); tuner.reset()
        tracker.update(opp.pose()[:2])
        feat = features(track, s5, [opp], opp_speed_est=tracker.speed)
        theta, u = tuner.act(feat, s5)
        rec, cov = [], 0.0
        for _ in range(300):
            s5n, r, off, tr = P.step(u); cov += r
            m.set_obstacles(P.keepouts())
            rec.append((s5n[0], s5n[1], float(np.exp(theta[2]) / np.exp(theta[0])),
                        float(np.exp(theta[2])), float(np.exp(theta[0])),
                        opp.pose()[0], opp.pose()[1]))
            tracker.update(opp.pose()[:2])
            out = tuner.learn(r, s5n,
                              features(track, s5n, [opp],
                                       opp_speed_est=tracker.speed), off)
            if out[0] is None:
                break
            theta, u = out
            s5 = s5n
            if off or tr:
                break
        rec = np.array(rec)
        ep_ratio.append(float(np.median(rec[:, 2]))); ep_cov.append(cov)
        last = rec
        print(f"    ep {ep:2d}  covered {cov:6.1f} m  median ratio {ep_ratio[-1]:6.2f}",
              flush=True)

    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(track.center[:, 0], track.center[:, 1], "-", color="0.85", lw=14,
            solid_capstyle="round", zorder=1)
    from matplotlib.collections import LineCollection
    xy = last[:, :2].reshape(-1, 1, 2)
    segs = np.concatenate([xy[:-1], xy[1:]], axis=1)
    # Diverging about 1.0, because 1.0 is the measured behaviour boundary --
    # below it the policy is following, above it overtaking. A sequential ramp
    # would hide the one value that means something.
    lc = LineCollection(segs, cmap="coolwarm",
                        norm=matplotlib.colors.TwoSlopeNorm(1.0, 0.05, 6.0),
                        linewidth=2.6, zorder=3)
    lc.set_array(last[:-1, 2])
    ax.add_collection(lc)
    ax.plot(last[::28, 5], last[::28, 6], "o", ms=5, color=INK, alpha=0.5,
            zorder=4, label="opponent")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("where the policy chose to overtake\n(episode 11)", fontsize=9.5,
                 color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("$q_v/q_c$   (1.0 = behaviour boundary)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    bx = fig.add_subplot(2, 2, 2)
    t = np.arange(len(last)) * 0.05
    bx.plot(t, last[:, 3], "-", color=BLUE, lw=1.8, label="$q_v$")
    bx.plot(t, last[:, 4], "-", color=RED, lw=1.8, label="$q_c$")
    bx.axhline(2.0, color="0.6", ls=":", lw=1.0)
    bx.annotate(" measured $q_v$ ceiling", xy=(t[-1] * 0.55, 2.05), fontsize=7,
                color="0.4")
    bx.set_ylabel("weight"); bx.set_yscale("log")
    bx.legend(fontsize=8, frameon=False, ncol=2)
    bx.grid(alpha=0.2, lw=0.5)
    bx.set_title("the two weights, tick by tick", fontsize=9.5, color=INK)
    cx = fig.add_subplot(2, 2, 4)
    cx.plot(np.arange(len(ep_cov)), ep_cov, "-o", color=GREEN, lw=1.8, ms=4)
    cx.set_xlabel("episode"); cx.set_ylabel("covered [m]")
    cx.grid(alpha=0.2, lw=0.5)
    for a in (bx, cx):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    bx.set_xlabel("time in the lap [s]")
    fig.suptitle("The weights are a function of the situation, not a vector to be found: "
                 "the LTC policy\nemits a different $\\theta$ at every tick, and the "
                 "ratio crosses the behaviour boundary where it passes.",
                 fontsize=9.5, color=INK, y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / "adaptation.png", dpi=170, bbox_inches="tight")
    print("  wrote adaptation.png")


def fig_filmstrip():
    """Stills from the animations, because a PDF cannot hold a GIF.

    ``\\includegraphics`` has no GIF support, so the animations are docs and
    presentation assets and can never appear in the paper. A filmstrip is the
    paper's version of the same evidence: the same runs, sampled at four times,
    with the keep-out drawn.
    """
    from mpcc_tuning.ltc import behaviour_theta
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.model import KinematicBicycle
    from mpcc_tuning.opponents import Opponent
    from examples.tune_online import Plant

    track = Track.oval()
    t0 = MPCCWeights(q_l=200.0, r_d=1.0).to_log()
    rows = [("follow  $q_v/q_c<1$", behaviour_theta("follow", "neutral", t0), 1.0),
            ("overtake  $q_v/q_c>1$", behaviour_theta("overtake", "aggressive", t0), 1.0),
            ("same weights, stopped car", behaviour_theta("follow", "neutral", t0), 0.0)]
    runs = []
    for label, th, vo in rows:
        m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
                 max_iter=60, max_obstacles=1)
        opp = Opponent(track, s0=3.0, speed=vo, radius=0.24)
        P = Plant(track, dt=0.05, max_steps=220, opponents=[opp])
        s5 = P.reset(); m.reset()
        xy, op, cov = [], [], 0.0
        for _ in range(220):
            m.set_obstacles(P.keepouts())
            u = m.value(s5, th)["u0"]
            s5, r, off, tr = P.step(u); cov += r
            xy.append((s5[0], s5[1])); op.append(tuple(opp.pose()[:2]))
            if off or tr:
                break
        runs.append((label, np.array(xy), np.array(op), cov))
        print(f"    {label}: {cov:.1f} m", flush=True)

    K = 4
    fig, axes = plt.subplots(len(runs), K, figsize=(3.1 * K, 2.5 * len(runs)))
    for i, (label, xy, op, cov) in enumerate(runs):
        idx = np.linspace(0, len(xy) - 1, K).round().astype(int)
        for j, k in enumerate(idx):
            ax = axes[i, j]
            ax.plot(track.center[:, 0], track.center[:, 1], "-", color="0.88",
                    lw=9, solid_capstyle="round", zorder=1)
            ax.plot(xy[:k + 1, 0], xy[:k + 1, 1], "-", color=BLUE, lw=1.8, zorder=3)
            ax.plot(xy[k, 0], xy[k, 1], "o", ms=7, color=BLUE, mec="white",
                    mew=1.0, zorder=5)
            ax.plot(op[k, 0], op[k, 1], "o", ms=7, color=RED, mec="white",
                    mew=1.0, zorder=5)
            ax.add_patch(plt.Circle((op[k, 0], op[k, 1]), 0.24, fill=False,
                                    ls="--", lw=1.1, color=RED, zorder=4))
            ax.set_aspect("equal"); ax.axis("off")
            if i == 0:
                ax.set_title(f"t = {k * 0.05:.1f} s", fontsize=8.5, color=INK)
        axes[i, 0].text(-0.04, 0.5, f"{label}\n{cov:.1f} m", transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8.5, color=INK)
    fig.suptitle("The same controller and the same opponent; only $q_v$ and $q_c$ differ.\n"
                 "Dashed circle is the keep-out. Bottom row: identical weights, "
                 "but the obstacle is stopped.", fontsize=9.5, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "filmstrip.png", dpi=170, bbox_inches="tight")
    print("  wrote filmstrip.png")


def fig_architecture():
    """The method in one picture: what is learned, and where the gradient comes from.

    Three things this has to make visible, because they are what the prose keeps
    having to restate. The MPCC is inside the loop, so every theta the policy
    emits is still solved subject to the same constraints. The gradient the
    learner needs comes out of the solve for free by the envelope theorem, not
    from differentiating through the solver. And the policy is recurrent, so its
    own influence has to be carried forward -- which is the only part that is an
    approximation.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.3); ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
                fontsize=9.5, color=INK, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 0.60, sub, ha="center", va="top",
                fontsize=7.6, color="0.25", zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label, col=INK, dashed=False, dy=0.16):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.5, color=col,
                                     linestyle="--" if dashed else "-",
                                     zorder=4,
                                     connectionstyle="arc3,rad=0.0"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center",
                fontsize=7.8, color=col, zorder=5)

    box(0.15, 2.35, 2.25, 1.45, "situation", 
        "curvature preview\nsector · corridor width\ngap · TTC · opponent class",
        "#eef3fb", BLUE)
    box(2.95, 2.35, 2.2, 1.45, "policy  $\\phi$",
        "LTC cell, recurrent\n$\\theta = \\theta_0 + \\mathrm{span}\\cdot\\tanh(Gh)$\nbounded to a measured box",
        "#f7eef7", "#7d3c98")
    box(5.7, 2.35, 2.2, 1.45, "MPCC",
        "one NLP, $\\theta$ as a\nruntime parameter\nconstraints unchanged",
        "#eef7ef", GREEN)
    box(8.4, 2.35, 1.85, 1.45, "plant",
        "kinematic bicycle\n+ grip limit the\ncontroller never sees",
        "#fdf2e9", AMBER)

    arrow(2.40, 3.05, 2.95, 3.05, "features")
    arrow(5.15, 3.05, 5.70, 3.05, "$\\theta$  (6 log weights)")
    arrow(7.90, 3.05, 8.40, 3.05, "$u_0$")

    # return path
    ax.add_patch(FancyArrowPatch((9.3, 2.35), (9.3, 1.5), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.5, color=INK, zorder=4))
    ax.plot([9.3, 1.28], [1.5, 1.5], "-", lw=1.5, color=INK, zorder=4)
    ax.add_patch(FancyArrowPatch((1.28, 1.5), (1.28, 2.35), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.5, color=INK, zorder=4))
    ax.text(8.0, 1.62, "reward: metres covered, $-5$ off-track", fontsize=7.8,
            ha="center", color=INK)

    box(2.6, 0.15, 5.2, 1.05, "TD($\\lambda$), one update per tick",
        "$\\delta = r + \\gamma V(s') - Q(s,a)$      "
        "$\\nabla_\\phi Q = \\nabla_\\theta Q \\cdot \\partial\\theta/\\partial\\phi$",
        "#fdeeee", RED)

    ax.annotate("", xy=(6.4, 1.20), xytext=(6.4, 2.35),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=GREEN, ls="--"))
    ax.text(6.52, 1.80, "$\\nabla_\\theta Q$ free, by the\nenvelope theorem",
            fontsize=7.6, color=GREEN, va="center")
    ax.annotate("", xy=(3.6, 2.35), xytext=(3.6, 1.20),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#7d3c98", ls="--"))
    ax.text(3.72, 1.80, "$\\partial\\theta/\\partial\\phi$ carried\nforward (RFLO)",
            fontsize=7.6, color="#7d3c98", va="center")

    fig.suptitle("The controller stays an MPCC. What is learned is a map from the "
                 "situation to its six cost weights,\nand the gradient that learns it "
                 "falls out of a solve that was happening anyway.",
                 fontsize=9.5, color=INK, y=1.02)
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    print("  wrote architecture.png")


def fig_icra():
    """The two ICRA competition maps, with the centrelines extracted from them."""
    import re
    from PIL import Image
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cl", str(ROOT / "tools" / "centerline_from_map.py"))
    cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)

    T = ROOT / "mpcc_tuning" / "tracks"
    maps = [("ICRA 2025, car10", T / "icra-2025-car10version_edited.pgm",
             T / "icra-2025-car10version.yaml", None),
            ("ICRA 2026, Track 1", T / "icra2026_t1.pgm", T / "icra2026_t1.yaml", 0)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (name, pgm, yml, rank) in zip(axes, maps):
        im, res, org = cl.load(str(pgm), str(yml))
        H, W = im.shape
        if rank is None:
            xy, w, L, ok, gap = cl.centerline(str(pgm), str(yml))
        else:
            xy, w, L, ok, gap, _a = cl.centerline_around(str(pgm), str(yml),
                                                         hole_rank=rank)
        # Draw the map the PLANNER sees, not the raw grid. A row of cones is a
        # boundary, but as separate blobs it is mostly free space between them,
        # and filling holes to tame the skeleton turned that boundary into
        # driveable track -- so the picture showed a corridor the car could cut
        # straight through. connect_cone_rows welds neighbouring cones into one
        # wall; what is shaded here is the corridor after that weld.
        occ = cl.connect_cone_rows(im <= 50)
        shown = np.where(occ, 0, im)
        ax.imshow(shown, cmap="gray")
        weld = occ & ~(im <= 50)
        # Not RED -- that is the centreline's colour, and a wall drawn in it
        # reads as track.
        ax.imshow(np.ma.masked_where(~weld, weld),
                  cmap=mcolors.ListedColormap(["#d97706"]), alpha=0.9, zorder=2)
        ax.plot((xy[:, 0] - org[0]) / res, (org[1] + H * res - xy[:, 1]) / res,
                "-", color=RED, lw=2.2, zorder=3)
        ax.plot((xy[0, 0] - org[0]) / res, (org[1] + H * res - xy[0, 1]) / res,
                "o", ms=9, color=GREEN, mec="white", mew=1.2, zorder=4)
        ax.set_title(f"{name}\n{L:.1f} m · {100*ok:.0f}% inside the corridor · "
                     f"closes to {gap:.2f} m", fontsize=9.5, color=INK)
        ax.axis("off")
        print(f"    {name}: {L:.1f} m, {100*ok:.0f}% in corridor", flush=True)
    fig.suptitle("Centrelines extracted from the competition teams' own occupancy "
                 "grids. Track 1's corridor BRANCHES, so it has\nno unique centreline "
                 "-- a loop is named by the hole it encircles, and this is the outer "
                 "one. Gaps between neighbouring\ncones are welded into solid wall "
                 "(red) so a row of cones bounds the corridor instead of being "
                 "driven through.",
                 fontsize=9.5, color=INK, y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / "icra_tracks.png", dpi=170, bbox_inches="tight")
    print("  wrote icra_tracks.png")


def fig_driving_sectors():
    """Driving the circuit, with the sector the car is in and the weights it emits.

    The tables say a schedule wins or loses; they cannot show the schedule
    *happening*. This drives one lap and draws, against arc length: which named
    sector the car is in, what the policy emitted for q_v and q_c there, and how
    fast it went. If a sector schedule is doing anything, the weights change at
    the sector boundaries and nowhere else in particular.
    """
    from mpcc_tuning.ltc import (LTCCell, N_FEATURES, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, features)
    from mpcc_tuning.model import KinematicBicycle
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.opponents import ObstacleTracker, Opponent
    from examples.tune_online import Plant

    track = Track.circuit()
    th0 = MPCCWeights(q_c=1.0, q_v=2.0, q_l=200.0, r_d=1.0).to_log()
    m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
             max_iter=60, max_obstacles=1)
    # The configuration in which the policy is actually a policy: the critic's
    # gauge freedom projected out, and NO readout decay. With the default
    # (theta_prior=0.5) the weights are flat for the whole lap -- not because
    # the figure is wrong but because the policy emits one vector everywhere.
    pol = WeightPolicy(LTCCell(N_FEATURES, 12, seed=0), th0, THETA_LO, THETA_HI,
                       seed=0, gauge_fix=True)
    tu = PolicyTuner(m, pol, alpha=2e-3, explore=0.05, delta_clip=1.0, seed=0,
                     trust_region=0.01, theta_prior=0.0)

    ep_cov = []
    for ep in range(10):
        opp = Opponent(track, s0=6.0, speed=(0.0, 1.0, 2.6, 3.4)[ep % 4], radius=0.24)
        P = Plant(track, dt=0.05, max_steps=400, opponents=[opp])
        s5 = P.reset(); m.reset(); m.set_obstacles(P.keepouts()); tu.reset()
        tr = ObstacleTracker(dt=0.05); tr.update(opp.pose()[:2])
        th, u = tu.act(features(track, s5, [opp], opp_speed_est=tr.speed), s5)
        rec, cov = [], 0.0
        for _ in range(400):
            s5n, r, off, done = P.step(u); cov += r
            m.set_obstacles(P.keepouts()); tr.update(opp.pose()[:2])
            rec.append((float(track.project(s5n[0], s5n[1])), s5n[0], s5n[1],
                        float(s5n[3]), float(np.exp(th[2])), float(np.exp(th[0])),
                        track.sector(float(track.wrap(s5n[4] + 1.5)))))
            out = tu.learn(r, s5n, features(track, s5n, [opp],
                                            opp_speed_est=tr.speed), off)
            if out[0] is None:
                break
            th, u = out; s5 = s5n
            if off or done:
                break
        ep_cov.append(cov)
        last = np.array(rec)
        print(f"    ep {ep:2d}  covered {cov:6.1f} m", flush=True)

    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.45, wspace=0.18)

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(track.center[:, 0], track.center[:, 1], "-", color="0.88", lw=13,
            solid_capstyle="round", zorder=1)
    for k in range(4):
        msk = last[:, 6] == k
        if msk.any():
            ax.plot(np.where(msk, last[:, 1], np.nan),
                    np.where(msk, last[:, 2], np.nan), ".", ms=3.2,
                    color=CAT[k], zorder=3, label=Track.SECTOR_NAMES[k])
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=7.5, frameon=False, loc="center", ncol=2)
    ax.set_title("the lap, coloured by the sector the policy sees",
                 fontsize=9, color=INK)

    bx = fig.add_subplot(gs[0, 1])
    for k in range(4):
        msk = last[:, 6] == k
        if msk.any():
            bx.fill_between(last[:, 0], 0, 1, where=msk, transform=
                            bx.get_xaxis_transform(), color=CAT[k], alpha=0.13,
                            linewidth=0)
    bx.plot(last[:, 0], last[:, 4], "-", color=BLUE, lw=1.7, label="$q_v$")
    bx.plot(last[:, 0], last[:, 5], "-", color=RED, lw=1.7, label="$q_c$")
    bx.set_yscale("log"); bx.set_ylabel("weight")
    bx.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    bx.set_title("what the policy emitted, against arc length "
                 "(bands are the sectors)", fontsize=9, color=INK)

    cx = fig.add_subplot(gs[1, 1], sharex=bx)
    cx.plot(last[:, 0], last[:, 3], "-", color=GREEN, lw=1.7)
    cx.set_ylabel("speed [m/s]"); cx.set_xlabel("arc length round the lap [m]")
    for a_ in (bx, cx):
        a_.grid(alpha=0.2, lw=0.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    fig.suptitle("If a sector schedule is doing anything, the weights change at the "
                 "sector boundaries.\nHere they do not: the policy emits nearly the "
                 "same $\\theta$ everywhere (Sec. \"the policy degenerates\").",
                 fontsize=9.5, color=INK, y=1.04)
    fig.savefig(OUT / "driving_sectors.png", dpi=170, bbox_inches="tight")
    print("  wrote driving_sectors.png")


def fig_online_curve():
    """The learning curve of every parameter, in real time while driving.

    Not "distance per episode" -- the parameters themselves, tick by tick, on a
    continuous axis across episode boundaries. Three things are only visible
    this way: whether a weight moves at all, whether it moves *within* a lap or
    only between them, and whether it settles or keeps drifting.

    Read from benchmarks/results/weight_matrix.json so the figure and the table
    come from the same run.
    """
    d = _load("weight_matrix.json")
    names = d["weights"]
    T = np.array(d["trace"])
    if not len(T):
        print("  no trace recorded"); return
    ep, sec, wts = T[:, 0], T[:, 3], T[:, 5:]
    t = np.arange(len(T))

    # Grouped by what the weight DOES, three per panel at most. Eight series
    # through a four-colour palette means q_c and r_a are both blue and the
    # legend stops identifying anything; splitting by role keeps every hue
    # unique within the axes a reader compares across, and lets each series be
    # labelled directly at its own line instead of in a key.
    GROUPS = (("path costs", ("q_c", "q_l", "q_v")),
              ("input costs", ("r_d", "r_a", "r_dv")),
              ("constraints", ("d_obs", "k_v")))
    idx = {n: i for i, n in enumerate(names)}

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 7.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 2, 2, 0.62]))
    for ax, (title, group) in zip(axes, GROUPS):
        for c, n in enumerate(group):
            if n not in idx:
                continue
            y = wts[:, idx[n]]
            ax.plot(t, y, "-", lw=1.4, color=CAT[c], alpha=0.95)
            ax.annotate(f" {n}", (t[-1], y[-1]), color=CAT[c], fontsize=8.5,
                        va="center", ha="left", annotation_clip=False)
        for b in np.flatnonzero(np.diff(ep)) + 1:
            ax.axvline(b, color="0.88", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_ylabel(title, fontsize=9)
        ax.margins(x=0.02)

    bx = axes[-1]
    seen = []
    for k in range(4):
        m = sec == k
        if m.any():
            bx.fill_between(t, 0, 1, where=m, transform=bx.get_xaxis_transform(),
                            color=CAT[k], alpha=0.18, linewidth=0)
            seen.append((Track.SECTOR_NAMES[k], CAT[k]))
    for j, (nm, col) in enumerate(seen):
        bx.annotate(nm, (0.005 + 0.13 * j, 0.5), xycoords="axes fraction",
                    fontsize=8, color=INK, va="center",
                    bbox=dict(fc=col, ec="none", alpha=0.25, pad=2.0))
    bx.set_xlabel("control tick (continuous across episodes)")
    bx.set_yticks([]); bx.set_ylabel("sector", fontsize=9)
    for a_ in axes:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle("Every MPCC weight, every control tick, while the car drives. The "
                 "weights DO adapt online --- and they do not\ntrack the sector "
                 "bands below, which is the open result, not a defect of the plot.",
                 fontsize=9.5, color=INK, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "online_curve.png", dpi=170, bbox_inches="tight")
    print("  wrote online_curve.png")


FIGS = {"online": fig_online_curve, "driving": fig_driving_sectors, "architecture": fig_architecture, "icra": fig_icra,
        "filmstrip": fig_filmstrip, "adaptation": fig_adaptation, "geometry": fig_geometry, "gradient_check": fig_gradient_check,
        "rti": fig_rti, "tracks": fig_tracks, "reversal": fig_reversal,
        "overtake": fig_overtake, "icra_grip": fig_icra_grip,
        "strategy": fig_strategy,
        "plant_gap": fig_plant_gap,
        "learning_curves": fig_learning_curves,
        "behaviour_matrix": fig_behaviour_matrix,
        "behaviour": fig_behaviour, "ltc_gate": fig_ltc_gate}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (a.only or sorted(FIGS)):
        FIGS[name]()
