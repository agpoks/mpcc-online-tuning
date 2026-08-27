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


FIGS = {"geometry": fig_geometry, "gradient_check": fig_gradient_check}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (a.only or sorted(FIGS)):
        FIGS[name]()
