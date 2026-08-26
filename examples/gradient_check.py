"""Check the claim the whole approach rests on: the gradient is free, and exact.

    python examples/gradient_check.py
    python examples/gradient_check.py --n-states 12 --plot runs/gradient_check.png

If this fails, the tuner is following something that is not a gradient and
every other number in the repo is meaningless. It is the first thing to run.
"""

# %% [markdown]
# # Is the envelope-theorem gradient the real gradient?
#
# Tuning an MPC's cost weights online is only affordable because of one fact.
# At the solution of the MPCC's NLP, the derivative of the optimal value with
# respect to a cost weight is the **partial** derivative of the Lagrangian,
# with the primal and dual variables held fixed:
#
# $$\frac{\mathrm{d}J^*}{\mathrm{d}\theta}
#   \;=\; \frac{\partial \mathcal{L}}{\partial \theta}\Big|_{w^*,\lambda^*},
#   \qquad \mathcal{L} = f(w,\theta) + \lambda^\top g(w,s,\theta)$$
#
# There is no $\mathrm{d}w^*/\mathrm{d}\theta$ term — it is annihilated by the
# stationarity condition. So no implicit function theorem, no differentiating
# through the solver, no adjoint sweep: the gradient falls out of a solve that
# was happening anyway.
#
# That is a strong claim, and it is checkable. Below it is compared against
# central finite differences on the optimal value itself, at a spread of states
# around the track and a spread of weight settings.

# %%
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.mpcc import MPCC, WEIGHT_NAMES, MPCCWeights
from mpcc_tuning.track import Track

# %% [markdown]
# ## Sample states around the track
#
# Not one state. A gradient identity that holds at the start/finish straight
# and nowhere else is not an identity — and the interesting places are the
# corners, where the track constraint is active and the multipliers are not
# zero.

# %%
def sample_states(track, n, speed=1.8, jitter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in np.linspace(0, track.length, n, endpoint=False):
        k = int(s / track.ds) % len(track.center)
        p = track.center[k]
        nxt = track.center[(k + 1) % len(track.center)]
        psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
        # nudge off the line and off the heading, so the constraint is
        # sometimes active and the solve is not trivial
        n_hat = np.array([-np.sin(psi), np.cos(psi)])
        off = rng.uniform(-jitter, jitter)
        out.append(np.array([p[0] + n_hat[0] * off, p[1] + n_hat[1] * off,
                             psi + rng.uniform(-0.15, 0.15),
                             speed * rng.uniform(0.7, 1.3), s]))
    return out


# %%
def compare(mpcc, state, theta, eps=1e-4):
    """Envelope gradient vs central finite differences at one state."""
    sol = mpcc.value(state, theta)
    if not sol["ok"]:
        return None
    analytic = mpcc.grad_theta(sol, state, theta)
    fd = np.empty(len(theta))
    for i in range(len(theta)):
        step = eps * np.eye(len(theta))[i]
        up = mpcc.action_value(state, theta + step, sol["u0"])
        dn = mpcc.action_value(state, theta - step, sol["u0"])
        if not (up["ok"] and dn["ok"]):
            return None
        fd[i] = (up["value"] - dn["value"]) / (2 * eps)
    cos = float(analytic @ fd / (np.linalg.norm(analytic) * np.linalg.norm(fd) + 1e-12))
    rel = float(np.linalg.norm(analytic - fd) / (np.linalg.norm(fd) + 1e-12))
    return dict(analytic=analytic, fd=fd, cos=cos, rel=rel)


# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-states", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--mpc-dt", type=float, default=0.15)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    track = Track.oval()
    mpcc = MPCC(track, model=KinematicBicycle(dt=args.mpc_dt), horizon=args.horizon,
                dt=args.mpc_dt)

    settings = [("nominal", MPCCWeights()),
                ("crawling", MPCCWeights(q_l=200.0, q_v=0.05, r_d=1.0)),
                ("aggressive", MPCCWeights(q_c=1.0, q_l=2.0, q_v=20.0, r_d=0.01))]

    rows = []
    print(f"  {'weights':<12}{'state':>7}{'cosine':>10}{'rel err':>10}{'|grad|':>12}")
    for name, w in settings:
        theta = w.to_log()
        for j, state in enumerate(sample_states(track, args.n_states)):
            mpcc.reset()
            r = compare(mpcc, state, theta)
            if r is None:
                print(f"  {name:<12}{j:>7}     solver did not converge -- skipped")
                continue
            rows.append((name, j, r))
            print(f"  {name:<12}{j:>7}{r['cos']:>10.5f}{r['rel']:>10.2e}"
                  f"{np.linalg.norm(r['analytic']):>12.3f}")

    cos = np.array([r["cos"] for _n, _j, r in rows])
    rel = np.array([r["rel"] for _n, _j, r in rows])
    print(f"\n  {len(rows)} comparisons")
    print(f"  cosine     min {cos.min():.5f}   mean {cos.mean():.5f}")
    print(f"  rel error  max {rel.max():.2e}   mean {rel.mean():.2e}")
    ok = cos.min() > 0.999 and rel.max() < 5e-2
    print(f"\n  {'PASS -- the envelope gradient is the gradient' if ok else 'FAIL'}")

    t0 = time.perf_counter()
    state = sample_states(track, 1)[0]
    sol = mpcc.value(state, settings[0][1].to_log())
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(200):
        mpcc.grad_theta(sol, state, settings[0][1].to_log())
    t_grad = (time.perf_counter() - t0) / 200
    print(f"\n  one NLP solve      {t_solve * 1e3:8.1f} ms")
    print(f"  one gradient       {t_grad * 1e6:8.1f} us   "
          f"({t_grad / t_solve * 100:.3f}% of the solve)")
    print("  -- which is the entire argument for why this can run at control rate.")

    if args.plot:
        plot(rows, args.plot)
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = np.concatenate([r["analytic"] for _n, _j, r in rows])
    f = np.concatenate([r["fd"] for _n, _j, r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = max(np.abs(a).max(), np.abs(f).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], color="0.7", lw=1, ls="--", label="y = x")
    ax1.scatter(f, a, s=14, alpha=0.7)
    ax1.set_xlabel("finite difference")
    ax1.set_ylabel("envelope theorem")
    ax1.set_title("every component, every state")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.bar(range(len(rows)), [r["cos"] for _n, _j, r in rows])
    ax2.set_ylim(0.99, 1.001)
    ax2.set_xlabel("comparison")
    ax2.set_ylabel("cosine to finite differences")
    ax2.set_title("agreement per state")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Why this matters more than it looks
#
# The alternative — deterministic policy gradient — needs
# $\mathrm{d}u_0^*/\mathrm{d}\theta$, the derivative of the *solution* rather
# than of the optimal value. That requires differentiating the KKT system: one
# linear solve with the KKT matrix per tick. Affordable, but a different and
# larger piece of machinery, and one more thing to get wrong.
#
# Q-learning needs only the value's gradient, and that is the quantity the
# envelope theorem hands over for nothing. It is the cheap door, which is why
# this spike goes through it.
