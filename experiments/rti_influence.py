"""Is the memoryless gradient actually wrong under real-time iteration?

    python experiments/rti_influence.py
    PYTHONPATH=/path/to/scuderia_gym_jax python experiments/rti_influence.py --plant scuderia

This is claim 1 of ``docs/source/influence_through_a_solver.md``, and it is the
load-bearing one. The note argues that a warm-started RTI solver is a recurrent
system,

    w_{t+1} = Phi(w_t, s_t, theta),

so the sensitivity of the applied input to the cost weights has a term the
envelope theorem cannot see -- the influence carried in the warm start. If that
term is negligible the whole idea is unnecessary, and finding that out costs an
afternoon.

## The experiment, and why it is shaped this way

The obvious version -- perturb theta, run the closed loop, difference the
result -- conflates two things: the solver's memory, and the fact that a
different theta drives the car somewhere else. Those are both real, but only
the first is what the note claims.

So the state sequence is **replayed**. A reference trajectory of states is
recorded once, and every subsequent run is driven through the *same* states.
The plant is then out of the loop entirely, and the only path from theta to the
solution at tick T is the chain of warm starts. That isolates exactly the term
in question.

Two gradients are then compared at tick T:

``memoryless``  solve at s_T from a cold start, to convergence, and take the
                sensitivity there -- what every MPC-as-function-approximator
                method assumes.
``through``     finite-difference the actual warm-started RTI recursion,
                replaying the same states from tick 0.

If they agree, the note is wrong and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

NAMES = ("q_c", "q_l", "q_v", "r_d", "r_a", "r_dv")


def reference_states(track, mpcc, theta, n, plant="bicycle"):
    """One closed-loop run, recorded. Every later run replays these states."""
    if plant == "scuderia":
        from mpcc_tuning.plant_scuderia import ScuderiaPlant
        p = ScuderiaPlant(track, model="st", dt=0.05)
    else:
        from examples.tune_online import Plant
        p = Plant(track, dt=0.05)
    p.max_steps = n + 5
    s5 = p.reset()
    mpcc.reset()
    states = [s5.copy()]
    for _ in range(n):
        u = mpcc.value(s5, theta)["u0"]
        s5, _r, off, tr = p.step(u)
        states.append(s5.copy())
        if off or tr:
            break
    return states


def replay_rti(mpcc, states, theta, iters=1):
    """Drive the solver through a fixed state sequence, warm-started.

    ``iters=1`` is real-time iteration: one QP/Newton step per tick, the
    previous solution carried forward. This is the deployed setting.
    """
    mpcc.reset()
    u_last = None
    for s5 in states:
        out = mpcc._solve(s5, theta)
        mpcc._w0 = out["w"]                 # the warm start IS the memory
        u_last = out["u0"][:2].copy()
    return u_last


def replay_sqp_rti(mpcc, states, theta, step=1.0):
    """A **genuine** real-time iteration: one full QP per tick, warm-started.

    This is the comparison that matters. Capping an interior-point solver at
    one iteration does not approximate RTI -- it fails, moving the iterate an
    order of magnitude further than a converged step and reporting success on
    none of them -- and the sensitivity of that failure is noise.
    """
    from mpcc_tuning.rti import RTISolver
    r = RTISolver(mpcc, step=step)
    r.reset()
    u = None
    for s5 in states:
        u = r.solve(s5, theta)["u0"][:2].copy()
    return u


def replay_cold(mpcc, states, theta):
    """The memoryless assumption: solve the last state alone, converged."""
    mpcc.reset()
    return mpcc._solve(states[-1], theta)["u0"][:2].copy()


def jac(fn, theta, eps=1e-3):
    """Central-difference Jacobian of a 2-vector w.r.t. the 6 log-weights."""
    J = np.zeros((2, len(theta)))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        J[:, j] = (np.asarray(fn(tp)) - np.asarray(fn(tm))) / (2 * eps)
    return J


def contraction(mpcc, states, theta, eps=1e-6):
    """Empirical rho(D): how fast a perturbation of w decays under replay.

    Perturb the warm start once, replay the same states, and watch the gap to
    the unperturbed run. The note predicts a geometric decay whose rate is the
    spectral radius of the RTI step's Jacobian in w -- and that this rate sets
    how far back the influence has to be carried.
    """
    mpcc.reset()
    base = []
    for s5 in states:
        out = mpcc._solve(s5, theta)
        mpcc._w0 = out["w"]
        base.append(out["w"].copy())

    mpcc.reset()
    rng = np.random.default_rng(0)
    pert = []
    w0 = None
    for i, s5 in enumerate(states):
        out = mpcc._solve(s5, theta)
        w = out["w"].copy()
        if i == 0:
            d = rng.normal(size=w.shape)
            w = w + eps * d / np.linalg.norm(d)
        mpcc._w0 = w
        pert.append(w)
    gap = [float(np.linalg.norm(a - b)) for a, b in zip(base, pert)]
    return gap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="bicycle", choices=["bicycle", "scuderia"])
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", nargs=6, type=float, default=None,
                    metavar=("Q_C", "Q_L", "Q_V", "R_D", "R_A", "R_DV"),
                    help="cost weights to linearise about. On the scuderia "
                         "plant the defaults crash in ~17 steps, and a "
                         "reference trajectory that ends in a wall makes every "
                         "solve after it ill-conditioned -- the finite "
                         "differences then measure solver noise, not "
                         "sensitivity. Use weights the controller survives on.")
    a = ap.parse_args(argv)

    track = Track.oval()
    theta = (np.log(np.array(a.weights)) if a.weights
             else MPCCWeights().to_log())
    ref_solver = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=a.horizon,
                      dt=0.05, max_iter=200)
    states = reference_states(track, ref_solver, theta, a.ticks, plant=a.plant)
    print(f"  plant: {a.plant};  replaying {len(states)} states")
    print("  weights: " + "  ".join(f"{n}={v:g}" for n, v in
                                    zip(NAMES, np.exp(theta))))
    if len(states) < a.ticks:
        print(f"  WARNING: the controller did not survive {a.ticks} ticks on "
              f"this plant, so the reference trajectory ends in a crash and "
              f"the numbers below are solver noise. Pass --weights.")
    print()

    rti = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=a.horizon,
               dt=0.05, max_iter=1)          # real-time iteration
    conv = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=a.horizon,
                dt=0.05, max_iter=200)       # converged

    J_sqp = jac(lambda th: replay_sqp_rti(conv, states, th), theta)
    J_through = jac(lambda th: replay_rti(rti, states, th), theta)
    J_memless = jac(lambda th: replay_cold(conv, states, theta=th), theta)
    # And the same solver, converged but still warm-started -- to separate "RTI
    # is truncated" from "the warm start carries information".
    J_warmconv = jac(lambda th: replay_rti(conv, states, th), theta)

    def rel(A, B):
        return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-12))

    def cos(A, B):
        a_, b_ = A.ravel(), B.ravel()
        return float(a_ @ b_ / max(np.linalg.norm(a_) * np.linalg.norm(b_), 1e-12))

    print("  d u0 / d theta at the final tick, three ways\n")
    print(f"  {'':<26}{'|J|':>10}{'rel. err vs memoryless':>24}{'cosine':>10}")
    for name, J in (("SQP-RTI (one full QP)", J_sqp),
                    ("IPOPT max_iter=1", J_through),
                    ("memoryless (converged)", J_memless),
                    ("warm-started, converged", J_warmconv)):
        print(f"  {name:<26}{np.linalg.norm(J):10.4f}"
              f"{rel(J, J_memless):22.3f}{cos(J, J_memless):10.4f}")

    print("\n  per-weight, d(delta)/d(log w):\n")
    print(f"  {'weight':<8}{'through':>12}{'memoryless':>12}{'ratio':>9}")
    for j, nm in enumerate(NAMES):
        t_, m_ = J_through[0, j], J_memless[0, j]
        r = m_ / t_ if abs(t_) > 1e-9 else float("nan")
        print(f"  {nm:<8}{t_:12.5f}{m_:12.5f}{r:9.2f}")

    gap = contraction(rti, states, theta)
    g0 = max(gap[0], 1e-30)
    decay = [g / g0 for g in gap]
    print("\n  perturbation of the warm start, decay under replay:")
    for k in (0, 1, 2, 4, 8, 16):
        if k < len(decay):
            print(f"    after {k:2d} ticks: {decay[k]:.3e}")
    ratios = [gap[i + 1] / gap[i] for i in range(min(12, len(gap) - 1))
              if gap[i] > 1e-25]
    rho = float(np.median(ratios)) if ratios else float("nan")
    print(f"    median per-tick ratio (empirical rho): {rho:.4f}")

    # How converged do you have to be before the envelope assumption is safe?
    # IPOPT with max_iter=1 is NOT acados' SQP_RTI -- one interior-point
    # iteration is a much smaller step than a full QP solve -- so the honest
    # version of the question is a sweep over solver effort rather than a
    # single "RTI" point.
    print("\n  solver effort vs the memoryless assumption:\n")
    print(f"  {'max_iter':>9}{'|J|':>11}{'rel. err':>11}{'cosine':>10}")
    sweep = {}
    for mi in (1, 2, 3, 5, 10, 25, 50, 100, 200):
        sv = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=a.horizon,
                  dt=0.05, max_iter=mi)
        J = jac(lambda th: replay_rti(sv, states, th), theta)
        sweep[mi] = dict(norm=float(np.linalg.norm(J)),
                         rel=rel(J, J_memless), cos=cos(J, J_memless))
        print(f"  {mi:9d}{np.linalg.norm(J):11.4f}"
              f"{rel(J, J_memless):11.3f}{cos(J, J_memless):10.4f}")

    out = dict(plant=a.plant, ticks=len(states), rho=rho, sweep=sweep,
               rel_err_memoryless=rel(J_memless, J_through),
               cos_memoryless=cos(J_memless, J_through),
               rel_err_warmconv=rel(J_warmconv, J_through),
               rel_err_sqp=rel(J_sqp, J_memless), cos_sqp=cos(J_sqp, J_memless),
               J_sqp=J_sqp.tolist(),
               J_through=J_through.tolist(), J_memoryless=J_memless.tolist(),
               decay=decay)
    path = Path(a.out or ROOT / "benchmarks" / "results" / f"rti_{a.plant}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
