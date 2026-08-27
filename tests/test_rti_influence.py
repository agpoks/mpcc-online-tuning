"""The claim the paper rests on: a warm-started solver carries information.

These are cheap versions of ``experiments/rti_influence.py``. They assert the
*structure* of the result -- that the memoryless assumption fails at one
iteration and holds at many -- rather than the exact numbers, which move with
the horizon and the track.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.rti_influence import (
    jac, reference_states, replay_cold, replay_rti, contraction,
)
from mpcc_tuning.model import KinematicBicycle
from mpcc_tuning.mpcc import MPCC, MPCCWeights
from mpcc_tuning.track import Track


@pytest.fixture(scope="module")
def setup():
    track = Track.oval()
    theta = MPCCWeights().to_log()
    ref = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=8, dt=0.05, max_iter=200)
    states = reference_states(track, ref, theta, 16)
    return track, theta, states


def _solver(track, max_iter):
    return MPCC(track, model=KinematicBicycle(dt=0.05), horizon=8, dt=0.05,
                max_iter=max_iter)


def test_warm_started_converged_matches_cold_converged(setup):
    """The sanity check the whole experiment depends on.

    If the solver is run to convergence the warm start cannot matter -- both
    paths reach the same optimum -- so these two must agree. If they do not,
    the harness is measuring something other than solver memory and every other
    number here is meaningless.
    """
    track, theta, states = setup
    conv = _solver(track, 200)
    J_warm = jac(lambda th: replay_rti(conv, states, th), theta)
    J_cold = jac(lambda th: replay_cold(conv, states, th), theta)
    num = np.linalg.norm(J_warm - J_cold)
    den = max(np.linalg.norm(J_cold), 1e-12)
    assert num / den < 0.05, f"warm and cold converged disagree by {num/den:.1%}"


def test_a_genuine_rti_agrees_with_the_memoryless_gradient(setup):
    """The result the paper reports: the assumption everyone makes is fine.

    One full QP per tick, warm-started -- what acados' SQP_RTI does -- gives a
    sensitivity in the *same direction* as the memoryless one. If this ever
    fails, the paper's conclusion has flipped and needs rewriting, not patching.
    """
    from experiments.rti_influence import replay_sqp_rti
    track, theta, states = setup
    J_rti = jac(lambda th: replay_sqp_rti(_solver(track, 200), states, th), theta)
    J_mem = jac(lambda th: replay_cold(_solver(track, 200), states, th), theta)
    a, b = J_rti.ravel(), J_mem.ravel()
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    assert cos > 0.95, f"SQP-RTI disagreed with memoryless in direction (cos {cos:.4f})"


def test_capping_an_interior_point_solver_is_not_an_rti(setup):
    """The trap, pinned as a regression.

    Capping IPOPT at one iteration looks like a cheap stand-in for real-time
    iteration and is not: the solves fail, the iterate moves further than a
    converged step, and the sensitivity of that non-solution reads as evidence
    that the memoryless gradient is wrong. An earlier version of this work
    reported exactly that. The assertion is on the *diagnosis* -- no solve
    succeeds -- rather than on the misleading number itself.
    """
    track, theta, states = setup
    capped, conv = _solver(track, 1), _solver(track, 200)
    capped.reset()
    conv.reset()
    n_ok, moves_capped, moves_conv = 0, [], []
    for s5 in states:
        w_in = capped._w0 if capped._w0 is not None else capped._initial_guess(s5)
        out = capped._solve(s5, theta)
        capped._w0 = out["w"]
        n_ok += bool(out["ok"])
        moves_capped.append(np.linalg.norm(out["w"] - w_in))

        w_in = conv._w0 if conv._w0 is not None else conv._initial_guess(s5)
        out = conv._solve(s5, theta)
        conv._w0 = out["w"]
        moves_conv.append(np.linalg.norm(out["w"] - w_in))

    # The primary invariant: a capped interior-point solve is a *failed* solve.
    assert n_ok == 0, f"{n_ok} capped solves reported success; the trap has changed"
    # And it overshoots. The multiple depends on the horizon -- 11x at N=12,
    # 2.5x at N=8 -- so the assertion is on the sign of the effect, not a
    # threshold tuned to one problem size.
    assert np.mean(moves_capped) > 1.5 * np.mean(moves_conv), (
        f"the capped solver moved {np.mean(moves_capped):.3g} against the "
        f"converged {np.mean(moves_conv):.3g}; it no longer overshoots and "
        f"this regression no longer describes the trap")


def test_enough_iterations_recovers_the_memoryless_gradient(setup):
    """The other half: the disagreement is about truncation, not a bug."""
    track, theta, states = setup
    J_many = jac(lambda th: replay_rti(_solver(track, 50), states, th), theta)
    J_mem = jac(lambda th: replay_cold(_solver(track, 200), states, th), theta)
    a, b = J_many.ravel(), J_mem.ravel()
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    assert cos > 0.9, f"50 iterations still disagreed (cos {cos:.3f})"


def test_the_warm_start_perturbation_decays(setup):
    """rho < 1: the warm start really is memory, and it is finite.

    This is the half of the note that survives. The solver carries state and it
    forgets geometrically; what the measurement shows is that the part of that
    state which reaches the gradient is negligible, not that the state is not
    there.
    """
    track, theta, states = setup
    gap = contraction(_solver(track, 200), states, theta)
    ratios = [gap[i + 1] / gap[i] for i in range(len(gap) - 1) if gap[i] > 1e-25]
    rho = float(np.median(ratios))
    assert 0.0 < rho < 1.0, f"measured rho = {rho:.3f}, not a contraction"
    assert gap[-1] < gap[0], "the perturbation did not decay at all"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("scuderia_gym_jax") is None,
    reason="scuderia_gym_jax not on the path")
def test_it_runs_on_the_fitted_tyre_plant():
    """The same experiment against real fitted tyres.

    Uses weights the controller survives on: with the defaults the scuderia
    plant crashes in ~17 steps, the reference trajectory ends in a wall, and
    every solve after that is ill-conditioned -- the finite differences then
    measure solver noise. That failure is quiet, so it is pinned here.
    """
    track = Track.oval()
    theta = np.log(np.array([10.0, 200.0, 0.02, 10.0, 0.01, 0.1]))
    ref = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=8, dt=0.05,
               max_iter=200)
    states = reference_states(track, ref, theta, 20, plant="scuderia")
    assert len(states) >= 20, (
        "the controller crashed on the scuderia plant, so this reference "
        "trajectory cannot be used for a sensitivity measurement")
    J_rti = jac(lambda th: replay_rti(_solver(track, 1), states, th), theta)
    assert np.all(np.isfinite(J_rti))
