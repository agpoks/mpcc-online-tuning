<p align="center">
  <img src="docs/source/_static/logo-banner.svg" alt="mpcc-online-tuning" width="560">
</p>

# mpcc-online-tuning

**Safe online learning of an MPCC's cost weights — refining a stable base
controller *while the car drives*, without ever letting it construct an unsafe
one.**

A model predictive contouring controller (MPCC) solves a constrained
optimal-control problem every 50 ms; what it *does* inside those constraints is
set by a handful of cost weights. This repo treats those weights as the output of
a small recurrent policy and learns them online: a base policy is fitted offline,
then kept improving on the real vehicle, surface and opponents it meets — a
direct route to closing the sim-to-real gap.

The hard part is not finding better weights but **moving them safely**. An
unconstrained tuner walks a weight to a bound, saturates, and silently claims
more grip or aggression than the car has — and leaves the track. The method makes
that impossible by construction, and it is the subject of the accompanying RA-L
paper.

Docs: [`docs/`](docs) (built on Read the Docs). **New here?** Start with the
tutorial, [`docs/source/tutorial/learning_algorithm.ipynb`](docs/source/tutorial/learning_algorithm.ipynb),
which builds the whole learning rule from the ground up with only `numpy`.

## The method

- **The learned weights are a run-time OCP parameter.** They enter only the cost,
  so the hard constraints (track, grip, actuators) hold whatever the policy
  emits. A weight can change behaviour; it cannot relax a constraint.
- **Anchored and boxed.** The policy is anchored at a hand-verified baseline and
  squashed into a box *measured* from where each weight stops helping — it bends
  the weights inside the box but can never leave it.
- **Class-conditioned bands** clamp the weights that carry safety ($k_v$ grip
  band, $r_a$ floor, $q_c$ floor), each removing one measured failure.
- **Validated keep-best.** An update is banked only after a frozen, noise-free lap
  confirms it, and reverted on any regression — the stopping criterion the
  gradient lacks, and the on-vehicle adaptation/safety layer.
- **The learning signal is the measured return**, read by a small return-fitted
  critic and a score-function actor, carried online by real-time recurrent
  learning (no backprop through time). It needs no exact Hessian and no second
  solve — the same single acados solve runs whether it is learning or frozen.

## Run it

```bash
pip install -e .

# understand it, no solver needed:
python examples/tune_online.py            # the learning rule on a toy plant
python examples/gradient_check.py          # the weight gradient vs finite differences
```

The real experiments use a C-code [acados](https://github.com/acados/acados) OCP
and the dynamic drift plant from
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax):

```bash
export ACADOS_SOURCE_DIR=$HOME/acados-v060
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$HOME/github/scuderia_gym_jax:.

# refine a base policy online (return critic + validated keep-best):
python3 experiments/online_from_baseline.py --tracks icra_t2_smooth \
    --conditions tuner --critic return --clock progress \
    --theta-explore 0.15 --keep-best --validate --seeds 3 --episodes 15

# the main comparison table (baseline / best constant / fitted / online):
bash scripts/reproduce_frozen_summary.sh
```

## What it does, measured

- **Safe online refinement.** In qualifying, lap time falls 54.9 → 44.3 s with the
  keep-best staircase monotone down and every regression reverted. Run
  continuously on the deployment map, the tuner drove 14.3 laps through 38
  track-departure ticks, each recovered by a revert, none terminal. The grip band
  cuts grip-over-claim departures from 100% to 12% of runs.
- **One policy, a variable occluded field.** A single online-learned policy races
  one to three opponents under a 270° occluding sensor, no retraining:
  100 / 100 / 67% clean ($N=1/2/3$), passes scaling with the field. Zero track
  departures; the three-car residual is occluded-opponent contact.
- **Memory, by ablation.** On the equal-pace occluded opponent the recurrent
  policy completes 0.89 of passes frozen against the memoryless 0.67 — stable from
  learning to deployment — by tracking the car through the blind arc. A design
  choice, reported with its ablation, not the contribution.
- **Real-time.** One acados solve holds 20 Hz, 41 / 48 ms (p50/p95) on the
  deployment map in simulation; policy inference 61 µs (LTC) / 29 µs (MLP).

## The spike it grew from

This began as a throwaway spike that asked whether an MPC's weight gradient is
free (the **envelope theorem**: $\mathrm{d}J^*/\mathrm{d}\theta =
\partial\mathcal{L}/\partial\theta$ at the solution) and whether the loop is
real-time. Both answers were yes — but the method that shipped is not the spike:

- the envelope / `V=−J*` critic was **shelved** — it points the same way in every
  situation, collapses the opponent classes onto one behaviour, and drifts into
  over-claiming grip. The **return-fitted critic + score-function actor** replaced
  it;
- the predictive **safety filter** was **replaced** by the box-and-bands-and-
  keep-best construction — a filter absorbs a bad weight *after* the solve, so
  nothing then stops the tuner pushing $q_c$ to zero;
- the plant became a **dynamic single-track drift model on acados**, not a
  kinematic bicycle on IPOPT.

That path — including the measured dead ends, which are *why* the current design
looks the way it does — is kept honestly in the docs' **"How we got here"**
section ([`docs/source/history.md`](docs/source/history.md)).

## What is deliberately wrong with the controller

The evaluation plant is a dynamic drift model with slip, load transfer and a
rate-limited servo; the controller predicts with a simpler model and does **not**
share the plant's parameters. That mismatch is the point — absorbing it online,
rather than re-identifying offline, is what the learning is for. A shared model
would make the question unaskable.

## Layout

```
mpcc-online-tuning/
├── mpcc_tuning/     acados_mpcc.py / acados_ocp.py (the OCP), ltc.py (the weight
│                    policy + tuner), baselines.py (anchor + bands), track.py
├── examples/        gradient_check.py, tune_online.py — the toy tutorial (.py/.ipynb)
├── experiments/     online_from_baseline.py and the evaluation studies
├── results/         seeded data + figures (incl. deploy_labtest/ for the lab map)
├── node/            the ROS 2 deployment node (branch deploy-car)
└── docs/            Sphinx / Read the Docs source
```

Every number is reproducible from a seed; the paper's figures regenerate
data-only from the committed `results/`.

## References

* Brunke et al., *Safe Learning in Robotics*, Annual Reviews 2022 — the survey to
  read first
* Hasani et al., *Liquid Time-Constant Networks*, AAAI 2021 — the recurrent policy
  cell
* Williams & Zipser, *A Learning Algorithm for Continually Running Fully Recurrent
  Neural Networks*, 1989 — the RTRL the policy update uses
* Nguyen et al., *AD-MPCC: Adaptive Differentiable MPCC for Autonomous Racing*,
  2026 — the closest neighbour, adapting MPCC weights per tick through the solver
* Liniger, Domahidi & Morari, *Optimization-based autonomous racing of 1:43 scale
  RC cars*, 2015 — the MPCC formulation itself
* Gros & Zanon, *Reinforcement Learning for MPC*, IFAC 2023 — the
  MPC-as-function-approximator line the spike was built on

The TD(λ) machinery is shared in spirit with
[`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground), where the same
outer loop drives a recurrent network instead of a solver.

## License

MIT, see [`LICENSE`](LICENSE).
