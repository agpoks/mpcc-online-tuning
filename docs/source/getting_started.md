# Getting started

## Install

```bash
git clone https://github.com/agpoks/mpcc-online-tuning.git
cd mpcc-online-tuning
pip install -e .
```

There are two tiers, and only the second needs a build:

- **The tutorial runs out of the box.** {doc}`tutorial/learning_algorithm`
  demonstrates every piece of the learning rule on toy problems with only
  `numpy` and `matplotlib` — no solver, no car. Start there.
- **The real experiments need acados and the plant.** The controller is a
  C-code-generated [acados](https://github.com/acados/acados) OCP, and the
  evaluation plant is the dynamic single-track drift model from
  [`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax). Both are
  deliberate: the paper's claims are all on the generated solver the car runs and
  a plant the controller does **not** model.

```bash
# the acados build this repo targets (v0.6.0), plus the plant on the path
export ACADOS_SOURCE_DIR=$HOME/acados-v060
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$HOME/github/scuderia_gym_jax:.
export OMP_NUM_THREADS=1
```

```{admonition} Why a specific acados version
:class: note
The weight policy adapts $k_v$ as a *global* run-time parameter and the online
tuner needs the SQP-with-feasible-QP path, both of which require acados ≥ 0.6.
The default `~/acados` on many machines is 0.1 and cannot code-generate it.
```

## The three things to run

**1. Understand it (no acados).** The tutorial as a script or notebook:

```bash
python examples/tune_online.py          # the learning rule on the toy plant
python examples/gradient_check.py        # the weight gradient vs finite differences
```

**2. Reproduce the main comparison (acados).** Five ways of choosing the weights
— the safe baseline, the best constant, the offline-fitted network, and the
network refined online by two critics — driven from three physical starts. This
is {doc}`policies_in_the_table`:

```bash
# drive any banked policy network cold on three starts
python3 scripts/drive_policy.py <network.npz> [--box-from <fitted.npz>]

# the whole table from scratch (~6 h on 16 cores)
bash scripts/reproduce_frozen_summary.sh
```

**3. Watch it learn online (acados).** The online tuner — the deployed learning
loop — refining a base policy with the return-fitted critic and validated
keep-best:

```bash
python3 experiments/online_from_baseline.py --tracks icra_t2_smooth \
    --conditions tuner --critic return --clock progress \
    --theta-explore 0.15 --keep-best --validate --seeds 3 --episodes 15
```

## The knobs that matter

| flag | what it does |
|---|---|
| `--critic` | `return` is the deployed critic — a small linear value fitted to the measured return, blind to the weights. `mpcc` is the shelved `V=−J*` critic, kept only as the cautionary case (it over-claims grip; see {doc}`policies_in_the_table`) |
| `--theta-explore` | σ of the Gaussian exploration **in weight space** (the score-function actor reads its direction off the realized perturbation). Replaces the spike's actuator noise, which cost a second solve for no information |
| `--keep-best` / `--validate` | bank a policy only after a **frozen, noise-free** lap confirms it beats the incumbent, and revert on any slower lap or track departure. Without it the tuner can bank an exploration fluke |
| `--clock` | `progress` learns once per metre of track rather than once per tick, so credit spans a fixed length of track at any speed (the reward then becomes $-\Delta t$, the lap time stated directly) |
| `--geom-narrow` | a raceline-referenced corridor to drive (e.g. the lab map); the representation the net trains on, built from the optimiser's own world CSV |

The class-conditioned safety bands ($k_v$ grip band, $r_a$ floor, $q_c$ floor
for the recurrent arm) are part of the policy, not flags — see {doc}`safety`.

## What is deliberately wrong

The evaluation plant is a **dynamic single-track drift model** with slip angles,
load transfer, and a rate-limited servo; the controller predicts with a simpler
model and does **not** share the plant's parameters. That mismatch is the point:
a controller tuned in simulation meets a real vehicle it was not tuned for, and
absorbing that gap online — rather than re-tuning offline from scratch — is
exactly what the online learning is for. A shared model would make the question
unaskable. See {doc}`plant`.
