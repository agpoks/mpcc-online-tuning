# The policies, one by one — the paper's result table

This page is the reference for the paper's main comparison on ICRA T2. Five
ways of choosing the MPCC's eight cost weights
`(q_c, q_l, q_v, r_d, r_a, r_dv, d_obs, k_v)` are compared as **fixed
policies**, each driven from three physically different starts with learning
and exploration switched off. Each has its own subsection below: what it is,
how it works, its result, and the exact command to reproduce it.

Two things are constant across every row. The **controller** is always the same
acados MPCC solving the same optimal-control problem every 50 ms. The **car** is
the dynamic drift model on the scuderia STD plant. Only the weights change, and
only *how* they are chosen.

The eight weights in one line each: `q_c` how expensive it is to leave the
reference line, `q_l` lag along the path, `q_v` how much progress is rewarded,
`r_d`/`r_a`/`r_dv` how much steering, acceleration and their rates are damped,
`d_obs` the berth around an opponent, `k_v` how much grip the plan claims.

## The story in one paragraph

A team never starts a race with optimal weights. It starts from a **safe
baseline** that drives but leaves time on the table, and adapts. This work
replaces the single constant with a small recurrent network that emits the
weights afresh every tick from what the car sees, learns a **situation-dependent
schedule** offline from a search over per-sector demands, and then **adapts it
online while driving**. The table below is that progression: the safe baseline,
the best a constant can do, the fitted network, and the network refined online
by two different critics.

## Results (ICRA T2, laps in 2500 steps ≈ 125 s; driven cold from a standstill unless noted)

| policy | seed 0 | seed 1 | seed 2 | mean | clean? |
|---|---|---|---|---|---|
| START — safe baseline (constant) | 1.87 | 2.01 | 2.00 | 1.96 | yes |
| best constant by search (context) | 2.32 | 2.33 | 2.40 | 2.35 | yes |
| **grid-fitted network** (deliverable) | 2.48 | 2.49 | 2.76 | **2.58** | yes |
| online, RETURN critic | 2.42 | 2.33 | 2.74 | 2.50 | yes |
| online, MPCC critic | 2.31✗ | 1.06✗ | 1.01✗ | — | no (cold) |
| online, MPCC critic — flying start | 2.79 | 2.74 | 2.84 | 2.79 | yes |

`✗` = left the track. The MPCC-critic row is the paper's cautionary case: it is
genuinely the fastest policy once the car is moving (2.79 laps, and 5.75 in a
long run), but from a cold standstill it over-drives the tightest corner and the
solver's QP goes infeasible. See its subsection.

Figures: `paper/figures/cold_summary.png` (all rows, cold) and
`paper/figures/scenario_summary.png` (standing vs flying start).

---

## START — the safe baseline (a constant)

**What it is.** One fixed weight vector, the same at every point of the track,
`baselines.START`: `q_c=1.0, q_l=50, q_v=0.2, r_a=6.0, k_v=0.40`. Chosen to be
stable and *not* aggressive — a low grip claim, heavy input damping. It drives
cleanly but slowly, peaking around 1.5 m/s where the track allows much more.

**How it works.** No learning. The MPCC solves with these weights every tick.
This is the honest starting point of the experiment and the thing every other
row is measured against — not a target, a floor.

**Result.** 1.96 laps, clean from every start.

**Reproduce.**

    # START is baselines.START (a constant); drive it held-constant for 1 episode:
    python3 experiments/online_from_baseline.py --tracks icra_t2_raceline \
        --conditions fixed --seeds 3 --episodes 1
    # reports laps per seed for START held constant

## Best constant by search — context, not a target (a constant)

**What it is.** The single best weight vector found by exhaustive search over a
grid of `(q_v, k_v, q_c)`, keeping the one with the highest mean laps over the
whole track (`k_v=0.50`). `baselines.BEST`.

**How it works.** `experiments/situation_demands_acados.py` drives every vector
in every `(sector, entry-speed)` cell and reports both the best per cell and the
best single constant across all cells. This row is that single constant. It is
an **upper bound on what any constant can do**, drawn in the figures as a
reference line: a policy that beats it has done something no fixed set of
weights can. Nobody has this vector at the start of a race — it is the answer to
a search, shown for scale.

**Result.** 2.35 laps, clean.

**Reproduce.**

    # the search that finds it (prints the best single constant + per-cell winners):
    python3 experiments/situation_demands_acados.py --tracks icra_t2_raceline \
        --with-qc --steps 400 --jobs 1
    # the winner is stored as baselines.BEST; drive it the same way as START.

## Grid-fitted network — the deliverable (a situation-dependent policy)

**What it is.** The point of the project. A small liquid-time-constant (LTC)
recurrent cell plus a linear readout emits the eight weights **afresh every
tick** from the features the car sees: the sector ahead (a soft membership that
ramps over ~2 m), the car's speed, the curvature preview, the corridor width.

**How it works, in three steps.**

1. **The situation grid** (`situation_demands_acados.py --with-qc`) drives every
   `(q_v, k_v, q_c)` combination in every `(sector, entry-speed)` cell for 400
   steps and keeps the winner of each cell — a measured table of *what weights
   each situation wants*.
2. **Supervised fit** (`scripts/fit_policy_to_grid.py`) trains the network to
   reproduce those per-cell winners. The readout sees `[hidden state; features;
   1]` — a **direct path** from the features, because the LTC hidden state alone
   barely varies with the input and cannot express sector-dependence. A
   closed-form ridge solve on the invertible squash plus a few SGD epochs fits
   it (RMSE ≈ 0.35 in log-weight space).
3. **A safety cap** on the target grip claim, `--kv-cap 0.50`: the grid's cells
   are 20-second bursts and its favourite `k_v=0.85` is lap-fragile, so the
   schedule is kept at the grip claim shown clean over full laps.

The result is a policy that **lowers `q_c` on the straights** — letting the car
leave the centre line for a faster line — and **raises it in the corners**, with
`q_v` moving the opposite way: a light, smooth switching by sector, learned from
data, no hand-set thresholds. It is produced with **no critic and no
exploration noise** — a supervised fit plus one cap.

**Result.** 2.48 / 2.49 / 2.76 = **2.58 laps, clean, reproducible cold** — above
the best constant, and the strongest fixed policy in the table. The gain over
the constant is exactly the sector-dependence, since the grip claim is the same.

**Reproduce.**

    python3 scripts/fit_policy_to_grid.py --track icra_t2_raceline \
        --grid situation_demands_qc.json --kv-cap 0.50 --epochs 10 --eval
    # then drive the saved network cold on three starts:
    python3 scripts/drive_policy.py results/fitted_policy_icra_t2_raceline_kv0.50.npz

## Online, RETURN critic — adaptation that holds (situation-dependent + online)

**What it is.** Starts from the grid-fitted network and keeps learning **while
driving**, with TD(λ). Its **critic** — the learner's estimate of how good the
situation is — is a small separate linear model fitted to the **actual return**
(lap time), which never sees the weights or the controller's cost.

**How it works.** The actor is nudged by small Gaussian perturbations of the
emitted weights, reinforced when they help via the score-function estimator; the
critic is trained by TD(λ) on the reward. θ enters the return only through the
policy. A **keep-best-and-validate** rule banks the network only after a
frozen validation episode confirms an improvement (see the algorithm).

**Result.** 2.42 / 2.33 / 2.74 = 2.50 laps, clean — it **holds** the fitted
network's performance but does not improve on it here.

**Reproduce.**

    python3 experiments/online_from_baseline.py --tracks icra_t2_raceline \
        --init-policy results/fitted_policy_icra_t2_raceline_kv0.50.npz \
        --clock progress --critic return --theta-explore 0.05 --explore 0 \
        --keep-best --validate --eval-episodes 1 --seeds 3 --episodes 10

## Online, MPCC critic — the fast, cautionary case (situation-dependent + online)

**What it is.** Same online setup, but the critic is the controller's **own
optimal cost**, `V = −J*`, which is free to compute every tick.

**How it works, and why it is the cautionary row.** `V = −J*` points the learner
in a direction fixed by the *units* of the cost rather than by how the car drove
(the envelope gradient of a penalty weight is always non-negative, of the
progress weight always negative), so the actor drifts to aggressive weights. It
is genuinely the **fastest** policy once the car is moving: **2.79 laps from a
flying start, 5.75 in a long run**. But those aggressive weights build ~4 m/s
into the tightest hairpin; from a **cold standstill** the recurrent policy needs
a lap to settle into a moderate speed regime, and before it does, the car
over-drives the corner, the tyres saturate, the MPCC's QP goes ill-conditioned
and acados returns a failure. Driven cold it leaves the track (1.0–2.3 laps).
Given a warm-up lap — the normal race procedure — it drives clean.

This row is kept in the paper because it separates two things a single number
hides: the **policy** (fast) from the **deployment condition** (needs a warm-up
lap, or a solver-failure guard in the control loop). See {doc}`safety` and the
start-scenario analysis.

**Result.** cold: crashes (1.0–2.3 laps); **flying start: 2.79 laps, clean**.

**Reproduce.**

    # learn it:
    python3 experiments/online_from_baseline.py --tracks icra_t2_raceline \
        --init-policy results/fitted_policy_icra_t2_raceline_kv0.50.npz \
        --clock progress --critic mpcc --explore 0.05 \
        --keep-best --validate --eval-episodes 1 --seeds 3 --episodes 10
    # drive a banked network both ways (standing vs flying):
    python3 scripts/racing_check.py \
        results/paper/frozen_summary_v1/best_policy_icra_t2_raceline_2_progress_mpcc_val.npz \
        --box-from results/paper/frozen_summary_v1/fitted_policy_icra_t2_raceline_kv0.50.npz

---

## Standing start vs flying start — a solver-init property, not a tuning one

The MPCC solver carries a warm-start from tick to tick; from a dead stop it
starts cold. Measured on the aggressive network: cold it drifts to ~4 m/s and
the hairpin QP fails; after a warm-up lap it stays ≤2.9 m/s and drives clean for
5.75 laps. What the warm-up lap initialises is the **recurrent policy's hidden
state** (which sets the speed regime), not the solver's warm-start — a standard
recurrent-controller property, exactly like a driver's warm-up lap. The
grid-fitted network needs no warm-up because it was fitted to safe per-sector
targets. Full analysis and the reproducible scenarios: {doc}`safety` and
`results/paper/scenarios/README.md`.

## Reproducing the whole table

All runs are deterministic given the acados build: same start, same plant, same
seed. Environment for every command:

    export ACADOS_SOURCE_DIR=$HOME/acados-v060
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
    export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$HOME/github/scuderia_gym_jax:.

The **archive** `results/paper/frozen_summary_v1/` holds the three banked
networks, the fitted start network, every run's JSON, the situation grid, the
reference rows and the figures, with its own README. One command regenerates the
whole table from scratch (~6 h on 16 cores):

    bash scripts/reproduce_frozen_summary.sh          # all steps
    bash scripts/reproduce_frozen_summary.sh --from 7 # resume at a step

Re-drive any saved policy network cold in minutes:

    python3 scripts/drive_policy.py <network.npz> [--box-from <fitted.npz>]

**Controller version matters.** The numbers above were measured on the
smoothed-corridor controller *without* the soft grip constraint (git tag
`pre-grip-dynamic`). A separate investigation adds a soft grip cap and re-fits
every policy on it; that changes the numbers (it slows corners for safety). To
reproduce the table above exactly, check out `pre-grip-dynamic` first. The
method and the per-policy explanations are unchanged by the controller version.

## The algorithm, for the paper

The LaTeX pseudo-code for the online tuner (TD(λ) on the weight policy with
keep-best-and-validate) and the offline grid-fit warm-start is in
`paper/algorithms.tex`, ready to `\input` into the paper.
