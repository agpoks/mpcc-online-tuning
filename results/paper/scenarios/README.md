# Start-scenario results — reproducible

How each policy drives from three different start scenarios on ICRA T2. The
point is a distinction the earlier tables missed: whether a policy needs a
**warm-up lap**, and why. All numbers are laps in 2500 steps, three physically
different starts each, driven with **learning off and exploration off** (a fixed
policy). Figure: `paper/figures/scenario_summary.png`.

## The three scenarios

| scenario | what it is | how to run |
|---|---|---|
| **standing (cold)** | fresh solver, fresh policy memory, from a stop. The car switched on cold. | `scripts/drive_policy.py <net.npz>` |
| **grid start** | a warm-up (formation) lap under the safe baseline, then the car lines up at the start line **at rest**, keeping the warmed solver and policy memory, then races. **This is the normal race procedure.** | `scripts/racing_check.py <net.npz>` (grid column) |
| **flying (moving)** | a warm-up lap, then the race policy takes over **while still moving** at the grid. | `scripts/racing_check.py <net.npz>` (flying column) |

## What the scenarios showed (the mechanism)

A policy that crashes cold does so because from a standing start its recurrent
memory drifts into a **fast speed regime**; it reaches ~4 m/s, and the tightest
hairpin then becomes infeasible for the solver (acados returns status 4 — QP
failure — for many ticks, the control thrashes, the car leaves the track). The
solver is *warm* by then (it has solved hundreds of times); the problem is the
speed, set by the policy's memory. A warm-up lap settles that memory into a
moderate regime (~2.9 m/s), and the car drives clean. So the warm-up lap
initialises **the policy, not the solver** — a normal recurrent-controller
property, exactly like a driver's warm-up lap.

The **grid-start** scenario tests the real race procedure: warm-up lap, stop on
the grid, then race. It answers whether the moderate regime survives the brief
stop (it lives in the policy's memory) or is lost with the momentum. See the
table in `scenario_summary.json` / the figure for the measured answer.

## Files

| file | what |
|---|---|
| `../frozen_summary_v1/best_policy_*_mpcc_val.npz` | the online MPCC-critic networks (fast flying, need a warm-up) |
| `../frozen_summary_v1/best_policy_*_return_val.npz` | the online RETURN-critic networks |
| `../frozen_summary_v1/fitted_policy_*_kv0.50.npz` | the grid-fitted network (robust cold; the deliverable) |
| `results/scenario_summary.json` | the measured laps per policy per scenario (input to the figure) |
| `paper/figures/scenario_summary.png` | the two/three-panel figure |

## Rerun everything

    # one policy, all three scenarios, three starts:
    PYTHONPATH=/path/to/scuderia_gym_jax python3 scripts/racing_check.py \
        results/paper/frozen_summary_v1/best_policy_icra_t2_raceline_2_progress_mpcc_val.npz \
        --box-from results/paper/frozen_summary_v1/fitted_policy_icra_t2_raceline_kv0.50.npz

    # the robust deliverable, all three scenarios:
    PYTHONPATH=... python3 scripts/racing_check.py \
        results/paper/frozen_summary_v1/fitted_policy_icra_t2_raceline_kv0.50.npz

    # a cold standing start only (fast):
    PYTHONPATH=... python3 scripts/drive_policy.py <net.npz> [--box-from <fitted.npz>]

    # the crash trace (speed, solver status, lateral, per tick):
    #   scratchpad why_crash.py -- see TODO 2i for the numbers it produced

    # rebuild the figure from scenario_summary.json:
    python3 scripts/make_scenario_summary.py

All runs are deterministic given the acados build: same start, same plant, same
seed. The environment for every command:

    export ACADOS_SOURCE_DIR=$HOME/acados-v060
    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
    export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$HOME/github/scuderia_gym_jax:.

## Caveats recorded honestly

* The **flying (moving)** scenario in `racing_check.py` hands the policy the
  wheel wherever the ~1-lap warm-up ends, which for some starts is *inside* the
  hairpin — a bad handoff that crashed two otherwise-clean return-critic
  networks. Those two flying crashes are a protocol artefact, not a policy
  property, and are marked as such on the figure. The **grid-start** scenario
  fixes this by handing off at the start line. (TODO 2i.)
* A policy learned on one track geometry does not transfer to another. After any
  change to `mpcc_tuning/track.py`, re-run these before quoting the numbers.
