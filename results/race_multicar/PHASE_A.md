# Phase A — multi-car (branch `race-multicar`, off the phase-2 sensor branch)

Goal: one online policy that races a VARIABLE number of opponents (1/2/3) of mixed pace under the real
270°+occlusion sensor, overtaking the beatable ones while managing risk — and is the policy we deploy on
the real RC car (the bunker demo has 2 opponents, so multi-car is a PREREQUISITE for deployment).

Inherits from the sensor branch: the online method (PolicyTuner: explore + RTRL + keep-best), the
`r_a`/`k_v` bands (RA_FLOOR=10 classes 1-3), the `_visible` 270°+occlusion sensor. Results go in
`results/race_multicar/` (NOT mixed with results/race_phase2/).

## Design decisions
- **Opponent-count flexibility (one net for 1/2/3 cars, no retraining):**
  1. MPCC `max_obstacles = MAX_OPP (3)`; fewer active = inactive `off` slots. Flexible to <=3.
  2. **Aggregate observation (count-invariant):** the 31-feature vector carries the **FASTEST detected
     rival** (overtake target) in its opponent slots, plus **3 appended field aggregates** —
     `[31]` crowding (mean closeness over all detected opponents, R_CROWD=8 m), `[32]` avg closing-rate,
     `[33]` opponent count / MAX_OPP. `N_RACE_FEATURES_MULTI = 34`. Implemented: `race_features(..., field=)`
     + `field_summary()` in experiments/race_mode.py (field=None keeps the 31-feat single-opponent path).
  3. **Train with VARYING N per episode** (randomise N in {1,2,3}) so the aggregates generalise across counts.
- **Objective = overtake-and-win, not stuck-in-the-middle:** chase/pass the fastest rival; the crowding +
  avg-closing terms set a risk budget. Reward = sum per-opponent (pass bonus - contact) + a finish-ahead term.

## Steps / status
1. [BLOCKED - needs idle machine] **Solve-time vs max_obstacles (1,2,3)** — does the N-obstacle QP still fit
   50 ms / 20 Hz? (`tools/solve_timing_obstacles.py`; the real-time gate + the real-car feasibility check).
   Deferred: the box is loaded by other jobs (record_states); re-run when idle. Onboard compute re-check
   before the real race.
2. [DONE] **Aggregate observation** — `race_features(field=)` + `field_summary()` (fastest + crowding +
   avg-closing + count), N_RACE_FEATURES_MULTI=34, MAX_OPP=3, R_CROWD=8. Backward-compatible.
3. [NEXT] **Baseline** — single-opponent net vs a mixed 2-3 car field under the real sensor (what the
   current policy already does before the aggregate-obs retrain). Extend experiments/race_multi.py.
4. [NEXT] **Multi-car training** — varied-N opponents/episode, aggregate obs, the multi-opponent reward;
   both arms (LTC vs MLP -- memory may finally win juggling several occluded opponents). New nets in
   results/race_multicar/nets/.
5. [NEXT] **Eval + artifacts** — vs 1/2/3 cars (flexibility), clean rate + overtakes + the field behaviour;
   data/csv/png/pdf/tikz/gif, reproducible via a `make` target + `--from-saved`.

## Then: phase B (real car) on its OWN new branch -- see [[phase-plan-and-branching]] / paper/deployment_plan.md.
