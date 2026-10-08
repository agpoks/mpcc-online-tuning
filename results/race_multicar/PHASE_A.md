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
4. [CODE DONE, run deferred] **Multi-car training** — `experiments/race_multi_train.py`: varied-N
   opponents/episode (N in 1..MAX_OPP, mixed pace), the 34-feat aggregate obs, MPCC avoids all detected
   (<=MAX_OPP), reward = race_reward on the fastest rival + extra pass credit, PolicyTuner + keep-best,
   both arms. Statically validated (parses/imports; both arms build the 34-feat net + forward). Needs a
   1-2 episode smoke-run on an IDLE box, then the full run (seeds) -> results/race_multicar/nets/.
5. [DONE] **Eval + artifacts** — `experiments/race_multi_eval.py`: the keep-best nets FIXED vs N=1/2/3
   (6 runs/cell). Result (see EVAL.md): flexibility holds (one net, passes scale 1.2→2.7); **LTC safer +
   more consistent** (clean 100/67/67, contact 0–33%) vs **MLP erratic/reckless** (clean 50/83/33, contact
   up to 50%) — the memory-helps-juggle-occluded-opponents signal. Fixed eval FLIPPED the training-noise
   "MLP wins" read. Artifacts: data/csv/png/pdf/tikz/gif + `--from-saved`. Remaining gap: occluded-opponent
   contact at N≥2 (next lever = a bounded last-seen keep-out during blind windows + small retrain).

## Then: phase B (real car) on its OWN new branch -- see [[phase-plan-and-branching]] / paper/deployment_plan.md.
