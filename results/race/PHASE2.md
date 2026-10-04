# Phase 2 — realistic racing: multiple opponents + partial observability

Branch: `race-phase2-multiopp-sensor` (off `race-mode`, which has the deployed single-opponent band fix).
Goal: a real race — **several opponents of mixed relative pace at once**, seen only through a
**limited sensor** (camera+lidar ~15–20 m + FOV). Enabled by the solve being ~34 ms (idle), well under
the 50 ms/20 Hz budget, so a longer horizon is NOT real-time-gated (results/race/solve_timing.md).

## A. Multiple opponents (mixed faster/slower/equal in one race)
1. **MPCC `max_obstacles = N`** — the OCP + `pack_params` already build N keep-out rows; the wrapper
   `AcadosMPCC._p` only packed slot 0. **FIXED**: `_p` now packs all N (active from `set_obstacles`,
   the rest the inactive `off = [0,0,-margin]` so `r_eff=0`). N=1 unchanged. *(step 1, done)*
2. **Features — drop the discrete class, go continuous.** A single `pol.cls` can't represent "faster
   ahead + slower behind". Use per-opponent **nearest-ahead** (overtake target) and **nearest-behind**
   (defend) slots with the continuous relative pace `diff = v_opp/v_ego` the reward already uses, plus
   `side_open` per slot. (`side_open`/`race_features` currently use `opponents[0]`.)
3. **Training** — N mixed-pace opponents per episode; reward sums per-opponent pass/contact.
4. **Eval** — multi-opponent races; LTC-vs-MLP (memory should help juggle several).

## B. Sensor range / partial observability (we don't always see the opponent)
1. **Gate**: `detected = (dist ≤ R_detect ~15–20 m) and in-FOV`; only detected opponents populate the
   features + MPCC obstacle (and the nearest-ahead/behind slots).
2. **New features**: `detected` flag + time-since-last-seen.
3. **On loss**: dead-reckon (last-seen pose + tracker speed) for a short grace (~0.5–1 s, rides through
   corner occlusion), then decay to "no opponent" → the large-gap regime the policy already drives clean.
4. **LTC memory** remembers + extrapolates across the blind window where the MLP can't → a clean
   ablation (memory wins under *both* partial observability and multi-agent).
5. **Eval the new cases**: overtake→lose-the-behind-car (stay clean); faster opponent re-closing from
   behind (react only on re-entry).

## C. Raise the ~2.5 m/s corridor ceiling (stretch)
Corridor-aware `v_ref` (slow where the corridor narrows, not just where grip runs out) + a longer
horizon (N=40 ≈ 36 ms, still under budget). Lets the car go faster than ~2.5 m/s *cleanly*.

## Sequence
1. [done] MPCC `max_obstacles=N` (`_p` fix) + multi-obstacle avoidance test.
2. Multi-opponent features (nearest-ahead/behind, continuous pace) + baseline eval of the band net vs N.
3. Multi-opponent training + reward.
4. Sensor-range gate + retrain + the two eval cases.
5. (C) corridor-aware v_ref + longer horizon, if time.

## Reproducibility
New `experiments/race_multi.py` + `tools/` additions, wired into `Makefile.race`; seeded/deterministic
like race-mode. Single-opponent canonical on `race-mode` stays untouched.
