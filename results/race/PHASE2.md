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

## Decisions (2026-10-04)
- **Order: sensor-range gate FIRST** (single-opponent), then multi-opponent. Results go to a NEW
  folder **`results/race_phase2/`** (do NOT mix with `results/race/`).
- **Multi-opponent objective = OVERTAKE AND WIN, not just survive.** Nearest-ahead + nearest-behind
  relative pace risks the policy getting **stuck in the middle and never overtaking**. Instead encode
  the field as: (a) the **FASTEST of all others** = the rival to beat / the target; (b) a **RISK that
  depends on the AVERAGE over all others** (crowding/aggression budget). So the obs is a *comparison of
  the whole field* (fastest + average), not just the two neighbours — this pushes the policy to pick off
  the fastest and manage risk by how crowded the field is, rather than parking between two cars.
- Sensor gate baseline result (single-opponent, band net vs N): overtakes all 3 but off at 5.63 laps
  (multi-car navigation, not over-speed) -- `experiments/race_multi.py`, states_multi_ltc_0.npz.
- **Sensor model = forward FOV + range (user-confirmed 2026-10-04).** A RANGE-only gate (18 m euclidean)
  loses the opponent ~0% of the time on this folded track (detected ~99-100%), so "don't see the car we
  passed" never happens. Added a forward-FOV cone (`FOV_DEG`=120 total, i.e. +-60 deg of heading) to the
  gate (`_visible()` in race_mode): a car outside the cone (e.g. directly behind, just overtaken) is
  unseen even in range. Post-hoc on the saved trajectories this flips detected to 34% (overtake case) /
  17% (faster-reapproach case) -> 66-83% BLIND. This is the regime where LTC memory should beat MLP.
- **Retrain under range+FOV, both ltc AND mlp** (the memory ablation), to `results/race_phase2/`.
  The first phase-2 net (`race_ltc_0`) was range-only (~always-seen) -> superseded.

## Reproducibility
New `experiments/race_multi.py` + a sensor-gated training path + `tools/` additions; seeded/deterministic
like race-mode; outputs in `results/race_phase2/`. Single-opponent canonical on `race-mode` untouched.
