# Phase-A multi-car eval — one online net, variable field (fixed eval)

The trained keep-best nets (`nets/race_{ltc,mlp}_{0,1}.npz`) run **FIXED** (explore off, deterministic
`pol.step`, no RTRL) vs **controlled fields of N=1/2/3 opponents** under the real 270°+occlusion sensor.
6 runs per cell (3 physical start variations × 2 net-seeds). This isolates what the net learned (the
deployed method adds online keep-best adaptation on top — phase B).

## Result (clean % / passes / contact % / off %, per arm per field size)

| arm | N | clean % | passes | contact % | off % | laps |
|---|---|---|---|---|---|---|
| **ltc** | 1 | **100** | 1.17 | 0 | 0 | 2.93 |
| **ltc** | 2 | 67 | 2.17 | 17 | 17 | 2.32 |
| **ltc** | 3 | 67 | 2.67 | 33 | 0 | 2.26 |
| mlp | 1 | 50 | 0.83 | 33 | 17 | 1.92 |
| mlp | 2 | 83 | 2.67 | 0 | 17 | 2.72 |
| mlp | 3 | 33 | 3.17 | 50 | 17 | 2.38 |

## Read (honest)
- **Flexibility holds:** the ONE net races 1/2/3 cars with no retraining; passes scale with the field
  (LTC 1.2→2.2→2.7, MLP 0.8→2.7→3.2).
- **LTC is the safer, more consistent net:** clean 100/67/67 across N (graceful degradation), 0–33% contact.
  **MLP is erratic** (clean 50/83/33) and trades safety for passes — 33% contact even at N=1 (it hits a
  single car) and **50% contact at N=3**. This is the memory-helps-juggle-occluded-opponents signal the
  single-opponent tie did not show.
- **The fixed eval flips the training-noise read.** The training last-8 (with exploration) looked like
  "MLP more consistent"; deterministic eval shows the opposite. Measure before asserting.
- **Remaining gap (both arms): contact with OCCLUDED opponents at N≥2** (LTC 17–33%, MLP up to 50%). The
  MPCC drops a blind opponent from its keep-out set → it can plan through where an unseen car actually is.
  n=6/cell is coarse (±17% steps) — the ranking is clear, the exact rates are indicative.

## Next lever (not yet done): cut the occluded-opponent contact
A **bounded** last-seen estimate — propagate a tracked opponent at its observed speed along the racing line
for a short horizon and keep its MPCC keep-out during the blind window, decaying after ~1–2 s — targets the
contact directly, without the over-swerve of the earlier growing-radius dead-reckon (which we stripped).
Needs a small retrain. See [[offtrack-corridor-buyoff-and-equal-overtake-trilemma]].

## Reproduce (deterministic/seeded)
    # full eval (needs the trained nets in nets/):
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_multi_eval.py --jobs 4 --steps 2600
    # plots/csv/tikz/gif from saved data only:
    python3 experiments/race_multi_eval.py --from-saved

Artifacts: `multi_eval_rows.npz` (scalars) · `multi_eval.csv` / `multi_eval_summary.csv` ·
`multi_flexibility.{png,pdf}` · `tikz/multi_flexibility_{clean,passes}.tex` · `multi_race_{ltc,mlp}.gif`
(filled square = opponent seen, hollow = blind). Nets + training: `TRAIN_SUMMARY.md`, `train_full.log`.
