# Phase-2 sensor gate — LTC vs MLP under partial observability (robust training, 4 seeds)

**Sensor model:** opponent observed only within **range 18 m AND a forward FOV cone ±60°** (`FOV_DEG`=120
total). A car outside the cone — e.g. directly behind, just overtaken — is unseen. (Range alone never
loses the opponent on this folded track; the FOV creates the 60–75 % blind window.)

**Training:** each net trained online under the gate over **varied physical starts** (s0/v0 cycle all 4
corners per episode, decorrelated from the opponent class) — lower-variance policies, not overfit to one
scenario — 18 episodes, one net per seed per arm. Evaluated FROZEN on the two cases, 4 seeds = 4 starts.

## Result (clean runs out of 4 seeds)
| arm | case | clean | passes | cross-track blind | cross-track seen |
|-----|------|:---:|:---:|:---:|:---:|
| LTC | overtake-then-blind (slower ahead) | **0/4** | 2.25 | 0.023 | 0.048 |
| MLP | overtake-then-blind | **2/4** | 2.00 | 0.020 | 0.019 |
| LTC | faster re-approach (faster behind) | **2/4** | 0.75 | 0.028 | 0.026 |
| MLP | faster re-approach | **3/4** | 1.00 | 0.033 | 0.036 |

## Conclusion — no LTC/memory advantage; if anything the MLP is more reliable
With the lower-variance (varied-start) training the user asked for, the picture is now **clear and it
refutes the memory hypothesis**: the MLP matches or beats the LTC on *both* cases —
**overtake-then-blind MLP 2/4 vs LTC 0/4**, **faster-re-approach MLP 3/4 vs LTC 2/4**. The recurrent
state gives no benefit under this partial-observability setup, and the LTC is actually *worse* on the
overtake case (fails all 4). The earlier seed-0 "LTC wins" and the first 4-seed "LTC marginally ahead"
were both noise from single-start, high-variance training — removing that variance flips/erases the gap.

**Net:** memory does not help here. The deployed single-opponent policy ([[faster-10lap-walllout-is-overdriven-ra]]
band fix) should stay the paper's result; this partial-observability ablation is an **honest negative** —
LTC ≈ MLP (MLP slightly better), so there is no memory story to tell for this setup. overtake-then-blind
(overtake a slower car you then go blind behind) is hard for *both* — the open problem is the plan/
constraint side under blindness, not the observation memory.

## Artifacts (all here, deterministic/seeded, regenerable)
- **data** `sensor_<case>_<arm>_<seed>.npz` (×16; ego+opp trajectory, detected flag, gap, weights, scalars)
- **csv** `sensor_eval.csv` (per-seed) + `sensor_eval_summary.csv` (clean-rate X/4 + means)
- **png+pdf** `fig_sensor_compare.*` (PAPER fig: clean-rate by case + reaction amplitude, LTC vs MLP),
  `fig_sensor_tracks.*` / `fig_sensor_timeline.*` (detailed, seed 0)
- **tikz** `tikz/fig_sensor_{compare,tracks,timeline}.tex` (all compile; need `\usepgfplotslibrary{groupplots}`)
- **gif** `sensor_<case>_<arm>_0.gif` (×4, seed 0)

## Reproduce
```
make -f Makefile.race sensor-train                                   # varied-start training (use --jobs 2 for memory)
python3 experiments/race_sensor_eval.py --seeds 0 1 2 3              # eval all 4 seeds + every format
python3 experiments/race_sensor_eval.py --seeds 0 1 2 3 --from-saved # regenerate figs/csv/tikz from the .npz (no acados)
```
