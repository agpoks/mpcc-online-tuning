# Phase-2 sensor gate — LTC vs MLP under partial observability (4 seeds)

**Sensor model:** opponent observed only within **range 18 m AND a forward FOV cone ±60°** (`FOV_DEG`=120
total). A car outside the cone — e.g. directly behind, just overtaken — is unseen. (Range alone never
loses the opponent on this folded track: detected ~99–100%. The FOV creates the 60–75 % blind window.)

Both arms trained online under this gate (`make -f Makefile.race sensor-train`, one net per seed), then
evaluated FROZEN (deployed: no explore/learn) on the two cases. 4 seeds = 4 distinct physical starts.

## Hardened result (clean runs out of 4 seeds)
| arm | case | clean | passes | cross-track blind | cross-track seen |
|-----|------|:---:|:---:|:---:|:---:|
| LTC | overtake-then-blind (slower ahead) | **2/4** | 1.5 | 0.041 | 0.041 |
| MLP | overtake-then-blind | **1/4** | 2.0 | 0.029 | 0.024 |
| LTC | faster re-approach (faster behind) | **3/4** | 1.0 | 0.022 | 0.026 |
| MLP | faster re-approach | **3/4** | 1.0 | 0.024 | 0.027 |

## Honest reading — the single-seed advantage does NOT robustly generalise
On seed 0 alone the LTC looked like a clean win (clean 2/2, MLP off-track on overtake, and a tidy
"MLP more erratic when blind" cross-track signal). **Across 4 seeds that largely washes out:**
- **overtake-then-blind:** LTC 2/4 vs MLP 1/4 — LTC marginally better, but *both* fail more often than
  not. LTC is **bimodal** (clean on seeds 0,3; off-track on 1,2), not reliably clean.
- **faster re-approach:** 3/4 vs 3/4 — **tied**.
- The cross-track blind-vs-seen "mechanism" does **not** survive: LTC overtake is 0.041 blind ≈ 0.041
  seen (seed 2 even spikes to 0.082 on an off-track), and on faster re-approach both arms are slightly
  *calmer* when blind — no arm difference.

**Why it's noisy:** each net is trained on a *single* scenario (1 seed), and the training clean-rate
swung 25–88 % across seeds — so a per-seed eval reflects training luck more than a systematic LTC-vs-MLP
difference. The overtake-then-blind case is simply hard for both (overtake a slower car you go blind
behind). **Conclusion: under this exact setup there is no robust memory advantage** — the seed-0 result
was not representative. To make a real claim we'd need multiple training seeds *per* net (lower-variance
policies), more eval scenarios, or a setup that stresses memory harder.

## Artifacts (all in this folder, deterministic/seeded, regenerable)
- **data** `sensor_<case>_<arm>_<seed>.npz` (×16; ego+opp trajectory, detected flag, gap, weights, scalars)
- **csv** `sensor_eval.csv` (per-seed) + `sensor_eval_summary.csv` (clean-rate X/4 + means)
- **png+pdf** `fig_sensor_compare.*` (PAPER fig: clean-rate by case + reaction amplitude, LTC vs MLP),
  `fig_sensor_tracks.*` / `fig_sensor_timeline.*` (detailed, seed 0)
- **tikz** `tikz/fig_sensor_{compare,tracks,timeline}.tex` (all compile with tectonic; need `\usepgfplotslibrary{groupplots}`)
- **gif** `sensor_<case>_<arm>_0.gif` (×4, seed 0)

## Reproduce
```
make -f Makefile.race sensor-train                                   # retrain nets (use --jobs 2 for memory)
python3 experiments/race_sensor_eval.py --seeds 0 1 2 3              # eval all 4 seeds + every format
python3 experiments/race_sensor_eval.py --seeds 0 1 2 3 --from-saved # regenerate figs/csv/tikz from the .npz (no acados)
```
