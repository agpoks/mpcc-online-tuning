# lab_test_ev8 deploy qualifier — corridor fix + trained net (2026-10-16)

Deploy map `lab_test_ev8` (figure-8, 27 m, SLAM-mapped walls). Branch `deploy-car`.

## 1. The corridor was the bug (open ends), not the net
The first `lab_test_ev8_corridor.npz` (commit a0d0cb3) built its boundaries by RAY-CASTING the SLAM map walls
from a RECOVERED centre. Two failures: where a map wall has a HOLE the cast saturated to the cap and the
boundary ran out into free space (an OPEN END); at the figure-8 CROSSOVER the XY-nearest wall is the wrong
loop. 10 open-end clusters (worst wrapping through the start s=0 + the crossover), and the recovered centre
sat up to **0.855 m** off the raceline. The frozen icra qualifier drove straight off it from every start
(best 5.85 m of 27) — and this was **not** fixable by k_v (inert), a speed cap (a 1.0 m/s crawl also left),
or the start pose. See `corridor_openends_BEFORE.png`.

**Fix:** build the corridor RACELINE-REFERENCED, like `Track.icra_t2_smooth` (the representation the net
trains on): centre = the raceline, per-side half-widths = the optimiser's own `w_left_m`/`w_right_m` from the
WORLD csv (arc-length parameterised → correct at the crossover, hole-free). No ray-cast, no centre recovery
(`tools/build_corridor_from_world_csv.py`, committed 14a45e7). Regenerated corridor: centre + both edges
**100% on the free map, 0 open ends, 0 boundary jumps**, 27.0 m, corridor 0.39–2.27 m. See
`corridor_fixed_AFTER.png`.

## 2. The frozen icra net transfers on the fixed corridor
No retraining needed to WORK: the frozen icra qualifier drives **4.47 clean laps** from s=0 on the fixed
corridor (`frozen_transfer_drive.png`). The net always transferred; it needed a correct corridor.

## 3. Trained a faster net ON the fixed corridor (for pace)
3 seeds, keep-best + return critic + two-layer corridor, pinned to the start line (`--start-s0 0`, because
the figure-8's mid-track spawns are infeasible). High seed variance as usual; seed 2 wins.

### Head-to-head (committed corridor, frozen, s=0, v0=1.0, kv_launch=0.45, 1200 steps) — `compare_nets.png`
| net | clean | laps | mean speed |
|-----|:-----:|-----:|-----------:|
| **labtest seed 2 (deploy default)** | ✅ | **3.45** | **1.55 m/s** |
| icra-frozen (fallback) | ✅ | 2.61 | 1.18 m/s |
| labtest seed 0 | ✅ | 2.48 | 1.13 m/s |
| labtest seed 1 | ❌ off | 2.22 | 1.36 (over-drives) |

### 5-min continuous hardening drive (6000 steps = 300 s) — `labtest_5min.{json,png}`
| net | full laps | fastest lap | mean speed | clean | solve p50/p95 |
|-----|----------:|------------:|-----------:|:-----:|:-------------:|
| **seed 2 (deploy default)** | **16** | **14.6 s** | **1.53 m/s** | ✅ | 41 / 48 ms |
| icra-frozen (fallback) | 12 | 20.2 s | 1.16 m/s | ✅ | 41 / 48 ms |

Seed 2 is ~33% more laps and a 28%-faster fastest lap, clean over the full 5 min, within the 20 Hz budget.
**`node/config/deploy.yaml` defaults `net_path` to seed 2**, with the frozen icra net documented as the
conservative first-run fallback.

## Deploy nets
- **`best_policy_icra_t2_smooth_2_time_return.npz`** — deploy default (faster, clean).
- `best_policy_icra_t2_smooth_0_time_return.npz` — clean, slower (backup).
- `best_policy_icra_t2_smooth_1_time_return.npz` — kept for the record; over-drives, do not deploy.

## Reproduce
```bash
# env: ACADOS_SOURCE_DIR=/home/poxx/acados-v060  LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib  OMP_NUM_THREADS=1
#      PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:../scuderia_gym_jax:.

# (a) build the corridor from the optimiser WORLD csv (raceline-referenced)
python3 tools/build_corridor_from_world_csv.py \
    mpcc_tuning/tracks/LABTESTEV8_v080_raceline_WORLD.csv \
    mpcc_tuning/tracks/lab_test_ev8_corridor.npz --margin 0.05 --map lab_test_ev8.yaml

# (b) train 3 seeds on the fixed corridor (nets -> results/deploy_labtest/)
python3 experiments/online_from_baseline.py --tracks icra_t2_smooth \
    --geom-narrow mpcc_tuning/tracks/lab_test_ev8_corridor.npz \
    --seeds 3 --episodes 15 --conditions tuner --keep-best --clock time --critic return \
    --theta-explore 0.15 --two-layer --grad native --jobs 3 --eval-episodes 3 \
    --start-s0 0 --out qual_labtest.json --out-dir results/deploy_labtest

# (c) head-to-head + 5-min hardening (scripts saved here)
python3 results/deploy_labtest/compare_nets.py  <scratch_out_dir>
python3 results/deploy_labtest/harden_5min.py
```
