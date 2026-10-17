# Deploy status — branch `deploy-car`

A snapshot of where the real-car deployment of the online-tuned MPCC stands. This branch is **standalone**
(not merged into `main`); it holds the ROS2 node, the deploy map/corridor/net, and the car21 bring-up docs.
Last updated **2026-10-18**.

## TL;DR
- **Deploy map: `lab_test_ev8`** (single figure-8, 27 m corridor). This is the one deploy target — `try5_ev7`
  was dropped (see [Decisions](#decisions)).
- **Done & reproducible**: trained net, raceline-referenced corridor, 16 clean laps / 5 min in sim, verified
  end-to-end through the ROS node. Results + reproduction in [`results/deploy_labtest/RESULTS.md`](results/deploy_labtest/RESULTS.md).
- **What still gates a real drive**: three hardware-side checks at the car (below). No code work is pending.

## The package
`node/` is the ament_python package **`mpcc_online`** — a drop-in replacement for scale-mpcc's `mpcc_node`
(same acados MPCC family + car21 I/O), plus optional on-car online weight-learning and a race mode.

| file | what |
|------|------|
| [`node/DEPLOY_CAR21.md`](node/DEPLOY_CAR21.md) | copy→build→launch on car21; the drive chain; the one config step to get right |
| [`node/SIM_WS_TEST.md`](node/SIM_WS_TEST.md) | sim validation recipe (`~/sim_ws`, ROS2 Humble) + copy-from-backup list |
| [`node/launch/mpcc_online.launch.py`](node/launch/mpcc_online.launch.py) | self-contained launch (sets acados + repo on PYTHONPATH) |
| [`node/config/deploy.yaml`](node/config/deploy.yaml) | params (net, corridor, kv_launch, limits, topics) |
| [`node/foxglove/`](node/foxglove/) | live-monitoring Foxglove layout, modelled on the real car21 race dashboard |
| [`mpcc_tuning/tracks/lab_test_ev8_corridor.npz`](mpcc_tuning/tracks/lab_test_ev8_corridor.npz) | the deploy corridor (raceline-referenced) |
| `results/deploy_labtest/best_policy_icra_t2_smooth_2_time_return.npz` | the shipping net (seed 2; `deploy.yaml` default) |

## Verified
- **Offline (ScuderiaPlant)**: qualifier 16 clean laps / 5 min, fastest 14.6 s, 1.53 m/s, 0 crash; solve p95 ≤ 48 ms.
- **ROS end-to-end (`~/sim_ws`, Humble)**: the exact deploy config (seed-2 net + `lab_test_ev8_corridor.npz`,
  spawned at corridor s=0) builds, comes up, subscribes `/odom`, publishes the drive command + all debug topics
  incl. the online-tuned weights (`k_v`, `q_c`, `q_l`, `r_a`), **0 % off-track**. Pace in that run was not
  assessable (box under heavy parallel load → solve ~131 ms → ~7 Hz); re-run idle or on the car's CPU for pace.
- **Longitudinal units (A vs m/s²) — RESOLVED** against the car21 backup (`ackermann_to_vesc.cpp`, `vesc.yaml`,
  `ackermann_mux.toml`): the car's `ackermann_to_vesc` reads `drive.speed` as a sentinel — `+inf` ⇒
  `drive.acceleration` is an **a_x reference [m/s²]** tracked by a closed-loop PID → current. **Our node already
  emits `speed=+inf` + accel in m/s²**, so the output is correct as-is; no accel→current map needed. Only keep
  the car's `use_current_control:=true` (its default). Details in [`node/DEPLOY_CAR21.md`](node/DEPLOY_CAR21.md) item 2.

## Online learning
seed 2 is itself online-learned (PolicyTuner RTRL + per-lap keep-best); "frozen" is just the deploy phase. The
demonstration (keep-best ratchet + a live-drive run) and the paper TikZ figures live in `results/deploy_labtest/`.
**Banking lesson**: on-car keep-best must *frozen-validate* a lap before banking a net you intend to freeze —
direct-banking a dithered learning lap captures exploration flukes (one such fluke net is committed only to
document that it is NOT deployable). Continuous learning stays valuable as the on-car adaptation/safety layer
(every off-track tick in the live run was recovered by a keep-best revert; 0 terminal crashes).

## Still gates a real drive (hardware-side, at the car)
- [ ] Set **`drive_topic:=/drive`** on car21 (mux autonomous input, deadman button 4) — the default is
      `/ackermann_cmd`, correct for the sim but wrong for the car.
- [ ] Confirm the live car still has **`use_current_control=true`** (`ros2 param get /ackermann_to_vesc_node use_current_control`).
- [ ] Measure **solve time on the car's own CPU** (≤ 50 ms p95 for 20 Hz) — race solve is heavier; measure idle.

See [`node/DEPLOY_CAR21.md`](node/DEPLOY_CAR21.md) for the full deploy-safety checklist.

## Decisions
- **`try5_ev7` dropped (2026-10-18): `lab_test_ev8` is enough.** If ev7 is ever needed, it requires **no optimised
  raceline** — the deploy's speed reference is **curvature-derived** (`acados_ocp.py` builds it from
  `track_speed_profile(...)` whenever `use_track_vref` is off, which is how every deployed net is built; the
  optimiser speed column is used only when `use_track_vref=True`, not set at deploy). A smoothed **centerline**
  therefore suffices — the raceline would only buy lap time. ev7 is a single loop (no figure-8 crossover), so a
  distance-transform centerline would be clean; it would need a small `build_corridor_from_map.py` + light SLAM
  cleanup (close wall gaps, mask unknown). Not built.
