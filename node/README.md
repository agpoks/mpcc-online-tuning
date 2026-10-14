# mpcc_online — deploy node (ROS2)

Real-time **online-tuned MPCC** for the car21 / F1TENTH car, with **on-car online learning**. Reuses the
repo's acados MPCC + LTC weight-policy net + `PolicyTuner` unchanged; this folder is the real-time wrapper.
Structure mirrors `scale-mpcc` (three-layer split, enable latch, failed-solve decay, per-lap checkpoint).
Branch: `deploy-car`.

## Pieces
```
node/mpcc_online/
  controller.py    ROS-free core. OnlineMPCCController:
                     - frozen mode: net drives, LTC weight-tuning only
                     - learn=True: PolicyTuner RTRL (act/learn) + per-lap keep-best (the on-car learner)
                     - save_ckpt / load_ckpt (per-lap resume, like scale-mpcc)
  ros2_node.py     rclpy wrapper -> car21 interface (below). No learning math.
  sim_check.py        verify the frozen controller vs the plant (no ROS)
  sim_check_learn.py  verify the ONLINE-LEARNING loop vs the plant (no ROS)
node/config/deploy.yaml · launch/ · package.xml · setup.py
```

## Interface (car21 / scuderia_gym_ros — confirmed)
| dir | topic | msg | fields |
|---|---|---|---|
| in | `/pf/viz/inferred_pose` (real) or `/odom` pose (sim) | PoseStamped / Odometry | map pose; laser→CoG shift 0.0678 m |
| in | `/odom` | Odometry | BODY-frame twist: vx, vy, yaw-rate |
| in | `/mpcc/enable` | Bool | drive + learn latch |
| in | `/ackermann_cmd` | AckermannDriveStamped | mux echo (authority detection) |
| out | `/ackermann_cmd` | AckermannDriveStamped | `steering_angle`=δ; longitudinal = **acceleration** [m/s²] |

**Drive output:** our MPCC emits acceleration natively. `longitudinal_field: acceleration` puts a_x in
`drive.acceleration` (real car `ackermann_to_vesc`: `current = 8.938·a + 3.693·v`). For the **sim bridge**
set `longitudinal_field: speed` (bridge `ctrl_mode:='accl'` reads `drive.speed` as m/s²) **or** run
`mpcc_bringup/drive_adapter` in front of the bridge. Safety mirrors scale-mpcc: enable latch + pose-timeout
(0.5 s → publish nothing, mux stops the car) + failed-solve decay (accel ×0.7) + mux authority + warm-start drop.

## Progress state (important)
`s` is the MPCC's own VIRTUAL progress, integrated from its progress-rate output (`s += u2·dt`), exactly as
the training plant does — NOT `track.project(x,y)` (quantized to the point grid; using it dropped laps ~3×).
Start the run on the track start line (s=0), matching training.

## Online learning (the method)
With `learn: true` the node runs `PolicyTuner` behind the drive latch: per tick `act`/`learn` (RTRL,
clock='time', critic='return', reward = physical-progress − 5·off), and per lap `end_episode` keep-best with
`validate=False` — **banks a faster clean lap, reverts a slower/crashed one** (the on-car safety guard). The
LTC net adapts to the real track online. The banked net + bank are checkpointed (`save_ckpt`) on disable and
can be resumed (`load_ckpt`), like scale-mpcc's per-lap map save/resume.

## Verify on the dev box FIRST (no car, no ROS)
```bash
export ACADOS_SOURCE_DIR=/home/poxx/acados-v060 LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:/home/poxx/github/scuderia_gym_jax:.
python3 node/mpcc_online/sim_check.py          # frozen: 6 clean laps/5min, solve p95 ~43 ms < 50 ms
python3 node/mpcc_online/sim_check_learn.py     # learning: drives clean + RTRL + keep-best banks
```

## On the car / sim
1. acados on the machine (same env vars). `mpcc_tuning` + `scuderia_gym_jax` importable.
2. Build the package: copy/symlink `node/` as `mpcc_online` into your ROS2 ws `src/`, `colcon build`.
3. Test in sim first: `ros2 launch scuderia_gym_ros sim_std_accl.launch.py` + this node
   (`longitudinal_field: speed`, or route through `drive_adapter`). Then `ros2 topic pub /mpcc/enable ...`.
4. Edit `config/deploy.yaml`: `net_path`, `raceline_npz` (the deploy map, e.g. `try5_ev7_corridor.npz`),
   `longitudinal_field`, `steer_max`/`accel_*` to the car's limits, `learn: true/false`.

## Race mode (`mode: race`)
Frozen physical2 race net vs a field. Mirrors the validated `race_multi_eval` (LTC 100/100/67 over
N=1/2/3): per tick it ingests the detected opponents (`control(..., opponents=[{x,y,v,id,s,d}, ...])`),
picks the fastest rival + field aggregates, sets MPCC keep-outs (detected + a bounded last-seen prediction
when a car is occluded), and emits class/crowd-conditioned weights (k_v/r_a/q_c safety bands + the crowd
weave band). On the car the `opponents` list comes from perception (sim `/perception/tracks`
`TrackedObstacleArray`; real car `/obstacles` `ObstacleList`). `sim_check_race.py` verifies it against the
plant + scripted opponents. Online race-learning is a later increment — the race net runs frozen here.

## Status / TODO
- ✅ frozen qualifier + online-learning loop verified in sim (this repo, no ROS): learning banks faster laps.
- ✅ **solve p95 under learning fixed** (`skip_unused_q=True`: the return critic never reads action_value's
  Q, so skip its redundant 2nd acados solve → 1 solve/tick, p95 ~48 ms < 50 ms, learning byte-identical).
  (Turning action-dither off instead also fixed the time but ran the learner off-track ~17% — kept dither.)
- ✅ **off-track during learning no longer crashes**: a terminated tick reverts-to-best + recovers (was a
  0-d-array crash on the first off-track tick — a real deploy-safety bug). Both control paths guard NaN solves.
- ✅ **racing wired + verified in sim** (`sim_check_race.py`: N=2 → 2.63 laps, 0 off, 2/2 passes, p95 46.7 ms).
- ✅ small map `02092026-try5_ev7` extracted → `mpcc_tuning/tracks/try5_ev7_corridor.npz`.
- ⚠️ ROS node built to the confirmed sim/real interfaces but **UNTESTED on ROS** — see `SIM_WS_TEST.md`
  (sim test is possible now in `~/sim_ws`; nothing to copy from the backup for the qualifier).
- ▢ online race-learning (class-residual PolicyTuner + per-class keep-best) — not yet wired.
