# Deploying `mpcc_online` on **car21** (F1TENTH / scale-mpcc hardware)

How to run this node on the real car, what it needs, and the two things that **must** be resolved first.
Grounded in the real `scale-mpcc/ros2/mpcc_node` (`car21.launch.py`, `node.py`) — the stack this node replaces.

---

## ⚠️ MUST RESOLVE before the first drive (both are safety-relevant)

**1. Drive topic — publish to `/drive`, NOT `/ackermann_cmd`.**
On car21 the chain is: controller → **`/drive`** → `ackermann_mux_ng` (joystick deadman, button 4) → **`/ackermann_cmd`** (mux output) → `ackermann_to_vesc` → VESC. scale-mpcc's node publishes `/drive` and only *listens* to `/ackermann_cmd` for authority. **Our `deploy.yaml` currently sets `drive_topic: /ackermann_cmd`** — on car21 that would publish straight to the mux *output*, bypassing the joystick deadman. For car21 set:
```
drive_topic:   "/drive"          # mux autonomous input (deadman-gated)
mux_cmd_topic: "/ackermann_cmd"  # mux output, for authority/override tracking (already correct)
```

**2. Longitudinal units — Amps vs m/s². THIS is why the car would crawl if left as-is.**
scale-mpcc's node puts **motor current [A]** in `drive.acceleration` (`speed = -inf`): it computes the current itself (`current ≈ 8.938·a + 3.693·v`, plus a traction cap) and `ackermann_to_vesc` passes that field through as current. **Our node puts raw acceleration [m/s²] in `drive.acceleration`** (e.g. 0.15) — if car21's `ackermann_to_vesc` treats that field as **current**, 0.15 A ≈ a parked car. Before driving, confirm which `ackermann_to_vesc` expects, then either:
- **(a)** keep our accel output and configure/confirm `ackermann_to_vesc` does `current = Ca·a + Cv·v` (then our output is correct), **or**
- **(b)** add the `accel → current` map (the scale-mpcc formula + traction cap) to `ros2_node._publish` so we emit current like scale-mpcc does.
This is the single highest-risk item — it is a units mismatch, not a tuning knob. (Resolving it needs `ackermann_to_vesc`'s config from `car21_ws` — in the backup.)

---

## What this node is (and is NOT)

- **Is:** a drop-in replacement for scale-mpcc's `mpcc_node` — same acados MPCC family, same car21 I/O, **plus** optional on-car online weight-learning (`learn:true`, qualify mode) and a race mode vs detected opponents.
- **Bakes its own raceline from a file** (`track_name` / `raceline_npz`) at startup. It does **NOT** subscribe `/raceline_name`, so **`raceline_manager` is not required** (and live raceline-switching is not supported — restart to change the line). scale-mpcc followed `/raceline_name`; we don't.
- Progress `s` is the MPCC's own virtual progress (`s += u₂·dt`), **not** a map projection. **Start the car on the raceline start line (s=0).**

## States & inputs (the controller core)

| | vector | notes |
|---|---|---|
| MPCC state (8) | `[x, y, ψ, vₓ, s, v_y, r, δ]` | map pose + **body-frame** long/lat velocity + yaw-rate + steering angle; `s` = virtual progress |
| MPCC input (3) | `[δ_cmd, a_cmd, ṡ]` | the car actuates **δ + a** only; `ṡ` is internal (we integrate it; the car measures s from localization) |
| tuned weights (9) | `q_c, q_l, q_v, r_d, r_a, r_dv, d_obs, k_v, d_bound` | the LTC net outputs these per tick; `k_v` is the bounded grip fraction (cold-start cap `kv_launch`) |

## ROS interface (set these in `config/deploy.yaml`)

| dir | topic (car21) | msg | our param | field semantics |
|---|---|---|---|---|
| in | `/pf/viz/inferred_pose` | `PoseStamped` | `pose_source: pose_topic`, `pose_topic` | PF pose in `map`; laser→CoG shift `pose_forward_offset` (0.0678 m) |
| in | `/odom` | `nav_msgs/Odometry` | `odom_topic` | **body-frame** twist: `twist.linear.{x,y}`, `twist.angular.z`. **Verify this is body-frame vₓ on car21** (scale-mpcc also fuses VESC speed + IMU; we trust `/odom` twist only). |
| in | `/mpcc/enable` | `Bool` | `enable_topic` | drive + learn latch (with `autostart:false`) |
| in | `/ackermann_cmd` | `AckermannDriveStamped` | `mux_cmd_topic` | mux **output** echo → authority/override detection (drops warm-start when someone else drives) |
| in *(race)* | `/obstacles` | `bcl_interfaces/ObstacleList` | `obstacle_topic`, `obstacle_msg: bcl` | points + id, **no velocity** → our tracker estimates speed |
| out | **`/drive`** | `AckermannDriveStamped` | `drive_topic` | `steering_angle = δ`; longitudinal per item 2 above (`longitudinal_field: acceleration`) |

## Nodes to bring up on car21

Reuse scale-mpcc's `car21.launch.py` infra; it starts everything from `car21_ws` **unchanged** and you only swap its `mpcc` node for ours.

| node (pkg) | provides | needed? |
|---|---|---|
| **`localization.launch.py`** (`racecar_launches`) | lidar, VESC, `ackermann_to_vesc`, joystick, **`ackermann_mux_ng`**, **particle_filter** (`/pf/viz/inferred_pose`, `/pf/pose/odom`), map_server, Foxglove | **required** — pose + drive path + deadman |
| **`mpcc_online`** (this pkg) | the controller | **required** (replaces scale-mpcc `mpcc_node`) |
| `supreme_obstacle_detection` | `/obstacles` (`bcl_interfaces/ObstacleList`) | race only |
| `drive_watchdog` | warns when `/drive` goes silent | recommended |
| `timekeeping_node` | lap times (remap `odom`→`/pf/pose/odom`) | optional |
| `raceline_manager` | `/raceline_name` live raceline switching | **NOT needed** (we bake the raceline) |
| `supreme_mpc`, `raceline_slicer/switcher`, FTG backup | — | **NOT started** (never run a 2nd `/drive` publisher) |

## Install on car21 (copy → build → one-time)

**1. What to copy.** The node reuses the repo's `mpcc_tuning` + acados directly; there are **no sim deps**
(`scuderia_gym_jax` is sim-only and NOT needed — verified: the deploy controller imports without it). Copy the
**whole repo** to the car, e.g. `~/mpcc-online-tuning`, including the trained net:
```bash
rsync -a <devbox>:~/github/mpcc-online-tuning/ car21:~/mpcc-online-tuning/
#   minimum set if trimming: node/  mpcc_tuning/ (incl. tracks/*_corridor.npz)  experiments/ (race mode)
#   results/deploy_labtest/best_policy_icra_t2_smooth_2_time_return.npz   node/config/deploy.yaml
```

**2. acados on the car.** car21 (NUC, x86) reuses a prebuilt acados tree or builds it once (docs.acados.org).
Note its root as `ACADOS_SOURCE_DIR` (it has `lib/` and `interfaces/acados_template`). A C compiler (`cc`) must
be present — the node codegens + compiles its solver on the first launch (~90 s, once).

**3. Make it a ROS2 package.** `node/` **is** the `ament_python` package `mpcc_online`. Symlink it into a colcon
workspace and build (pure-python install, ~2 s):
```bash
ln -s ~/mpcc-online-tuning/node ~/car21_ws/src/mpcc_online          # or copy
cd ~/car21_ws && colcon build --packages-select mpcc_online && source install/setup.bash
```
colcon installs only the `mpcc_online` package + its `config/` + `launch/`; `mpcc_tuning`/acados stay in the repo
and are put on `PYTHONPATH` by the launch file (next section) via `repo_dir` / `acados_dir`.

## Bring-up (once the two ⚠️ items are resolved)

For **lab_test_ev8** the corridor is already built (`mpcc_tuning/tracks/lab_test_ev8_corridor.npz`) and
`deploy.yaml` already points `raceline_npz` at it with the faster trained net as `net_path`. For a different map,
first convert its raceline:
`python3 tools/build_corridor_from_world_csv.py <raceline_WORLD.csv> <out_corridor.npz> --margin 0.05 --map <car_map.yaml>`.

```bash
# 1. car stack (unchanged from car21_ws) WITHOUT scale-mpcc's controller:
ros2 launch racecar_launches localization.launch.py map:=<car map>     # joystick/mux/VESC/PF/map_server
# (or copy scale-mpcc car21.launch.py and replace its `mpcc` Node with `mpcc_online`)

# 2. our controller via the launch file — it sets acados + the repo on PYTHONPATH for the node process,
#    so prefer this over a bare `ros2 run` (which needs you to export PYTHONPATH/LD_LIBRARY_PATH yourself):
ros2 launch mpcc_online mpcc_online.launch.py \
  repo_dir:=$HOME/mpcc-online-tuning acados_dir:=$ACADOS_SOURCE_DIR \
  drive_topic:=/drive \
  raceline_npz:=mpcc_tuning/tracks/lab_test_ev8_corridor.npz \
  learn:=true
#   first launch compiles the acados solver (~90 s) then prints "mpcc_online up."
#   (repo_dir/acados_dir also read from env MPCC_REPO / ACADOS_SOURCE_DIR if you omit them. Longitudinal units
#    and kv_launch come from deploy.yaml — edit it per ⚠️ item 2.)

# 3. open node/foxglove/mpcc_online_layout.json, set the initial pose on the start line (s=0),
#    then HOLD the deadman (button 4) and latch us on:
ros2 topic pub -r 2 /mpcc/enable std_msgs/Bool "{data: true}"
```

**Live monitoring:** import `node/foxglove/mpcc_online_layout.json` in Foxglove Studio (its README lists the
topics). The node publishes `/mpcc_online/debug/{vx,solve_ms,lateral,s,accel,off}` and the online-tuned weights
`/mpcc_online/debug/{k_v,q_c,q_l,r_a}` as `std_msgs/Float32` — watch `solve_ms` (< 50 ms), `off`, and the
weights adapting when `learn:true`.

**Sim-verified (2026-10-17, `~/sim_ws` ROS2 Humble, lab_test_ev8 map).** The exact deploy config — the seed-2
net + `lab_test_ev8_corridor.npz`, spawned at the corridor s=0 — ran end-to-end through this node: builds,
comes up, subscribes `/odom`, publishes `/ackermann_cmd` (gym steps) and all debug topics including the new
weight topics (`k_v≈0.49` cold-start-capped, `q_c≈4.8`, `q_l≈209`, `r_a≈6.4`), **0 % off-track**. Pace was
NOT assessed in that run: the dev box was under a heavy parallel workload, so solve was ~131 ms p50 → the node
ran at ~7 Hz (not 20) and the car only crawled. Idle, this config solves ~40 ms (the icra ROS test hit p95
40 ms, 20 Hz, 2+ clean laps); re-run on an idle box, or on the car's own CPU, to judge pace. This confirms the
integration; **the solve-time-on-onboard-compute checklist item below is the one that still gates real driving.**

## Deploy-safety checklist

- [ ] **Drive topic = `/drive`** (deadman-gated), not `/ackermann_cmd`.
- [ ] **Longitudinal units resolved** (A vs m/s²) — confirm `ackermann_to_vesc`, add accel→current if needed.
- [ ] `/odom` twist is **body-frame** vₓ on car21 (else velocity feedback is wrong → bad speed tracking).
- [ ] Raceline CSV → corridor npz, geometry matches the car's map; **start on s=0**.
- [ ] `kv_launch ≤ 0.50` for the cold first lap (grip over-claim guard); `steer_max`/`accel_*` = car limits.
- [ ] **Solve time on the car's onboard compute** ≤ 50 ms p95 (dev box: qualify p95 38 ms idle; **race solve is heavier — measure it on the car's CPU, idle, before trusting 20 Hz**).
- [ ] `autostart:false` → the car moves only while the deadman is held **and** `/mpcc/enable` is true.
- [ ] Never run a second `/drive` publisher (supreme_mpc / slicer) alongside.

See `SIM_WS_TEST.md` for the sim validation and the full copy-from-backup list.
