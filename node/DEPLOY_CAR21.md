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

## Bring-up (once the two ⚠️ items are resolved)

```bash
# 0. prep: convert the car's raceline CSV -> our corridor npz (like try5_ev7_corridor.npz), set
#    deploy.yaml raceline_npz + track_name to it; set drive_topic:/drive, longitudinal per item 2.
# 1. car stack (unchanged from car21_ws) WITHOUT scale-mpcc's controller:
ros2 launch racecar_launches localization.launch.py map:=<car map>     # + joystick/mux/VESC/PF
# (or copy scale-mpcc car21.launch.py and replace its `mpcc` Node with `mpcc_online`)
# 2. our controller (acados env + this repo on PYTHONPATH, same as the dev box):
ros2 run mpcc_online mpcc_online --ros-args --params-file node/config/deploy.yaml \
  -p drive_topic:=/drive -p pose_source:=pose_topic -p autostart:=false \
  -p raceline_npz:=<car raceline corridor npz> -p kv_launch:=0.50
# 3. set the initial pose in Foxglove/RViz on the start line, then HOLD the deadman (button 4)
#    and latch us on:
ros2 topic pub -r 2 /mpcc/enable std_msgs/Bool "{data: true}"
```

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
