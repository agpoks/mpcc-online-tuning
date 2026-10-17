# Foxglove layout — mpcc_online deploy node

`mpcc_online_layout.json` is a [Foxglove Studio](https://foxglove.dev) layout for live-monitoring the deployed
online-tuned MPCC node. Import it with **Foxglove → Layouts → Import from file…**, then connect to the
robot/sim (Foxglove bridge or a ROS2 live connection). On car21 the bridge is `ros2 launch racecar_launches
foxglove.launch.py` → `ws://<car-ip>:8765` (see `car21_ws/README_FOXGLOVE.md`).

**Modelled on the real car21 race dashboard** `car21_ws/foxglove_dashboards/race_dashboard_mpc_ax.json` (from
the car7nuc backup): same studio-layout schema, same panel vocabulary (a `Parameters | Tab` top split; `Plot`
with a floating legend; `Indicator` with background status rules; a `3D` view with a grid + green pose arrow +
scan + a fading car-trail marker; `StateTransitions`; a `Log` filtered to our node; `Publish` buttons), and the
same two reusable user scripts — the **car-trajectory trail** and the **sideslip-beta** derivation — carried
over verbatim and rebound to our topics. The scale-mpcc-only panels (slicer lateral/angular error, `mpc_diag`
cost breakdown, disparity-extender sectors, the raceline-viz publisher, lap-timer feeds) are dropped rather
than bound to dead topics. Our **online-tuned-weights** plot is the analog of the real dashboard's MPC-sectors
Q-weights tab.

Regenerate with `python3 node/foxglove/make_layout.py` (options: `--drive`, `--pose`, `--odom`, `--enable`,
`--obstacles`, `--node`).

## Tabs
| tab | shows |
|-----|-------|
| **Setup** | `Parameters` (expand `/mpcc_online` to tune net_path/kv_launch/mode/learn live) · `Publish` buttons to latch `/mpcc/enable` on/off and to clear the trail |
| **Overview** | 3D (map · pose arrow · scan · fading trail) · speed (vx) · drive command (accel + steer) · off-track & lateral status indicators |
| **Online tuning** | the weights `k_v/q_c/q_l/r_a` **adapting live** (big) · progress `s` · lateral+off · solve time · sideslip β |
| **Health** | solve-time plot · enable/off-track `StateTransitions` timeline · `/rosout` log (mpcc_online) · raw drive + enable |
| **Obstacles (race)** | 3D with the obstacle topic · raw obstacle messages |

## User scripts (vendored from the car21 dashboard, rebound)
| script | input → output | note |
|--------|----------------|------|
| **Car Trajectory Trail** | `--pose` → `/mpcc_online/viz/car_trail` | fading LINE_STRIP of pose history; clear via the `Publish` button on `/mpcc_online/viz/trail_clear` |
| **Sideslip Beta Angle** | `--odom` → `/mpcc_online/diag/beta_angle` | `atan2(vy, vx)`; only meaningful if the odom source estimates `vy` (VESC wheel odometry has `vy=0`) |

## Topics it binds to
The node (`ros2_node.py`) publishes one `std_msgs/Float32` per diagnostic under its private namespace, plus the
drive command:

| topic | meaning |
|-------|---------|
| `/mpcc_online/debug/k_v`, `/q_c`, `/q_l`, `/r_a` | **online-tuned MPCC weights** (the paper's adaptation signal) |
| `/mpcc_online/debug/vx` | body-frame speed |
| `/mpcc_online/debug/solve_ms` | acados solve time (keep < 50 ms for 20 Hz) |
| `/mpcc_online/debug/lateral` | signed lateral error to the corridor centre |
| `/mpcc_online/debug/s` | arc-length progress |
| `/mpcc_online/debug/accel`, `/off` | accel command, off-track flag (0/1) |
| `/odom` | `nav_msgs/Odometry` (vx = `twist.twist.linear.x`, vy = `linear.y` for β) |
| `/pf/viz/inferred_pose` | `geometry_msgs/PoseStamped` (particle-filter pose; feeds the 3D view + trail) |
| `/mpcc/enable` | `std_msgs/Bool` drive+learn latch (Publish buttons on the Setup tab) |
| `/ackermann_cmd` | `ackermann_msgs/AckermannDriveStamped` drive command |
| `/obstacles` | obstacle topic (race mode; `--obstacles` to rebind) |

**Real car:** `deploy.yaml` ships `drive_topic: /ackermann_cmd`, but `DEPLOY_CAR21.md` requires `/drive` on
car21 (the mux's autonomous input). If you set `drive_topic: /drive`, regenerate with `--drive /drive` so the
Overview/Health tabs point at the right topic.
