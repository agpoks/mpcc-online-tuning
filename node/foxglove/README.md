# Foxglove layout — mpcc_online deploy node

`mpcc_online_layout.json` is a [Foxglove Studio](https://foxglove.dev) layout for live-monitoring the deployed
online-tuned MPCC node, in the same schema as `~/sim_ws/src/scuderia_gym_ros/foxglove/`. Import it with
**Foxglove → Layouts → Import from file…**, then connect to the robot/sim (Foxglove bridge or a ROS2 live
connection).

Regenerate with `python3 node/foxglove/make_layout.py` (options: `--drive`, `--pose`, `--odom`, `--enable`,
`--node`).

## Tabs
| tab | shows |
|-----|-------|
| **MPCC Online** | 3D pose/odom view · gauges (speed, solve time, off-track) · **online-tuned weights adapting live** · pace (vx vs command) |
| **Online tuning** | the weights `k_v/q_c/q_l/r_a` (big) · lateral error + off-track · solve time · enable latch |
| **Control** | pace + progress `s` · raw drive command · off-track gauge |

## Topics it binds to
The node (`ros2_node.py`) publishes one `std_msgs/Float32` per diagnostic under its private namespace, plus the
drive command:

| topic | meaning |
|-------|---------|
| `/mpcc_online/debug/k_v`, `/q_c`, `/q_l`, `/r_a` | **online-tuned MPCC weights** (added for monitoring; the paper's adaptation signal) |
| `/mpcc_online/debug/vx` | body-frame speed |
| `/mpcc_online/debug/solve_ms` | acados solve time (keep < 50 ms for 20 Hz) |
| `/mpcc_online/debug/lateral` | signed lateral error to the corridor centre |
| `/mpcc_online/debug/s` | arc-length progress |
| `/mpcc_online/debug/accel`, `/off` | accel command, off-track flag (0/1) |
| `/odom` | `nav_msgs/Odometry` (vx = `twist.twist.linear.x`) |
| `/pf/viz/inferred_pose` | `geometry_msgs/PoseStamped` (particle-filter pose) |
| `/mpcc/enable` | `std_msgs/Bool` drive+learn latch |
| `/ackermann_cmd` | `ackermann_msgs/AckermannDriveStamped` drive command |

**Real car:** `deploy.yaml` ships `drive_topic: /ackermann_cmd`, but `DEPLOY_CAR21.md` requires `/drive` on
car21 (the mux's autonomous input). If you set `drive_topic: /drive`, regenerate the layout with
`--drive /drive` so the Control tab points at the right topic.
