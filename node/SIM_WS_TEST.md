# Testing `mpcc_online` in the WSL simulator (`~/sim_ws`)

**Yes — the node can be tested entirely in `~/sim_ws` with no hardware and (for the qualifier) nothing
copied from the `car7nuc` backup.** `sim_ws` is already a complete, colcon-built MPCC sim: the
`scuderia_gym_ros` `gym_bridge` *is* the simulator (it has built-in opponents), `nav2_map_server` gives
`/map`, pose comes from the bridge's ground-truth TF/`/odom`, `mpcc_bringup/drive_adapter` and the
opponent perception stack are present. The backup is only needed for the **real car** (see the last
section).

ROS2 Humble is installed and `sim_ws/{build,install}` exist, so this runs today.

---

## The sim interface (confirmed against `scuderia_gym_ros/gym_bridge.py`)

| what | sim topic / msg | our node param |
|---|---|---|
| drive OUT | `/ackermann_cmd` `AckermannDriveStamped` — bridge reads **`drive.speed`** + `drive.steering_angle` only | `drive_topic:/ackermann_cmd`, **`longitudinal_field: speed`** |
| accel semantics | bridge `ctrl_mode:=accl` → **`drive.speed` carries acceleration [m/s²]** (NOT `drive.acceleration`) | we put accel in `drive.speed` ✓ |
| pose IN | `/odom` `nav_msgs/Odometry` (pose in `map`, body twist) **or** `ego_racecar/pose` `PoseStamped` | **`pose_source: odom`**, `odom_topic:/odom` |
| body twist | `/odom` `twist.linear.{x,y}`, `twist.angular.z` | read from `/odom` ✓ (vy may be ~0 in odom; fine) |
| enable | you publish `/mpcc/enable` `std_msgs/Bool` | `enable_topic:/mpcc/enable` |
| opponents (race) | `/perception/tracks` `obstacle_perception_msgs/TrackedObstacleArray` (id, map x/y, Frenet s/d, v_s/v_d) | `mode:race`, `obstacle_topic:/perception/tracks`, `obstacle_msg:tracked` |

> **KEY FIX vs the real-car default:** the real car reads `drive.acceleration`; the **sim reads
> `drive.speed`**. So for sim always set `longitudinal_field: speed`. (Or route our output through
> `mpcc_bringup/drive_adapter`, but publishing straight to `/ackermann_cmd` with accel in `.speed` is
> simpler and is a supported sim socket — there is no mux in sim.)

> **Track-match prerequisite (important):** the gym simulates on a `.pgm` map; our controller plans on its
> OWN `Track`. They must be the SAME physical track. The qualifier/race nets were trained on
> `Track.icra_t2_smooth()`, whose walls come from **`mpcc_tuning/tracks/icra2026_t2.pgm`** (res 0.05, origin
> [-2.8,-7.25]) — that is the gym map to load. NOTE the `sim_ws/src/tracks/icra*` maps are 2025/`icra25`
> variants with **different geometry** — do NOT use them; point the gym straight at the repo's
> `icra2026_t2`. The small real map `try5_ev7` has no gym `.pgm` yet; to sim-test *that* track, add its
> `.pgm/.yaml` and set `raceline_npz` to match.

---

## One-time setup: make the repo importable inside ROS

The node imports `mpcc_tuning`, `experiments.*`, `scuderia_gym_jax`, and acados. colcon/ROS will **not**
put this repo on `PYTHONPATH` automatically. Easiest: export it in the shell that launches the node (same
env vars we already use on the dev box):

```bash
export ACADOS_SOURCE_DIR=/home/poxx/acados-v060
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:/home/poxx/github/scuderia_gym_jax:/home/poxx/github/mpcc-online-tuning:$PYTHONPATH
export OMP_NUM_THREADS=1
```

Then put the package in the workspace and build:
```bash
ln -s /home/poxx/github/mpcc-online-tuning/node ~/sim_ws/src/mpcc_online   # or copy
cd ~/sim_ws && colcon build --packages-select mpcc_online && source install/setup.bash
```
(The first controller build compiles the acados solver, ~90 s; it is cached afterwards.)

---

## Stage 1 — qualifier (solo), the lowest-risk first test

> **✅ VERIFIED LIVE 2026-10-13 (ROS2 Humble, this `~/sim_ws`).** The exact commands below were run:
> node subscribes `/odom`, publishes `/ackermann_cmd`, honours the enable latch, drives **2+ clean laps,
> 0 off-track, 0 NaN**, cruise vx≈1.77-1.9 m/s (the net's pace), solve idle p95 40 ms (p95 51 / p99 58 ms
> under the sim+recorder sharing cores — lower on dedicated car compute). Build: `colcon build
> --packages-select mpcc_online` → 1.4 s (pure-python).

Five things that are easy to get wrong (all baked into the commands below):
> 1. **Remap the bridge node name:** the executable is named `gym_bridge` but `sim.yaml` keys its params
>    under `bridge:` → pass **`-r __node:=bridge`** or the whole params-file (including `ctrl_mode:=accl`)
>    is silently ignored and the car won't move.
> 2. **`-p map_img_ext:=.pgm`** — the gym defaults to `.png`; our ICRA map is a `.pgm`.
> 3. **No `map_server` needed** — the gym loads the map image itself from `map_path`; our node never
>    subscribes `/map`. (Add map_server only if you want the map in RViz.)
> 4. **Spawn the ego on the raceline:** `-p sx:=9.24 -p sy:=9.01 -p stheta:=3.02` = `icra_t2_smooth` s=0.
>    Spawn elsewhere and it starts off-line.
> 5. **Cold start is slow:** the gym spawns at v=0 and this (conservative) net accelerates gently, so it
>    takes ~30-40 s to reach cruise. That is the net, not a bug (offline `sim_check.py` hits only vmax 1.36
>    from v0=1.0 too). Let it run ≥1 min before judging.

```bash
# --- term 1: the simulator only (bridge loads the map itself; no map_server, no other controllers) ---
source /opt/ros/humble/setup.bash && source ~/sim_ws/install/setup.bash
MAP=/home/poxx/github/mpcc-online-tuning/mpcc_tuning/tracks/icra2026_t2    # matches Track.icra_t2_smooth
ros2 run scuderia_gym_ros gym_bridge --ros-args -r __node:=bridge \
  --params-file ~/sim_ws/src/scuderia_gym_ros/config/sim.yaml \
  -p model:=std -p ctrl_mode:=accl -p integrator:=rk4 -p tire_model:=full_pacejka_combined \
  -p map_path:=$MAP -p map_img_ext:=.pgm \
  -p sx:=9.24 -p sy:=9.01 -p stheta:=3.02 -p num_agent:=1

# --- term 2: our controller (env = the dev-box env, see above) ---
cd /home/poxx/github/mpcc-online-tuning          # so relative net_path/raceline resolve
ros2 run mpcc_online mpcc_online --ros-args --params-file node/config/deploy.yaml \
  -p mode:=qualify -p learn:=false \
  -p pose_source:=odom -p longitudinal_field:=speed \
  -p raceline_npz:='' -p track_name:=icra_t2_smooth      # '' -> icra_t2_smooth (NOT try5_ev7)
#   first start compiles the acados solver (~90 s) then prints "mpcc_online up."

# --- term 3: go (latch is Bool; hold it or publish repeatedly — our enable sub is volatile QoS) ---
ros2 topic pub -r 2 /mpcc/enable std_msgs/Bool "{data: true}"
```
Watch the debug topics — note they live under the **node name** `mpcc_online`:
`/mpcc_online/debug/solve_ms`, `/mpcc_online/debug/s`, `/mpcc_online/debug/accel`, `/mpcc_online/debug/lateral`,
`/mpcc_online/debug/off`. Set `learn:=true` to exercise on-car learning (keep-best banks a faster lap,
reverts a slower/off one).

A convenience launch that starts the sim + node together is `node/launch/sim_ws_bringup.launch.py`; the
verified manual recipe above is the ground truth for its params (it still needs `map_img_ext`/spawn/node-name
added if you adapt it).

## Stage 2 — racing (vs a field)

Add opponents + perception, and switch the node to race mode:
```bash
# sim with an opponent that keeps publishing /opp_drive (the gym won't step otherwise):
#   gym_bridge num_agent:=2  +  mpcc_bringup/opponent_driver (opp_speed>0)  on the ICRA map, ctrl_mode:=accl
# opponent perception -> /perception/tracks:
ros2 launch obstacle_perception sim_two_car_drive.launch.py    # detector_node + tracker_node

# our controller in race mode, physical2 net:
ros2 run mpcc_online mpcc_online --ros-args --params-file .../node/config/deploy.yaml \
  -p mode:=race -p longitudinal_field:=speed -p pose_source:=odom \
  -p net_path:=results/race_multicar/nets_physical2/race_ltc_0.npz \
  -p obstacle_topic:=/perception/tracks -p obstacle_msg:=tracked -p ego_pace:=1.77
```
The node consumes `TrackedObstacleArray` (filters out STATIC), tracks opponents by id, sets MPCC keep-outs
(+ a bounded blind prediction when a car is occluded), and emits class/crowd-conditioned weights — the same
logic validated in `race_multi_eval` (LTC 100/100/67 over N=1/2/3). Start with
`/perception/ground_truth/obstacles` (also a `TrackedObstacleArray`) to separate controller behaviour from
detector noise.

---

## What to copy from `D:\car7nuc_backup_2026-10-04` → WSL

**For the sim: nothing.** Everything the node needs in sim is already in `sim_ws`.

**For the REAL car**, `scale-mpcc`'s `car21.launch.py` pulls these from the car's `car21_ws` (the backup).
Copy the ones you don't already have into `~/sim_ws/src` (or a `car21_ws`) and `colcon build`:

| package (in backup / car21_ws) | provides | needed for |
|---|---|---|
| `racecar_launches` (`localization.launch.py`) | VESC driver, `ackermann_to_vesc`, **`ackermann_mux_ng`**, **particle_filter** (`/pf/viz/inferred_pose`), map_server, lidar, joystick | real-car drive + localization |
| `supreme_obstacle_detection` | lidar opponent detector → `/obstacles` (`bcl_interfaces/ObstacleList`) | real-car racing perception |
| `bcl_interfaces` | `ObstacleList` + other custom msgs | building the detector / its msgs |
| `raceline_manager` | `/raceline_name`, `/load_raceline` | real-car raceline selection (optional for us) |
| `timekeeping_node` | lap timing | optional |
| `drive_watchdog` | warns on `/drive` dropout | optional safety |

Our node's **real-car** mapping (already the `deploy.yaml` defaults): `pose_source:pose_topic`,
`pose_topic:/pf/viz/inferred_pose`, `longitudinal_field:acceleration`, `obstacle_msg:bcl` (parses
`ObstacleList` — note: points + id only, no velocity, so the on-board tracker estimates speed). You do
**not** need scale-mpcc's `mpcc_node` itself — `mpcc_online` replaces it.
