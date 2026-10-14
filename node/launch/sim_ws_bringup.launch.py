"""Bring up the scuderia_gym_ros simulator + the mpcc_online controller in one command, for testing in
~/sim_ws. The solo default starts ONLY gym_bridge + our node (+ a trivial laser TF) -- NOT the other MPCC
controllers (supreme_mpc/mpcc_planner), and NOT nav2 map_server / opponent_driver, which are dead weight for
a solo qualifier and starved the 20 Hz control loop on a modest box in testing (solve time 40->140 ms -> the
car diverged). map_server (RViz only; our node never uses /map) is behind map_server:=true; opponent_driver
is auto-added only for num_agent>=2. The DEFAULTS are the qualifier recipe verified live on ROS2 Humble
2026-10-13 (map=icra2026_t2, spawn at icra_t2_smooth s=0, ctrl_mode=accl) -- see SIM_WS_TEST.md. The
equivalent manual `ros2 run` recipe there is the ground truth.

  # qualifier (solo) -- runs as-is, no args needed:
  ros2 launch mpcc_online sim_ws_bringup.launch.py
  ros2 topic pub -r 2 /mpcc/enable std_msgs/Bool "{data: true}"   # hold the latch (volatile QoS)
  # cold start: gym spawns at v=0 and the net accelerates gently -> ~30-40 s to reach cruise (not a bug).
  # racing: mode:=race num_agent:=2 opp_speed:=1.2  AND launch perception (obstacle_perception) separately,
  #         then set the node's obstacle_topic:=/perception/tracks (see SIM_WS_TEST.md).

The controller needs this repo on PYTHONPATH + the acados env (see SIM_WS_TEST.md); export them in the
launching shell. The first controller tick builds the acados solver (~90 s), hence the 8 s start delay is
only a courtesy -- the node will spend its first seconds compiling before it publishes.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gym_share = get_package_share_directory("scuderia_gym_ros")
    sim_yaml = os.path.join(gym_share, "config", "sim.yaml")           # bridge defaults
    deploy = os.path.join(get_package_share_directory("mpcc_online"), "config", "deploy.yaml")

    A = DeclareLaunchArgument
    # Defaults are the VERIFIED (2026-10-13) qualifier recipe on icra_t2_smooth (see SIM_WS_TEST.md).
    repo = "/home/poxx/github/mpcc-online-tuning"
    args = [
        # the gym map WHOSE WALLS match Track.icra_t2_smooth -- the repo's icra2026_t2 (NOT sim_ws icra* 2025 maps)
        A("map", default_value=repo + "/mpcc_tuning/tracks/icra2026_t2"),
        A("map_ext", default_value=".pgm"),         # gym defaults .png; our ICRA map is .pgm
        # spawn ego on the raceline at icra_t2_smooth s=0 (else it starts off-line)
        A("sx", default_value="9.24"), A("sy", default_value="9.01"), A("stheta", default_value="3.02"),
        A("net", default_value="results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz"),
        A("mode", default_value="qualify"),         # qualify | race
        A("learn", default_value="false"),          # qualify-mode online learning
        A("num_agent", default_value="1"),          # 2/3 for racing -> auto-adds opponent_driver
        A("opp_speed", default_value="0.0"),
        A("map_server", default_value="false"),     # true = also run nav2 map_server (for RViz; our node
    ]                                                # never uses /map). OFF by default: on modest boxes the
    #                                                  extra nodes starve the 20 Hz control loop (solve time
    #                                                  blew 40->140 ms in testing -> the car diverged).
    L = LaunchConfiguration
    Lf = lambda k: ParameterValue(L(k), value_type=float)   # launch args arrive as strings; type them so
    Li = lambda k: ParameterValue(L(k), value_type=int)     # the nodes' typed params accept the overrides
    Lb = lambda k: ParameterValue(L(k), value_type=bool)

    # name="bridge" == `-r __node:=bridge`: sim.yaml keys params under `bridge:`, so this MUST match or the
    # params-file (incl. ctrl_mode) is silently ignored and the car never moves.
    bridge = Node(
        package="scuderia_gym_ros", executable="gym_bridge", name="bridge", output="screen",
        parameters=[sim_yaml, {
            "model": "std", "ctrl_mode": "accl",            # ctrl_mode=accl -> drive.speed carries accel [m/s^2]
            "opp_ctrl_mode": "speed", "integrator": "rk4",
            "publish_extra_states": True, "publish_inner_states": True,
            "num_agent": Li("num_agent"), "map_path": L("map"), "map_img_ext": L("map_ext"),
            "sx": Lf("sx"), "sy": Lf("sy"), "stheta": Lf("stheta"),
        }])
    # /map (nav2) -- ONLY for RViz; our node never subscribes /map (the gym loads its own map image).
    # OFF by default (map_server:=false) to keep the control loop real-time on modest machines.
    map_cond = IfCondition(L("map_server"))
    map_server = Node(
        package="nav2_map_server", executable="map_server", name="map_server", output="screen",
        condition=map_cond,
        parameters=[{"yaml_filename": PythonExpression(["'", L("map"), "' + '.yaml'"]), "frame_id": "map"}])
    map_life = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map",
        output="screen", condition=map_cond,
        parameters=[{"autostart": True, "node_names": ["map_server"]}])
    # alias ego_racecar/laser -> laser (harmless; some tools look up 'laser'). Pose for OUR node is /odom.
    laser_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", name="laser_frame_alias",
        arguments=["0", "0", "0", "0", "0", "0", "ego_racecar/laser", "laser"])
    # opponent driver so the gym steps with num_agent>=2 (pure-pursuit). ONLY launched for num_agent>=2 --
    # at num_agent=1 it is dead weight that steals CPU from the control loop.
    opp = Node(
        package="mpcc_bringup", executable="opponent_driver", name="opponent_driver", output="screen",
        parameters=[{"opp_speed": Lf("opp_speed")}],
        condition=IfCondition(PythonExpression([L("num_agent"), " >= 2"])))

    mpcc = Node(
        package="mpcc_online", executable="mpcc_online", name="mpcc_online", output="screen",
        parameters=[deploy, {
            "mode": L("mode"), "learn": Lb("learn"), "net_path": L("net"),
            "pose_source": "odom", "odom_topic": "/odom",               # gym ground-truth pose+twist
            "longitudinal_field": "speed",                             # sim reads drive.speed as accel
            "mux_cmd_topic": "/ackermann_cmd",
            # race: point at perception once it is up (launched separately)
            "obstacle_topic": PythonExpression(["'/perception/tracks' if '", L("mode"), "' == 'race' else ''"]),
            "obstacle_msg": "tracked",
        }])

    return LaunchDescription(args + [
        bridge, map_server, map_life, laser_tf, opp,
        TimerAction(period=8.0, actions=[mpcc]),   # let the sim + map settle before the controller starts
    ])
