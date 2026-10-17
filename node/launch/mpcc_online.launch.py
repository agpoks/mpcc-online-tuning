"""Launch the online-tuned MPCC deploy node with its env set up.

The node imports `mpcc_tuning` + acados (NOT installed by colcon -- they live in the repo + the acados tree),
so this launch file puts the repo root and acados on PYTHONPATH/LD_LIBRARY_PATH for the node process. Point it
at the repo and acados with launch args or the env vars MPCC_REPO / ACADOS_SOURCE_DIR.

  ros2 launch mpcc_online mpcc_online.launch.py \
      repo_dir:=/home/car21/mpcc-online-tuning acados_dir:=/home/car21/acados \
      drive_topic:=/drive raceline_npz:=mpcc_tuning/tracks/lab_test_ev8_corridor.npz

Everything else (net_path, kv_launch, limits, ...) comes from params_file (default: the installed deploy.yaml).
Car-specific values that MUST change for car21 are drive_topic:=/drive and the longitudinal units -- see
DEPLOY_CAR21.md. The node resolves relative net_path/raceline_npz against repo_dir (cwd is set there).
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("mpcc_online"), "config", "deploy.yaml")
    repo = LaunchConfiguration("repo_dir")
    acados = LaunchConfiguration("acados_dir")
    args = [
        DeclareLaunchArgument("repo_dir", default_value=EnvironmentVariable("MPCC_REPO", default_value=os.getcwd()),
                              description="path to the mpcc-online-tuning repo (holds mpcc_tuning/, the nets, corridors)"),
        DeclareLaunchArgument("acados_dir", default_value=EnvironmentVariable("ACADOS_SOURCE_DIR", default_value="/opt/acados"),
                              description="acados install root (has lib/ and interfaces/acados_template)"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("drive_topic", default_value="/ackermann_cmd",
                              description="CAR21: set to /drive (mux deadman input), NOT /ackermann_cmd"),
        DeclareLaunchArgument("raceline_npz", default_value="mpcc_tuning/tracks/lab_test_ev8_corridor.npz"),
        DeclareLaunchArgument("mode", default_value="qualify"),
        DeclareLaunchArgument("learn", default_value="true"),
    ]
    # env for the node process: repo + acados importable, single-threaded acados
    set_env = [
        SetEnvironmentVariable("ACADOS_SOURCE_DIR", acados),
        SetEnvironmentVariable("OMP_NUM_THREADS", "1"),
        SetEnvironmentVariable("PYTHONPATH", [PathJoinSubstitution([acados, "interfaces", "acados_template"]),
                                              ":", repo, ":", EnvironmentVariable("PYTHONPATH", default_value="")]),
        SetEnvironmentVariable("LD_LIBRARY_PATH", [PathJoinSubstitution([acados, "lib"]),
                                                   ":", EnvironmentVariable("LD_LIBRARY_PATH", default_value="")]),
    ]
    node = Node(
        package="mpcc_online", executable="mpcc_online", name="mpcc_online", output="screen",
        cwd=repo,                                   # so relative net_path / raceline_npz resolve against the repo
        parameters=[LaunchConfiguration("params_file"), {
            "drive_topic": LaunchConfiguration("drive_topic"),
            "raceline_npz": LaunchConfiguration("raceline_npz"),
            "mode": LaunchConfiguration("mode"),
            "learn": ParameterValue(LaunchConfiguration("learn"), value_type=bool),
        }],
    )
    return LaunchDescription(args + set_env + [node])
