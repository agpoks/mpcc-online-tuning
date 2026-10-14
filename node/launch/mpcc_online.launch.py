import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("mpcc_online"), "config", "deploy.yaml")
    return LaunchDescription([
        Node(
            package="mpcc_online",
            executable="mpcc_online",
            name="mpcc_online",
            output="screen",
            parameters=[cfg],
        ),
    ])
