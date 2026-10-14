"""Online-tuned MPCC deploy node (ROS2) for the RC car.

`controller.OnlineMPCCController` is the ROS-agnostic core (state in -> steer/accel out); it reuses the
repo's acados MPCC + trained weight-policy net + Track with NO changes to them. `ros2_node` is the thin
rclpy wrapper; `sim_check` drives the core against the ScuderiaPlant with no ROS, to verify it end-to-end.
"""
