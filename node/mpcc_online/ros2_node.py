"""ROS2 (rclpy) node for the online-tuned MPCC, wired to the car21 / scuderia_gym_ros interface.
Mirrors scale-mpcc's node structure (enable latch, pose-timeout, failed-solve decay, warm-start lifecycle)
but drives OUR OnlineMPCCController (+ optional on-car PolicyTuner learning). rclpy is imported lazily so
this file is inspectable without ROS.

Interface (confirmed against scuderia_gym_ros + the real car21 stack):
  IN  pose   : /pf/viz/inferred_pose (PoseStamped, laser frame -> shifted to CoG)  [real car]
               or /odom pose           (sim bridge provides full odom)             [pose_source=odom]
      vx     : /odom (nav_msgs/Odometry, BODY-frame twist.linear.x; twist.angular.z = yaw rate)
      enable : /mpcc/enable (Bool) -- drive + learn latch
      mux    : /ackermann_cmd (AckermannDriveStamped) -- who owns the car
  OUT drive  : /ackermann_cmd (AckermannDriveStamped)
               longitudinal_field='acceleration' -> drive.acceleration = a_x [m/s^2]  (real car ackermann_to_vesc)
               longitudinal_field='speed'        -> drive.speed        = a_x [m/s^2]  (sim bridge ctrl_mode='accl')
               drive.steering_angle = delta [rad]
Our MPCC outputs acceleration natively, so the car's ackermann_to_vesc (current = C_acc*a + C_v*v) handles it.
"""
from __future__ import annotations
import math
import numpy as np
from .controller import OnlineMPCCController


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool, Float32
    from ackermann_msgs.msg import AckermannDriveStamped

    class MpccOnlineNode(Node):
        def __init__(self):
            super().__init__("mpcc_online")
            P = self.declare_parameters("", [
                ("net_path", ""), ("raceline_npz", ""), ("track_name", "icra_t2_smooth"),
                ("mode", "qualify"), ("learn", False), ("kv_launch", 0.50),
                ("ego_pace", 1.77), ("pred_r", 0.0),
                ("obstacle_topic", ""), ("obstacle_msg", "tracked"),   # race: opponents in (sim: /perception/tracks)
                ("pose_source", "pose_topic"), ("pose_topic", "/pf/viz/inferred_pose"),
                ("odom_topic", "/odom"), ("enable_topic", "/mpcc/enable"),
                ("drive_topic", "/ackermann_cmd"), ("mux_cmd_topic", "/ackermann_cmd"),
                ("longitudinal_field", "acceleration"), ("pose_forward_offset", 0.0678),
                ("rate_hz", 20.0), ("pose_timeout_s", 0.5), ("fail_decay", 0.7),
                ("accel_min", -8.0), ("accel_max", 8.0), ("steer_max", 0.42), ("autostart", False)])
            g = lambda k: self.get_parameter(k).value
            cfg = dict(track_name=g("track_name"), kv_launch=float(g("kv_launch")),
                       mode=str(g("mode")), learn=bool(g("learn")), ego_pace=float(g("ego_pace")))
            if float(g("pred_r")) > 0:
                cfg["pred_r"] = float(g("pred_r"))
            if g("raceline_npz"):
                cfg["raceline_npz"] = g("raceline_npz")
            self.get_logger().info(f"building MPCC + net {g('net_path')}  learn={cfg['learn']} ...")
            self.ctl = OnlineMPCCController(g("net_path"), cfg)
            self._lon = g("longitudinal_field"); self._off = float(g("pose_forward_offset"))
            self._amin, self._amax, self._smax = float(g("accel_min")), float(g("accel_max")), float(g("steer_max"))
            self._timeout = float(g("pose_timeout_s")); self._decay = float(g("fail_decay"))
            self.enabled = bool(g("autostart")); self.warm = False
            self.pose = None; self.pose_t = None; self.vx = 0.0; self.vy = 0.0; self.yawrate = 0.0
            self.delta_est = 0.0; self.a_last = 0.0; self._sent = []
            self._opponents = []; self._opp_t = None     # race mode: latest detected field (map frame)

            self.create_subscription(Odometry, g("odom_topic"), self._odom_cb, qos_profile_sensor_data)
            if g("pose_source") == "pose_topic":
                self.create_subscription(PoseStamped, g("pose_topic"), self._pose_cb, qos_profile_sensor_data)
            self.create_subscription(Bool, g("enable_topic"), self._enable_cb, 10)
            self.create_subscription(AckermannDriveStamped, g("mux_cmd_topic"), self._mux_cb, 10)
            # RACE mode: subscribe opponent perception. Sim: /perception/tracks (obstacle_perception_msgs/
            # TrackedObstacleArray -- has id, classification, map x/y, Frenet s/d, v_s/v_d). Real car
            # (scale-mpcc): bcl_interfaces/ObstacleList on /obstacles (points + identifier, no velocity).
            # obstacle_msg selects the parser; the import is lazy so this file loads without those msgs.
            if cfg["mode"] == "race" and g("obstacle_topic"):
                self._sub_obstacles(g("obstacle_topic"), g("obstacle_msg"))
            self._pub = self.create_publisher(AckermannDriveStamped, g("drive_topic"), 10)
            self._dbg = {n: self.create_publisher(Float32, f"~/debug/{n}", 10)
                         for n in ("vx", "solve_ms", "lateral", "s", "accel", "off")}
            self.create_timer(1.0 / float(g("rate_hz")), self._tick)
            self.get_logger().info("mpcc_online up.")

        # ---- inputs ----
        def _pose_cb(self, m):
            yaw = _yaw(m.pose.orientation)
            self.pose = (m.pose.position.x - self._off * math.cos(yaw),
                         m.pose.position.y - self._off * math.sin(yaw), yaw)   # laser -> CoG
            self.pose_t = self.get_clock().now()

        def _odom_cb(self, m):
            self.vx = m.twist.twist.linear.x; self.vy = m.twist.twist.linear.y
            self.yawrate = m.twist.twist.angular.z
            if self.get_parameter("pose_source").value == "odom":
                yaw = _yaw(m.pose.pose.orientation)
                self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)
                self.pose_t = self.get_clock().now()

        def _enable_cb(self, m):
            if m.data and not self.enabled:
                self.warm = False; self.ctl.reset(); self.get_logger().info("MPCC ENABLED")
            if not m.data and self.enabled:
                self.get_logger().info("MPCC disabled")
                if self.ctl.learn:
                    self._save_ckpt()
            self.enabled = bool(m.data)

        def _mux_cb(self, m):
            # if the mux is forwarding a command that is not one of ours, someone else owns the car
            cur = (round(m.drive.acceleration, 3), round(m.drive.speed, 3), round(m.drive.steering_angle, 3))
            if self.enabled and self._sent and cur not in self._sent:
                self.delta_est = m.drive.steering_angle; self.a_last = 0.0; self.warm = False

        # ---- opponents (race mode) ----
        def _sub_obstacles(self, topic, kind):
            """Subscribe the opponent-perception topic. Lazy msg import so this node loads without the
            perception msgs installed; if they are missing, log and run solo (empty field)."""
            try:
                if kind == "tracked":                                  # sim: obstacle_perception stack
                    from obstacle_perception_msgs.msg import TrackedObstacleArray as Msg
                elif kind == "bcl":                                     # real car: scale-mpcc detector
                    from bcl_interfaces.msg import ObstacleList as Msg
                else:                                                   # sim: obstacle_detection_node
                    from sim_planning_msgs.msg import ObstacleList as Msg
            except Exception as e:
                self.get_logger().warn(f"opponent msg for '{kind}' unavailable ({e}); racing SOLO")
                return
            self._obs_kind = kind
            self.create_subscription(Msg, topic, self._obstacle_cb, qos_profile_sensor_data)
            self.get_logger().info(f"race: opponents from {topic} ({kind})")

        def _obstacle_cb(self, m):
            opp = []
            if getattr(self, "_obs_kind", "tracked") == "tracked":
                for o in m.obstacles:
                    if getattr(o, "classification", 2) == 1:       # 1=STATIC -> wall, not a rival
                        continue
                    d = dict(x=float(o.x), y=float(o.y), id=int(o.id),
                             v=float(math.hypot(getattr(o, "v_s", 0.0), getattr(o, "v_d", 0.0))))
                    if getattr(o, "has_frenet", False):
                        d["s"] = float(o.s); d["d"] = float(o.d)
                    opp.append(d)
            else:  # ObstacleList: points + identifier, no velocity -> centroid, speed left to the tracker
                for o in m.obstacles:
                    pts = [(p.point.x, p.point.y) for p in o.points] if o.points else []
                    if not pts:
                        continue
                    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
                    opp.append(dict(x=float(cx), y=float(cy), id=str(o.identifier)))
            self._opponents = opp; self._opp_t = self.get_clock().now()

        def _current_opponents(self, now):
            """The latest detected field, or () when stale/absent -- matched to the pose timeout so a
            perception dropout degrades to solo driving rather than acting on a frozen field."""
            if self._opp_t is None:
                return ()
            if (now - self._opp_t).nanoseconds * 1e-9 > self._timeout:
                return ()
            return self._opponents

        # ---- control loop ----
        def _tick(self):
            now = self.get_clock().now()
            fresh = self.pose is not None and self.pose_t is not None and \
                (now - self.pose_t).nanoseconds * 1e-9 < self._timeout
            if not (self.enabled and fresh):
                return                      # silent: mux stops the car on its own /drive gap
            if not self.warm:
                self.ctl.reset(); self.warm = True
            x, y, psi = self.pose
            out = self.ctl.control(x, y, psi, self.vx, self.vy, self.yawrate, self.delta_est,
                                   opponents=self._current_opponents(now))
            u = out["u0"]
            if u is None or not np.all(np.isfinite(u)):
                accel = self._decay * self.a_last; steer = self.delta_est          # failed solve
            else:
                accel = float(np.clip(out["accel"], self._amin, self._amax))
                steer = float(np.clip(out["steer"], -self._smax, self._smax))
            self.delta_est = steer; self.a_last = accel
            self._publish(accel, steer)
            for n, val in (("vx", self.vx), ("solve_ms", out["solve_ms"]), ("lateral", out["lateral"]),
                           ("s", out["s"]), ("accel", accel), ("off", float(out.get("off", 0)))):
                self._dbg[n].publish(Float32(data=float(val)))

        def _publish(self, accel, steer):
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = "base_link"
            m.drive.steering_angle = float(steer)
            if self._lon == "acceleration":
                m.drive.acceleration = float(accel); m.drive.speed = float("inf")   # real car convention
            else:
                m.drive.speed = float(accel)                                        # sim ctrl_mode='accl'
            self._pub.publish(m)
            self._sent = (self._sent + [(round(m.drive.acceleration, 3), round(m.drive.speed, 3),
                                         round(float(steer), 3))])[-3:]

        def _save_ckpt(self):
            try:
                p = self.get_parameter("net_path").value + ".ckpt.npz"
                self.ctl.save_ckpt(p); self.get_logger().info(f"saved learn checkpoint -> {p}")
            except Exception as e:
                self.get_logger().warn(f"checkpoint save failed: {e}")

    rclpy.init(args=args)
    node = MpccOnlineNode()
    try:
        rclpy.spin(node)
    finally:
        if node.ctl.learn:
            node._save_ckpt()
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
