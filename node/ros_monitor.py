#!/usr/bin/env python3
"""Monitor the lab_test sim_ws ROS drive: subscribe the ego odom + the node's /mpcc_online/debug/* topics,
run for DURATION s, then report laps (odom projected onto the corridor), off-track fraction, solve-time
percentiles, and CONFIRM the new online-tuned-weight topics publish + vary. Pure functional verification."""
import sys, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
CORR = "/home/poxx/github/mpcc-online-tuning/mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
g = np.load(CORR); cx, cy = g["cx"], g["cy"]
cums = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))]); L = float(cums[-1])
P = np.column_stack([cx, cy])


class Mon(Node):
    def __init__(self):
        super().__init__("mpcc_monitor")
        self.xy = []; self.off = []; self.solve = []; self.vx = []
        self.w = {k: [] for k in ("k_v", "q_c", "q_l", "r_a")}
        self.odom_topic = None
        for t in ("/ego_racecar/odom", "/odom"):
            self.create_subscription(Odometry, t, lambda m, tt=t: self._odom(m, tt), qos_profile_sensor_data)
        D = "/mpcc_online/debug"
        self.create_subscription(Float32, f"{D}/off", lambda m: self.off.append(m.data), 10)
        self.create_subscription(Float32, f"{D}/solve_ms", lambda m: self.solve.append(m.data), 10)
        self.create_subscription(Float32, f"{D}/vx", lambda m: self.vx.append(m.data), 10)
        for k in self.w:
            self.create_subscription(Float32, f"{D}/{k}", lambda m, kk=k: self.w[kk].append(m.data), 10)

    def _odom(self, m, t):
        if self.odom_topic is None:
            self.odom_topic = t
        if t == self.odom_topic:
            self.xy.append((m.pose.pose.position.x, m.pose.pose.position.y))


def main():
    rclpy.init()
    n = Mon()
    t0 = time.time()
    while time.time() - t0 < DURATION:
        rclpy.spin_once(n, timeout_sec=0.1)
    xy = np.array(n.xy)
    # laps: project each odom sample to nearest corridor node -> s, unwrap, accumulate progress
    laps = 0.0
    if len(xy) > 5:
        s = cums[np.argmin(((xy[:, None, :] - P[None, :, :]) ** 2).sum(-1), axis=1)]
        ds = np.diff(s); ds += np.where(ds < -0.5 * L, L, 0) - np.where(ds > 0.5 * L, L, 0)
        laps = float(ds[np.abs(ds) < 0.5 * L].sum()) / L
    off = np.array(n.off); solve = np.array(n.solve); vx = np.array(n.vx)
    off_frac = float((off > 0.5).mean()) if off.size else float("nan")
    print("=" * 72)
    print(f"odom topic used: {n.odom_topic}   samples: odom={len(xy)} off={off.size} solve={solve.size}")
    print(f"LAPS (projected): {laps:.2f}   off-track fraction: {off_frac*100:.1f}%   "
          f"({int((off>0.5).sum())}/{off.size} ticks)")
    if solve.size:
        print(f"solve_ms: p50={np.percentile(solve,50):.0f} p95={np.percentile(solve,95):.0f} "
              f"max={solve.max():.0f}  (BOX LOADED -> contended, not a clean number)")
    if vx.size:
        print(f"speed vx: mean={np.mean(vx[vx>0.3]):.2f} max={vx.max():.2f} m/s")
    print("-- online-tuned weight topics (the new publishers) --")
    for k, v in n.w.items():
        a = np.array(v)
        if a.size:
            print(f"  /mpcc_online/debug/{k:4s}: {a.size} msgs, range [{a.min():.3f}, {a.max():.3f}]  "
                  f"{'VARIES' if np.ptp(a)>1e-4 else 'constant'}")
        else:
            print(f"  /mpcc_online/debug/{k:4s}: NO MESSAGES")
    print("=" * 72)
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
