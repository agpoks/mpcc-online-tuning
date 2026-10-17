#!/usr/bin/env python3
"""Generate a Foxglove Studio layout for the mpcc_online deploy node (schema = studio-layout.json, same as
~/sim_ws/src/scuderia_gym_ros/foxglove/). Binds to the node's real topics: the /mpcc_online/debug/* Float32
diagnostics (incl. the online-tuned weights k_v/q_c/q_l/r_a that ros2_node now publishes), /odom, the pose
topic, /mpcc/enable, and the drive command. Mirrors the offline figure: weights adapting + pace + solve time
+ lateral/off. Run:  python3 node/foxglove/make_layout.py  ->  node/foxglove/mpcc_online_layout.json
If you change drive_topic to /drive (real car, see DEPLOY_CAR21.md), pass --drive /drive."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--node", default="mpcc_online", help="ROS node name (debug topics live under /<node>/debug/*)")
ap.add_argument("--drive", default="/ackermann_cmd", help="drive_topic (deploy.yaml default; real car = /drive)")
ap.add_argument("--pose", default="/pf/viz/inferred_pose", help="pose_topic (PoseStamped)")
ap.add_argument("--odom", default="/odom")
ap.add_argument("--enable", default="/mpcc/enable")
ap.add_argument("--out", default=str(Path(__file__).with_name("mpcc_online_layout.json")))
A = ap.parse_args()
D = f"/{A.node}/debug"                       # /mpcc_online/debug
dbg = lambda n: f"{D}/{n}.data"              # Float32 -> .data

# palette matches the offline figure (panel D): k_v blue, q_c orange, q_l green, r_a red
WCOL = {"k_v": "#2196F3", "q_c": "#FF9800", "q_l": "#4CAF50", "r_a": "#F44336"}


def plot(title, paths, synced=True):
    return {"title": title, "paths": paths, "xAxisVal": "timestamp", "isSynced": synced,
            "showLegend": True, "showPlotValuesInLegend": True, "legendDisplay": "floating",
            "followingViewWidth": 30}


def p(value, label, color, enabled=True):
    return {"value": value, "label": label, "color": color, "enabled": enabled,
            "timestampMethod": "receiveTime"}


def gauge(title, path, lo, hi, cmap="turbo", reverse=False):
    return {"title": title, "path": path, "minValue": lo, "maxValue": hi,
            "colorMode": "colormap", "colorMap": cmap, "reverse": reverse}


cfg = {}
# ---- panels ----
cfg["3D!m3d"] = {
    "followMode": "follow-pose",
    "scene": {"enableStats": False},
    "transforms": {},
    "topics": {A.pose: {"visible": True}, A.odom: {"visible": True}, A.drive: {"visible": False}},
    "layers": {},
    "cameraState": {"distance": 18.0, "perspective": True, "phi": 35.0, "thetaOffset": 0.0,
                    "targetOffset": [0, 0, 0], "target": [0, 0, 0], "targetOrientation": [0, 0, 0, 1],
                    "fovy": 45.0, "near": 0.5, "far": 5000.0},
    "publish": {"type": "point", "poseTopic": "/move_base_simple/goal"},
    "imageMode": {}, "synchronize": False,
}
cfg["Gauge!g_speed"] = gauge("Speed vx [m/s]", f"{A.odom}.twist.twist.linear.x", 0, 8)
cfg["Gauge!g_solve"] = gauge("Solve time [ms]  (20 Hz budget = 50)", dbg("solve_ms"), 0, 80, reverse=True)
cfg["Gauge!g_off"] = gauge("Off-track flag (0=in / 1=OUT)", dbg("off"), 0, 1, cmap="red-yellow-green", reverse=True)

# THE signal: the online-tuned weights adapting live (mirrors the offline figure panel D)
cfg["Plot!w"] = plot("Online-tuned MPCC weights (adapting live)",
                     [p(dbg(w), w, WCOL[w]) for w in ("k_v", "q_c", "q_l", "r_a")])
cfg["Plot!pace"] = plot("Pace — vx actual vs drive command",
                        [p(f"{A.odom}.twist.twist.linear.x", "vx actual [m/s]", "#4CAF50"),
                         p(dbg("accel"), "accel cmd [m/s^2]", "#9C27B0"),
                         p(f"{A.drive}.drive.steering_angle", "steer cmd [rad]", "#FFC107")])
cfg["Plot!track"] = plot("Tracking — lateral error & off-track",
                         [p(dbg("lateral"), "lateral error [m]", "#00BCD4"),
                          p(dbg("off"), "off-track (0/1)", "#F44336")])
cfg["Plot!solve"] = plot("Solve time [ms] (keep < 50 for 20 Hz)",
                         [p(dbg("solve_ms"), "solve_ms", "#FF5722")])
cfg["Plot!prog"] = plot("Progress s [m] along the corridor",
                        [p(dbg("s"), "s [m]", "#3F51B5")])
cfg["RawMessages!drive"] = {"topicPath": A.drive}
cfg["RawMessages!enable"] = {"topicPath": A.enable}

# ---- tabs ----
cfg["Tab!main"] = {"activeTabIdx": 0, "tabs": [
    {"title": "MPCC Online", "layout": {
        "direction": "row", "first": "3D!m3d", "splitPercentage": 44,
        "second": {"direction": "column", "splitPercentage": 26,
                   "first": {"direction": "row", "first": "Gauge!g_speed", "splitPercentage": 34,
                             "second": {"direction": "row", "first": "Gauge!g_solve",
                                        "second": "Gauge!g_off", "splitPercentage": 50}},
                   "second": {"direction": "column", "first": "Plot!w",
                              "second": "Plot!pace", "splitPercentage": 58}}}},
    {"title": "Online tuning", "layout": {
        "direction": "row", "splitPercentage": 58,
        "first": "Plot!w",
        "second": {"direction": "column", "splitPercentage": 50,
                   "first": "Plot!track",
                   "second": {"direction": "column", "first": "Plot!solve",
                              "second": "RawMessages!enable", "splitPercentage": 60}}}},
    {"title": "Control", "layout": {
        "direction": "row", "splitPercentage": 60,
        "first": {"direction": "column", "first": "Plot!pace", "second": "Plot!prog", "splitPercentage": 55},
        "second": {"direction": "column", "first": "RawMessages!drive",
                   "second": "Gauge!g_off", "splitPercentage": 55}}},
]}

layout = {"$schema": "https://foxglove.dev/schemas/studio-layout.json",
          "configById": cfg, "globalVariables": {}, "userNodes": {},
          "playbackConfig": {"speed": 1.0}, "layout": "Tab!main"}

Path(A.out).write_text(json.dumps(layout, indent=2))
# sanity: every panel id referenced in the tab tree must exist in configById
def ids(n):
    if isinstance(n, str):
        return {n}
    out = set()
    for k in ("first", "second"):
        if k in n:
            out |= ids(n[k])
    return out
ref = set()
for t in cfg["Tab!main"]["tabs"]:
    ref |= ids(t["layout"])
missing = ref - set(cfg)
assert not missing, f"tab references missing panels: {missing}"
print(f"wrote {A.out}  ({len(cfg)-1} panels, {len(cfg['Tab!main']['tabs'])} tabs, drive={A.drive})")
