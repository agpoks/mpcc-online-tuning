#!/usr/bin/env python3
"""Generate a Foxglove Studio layout for the mpcc_online deploy node.

MODELLED ON the real car21 race dashboard `car21_ws/foxglove_dashboards/race_dashboard_mpc_ax.json`
(from the car7nuc_backup_2026-10-04 backup): same schema (studio-layout.json), same panel vocabulary
(Parameters | Tab top split; Plot with floating legend + foxglovePanelTitle; Indicator with background
rules; 3D with a grid layer + pose arrow + scan + a car-trail marker; StateTransitions; Log filtered to
our node; Publish buttons), and the same two reusable user scripts — the fading car-trajectory trail and
the sideslip-beta derivation — carried over verbatim and REBOUND to our node's topics (no cross-repo dep;
the TS is vendored below with this source note).

What the real dashboard shows for /supreme_mpc + /slicer + /mpc_diag + disparity_extender, this shows for
OUR node: the online-tuned weights k_v/q_c/q_l/r_a adapting live (our analog of the MPC-sectors Q-weights
tab), plus pace, tracking, solve time, and controller-enable switching. Panels with no equivalent in our
node (slicer lateral/angular error, mpc_diag cost breakdown, disparity-extender sectors, the raceline-viz
publisher) are dropped rather than bound to dead topics.

Binds to the node's real topics: /mpcc_online/debug/* Float32 diagnostics (incl. the online-tuned weights
ros2_node publishes), /odom, the pose topic, /mpcc/enable, the drive command, and the obstacle topic.

Run:  python3 node/foxglove/make_layout.py           ->  node/foxglove/mpcc_online_layout.json
Real car (see DEPLOY_CAR21.md): python3 node/foxglove/make_layout.py --drive /drive
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--node", default="mpcc_online", help="ROS node name (debug topics live under /<node>/debug/*)")
ap.add_argument("--drive", default="/ackermann_cmd", help="drive_topic (deploy.yaml default; real car = /drive)")
ap.add_argument("--pose", default="/pf/viz/inferred_pose", help="pose_topic (PoseStamped) for the 3D view + trail")
ap.add_argument("--odom", default="/odom", help="Odometry (body-frame twist) for speed + sideslip")
ap.add_argument("--enable", default="/mpcc/enable", help="Bool drive/learn latch")
ap.add_argument("--obstacles", default="/obstacles", help="obstacle topic (race mode)")
ap.add_argument("--out", default=str(Path(__file__).with_name("mpcc_online_layout.json")))
A = ap.parse_args()

D = f"/{A.node}/debug"                           # /mpcc_online/debug
dbg = lambda n: f"{D}/{n}.data"                  # Float32 -> .data
VIZ = f"/{A.node}/viz"                           # trail marker + clear live here
DIAG = f"/{A.node}/diag"                         # derived signals (beta) live here
TRAIL, CLEAR, BETA = f"{VIZ}/car_trail", f"{VIZ}/trail_clear", f"{DIAG}/beta_angle"

# palette matches the offline paper figure (panel D): k_v blue, q_c orange, q_l green, r_a red
WCOL = {"k_v": "#2196F3", "q_c": "#FF9800", "q_l": "#4CAF50", "r_a": "#F44336"}

# ---------------------------------------------------------------- panel builders (D: dashboard schema)
def plot(title, paths, ymin=None, ymax=None, synced=True):
    c = {"paths": paths, "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
         "legendDisplay": "floating", "showPlotValuesInLegend": True, "isSynced": synced,
         "xAxisVal": "timestamp", "sidebarDimension": 240, "foxglovePanelTitle": title}
    if ymin is not None: c["minYValue"] = ymin
    if ymax is not None: c["maxYValue"] = ymax
    return c

def pth(value, label, color, enabled=True):
    return {"timestampMethod": "receiveTime", "value": value, "enabled": enabled, "color": color, "label": label}

def indicator(title, path, rules, fallback_color="#424242", fallback_label="—", font=18):
    return {"path": path, "style": "background", "fallbackColor": fallback_color, "fallbackLabel": fallback_label,
            "rules": rules, "foxglovePanelTitle": title, "fontSize": font}

def rule(op, val, color, label):
    return {"operator": op, "rawValue": str(val), "color": color, "label": label}

def raw(title, topic_path, expanded=False):
    return {"diffEnabled": False, "diffMethod": "custom", "diffTopicPath": "", "showFullMessageForDiff": False,
            "topicPath": topic_path, "fontSize": 13, "foxglovePanelTitle": title, "defaultExpanded": expanded}

def view3d(title, extra_topics=None):
    topics = {"/map": {"visible": True, "alpha": 1, "drawBehind": True, "frameLocked": True, "colorMode": "custom"},
              "/scan": {"visible": True, "colorField": "range", "colorMode": "colormap", "colorMap": "turbo",
                        "pointSize": 3, "decayTime": 0},
              A.pose: {"visible": True, "color": "#4CAF50", "size": 0.25, "type": "arrow"},
              TRAIL: {"visible": True}}
    if extra_topics:
        topics.update(extra_topics)
    return {
        "layers": {"grid-map": {"visible": True, "frameLocked": True, "label": "Grid", "instanceId": "grid-map",
                                "layerId": "foxglove.Grid", "size": 80, "divisions": 80, "lineWidth": 1,
                                "color": "#248eff5c", "position": [0, 0, 0], "rotation": [0, 0, 0], "order": 1}},
        "cameraState": {"perspective": False, "distance": 16.0, "phi": 0.0, "thetaOffset": 0.0,
                        "targetOffset": [0, 0, 0], "target": [0, 0, 0], "targetOrientation": [0, 0, 0, 1],
                        "fovy": 45, "near": 0.5, "far": 5000, "logDepth": False},
        "followMode": "follow-position",
        "scene": {"enableStats": False, "meshUpAxis": "z_up", "transforms": {"showLabel": False, "axisSize": 0.2}},
        "transforms": {}, "topics": topics, "publish": {"type": "point", "poseTopic": "/move_base_simple/goal"},
        "foxglovePanelTitle": title,
    }

def state_transitions(title, paths):
    return {"paths": [{"value": v, "timestampMethod": "receiveTime", "label": l} for v, l in paths],
            "isSynced": True, "foxglovePanelTitle": title, "showPoints": False}

def log_panel(title, name):
    return {"searchTerms": [], "minLogLevel": 1, "nameFilter": {name: {"visible": True}},
            "topicToRender": "/rosout", "foxglovePanelTitle": title}

def params(title):
    return {"foxglovePanelTitle": title}

def publish(title, topic, datatype, value, text, color):
    return {"topicName": topic, "datatype": datatype, "buttonText": text, "buttonColor": color,
            "advancedView": False, "value": value, "foxglovePanelTitle": title}

# ---------------------------------------------------------------- user scripts (vendored from the D:
# race dashboard's userNodes, rebound to our topics -- see module docstring)
TRAIL_TS = r'''import { Input, Message } from "./types";

// Accumulates the car's pose history into a single LINE_STRIP marker, with
// distance-based sampling, a bounded buffer, a rate-limited emit and an
// oldest->newest alpha fade. A Publish of std_msgs/Empty on the clear topic
// drops the buffer and emits a DELETE marker.
// Vendored from the car21 race dashboard ("Car Trajectory Trail"), rebound.

type Marker = Message<"visualization_msgs/msg/Marker">;
type Point = { x: number; y: number; z: number };
type Color = { r: number; g: number; b: number; a: number };
type ClearEvent = { topic: "__CLEAR__"; message: Record<string, never> };

export const inputs = ["__POSE__", "__CLEAR__"];
export const output = "__TRAIL__";

const MAX_POINTS = 1500;
const MIN_DIST_M = 0.10;
const MIN_DIST_SQ = MIN_DIST_M * MIN_DIST_M;
const EMIT_PERIOD_NS = 200_000_000; // 5 Hz

let points: Point[] = [];
let last_emit_ns = 0;

function emptyDeleteMarker(): Marker {
  return {
    header: { stamp: { sec: 0, nsec: 0 }, frame_id: "map" },
    ns: "__NS__", id: 1001, type: 4, action: 2,
    pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    scale: { x: 0, y: 0, z: 0 }, color: { r: 0, g: 0, b: 0, a: 0 },
    lifetime: { sec: 0, nsec: 0 }, frame_locked: false, points: [], colors: [],
    texture_resource: "",
    texture: { header: { stamp: { sec: 0, nsec: 0 }, frame_id: "" }, format: "",
               data: new Uint8Array(0) as Uint8Array<ArrayBufferLike> },
    uv_coordinates: [], text: "", mesh_resource: "",
    mesh_file: { filename: "", data: new Uint8Array(0) as Uint8Array<ArrayBufferLike> },
    mesh_use_embedded_materials: false,
  };
}

export default function script(event: Input<"__POSE__"> | ClearEvent): Marker | undefined {
  if (event.topic === "__CLEAR__") {
    points = [];
    last_emit_ns = 0;
    return emptyDeleteMarker();
  }

  const p = event.message.pose.position;
  const last = points.length > 0 ? points[points.length - 1] : null;
  if (last === null) {
    points.push({ x: p.x, y: p.y, z: 0.15 });
  } else {
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    if (dx * dx + dy * dy >= MIN_DIST_SQ) {
      points.push({ x: p.x, y: p.y, z: 0.15 });
      if (points.length > MAX_POINTS) { points.shift(); }
    }
  }

  const stamp = event.message.header.stamp;
  const now_ns = stamp.sec * 1_000_000_000 + stamp.nsec;
  if (now_ns - last_emit_ns < EMIT_PERIOD_NS) { return undefined; }
  last_emit_ns = now_ns;

  const n = points.length;
  const denom = Math.max(1, n - 1);
  const colors: Color[] = points.map((_, i) => {
    const t = i / denom;
    return { r: 1.0, g: 0.5, b: 0.0, a: 0.15 + 0.85 * t };
  });

  return {
    header: { stamp: event.message.header.stamp, frame_id: "map" },
    ns: "__NS__", id: 1001, type: 4, action: 0,
    pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    scale: { x: 0.04, y: 0, z: 0 }, color: { r: 1.0, g: 0.5, b: 0.0, a: 1.0 },
    lifetime: { sec: 0, nsec: 0 }, frame_locked: false,
    points: points.slice(), colors: colors, texture_resource: "",
    texture: { header: { stamp: { sec: 0, nsec: 0 }, frame_id: "" }, format: "",
               data: new Uint8Array(0) as Uint8Array<ArrayBufferLike> },
    uv_coordinates: [], text: "", mesh_resource: "",
    mesh_file: { filename: "", data: new Uint8Array(0) as Uint8Array<ArrayBufferLike> },
    mesh_use_embedded_materials: false,
  };
}
'''

BETA_TS = r'''import { Input } from "./types";

// Body-frame sideslip angle beta = atan2(vy, vx) from the odometry twist
// (base_link frame: linear.x longitudinal, linear.y lateral). Held at 0
// below MIN_SPEED where beta is meaningless. Valid only if the source
// actually estimates vy (VESC wheel odometry has vy=0).
// Vendored from the car21 race dashboard ("Sideslip Beta Angle"), rebound.

type Output = { data: number };
const MIN_SPEED = 0.3; // m/s

export const inputs = ["__ODOM__"];
export const output = "__BETA__";

export default function script(event: Input<"__ODOM__">): Output {
  const vx = Number(event.message.twist.twist.linear.x ?? 0);
  const vy = Number(event.message.twist.twist.linear.y ?? 0);
  if (Math.abs(vx) < MIN_SPEED) { return { data: 0 }; }
  return { data: Math.atan2(vy, vx) };
}
'''

def rebind(ts, **kw):
    for k, v in kw.items():
        ts = ts.replace(k, v)
    return ts

user_nodes = {
    "mpcc_car_trail": {"name": "Car Trajectory Trail",
                       "sourceCode": rebind(TRAIL_TS, __POSE__=A.pose, __CLEAR__=CLEAR, __TRAIL__=TRAIL, __NS__=A.node)},
    "mpcc_beta_angle": {"name": "Sideslip Beta Angle (atan2(vy, vx))",
                        "sourceCode": rebind(BETA_TS, __ODOM__=A.odom, __BETA__=BETA)},
}

# ---------------------------------------------------------------- panels
cfg = {}
cfg["Parameters!params"] = params(f"Live tuning — expand /{A.node} here (net_path, kv_launch, mode, learn)")
cfg["Publish!enable_on"] = publish("Latch MPCC ON", A.enable, "std_msgs/msg/Bool", '{"data": true}',
                                   "Enable MPCC (hold deadman!)", "#1b5e20")
cfg["Publish!enable_off"] = publish("Latch MPCC OFF", A.enable, "std_msgs/msg/Bool", '{"data": false}',
                                    "Disable MPCC", "#b71c1c")
cfg["Publish!clear_trail"] = publish("Clear trail", CLEAR, "std_msgs/msg/Empty", "{}", "Clear car trail", "#f5774d")

cfg["3D!view"] = view3d("Track / pose / trail")
cfg["3D!obs"] = view3d("Obstacles", extra_topics={A.obstacles: {"visible": True}})

cfg["Plot!speed"] = plot("Speed — vx [m/s]",
                         [pth(f"{A.odom}.twist.twist.linear.x", "vx (odom)", "#4CAF50"),
                          pth(dbg("vx"), "vx (node)", "#81C784")], ymin=0, ymax=8)
cfg["Plot!drive"] = plot("Drive command — accel & steer",
                         [pth(dbg("accel"), "accel cmd [m/s^2]", "#9C27B0"),
                          pth(f"{A.drive}.drive.steering_angle", "steer cmd [rad]", "#FFC107")])
# THE signal: the online-tuned weights adapting live (mirrors the paper figure panel D)
cfg["Plot!weights"] = plot("Online-tuned MPCC weights (adapting live)",
                           [pth(dbg(w), w, WCOL[w]) for w in ("k_v", "q_c", "q_l", "r_a")])
cfg["Plot!track"] = plot("Tracking — lateral error & off-track",
                         [pth(dbg("lateral"), "lateral error [m]", "#00BCD4"),
                          pth(dbg("off"), "off-track (0/1)", "#F44336")])
cfg["Plot!beta"] = plot("Sideslip beta [rad]  (atan2(vy, vx))", [pth(f"{BETA}.data", "beta", "#E91E63")])
cfg["Plot!solve"] = plot("Solve time [ms]  (20 Hz budget = 50)", [pth(dbg("solve_ms"), "solve_ms", "#FF5722")],
                         ymin=0, ymax=80)
cfg["Plot!prog"] = plot("Progress s [m] along the corridor", [pth(dbg("s"), "s [m]", "#3F51B5")])

cfg["Indicator!off"] = indicator("Off-track status (1 = OUT of corridor)", dbg("off"),
                                 [rule(">", 0.5, "#b71c1c", "OFF-TRACK"), rule("<", 0.5, "#1b5e20", "in corridor")],
                                 fallback_color="#1b5e20", fallback_label="in corridor", font=20)
cfg["Indicator!lateral"] = indicator("Lateral-error status", dbg("lateral"),
                                     [rule(">", 0.3, "#b71c1c", "Diverged (e > +0.3)"),
                                      rule("<", -0.3, "#b71c1c", "Diverged (e < -0.3)"),
                                      rule(">", 0.1, "#f9a825", "Warning (e > +0.1)"),
                                      rule("<", -0.1, "#f9a825", "Warning (e < -0.1)")],
                                     fallback_color="#1b5e20", fallback_label="Tracking", font=20)
cfg["Indicator!solve"] = indicator("Solve-time budget (ms)", dbg("solve_ms"),
                                   [rule(">", 50, "#b71c1c", "OVER 20 Hz budget"),
                                    rule(">", 40, "#f9a825", "near budget"),
                                    rule("<", 40, "#1b5e20", "OK")],
                                   fallback_color="#424242", fallback_label="—", font=20)

cfg["StateTransitions!switch"] = state_transitions("Enable / off-track timeline",
                                                   [(f"{A.enable}.data", "MPCC enabled"), (dbg("off"), "off-track")])
cfg["Log!log"] = log_panel(f"{A.node} log (rosout, INFO+)", A.node)
cfg["RawMessages!drive"] = raw("Drive cmd", A.drive)
cfg["RawMessages!enable"] = raw("Enable latch", f"{A.enable}.data")
cfg["RawMessages!obstacles"] = raw("Obstacles (race)", A.obstacles, expanded=True)

# ---------------------------------------------------------------- tabs (mirror the D: dashboard's intent)
def col(a, b, split): return {"direction": "column", "first": a, "second": b, "splitPercentage": split}
def row(a, b, split): return {"direction": "row", "first": a, "second": b, "splitPercentage": split}

cfg["Tab!root"] = {"activeTabIdx": 1, "tabs": [
    {"title": "Setup", "layout":
        row("Parameters!params",
            {"direction": "column", "first": "Publish!enable_on",
             "second": col("Publish!enable_off", "Publish!clear_trail", 50), "splitPercentage": 34}, 60)},
    {"title": "Overview", "layout":
        col(row("3D!view", col("Plot!speed", "Plot!drive", 50), 48),
            row("Indicator!off", row("Indicator!lateral", "RawMessages!drive", 55), 34), 66)},
    {"title": "Online tuning", "layout":
        row(col("Plot!weights", "Plot!prog", 64),
            col("Plot!track", col("Plot!solve", "Plot!beta", 50), 50), 58)},
    {"title": "Health", "layout":
        col(row("Plot!solve", "StateTransitions!switch", 50),
            row("Log!log", col("RawMessages!drive", "RawMessages!enable", 50), 55), 38)},
    {"title": "Obstacles (race)", "layout": row("3D!obs", "RawMessages!obstacles", 68)},
]}

layout = {"$schema": "https://foxglove.dev/schemas/studio-layout.json",
          "configById": cfg, "globalVariables": {}, "userNodes": user_nodes,
          "playbackConfig": {"speed": 1.0}, "layout": row("Parameters!params", "Tab!root", 16)}

Path(A.out).write_text(json.dumps(layout, indent=2))

# ---- sanity: every panel id referenced in the top layout + tab trees must exist in configById ----
def ids(n):
    if isinstance(n, str):
        return {n}
    out = set()
    for k in ("first", "second"):
        if k in n:
            out |= ids(n[k])
    return out
ref = ids(layout["layout"])
for t in cfg["Tab!root"]["tabs"]:
    ref |= ids(t["layout"])
ref.discard("Tab!root")
missing = ref - set(cfg)
assert not missing, f"layout references missing panels: {missing}"
npanels = len([k for k in cfg if not k.startswith("Tab!")])
print(f"wrote {A.out}  ({npanels} panels, {len(cfg['Tab!root']['tabs'])} tabs, "
      f"{len(user_nodes)} user scripts, drive={A.drive})")
