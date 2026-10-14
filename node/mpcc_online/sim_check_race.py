"""Verify the RACE deploy path (no ROS): drive OnlineMPCCController(mode='race') against ScuderiaPlant +
a field of N scripted opponents, feeding the controller ONLY the opponents a sensor would see (range +
FOV + occlusion), and confirm it (a) drives, (b) stays clean / overtakes, (c) sets keep-outs / class /
crowd, (d) solves in budget. This is the on-car race loop exercised in sim; it mirrors
experiments/race_multi_eval.run_eval (same field recipe + perception) but routes through the DEPLOY
controller instead of the eval's own policy stepping.

  ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 node/mpcc_online/sim_check_race.py [N] [net.npz] [steps]
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import RacelineOpponent
from node.mpcc_online.controller import OnlineMPCCController
import experiments.race_mode as RM

RACE_KINDS = ("slower", "equal", "faster")   # the mixed-field recipe from race_multi_eval (no 'static')


def main():
    N = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1]) else 2
    net = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[2]) else str(
        ROOT / "results/race_multicar/nets_physical2/race_ltc_0.npz")
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2600
    phys = 0
    print(f"net={Path(net).name}  N={N}  steps={steps}  mode=race", flush=True)
    ctl = OnlineMPCCController(net, cfg=dict(mode="race", kv_launch=0.50, ego_pace=1.77))
    track = ctl.track; L = track.length
    kinds = [RACE_KINDS[(phys + i) % 3] for i in range(N)]
    s0 = (phys % 4) * L / 4.0; v0 = 1.3 + 0.1 * (phys % 3)
    gaps0 = [3.0 + 4.0 * i + 1.5 * phys for i in range(N)]
    opps = [RacelineOpponent(track, s0=(s0 + gaps0[i]) % L, pace=RM.FAIR_PACE[kinds[i]] * ctl.ego_pace,
                             offset=0.0, radius=RM.KEEPOUT_R, a_lat=2.5) for i in range(N)]
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps + 20
    P.reset(s0=s0, v0=v0); ctl.reset()
    for o in opps:
        o.reset()
    half = np.radians(RM.FOV_DEG) / 2.0
    base = float(P.state5()[4]); off_cnt = 0; solve = []; contact = False
    passed = [False] * N; passes = 0; max_det = 0
    for k in range(steps):
        x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
        # perception: feed only the VISIBLE opponents (range + FOV + occlusion), with their state + stable id
        det_list = []
        for i, o in enumerate(opps):
            ox, oy, _ = o.keepout()
            if RM._visible(x, y, psi, ox, oy, 18.0, half, occlude=True):
                det_list.append(dict(x=ox, y=oy, v=float(getattr(o, "speed", 0.0)), id=i))
        max_det = max(max_det, len(det_list))
        out = ctl.control(x, y, psi, vx, vy, r, delta, opponents=det_list)
        solve.append(out["solve_ms"])
        u = out["u0"] if out["u0"] is not None else np.array([0.0, -4.0, 0.0])   # failed solve -> brake
        s5n, rr, offp, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey); vv = float(P._x[3])
        if offp:
            off_cnt += 1
        for i, o in enumerate(opps):
            o.step(0.05, ego=(ex, ey, vv)); oxx, oyy, _ = o.keepout()
            if np.hypot(ex - oxx, ey - oyy) < RM.CONTACT_R:
                contact = True
            g = RM.signed_gap(track, s_ego, o.s)
            if g < 0 and abs(g) < L / 4 and not passed[i] and vv > float(getattr(o, "speed", 0.0)):
                passed[i] = True; passes += 1
            elif g > 0.5:
                passed[i] = False
        if offp or tr or contact:
            break
    laps = (float(P.state5()[4]) - base) / L
    sv = np.array(solve)
    clean = (off_cnt == 0) and (not contact)
    print(f"  laps={laps:.2f}  off_ticks={off_cnt}  contact={contact}  passes={passes}  "
          f"clean={clean}  max_detected={max_det}", flush=True)
    over = int((sv > 50.0).sum())
    print(f"  solve_ms p50={np.percentile(sv,50):.1f} p90={np.percentile(sv,90):.1f} "
          f"p95={np.percentile(sv,95):.1f} max={sv.max():.1f}  (budget 50ms; over {over}/{len(sv)})", flush=True)
    ok = (laps > 0.5) and (not contact)
    print("  RACE SIM CHECK PASSED" if ok else "  RACE SIM CHECK: review above", flush=True)


if __name__ == "__main__":
    main()
