"""Verify the deploy controller core WITHOUT ROS or a car: drive OnlineMPCCController against the
ScuderiaPlant for a few laps and report laps / clean / solve-time. Run on the dev box before flashing
the node. This exercises the exact odom->state->MPCC->drive path the ROS node uses.

  ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 node/mpcc_online/sim_check.py [net.npz] [steps]
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from node.mpcc_online.controller import OnlineMPCCController


def main():
    net = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1]) else str(
        ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    print(f"net={Path(net).name}  steps={steps}")
    ctl = OnlineMPCCController(net, cfg=dict(kv_launch=0.50))
    P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = steps + 20
    P.reset(s0=0.0, v0=1.0); ctl.reset()
    base = float(P.state5()[4]); off = False; solve = []; vmax = 0.0
    for k in range(steps):
        x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
        out = ctl.control(x, y, psi, vx, vy, r, delta)
        solve.append(out["solve_ms"]); vmax = max(vmax, float(np.hypot(vx, vy)))
        _, _, off, tr = P.step(out["u0"])
        if off:
            print(f"  OFF-TRACK at step {k} (s={out['s']:.1f})"); break
    laps = (float(P.state5()[4]) - base) / ctl.track.length
    sv = np.array(solve)
    print(f"  laps={laps:.2f}  clean={not off}  vmax={vmax:.2f} m/s")
    print(f"  solve_ms: p50={np.percentile(sv,50):.1f}  p95={np.percentile(sv,95):.1f}  "
          f"max={sv.max():.1f}  (20 Hz budget = 50 ms)")
    print(f"  last weights: " + "  ".join(f"{n}={w:.2f}" for n, w in
          zip(['q_c','q_l','q_v','r_d','r_a','r_dv','d_obs','k_v','d_bound'], ctl.last_theta)))
    print("  SIM CHECK PASSED" if (not off and laps > 1.0) else "  SIM CHECK: review above")


if __name__ == "__main__":
    main()
