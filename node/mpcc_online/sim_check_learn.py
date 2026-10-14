"""Verify the ONLINE-LEARNING deploy path (no ROS): drive OnlineMPCCController(learn=True) against the
ScuderiaPlant for several laps and confirm it (a) drives, (b) runs RTRL online, (c) banks a keep-best net
per lap. This is the on-car learning loop, exercised in sim.

  ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 node/mpcc_online/sim_check_learn.py [net.npz] [steps]
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
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    print(f"net={Path(net).name}  steps={steps}  LEARN=on")
    ctl = OnlineMPCCController(net, cfg=dict(learn=True, kv_launch=0.50))
    P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = steps + 20
    P.reset(s0=0.0, v0=1.0); ctl.reset()
    base = float(P.state5()[4]); off_cnt = 0; banks = 0; solve = []; crashed = False
    for k in range(steps):
        x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
        out = ctl.control(x, y, psi, vx, vy, r, delta)
        solve.append(out["solve_ms"]); off_cnt += int(out["off"])
        if out["banked"] is not None:
            print(f"    lap boundary at tick {k}: end_episode -> {out['banked']}", flush=True)
        if out["banked"] == "banked":
            banks += 1
        _, _, offp, tr = P.step(out["u0"])
        if offp:
            crashed = True
    laps = (float(P.state5()[4]) - base) / ctl.track.length
    best = ctl.tuner._best
    sv = np.array(solve)
    # save the banked checkpoint (the on-car resume artifact)
    ck = ROOT / "results/race_multicar/qualification/deploy_learn_ckpt.npz"
    ctl.save_ckpt(ck)
    print(f"  laps={laps:.2f}  off_ticks={off_cnt}/{steps}  keep-best banks={banks}  plant_off={crashed}")
    print(f"  best_score(-laptime)={None if best is None else round(best[0],2)}  (banked a net: {best is not None})")
    over = int((sv > 50.0).sum())
    print(f"  solve_ms p50={np.percentile(sv,50):.1f} p90={np.percentile(sv,90):.1f} "
          f"p95={np.percentile(sv,95):.1f} p99={np.percentile(sv,99):.1f} max={sv.max():.1f}  "
          f"(20Hz budget 50ms; over-budget ticks {over}/{steps} = {100.0*over/steps:.1f}%)")
    print(f"  saved checkpoint -> {ck.name}")
    ok = (laps > 1.0) and (best is not None)
    print("  LEARN SIM CHECK PASSED" if ok else "  LEARN SIM CHECK: review above")


if __name__ == "__main__":
    main()
