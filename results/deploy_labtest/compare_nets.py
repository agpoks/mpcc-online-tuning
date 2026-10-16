#!/usr/bin/env python3
"""Fair comparison: drive the frozen icra qualifier + each trained-on-lab_test net on the COMMITTED
lab_test corridor under identical conditions (s=0, v0=1.0, kv_launch=0.45, 1200 steps). Report laps/clean/
vmax/solve and plot all trajectories."""
import sys, os, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning")
sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from node.mpcc_online.controller import OnlineMPCCController
SP = sys.argv[1]
CORR = "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
CANDS = [("icra-frozen", str(ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz"))]
for f in sorted(glob.glob(str(ROOT / "results/deploy_labtest/best_policy_*_time_return.npz"))):
    CANDS.append(("labtest-s" + Path(f).stem.split("_")[5], f))   # seed index (best_policy_icra_t2_smooth_<seed>_...)

STEPS = 1200
rows = []; trajs = {}
for name, net in CANDS:
    try:
        ctl = OnlineMPCCController(net, cfg=dict(raceline_npz=CORR, kv_launch=0.45))
    except Exception as e:
        print(f"{name:16s}: FAILED to load ({e})"); continue
    P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
    P.reset(s0=0.0, v0=1.0); ctl.reset()
    base = float(P.state5()[4]); off = False; tr = []; sv = []
    for k in range(STEPS):
        x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
        out = ctl.control(x, y, psi, vx, vy, r, delta)
        tr.append((x, y, float(np.hypot(vx, vy)))); sv.append(out["solve_ms"])
        u = out["u0"]
        if u is None:
            u = np.array([0.0, -2.0, 0.0])
        _, _, off, _ = P.step(u)
        if off:
            break
    tr = np.array(tr); laps = (float(P.state5()[4]) - base) / ctl.track.length
    meanv = tr[:, 2][tr[:, 2] > 0.3].mean() if (tr[:, 2] > 0.3).any() else 0.0
    rows.append((name, laps, not off, tr[:, 2].max(), meanv, np.percentile(sv, 50), np.percentile(sv, 95)))
    trajs[name] = (tr, not off)
    print(f"{name:16s}: laps={laps:5.2f} clean={not off!s:5} vmax={tr[:,2].max():.2f} meanv={meanv:.2f} "
          f"solve p50={np.percentile(sv,50):.0f} p95={np.percentile(sv,95):.0f}ms")

print("\n=== ranking (clean first, then laps) ===")
for name, laps, clean, vmax, meanv, p50, p95 in sorted(rows, key=lambda r: (not r[2], -r[1])):
    print(f"  {name:16s} {'CLEAN' if clean else 'off  '} laps {laps:.2f}  mean {meanv:.2f} m/s")

# save raw comparison data (reproducibility)
np.savez(f"{SP}/compare_nets.npz",
         names=np.array([r[0] for r in rows]),
         laps=np.array([r[1] for r in rows]), clean=np.array([r[2] for r in rows]),
         vmax=np.array([r[3] for r in rows]), meanv=np.array([r[4] for r in rows]),
         p50=np.array([r[5] for r in rows]), p95=np.array([r[6] for r in rows]),
         **{f"traj_{n}": trajs[n][0] for n in trajs})

# plot
d = np.load(CORR); cx, cy, wl, wr, rl = d["cx"], d["cy"], d["wl"], d["wr"], d["raceline"]
t = np.gradient(np.column_stack([cx, cy]), axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
nn = np.column_stack([-t[:, 1], t[:, 0]])
left = np.column_stack([cx, cy]) + wl[:, None] * nn; right = np.column_stack([cx, cy]) - wr[:, None] * nn
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(left[:, 0], left[:, 1], color="0.5", lw=1.0); ax.plot(right[:, 0], right[:, 1], color="0.5", lw=1.0, label="boundaries")
ax.plot(rl[:, 0], rl[:, 1], "--", color="tab:red", lw=0.8, label="raceline")
cols = plt.cm.tab10(np.linspace(0, 1, len(trajs)))
for (name, (tr, clean)), c in zip(trajs.items(), cols):
    laps = [r[1] for r in rows if r[0] == name][0]
    ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.6, label=f"{name}: {'clean' if clean else 'OFF'} {laps:.2f} lap")
    if not clean:
        ax.plot(tr[-1, 0], tr[-1, 1], "X", color=c, ms=12, mec="k")
ax.plot(cx[0], cy[0], "ko", ms=7, label="start")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("lab_test_ev8: frozen icra qualifier vs trained-on-lab_test nets (committed corridor, s=0)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{SP}/compare_nets.png", dpi=130); fig.savefig(f"{SP}/compare_nets.pdf")
print("wrote compare_nets.png + .pdf")
