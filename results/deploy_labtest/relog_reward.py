#!/usr/bin/env python3
"""Re-drive the SAME online-learning run (deterministic: tuner RNG = default_rng(seed), plant deterministic)
and log the AUTHORITATIVE per-tick reward the controller uses: reward = dprog - 5*off, with off taken from the
controller's own out['off'] (not reconstructed by projection). Verifies the lap times reproduce the committed
online_learn.npz, then AUGMENTS that npz with per-tick reward/off/return so the figure is exact + reproducible
without a re-drive next time."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.mpcc import WEIGHT_NAMES
from node.mpcc_online.controller import OnlineMPCCController
OUT = ROOT / "results/deploy_labtest"
CORR = "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
START_NET = str(ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz")
STEPS = 6000

ctl = OnlineMPCCController(START_NET, cfg=dict(learn=True, kv_launch=0.45, raceline_npz=CORR))
L = ctl.track.length
P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
P.reset(s0=0.0, v0=1.0); ctl.reset()
base = float(P.state5()[4])
th_hist = []; xy = []; vh = []; sv = []; reward = []; offh = []
lap_end_tick = []; lap_banked = []; prev_s = base
for k in range(STEPS):
    x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
    out = ctl.control(x, y, psi, vx, vy, r, delta)
    # dprog from the plant arc-length (same track the controller projects onto), wrap-corrected
    dprog = s - prev_s; dprog += L if dprog < -0.5 * L else (-L if dprog > 0.5 * L else 0.0)
    prev_s = s
    rw = dprog - 5.0 * float(out["off"])
    th_hist.append(out["theta"]); xy.append((x, y)); vh.append(float(np.hypot(vx, vy)))
    sv.append(out["solve_ms"]); reward.append(rw); offh.append(bool(out["off"]))
    if out["banked"] is not None:
        lap_end_tick.append(k); lap_banked.append(out["banked"])
    u = out["u0"]
    if u is None:
        u = np.array([0.0, -2.0, 0.0])
    P.step(u)
th_hist = np.array(th_hist); xy = np.array(xy); vh = np.array(vh); sv = np.array(sv)
reward = np.array(reward); offh = np.array(offh); ret = np.cumsum(reward)
laps = (float(P.state5()[4]) - base) / L
lt = np.diff([0] + lap_end_tick).astype(float) * ctl.dt if lap_end_tick else np.array([])
banks = sum(b == "banked" for b in lap_banked)

# reproducibility check against the committed npz
old = np.load(OUT / "online_learn.npz", allow_pickle=True)
old_lt = np.asarray(old["lap_times"], float)
match = len(old_lt) == len(lt) and np.allclose(old_lt, lt, atol=1e-6)
print(f"re-drive: laps={laps:.3f} off_ticks={int(offh.sum())} banks={banks}/{len(lap_banked)} "
      f"return={ret[-1]:.1f}")
print(f"lap_times reproduce committed npz: {match}  (n={len(lt)} vs {len(old_lt)})")
if not match:
    print("  committed:", [round(float(x), 2) for x in old_lt])
    print("  re-drive :", [round(float(x), 2) for x in lt])

np.savez(OUT / "online_learn.npz", th_hist=th_hist, xy=xy, vh=vh, solve=sv,
         lap_end_tick=np.array(lap_end_tick), lap_banked=np.array(lap_banked, dtype=object),
         lap_times=lt, laps=laps, banks=banks, weight_names=np.array(WEIGHT_NAMES),
         start_net=Path(START_NET).name, reward=reward, off=offh, ret=ret)
print(f"augmented online_learn.npz with per-tick reward/off/return (off_ticks={int(offh.sum())})")
