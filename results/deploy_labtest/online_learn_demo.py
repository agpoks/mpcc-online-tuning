#!/usr/bin/env python3
"""ONLINE-LEARNING demo on the fixed lab_test_ev8 corridor: start from the TRANSFERRED frozen icra net and
let the on-car PolicyTuner (RTRL + per-lap keep-best) ADAPT the weights live. Shows the method the project is
about -- learning ON the car on a new track -- vs the frozen baseline and the offline-trained seed2.
Tracks lap times (learning curve), weights adapting, trajectory over time. Saves results/deploy_labtest/
online_learn.{png,npz} + the online-learned checkpoint."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.mpcc import WEIGHT_NAMES
from node.mpcc_online.controller import OnlineMPCCController
OUT = ROOT / "results/deploy_labtest"
CORR = "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
START_NET = str(ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

ctl = OnlineMPCCController(START_NET, cfg=dict(learn=True, kv_launch=0.45, raceline_npz=CORR))
L = ctl.track.length
P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
P.reset(s0=0.0, v0=1.0); ctl.reset()
base = float(P.state5()[4])
th_hist = []; xy = []; vh = []; sv = []; off_ticks = 0
lap_end_tick = []; lap_banked = []; last_tick0 = 0
for k in range(STEPS):
    x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
    out = ctl.control(x, y, psi, vx, vy, r, delta)
    th_hist.append(out["theta"]); xy.append((x, y)); vh.append(float(np.hypot(vx, vy))); sv.append(out["solve_ms"])
    off_ticks += int(out["off"])
    if out["banked"] is not None:                 # a lap boundary (banked or reverted)
        lap_end_tick.append(k); lap_banked.append(out["banked"])
    u = out["u0"]
    if u is None:
        u = np.array([0.0, -2.0, 0.0])
    _, _, offp, _ = P.step(u)
th_hist = np.array(th_hist); xy = np.array(xy); vh = np.array(vh); sv = np.array(sv)
laps = (float(P.state5()[4]) - base) / L
# per-lap times from lap boundaries
lt = np.diff([0] + lap_end_tick).astype(float) * ctl.dt if lap_end_tick else np.array([])
banks = sum(b == "banked" for b in lap_banked)
best = ctl.tuner._best
ck = OUT / "online_learned_ckpt.npz"; ctl.save_ckpt(str(ck))
print(f"ONLINE LEARN on lab_test: laps={laps:.2f} off_ticks={off_ticks}/{STEPS} keep-best banks={banks}/"
      f"{len(lap_banked)} boundaries  best(-laptime)={None if best is None else round(best[0],2)}")
print(f"  lap times [s]: {[round(x,1) for x in lt.tolist()]}")
print(f"  solve p50={np.percentile(sv,50):.0f} p95={np.percentile(sv,95):.0f} ms")
print(f"  saved {ck.name}")

# reference lines (from the 5-min hardening): frozen icra 20.2s fastest / seed2 14.6s fastest
FROZEN_FAST, SEED2_FAST = 20.2, 14.6
np.savez(OUT / "online_learn.npz", th_hist=th_hist, xy=xy, vh=vh, solve=sv, lap_end_tick=np.array(lap_end_tick),
         lap_banked=np.array(lap_banked, dtype=object), lap_times=lt, laps=laps, banks=banks,
         weight_names=np.array(WEIGHT_NAMES), start_net=Path(START_NET).name)

# ---------- figure ----------
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2)
# (A) learning curve: lap time vs lap #
axA = fig.add_subplot(gs[0, 0])
cols = ["tab:green" if b == "banked" else "tab:red" for b in lap_banked]
axA.plot(range(1, len(lt) + 1), lt, "-", color="0.6", lw=1.0, zorder=1)
axA.scatter(range(1, len(lt) + 1), lt, c=cols, s=45, zorder=3, label="lap (green=banked / red=reverted)")
axA.axhline(FROZEN_FAST, ls="--", color="tab:orange", lw=1.2, label=f"frozen icra fastest {FROZEN_FAST}s")
axA.axhline(SEED2_FAST, ls="--", color="tab:blue", lw=1.2, label=f"offline seed2 fastest {SEED2_FAST}s")
axA.set_xlabel("lap #"); axA.set_ylabel("lap time [s]")
axA.set_title("Online learning on lab_test: lap time per lap (lower=faster)"); axA.grid(alpha=0.3); axA.legend(fontsize=8)
# (B) weights adapting over time
axB = fig.add_subplot(gs[0, 1])
tmin = np.arange(len(th_hist)) * ctl.dt / 60.0
for w in ["k_v", "q_c", "q_l", "r_a"]:
    i = WEIGHT_NAMES.index(w)
    tw = th_hist[:, i]; axB.plot(tmin, tw / (np.abs(tw[0]) + 1e-9), lw=1.3, label=f"{w} (×{tw[0]:.2g})")
axB.set_xlabel("time [min]"); axB.set_ylabel("weight / initial"); axB.set_title("Weights adapting online (normalised to start)")
axB.grid(alpha=0.3); axB.legend(fontsize=8)
# (C) trajectory colored by time (early->late)
axC = fig.add_subplot(gs[1, 0])
d = np.load(CORR); cx, cy, wl, wr, rl = d["cx"], d["cy"], d["wl"], d["wr"], d["raceline"]
t = np.gradient(np.column_stack([cx, cy]), axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
nn = np.column_stack([-t[:, 1], t[:, 0]])
axC.plot((np.column_stack([cx, cy]) + wl[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) + wl[:, None] * nn)[:, 1], color="0.6", lw=1.0)
axC.plot((np.column_stack([cx, cy]) - wr[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) - wr[:, None] * nn)[:, 1], color="0.6", lw=1.0)
pts = xy.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, len(xy) * ctl.dt / 60.0), lw=1.4)
lc.set_array(np.arange(len(xy) - 1) * ctl.dt / 60.0); axC.add_collection(lc)
plt.colorbar(lc, ax=axC, fraction=0.046, label="time [min]")
axC.set_aspect("equal"); axC.set_xlabel("x [m]"); axC.set_ylabel("y [m]"); axC.set_title("Trajectory over the drive (blue=early, yellow=late)")
# (D) speed over time
axD = fig.add_subplot(gs[1, 1])
axD.plot(tmin, vh, color="tab:purple", lw=0.7)
# rolling mean
w = 200; rm = np.convolve(vh, np.ones(w) / w, mode="valid")
axD.plot(tmin[w - 1:], rm, color="k", lw=1.6, label="rolling mean")
axD.set_xlabel("time [min]"); axD.set_ylabel("speed [m/s]"); axD.set_title("Speed over the drive"); axD.grid(alpha=0.3); axD.legend(fontsize=8)
fig.suptitle(f"Online-learned MPCC on lab_test_ev8 (from the transferred frozen net) — {laps:.1f} laps, "
             f"{banks} keep-best banks, 0 crash" if off_ticks == 0 else
             f"Online-learned MPCC on lab_test_ev8 — {laps:.1f} laps, {banks} banks, {off_ticks} off-ticks (recovered)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "online_learn.png", dpi=130); fig.savefig(OUT / "online_learn.pdf")
print("wrote results/deploy_labtest/online_learn.{png,pdf,npz}")
