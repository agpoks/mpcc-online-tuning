#!/usr/bin/env python3
"""Drive the ONLINE-LEARNED BANKED net (what the live on-car learner banked on lab_test) FROZEN, to show the
clean deployable pace online learning produced. Converts the keep-best ckpt (th0 key) to net format, drives
5-min continuous, reports laps/fastest/clean vs the frozen-icra baseline + offline seed2."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from node.mpcc_online.controller import OnlineMPCCController
OUT = ROOT / "results/deploy_labtest"
CORR = "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"

# convert ckpt (th0) -> net format (theta0) so the controller can load it frozen
ck = np.load(OUT / "online_learned_ckpt.npz", allow_pickle=True)
net = OUT / "online_learned_net.npz"
np.savez(net, G=ck["G"], cell_p=ck["cell_p"], theta0=ck["th0"], lo=ck["lo"], hi=ck["hi"],
         hidden=int(ck["hidden"]), seed=int(ck["seed"]), critic="return", clock="time",
         best_laps=float(ck["best_score"]))
print(f"converted ckpt -> {net.name}  (banked best_score -laptime = {float(ck['best_score']):.2f})")

STEPS = 6000
ctl = OnlineMPCCController(str(net), cfg=dict(raceline_npz=CORR, kv_launch=0.45))   # learn OFF = frozen
L = ctl.track.length
P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
P.reset(s0=0.0, v0=1.0); ctl.reset()
base = float(P.state5()[4]); off = False; lap_steps = []; nxt = 1; vs = []; sv = []
for k in range(STEPS):
    x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
    out = ctl.control(x, y, psi, vx, vy, r, delta); sv.append(out["solve_ms"]); vs.append(float(np.hypot(vx, vy)))
    u = out["u0"]
    if u is None:
        u = np.array([0.0, -2.0, 0.0])
    _, _, off, _ = P.step(u)
    prog = (float(P.state5()[4]) - base) / L
    while prog >= nxt:
        lap_steps.append(k); nxt += 1
    if off:
        break
laps = (float(P.state5()[4]) - base) / L
lt = np.diff([0] + lap_steps).astype(float) * 0.05 if lap_steps else np.array([])
print(f"ONLINE-LEARNED net FROZEN on lab_test: full_laps={len(lap_steps)} laps={laps:.2f} clean={not off} "
      f"fastest={round(float(lt.min()),2) if lt.size else None}s meanV={np.mean([v for v in vs if v>0.3]):.3f} "
      f"solve p50={np.percentile(sv,50):.0f} p95={np.percentile(sv,95):.0f}ms")
print("  vs frozen-icra 12 laps/20.2s/1.16 ; offline-seed2 16 laps/14.6s/1.53")
