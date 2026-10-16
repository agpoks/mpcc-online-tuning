#!/usr/bin/env python3
"""5-minute hardening drive on the lab_test_ev8 deploy corridor: drive the trained seed-2 net (deploy
default) + the frozen icra qualifier FROZEN for 6000 steps (=300 s @ 20 Hz), counting continuous clean laps,
per-lap times (fastest lap) and mean speed -- the qual_5min check applied to the deploy map. Saves
results/deploy_labtest/labtest_5min.{json,png}. Uses the committed OnlineMPCCController + committed corridor.
"""
import sys, os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from node.mpcc_online.controller import OnlineMPCCController
OUT = ROOT / "results/deploy_labtest"
CORR = "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
STEPS = 6000; V0 = 1.0; S0 = 0.0; KV = 0.45
NETS = [("seed2 (deploy default)", str(OUT / "best_policy_icra_t2_smooth_2_time_return.npz")),
        ("icra-frozen (fallback)", str(ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz"))]


def drive(net):
    ctl = OnlineMPCCController(net, cfg=dict(raceline_npz=CORR, kv_launch=KV))
    L = ctl.track.length
    P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
    P.reset(s0=S0, v0=V0); ctl.reset()
    base = float(P.state5()[4]); off = False; crash_step = None
    lap_steps = []; next_lap = 1; vs = []; sv = []
    for k in range(STEPS):
        x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
        out = ctl.control(x, y, psi, vx, vy, r, delta); sv.append(out["solve_ms"])
        u = out["u0"]
        if u is None:
            u = np.array([0.0, -2.0, 0.0])
        _, _, off, _ = P.step(u); vs.append(float(np.hypot(vx, vy)))
        prog = (float(P.state5()[4]) - base) / L
        while prog >= next_lap:
            lap_steps.append(k); next_lap += 1
        if off:
            crash_step = k; break
    laps = (float(P.state5()[4]) - base) / L
    lt = np.diff([0] + lap_steps).astype(float) * 0.05 if lap_steps else np.array([])
    return dict(laps=round(laps, 3), clean=bool(not off), crash_step=crash_step, full_laps=len(lap_steps),
                lap_times=[round(x, 2) for x in lt.tolist()],
                fastest_lap=round(float(lt.min()), 2) if lt.size else None,
                mean_speed=round(float(np.mean([v for v in vs if v > 0.3])), 3),
                solve_p50=round(float(np.percentile(sv, 50)), 1), solve_p95=round(float(np.percentile(sv, 95)), 1),
                steps_run=len(vs))


OUT.mkdir(parents=True, exist_ok=True)
rows = []
for name, net in NETS:
    r = drive(net); r["name"] = name; r["net"] = Path(net).name; rows.append(r)
    ft = f"{r['fastest_lap']}s" if r["fastest_lap"] else "--"
    print(f"{name:26s}: {r['full_laps']} full laps ({r['laps']} progress) clean={r['clean']} "
          f"fastest={ft} meanV={r['mean_speed']} solve p50={r['solve_p50']} p95={r['solve_p95']}ms", flush=True)
json.dump(rows, open(OUT / "labtest_5min.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for r in rows:
    if r["lap_times"]:
        ax.plot(range(1, len(r["lap_times"]) + 1), r["lap_times"], "-o",
                label=f"{r['name']}: {r['full_laps']} laps, fastest {r['fastest_lap']}s, meanV {r['mean_speed']}")
ax.set_xlabel("lap #"); ax.set_ylabel("lap time [s]")
ax.set_title("lab_test_ev8 — 5-min continuous drive (lower = faster)"); ax.grid(alpha=0.3)
if ax.get_legend_handles_labels()[0]:
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "labtest_5min.png", dpi=150)
print("saved results/deploy_labtest/labtest_5min.{json,png}")
