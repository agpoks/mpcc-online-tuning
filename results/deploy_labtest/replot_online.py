#!/usr/bin/env python3
"""Replot the online-learning run cleanly from the saved npz (no re-drive): filter the spurious
off-recovery 'laps', smooth the weight-adaptation trend out of the per-tick exploration dither."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT = Path("/home/poxx/github/mpcc-online-tuning")
OUT = ROOT / "results/deploy_labtest"
d = np.load(OUT / "online_learn.npz", allow_pickle=True)
th = d["th_hist"]; xy = d["xy"]; vh = d["vh"]; wn = list(d["weight_names"])
lt = d["lap_times"]; lb = list(d["lap_banked"])
DT = 0.05
FROZEN_FAST, SEED2_FAST = 20.2, 14.6
FROZEN_MEANV, SEED2_MEANV = 1.16, 1.53

# real laps = lap-boundary events with a plausible lap time (drop the <3s off-recovery artifacts)
real = [(i, t, b) for i, (t, b) in enumerate(zip(lt, lb)) if t > 3.0]
rt = [t for _, t, _ in real]; rb = [b for _, _, b in real]
banked_fast = min([t for t, b in zip(lt, lb) if b == "banked"], default=None)

fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 2)
# (A) learning curve (real laps only)
axA = fig.add_subplot(gs[0, 0])
cols = ["tab:green" if b == "banked" else "tab:red" for b in rb]
axA.plot(range(1, len(rt) + 1), rt, "-", color="0.6", lw=1.0, zorder=1)
axA.scatter(range(1, len(rt) + 1), rt, c=cols, s=55, zorder=3, edgecolor="k", lw=0.4,
            label="completed lap (green=banked / red=reverted)")
axA.axhline(FROZEN_FAST, ls="--", color="tab:orange", lw=1.3, label=f"frozen icra fastest {FROZEN_FAST}s")
axA.axhline(SEED2_FAST, ls="--", color="tab:blue", lw=1.3, label=f"offline seed2 fastest {SEED2_FAST}s")
if banked_fast:
    axA.axhline(banked_fast, ls=":", color="tab:green", lw=1.6, label=f"online BANKED {banked_fast:.1f}s")
axA.set_xlabel("completed lap #"); axA.set_ylabel("lap time [s]")
axA.set_title("Online learning on lab_test: completed-lap times"); axA.grid(alpha=0.3); axA.legend(fontsize=8)
# (B) weights adapting -- rolling-mean trend (strip the per-tick exploration dither)
axB = fig.add_subplot(gs[0, 1]); tmin = np.arange(len(th)) * DT / 60.0
W = 600
def roll(a):
    return np.convolve(a, np.ones(W) / W, mode="valid")
for w, c in zip(["k_v", "q_c", "q_l", "r_a"], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    i = wn.index(w); tw = th[:, i]
    axB.plot(tmin[W - 1:], roll(tw) / (np.abs(tw[0]) + 1e-9), color=c, lw=1.8, label=f"{w} (start {tw[0]:.2g})")
axB.set_xlabel("time [min]"); axB.set_ylabel("weight / initial (20 s rolling mean)")
axB.set_title("Weights adapting online (trend, dither removed)"); axB.grid(alpha=0.3); axB.legend(fontsize=8)
# (C) trajectory over time
axC = fig.add_subplot(gs[1, 0])
cd = np.load(ROOT / "mpcc_tuning/tracks/lab_test_ev8_corridor.npz")
cx, cy, wl, wr = cd["cx"], cd["cy"], cd["wl"], cd["wr"]
t = np.gradient(np.column_stack([cx, cy]), axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
nn = np.column_stack([-t[:, 1], t[:, 0]])
axC.plot((np.column_stack([cx, cy]) + wl[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) + wl[:, None] * nn)[:, 1], color="0.6", lw=1.0)
axC.plot((np.column_stack([cx, cy]) - wr[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) - wr[:, None] * nn)[:, 1], color="0.6", lw=1.0)
pts = xy.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, len(xy) * DT / 60.0), lw=1.3)
lc.set_array(np.arange(len(xy) - 1) * DT / 60.0); axC.add_collection(lc)
plt.colorbar(lc, ax=axC, fraction=0.046, label="time [min]")
axC.set_aspect("equal"); axC.set_xlabel("x [m]"); axC.set_ylabel("y [m]"); axC.set_title("Trajectory over the learning drive")
# (D) speed rolling mean vs references
axD = fig.add_subplot(gs[1, 1]); w2 = 300
rm = np.convolve(vh, np.ones(w2) / w2, mode="valid")
axD.plot(tmin[w2 - 1:], rm, color="tab:purple", lw=1.6, label="online learner (15 s mean)")
axD.axhline(FROZEN_MEANV, ls="--", color="tab:orange", lw=1.2, label=f"frozen icra {FROZEN_MEANV} m/s")
axD.axhline(SEED2_MEANV, ls="--", color="tab:blue", lw=1.2, label=f"offline seed2 {SEED2_MEANV} m/s")
axD.set_xlabel("time [min]"); axD.set_ylabel("speed [m/s]"); axD.set_title("Pace over the learning drive"); axD.grid(alpha=0.3); axD.legend(fontsize=8)
off_ticks = int((vh < 0).sum())  # placeholder; off count is in the log
fig.suptitle(f"Online-learning MPCC live on lab_test_ev8 — banks a {banked_fast:.1f}s net "
             f"(< frozen {FROZEN_FAST}s and offline seed2 {SEED2_FAST}s); exploratory, keep-best recovers excursions",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "online_learn.png", dpi=130); fig.savefig(OUT / "online_learn.pdf")
print(f"real laps kept: {len(rt)} (of {len(lt)} boundaries); banked fastest {banked_fast}s")
print("rewrote results/deploy_labtest/online_learn.{png,pdf}")
