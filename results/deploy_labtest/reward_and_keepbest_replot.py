#!/usr/bin/env python3
"""Re-plot the online-learning drive from the saved online_learn.npz (NO re-drive) with two things added that
the committed figure was missing:

  (1) the KEEP-BEST INCUMBENT ("take the best net") drawn ON the lap-time curve -- a staircase of the banked
      net's lap time, a gold star where it banks, and the net's true FROZEN pace as a reference band. This is
      what answers "why faster then slower again": the single 13.8 s lap was a lucky-dither lap; keep-best
      banks it, and because it was a fluke, every later lap runs the net's real ~20 s pace -> SLOWER than the
      incumbent -> reverted (weights snap back to the banked net). The oscillation is the exploration dither;
      the ~20 s floor is the net's actual capability (matches the 20.25 s frozen drive).

  (2) the ONLINE REWARD the learner actually sees, reconstructed EXACTLY as the controller computes it
      (controller._control_learn:175 reward = dprog - 5*off, dprog = projected progress since last tick),
      straight from the saved per-tick xy -- per-tick reward (pace + off-track -5 spikes) and the cumulative
      return. No re-run: the committed npz is the single source of truth.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
OUT = ROOT / "results/deploy_labtest"
CORR = ROOT / "mpcc_tuning/tracks/lab_test_ev8_corridor.npz"
DT = 0.05
FROZEN_FAST = 20.2        # the net's true frozen fastest lap (drive_online_learned.py: 20.25 s)
SEED2_FAST = 14.6
OFF_PEN = 5.0

# ---------- load the committed drive ----------
d = np.load(OUT / "online_learn.npz", allow_pickle=True)
xy = d["xy"]; th = d["th_hist"]; vh = d["vh"]; wn = [str(w) for w in d["weight_names"]]
lt = np.asarray(d["lap_times"], float); lb = [str(b) for b in d["lap_banked"]]
let = np.asarray(d["lap_end_tick"], int)
N = len(xy); tmin = np.arange(N) * DT / 60.0

# ---------- the per-tick reward: AUTHORITATIVE (logged by relog_reward.py) if present, else reconstruct ----------
g = np.load(CORR)
if "reward" in d.files and "off" in d.files:
    reward = np.asarray(d["reward"], float); off = np.asarray(d["off"], bool)
    ret = np.asarray(d["ret"], float) if "ret" in d.files else np.cumsum(reward)
    dprog = reward + OFF_PEN * off.astype(float)                        # reward = dprog - 5*off
    print(f"AUTHORITATIVE reward from npz: off_ticks={int(off.sum())}, total return={ret[-1]:.1f}, "
          f"mean progress reward={dprog[dprog>0].mean():.4f} m/tick")
else:
    trk = Track(g["cx"], g["cy"], ds=0.1, w_left=g["wl"], w_right=g["wr"])
    trk.raceline = g["raceline"]; trk.v_ref = g["vref"]; trk.width_vehicle_adjusted = False
    Lt = trk.length
    proj = np.array([float(trk.project(float(x), float(y))) for x, y in xy])
    lat = np.array([float(trk.lateral(float(x), float(y))) for x, y in xy])
    dprog = np.diff(proj, prepend=proj[0])
    dprog += np.where(dprog < -0.5 * Lt, Lt, 0.0) - np.where(dprog > 0.5 * Lt, Lt, 0.0)
    dprog[0] = 0.0
    off = np.zeros(N, bool)
    for i in range(N):
        wl, wr = trk.width(trk.wrap(proj[i]))
        off[i] = (lat[i] > float(wr) - 0.12) or (-lat[i] > float(wl) - 0.12)
    reward = dprog - OFF_PEN * off.astype(float); ret = np.cumsum(reward)
    print(f"reconstructed reward: off_ticks={int(off.sum())}, total return={ret[-1]:.1f}")

# ---------- real completed laps + keep-best incumbent ----------
real = [(int(let[i]), float(lt[i]), lb[i]) for i in range(len(lt)) if float(lt[i]) > 3.0]
rtick = [t for t, _, _ in real]; rtime = [v for _, v, _ in real]; rbank = [b for _, _, b in real]
rmin = [tk * DT / 60.0 for tk in rtick]
# keep-best incumbent = running best lap time among BANKED laps (keep-best banks on -laptime, reverts slower)
inc = []; cur = np.inf
for t, b in zip(rtime, rbank):
    if b == "banked":
        cur = min(cur, t)
    inc.append(cur)
bank_idx = [i for i, b in enumerate(rbank) if b == "banked"]

# ======================= FIGURE =======================
fig = plt.figure(figsize=(17.5, 9.2)); gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.26)

# (A) lap-time learning curve + keep-best incumbent ("take the best net")
axA = fig.add_subplot(gs[0, 0])
axA.axhspan(FROZEN_FAST - 1.3, max(rtime) + 0.5, color="tab:red", alpha=0.05)
cols = ["tab:green" if b == "banked" else ("tab:red" if b == "reverted" else "0.5") for b in rbank]
axA.plot(range(1, len(rtime) + 1), rtime, "-", color="0.6", lw=1.0, zorder=1)
axA.scatter(range(1, len(rtime) + 1), rtime, c=cols, s=60, zorder=3, edgecolor="k", lw=0.5)
# the incumbent net "in place" (staircase): flat once it banks, since the bank was never beaten
xs = np.arange(1, len(rtime) + 1); inc_arr = np.array(inc, float)
fin = np.isfinite(inc_arr)
axA.step(xs[fin], inc_arr[fin], where="post", color="tab:green", lw=2.6, zorder=2,
         label="keep-best net IN PLACE (incumbent)")
for i in bank_idx:
    axA.scatter([i + 1], [rtime[i]], marker="*", s=360, color="gold", edgecolor="k", lw=0.8, zorder=5)
    axA.annotate(f"keep-best BANKS\n{rtime[i]:.1f}s (lucky dither)", (i + 1, rtime[i]),
                 textcoords="offset points", xytext=(14, -6), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
axA.axhline(FROZEN_FAST, ls="--", color="k", lw=1.3, label=f"net's true FROZEN pace {FROZEN_FAST}s")
axA.text(0.5, FROZEN_FAST + 1.6, "later laps run the net's real ~20 s pace →\n"
         "slower than the bank → REVERTED (snap back)", fontsize=7.8, color="tab:red")
axA.set_xlabel("completed lap #"); axA.set_ylabel("lap time [s]")
axA.set_title("(A) Lap time + keep-best incumbent\ngreen=banked ★ / red=reverted"); axA.grid(alpha=0.3)
axA.legend(loc="lower right", fontsize=7.6)

# (B) ONLINE REWARD per tick (the signal the RTRL learner optimizes) + off-track penalties
axB = fig.add_subplot(gs[0, 1])
W = 200; rm = np.convolve(reward, np.ones(W) / W, mode="valid")
axB.axhline(0, color="0.7", lw=0.8)
axB.plot(tmin[:len(dprog)], dprog, color="0.78", lw=0.5, zorder=1, label="progress reward/tick (m)")
axB.plot(tmin[W - 1:], rm, color="tab:purple", lw=1.8, zorder=3, label="reward (10 s rolling mean)")
ot = np.where(off)[0]
axB.scatter(tmin[ot], reward[ot], color="tab:red", s=14, zorder=4, label=f"off-track −5 ({len(ot)} ticks)")
axB.set_xlabel("time [min]"); axB.set_ylabel("reward per tick")
axB.set_title("(B) Online reward  = progress − 5·off-track\n(controller._control_learn, logged on the drive)")
axB.grid(alpha=0.3); axB.legend(loc="lower left", fontsize=7.6)

# (C) cumulative return (what critic='return' fits) + where keep-best banks/reverts
axC = fig.add_subplot(gs[0, 2])
axC.plot(tmin, ret, color="tab:blue", lw=1.8, label="cumulative return Σ reward")
for tk, b in zip(rtick, rbank):
    tm = tk * DT / 60.0; k = min(tk, N - 1)
    axC.scatter([tm], [ret[k]], color=("tab:green" if b == "banked" else "tab:red"),
                s=55, edgecolor="k", lw=0.5, zorder=4)
axC.scatter([], [], color="tab:green", label="lap banked"); axC.scatter([], [], color="tab:red", label="lap reverted")
axC.set_xlabel("time [min]"); axC.set_ylabel("return")
axC.set_title("(C) Cumulative return\n(down-steps = off-track penalties)"); axC.grid(alpha=0.3)
axC.legend(loc="upper left", fontsize=7.6)

# (D) weights adapting -- rolling-mean trend (strip per-tick dither)
axD = fig.add_subplot(gs[1, 0]); Ww = 600
roll = lambda a: np.convolve(a, np.ones(Ww) / Ww, mode="valid")
for w, c in zip(["k_v", "q_c", "q_l", "r_a"], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    i = wn.index(w); tw = th[:, i]
    axD.plot(tmin[Ww - 1:], roll(tw) / (abs(tw[0]) + 1e-9), color=c, lw=1.8, label=f"{w} (start {tw[0]:.2g})")
axD.set_xlabel("time [min]"); axD.set_ylabel("weight / initial (30 s mean)")
axD.set_title("(D) Weights adapting online (trend)"); axD.grid(alpha=0.3); axD.legend(fontsize=7.6)

# (E) trajectory over time
axE = fig.add_subplot(gs[1, 1])
cx, cy, wl_, wr_ = g["cx"], g["cy"], g["wl"], g["wr"]
tg = np.gradient(np.column_stack([cx, cy]), axis=0); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
nn = np.column_stack([-tg[:, 1], tg[:, 0]])
axE.plot((np.column_stack([cx, cy]) + wl_[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) + wl_[:, None] * nn)[:, 1], color="0.6", lw=1.0)
axE.plot((np.column_stack([cx, cy]) - wr_[:, None] * nn)[:, 0], (np.column_stack([cx, cy]) - wr_[:, None] * nn)[:, 1], color="0.6", lw=1.0)
pts = xy.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, N * DT / 60.0), lw=1.3)
lc.set_array(np.arange(N - 1) * DT / 60.0); axE.add_collection(lc)
if ot.size:
    axE.scatter(xy[ot, 0], xy[ot, 1], color="tab:red", s=10, zorder=5, label="off-track")
    axE.legend(loc="upper right", fontsize=7.6)
plt.colorbar(lc, ax=axE, fraction=0.046, label="time [min]")
axE.set_aspect("equal"); axE.set_xlabel("x [m]"); axE.set_ylabel("y [m]"); axE.set_title("(E) Trajectory over the drive")

# (F) pace over time
axF = fig.add_subplot(gs[1, 2]); w2 = 300; rmv = np.convolve(vh, np.ones(w2) / w2, mode="valid")
axF.plot(tmin[w2 - 1:], rmv, color="tab:purple", lw=1.7, label="online learner (15 s mean)")
axF.axhline(1.16, ls="--", color="tab:orange", lw=1.2, label="frozen icra 1.16 m/s")
axF.axhline(1.53, ls="--", color="tab:blue", lw=1.2, label="offline seed2 1.53 m/s")
axF.set_xlabel("time [min]"); axF.set_ylabel("speed [m/s]"); axF.set_title("(F) Pace over the drive")
axF.grid(alpha=0.3); axF.legend(fontsize=7.6)

fig.suptitle("Online-learning MPCC on lab_test_ev8 — keep-best banks ONE lucky-dither lap (13.8 s) then "
             "reverts every slower lap to it; reward = progress − 5·off, all 38 excursions recovered",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "online_learn.png", dpi=130); fig.savefig(OUT / "online_learn.pdf")
print("rewrote results/deploy_labtest/online_learn.{png,pdf}")

# ---------- standalone reward figure (explicitly requested) ----------
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
a1.axhline(0, color="0.7", lw=0.8)
a1.plot(tmin[:len(dprog)], dprog, color="0.8", lw=0.5, label="progress reward/tick (m)")
a1.plot(tmin[W - 1:], rm, color="tab:purple", lw=1.9, label="reward (10 s rolling mean)")
a1.scatter(tmin[ot], reward[ot], color="tab:red", s=14, label=f"off-track −5 ({len(ot)} ticks, all recovered)")
a1.set_xlabel("time [min]"); a1.set_ylabel("reward per tick")
a1.set_title("Online reward per tick = progress − 5·off-track"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
a2.plot(tmin, ret, color="tab:blue", lw=1.9, label="cumulative return Σ reward")
for tk, b in zip(rtick, rbank):
    k = min(tk, N - 1)
    a2.scatter([tk * DT / 60.0], [ret[k]], color=("tab:green" if b == "banked" else "tab:red"), s=55, edgecolor="k", lw=0.5)
a2.scatter([], [], color="tab:green", label="lap banked"); a2.scatter([], [], color="tab:red", label="lap reverted")
a2.set_xlabel("time [min]"); a2.set_ylabel("return"); a2.set_title("Cumulative return (critic='return' target)")
a2.grid(alpha=0.3); a2.legend(fontsize=8)
fig2.suptitle("Online reward on lab_test_ev8 (logged on the online-learning drive: reward = progress − 5·off)", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(OUT / "online_reward.png", dpi=135); fig2.savefig(OUT / "online_reward.pdf")
print("wrote results/deploy_labtest/online_reward.{png,pdf}")
