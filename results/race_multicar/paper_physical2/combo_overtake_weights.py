#!/usr/bin/env python3
"""Combined 3-row figure for the RA-L paper (DATA-ONLY, reads committed physical2 npz).
Row 1: bird's-eye race, N=1,2,3 -- ego path by speed + OPPONENT paths + overtake stars.
Row 2: the online-tuned weights for the SAME runs (q_c, q_v left; q_l right twin axis);
       r_a, k_v held at their safety band (constant) are omitted.
Row 3: the per-tick reward the policy accrues over the race (raw + smoothed).
Shared legend outside the panels; compact panels, readable font. No acados.
Reproduce: python3 combo_overtake_weights.py
"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/home/poxx/github/mpcc-online-tuning")
B = ROOT / "results/race_multicar/paper_physical2/data"
OUT = Path("/home/poxx/github/R-AL-MPCC_online_learning/figures/overtake_multiN.pdf")
bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
# Times-compatible serif (Nimbus Roman) + STIX math so this matches the paper's Times type and
# the pgfplots (TikZ) figures -- not matplotlib's default DejaVu Serif. Smaller to match Fig 3.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
vmin, vmax = 1.2, 2.0
OPP_PATH = "#EE7733"   # clear orange (Tol): reads against the grey ground and the viridis ego
xlo, xhi = bnd["left_x"].min() - 0.3, bnd["left_x"].max() + 0.3
ylo_m, yhi_m = bnd["left_y"].min() - 0.3, bnd["left_y"].max() + 0.3

def tel(N):
    return np.load(B / f"telemetry_physical2_N{N}.npz", allow_pickle=True)

def smooth(x, w=51):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

# base weights (icra_t2_smooth anchor, theta0); labels + per-weight tab10 colours
BASE = np.array([1.0, 50.0, 0.2, 1.0, 6.0, 0.1, 0.15, 0.45, 0.07])
WLAB = [r"$q_c$", r"$q_l$", r"$q_v$", r"$r_\delta$", r"$r_a$",
        r"$r_{\dot s}$", r"$d_\mathrm{obs}$", r"$k_v$", r"$d_\mathrm{bnd}$"]
C = plt.cm.tab10
# two y-axes by scale: LEFT = O(<=1.3) weights, RIGHT = q_l (~25-50)
LEFT = [0, 2, 3, 5, 6, 8]           # q_c, q_v, r_delta, r_sdot, d_obs, d_bnd
RIGHT = 1                           # q_l
# r_a, k_v ADAPT within a class-conditioned band but saturated flat in these runs
BAND_L, BAND_R = 7, 4               # k_v (left scale), r_a (right scale)
WCOL = {0: C(0), 2: C(2), 3: C(3), 5: C(4), 6: C(5), 8: C(6), 1: C(1)}

fig = plt.figure(figsize=(7.5, 4.2), constrained_layout=True)   # ~\textwidth so font 9 ~ 9 pt
# 4-row grid: a THIN spacer row (0.12) between the maps and the weights holds the one-row
# weight legend; every other gap stays tight, as before.
gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.12, 0.8, 0.8])
axs = np.empty((3, 3), dtype=object)
for _j in range(3):
    axs[0, _j] = fig.add_subplot(gs[0, _j])   # maps
    axs[1, _j] = fig.add_subplot(gs[2, _j])   # weights
    axs[2, _j] = fig.add_subplot(gs[3, _j])   # reward
fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.03)
sc = None
# one common y-range for the reward row so the three panels share a scale (no scaling "gap")
_rlo, _rhi = np.inf, -np.inf
for _N in (1, 2, 3):
    _s = smooth(np.asarray(tel(_N)["R"], float))
    _rlo = min(_rlo, float(_s.min())); _rhi = max(_rhi, float(_s.max()))
RLO, RHI = _rlo - 0.3, _rhi + 0.4
for j, N in enumerate((1, 2, 3)):
    d = tel(N); T = d["THETA"].shape[0]; t = np.arange(T) * 0.05
    inview = np.asarray(d["DET"]).astype(float) > 0

    # ---- row 1: bird's-eye race, ICRA style (filled corridor + raceline + scale bar) ----
    a = axs[0, j]                                                     # clean paper style, as Fig 3
    a.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.8, zorder=1)
    a.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.8, zorder=1)
    a.plot(bnd["ref_x"], bnd["ref_y"], color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=1.5)  # raceline
    OPP = d["OPP"]  # (T, 3, 3): per opponent x,y,(s)
    for i in range(N):
        ox, oy = OPP[:, i, 0], OPP[:, i, 1]
        m = np.isfinite(ox) & np.isfinite(oy) & (ox > xlo) & (ox < xhi) & (oy > ylo_m) & (oy < yhi_m)
        if m.sum() > 5:
            a.plot(ox[m], oy[m], color=OPP_PATH, lw=1.8, alpha=0.8, solid_capstyle="round", zorder=2)
    sc = a.scatter(d["EX"], d["EY"], c=np.clip(d["V"], vmin, vmax), cmap="viridis",
                   s=2.2, vmin=vmin, vmax=vmax, zorder=3)
    for (tk, oi) in d["PASS"]:
        a.scatter(d["EX"][tk], d["EY"][tk], marker="D", s=32, facecolor="gold",
                  edgecolor="k", lw=0.6, zorder=5)         # overtake marker (diamond)
    a.plot(d["EX"][0], d["EY"][0], marker="o", mfc="none", mec="k", mew=1.2, ms=7, zorder=6)  # start = open circle
    if j == 0:                                                        # 1 m scale bar once, in the bottom margin
        x0, y0 = xlo + 0.15, ylo_m + 0.12
        a.plot([x0, x0 + 1.0], [y0, y0], color="k", lw=1.8, zorder=6)
        a.text(x0 + 0.5, y0 - 0.04, "1 m", ha="center", va="top", fontsize=7, zorder=6)
    a.set_title(f"({'abc'[j]})", fontsize=10, fontweight="bold", loc="left")
    a.set_aspect("equal")
    a.set_xlim(xlo, xhi); a.set_ylim(ylo_m, yhi_m)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)
    mx = np.r_[bnd["left_x"], bnd["right_x"]]; my = np.r_[bnd["left_y"], bnd["right_y"]]
    a.set_xlim(mx.min() - 0.1, mx.max() + 0.1); a.set_ylim(my.min() - 0.1, my.max() + 0.1)  # outer track limits

    # ---- row 2: all nine tuned weights as a RATIO to their base, theta/theta0, ONE axis ----
    # One shared scale avoids the dual-axis misread. k_v and r_a sit on their class-conditioned
    # safety band (flat here, dashed): adaptable within the band, but the reward optimum is at
    # the band edge so the tuner pins them there. (q_l base is 50, so its ~0.56 ratio = driven down.)
    b = axs[1, j]
    R = np.asarray(d["THETA"], float) / BASE
    if inview.any():
        b.fill_between(t, 0.2, 2.15, where=inview, color="0.88", step="mid", zorder=0)
    b.axhline(1.0, color="0.6", lw=0.7, zorder=0.5)                              # base = theta0
    hs = [b.plot(t, R[:, w], color=WCOL[w], lw=1.3, zorder=3)[0] for w in LEFT]
    hql = b.plot(t, R[:, 1], color=C(1), lw=1.5, zorder=3)[0]                    # q_l
    hkv = b.plot(t, R[:, 7], color=C(9), lw=1.6, ls="--", zorder=3)[0]           # k_v (at band)
    hra = b.plot(t, R[:, 4], color=C(8), lw=1.6, ls="--", zorder=3)[0]           # r_a (at band)
    for (tk, oi) in d["PASS"]:
        b.axvline(tk * 0.05, color="gold", lw=1.0, ls=":", zorder=1)
    b.set_ylim(0.2, 2.15); b.set_xlim(0, t[-1]); b.set_xticklabels([]); b.tick_params(labelsize=9)
    if j == 0:
        b.set_ylabel(r"weight / base  $\theta/\theta_0$")
        wleg_h = hs + [hql, hkv, hra]        # keep handles for a figure-level legend (below, out of the data)
        wleg_l = [WLAB[w] for w in LEFT] + [r"$q_l$", r"$k_v$ (band)", r"$r_a$ (band)"]

    # ---- row 3: per-tick reward over the race ----
    c = axs[2, j]
    R = np.asarray(d["R"]).astype(float)
    if inview.any():
        c.fill_between(t, RLO, RHI, where=inview, color="0.88", step="mid", zorder=0)
    c.plot(t, R, color="0.7", lw=0.5, zorder=1)
    c.plot(t, smooth(R), color="#6A1B9A", lw=1.6, zorder=2)
    for (tk, oi) in d["PASS"]:
        c.axvline(tk * 0.05, color="gold", lw=1.0, ls=":", zorder=3)
    c.set_xlim(0, t[-1]); c.set_ylim(RLO, RHI)      # shared reward scale across the three panels
    c.set_xlabel("race time (s)"); c.tick_params(labelsize=9)
    if j == 0:
        c.set_ylabel("reward / tick")

cb = fig.colorbar(sc, ax=axs[0, :], shrink=0.85, aspect=24, pad=0.01)
cb.set_label("Speed [m/s]", fontsize=9); cb.ax.tick_params(labelsize=8)
# The 2D-map markers legend sits ABOVE the maps row (near what it explains); the nine weight
# colours sit in the gap BETWEEN the maps and the weight panels, as a single row -- there is
# whitespace there, and it frees the bottom margin the two-row legend used to take.
fig.legend(handles=[Line2D([], [], marker="D", color="w", markerfacecolor="gold", markeredgecolor="k", ms=8),
                    Line2D([], [], marker="o", color="w", markerfacecolor="none", markeredgecolor="k", mew=1.4, ms=9),
                    Line2D([], [], color="0.2", lw=0.9, ls=(0, (4, 3))),
                    Line2D([], [], color=OPP_PATH, lw=2.4, alpha=0.9),
                    Line2D([], [], color="0.85", lw=7)],
           labels=["overtake", "ego start", "raceline", "opponent path", "opponent in view"],
           loc="lower center", ncol=5, fontsize=8.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.005))
# nine weight colours as ONE row, anchored just above the centre weight panel (in the gap)
axs[1, 1].legend(wleg_h, wleg_l, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                 ncol=9, fontsize=7.5, frameon=True, framealpha=0.9, handlelength=1.1,
                 columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
for N in (1, 2, 3):
    d = tel(N)
    print(f"  N={N}: {len(d['PASS'])} passes, reward mean {np.mean(d['R']):.2f}, det-in-view {100*np.mean(np.asarray(d['DET'])>0):.0f}%")
