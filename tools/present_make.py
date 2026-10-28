#!/usr/bin/env python3
"""Presentation assets for the safe-online-MPCC talk.

  python3 tools/present_make.py method     # how-it-works loop animation (schematic)
  python3 tools/present_make.py multiN     # one policy vs 3 mixed-pace opponents (RQ2)
  python3 tools/present_make.py weights     # overtake + live weight adaptation (method in action)
  python3 tools/present_make.py memory      # LTC vs MLP on the occluded equal-pace pass (ablation)
  python3 tools/present_make.py all

Data-only: reads cached state logs under results/race{,_multicar,_phase2}; no acados.
Writes the GIFs into results/present/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.font_manager as fm
from present_common import walls
from mpcc_tuning.track import Track

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "present"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 13})

# ---- house palette ----
BG = "0.90"; CORR = "white"; WALL = "0.42"
C_SENSE, C_LEARN, C_CTRL, C_PLANT = "#2f6fb0", "#2a8f3a", "#d08020", "#555555"
OPPC = "#e4572e"

def draw_map(ax, left, right):
    ax.set_facecolor(BG)
    L = np.vstack([left, left[:1]]); R = np.vstack([right, right[:1]])
    quads = [[L[i], L[i+1], R[i+1], R[i]] for i in range(len(L)-1)]
    ax.add_collection(PolyCollection(quads, facecolors=CORR, edgecolors="none", zorder=0))
    ax.plot(L[:, 0], L[:, 1], color=WALL, lw=0.9, zorder=1)
    ax.plot(R[:, 0], R[:, 1], color=WALL, lw=0.9, zorder=1)
    ax.set_aspect("equal"); ax.axis("off")

def fading_trail(ax, lw=3.2, cmap="viridis", zorder=4):
    lc = LineCollection([], cmap=cmap, lw=lw, zorder=zorder); ax.add_collection(lc); return lc

def set_trail(lc, X, Y, C, j, win, vmin, vmax):
    a = max(0, j - win)
    pts = np.array([X[a:j+1], Y[a:j+1]]).T.reshape(-1, 1, 2)
    if len(pts) < 2:
        lc.set_segments([]); return
    segs = np.concatenate([pts[:-1], pts[1:]], 1)
    lc.set_segments(segs); lc.set_array(np.asarray(C[a:j]))
    lc.set_clim(vmin, vmax)
    n = len(segs); al = np.linspace(0.15, 1.0, n); lc.set_alpha(al)

# ======================================================================= METHOD
def gif_method():
    fig, ax = plt.subplots(figsize=(12.8, 7.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 72)
    ax.axis("off"); fig.patch.set_facecolor("white")
    boxes = {  # name: (cx, cy, w, h, color, title, lines)
        "sit":  (14, 52, 20, 15, C_SENSE, "SITUATION",
                 ["curvature preview", "gap · TTC · margin", "opponent class"]),
        "pol":  (40, 52, 20, 15, C_LEARN, "POLICY  $\\pi_\\phi$",
                 ["recurrent LTC cell", "emits 9 MPCC weights", "anchored · box · bands"]),
        "mpc":  (66, 52, 20, 15, C_CTRL, "MPCC  (acados)",
                 ["one OCP, 20 Hz", "$\\theta$ = runtime param", "hard constraints fixed"]),
        "plt":  (90, 52, 16, 15, C_PLANT, "CAR + SENSOR",
                 ["single-track drift", "270° occluding", "LiDAR"]),
        "tun":  (53, 20, 46, 14, C_LEARN, "ONLINE TUNER",
                 ["return critic  $\\delta_t=r_t+\\gamma V'-V$",
                  "score function  $g=\\varepsilon/\\sigma^2$   ·   RTRL  $\\partial\\theta/\\partial\\phi$",
                  "keep-best: bank only a frozen-validated gain"]),
    }
    patch = {}
    for k, (cx, cy, w, h, col, title, lines) in boxes.items():
        p = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                           linewidth=2.0, edgecolor=col, facecolor="white", zorder=3)
        ax.add_patch(p); patch[k] = p
        ax.text(cx, cy+h/2-2.6, title, ha="center", va="center", fontsize=14.5,
                fontweight="bold", color=col, zorder=5)
        for i, ln in enumerate(lines):
            ax.text(cx, cy+h/2-5.8-i*3.1, ln, ha="center", va="center", fontsize=10.5,
                    color="0.2", zorder=5)

    def arrow(p0, p1, color="0.25", dashed=False, lw=2.2, rad=0.0):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, lw=lw,
                            color=color, zorder=2,
                            linestyle=("--" if dashed else "-"),
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a); return a
    # forward path
    arrow((24, 52), (30, 52)); ax.text(27, 54.4, "features", ha="center", fontsize=10, color="0.3")
    arrow((50, 52), (56, 52)); ax.text(53, 54.4, "$\\theta$", ha="center", fontsize=12, color="0.3")
    arrow((76, 52), (82, 52)); ax.text(79, 54.4, "$u_0$", ha="center", fontsize=11, color="0.3")
    # reward loop (plant -> bottom -> situation)
    ax.plot([90, 90, 14, 14], [44.5, 8, 8, 44.5], color="0.45", lw=2.0, zorder=1)
    arrow((14, 10), (14, 44.5)); ax.text(52, 6.0, "reward  $r_t=\\Delta s_t - c_{\\mathrm{off}}\\,\\mathbf{1}[\\text{off-track}]$",
                                         ha="center", fontsize=11, color="0.3")
    # gradient arrows (MPCC -> tuner, tuner -> policy)
    arrow((66, 44.5), (60, 27), color=C_LEARN, dashed=True, rad=-0.15)
    arrow((46, 27), (40, 44.5), color=C_LEARN, dashed=True, rad=-0.15)
    # moving dot + caption
    dot = Circle((0, 0), 0.9, color="#111", zorder=8, visible=False); ax.add_patch(dot)
    cap = ax.text(50, 68.5, "", ha="center", va="center", fontsize=15, color="#111")
    glow = {}
    for k, (cx, cy, w, h, col, *_) in boxes.items():
        g = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                           linewidth=5.5, edgecolor=col, facecolor="none", zorder=6, alpha=0.0)
        ax.add_patch(g); glow[k] = g

    # stages: (active box, caption, edge p0, p1)
    stages = [
        ("sit", "Sense the situation: curvature, traffic, opponent class", (24, 52), (30, 52)),
        ("pol", "Policy maps it to the 9 MPCC weights — inside a measured box + safety bands", (50, 52), (56, 52)),
        ("mpc", "MPCC solves one OCP with those weights — hard constraints hold by construction", (76, 52), (82, 52)),
        ("plt", "The car drives; a 270° sensor sees traffic, often occluded", (90, 44.5), (90, 8)),
        ("plt", "Measured return flows back — progress, minus going off-track", (90, 8), (14, 8)),
        ("tun", "Online tuner nudges the policy toward more return (critic · score function · RTRL)", (46, 27), (40, 44.5)),
        ("pol", "Keep-best banks the update only if a frozen lap confirms it — else revert", (50, 52), (56, 52)),
    ]
    FPS, PER = 12, 26
    def frame(f):
        si = (f // PER) % len(stages); t = (f % PER) / PER
        key, text, p0, p1 = stages[si]
        cap.set_text(text)
        for k, g in glow.items():
            g.set_alpha(0.55 if k == key else 0.0)
        dot.set_visible(True)
        dot.center = (p0[0] + (p1[0]-p0[0]) * t, p0[1] + (p1[1]-p0[1]) * t)
        return list(glow.values()) + [dot, cap]
    an = FuncAnimation(fig, frame, frames=PER*len(stages), interval=1000/FPS, blit=False)
    out = OUT / "method_loop.gif"
    an.save(out, writer=PillowWriter(fps=FPS)); plt.close(fig)
    print("wrote", out, sz(out)); return out

def sz(p): return f"{Path(p).stat().st_size/1e6:.1f} MB"

TRACK = None
def track():
    global TRACK
    if TRACK is None: TRACK = Track.icra_t2_smooth()
    return TRACK

# ======================================================================= MULTI-N
def gif_multiN(stride=20, fps=12):
    z = np.load(ROOT / "results/race/paper/states_multi_ltc_0.npz", allow_pickle=True)
    EX, EY, V = z["EX"], z["EY"], z["V"]; mix = [str(m) for m in z["mix"]]
    OPP = [(z["O0X"], z["O0Y"]), (z["O1X"], z["O1Y"]), (z["O2X"], z["O2Y"])]
    c, left, right = walls(track())
    fig, ax = plt.subplots(figsize=(8.2, 7.2)); draw_map(ax, left, right)
    vmin, vmax = float(np.percentile(V, 4)), float(np.percentile(V, 96))
    ax.plot(EX, EY, color="0.75", lw=0.8, zorder=2)
    trail = fading_trail(ax)
    ego = ax.scatter([], [], s=70, c="k", zorder=7)
    opp_dots = [ax.scatter([], [], s=60, color=OPPC, edgecolors="k", lw=0.6, zorder=6) for _ in OPP]
    rings = [Circle((0, 0), 0.30, fill=False, ec=OPPC, lw=1.4, alpha=0.8, zorder=5) for _ in OPP]
    for r in rings: ax.add_patch(r)
    ax.scatter([EX[0]], [EY[0]], s=110, facecolors="none", edgecolors="k", lw=1.8, zorder=7)
    cb = fig.colorbar(trail, ax=ax, fraction=0.035, pad=0.01); cb.set_label("speed (m/s)", fontsize=11)
    ax.set_title("One learned policy vs three mixed-pace opponents — no retraining",
                 fontsize=14, pad=10)
    lab = {"slower": "slower", "equal": "equal", "faster": "faster", "static": "static"}
    hud = ax.text(0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left", fontsize=12,
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    idxs = list(range(0, len(EX), stride))
    def frame(fi):
        j = idxs[fi]
        set_trail(trail, EX, EY, V, j, 120, vmin, vmax)
        ego.set_offsets([[EX[j], EY[j]]])
        for d, r, (ox, oy) in zip(opp_dots, rings, OPP):
            d.set_offsets([[ox[j], oy[j]]]); r.center = (ox[j], oy[j])
        hud.set_text("opponents: " + " · ".join(lab.get(m, m) for m in mix) +
                     f"\nspeed {V[j]:.2f} m/s")
        return [trail, ego, hud] + opp_dots
    an = FuncAnimation(fig, frame, frames=len(idxs), interval=1000/fps, blit=False)
    out = OUT / "race_multiN.gif"; an.save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    print("wrote", out, sz(out)); return out

# ======================================================================= WEIGHTS
def gif_weights(kind="equal", stride=12, fps=12):
    z = np.load(ROOT / f"results/race/paper/states_online_ltc_0_{kind}.npz")
    EX, EY, V, OX, OY = z["EX"], z["EY"], z["V"], z["OX"], z["OY"]
    TH = z["THETA"]; PASS = np.asarray(z["PASS"], float)
    c, left, right = walls(track())
    fig, (axr, axw) = plt.subplots(1, 2, figsize=(13.4, 6.6),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    draw_map(axr, left, right)
    vmin, vmax = float(np.percentile(V, 4)), float(np.percentile(V, 96))
    axr.plot(EX, EY, color="0.78", lw=0.8, zorder=2)
    trail = fading_trail(axr)
    ego = axr.scatter([], [], s=70, c="k", zorder=7)
    opp = axr.scatter([], [], s=62, color=OPPC, edgecolors="k", lw=0.6, zorder=6)
    ring = Circle((0, 0), 0.30, fill=False, ec=OPPC, lw=1.4, zorder=5); axr.add_patch(ring)
    axr.scatter([EX[0]], [EY[0]], s=110, facecolors="none", edgecolors="k", lw=1.8, zorder=7)
    star = axr.scatter([], [], s=220, marker="*", color="#f2c200", edgecolors="k", lw=0.6, zorder=8)
    art = "an" if kind[:1].lower() in "aeiou" else "a"
    axr.set_title(f"Overtaking {art} {kind}-pace opponent", fontsize=14, pad=8)
    SHOW = [(7, "$k_v$", "#2a8f3a"), (4, "$r_a$", "#c23b3b"),
            (0, "$q_c$", "#1f4e8c"), (2, "$q_v$", "#d08020")]
    rel0 = (TH - TH[0]) / np.log(2.0)   # log2 change from start (raw, per-tick exploration)
    def causal_ma(x, W=51):
        cs = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0); out = np.empty_like(x)
        for tt in range(len(x)):
            a = max(0, tt - W + 1); out[tt] = (cs[tt+1] - cs[a]) / (tt + 1 - a)
        return out
    rel = causal_ma(rel0)              # smoothed trend the tuner settles into
    t = np.arange(len(EX)) / 20.0
    axw.axhline(0, color="0.6", lw=1.0)
    lines = {i: axw.plot([], [], color=col, lw=2.4, label=nm)[0] for i, nm, col in SHOW}
    heads = {i: axw.scatter([], [], s=28, color=col, zorder=5)[0] if False else
             axw.plot([], [], marker="o", ms=6, color=col)[0] for i, nm, col in SHOW}
    axw.set_xlim(0, t[-1]); lo = float(rel[:, [7, 4, 0, 2]].min()); hi = float(rel[:, [7, 4, 0, 2]].max())
    pad = 0.15 * (hi - lo + 1e-6); axw.set_ylim(lo - pad, hi + pad)
    axw.set_xlabel("time (s)"); axw.set_ylabel("weight change  $\\log_2(w/w_0)$")
    axw.set_title("Online weight adaptation  (2.5 s moving average)", fontsize=13.5, pad=8)
    axw.legend(loc="upper left", ncol=4, frameon=False, fontsize=11)
    axw.grid(alpha=0.25)
    idxs = list(range(0, len(EX), stride)); passed = [0]
    def frame(fi):
        j = idxs[fi]
        set_trail(trail, EX, EY, V, j, 110, vmin, vmax)
        ego.set_offsets([[EX[j], EY[j]]]); opp.set_offsets([[OX[j], OY[j]]]); ring.center = (OX[j], OY[j])
        if j > 0 and PASS[j] > passed[0]:
            passed[0] = PASS[j]; star.set_offsets([[EX[j], EY[j]]])
        for i, nm, col in SHOW:
            lines[i].set_data(t[:j+1], rel[:j+1, i])
            heads[i].set_data([t[j]], [rel[j, i]])
        return [trail, ego, opp, star] + list(lines.values()) + list(heads.values())
    an = FuncAnimation(fig, frame, frames=len(idxs), interval=1000/fps, blit=False)
    fig.tight_layout()
    out = OUT / f"overtake_weights_{kind}.gif"; an.save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    print("wrote", out, sz(out)); return out

# ======================================================================= MEMORY
def gif_memory(seed=3, stride=10, fps=12, win=700):
    c, left, right = walls(track())
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.9))
    data = {}
    for ax, arm in zip(axes, ("ltc", "mlp")):
        z = np.load(ROOT / f"results/race_phase2/sensor_equal_{arm}_{seed}.npz", allow_pickle=True)
        EX, EY, DET, OX, OY = z["EX"], z["EY"], np.asarray(z["DET"], float), z["OX"], z["OY"]
        GAP = np.abs(np.asarray(z["GAP"], float))
        clean = bool(z["clean"]); passes = int(z["passes"]); detf = float(z["det_frac"])
        k = int(np.argmin(GAP)); a0 = max(0, k - win); a1 = min(len(EX), k + win)  # overtake window
        EX, EY, DET, OX, OY = EX[a0:a1], EY[a0:a1], DET[a0:a1], OX[a0:a1], OY[a0:a1]
        detf = float(DET.mean())   # in-view fraction over the shown overtake window
        draw_map(ax, left, right)
        ax.plot(EX, EY, color="0.80", lw=0.7, zorder=2)
        lc = LineCollection([], lw=3.4, zorder=4); ax.add_collection(lc)
        ego = ax.scatter([], [], s=72, c="k", zorder=7)
        opp = ax.scatter([], [], s=64, color=OPPC, edgecolors="k", lw=0.6, zorder=6)
        ring = Circle((0, 0), 0.30, fill=False, ec=OPPC, lw=1.4, zorder=5); ax.add_patch(ring)
        ax.scatter([EX[0]], [EY[0]], s=110, facecolors="none", edgecolors="k", lw=1.8, zorder=7)
        star = ax.scatter([], [], s=240, marker="*", color="#f2c200", edgecolors="k", lw=0.6, zorder=8)
        name = "LTC (memory)" if arm == "ltc" else "MLP (memoryless)"
        ax.set_title(f"{name} — completes the pass, clean\n"
                     f"keeps the car in view {detf*100:.0f}% of the overtake  (blind {(1-detf)*100:.0f}%)",
                     fontsize=12.5, pad=8)
        data[arm] = dict(EX=EX, EY=EY, DET=DET, OX=OX, OY=OY,
                         lc=lc, ego=ego, opp=opp, ring=ring, star=star, n=len(EX), passes=passes)
    # legend for in-view / occluded
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color="#1f6fb0", lw=3), Line2D([0], [0], color="0.55", lw=3)],
               ["opponent in view", "occluded (blind)"], loc="lower center", ncol=2,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Equal-pace overtake under a 270° occluding sensor", fontsize=15, y=0.99)
    nmax = max(d["n"] for d in data.values()); idxs = list(range(0, nmax, stride))
    def seg_colate(EX, EY, DET, j, win):
        a = max(0, j - win); pts = np.array([EX[a:j+1], EY[a:j+1]]).T.reshape(-1, 1, 2)
        if len(pts) < 2: return [], []
        segs = np.concatenate([pts[:-1], pts[1:]], 1)
        cols = np.where(DET[a:j] > 0.5, "#1f6fb0", "0.55")
        return segs, cols
    def frame(fi):
        arts = []
        for arm, d in data.items():
            j = min(idxs[fi], d["n"] - 1)
            segs, cols = seg_colate(d["EX"], d["EY"], d["DET"], j, 140)
            d["lc"].set_segments(segs); d["lc"].set_color(list(cols))
            d["ego"].set_offsets([[d["EX"][j], d["EY"][j]]])
            d["opp"].set_offsets([[d["OX"][j], d["OY"][j]]]); d["ring"].center = (d["OX"][j], d["OY"][j])
            arts += [d["lc"], d["ego"], d["opp"]]
        return arts
    an = FuncAnimation(fig, frame, frames=len(idxs), interval=1000/fps, blit=False)
    out = OUT / "memory_occlusion.gif"; an.save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    print("wrote", out, sz(out)); return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("method", "all"): gif_method()
    if which in ("multiN", "all"): gif_multiN()
    if which in ("weights", "all"): gif_weights("equal")
    if which in ("memory", "all"): gif_memory()
