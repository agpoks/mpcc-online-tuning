#!/usr/bin/env python3
"""Closed-loop dataflow diagram for docs/source/code_map.md.

One 50 ms control tick: track geometry -> features -> WeightPolicy -> AcadosMPCC
-> ScuderiaPlant -> PolicyTuner, each block labelled with its file:function and
the equation it computes. House style (serif + STIX math). Writes an SVG (crisp
in the RTD theme) plus a PNG preview. No solver, no data -- a static figure.

    python3 scripts/make_code_map_diagram.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/poxx/github/mpcc-online-tuning/docs/source/_static/code_map_dataflow.svg"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10,
})

# role -> (face, edge)
COL = {
    "sense": ("#e8f1fb", "#2d6cb3"),   # track / features  (blue)
    "learn": ("#e6f4ea", "#2e7d46"),   # policy / tuner     (green)
    "ctrl":  ("#fdecd9", "#d6821a"),   # MPCC               (orange)
    "plant": ("#ededed", "#666666"),   # plant              (grey)
}

fig, ax = plt.subplots(figsize=(11.6, 6.2))
ax.set_xlim(0, 116)
ax.set_ylim(0, 62)
ax.set_aspect("equal")
ax.axis("off")

BW, BH = 25.0, 12.0          # block width / height


def block(cx, cy, role, title, file, eqs, w=BW, h=BH, num=None):
    face, edge = COL[role]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.4,rounding_size=1.6",
                 fc=face, ec=edge, lw=1.4, zorder=2))
    if num is not None:   # section badge, matching the numbered sections in the page
        ax.scatter([cx - w / 2 + 2.2], [cy + h / 2 - 2.2], s=200, marker="o",
                   facecolor=edge, edgecolor="white", linewidths=1.0, zorder=4)
        ax.text(cx - w / 2 + 2.2, cy + h / 2 - 2.2, str(num), ha="center",
                va="center", fontsize=8.5, fontweight="bold", color="white", zorder=5)
    ax.text(cx, cy + h / 2 - 2.6, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=edge, zorder=3)
    ax.text(cx, cy + h / 2 - 5.4, file, ha="center", va="center",
            fontsize=7.6, style="italic", color="#444", family="monospace", zorder=3)
    y = cy + h / 2 - 8.0
    for e in eqs:
        ax.text(cx, y, e, ha="center", va="center", fontsize=8.6, color="#111", zorder=3)
        y -= 2.9
    return (cx, cy)


def arrow(p, q, label="", dashed=False, color="#333", rad=0.0,
          lx=0.0, ly=1.6, fs=8.2, lcolor=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                 lw=1.4, color=color, zorder=1,
                 linestyle=(0, (5, 3)) if dashed else "-",
                 shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=lcolor or color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


# ----- top forward row: track -> features -> policy -> MPCC -----------------
yT = 48
A = block(15, yT, "sense", "track geometry", "track.py",
          [r"$\kappa(s),\ $sector$,\ $width"], num=1)
B = block(46, yT, "sense", "features()", "ltc.py",
          [r"18-dim obs $x$", r"(dimensionless)"], num=2)
C = block(78, yT, "learn", "WeightPolicy.step()", "ltc.py + baselines.py",
          [r"$\theta=\theta_0+\mathrm{span}\odot\tanh(Gu)$", r"+ residual + bands"], num=3)
D = block(103, 34, "ctrl", "AcadosMPCC.value()", "acados_ocp / acados_mpcc.py",
          [r"solve $\min J(\theta)$", r"$\rightarrow u_0$"], num=4)

arrow(A, B, "state")
arrow(B, C, r"$x$")
# policy -> MPCC (down-right into D)
arrow((C[0] + BW / 2, C[1]), (D[0], D[1] + BH / 2), r"$\theta$  (9 log-weights)",
       lx=2, ly=2.2, rad=-0.15)

# ----- down the right: MPCC -> plant ----------------------------------------
E = block(103, 14, "plant", "ScuderiaPlant.step(u)", "plant_scuderia.py",
          [r"dynamic drift model", r"$\rightarrow s',\ r$"], num=5)
arrow((D[0], D[1] - BH / 2), (E[0], E[1] + BH / 2), r"$u_0$", lx=3.2, ly=0)

# ----- bottom row: plant -> tuner -------------------------------------------
F = block(58, 14, "learn", "PolicyTuner.learn()", "ltc.py",
          [r"TD$(\lambda)$: $\delta=r+\gamma^{n}V'-V$", r"actor $g=\varepsilon/\sigma^2$ (RTRL)"], num=6)
arrow((E[0] - BW / 2, E[1]), (F[0] + BW / 2, F[1]), r"$r,\ s'$")

# tuner -> policy (the online update), dashed green, up the middle
arrow((F[0], F[1] + BH / 2), (C[0], C[1] - BH / 2),
      "update  $G,\\ $cell$.p,\\ w$", dashed=True, color="#2e7d46", rad=0.12,
      lx=8.5, ly=0, fs=8.0)

# plant state -> features (next tick), light curved feedback along the bottom
arrow((E[0] - BW / 2, E[1] - BH / 2 + 2), (A[0], A[1] - BH / 2),
      "next tick  $s'$", dashed=False, color="#9aa0a6", rad=0.28,
      lx=0, ly=-2.2, fs=7.8, lcolor="#777")

# ----- per-episode keep-best (dashed, off the tuner) ------------------------
G = block(18, 14, "learn", "end_episode()  /", "ltc.py",
          [r"confirm_candidate()", r"keep-best + validate"], w=26, h=12)
arrow((F[0] - BW / 2, F[1]), (G[0] + 13, G[1]), "per episode", dashed=True,
      color="#2e7d46", fs=7.8)

# ----- framing title --------------------------------------------------------
ax.text(58, 60, "Closed loop — once per 50 ms control tick",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
ax.text(58, 56.6,
        "the MPCC solves and drives every tick, learning or frozen; "
        "the weights are a cost parameter, so the hard constraints always hold",
        ha="center", va="center", fontsize=8.4, style="italic", color="#555")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, bbox_inches="tight")
png = OUT.rsplit(".", 1)[0] + ".png"
fig.savefig(png, dpi=150, bbox_inches="tight")
print("wrote", OUT, "and", png)
