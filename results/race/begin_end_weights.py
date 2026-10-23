#!/usr/bin/env python3
"""Begin-of-race vs best-policy emitted weights, per opponent class (DATA-ONLY).

Within ONE consistent run of the current method (no cross-code-state comparison):
  LEFT  = start of the race  -- emitted weights in the FIRST online episode
          (the anchored base policy, before keep-best has refined it).
  RIGHT = best policy         -- emitted weights in the LAST online episode,
          after the validated keep-best rule has banked the winning policy.
Both as log2(mean emitted / START), all 9 weights, averaged over the 3 seeds.
Reads only results/race/race_phase_recent_3seed.json. No acados.
Reproduce: python3 results/race/begin_end_weights.py
"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Times-compatible serif (Nimbus Roman) + STIX math so this matches the paper's Times type
# and the other figures -- without it matplotlib falls back to DejaVu Sans.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})

ROOT = Path("/home/poxx/github/mpcc-online-tuning")
SRC  = ROOT / "results/race/race_phase_recent_3seed.json"
OUT  = Path("/home/poxx/github/R-AL-MPCC_online_learning/figures/weights_by_opponent.pdf")
WN = [r"$q_c$", r"$q_l$", r"$q_v$", r"$r_\delta$", r"$r_a$", r"$r_{\dot s}$",
      r"$d_{\mathrm{obs}}$", r"$k_v$", r"$d_{\mathrm{bnd}}$"]
START = np.array([1.0, 50.0, 0.2, 1.0, 6.0, 0.1, 0.15, 0.45, 0.07])   # anchored base (icra_t2_smooth)
KINDS = ["static", "slower", "equal", "faster"]


def begin_end(path):
    """mean emitted weights per class at the FIRST vs LAST online episode, over seeds."""
    runs = json.load(open(path))["runs"]
    BEG, END = {}, {}
    for k in KINDS:
        firsts, lasts = [], []
        for r in runs:
            if r.get("arm") != "ltc":
                continue
            rows = sorted((row for row in r["rows"] if row.get("kind") == k),
                          key=lambda x: x["ep"])
            if rows:
                firsts.append(rows[0]["theta_mean"])
                lasts.append(rows[-1]["theta_mean"])
        BEG[k] = np.mean(np.array(firsts, float), axis=0) if firsts else np.full(9, np.nan)
        END[k] = np.mean(np.array(lasts, float), axis=0) if lasts else np.full(9, np.nan)
    return BEG, END


def heat(ax, M, title, xlabels=True):
    Z = np.array([np.log2(M[k] / START) for k in KINDS])
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(9))
    ax.set_xticklabels(WN if xlabels else [], fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(KINDS, fontsize=8)
    for i in range(4):
        for j in range(9):
            v = Z[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title(title, fontsize=9, loc="left")
    return im


# Single-column IEEE figure: the two heatmaps are STACKED (start on top, best
# policy below) so each keeps full column width and the fonts stay readable.
BEG, END = begin_end(SRC)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.7), constrained_layout=True)   # a bit smaller; font stays 9 pt
heat(axs[0], BEG, "(a) start of race (base policy)", xlabels=False)
im = heat(axs[1], END, "(b) banked best (keep-best)", xlabels=True)
for ax in axs:
    ax.set_ylabel("opp. class", fontsize=8); ax.tick_params(labelsize=8)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.95, pad=0.02, aspect=42)
cb.set_label(r"$\log_2(\mathrm{emitted}/\mathrm{START})$", fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("wrote", OUT)
for tag, M in [("BEGIN (first ep)", BEG), ("END (best policy)", END)]:
    print(f"--- {tag} mean emitted weights per class ---")
    for k in KINDS:
        print("  ", f"{k:7s}", "  ".join(
            f"{n}={v:.3f}" for n, v in zip(
                ["q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v", "d_bnd"], M[k])))
