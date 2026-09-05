"""The honest frozen-policy table: every policy driven COLD (fresh solver).

The in-experiment frozen evaluation ran on an acados solver warmed by the
preceding learning episodes, and acados' sv.reset() does not fully clear that
state -- so the online-adaptation numbers were optimistic (MPCC critic: banked
2.99, cold 1.01 and off the track). This figure uses only numbers reproduced
COLD with a freshly built solver (scripts/drive_policy.py), which is what the
car would see. Source: results/frozen_summary_cold.json.
"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
COL = {"constant": MUT, "fit": "#0CA678", "online": "#AE3EC9"}

def main(track="icra_t2_raceline"):
    d = json.loads((ROOT / "results" / "frozen_summary_cold.json").read_text())[track]
    rows = list(d.items())
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 1.5)); fig.patch.set_facecolor("white")
    for i, (lab, r) in enumerate(rows):
        y = len(rows) - 1 - i; c = COL[r["kind"]]
        for l_, o_ in zip(r["laps"], r["off"]):
            ax.plot(l_, y, "o", ms=9, color=c, mfc="white" if o_ else c, mew=1.7, zorder=3)
        ok = [l_ for l_, o_ in zip(r["laps"], r["off"]) if not o_]
        if ok: ax.plot([min(ok), max(ok)], [y, y], "-", color=c, lw=2, alpha=0.5, zorder=2)
        clean = "" if all(not o for o in r["off"]) else "  (leaves the track)"
        ax.text(-0.02, y, lab + clean, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5, color=INK)
    for k, (val, c, lab) in enumerate(((1.96, "#E8590C", "START, the safe baseline"),
                                       (2.35, MUT, "best constant by search"))):
        ax.axvline(val, color=c, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(val, len(rows) - 0.45 + 0.3 * k, f" {lab} {val:.2f}", fontsize=8, color=c, va="bottom")
    ax.set_yticks([]); ax.set_xlim(0, 3.1); ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xlabel("laps in 2500 steps, driven COLD on a fresh solver (learning off, noise off); hollow = left the track", color=MUT, fontsize=9.5, labelpad=8)
    ax.grid(True, axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8.5)
    ax.set_title("ICRA T2: what each approach delivers as a FIXED policy, driven cold (3 starts each)",
                 loc="left", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(0, -0.9, "green = supervised fit to the situation grid (the best, reproducible);  purple = + online adaptation;  "
            "grey = constants.\nThe MPCC-critic online row banked 2.72/2.78/2.99 on a WARM eval solver; cold it collapses -- "
            "its weights only stay feasible given a warm start.", fontsize=8, color=MUT, transform=ax.get_yaxis_transform())
    fig.savefig(OUT / "cold_summary.png", dpi=190, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  wrote", OUT / "cold_summary.png")

if __name__ == "__main__":
    main()
