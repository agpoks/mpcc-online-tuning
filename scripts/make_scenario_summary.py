"""The result table as two deployment scenarios: standing start and flying start.

Reads results/scenario_summary.json -- laps per policy for each scenario, three
starts each -- and draws them side by side. Standing start = everything cold
from a stop (drive_policy.py); flying start = after a warm-up lap to the grid
(racing_check.py), the deployment condition. A policy fast flying but off cold
needs the warm-up lap every race gives it; one clean both ways needs nothing.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
INK, MUT, GRID = "#212529", "#868E96", "#DEE2E6"
COL = {"constant": MUT, "fit": "#0CA678", "online": "#AE3EC9"}


def main(track="icra_t2_raceline"):
    d = json.loads((ROOT / "results" / "scenario_summary.json").read_text())[track]
    rows = list(d.items())
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.62 * len(rows) + 1.6), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, scen, title in ((axes[0], "standing", "STANDING start (cold, from a stop)"),
                            (axes[1], "flying", "FLYING start (after a warm-up lap -- racing)")):
        for i, (lab, r) in enumerate(rows):
            y = len(rows) - 1 - i; c = COL[r["kind"]]
            laps, off = r[scen]["laps"], r[scen]["off"]
            for l_, o_ in zip(laps, off):
                ax.plot(l_, y, "o", ms=9, color=c, mfc="white" if o_ else c, mew=1.7, zorder=3)
            ok = [l_ for l_, o_ in zip(laps, off) if not o_]
            if ok:
                ax.plot([min(ok), max(ok)], [y, y], "-", color=c, lw=2, alpha=0.5, zorder=2)
        for val, cc, t in ((1.96, "#E8590C", "START"), (2.35, MUT, "best const")):
            ax.axvline(val, color=cc, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.text(val, len(rows) - 0.4, f" {t}", fontsize=7.5, color=cc, va="bottom")
        ax.set_xlim(0, 3.3); ax.set_ylim(-0.7, len(rows) - 0.1)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", color=INK)
        ax.set_xlabel("laps in 2500 steps; hollow = left the track", color=MUT, fontsize=9)
        ax.grid(True, axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUT, labelsize=8.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([lab for lab, _ in rows][::-1], fontsize=9.5, color=INK)
    fig.suptitle(f"{track}: what each policy delivers, by start scenario (3 starts each)\\n"
                 "warm/cold is solver initialisation, not tuning; a race gives every policy the warm-up lap",
                 fontsize=11.5, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.savefig(OUT / "scenario_summary.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", OUT / "scenario_summary.png")


if __name__ == "__main__":
    main()
