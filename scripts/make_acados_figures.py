"""Figures for the acados investigation: globalization, tracks, overtaking."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUT = "#212529", "#868E96"
# categorical, fixed order, never cycled
C_FIXED, C_MERIT, C_FUNNEL = "#868E96", "#4C6EF5", "#0CA678"
C_IPOPT = "#E8590C"


def fig_globalization():
    """The headline: step acceptance, not iteration count, was the gap."""
    data = {
        "soft SQP":            (2.27, 4.79, 5.52, 0.53, 2.05, 2.57),
        "feasible-QP, soft":   (2.17, 5.46, 7.12, 0.66, 2.36, 1.67),
        "feasible-QP, hard":   (0.97, 1.53, 2.95, 0.31, 0.50, 0.79),
        "feasible-QP, hard\n+ half-space": (1.94, 1.90, 3.22, 0.21, 0.37, 0.88),
    }
    names = list(data)
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    fig.patch.set_facecolor("white")
    for i, (lab, c) in enumerate([("FIXED_STEP", C_FIXED),
                                  ("MERIT_BACKTRACKING", C_MERIT),
                                  ("FUNNEL_L1PEN", C_FUNNEL)]):
        vals = [data[n][i] for n in names]
        errs = [data[n][i + 3] for n in names]
        b = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
                   color=c, label=lab, zorder=3,
                   error_kw=dict(ecolor="#495057", lw=1.0))
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(5.12, color=C_IPOPT, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.text(len(names) - 0.42, 5.32, "IPOPT reference 5.12",
            va="bottom", ha="right", fontsize=9, color=C_IPOPT,
            fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("laps completed (mean of 3, perturbed starts)", fontsize=10)
    fig.suptitle("Globalization, not iteration count, was the acados-IPOPT gap",
                 fontsize=12.5, fontweight="bold", x=0.012, ha="left", y=1.06,
                 color=INK)
    fig.text(0.012, 0.995, "acados defaults to FIXED_STEP: the full SQP step, no "
             "line search, no acceptance test. Raising iterations 8 / 50 / 300 "
             "changed nothing\n(3.16 / 3.06 / 2.43 laps). Rejecting bad steps "
             "changed everything.", fontsize=9, color=MUT, va="top", ha="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3,
              bbox_to_anchor=(0.0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 10.2)
    fig.savefig(OUT / "acados_globalization.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "acados_globalization.png")


def fig_tracks():
    """Where it generalises and where it does not."""
    f = ROOT / "benchmarks" / "results" / "acados_all_tracks.json"
    if not f.exists():
        print("skip tracks: no data"); return
    rows = [r for r in json.load(open(f))["rows"] if not r.get("skipped")]
    tracks, by = [], {}
    for r in rows:
        t = r["track"]
        if t not in by: by[t] = {}; tracks.append(t)
        by[t][r["name"]] = r
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                  gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor("white")
    x = np.arange(len(tracks)); w = 0.36
    for i, (nm, c) in enumerate([("fqp_soft_fixed", C_FIXED),
                                 ("fqp_soft_funnel", C_FUNNEL)]):
        v = [by[t].get(nm, {}).get("laps_mean", 0) for t in tracks]
        e = [by[t].get(nm, {}).get("laps_std", 0) for t in tracks]
        ax.bar(x + (i - 0.5) * w, v, w, yerr=e, capsize=3, color=c,
               label="FIXED_STEP" if i == 0 else "FUNNEL_L1PEN", zorder=3,
               error_kw=dict(ecolor="#495057", lw=1.0))
    ax.axhline(2.0, color="#C92A2A", lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.text(len(tracks) - 0.4, 2.05, " two-lap bar", ha="right", va="bottom",
            fontsize=9, color="#C92A2A", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_raceline", "").replace("icra_", "ICRA ")
                        for t in tracks], fontsize=9, rotation=12)
    ax.set_ylabel("laps", fontsize=10)
    ax.set_title("(a)  It clears two laps on two tracks of five",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top","right"]].set_visible(False)

    for i, (nm, c) in enumerate([("fqp_soft_funnel", C_FUNNEL)]):
        v = [by[t].get(nm, {}).get("ms_mean_mean", 0) for t in tracks]
        p = [by[t].get(nm, {}).get("ms_p99_mean", 0) for t in tracks]
        ax2.bar(x - 0.18, v, 0.36, color=c, label="mean", zorder=3)
        ax2.bar(x + 0.18, p, 0.36, color="#ADB5BD", label="p99", zorder=3)
    ax2.axhline(50, color="#C92A2A", lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax2.text(len(tracks) - 0.4, 52, " 50 ms budget", ha="right", va="bottom",
             fontsize=9, color="#C92A2A", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace("_raceline", "").replace("icra_", "ICRA ")
                         for t in tracks], fontsize=9, rotation=12)
    ax2.set_ylabel("solve time [ms]", fontsize=10)
    ax2.set_title("(b)  and the ICRA horizons blow the tick budget",
                  fontsize=11, fontweight="bold", loc="left", color=INK)
    ax2.legend(frameon=False, fontsize=9); ax2.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "acados_tracks.png", dpi=190, bbox_inches="tight",
                facecolor="white")
    print("wrote", OUT / "acados_tracks.png")


def fig_overtaking():
    f = ROOT / "benchmarks" / "results" / "overtaking_dynamic.json"
    if not f.exists():
        print("skip overtaking: no data"); return
    d = json.load(open(f)); rows = d["rows"]; pace = d["pace"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4),
                                  gridspec_kw=dict(width_ratios=[1, 1]))
    fig.patch.set_facecolor("white")
    lab = [f"{r['ratio']:.2f}x\n({r['ratio']*pace:.2f} m/s)" for r in rows]
    x = np.arange(len(rows))
    cols = ["#0CA678" if not r["off"] else "#E8590C" for r in rows]
    b = ax.bar(x, [r["passes"] for r in rows], 0.55, color=cols, zorder=3)
    for rect, r in zip(b, rows):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.05,
                "left track" if r["off"] else "clean", ha="center", va="bottom",
                fontsize=8.5, color="#E8590C" if r["off"] else "#0CA678",
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlabel("opponent speed, as a fraction of the ego's own pace", fontsize=9)
    ax.set_ylabel("completed passes", fontsize=10)
    ax.set_title("(a)  Passes made", fontsize=11, fontweight="bold",
                 loc="left", color=INK)
    ax.spines[["top","right"]].set_visible(False)

    g = [r["min_gap"] for r in rows]
    ax2.bar(x, g, 0.55, color="#4C6EF5", zorder=3)
    ax2.axhline(0.24, color="#C92A2A", lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax2.text(len(rows) - 0.45, 0.245, " keep-out 0.24 m", ha="right",
             va="bottom", fontsize=9, color="#C92A2A", fontweight="bold")
    for xx, gg in zip(x, g):
        ax2.text(xx, gg + 0.006, f"{gg:.3f}", ha="center", va="bottom",
                 fontsize=8.5, color=INK)
    ax2.set_xticks(x); ax2.set_xticklabels(lab, fontsize=9)
    ax2.set_ylabel("closest approach [m]", fontsize=10)
    ax2.set_ylim(0, max(g) * 1.25)
    ax2.set_title("(b)  Every pass cleared the keep-out", fontsize=11,
                  fontweight="bold", loc="left", color=INK)
    ax2.spines[["top","right"]].set_visible(False)
    fig.suptitle("Overtaking with the deployable controller "
                 "(acados, feasible-QP + FUNNEL, ~11 ms/tick)",
                 fontsize=11.5, x=0.012, ha="left", y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "overtaking_results.png", dpi=190, bbox_inches="tight",
                facecolor="white")
    print("wrote", OUT / "overtaking_results.png")


if __name__ == "__main__":
    fig_globalization(); fig_tracks(); fig_overtaking()
