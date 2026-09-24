"""Phase-1 race results -> tables + a decision figure, broken down BY OPPONENT CLASS.

    python3 tools/race_table.py results/race/race_phase1_fast.json

Reads a race_mode.py output JSON (summary + per-run rows) and produces, so every
Phase-1 result is reproducible and shown as a table AND a figure:
  - results/race/race_phase1_summary.csv    per (arm, kind): laps, passes, clean%, off%, contact%
  - results/race/race_phase1_byseed.csv     per (arm, seed): headline numbers + wall time
  - results/race/race_phase1_table.md       markdown table (paste-ready)
  - results/race/race_phase1_summary.{pdf,png}  grouped bars: laps / clean% / passes / off%
                                             by arm, grouped by opponent class

The per-class split is the point: "slower" we can cruise, "equal/faster" we must push --
the figure shows whether the online tuner (ltc) actually does that vs the fixed START
constant (const) and the frozen schedule (fixed).
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path("/home/poxx/github/mpcc-online-tuning")
ARM_ORDER = ["const", "fixed", "ltc", "mlp"]
KIND_ORDER = ["static", "slower", "equal", "faster"]
ARM_LABEL = {"const": "const (START)", "fixed": "fixed sched", "ltc": "ltc (online)", "mlp": "mlp (online)"}


def load(path):
    d = json.load(open(path))
    runs = d["runs"]
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in runs)]
    arms += [r["arm"] for r in runs if r["arm"] not in arms]  # any unexpected arms last
    # per (arm, kind) aggregation over the LAST-episode rows of every seed
    by_ak = defaultdict(lambda: defaultdict(list))   # by_ak[arm][kind] -> list of row dicts
    by_arm = defaultdict(list)                        # by_arm[arm] -> list of row dicts (all kinds)
    for r in runs:
        for row in r.get("rows", []):
            by_ak[r["arm"]][row["kind"]].append(row)
            by_arm[r["arm"]].append(row)
    return d, runs, arms, by_ak, by_arm


def agg(rows):
    if not rows:
        return None
    return dict(n=len(rows),
                laps=float(np.mean([x["laps"] for x in rows])),
                passes=float(np.mean([x["passes"] for x in rows])),
                clean=float(np.mean([x["clean"] for x in rows])),
                off=float(np.mean([x["off"] for x in rows])),
                contact=float(np.mean([x["contact"] for x in rows])))


def main(argv=None):
    path = (argv or sys.argv[1:] or ["results/race/race_phase1_fast.json"])[0]
    d, runs, arms, by_ak, by_arm = load(path)
    kinds = [k for k in KIND_ORDER if any(k in by_ak[a] for a in arms)]

    # ---- CSV 1: per (arm, kind) + an ALL row -------------------------------
    lines = ["arm,kind,n_ep,laps,passes,clean_frac,offtrack_frac,contact_frac"]
    for a in arms:
        for k in kinds + ["ALL"]:
            g = agg(by_arm[a]) if k == "ALL" else agg(by_ak[a].get(k, []))
            if g:
                lines.append(f"{a},{k},{g['n']},{g['laps']:.3f},{g['passes']:.3f},"
                             f"{g['clean']:.3f},{g['off']:.3f},{g['contact']:.3f}")
    (ROOT / "results/race/race_phase1_summary.csv").write_text("\n".join(lines) + "\n")

    # ---- CSV 2: per (arm, seed) --------------------------------------------
    bl = ["arm,seed,laps,passes,clean_frac,contact_frac,wall_s"]
    for r in sorted(runs, key=lambda r: (ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 9, r["seed"])):
        bl.append(f"{r['arm']},{r['seed']},{r['laps']:.3f},{r['passes']:.3f},"
                  f"{r['clean']:.3f},{r['contact']:.3f},{r.get('wall_s', 0):.0f}")
    (ROOT / "results/race/race_phase1_byseed.csv").write_text("\n".join(bl) + "\n")

    # ---- markdown table (overall) ------------------------------------------
    md = ["| arm | laps | passes | clean | off-track | contact |",
          "|-----|-----:|-------:|------:|----------:|--------:|"]
    for a in arms:
        g = agg(by_arm[a])
        md.append(f"| {ARM_LABEL.get(a, a)} | {g['laps']:.2f} | {g['passes']:.2f} | "
                  f"{g['clean']:.0%} | {g['off']:.0%} | {g['contact']:.0%} |")
    md += ["", "By opponent class (laps / clean%):", "",
           "| arm | " + " | ".join(kinds) + " |", "|---|" + "|".join(["---"] * len(kinds)) + "|"]
    for a in arms:
        cells = []
        for k in kinds:
            g = agg(by_ak[a].get(k, []))
            cells.append(f"{g['laps']:.2f} / {g['clean']:.0%}" if g else "--")
        md.append(f"| {ARM_LABEL.get(a, a)} | " + " | ".join(cells) + " |")
    md_txt = "\n".join(md) + "\n"
    (ROOT / "results/race/race_phase1_table.md").write_text(md_txt)
    print(md_txt)

    # ---- figure: grouped bars, metric x (arm x kind) -----------------------
    metrics = [("laps", "laps completed", 1.0), ("clean", "clean fraction", 1.0),
               ("passes", "overtakes", 1.0), ("off", "off-track fraction", 1.0)]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    acol = {a: colors[i] for i, a in enumerate(arms)}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(kinds)); w = 0.8 / max(len(arms), 1)
    for ax, (key, ylab, _) in zip(axs.ravel(), metrics):
        for i, a in enumerate(arms):
            vals = [(agg(by_ak[a].get(k, [])) or {}).get(key, np.nan) for k in kinds]
            ax.bar(x + i * w, vals, w, label=ARM_LABEL.get(a, a), color=acol[a], edgecolor="k", lw=0.4)
        ax.set_xticks(x + w * (len(arms) - 1) / 2); ax.set_xticklabels(kinds)
        ax.set_ylabel(ylab); ax.set_title(ylab); ax.grid(axis="y", alpha=0.3)
        if key in ("clean", "off"):
            ax.set_ylim(0, 1.05)
    axs.ravel()[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Phase-1 race: arms by opponent class  (ego_pace={d.get('ego_pace', 0):.2f} m/s, "
                 f"{len({r['seed'] for r in runs})} seeds)", fontsize=13)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(ROOT / f"results/race/race_phase1_summary.{e}", dpi=140)
    print("saved race_phase1_summary.{csv,pdf,png}, race_phase1_byseed.csv, race_phase1_table.md")


if __name__ == "__main__":
    main()
