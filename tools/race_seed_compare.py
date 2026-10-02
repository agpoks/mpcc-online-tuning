"""Compare two seed counts of the canonical run (first N1 vs first N2 seeds) so you can choose how
many seeds to report. Both are subsets of results/race/race_phase1.json (deterministic), so this is
reproducible. Writes results/race/seed_choice.md and prints it.

    python3 tools/race_seed_compare.py 5 6
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning")
KINDS = ["static", "slower", "equal", "faster"]


def rows(d, arm, seeds, kind=None):
    return [row for r in d["runs"] if r["arm"] == arm and r["seed"] in seeds
            for row in r["rows"] if kind is None or row["kind"] == kind]


def m(R, k):
    return float(np.mean([x[k] for x in R])) if R else float("nan")


def per_seed(d, arm, metric, seeds, kind=None):
    out = []
    for s in sorted(seeds):
        R = rows(d, arm, {s}, kind)
        if R:
            out.append(m(R, metric))
    return np.array(out)


def main(n1, n2, arm="ltc"):
    d = json.load(open(ROOT / "results/race/race_phase1.json"))
    S1, S2 = set(range(n1)), set(range(n2))
    L = [f"# Seed-count choice: {n1} vs {n2} seeds ({arm})", "",
         f"Both are the first-N seeds of the deterministic canonical run "
         f"(`race_phase1.json`); seeds 0..{min(n1, n2)-1} are bit-identical in both, so the only "
         f"difference is the extra seed(s).", "",
         "## Overall", "",
         f"| metric | {n1}-seed | {n2}-seed | {n2}-seed 95% CI |",
         "|---|---:|---:|---:|"]
    for name, key, sc in [("passes", "passes", 1), ("clean %", "clean", 100),
                          ("off-track %", "off", 100), ("contact %", "contact", 100)]:
        a = m(rows(d, arm, S1), key) * sc
        b = m(rows(d, arm, S2), key) * sc
        v = per_seed(d, arm, key, S2) * sc
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        L.append(f"| {name} | {a:.2f} | {b:.2f} | ±{ci:.1f} |")
    L += ["", "## By opponent class (passes / clean% / off%)", "",
          f"| class | {n1}-seed | {n2}-seed |", "|---|---|---|"]
    for k in KINDS:
        a = rows(d, arm, S1, k); b = rows(d, arm, S2, k)
        L.append(f"| {k} | {m(a,'passes'):.2f} / {m(a,'clean')*100:.0f} / {m(a,'off')*100:.0f} "
                 f"| {m(b,'passes'):.2f} / {m(b,'clean')*100:.0f} / {m(b,'off')*100:.0f} |")
    # per-seed spread (shows the bimodal metrics)
    L += ["", "## Per-seed (off-track %), to show spread / bimodality", "",
          f"| class | " + " | ".join(f"s{s}" for s in sorted(S2)) + " |",
          "|---|" + "|".join(["---"] * n2) + "|"]
    for k in KINDS:
        vs = [f"{m(rows(d, arm, {s}, k),'off')*100:.0f}" for s in sorted(S2)]
        L.append(f"| {k} | " + " | ".join(vs) + " |")
    txt = "\n".join(L) + "\n"
    (ROOT / "results/race/seed_choice.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]] or [5, 6]
    main(a[0], a[1])
