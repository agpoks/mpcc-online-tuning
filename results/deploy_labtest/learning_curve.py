#!/usr/bin/env python3
"""The CLEAN online-learning demonstration on lab_test: the keep-best ratchet. For each seed, per-episode
laps (with exploration) + the keep-best INCUMBENT (monotone non-decreasing: banks a faster clean net, reverts
a slower/off one) + the final net's FROZEN eval. This is the paper's mechanism -- online adaptation made safe
by keep-best. Plotted from the committed results/deploy_labtest/qual_labtest.json (no re-run)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); OUT = ROOT / "results/deploy_labtest"
d = json.load(open(OUT / "qual_labtest.json"))
seeds = [0, 1, 2]; cols = {0: "tab:green", 1: "tab:red", 2: "tab:blue"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.7, 1]})
for s in seeds:
    ep = d["episodes"][f"icra_t2_smooth|{s}|tuner"]
    n = [e["ep"] for e in ep]; laps = [e["laps"] for e in ep]
    best = [e["best"] if e["best"] is not None else 0 for e in ep]
    acts = [e["action"] for e in ep]; offs = [e["off"] for e in ep]
    # per-episode attempts: marker by outcome
    for x, y, a, o in zip(n, laps, acts, offs):
        mk = "x" if o else ("o" if a == "banked" else ".")
        mc = "k" if o else (cols[s] if a == "banked" else "0.6")
        axL.plot(x, y, mk, color=mc, ms=(9 if a == "banked" else 7), mew=1.4, zorder=3)
    # keep-best incumbent = the online-learned net so far (monotone)
    axL.step(n, best, where="post", color=cols[s], lw=2.2, label=f"seed {s}: keep-best net", zorder=2)
axL.set_xlabel("online-learning episode"); axL.set_ylabel("laps (per-episode budget)")
axL.set_title("Online learning on lab_test: keep-best ratchet\n"
              "thick line = banked net (monotone↑); o=banked attempt, x=off-track (reverted), ·=slower (reverted)")
axL.grid(alpha=0.3); axL.legend(loc="lower right", fontsize=8)

# frozen eval of each seed's learned net (the deployable number)
ev = {s: np.mean([e["laps"] for e in d["evals"][f"icra_t2_smooth|{s}|tuner"]]) for s in seeds}
evc = {s: all(not e["off"] for e in d["evals"][f"icra_t2_smooth|{s}|tuner"]) for s in seeds}
bars = axR.bar([f"seed {s}" for s in seeds], [ev[s] for s in seeds],
               color=[cols[s] for s in seeds], alpha=0.85)
for s, b in zip(seeds, bars):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{ev[s]:.2f}\n{'clean' if evc[s] else 'OFF'}",
             ha="center", fontsize=9)
axR.set_ylabel("laps (FROZEN eval, 2500 steps)")
axR.set_title("The learned net driven FROZEN\n(seed 2 = deploy choice)")
axR.grid(alpha=0.3, axis="y")
fig.suptitle("Online-learned MPCC on lab_test_ev8 — the keep-best mechanism (what produced the deployed seed-2 net)",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "online_learning_curve.png", dpi=135); fig.savefig(OUT / "online_learning_curve.pdf")
print("wrote results/deploy_labtest/online_learning_curve.{png,pdf}")
for s in seeds:
    ep = d["episodes"][f"icra_t2_smooth|{s}|tuner"]
    print(f"  seed {s}: keep-best {max(e['best'] or 0 for e in ep):.2f} -> frozen eval {ev[s]:.2f} ({'clean' if evc[s] else 'off'})")
