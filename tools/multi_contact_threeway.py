"""Phase-A three-way: the occluded-opponent keep-out across its settings -- pre-fix (no keep-out) ->
hard-fix (predicted r=0.36, cap 1.5s) -> gentler-fix (r=0.28, cap 1.0s). Loads the three saved eval-row
snapshots in results/race_multicar/ and emits a table + a 3-panel plot (clean/contact/off, per arm per N).
    python3 tools/multi_contact_threeway.py
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from experiments.race_multi_eval import load_rows_path, _agg, ARMS, NSET, OUT

VAR = [("pre-fix", "multi_eval_rows_prefix.npz"), ("hard-fix", "multi_eval_rows_fix.npz"),
       ("gentler", "multi_eval_rows.npz")]
ag = {name: _agg(load_rows_path(OUT / f)) for name, f in VAR}

print(f"\n{'':9}", end="")
for name, _ in VAR:
    print(f"{name:>22}", end="")
print(f"\n{'arm N':9}" + ("  clean/contact/off %   " * len(VAR)))
for arm in ARMS:
    for N in NSET:
        print(f"{arm:>4} {N:>2} ", end="")
        for name, _ in VAR:
            a = ag[name][arm].get(N, {})
            print(f"{a.get('clean', 0):5.0f}/{a.get('contact', 0):3.0f}/{a.get('off', 0):3.0f}     ", end="")
        print()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
metrics = [("clean", "clean races (%)  (higher better)"), ("contact", "contact races (%)  (lower better)"),
           ("off", "off-track races (%)  (lower better)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
x = np.arange(len(NSET)); w = 0.13
shades = {"pre-fix": 0.35, "hard-fix": 0.62, "gentler": 1.0}
col = {"ltc": "#1f77b4", "mlp": "#d62728"}
for ax, (metric, lbl) in zip(axes, metrics):
    for ai, arm in enumerate(ARMS):
        for vi, (name, _) in enumerate(VAR):
            vals = [ag[name][arm].get(N, {}).get(metric, np.nan) for N in NSET]
            off = (ai * 3 + vi - 2.5) * w
            ax.bar(x + off, vals, w, color=col[arm], alpha=shades[name],
                   label=f"{arm.upper()} {name}" if ax is axes[0] else None)
    ax.set_xticks(x); ax.set_xticklabels([f"{N}car" for N in NSET]); ax.set_ylabel(lbl); ax.set_ylim(0, 105)
axes[0].legend(fontsize=6.5, ncol=2, loc="lower center")
fig.suptitle("Occluded-opponent keep-out, three-way: pre-fix → hard (r0.36/1.5s) → gentler (r0.28/1.0s)")
fig.tight_layout()
fig.savefig(OUT / "multi_contact_threeway.png", dpi=130); fig.savefig(OUT / "multi_contact_threeway.pdf")
print(f"\nwrote {OUT}/multi_contact_threeway.{{png,pdf}}")
