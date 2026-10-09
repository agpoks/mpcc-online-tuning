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

VAR = [("pre-fix", "multi_eval_rows_prefix.npz"), ("gentler", "multi_eval_rows_gentler.npz"),
       ("q_c-floor", "multi_eval_rows_qcfloor.npz")]   # stable snapshot (multi_eval_rows.npz = latest/deploy eval)
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
shades = {"pre-fix": 0.3, "gentler": 0.6, "q_c-floor": 1.0}
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
fig.suptitle("Multi-car three-way: pre-fix → gentler keep-out → + q_c floor (0.5). "
             "q_c floor RECOVERS LTC (N=1,2 → 100%) but BREAKS MLP (off-track) — it is LTC-specific")
fig.tight_layout()
fig.savefig(OUT / "multi_contact_threeway.png", dpi=130); fig.savefig(OUT / "multi_contact_threeway.pdf")
print(f"\nwrote {OUT}/multi_contact_threeway.{{png,pdf}}")

# TikZ (pgfplots): clean% per arm per N across the three variants
TZ = OUT / "tikz"; TZ.mkdir(parents=True, exist_ok=True)
tex = ["% Multi-car three-way clean%: pre-fix -> gentler -> q_c-floor. Regen: tools/multi_contact_threeway.py",
       "\\begin{tikzpicture}",
       "\\begin{axis}[ybar,bar width=4pt,width=10cm,height=6cm,xtick={1,2,3},xlabel={opponents},",
       "  ylabel={clean races (\\%)},ymin=0,ymax=105,enlarge x limits=0.25,legend pos=south west,legend columns=2]"]
for arm in ARMS:
    for name, _ in VAR:
        coords = " ".join(f"({N},{ag[name][arm].get(N, {}).get('clean', 0):.0f})" for N in NSET)
        lbl = (arm.upper() + " " + name).replace("_", "-")   # no underscores (tikz math-mode gotcha)
        tex.append(f"\\addplot coordinates {{{coords}}}; \\addlegendentry{{{lbl}}}")
tex += ["\\end{axis}", "\\end{tikzpicture}"]
(TZ / "multi_contact_threeway_clean.tex").write_text("\n".join(tex) + "\n")
print(f"wrote {TZ}/multi_contact_threeway_clean.tex")
