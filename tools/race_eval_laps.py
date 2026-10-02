"""Long (full-race) evaluation of the trained policy: drive the ONLINE banked net for ~N laps per
opponent class and report laps completed, overtakes over the race, and clean/off -- because a real
race is ~10 laps, not the ~4.6 laps of a 4000-step training episode. Reproducible (seeded), and it
writes to a SEPARATE cache (states_online_10lap_*) so it never clobbers the paper-figure data.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/race_eval_laps.py [--seed 0] [--steps 11000]
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
import tools.paper_race_learning as P

OUT = ROOT / "results/race"
L = Track.icra_t2_smooth().length


def laps_of(S):
    """Net forward laps from the wrapped arc-length log S (unwrap forward wraps, sum, / L)."""
    S = np.asarray(S, float); dS = np.diff(S)
    dS[dS < -L / 2] += L                       # a forward lap-wrap
    dS[dS > L / 2] -= L                        # (rare) backward glitch
    return float(dS[dS > 0].sum() / L)


def passes_of(PASS):
    return int(np.asarray(PASS)[-1]) if len(PASS) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=11000)   # ~12 laps of headroom (train ep = 4000)
    ap.add_argument("--kinds", nargs="+", default=["static", "slower", "equal", "faster"])
    ap.add_argument("--frozen", action="store_true", help="deterministic banked net (no explore/learn)")
    a = ap.parse_args()
    mode = "frozen" if a.frozen else "online"; tag = "_10lapfrozen" if a.frozen else "_10lap"
    drive = P.redrive if a.frozen else P.redrive_online
    rows = []
    for k in a.kinds:
        d = drive(a.seed, k, steps=a.steps, tag=tag)
        off = bool(d["off"][0]); laps = laps_of(d["S"]); pss = passes_of(d["PASS"])
        n = len(d["S"]); t = n * 0.05
        dur = "ended OFF-TRACK" if off else "finished clean"
        rows.append((k, laps, pss, off, n, t, d["V"].mean()))
        print(f"{k:7s}: {laps:5.2f} laps in {t:5.0f}s ({n} steps, {dur})  overtakes={pss}  v_mean={d['V'].mean():.2f}")
    # markdown summary
    md = [f"# 10-lap race evaluation (LTC {mode}, seed {a.seed}, up to {a.steps} steps)", "",
          "| opponent | laps | overtakes | outcome | v_mean |", "|---|---:|---:|---|---:|"]
    for k, laps, pss, off, n, t, v in rows:
        md.append(f"| {k} | {laps:.2f} | {pss} | {'OFF-TRACK at '+f'{laps:.1f} laps' if off else 'clean finish'} | {v:.2f} |")
    (OUT / f"race_10lap_{mode}.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote results/race/race_10lap_{mode}.md")
