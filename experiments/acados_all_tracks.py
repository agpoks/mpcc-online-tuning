"""The winning acados configuration, on every track.

    python3 experiments/acados_all_tracks.py --acados-v060

Globalization turned out to be the whole IPOPT-acados gap: on the oval,
FIXED_STEP gives 2.17 laps and FUNNEL_L1PEN_LINESEARCH gives 7.12, beating
IPOPT's 5.12 at a fraction of the runtime. That was measured on ONE track, and
a controller that works on an oval is not a controller.

This runs the best configuration on all five, against the IPOPT reference and
against FIXED_STEP as the control, so the effect can be checked where the
geometry is harder: the ICRA circuits have hairpins tighter than the oval's
minimum radius and corridor widths from 0.67 m to 1.10 m.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: name, horizon. Longer horizons on the ICRA circuits: 0.6 s of lookahead
#: cannot see through a 0.7 m-radius hairpin.
TRACKS = (("oval", 12), ("circuit", 12), ("icra_t1_raceline", 40),
          ("icra_t2_raceline", 40), ("icra2025", 50))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*",
                    default=["fqp_soft_funnel", "fqp_soft_fixed"])
    ap.add_argument("--tracks", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    # IPOPT is ~1.2 s/tick and N=40-50 on the ICRA circuits, so including
    # it multiplies the run by hours. Off by default; ask for it when the
    # reference is what you want rather than the acados comparison.
    ap.add_argument("--with-ipopt", action="store_true")
    ap.add_argument("--acados-v060", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.acados_v060:
        from mpcc_tuning.acados_variants import ACADOS_V060
        if os.environ.get("_ACADOS_SWITCHED") != "1":
            env = dict(os.environ, _ACADOS_SWITCHED="1")
            env["ACADOS_SOURCE_DIR"] = ACADOS_V060["ACADOS_SOURCE_DIR"]
            for k in ("LD_LIBRARY_PATH", "PYTHONPATH"):
                env[k] = ACADOS_V060[k] + os.pathsep + env.get(k, "")
            os.execve(sys.executable, [sys.executable, *sys.argv], env)

    from experiments.acados_variant_sweep import aggregate, run_ipopt, run_variant
    from mpcc_tuning.acados_variants import BY_NAME

    picked = [(n, h) for n, h in TRACKS if a.tracks is None or n in a.tracks]
    print("  %-20s %-20s %8s %8s %8s %9s" %
          ("track", "config", "laps", "usable%", "ms mean", "ms p99"),
          flush=True)
    rows, any_ran = [], False
    for tname, hz in picked:
        for vname in a.variants:
            v = BY_NAME[vname]
            runs = [run_variant(v, track_name=tname, horizon=hz,
                                steps=a.steps, seed=sd)
                    for sd in range(a.repeats)]
            r = aggregate(runs); r["track"] = tname; r["horizon"] = hz
            rows.append(r)
            if r["skipped"]:
                print("  %-20s %-20s SKIPPED %s" % (tname, vname, r["reason"]),
                      flush=True)
                continue
            any_ran = True
            print("  %-20s %-20s %5.2f+-%4.2f %7.0f%% %8.2f %9.1f"
                  % (tname, vname, r["laps_mean"], r["laps_std"],
                     r["usable_pct_mean"], r["ms_mean_mean"], r["ms_p99_mean"]),
                  flush=True)
        if a.with_ipopt:
            runs = [run_ipopt(tname, hz, 0.05, a.steps, sd)
                    for sd in range(a.repeats)]
            r = aggregate(runs); r["track"] = tname; r["horizon"] = hz
            rows.append(r); any_ran = True
            print("  %-20s %-20s %5.2f+-%4.2f %7.0f%% %8.2f %9.1f"
                  % (tname, "IPOPT", r["laps_mean"], r["laps_std"],
                     r["usable_pct_mean"], r["ms_mean_mean"], r["ms_p99_mean"]),
                  flush=True)
    out = a.out or str(ROOT / "benchmarks" / "results" / "acados_all_tracks.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(rows=rows), indent=2, default=str) + "\n")
    print(f"\n  wrote {out}")
    if not any_ran:
        print("  SWEEP FAILED: nothing ran.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
