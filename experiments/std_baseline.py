"""Find a stable parameterisation on a TYRE model, not a kinematic bicycle.

    PYTHONPATH=/path/to/scuderia_gym_jax \
        python3 experiments/std_baseline.py --plant std --jobs 5

Step 1 of the project's stated order, done on a plant that represents the car:
a fixed setting that simply *drives*, several laps, not the fastest. Only then
does online tuning have a baseline to improve on, and only then can grip,
sectors and opponents be studied -- a kinematic bicycle has no friction to
lower, so "mu = 0.6 through this section" is a speed cap and not physics.

Measured on the oval, same weights and controller, changing only the plant:

    kinematic bicycle    7.71 laps, clean, 100% solve
    scuderia ST          0.20 laps, off,    72% solve
    scuderia STD         0.22 laps, off,    84% solve

The grid that passes on the bicycle does not contain a setting that drives the
tyre model: its best is 1.22 laps. That grid fixes ``q_v = 2`` and caps
``r_delta`` at 1, which is the wrong region. A car that can actually slide wants
**less progress pressure and smoother steering**, so this searches downward in
``q_v`` and upward in ``r_delta``, which the earlier grid never reached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

#: Downward in q_v, upward in r_delta -- the region the bicycle grid never
#: reached, and the one a sliding car should want.
GRID = [(qc, qv, rd)
        for qc in (0.3, 1.0, 3.0)
        for qv in (0.25, 0.5, 1.0)
        for rd in (1.0, 5.0, 20.0)]


def one(job):
    qc, qv, rd, plant, horizon, laps, track_name, max_iter = job
    t = getattr(Track, track_name)()
    steps = int(laps * t.length / (0.05 * 1.5)) + 300
    th = MPCCWeights(q_c=qc, q_l=50.0, q_v=qv, r_d=rd).to_log()
    m = MPCC(t, model=KinematicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=max_iter)
    if plant == "bicycle":
        from examples.tune_online import Plant
        P = Plant(t, dt=0.05, max_steps=steps)
    else:
        from mpcc_tuning.plant_scuderia import ScuderiaPlant
        P = ScuderiaPlant(t, model=plant, dt=0.05)
        P.max_steps = steps
    s5 = P.reset()
    m.reset()
    off = False
    s0 = float(s5[4])
    nok = k = 0
    vmax = 0.0
    for _ in range(steps):
        o = m.value(s5, th)
        nok += int(bool(o["ok"]))
        k += 1
        s5, _r, off, tr = P.step(o["u0"])
        vmax = max(vmax, float(s5[3]))
        if off or tr:
            break
    cov = float(s5[4]) - s0
    return (qc, qv, rd), cov / t.length, bool(off), 100.0 * nok / max(k, 1), vmax


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--plant", default="std")
    ap.add_argument("--laps", type=float, default=2.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(ROOT / "benchmarks" / "results"
                    / f"std_baseline_{a.track}_{a.plant}.json")

    jobs = [(qc, qv, rd, a.plant, a.horizon, a.laps, a.track, a.max_iter)
            for qc, qv, rd in GRID]
    t = getattr(Track, a.track)()
    print(f"  {a.track} on the {a.plant} plant, {t.length:.1f} m lap, "
          f"target {a.laps:g} laps, {len(jobs)} runs over {a.jobs} workers",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for key, laps, off, ok, vmax in ex.map(one, jobs):
            rows.append((key, laps, off, ok, vmax))
            print(f"    q_c={key[0]:<4.1f} q_v={key[1]:<5.2f} r_d={key[2]:<5.1f}"
                  f"  {laps:5.2f} laps {'OFF' if off else 'ok '}"
                  f"  solve {ok:3.0f}%  peak v {vmax:4.2f}", flush=True)

    good = [r for r in rows if not r[2] and r[1] >= a.laps - 0.05]
    print()
    if good:
        best = max(good, key=lambda r: (r[3], r[1]))
        print(f"  PASSES on the {a.plant} plant: q_c={best[0][0]}, "
              f"q_v={best[0][1]}, r_d={best[0][2]} -> {best[1]:.2f} laps, "
              f"solve {best[3]:.0f}%, peak {best[4]:.2f} m/s")
        print("  A baseline on a plant with tyres. Grip, sectors and opponents")
        print("  can now be studied on something that can actually slide.")
    else:
        best = max(rows, key=lambda r: (not r[2], r[1]))
        print(f"  NO SETTING COMPLETES {a.laps:g} LAPS on {a.plant}. Best: "
              f"q_c={best[0][0]}, q_v={best[0][1]}, r_d={best[0][2]} -> "
              f"{best[1]:.2f} laps, solve {best[3]:.0f}%")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"track": a.track, "plant": a.plant, "laps_target": a.laps,
         "rows": [{"q_c": r[0][0], "q_v": r[0][1], "r_d": r[0][2],
                   "laps": r[1], "off": r[2], "solve_ok": r[3], "peak_v": r[4]}
                  for r in rows]}, indent=2) + "\n")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
