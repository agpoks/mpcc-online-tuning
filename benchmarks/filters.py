"""Every safety filter, on the same plant, controller and track.

    python benchmarks/filters.py
    python benchmarks/filters.py --grip 0.7      # a plant the filters are wrong about

Two controllers are used deliberately:

``good``    the default weights, which never crash. A filter should be
            **invisible** here -- any intervention is pure cost.
``bad``     the weights the tuner collapsed to, which crash in 68 steps. A
            filter should keep the car alive.

A filter that fails the first test is a controller. A filter that fails the
second is decoration. Results land in ``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tune_online import Plant  # noqa: E402
from mpcc_tuning.filters import FILTERS  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

WEIGHTS = {
    "good": MPCCWeights().to_log(),
    "bad": MPCCWeights(q_c=0.41, q_l=1.40, q_v=22.0, r_d=0.019,
                    r_a=0.006, r_dv=0.019).to_log(),
}


def run(track, mpcc, theta, filt, grip=1.0, steps=400):
    plant = Plant(track, grip=grip, dt=0.05)
    plant.max_steps = steps
    s5 = plant.reset()
    mpcc.reset()
    R, off, t_filt, i = 0.0, False, 0.0, 0
    for i in range(steps):
        u = mpcc.value(s5, theta)["u0"]
        if filt is not None:
            t0 = time.perf_counter()
            u, _iv = filt(s5, u)
            t_filt += time.perf_counter() - t0
        s5, r, off, tr = plant.step(u)
        R += r
        if filt is not None and hasattr(filt, "observe"):
            filt.observe(s5)
        if off or tr:
            break
    return dict(covered=float(R), steps=i + 1, off=bool(off),
                iv=float(filt.intervention_rate) if filt else 0.0,
                no_safe=int(filt.n_no_safe_action) if filt else 0,
                us_per_tick=1e6 * t_filt / (i + 1) if filt else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    track = Track.oval()
    mpcc = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05, max_iter=60)
    out = {"_meta": dict(grip=a.grip, steps=a.steps)}

    print(f"  plant grip {a.grip}\n")
    print(f"  {'filter':<16}{'weights':<7}{'covered':>9}{'steps':>7}"
          f"{'outcome':>11}{'overridden':>12}{'no-safe':>9}{'us/tick':>9}", flush=True)
    for name in ["none"] + list(FILTERS):
        for wname, theta in WEIGHTS.items():
            filt = None if name == "none" else FILTERS[name](track)
            r = run(track, mpcc, theta, filt, grip=a.grip, steps=a.steps)
            out.setdefault(name, {})[wname] = r
            print(f"  {name:<16}{wname:<7}{r['covered']:9.1f}{r['steps']:7d}"
                  f"{('OFF-TRACK' if r['off'] else 'survived'):>11}"
                  f"{r['iv']:11.0%}{r['no_safe']:9d}{r['us_per_tick']:9.0f}",
                  flush=True)
    path = Path(a.out or ROOT / "benchmarks" / "results" /
                f"filters_grip{a.grip:g}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
