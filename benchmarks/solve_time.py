"""Does the controller fit in a control tick? Measured, not asserted.

    python benchmarks/solve_time.py

The repo shipped for a long time with a "what is not real-time yet" section
saying the solve took ~150 ms against a 50 ms budget. That was true of IPOPT
solved to convergence. It is not true of the SQP-RTI in ``mpcc_tuning/rti.py``,
and the difference is large enough that the claim had to be re-measured rather
than edited.

Reported: mean *and worst case*. For a control loop the worst case is the number
that matters -- a mean inside budget with a tail outside it is a controller that
misses deadlines, and IPOPT at N=12 is exactly that.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.rti_influence import reference_states  # noqa: E402
from mpcc_tuning.model import KinematicBicycle  # noqa: E402
from mpcc_tuning.mpcc import MPCC, MPCCWeights  # noqa: E402
from mpcc_tuning.rti import RTISolver  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402

BUDGET_MS = {20: 50.0, 50: 20.0, 100: 10.0}


def main():
    track = Track.oval()
    theta = MPCCWeights().to_log()
    ref = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=12, dt=0.05,
               max_iter=200)
    states = reference_states(track, ref, theta, 60)
    rows = {}

    print(f"  {'solver':<30}{'mean ms':>9}{'worst ms':>10}"
          f"{'  20 Hz':>9}{'  50 Hz':>9}{' 100 Hz':>9}")
    for N in (12, 20):
        m = MPCC(track, model=KinematicBicycle(dt=0.05), horizon=N, dt=0.05,
                 max_iter=200)
        m.reset()
        ts = []
        for s5 in states:
            t0 = time.perf_counter()
            m.value(s5, theta)
            ts.append(1e3 * (time.perf_counter() - t0))
        rows[f"ipopt_N{N}"] = dict(mean=float(np.mean(ts)), worst=float(np.max(ts)))

        r = RTISolver(m, qp="qrqp")
        r.reset()
        ts2 = []
        for s5 in states:
            t0 = time.perf_counter()
            r.solve(s5, theta)
            ts2.append(1e3 * (time.perf_counter() - t0))
        rows[f"rti_N{N}"] = dict(mean=float(np.mean(ts2)), worst=float(np.max(ts2)))

    for key, d in rows.items():
        fits = "".join(f"{'ok' if d['worst'] <= b else 'MISS':>9}"
                       for b in BUDGET_MS.values())
        name = key.replace("ipopt_", "IPOPT converged, ").replace("rti_", "SQP-RTI, ")
        print(f"  {name:<30}{d['mean']:>9.2f}{d['worst']:>10.2f}{fits}")

    print("\n  The worst case is the number that matters for a control loop.")
    p = ROOT / "benchmarks" / "results" / "solve_time.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
