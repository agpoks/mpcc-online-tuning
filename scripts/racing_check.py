"""Evaluate a policy the way a race actually starts: after a warm-up lap.

    python3 scripts/racing_check.py <network.npz> [--warm-from <safe.npz>]

Whether the acados solver starts warm or cold is a solver-initialisation
question, not a tuning one: an aggressive HAND-tuned weight set has the same
trouble from a dead stop. Real races solve it the obvious way -- a warm-up /
formation lap to the grid, so by the time the race starts the solver (and the
policy's own recurrent memory) are already initialised. This script measures
that deployment condition, and contrasts it with a cold standing start.

Two scenarios, both deterministic and rerunnable:

* **standing start** -- fresh solver, fresh policy memory, from a stop. What
  the car sees switched on cold. (= `drive_policy.py`.)

* **flying start (racing)** -- a warm-up lap is driven under a SAFE policy
  (``--warm-from``, default `baselines.START`), and CRUCIALLY the target
  network runs alongside it the whole warm-up lap, watching the same features,
  so at the grid BOTH the solver and the network's hidden state are warm. Then
  the target policy takes the wheel and is measured. No hidden-state reset at
  the handoff -- that reset was an artefact of an earlier, cruder check.

A policy that is fast flying but crashes from a standing start is not broken:
it simply needs the warm-up lap every race gives it. A policy that also
survives the standing start is more robust. Reporting both says which is which.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.ltc import LTCCell, N_FEATURES, WeightPolicy, features  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def _pol(z, box, th0, seed):
    p = WeightPolicy(LTCCell(N_FEATURES, 12, seed=seed), th0, box[0], box[1], seed=seed)
    p.G[...] = z["G"]; p.cell.p[...] = z["cell_p"]
    return p


def standing(m, pol, t, s0, v0, steps):
    """Fresh everything, from a stop."""
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = steps
    s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
    base = float(s5[4]); off = tr = False
    th = pol.step(features(t, s5))
    for _ in range(steps):
        u = m.value(P.state_dyn(), th)["u0"]
        s5, r, off, tr = P.step(u)
        th = pol.step(features(t, s5))
        if off or tr:
            break
    return (float(s5[4]) - base) / t.length, bool(off)


def flying(m, pol, t, s0, v0, steps, warm_theta, warm_steps):
    """A warm-up lap under a SAFE constant, with the network watching, then race.

    The warm-up drives fixed ``warm_theta`` (a clean baseline) so the lap is
    guaranteed to complete and warm the solver. The target network ``pol``
    steps every tick on the same features -- so its recurrent state is warm at
    the grid -- but its output is NOT used until the handoff. At the grid the
    network takes over with the solver AND its own memory initialised.
    """
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = warm_steps + steps
    s5 = P.reset(s0=s0, v0=v0); m.reset(); pol.reset()
    for _ in range(warm_steps):                 # warm-up lap: safe weights drive, network watches
        pol.step(features(t, s5))               # warms the network's hidden state
        u = m.value(P.state_dyn(), warm_theta)["u0"]
        s5, r, off, tr = P.step(u)
        if off or tr:
            return 0.0, True, "warm-up lap crashed (the safe baseline should not)"
    base = float(s5[4]); off = tr = False        # grid: start counting, everything warm
    th = pol.step(features(t, s5))
    for _ in range(steps):
        u = m.value(P.state_dyn(), th)["u0"]
        s5, r, off, tr = P.step(u)
        th = pol.step(features(t, s5))
        if off or tr:
            break
    return (float(s5[4]) - base) / t.length, bool(off), "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--warm-steps", type=int, default=1100, help="~one lap of warm-up")
    ap.add_argument("--box-from", default=None,
                    help="npz with lo/hi if the network file has no box (v1 archive)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)

    from mpcc_tuning.acados_mpcc import AcadosMPCC
    t = getattr(Track, a.track)(); st = B.start(a.track); th0 = np.asarray(st.theta(), float)
    warm_theta = np.asarray(st.theta(), float)   # the safe baseline drives the warm-up lap
    z = np.load(a.npz)
    if "lo" in z and "hi" in z:
        box = (z["lo"], z["hi"])
    else:
        bf = a.box_from or str(ROOT / "results" / f"fitted_policy_{a.track}_kv0.50.npz")
        b = np.load(bf); box = (b["lo"], b["hi"])

    print(f"  {Path(a.npz).name}" + (f"  (banked {float(z['best_laps']):.2f})" if "best_laps" in z else ""))
    print(f"  warm-up lap driven by baselines.START; race policy engages at the grid")
    print(f"  {'seed':>4} {'s0':>6} {'STANDING start':>16} {'FLYING start (racing)':>24}")
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, name=f"race_{os.getpid()}")
    rows = []
    for seed in a.seeds:
        s0 = (seed % 4) * t.length / 4.0; v0 = 1.0 + 0.1 * (seed % 3)
        s_laps, s_off = standing(m, _pol(z, box, th0, seed), t, s0, v0, st.steps)
        f_laps, f_off, note = flying(m, _pol(z, box, th0, seed), t, s0, v0, st.steps, warm_theta, a.warm_steps)
        rows.append((seed, s_laps, s_off, f_laps, f_off))
        print(f"  {seed:>4} {s0:>6.1f} {s_laps:>10.2f} {'OFF' if s_off else 'ok':>4} "
              f"{f_laps:>16.2f} {(('OFF' if f_off else 'ok') if note=='ok' else note):>7}", flush=True)
    print(f"  standing: {sum(1 for r in rows if not r[2])}/{len(rows)} clean, mean {np.mean([r[1] for r in rows]):.2f}; "
          f"flying: {sum(1 for r in rows if not r[4])}/{len(rows)} clean, mean {np.mean([r[3] for r in rows]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
