"""Drive a saved network COLD and WARM, reproducibly, and print both.

    python3 scripts/warm_cold_check.py <network.npz> [--warm-from <gentle.npz>]

The online-adaptation networks scored well in-experiment but not when
re-driven cold (drive_policy.py). This tells you WHY, reproducibly, by
separating two things that were conflated:

* **cold** -- a freshly built acados solver, seeded by rolling the model
  forward at low speed (`AcadosMPCC._seed`), then the network drives. This is
  a standing start: what the car sees when it is switched on. `drive_policy.py`
  measures exactly this.

* **warm** -- a freshly built solver that is first driven by a GENTLE network
  (``--warm-from``, default the grid-fitted START network, which drives
  cleanly) for one lap, so the solver accumulates a good working iterate; then
  WITHOUT resetting the solver the target network takes over and is measured.
  This reproduces the in-experiment condition, where the aggressive banked
  network was evaluated on a solver already warmed by the whole learning run.

If a network does well warm but crashes cold, its weights only stay feasible
given a warm start it cannot produce itself -- the score was real but belongs
to the (network + solver-history) pair, not the network. If cold ~ warm, the
network is genuinely robust. Both runs are deterministic: same start, same
plant, same seed, so you can rerun and check.

    cold  == drive_policy.py (a standing start; the deployable number)
    warm  == the in-experiment eval (a solver handed a good iterate)
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


def drive(m, pol, t, s0, v0, steps, warm_pol=None, warm_steps=0):
    """Drive ``pol`` for ``steps`` and return laps + off. If ``warm_pol`` is
    given, first drive it for ``warm_steps`` on the SAME solver without reset,
    so the solver is warm when ``pol`` takes over. The measured laps are those
    covered by ``pol``, from wherever the warmup left the car."""
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = warm_steps + steps
    s5 = P.reset(s0=s0, v0=v0); m.reset()
    if warm_pol is not None and warm_steps > 0:
        warm_pol.reset()
        th = warm_pol.step(features(t, s5))
        for _ in range(warm_steps):
            u = m.value(P.state_dyn(), th)["u0"]
            s5, r, off, tr = P.step(u)
            th = warm_pol.step(features(t, s5))
            if off or tr:                       # warmup itself left the track
                return 0.0, True, "warmup crashed"
    pol.reset()                                 # network hidden state fresh; SOLVER stays warm
    base = float(s5[4]); off = tr = False
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
    ap.add_argument("--warm-from", default=None,
                    help="network to warm the solver with (default: the grid-fitted "
                         "START network results/fitted_policy_<track>_kv0.50.npz)")
    ap.add_argument("--warm-steps", type=int, default=1100, help="~one lap of warmup")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)

    from mpcc_tuning.acados_mpcc import AcadosMPCC
    t = getattr(Track, a.track)(); st = B.start(a.track); th0 = np.asarray(st.theta(), float)
    warm_file = a.warm_from or str(ROOT / "results" / f"fitted_policy_{a.track}_kv0.50.npz")
    box = None
    zt = np.load(a.npz)
    if "lo" in zt and "hi" in zt:
        box = (zt["lo"], zt["hi"])
    else:                                       # v1 files: take the box from the warm-from network
        wb = np.load(warm_file); box = (wb["lo"], wb["hi"])
    zw = np.load(warm_file)

    print(f"  target: {Path(a.npz).name}"
          + (f"  (banked {float(zt['best_laps']):.2f})" if "best_laps" in zt else ""))
    print(f"  warm-from: {Path(warm_file).name}")
    print(f"  {'seed':>4} {'start s0':>9} {'COLD (standing start)':>24} {'WARM (solver pre-driven)':>26}")
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, name=f"wc_{os.getpid()}")
    rows = []
    for seed in a.seeds:
        s0 = (seed % 4) * t.length / 4.0; v0 = 1.0 + 0.1 * (seed % 3)
        pol = _pol(zt, box, th0, seed)
        c_laps, c_off, _ = drive(m, pol, t, s0, v0, st.steps)
        pol = _pol(zt, box, th0, seed)
        warm = _pol(zw, box, th0, seed)
        w_laps, w_off, note = drive(m, pol, t, s0, v0, st.steps, warm_pol=warm, warm_steps=a.warm_steps)
        rows.append((seed, c_laps, c_off, w_laps, w_off))
        print(f"  {seed:>4} {s0:>9.1f} {c_laps:>18.2f} {'OFF' if c_off else 'clean':>5} "
              f"{w_laps:>18.2f} {('OFF' if w_off else 'clean') if note=='ok' else note:>7}", flush=True)
    cold_ok = sum(1 for _, _, o, _, _ in rows if not o)
    warm_ok = sum(1 for _, _, _, _, o in rows if not o)
    print(f"  cold: {cold_ok}/{len(rows)} clean, mean {np.mean([r[1] for r in rows]):.2f}; "
          f"warm: {warm_ok}/{len(rows)} clean, mean {np.mean([r[3] for r in rows]):.2f}")
    print("  A network that is clean WARM but OFF COLD only works given a warm start it cannot")
    print("  produce from a standing start: the score was real but not the network's alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
