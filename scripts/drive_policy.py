"""Re-drive a saved weight-policy network, frozen, on the standard starts.

    python3 scripts/drive_policy.py results/paper/frozen_summary_v1/best_policy_icra_t2_raceline_2_progress_mpcc_val.npz
    python3 scripts/drive_policy.py <file.npz> --seeds 0 1 2 --track icra_t2_raceline

Loads the network (readout G, LTC cell parameters) and the box it was trained
in, anchors it on ``baselines.START`` of the track, and drives it with learning
and exploration OFF for one 2500-step episode per start -- the same three
physically different starts every table in this repo uses. Prints laps and
whether the car stayed on the track, and the banked score the file carries so
the two can be compared.

Run this after ANY change to ``mpcc_tuning/track.py``: a policy learned on one
boundary does not transfer to another (a network banked at 2.66 laps on the
notched corridor drove 1.81 laps and left the track on the smoothed one).
Networks saved before the direct-path readout (2026-09-05) have a smaller G
and are rejected with a message rather than silently misloaded.
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


def load_policy(path, track_name, hidden=12, seed=0):
    z = np.load(path, allow_pickle=False)
    st = B.start(track_name)
    th0 = np.asarray(st.theta(), float)
    lo = z["lo"] if "lo" in z else B.adaptation_box(track_name, 2.0)[0]
    hi = z["hi"] if "hi" in z else B.adaptation_box(track_name, 2.0)[1]
    pol = WeightPolicy(LTCCell(N_FEATURES, hidden, seed=seed), th0, lo, hi, seed=seed)
    G = z["G"]
    if G.shape != pol.G.shape:
        raise SystemExit(
            f"{path}: readout G is {G.shape}, this policy expects {pol.G.shape}. "
            f"The file predates the direct-path readout (or a different hidden size); "
            f"it cannot be driven by the current WeightPolicy.")
    pol.G[...] = G
    pol.cell.p[...] = z["cell_p"]
    banked = float(z["best_laps"]) if "best_laps" in z else None
    meta = {k: (z[k].item() if z[k].shape == () else z[k].tolist())
            for k in z.files if k in ("critic", "clock", "box", "factor", "validated", "grid")}
    return pol, banked, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--track", default="icra_t2_raceline")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=0, help="0 = the track's baseline step budget")
    a = ap.parse_args(argv)

    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from experiments.online_from_baseline import run_frozen

    t = getattr(Track, a.track)()
    st = B.start(a.track)
    steps = a.steps or st.steps
    pol, banked, meta = load_policy(a.npz, a.track)
    print(f"  {Path(a.npz).name}: {meta}" + (f"  banked frozen laps {banked:.2f}" if banked is not None else ""))
    m = AcadosMPCC(t, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
                   name=f"drive_{a.track}")
    out = []
    for seed in a.seeds:
        s0 = (seed % 4) * t.length / 4.0
        v0 = 1.0 + 0.1 * (seed % 3)
        laps, off = run_frozen(m, pol, t, s0, v0, steps, features)
        out.append((seed, laps, off))
        print(f"  start {seed} (s0 {s0:5.1f} m, v0 {v0:.1f} m/s): {laps:.2f} laps {'OFF' if off else 'clean'}", flush=True)
    clean = [l for _, l, o in out if not o]
    print(f"  {len(clean)}/{len(out)} clean; laps {np.mean([l for _, l, _ in out]):.2f} mean"
          + (f", START {st.laps:.2f}, best constant {B.best(a.track).laps:.2f}" if a.track in B.TRACKS else ""))
    return 0 if len(clean) == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
