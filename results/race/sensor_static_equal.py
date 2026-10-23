#!/usr/bin/env python3
"""Generate matched LTC vs MLP per-tick runs on a STATIC (parked) and an EQUAL-pace opponent under the
270-deg occluding sensor -- the two cases missing from results/race_phase2 (MLP was only ever recorded
for slower/faster). Reuses experiments.race_sensor_eval.run_case (same logging + sensor gate), so the
output npz match the existing sensor_* schema. NEEDS acados -> run in WSL sim_ws, not the offline box.
    python3 results/race/sensor_static_equal.py --seeds 1 2 3
Writes results/race_phase2/sensor_{static,equal}_{ltc,mlp}_<seed>.npz ; then build the figure with
    python3 tools/paper_ltc_vs_mlp_time.py --cases static equal --seed 1
"""
import sys, os, argparse
# --- select the deploy acados (SQP_WITH_FEASIBLE_QP) + the STD plant, via the environment ---
# AcadosMPCC's default variant is fqp_soft_funnel (needs acados >=0.4.5); the default ~/acados is
# v0.1.9. Re-exec once onto ~/acados-v060 + scuderia_gym_jax so a bare `python3 <this>` runs.
# (see mpcc_tuning/acados_variants.ACADOS_V060; scuderia_gym_jax is the one allowed external plant.)
_V060 = os.path.expanduser("~/acados-v060")
if os.environ.get("ACADOS_SOURCE_DIR") != _V060:
    _env = dict(os.environ)
    _env["ACADOS_SOURCE_DIR"] = _V060
    _env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (_V060 + "/lib", _env.get("LD_LIBRARY_PATH", "")) if p)
    _env["PYTHONPATH"] = os.pathsep.join(p for p in (
        _V060 + "/interfaces/acados_template", os.path.expanduser("~/github/scuderia_gym_jax"),
        _env.get("PYTHONPATH", "")) if p)
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from experiments.race_mode import measure_pace, A_LAT_RACE, CORRIDOR_KW, FOV_DEG
from experiments.race_sensor_eval import run_case, _load_pol

# (case name, opponent kind, start gap ahead [m]) -- a parked car, and an equal-pace car just ahead
CASES = [("static", "static", 4.0), ("equal", "equal", 2.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--arms", nargs="*", default=["ltc", "mlp"])
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--detect-range", type=float, default=18.0)
    ap.add_argument("--no-occlude", action="store_true", help="disable car-height line-of-sight occlusion")
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="sensor_eval_se")
    ego_pace = measure_pace(m, track, th0, 2000)
    print(f"ego solo pace {ego_pace:.2f} m/s; seeds {a.seeds}; cases {[c[0] for c in CASES]}"
          f"{'' if a.no_occlude else '; OCCLUSION'}", flush=True)
    for seed in a.seeds:
        for arm in a.arms:
            pol = _load_pol(arm, seed)
            if pol is None:
                print(f"  [seed {seed}][{arm}] no net at nets/race_{arm}_{seed}.npz -- skipped", flush=True); continue
            for (n, k, g) in CASES:
                run_case(m, ego_pace, pol, seed, n, k, g, a.detect_range, a.steps, FOV_DEG, arm,
                         tag="", occlude=not a.no_occlude)
    print("done -> results/race_phase2/sensor_{static,equal}_{ltc,mlp}_<seed>.npz")


if __name__ == "__main__":
    main()
