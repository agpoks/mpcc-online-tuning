"""Highest k_v that keeps the STD plant ON the line (solo, aggressive q_v) -> the k_v CEILING.

The corridor probe showed seed 2's off-track is NOT a plan buy-off (plan stays inside ~99%): it is
grip OVER-CLAIM -- at k_v=0.76 the plant eventually drifts off the racing line at 3+ m/s even with
fixed weights. k_v is a runtime parameter, so ONE build sweeps every k_v. Drives solo (buy-off is
solo-style) with q_v pushed, long enough to expose the drift, and reports the first off-track tick.
The ceiling is the highest k_v that survives with margin.

    ACADOS_SOURCE_DIR=... LD_LIBRARY_PATH=... PYTHONPATH=...:. python3 tools/kv_ceiling_probe.py
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW

track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
th0 = np.asarray(st.theta(), float)
m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
               discrete=True, max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW,
               name="kv_ceil_probe")


def drive(kv, qv_mult=1.6, steps=3000):
    theta = th0.copy(); theta[7] = float(np.log(kv)); theta[2] = th0[2] + float(np.log(qv_mult))
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=0.0, v0=1.4); m.reset()
    m.set_obstacles([(1e3, 1e3, 0.36)])
    vmax = 0.0
    for t in range(steps):
        out = m.value(P.state_dyn(), theta); u = out["u0"]
        s5n, r, off, tr = P.step(u); vmax = max(vmax, float(P._x[3]))
        if off or tr:
            return t, steps, vmax
    return None, steps, vmax


if __name__ == "__main__":
    kvs = [float(x) for x in (sys.argv[1:] or [0.50, 0.55, 0.60, 0.65, 0.70, 0.76])]
    print(f"solo aggressive drive (q_v x1.6), {3000} steps, outer_scale={CORRIDOR_KW['corridor_outer_scale']}:\n")
    for kv in kvs:
        off, n, vmax = drive(kv)
        verd = "SURVIVED (on-line whole run)" if off is None else f"OFF at tick {off}/{n}"
        print(f"  k_v={kv:.2f}:  {verd:>30s}  | vmax {vmax:.2f} m/s")
