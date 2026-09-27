"""Which outer-wall scale stops the aggressive plan buying off the corridor?

Seed 2's off-track (reward4) is a SOLO-style buy-off: the opponent is ~13 m away, so the aggressive
policy (high k_v/q_v) is cutting the two-layer corridor for progress, not avoiding anyone. This drives
SOLO with seed-2-like aggressive fixed weights through the SAME two-layer corridor at a few candidate
corridor_outer_scale values and reports whether/when the PLANT goes off and how often the PLAN itself
predicts off. Cheap (one build per scale) way to pick the scale before a 30-min training run.

    ACADOS_SOURCE_DIR=... LD_LIBRARY_PATH=... PYTHONPATH=...:. python3 tools/corridor_scale_probe.py
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


def clearance(px, py, keep):
    s = track.project(float(px), float(py)); lat = float(track.lateral(float(px), float(py)))
    wl, wr = track.width(track.wrap(s))
    return min(float(wr) - keep - lat, lat + float(wl) - keep)


def probe(outer_scale, kv=0.76, qv_mult=1.6, steps=1500):
    kw = dict(CORRIDOR_KW); kw["corridor_outer_scale"] = float(outer_scale)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
                   discrete=True, max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **kw,
                   name=f"cor_probe_{int(outer_scale)}")
    N = m.N
    theta = th0.copy(); theta[7] = float(np.log(kv)); theta[2] = th0[2] + float(np.log(qv_mult))
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=0.0, v0=1.4); m.reset()
    m.set_obstacles([(1e3, 1e3, 0.36)])            # SOLO: no opponent (buy-off is solo-style)
    off_tick = None; plan_off_ticks = 0; n = 0; vmax = 0.0
    for t in range(steps):
        out = m.value(P.state_dyn(), theta); u = out["u0"]
        plan_off = min(clearance(m.sv.get(k, "x")[0], m.sv.get(k, "x")[1], 0.12) for k in range(1, N + 1))
        if plan_off < 0: plan_off_ticks += 1
        s5n, r, off, tr = P.step(u); n += 1; vmax = max(vmax, float(P._x[3]))
        if off or tr:
            off_tick = t; break
    return dict(scale=outer_scale, n=n, off_tick=off_tick, plan_off_pct=100 * plan_off_ticks / max(n, 1),
                vmax=vmax, survived=off_tick is None)


if __name__ == "__main__":
    scales = [float(x) for x in (sys.argv[1:] or [60, 200, 500])]
    print(f"solo aggressive drive (k_v=0.76, q_v x1.6) through the two-layer corridor:\n")
    for sc in scales:
        r = probe(sc)
        verdict = "SURVIVED (no off)" if r["survived"] else f"went OFF at tick {r['off_tick']}/{r['n']}"
        print(f"  outer_scale={sc:>6.0f}:  {verdict:>26s}  | plan-predicts-off {r['plan_off_pct']:4.0f}% of ticks  | vmax {r['vmax']:.2f} m/s")
