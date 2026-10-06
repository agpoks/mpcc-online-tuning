"""Phase-A de-risk: acados QP solve time vs number of opponents (max_obstacles). Two opponents = more
keep-out constraints = a heavier QP than the N=1 case we measured. This checks whether the 2-3 car race
still fits the 50 ms / 20 Hz budget -- on the WORKSTATION (idle). The onboard RC-car compute will be
slower, so this is the relative scaling + a go/no-go sanity check, not the final onboard number.

Run on an IDLE machine (a concurrent CPU job inflates + flattens the numbers).
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/solve_timing_obstacles.py
"""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import RacelineOpponent
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, KEEPOUT_R, measure_pace


def time_obs(max_obs, steps=600):
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=max_obs, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"timing_obs{max_obs}")
    ego_pace = measure_pace(m, track, th0, 1500)
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps + 50; P.reset(s0=0.0, v0=1.4); m.reset()
    opps = [RacelineOpponent(track, s0=(2.5 + 3.5 * i) % track.length, pace=1.0 * ego_pace, offset=0.0,
                             radius=KEEPOUT_R, a_lat=2.5) for i in range(max_obs)]
    for o in opps:
        o.reset()
    theta = th0; m.set_obstacles([o.keepout() for o in opps]); u = m.value(P.state_dyn(), theta)["u0"]
    ts = []
    for _ in range(steps):
        P.step(u); ex, ey = float(P._x[0]), float(P._x[1])
        for o in opps:
            o.step(0.05, ego=(ex, ey, float(P._x[3])))
        m.set_obstacles([o.keepout() for o in opps])
        t0 = time.perf_counter(); out = m.value(P.state_dyn(), theta); ts.append((time.perf_counter() - t0) * 1e3)
        u = out["u0"]
    return np.array(ts[30:])   # drop warm-up


if __name__ == "__main__":
    print(f"acados solve time vs opponents (horizon={B.start('icra_t2_smooth').horizon}, workstation idle):")
    print(f"{'max_obstacles':>13} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8} {'mean ms':>8} {'p95<50ms?':>10}")
    rows = []
    for N in (1, 2, 3):
        ts = time_obs(N); p50, p95 = np.percentile(ts, 50), np.percentile(ts, 95)
        rows.append((N, p50, p95, ts.max(), ts.mean()))
        print(f"{N:>13} {p50:>8.1f} {p95:>8.1f} {ts.max():>8.1f} {ts.mean():>8.1f} {'YES' if p95 < 50 else 'NO':>10}", flush=True)
    with open(ROOT / "results/race_phase2/solve_timing_obstacles.md", "w") as fh:
        fh.write("# acados solve time vs opponents (max_obstacles), workstation idle\n\n")
        fh.write("| max_obstacles | p50 ms | p95 ms | max ms | mean ms |\n|---|---|---|---|---|\n")
        for N, p50, p95, mx, mn in rows:
            fh.write(f"| {N} | {p50:.1f} | {p95:.1f} | {mx:.1f} | {mn:.1f} |\n")
        fh.write("\n50 ms = 20 Hz budget. Onboard RC compute is slower -- re-measure there before the real race.\n")
    print("wrote results/race_phase2/solve_timing_obstacles.md")
