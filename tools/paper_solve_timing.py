"""Computational-time analysis of the acados MPCC solve (the shipping stack): per-solve wall time vs
horizon N, against the 20 Hz (50 ms) real-time budget. Builds the SAME dynamic/discrete/corridor OCP
race_mode deploys, warms up, then times pure m.value() calls along a drive. Writes a table and a plot.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/paper_solve_timing.py
Outputs: results/race/solve_timing.md , results/race/paper/fig_solve_timing.{pdf,png}
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant

BUDGET_MS = 50.0   # 20 Hz


def time_horizon(N, n_solve=250, warm=25):
    tr = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(tr, horizon=N, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"timing_N{N}")
    P = ScuderiaPlant(tr, model="std", dt=0.05); P.reset(s0=0.0, v0=1.5); m.reset(); m.set_obstacles([(1e3, 1e3, 0.36)])
    for _ in range(warm):
        P.step(m.value(P.state_dyn(), th0)["u0"])
    ts = []
    for _ in range(n_solve):
        t = time.time(); out = m.value(P.state_dyn(), th0); ts.append((time.time() - t) * 1e3); P.step(out["u0"])
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--horizons", type=int, nargs="+", default=[15, 20, 25, 30, 40])
    a = ap.parse_args()
    rows = []
    for N in a.horizons:
        ts = time_horizon(N)
        rows.append((N, ts.mean(), np.percentile(ts, 50), np.percentile(ts, 95), ts.max()))
        print(f"N={N:3d}: mean {ts.mean():6.1f}  p50 {np.percentile(ts,50):6.1f}  p95 {np.percentile(ts,95):6.1f}  max {ts.max():6.1f} ms", flush=True)
    # table
    md = ["# acados MPCC solve-time analysis (dynamic + discrete + two-layer corridor, STD plant)", "",
          f"Pure `m.value()` wall time, single process (acados built WITHOUT OpenMP), {len(a.horizons)} horizons, "
          f"250 solves each along a drive. Real-time budget = **{BUDGET_MS:.0f} ms (20 Hz)**. The cost is dominated "
          "by `hessian_approx=EXACT`, required because the cost is `EXTERNAL` (keeps progress linear) and the online "
          "weight-tuning gradient needs the solver's exact Hessian.", "",
          "| horizon N | lookahead | mean [ms] | p50 | p95 | max | vs 20 Hz |", "|---:|---:|---:|---:|---:|---:|:--|"]
    for N, mn, p50, p95, mx in rows:
        md.append(f"| {N} | {N*0.05:.2f} s | {mn:.1f} | {p50:.1f} | {p95:.1f} | {mx:.1f} | {'OK' if p95 < BUDGET_MS else 'OVER'} |")
    (ROOT / "results/race/solve_timing.md").write_text("\n".join(md) + "\n")
    # plot
    Ns = [r[0] for r in rows]; mean = [r[1] for r in rows]; p95 = [r[3] for r in rows]; mx = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(0, BUDGET_MS, color="#d6f0d6", alpha=.7, zorder=0); ax.axhline(BUDGET_MS, color="#2a8f3a", lw=1.2, ls="--", label="20 Hz budget (50 ms)")
    ax.fill_between(Ns, mean, p95, color="#1f4e8c", alpha=.18, label="mean–p95")
    ax.plot(Ns, mean, "-o", color="#1f4e8c", lw=1.8, label="mean solve time")
    ax.plot(Ns, mx, ":", color="#8a3ffc", lw=1.2, label="max")
    ax.axvline(25, color="0.5", ls=":", lw=1); ax.text(25.3, ax.get_ylim()[1]*0.05, "deployed N=25", fontsize=8, color="0.4")
    ax.set_xlabel("MPCC horizon N (steps, dt=0.05 s)"); ax.set_ylabel("per-solve wall time [ms]")
    ax.set_title("acados MPCC solve time vs horizon (single thread, no OpenMP)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5); ax.grid(alpha=.25)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(ROOT / f"results/race/paper/fig_solve_timing.{e}", dpi=150)
    print("wrote results/race/solve_timing.md + results/race/paper/fig_solve_timing.{pdf,png}")


if __name__ == "__main__":
    main()
