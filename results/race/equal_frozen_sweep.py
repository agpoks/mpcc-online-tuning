#!/usr/bin/env python3
"""DEPLOYED (frozen) memory comparison on the EQUAL-pace occluded opponent, over MATCHED starts.
The paper's fig:blind numbers (equal clean 0.91 vs 0.36) come from an ONLINE-LEARNING run averaged
over all episodes incl. the rough early ones -- i.e. partly a learning transient. This measures the
SHIPPED policy instead: each arm's trained net (seeds 1-3) replayed FROZEN (no learning) on a set of
matched starts (same s0/v0/opponent for LTC and MLP), under the 270-deg occluding sensor. Reports
deployed clean-rate, pass-rate and pass-TIMING with real n, so RQ2 can state a number the deployed
car reproduces. NEEDS acados -> self-re-execs onto ~/acados-v060 + scuderia_gym_jax.
    python3 results/race/equal_frozen_sweep.py --net-seeds 1 2 3 --starts 1 2 3
Writes results/race_phase2/sensor_eqsweep_{arm}_{start}_n{netseed}.npz and prints the table.
"""
import sys, os, argparse
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
OUT = ROOT / "results/race_phase2"; DT = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--starts", type=int, nargs="*", default=[1, 2, 3])   # each -> a distinct s0/v0 (matched across arms)
    ap.add_argument("--gap0", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=2600)                    # 130 s: catches normal passes; a very late one reads as "no pass in window"
    ap.add_argument("--detect-range", type=float, default=18.0)
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=DT, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="eqsweep")
    ego_pace = measure_pace(m, track, th0, 2000)
    print(f"ego solo pace {ego_pace:.2f} m/s; nets {a.net_seeds} x starts {a.starts}; equal; OCCLUSION; steps {a.steps}", flush=True)
    rows = []
    for ns in a.net_seeds:
        pols = {arm: _load_pol(arm, ns) for arm in ("ltc", "mlp")}
        if any(pols[arm] is None for arm in pols):
            print(f"  [net {ns}] missing a net -- skipped", flush=True); continue
        for sidx in a.starts:
            for arm in ("ltc", "mlp"):
                r = run_case(m, ego_pace, pols[arm], sidx, "eqsweep", "equal", a.gap0, a.detect_range,
                             a.steps, FOV_DEG, arm, tag=f"_n{ns}", occlude=True)
                # pass time from the saved npz (first + -> - crossing)
                d = np.load(OUT / f"sensor_eqsweep_{arm}_{sidx}_n{ns}.npz", allow_pickle=True)
                G = np.asarray(d["GAP"], float)
                cr = np.where((G[:-1] > 0) & (G[1:] <= 0) & (np.abs(G[1:]) < 10))[0]
                pt = float(cr[0] * DT) if len(cr) else np.nan
                rows.append(dict(net=ns, start=sidx, arm=arm, clean=bool(r["clean"]), passed=len(cr) > 0,
                                 npass=int(r["passes"]), pass_t=pt, off=bool(r["off"]), contact=bool(r["contact"])))

    def rate(arm, key): return np.mean([r[key] for r in rows if r["arm"] == arm]) if rows else np.nan
    def med_pt(arm):
        v = [r["pass_t"] for r in rows if r["arm"] == arm and r["passed"]]
        return float(np.median(v)) if v else np.nan
    n = len([r for r in rows if r["arm"] == "ltc"])
    print(f"\n=== DEPLOYED (frozen) equal, n={n} matched encounters per arm ===")
    print(f"{'arm':4s} {'clean':>6s} {'pass':>6s} {'contact':>8s} {'off':>6s} {'med pass-t':>11s}")
    for arm in ("ltc", "mlp"):
        print(f"{arm:4s} {rate(arm,'clean'):6.2f} {rate(arm,'passed'):6.2f} {rate(arm,'contact'):8.2f} "
              f"{rate(arm,'off'):6.2f} {med_pt(arm):10.1f}s")
    # matched pass-time deltas (LTC earlier = negative)
    print("\nmatched (net,start): LTC pass-t | MLP pass-t")
    for ns in a.net_seeds:
        for sidx in a.starts:
            gl = [r for r in rows if r["arm"] == "ltc" and r["net"] == ns and r["start"] == sidx]
            gm = [r for r in rows if r["arm"] == "mlp" and r["net"] == ns and r["start"] == sidx]
            if gl and gm:
                print(f"  n{ns} s{sidx}: LTC {gl[0]['pass_t']:6.1f}s  MLP {gm[0]['pass_t']:6.1f}s")
    np.savez(OUT / "equal_frozen_sweep.npz", rows=np.array(rows, dtype=object))
    print(f"\nwrote {OUT}/equal_frozen_sweep.npz  (+ per-run sensor_eqsweep_*.npz)")


if __name__ == "__main__":
    main()
