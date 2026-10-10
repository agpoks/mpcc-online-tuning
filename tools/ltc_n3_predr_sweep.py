"""Last N=3 swing: with the crowd band now opening LTC's line, sweep the PREDICTED keep-out radius PRED_R
(clearance from an OCCLUDED car) at N=3 to clear the remaining occluded-car clips. Eval-time only (PRED_R is
an MPCC obstacle param, not a net weight -- no retrain). Watch off-track: too big a PRED_R ran LTC wide in
the hard-fix. Floor+crowd LTC net, N=3, phys 0..5 x seeds 0,1 (n=12 per PRED_R).
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/ltc_n3_predr_sweep.py --jobs 3
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, PRED_R, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
OUTN = OUT / "ltc_n3"; OUTN.mkdir(parents=True, exist_ok=True)
PREDRS = [0.28, 0.32, 0.36]      # 0.28 = current; bigger = more occluded-car clearance (risks off-track)


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def job(j):
    pr, seed, ego_pace = j
    m = _mpcc(f"pr{int(pr * 100)}_{seed}"); pol = _load_pol("ltc", seed)   # floor + crowd band (deploy config)
    rows = []
    for phys in range(6):
        met, _ = run_eval(m, ego_pace, pol, 3, phys, steps=2600, pred_r=pr)
        met.update(pred_r=pr, seed=seed); rows.append(met)
        print(f"  [PRED_R={pr} s{seed}] N=3 phys={phys}: "
              f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("pr_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; LTC floor+crowd N=3, PRED_R sweep {PREDRS} (default {PRED_R})", flush=True)
    jobs = [(pr, s, ego_pace) for pr in PREDRS for s in (0, 1)]
    allrows = []
    if a.jobs <= 1:
        for j in jobs:
            allrows += job(j)
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for rr in pool.imap_unordered(job, jobs):
                allrows += rr
    print("\n==== LTC floor+crowd N=3 vs PRED_R (n=12 each) ====")
    best = None
    for pr in PREDRS:
        rs = [r for r in allrows if abs(r["pred_r"] - pr) < 1e-6]
        c = 100 * np.mean([r["clean"] for r in rs]); ct = 100 * np.mean([r["contact"] for r in rs])
        of = 100 * np.mean([r["off"] for r in rs]); pa = float(np.mean([r["passes"] for r in rs]))
        tag = "  <- current" if abs(pr - PRED_R) < 1e-6 else ""
        print(f"  PRED_R={pr}: clean {c:.0f}%  contact {ct:.0f}%  off {of:.0f}%  passes {pa:.2f}{tag}")
        if of == 0 and (best is None or c > best[1]):
            best = (pr, c)
    if best:
        print(f"\n  best (0 off-track, max clean): PRED_R={best[0]} at {best[1]:.0f}% clean")
    np.savez(OUTN / "ltc_n3_predr_sweep.npz", rows=np.array(allrows, dtype=object), predrs=np.array(PREDRS))


if __name__ == "__main__":
    main()
