"""Why is LTC N=3 < MLP? Hypothesis: the q_c floor (good for N=1,2 -- holds the line) RESTRICTS the deviation
LTC needs to weave through a 3-car crowd. Eval the LTC q_c-floor net at N=3 (phys 0..5 x seeds 0,1) with the
floor ON vs OFF, and log q_c to see whether the net WANTS to drop below the 0.5 floor mid-crowd.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/ltc_n3_floor_check.py --jobs 4
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
OUTF = OUT / "ltc_n3"; OUTF.mkdir(parents=True, exist_ok=True)


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def job(j):
    floor_on, seed, ego_pace = j
    m = _mpcc(f"nf_{int(floor_on)}_{seed}"); pol = _load_pol("ltc", seed, qc_floor_on=floor_on)
    rows = []; qc = None
    for phys in range(6):
        want = (seed == 0 and phys == 1)                       # log one N=3 run's q_c
        met, lg = run_eval(m, ego_pace, pol, 3, phys, steps=2600, log=want)
        met.update(floor_on=bool(floor_on), seed=seed, phys=phys); rows.append(met)
        if want and lg is not None:
            qc = np.asarray(lg["THETA"])[:, 0]                 # q_c over the run (idx 0)
        print(f"  [floor={'ON ' if floor_on else 'OFF'} s{seed}] N=3 phys={phys}: "
              f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')}", flush=True)
    return rows, (bool(floor_on), qc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("nf_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; LTC N=3 floor ON vs OFF, phys 0..5 x seeds 0,1", flush=True)
    jobs = [(True, s, ego_pace) for s in (0, 1)] + [(False, s, ego_pace) for s in (0, 1)]
    allrows = []; qcs = {}
    if a.jobs <= 1:
        for j in jobs:
            rr, (fo, qc) = job(j); allrows += rr
            if qc is not None:
                qcs[fo] = qc
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for rr, (fo, qc) in pool.imap_unordered(job, jobs):
                allrows += rr
                if qc is not None:
                    qcs[fo] = qc
    print("\n==== LTC N=3: floor ON vs OFF (n=12 each) ====")
    for fo in (True, False):
        rs = [r for r in allrows if r["floor_on"] == fo]
        c = 100 * np.mean([r["clean"] for r in rs]); ct = 100 * np.mean([r["contact"] for r in rs])
        of = 100 * np.mean([r["off"] for r in rs]); pa = float(np.mean([r["passes"] for r in rs]))
        print(f"  floor {'ON ' if fo else 'OFF'}: clean {c:.0f}%  contact {ct:.0f}%  off {of:.0f}%  passes {pa:.2f}")
    for fo in (True, False):
        if fo in qcs:
            q = qcs[fo]
            print(f"  q_c (floor {'ON ' if fo else 'OFF'}) over an N=3 run: min {q.min():.2f} mean {q.mean():.2f} "
                  f"%wanting<0.5 {(q < 0.5).mean() * 100:.0f}%")
    np.savez(OUTF / "ltc_n3_floor_check.npz", rows=np.array(allrows, dtype=object),
             qc_on=qcs.get(True, np.zeros(0)), qc_off=qcs.get(False, np.zeros(0)))


if __name__ == "__main__":
    main()
