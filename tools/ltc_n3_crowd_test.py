"""Cheap test of the crowd-aware q_c band on the EXISTING LTC net (trained WITHOUT it) at N=3:
  floor-only  = the current deploy (q_c held tight ~0.85 through the crowd) -- should reproduce ~58%.
  floor+crowd = force a looser WEAVING band once the 3-car field is REMEMBERED (q_c -> 0.30-0.45).
Confirms (a) the floor wasn't binding (floor-only ~ the known 58%, i.e. the floor is not the N=3 cause) and
(b) whether opening the line in a remembered crowd lifts N=3 -- before committing to a retrain. Logs q_c on a
floor+crowd run to verify the memory-crowd estimate actually reaches 3 under occlusion.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/ltc_n3_crowd_test.py --jobs 3
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
OUTN = OUT / "ltc_n3"; OUTN.mkdir(parents=True, exist_ok=True)
CFG = [("floor-only", True, False), ("floor+crowd", True, True)]   # (name, qc_floor_on, crowd_on)


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def job(j):
    name, fon, con, seed, ego_pace = j
    m = _mpcc(f"ct_{name.replace('+', '_').replace('-', '_')}_{seed}")
    pol = _load_pol("ltc", seed, qc_floor_on=fon, crowd_on=con)
    rows = []; qc = None
    for phys in range(6):
        want = (seed == 0 and phys == 1 and con)
        met, lg = run_eval(m, ego_pace, pol, 3, phys, steps=2600, log=want)
        met.update(cfg=name, seed=seed); rows.append(met)
        if want and lg is not None:
            qc = np.asarray(lg["THETA"])[:, 0]
        print(f"  [{name} s{seed}] N=3 phys={phys}: "
              f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')}", flush=True)
    return rows, (name, qc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("ct_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; LTC N=3 floor-only vs floor+crowd, phys 0..5 x seeds 0,1", flush=True)
    jobs = [(n, fon, con, s, ego_pace) for (n, fon, con) in CFG for s in (0, 1)]
    allrows = []; qcs = {}
    if a.jobs <= 1:
        for j in jobs:
            rr, (n, qc) = job(j); allrows += rr
            if qc is not None:
                qcs[n] = qc
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for rr, (n, qc) in pool.imap_unordered(job, jobs):
                allrows += rr
                if qc is not None:
                    qcs[n] = qc
    print("\n==== LTC N=3 (existing net): floor-only vs floor+crowd (n=12 each) ====")
    for name, _, _ in CFG:
        rs = [r for r in allrows if r["cfg"] == name]
        c = 100 * np.mean([r["clean"] for r in rs]); ct = 100 * np.mean([r["contact"] for r in rs])
        of = 100 * np.mean([r["off"] for r in rs]); pa = float(np.mean([r["passes"] for r in rs]))
        print(f"  {name:12}: clean {c:.0f}%  contact {ct:.0f}%  off {of:.0f}%  passes {pa:.2f}")
    if "floor+crowd" in qcs:
        q = qcs["floor+crowd"]
        print(f"  q_c on a floor+crowd N=3 run: min {q.min():.2f} mean {q.mean():.2f} max {q.max():.2f}; "
              f"%ticks weaving (q_c<=0.5) {(q <= 0.5).mean() * 100:.0f}%  <- crowd band active fraction")
    np.savez(OUTN / "ltc_n3_crowd_test.npz", rows=np.array(allrows, dtype=object), **{f"qc_{k}": v for k, v in qcs.items()})


if __name__ == "__main__":
    main()
