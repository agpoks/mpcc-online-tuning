"""Chase LTC's N=3 crowding contact (50% in the n=6 eval -- real or noise?). Runs LTC-q_c-floor (floor on)
and MLP-gentler (floor off, as trained) at N=3 across 6 physical fields (phys 0..5) x 2 seeds = 12 runs/arm
for a firmer contact%, and traces an LTC contact: which car it hits, where, at what speed -- is it crowding
(the car needs to deviate through 3 occluded cars) or noise?
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/ltc_n3_chase.py --jobs 4
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, CONTACT_R, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
OUTN = OUT / "ltc_n3"; OUTN.mkdir(parents=True, exist_ok=True)
PHYS6 = list(range(6))


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def job(j):
    arm, seed, ego_pace, floor_on = j
    m = _mpcc(f"n3_{arm}_{seed}"); pol = _load_pol(arm, seed, qc_floor_on=floor_on)
    rows = []; trace = None
    for phys in PHYS6:
        want = (arm == "ltc" and seed == 0)
        met, lg = run_eval(m, ego_pace, pol, 3, phys, steps=2600, log=want)
        met.update(arm=arm, seed=seed, phys=phys); rows.append(met)
        res = "CLEAN" if met["clean"] else ("OFF" if met["off"] else "CONTACT")
        if want and res == "CONTACT" and trace is None:
            trace = (phys, dict(met), {k: np.asarray(v) for k, v in lg.items()})
        print(f"  [{arm} s{seed}] N=3 phys={phys}: {res} laps {met['laps']:.2f} passes {met['passes']}", flush=True)
    return rows, trace


def fig_contact(phys, met, lg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    EX, EY, V, OPP = lg["EX"], lg["EY"], lg["V"], lg["OPP"]
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(12, 5.4))
    ax.set_aspect("equal"); ax.axis("off")
    ax.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.7); ax.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.7)
    pts = np.array([EX, EY]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(1.0, 3.0)); lc.set_array(V[:-1]); lc.set_linewidth(2.4)
    ax.add_collection(lc); fig.colorbar(lc, ax=ax, shrink=0.6, label="ego speed (m/s)")
    # opponent paths + the one hit (closest at the end)
    hit = None; dmin = 1e9
    for k in range(OPP.shape[1]):
        ax.plot(OPP[:, k, 0], OPP[:, k, 1], "-", color="0.55", lw=0.7, alpha=0.6)
        d = np.hypot(EX[-1] - OPP[-1, k, 0], EY[-1] - OPP[-1, k, 1])
        if d < dmin:
            dmin = d; hit = k
    ax.plot(OPP[-1, hit, 0], OPP[-1, hit, 1], "s", color="orange", ms=11, label=f"car hit (#{hit})")
    ax.plot(EX[-1], EY[-1], "x", color="red", ms=14, mew=3, label="ego contact")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"LTC q_c-floor N=3 phys={phys}: CONTACT @{met['laps']:.2f} laps (car gap {dmin:.2f} m)")
    axs.plot(lg["S"] - lg["S"][0], V, color="#1f77b4", lw=1.4)
    axs.set_xlabel("distance s (m)"); axs.set_ylabel("ego speed (m/s)"); axs.set_title("Speed into the contact"); axs.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTN / "ltc_n3_contact.png", dpi=125); plt.close(fig)
    print(f"wrote {OUTN}/ltc_n3_contact.png (hit car #{hit}, final gap {dmin:.2f} m)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("n3_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; N=3 x phys 0..5 x seeds 0,1", flush=True)
    jobs = [("ltc", s, ego_pace, True) for s in (0, 1)] + [("mlp", s, ego_pace, False) for s in (0, 1)]
    allrows = []; ctrace = None
    if a.jobs <= 1:
        for j in jobs:
            rr, tr = job(j); allrows += rr; ctrace = ctrace or tr
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for rr, tr in pool.imap_unordered(job, jobs):
                allrows += rr; ctrace = ctrace or tr
    print("\n==== N=3 firmer stats (n=12/arm) ====")
    for arm in ("ltc", "mlp"):
        rs = [r for r in allrows if r["arm"] == arm]
        c = 100 * np.mean([r["clean"] for r in rs]); ct = 100 * np.mean([r["contact"] for r in rs])
        of = 100 * np.mean([r["off"] for r in rs]); pa = float(np.mean([r["passes"] for r in rs]))
        lab = "q_c-floor" if arm == "ltc" else "gentler(no floor)"
        print(f"  {arm} {lab}: clean {c:.0f}%  contact {ct:.0f}%  off {of:.0f}%  passes {pa:.2f}  (n={len(rs)})")
    np.savez(OUTN / "ltc_n3_rows.npz", rows=np.array(allrows, dtype=object))
    if ctrace is not None:
        phys, met, lg = ctrace
        np.savez(OUTN / "ltc_n3_contact_trace.npz", phys=phys, **lg)
        fig_contact(phys, met, lg)
    else:
        print("no LTC contact logged in seed 0 (all clean there)")


if __name__ == "__main__":
    main()
