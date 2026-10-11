"""ADAPTIVE net (loose physical rails + crash-recovery training) vs the BANDED deploy.
Answers the user's two asks:
 (1) "where is the online learning?" -- plot the weights across a race for BOTH nets. The deploy's sit on
     their band clamps (k_v in [0.56,0.62], r_a pinned at 10, q_c on the crowd band); the adaptive net's
     MOVE, because the learner -- not a clamp -- sets them inside the wide physical rails.
 (2) does letting it learn (and recover from crashes instead of dying) still drive cleanly? -- fixed eval
     clean/contact/off per N vs the deploy (100/100/75).
No deploy data overwritten.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/adaptive_eval.py --jobs 4
    python3 tools/adaptive_eval.py --from-saved
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
from mpcc_tuning.mpcc import WEIGHT_NAMES
ADAPT = str(OUT / "nets_adaptive"); OUTA = OUT / "adaptive"; OUTA.mkdir(parents=True, exist_ok=True)
SHOW = ["q_c", "q_v", "r_a", "d_obs", "k_v"]   # the behaviourally meaningful weights (and the banded ones)


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def job(j):
    seed, ego_pace, net_dir, ra_on = j
    m = _mpcc(f"ad_{seed}")
    pol = _load_pol("ltc", seed, net_dir=net_dir, loose=True, ra_floor_on=ra_on)
    if pol is None:
        print(f"  seed {seed}: no adaptive net", flush=True); return []
    rows = []
    for N in (1, 2, 3):
        for phys in range(3):
            want = (seed == 0 and N == 3 and phys == 1)
            met, lg = run_eval(m, ego_pace, pol, N, phys, steps=2600, log=want)
            met.update(seed=seed); rows.append(met)
            if want and lg is not None:
                np.savez(OUTA / "adapt_trace.npz", **{k: np.asarray(v) for k, v in lg.items()})
            print(f"  [adapt s{seed}] N={N} phys={phys}: "
                  f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')}", flush=True)
    # the matching DEPLOY (banded) trace on the same scenario, for the side-by-side weight figure
    if seed == 0:
        dp = _load_pol("ltc", 0)
        _, lgd = run_eval(m, ego_pace, dp, 3, 1, steps=2600, log=True)
        if lgd is not None:
            np.savez(OUTA / "deploy_trace.npz", **{k: np.asarray(v) for k, v in lgd.items()})
    return rows


def fig_weights():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    da = np.load(OUTA / "adapt_trace.npz", allow_pickle=True)
    dd = np.load(OUTA / "deploy_trace.npz", allow_pickle=True) if (OUTA / "deploy_trace.npz").exists() else None
    sa = da["S"] - da["S"][0]; Wa = da["THETA"]             # THETA is logged as the LINEAR weights already
    sd = (dd["S"] - dd["S"][0], dd["THETA"]) if dd is not None else None
    idx = [WEIGHT_NAMES.index(n) for n in SHOW]
    fig, ax = plt.subplots(len(SHOW), 1, figsize=(11, 1.65 * len(SHOW)), sharex=True)
    for row, (n, i) in enumerate(zip(SHOW, idx)):
        if sd is not None:
            ax[row].plot(sd[0], sd[1][:, i], color="0.6", lw=1.3, label="deploy (banded)")
        ax[row].plot(sa, Wa[:, i], color="#1f77b4", lw=1.5, label="adaptive (learned)")
        rng_a = float(np.ptp(Wa[:, i])); rng_d = float(np.ptp(sd[1][:, i])) if sd is not None else 0.0
        ax[row].set_ylabel(n); ax[row].grid(alpha=0.25)
        ax[row].set_title(f"{n}: adaptive moves {rng_a:.3f}   vs   deploy moves {rng_d:.3f}",
                          fontsize=8.5, loc="left")
        if row == 0:
            ax[row].legend(fontsize=8, loc="upper right", ncol=2)
    ax[-1].set_xlabel("lap distance s (m)")
    fig.suptitle("Online learning drives the weights (N=3): adaptive net vs the banded deploy", y=1.005)
    fig.tight_layout()
    fig.savefig(OUTA / "adaptive_weights.png", dpi=130); fig.savefig(OUTA / "adaptive_weights.pdf"); plt.close(fig)
    print(f"wrote {OUTA}/adaptive_weights.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--net-dir", default=ADAPT, help="adaptive net dir (default nets_adaptive)")
    ap.add_argument("--physical", action="store_true", help="net was trained with the r_a corridor-floor band (nets_physical)")
    ap.add_argument("--from-saved", action="store_true"); a = ap.parse_args()
    if a.from_saved:
        fig_weights(); return
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("ad_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    tag = "PHYSICAL-rails" if a.physical else "loose-rails"
    print(f"ego solo pace {ego_pace:.2f} m/s ; ADAPTIVE ({tag}) net fixed eval N=1/2/3", flush=True)
    jobs = [(s, ego_pace, a.net_dir, a.physical) for s in (0, 1)]
    allrows = []
    if a.jobs <= 1:
        for j in jobs:
            allrows += job(j)
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for rr in pool.imap_unordered(job, jobs):
                allrows += rr
    print("\n==== ADAPTIVE net (loose rails + recovery) vs deploy 100/100/75 ====")
    for N in (1, 2, 3):
        rs = [r for r in allrows if r["N"] == N]
        if rs:
            c = 100 * np.mean([r["clean"] for r in rs]); ct = 100 * np.mean([r["contact"] for r in rs])
            of = 100 * np.mean([r["off"] for r in rs]); pa = float(np.mean([r["passes"] for r in rs]))
            print(f"  N={N}: clean {c:.0f}%  contact {ct:.0f}%  off {of:.0f}%  passes {pa:.2f}  (n={len(rs)})")
    np.savez(OUTA / "adapt_eval_rows.npz", rows=np.array(allrows, dtype=object))
    if (OUTA / "adapt_trace.npz").exists():
        fig_weights()


if __name__ == "__main__":
    main()
