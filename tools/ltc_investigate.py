"""Investigate what changed for LTC: pre-fix (no keep-out) vs gentler keep-out. Two parts:
 (1) net-vs-constraint ABLATION -- LTC {pre-fix net, gentler net} x {no keep-out, gentle keep-out}, same
     scenarios. Isolates whether LTC changed because of the RETRAINED NET or the KEEP-OUT CONSTRAINT.
 (2) a detailed TRACE of one scenario for pre-fix-LTC vs gentler-LTC: speed-coloured trajectory, speed &
     states (slip, yaw-rate) vs distance, learned weights (k_v/q_c/r_a) vs distance, + a side-by-side gif.

The pre-fix LTC/MLP nets are restored from git (commit 5a57d07) into the scratchpad prefix_nets/ dir.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/ltc_investigate.py --jobs 4
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, NSET, PHYS, _agg, load_rows_path, OUT

SCR = Path("/tmp/claude-1000/-home-poxx-github-mpcc-online-tuning/c2132c3b-15f2-4913-a334-92d54345de72/scratchpad")
PREFIX = SCR / "prefix_nets"
OUTI = OUT / "ltc_investigate"; OUTI.mkdir(parents=True, exist_ok=True)
WLAB = [("k_v", 7), ("q_c", 0), ("r_a", 4), ("q_v", 2)]   # weight index map (theta: q_c0 q_l1 q_v2 r_d3 r_a4 r_dv5 d_obs6 k_v7 d_bound8)


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


# ----------------------------------------------------------------- (1) ablation
def abl_job(job):
    tag, net_dir, keepout, seed, ego_pace = job
    m = _mpcc(f"abl_{tag}_{seed}"); pol = _load_pol("ltc", seed, net_dir)
    rows = []
    for N in NSET:
        for phys in PHYS:
            met, _ = run_eval(m, ego_pace, pol, N, phys, steps=2600, keepout=keepout)
            met.update(tag=tag, seed=seed); rows.append(met)
            print(f"  [{tag} s{seed}] N={N} phys={phys}: "
                  f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')}", flush=True)
    return rows


def _agg_tag(rows):
    out = {}
    for N in NSET:
        rs = [r for r in rows if r["N"] == N]
        if rs:
            out[N] = dict(clean=100 * np.mean([r["clean"] for r in rs]), contact=100 * np.mean([r["contact"] for r in rs]),
                          off=100 * np.mean([r["off"] for r in rs]), n=len(rs))
    return out


def ablation(ego_pace, jobs_n):
    # A (gentler+gentle) and C (prefix+none) are already computed -> load LTC rows from saved snapshots.
    A = [r for r in load_rows_path(OUT / "multi_eval_rows.npz") if r["arm"] == "ltc"]
    C = [r for r in load_rows_path(OUT / "multi_eval_rows_prefix.npz") if r["arm"] == "ltc"]
    jobs = [("B_gentler_noKO", None, "none", s, ego_pace) for s in (0, 1)] + \
           [("D_prefix_gentleKO", str(PREFIX), "gentle", s, ego_pace) for s in (0, 1)]
    results = []
    if jobs_n <= 1:
        for j in jobs:
            results += abl_job(j)
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(jobs_n, len(jobs))) as pool:
            for rr in pool.imap_unordered(abl_job, jobs):
                results += rr
    B_ = [r for r in results if r["tag"] == "B_gentler_noKO"]
    D_ = [r for r in results if r["tag"] == "D_prefix_gentleKO"]
    tab = {"prefix-net + NO keep-out  (= pre-fix)": _agg_tag(C),
           "prefix-net + gentle keep-out       ": _agg_tag(D_),
           "gentler-net + NO keep-out          ": _agg_tag(B_),
           "gentler-net + gentle keep-out (now) ": _agg_tag(A)}
    print("\n==== LTC net-vs-constraint ablation (clean% / contact% / off%) ====")
    print(f"{'config':40}" + "".join(f"  N={N}" + " " * 10 for N in NSET))
    for name, ag in tab.items():
        print(f"{name:40}", end="")
        for N in NSET:
            a = ag.get(N, {})
            print(f"  {a.get('clean', 0):3.0f}/{a.get('contact', 0):3.0f}/{a.get('off', 0):3.0f}    ", end="")
        print()
    np.savez(OUTI / "ablation.npz", B=np.array(B_, dtype=object), D=np.array(D_, dtype=object),
             A=np.array(A, dtype=object), C=np.array(C, dtype=object))
    return tab


# ----------------------------------------------------------------- (2) detailed trace
def trace_one(ego_pace, net_dir, keepout, N, phys, name):
    m = _mpcc(f"tr_{name}"); pol = _load_pol("ltc", 0, net_dir)
    met, lg = run_eval(m, ego_pace, pol, N, phys, steps=2600, keepout=keepout, log=True)
    np.savez(OUTI / f"trace_{name}.npz", **lg, **{k: np.asarray([v]) for k, v in met.items() if k != "kinds"})
    print(f"  trace {name}: {'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')} "
          f"laps {met['laps']:.2f} passes {met['passes']}", flush=True)
    return met


def fig_trace(N, phys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    pre = np.load(OUTI / "trace_prefix.npz", allow_pickle=True); gen = np.load(OUTI / "trace_gentler.npz", allow_pickle=True)
    traces = [("pre-fix LTC", pre, "#1f77b4"), ("gentler LTC", gen, "#d62728")]
    fig = plt.figure(figsize=(14, 8))
    # row 1: two speed-coloured trajectories
    for j, (nm, d, _) in enumerate(traces):
        ax = fig.add_subplot(2, 3, j + 1); ax.set_aspect("equal"); ax.axis("off")
        ax.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.7); ax.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.7)
        EX, EY, V = d["EX"], d["EY"], d["V"]
        pts = np.array([EX, EY]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(1.0, 3.2)); lc.set_array(V[:-1]); lc.set_linewidth(2.3)
        ax.add_collection(lc)
        OPP = d["OPP"]  # (T,N,3)
        if OPP.ndim == 3 and OPP.shape[0]:
            for k in range(OPP.shape[1]):
                ax.plot(OPP[:, k, 0], OPP[:, k, 1], color="0.5", lw=0.6, alpha=0.5)
        laps = float(d["laps"][0]); res = "CLEAN" if bool(d["clean"][0]) else ("OFF" if bool(d["off"][0]) else "CONTACT")
        if res == "CONTACT":
            ax.plot(EX[-1], EY[-1], "x", color="red", ms=13, mew=3)
        ax.set_title(f"{nm}: {res}, {laps:.2f} laps", color="red" if res != "CLEAN" else "black")
    fig.colorbar(lc, ax=fig.axes[:2], shrink=0.6, label="speed (m/s)", location="right")
    # row 1 col 3: speed vs distance
    axs = fig.add_subplot(2, 3, 3)
    for nm, d, c in traces:
        axs.plot(d["S"] - d["S"][0], d["V"], color=c, lw=1.3, label=nm)
    axs.set_xlabel("distance s (m)"); axs.set_ylabel("speed (m/s)"); axs.set_title("Speed"); axs.legend(fontsize=8); axs.grid(alpha=0.3)
    # row 2: slip, yaw-rate, and weights
    axsl = fig.add_subplot(2, 3, 4)
    for nm, d, c in traces:
        axsl.plot(d["S"] - d["S"][0], np.degrees(d["SLIP"]), color=c, lw=1.1)
    axsl.set_xlabel("s (m)"); axsl.set_ylabel("slip angle (deg)"); axsl.set_title("Slip"); axsl.grid(alpha=0.3)
    axyr = fig.add_subplot(2, 3, 5)
    for nm, d, c in traces:
        axyr.plot(d["S"] - d["S"][0], d["YAWRATE"], color=c, lw=1.1)
    axyr.set_xlabel("s (m)"); axyr.set_ylabel("yaw rate (rad/s)"); axyr.set_title("Yaw rate"); axyr.grid(alpha=0.3)
    axw = fig.add_subplot(2, 3, 6)
    for nm, d, c in traces:
        TH = d["THETA"]
        if TH.ndim == 2 and TH.shape[0]:
            s = d["S"] - d["S"][0]
            for (wl, wi), ls in zip(WLAB[:3], ["-", "--", ":"]):
                axw.plot(s, TH[:, wi], color=c, lw=1.1, ls=ls, label=f"{nm[:3]} {wl}" if wl == "k_v" else None)
    axw.set_xlabel("s (m)"); axw.set_ylabel("weight"); axw.set_title("Learned weights (solid k_v, -- q_c, : r_a)")
    axw.legend(fontsize=7); axw.grid(alpha=0.3)
    fig.suptitle(f"pre-fix LTC vs gentler LTC -- N={N} opponents, scenario phys={phys}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTI / "ltc_trace.png", dpi=125); fig.savefig(OUTI / "ltc_trace.pdf"); plt.close(fig)
    print(f"wrote {OUTI}/ltc_trace.png")


def gif_trace(N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    pre = np.load(OUTI / "trace_prefix.npz", allow_pickle=True); gen = np.load(OUTI / "trace_gentler.npz", allow_pickle=True)
    traces = [("pre-fix LTC", pre, "#1f77b4"), ("gentler LTC", gen, "#d62728")]
    T = max(len(pre["EX"]), len(gen["EX"])); stride = max(1, T // 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    arts = []
    for ax, (nm, d, c) in zip(axes, traces):
        ax.set_aspect("equal"); ax.axis("off")
        ax.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.7); ax.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.7)
        ego, = ax.plot([], [], "o", color=c, ms=9); trail, = ax.plot([], [], "-", color=c, lw=1.2, alpha=0.6)
        nopp = d["OPP"].shape[1] if d["OPP"].ndim == 3 and d["OPP"].shape[0] else 0
        oseen = [ax.plot([], [], "s", color="0.2", ms=7)[0] for _ in range(nopp)]
        oblind = [ax.plot([], [], "s", mfc="none", mec="0.2", ms=7)[0] for _ in range(nopp)]
        arts.append((d, ego, trail, oseen, oblind, nopp, ax, nm))
    ttl = fig.suptitle("")

    def frame(fk):
        for (d, ego, trail, oseen, oblind, nopp, ax, nm) in arts:
            i = min(fk * stride, len(d["EX"]) - 1)
            ego.set_data([d["EX"][i]], [d["EY"][i]]); trail.set_data(d["EX"][:i + 1], d["EY"][:i + 1])
            for k in range(nopp):
                ox, oy, det = d["OPP"][i, k]
                (oseen[k] if det > 0.5 else oblind[k]).set_data([ox], [oy])
                (oblind[k] if det > 0.5 else oseen[k]).set_data([], [])
            ax.set_title(f"{nm}  v={d['V'][i]:.1f} m/s", fontsize=10)
        ttl.set_text(f"N={N} cars  (filled=seen, hollow=blind)  t={fk*stride*0.05:.1f}s")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=T // stride + 1, interval=70)
    anim.save(OUTI / "ltc_trace.gif", writer=animation.PillowWriter(fps=14)); plt.close(fig)
    print(f"wrote {OUTI}/ltc_trace.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--trace-N", type=int, default=3); ap.add_argument("--trace-phys", type=int, default=0)
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    mp = _mpcc("inv_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s", flush=True)
    ablation(ego_pace, a.jobs)
    print("\n-- detailed trace --", flush=True)
    trace_one(ego_pace, str(PREFIX), "none", a.trace_N, a.trace_phys, "prefix")
    trace_one(ego_pace, None, "gentle", a.trace_N, a.trace_phys, "gentler")
    fig_trace(a.trace_N, a.trace_phys); gif_trace(a.trace_N)


if __name__ == "__main__":
    main()
