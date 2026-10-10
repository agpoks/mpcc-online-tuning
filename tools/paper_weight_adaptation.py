"""Online weight adaptation during a race: how the MPCC cost weights (q_c line-shape, k_v grip, r_a accel,
q_v progress) change tick-by-tick as the LTC net responds to the opponents + the track. Generates one clean
N=3 race of the deploy LTC net (q_c-floor + crowd band), logs the 9 weights + trajectory + opponents, then:
 - a static PLOT: weights vs lap-distance, with opponent-seen bands, pass markers, and the crowd-weave region;
 - a GIF: the car + 3 opponents driving the track (left) synced to the live weight traces (right).
Shows the crowd-aware q_c DROP when the field is remembered. Reproducible: --from-saved re-plots the log.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/paper_weight_adaptation.py           # generate + plot
    python3 tools/paper_weight_adaptation.py --from-saved                                      # re-plot only
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import A_LAT_RACE, CORRIDOR_KW, MAX_OPP, measure_pace
from experiments.race_multi_eval import _load_pol, run_eval, OUT
OUTW = OUT / "weight_adaptation"; OUTW.mkdir(parents=True, exist_ok=True)
LOG = OUTW / "weight_race.npz"
WLAB = [("q_c", 0, "#1f77b4", "line-keeping"), ("k_v", 7, "#2ca02c", "grip/speed"),
        ("r_a", 4, "#d62728", "accel damp"), ("q_v", 2, "#9467bd", "progress")]


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def generate():
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = _mpcc("wadapt"); ego_pace = measure_pace(m, track, th0, 1500)
    pol = _load_pol("ltc", 0)                       # deploy LTC: q_c floor + crowd band
    best = None
    for phys in (1, 0, 2, 3):                        # prefer a CLEAN, information-rich N=3 race
        met, lg = run_eval(m, ego_pace, pol, 3, phys, steps=3000, log=True)
        res = "CLEAN" if met["clean"] else ("OFF" if met["off"] else "CONTACT")
        print(f"  N=3 phys={phys}: {res} laps {met['laps']:.2f} passes {met['passes']} steps {len(lg['EX'])}", flush=True)
        score = (met["clean"], met["passes"], len(lg["EX"]))
        if best is None or score > best[0]:
            best = (score, phys, dict(met), {k: np.asarray(v) for k, v in lg.items()})
        if met["clean"] and met["passes"] >= 3:
            break
    _, phys, met, lg = best
    THw = np.exp(lg["THETA"]) if lg["THETA"].max() < 5 else lg["THETA"]   # store ACTUAL weights (exp if log)
    np.savez(LOG, phys=phys, passes=met["passes"], clean=met["clean"], W=THw,
             **{k: v for k, v in lg.items() if k != "THETA"})
    print(f"saved the N=3 phys={phys} race log -> {LOG}")


def _load():
    d = np.load(LOG, allow_pickle=True)
    return {k: d[k] for k in d.files}


def fig_plot(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, V, W, OPP = d["S"], d["V"], d["W"], d["OPP"]
    s = S - S[0]; T = len(s); N = OPP.shape[1] if OPP.ndim == 3 and OPP.shape[0] else 0
    seen_any = np.array([bool(OPP[t, :, 2].any()) if N else False for t in range(T)])
    crowd_seen = np.array([int(np.sum(OPP[:t + 1, :, 2].any(axis=0))) if N else 0 for t in range(T)])
    weave = W[:, 0] <= 0.5
    # small-multiples: each weight its own y-scale (q_c varies; k_v/r_a are band-held; q_v small)
    fig, axes = plt.subplots(6, 1, figsize=(11, 10), sharex=True,
                             height_ratios=[1.3, 1, 1, 1, 0.9, 0.7])
    for ax, (name, i, c, desc) in zip(axes[:4], WLAB):
        ax.plot(s, W[:, i], color=c, lw=1.7)
        ax.fill_between(s, *ax.get_ylim(), where=weave, color="#1f77b4", alpha=0.07, step="mid")
        ax.set_ylabel(f"{name}\n{desc}", fontsize=9); ax.grid(alpha=0.25); ax.margins(y=0.25)
    axes[0].set_title("Online weight adaptation during an N=3 race  (LTC: q_c-floor + crowd band). "
                      "Blue = crowd WEAVE (q_c capped once the field is remembered)")
    axes[0].text(0.01, 0.9, "q_c DROPS to weave when the 3-car crowd is remembered; k_v & r_a stay in their safety bands",
                 transform=axes[0].transAxes, fontsize=8, color="0.3", va="top")
    axes[4].plot(s, V, color="0.3", lw=1.4); axes[4].set_ylabel("speed\nm/s", fontsize=9); axes[4].grid(alpha=0.25)
    axes[5].step(s, crowd_seen, color="#d62728", lw=1.6, where="mid"); axes[5].set_ylabel("opp.\nseen", fontsize=9)
    axes[5].fill_between(s, 0, 1, where=~seen_any, color="0.6", alpha=0.18, step="mid",
                        transform=axes[5].get_xaxis_transform())
    axes[5].set_yticks([0, 1, 2, 3]); axes[5].set_xlabel("lap distance s (m)"); axes[5].grid(alpha=0.25)
    axes[5].text(0.01, 0.78, "grey = no opponent in view (occluded) — memory keeps the crowd count",
                 transform=axes[5].transAxes, fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(OUTW / "weight_adaptation.png", dpi=130); fig.savefig(OUTW / "weight_adaptation.pdf"); plt.close(fig)
    print(f"wrote {OUTW}/weight_adaptation.{{png,pdf}}")


def fig_gif(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    EX, EY, S, V, W, OPP = d["EX"], d["EY"], d["S"], d["V"], d["W"], d["OPP"]
    s = S - S[0]; T = len(EX); N = OPP.shape[1] if OPP.ndim == 3 and OPP.shape[0] else 0
    stride = max(1, T // 220)
    fig = plt.figure(figsize=(12.5, 6.2))
    axT = fig.add_subplot(1, 2, 1); axT.set_aspect("equal"); axT.axis("off")
    axT.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.8); axT.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.8)
    trail = LineCollection([], cmap="viridis", norm=plt.Normalize(1.0, 3.0), lw=2.4); axT.add_collection(trail)
    (ego,) = axT.plot([], [], "o", color="#1f77b4", ms=10, zorder=5)
    oseen = [axT.plot([], [], "s", color="#d62728", ms=8)[0] for _ in range(N)]
    oblind = [axT.plot([], [], "s", mfc="none", mec="#d62728", ms=8)[0] for _ in range(N)]
    ttlT = axT.set_title("")
    axW = fig.add_subplot(1, 2, 2)
    axW.plot(s, W[:, 0], color="#1f77b4", lw=1.9)                # q_c trace -- the adapting weight
    axW.fill_between(s, 0, W[:, 0].max() * 1.12, where=(W[:, 0] <= 0.5), color="#1f77b4", alpha=0.10, step="mid")
    axW.axhline(0.5, color="0.6", lw=0.9, ls=":")
    axW.text(s[-1], 0.46, "crowd weave band", fontsize=7, ha="right", va="top", color="0.45")
    qc_cur, = axW.plot([], [], "o", color="#1f77b4", ms=10, zorder=5)
    cursor = axW.axvline(0, color="0.5", lw=1.0, ls="--")
    axW.set_ylim(0, W[:, 0].max() * 1.12); axW.set_xlabel("lap distance s (m)")
    axW.set_ylabel("q_c  (line-keeping weight)")
    axW.set_title("q_c adapts to the remembered crowd  (k_v/r_a/q_v held in their bands)")
    axW.grid(alpha=0.25)
    txt = axW.text(0.02, 0.03, "", transform=axW.transAxes, fontsize=9.5, family="monospace",
                   va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    def frame(fk):
        k = min(fk * stride, T - 1)
        pts = np.array([EX[:k + 1], EY[:k + 1]]).T.reshape(-1, 1, 2)
        if len(pts) > 1:
            trail.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1)); trail.set_array(V[:k])
        ego.set_data([EX[k]], [EY[k]])
        for j in range(N):
            ox, oy, det = OPP[k, j]
            (oseen[j] if det > 0.5 else oblind[j]).set_data([ox], [oy])
            (oblind[j] if det > 0.5 else oseen[j]).set_data([], [])
        nseen = int(np.sum(OPP[:k + 1, :, 2].any(axis=0))) if N else 0
        weaving = W[k, 0] <= 0.5
        ttlT.set_text(f"t={k*0.05:.1f}s   v={V[k]:.1f} m/s   crowd remembered {nseen}/3  (filled=seen, hollow=blind)")
        qc_cur.set_data([s[k]], [W[k, 0]])
        cursor.set_xdata([s[k], s[k]])
        txt.set_text(f"q_c = {W[k,0]:.2f}  [{'WEAVE' if weaving else 'hold line'}]\n"
                     f"k_v = {W[k,7]:.2f}   r_a = {W[k,4]:.1f}   q_v = {W[k,2]:.2f}")
        return []

    anim = animation.FuncAnimation(fig, frame, frames=T // stride + 1, interval=70)
    fig.tight_layout(); anim.save(OUTW / "weight_adaptation.gif", writer=animation.PillowWriter(fps=15)); plt.close(fig)
    print(f"wrote {OUTW}/weight_adaptation.gif")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from-saved", action="store_true")
    a = ap.parse_args()
    if not a.from_saved:
        generate()
    d = _load(); fig_plot(d); fig_gif(d)


if __name__ == "__main__":
    main()
