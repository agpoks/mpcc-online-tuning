"""Visualise opponent-state PREDICTION under occlusion. Replays a saved sensor run and, from the last-
seen state, predicts the (occluded) opponent with the 'drives-like-us' model: it is the same car, so it
follows the racing line at a grip-limited speed profile, scaled by its observed PACE (EMA) and LINE bias
(EMA). Shows predicted vs ground-truth opponent through the blind windows -- the occluded opponent is
predictable because it stays on the known racing manifold. This is the EXPLICIT form of what a recurrent
policy's memory can encode; the deployed controller drives via the online-tuned weights and does not need
it, but it quantifies how well the hidden opponent CAN be tracked.

Reproducible from the saved .npz (no acados):
    python3 tools/paper_opp_prediction.py --case overtake_blind --seed 0 --tag _ra10conf
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.collections import LineCollection

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from experiments.race_mode import A_LAT_RACE
OUT = ROOT / "results/race_phase2"; TZ = OUT / "tikz"
SEEN, BLIND, TRUE, PRED = "#1f4e8c", "#cfcfcf", "#c23b3b", "#e8820e"
DT = 0.05


def _vref(track, a_lat=A_LAT_RACE, n=600):
    """Grip-limited speed profile v(s) = sqrt(a_lat / curvature), clipped -- the 'drives like us' baseline."""
    L = float(track.length); ss = np.linspace(0, L, n, endpoint=False); eps = L / n; kap = np.empty(n)
    for i, s in enumerate(ss):
        a0 = float(track.tangent_angle(s - eps)); a1 = float(track.tangent_angle(s + eps))
        kap[i] = abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi) / (2 * eps)
    return ss, np.clip(np.sqrt(a_lat / np.maximum(kap, 1e-3)), 0.8, 4.0), L


def predict_opponent(track, OX, OY, DET, tau=0.6):
    """Replay the drives-like-us predictor over the run. Returns predicted (px, py) per tick + the pace."""
    ss, vref, L = _vref(track); ema = np.exp(-DT / tau)
    def vlim(s): return float(np.interp(s % L, ss, vref))
    s_pred = None; pace = 1.0; lat = 0.0; prev = None
    PX = np.empty(len(OX)); PY = np.empty(len(OY))
    for i in range(len(OX)):
        if DET[i]:
            s_opp = float(track.project(float(OX[i]), float(OY[i])))
            lo = float(track.lateral(float(OX[i]), float(OY[i])))
            if prev is not None:
                v = float(np.hypot(OX[i] - prev[0], OY[i] - prev[1]) / DT)
                pace = ema * pace + (1 - ema) * float(np.clip(v / max(vlim(s_opp), 0.3), 0.3, 1.25))
            lat = ema * lat + (1 - ema) * lo; s_pred = s_opp; prev = (OX[i], OY[i])
        elif s_pred is not None:
            s_pred += vlim(s_pred) * pace * DT
        if s_pred is None:
            PX[i], PY[i] = OX[i], OY[i]
        else:
            p = np.array(track.pos(s_pred)).ravel(); psi = float(track.tangent_angle(s_pred))
            PX[i] = p[0] - np.sin(psi) * lat; PY[i] = p[1] + np.cos(psi) * lat
    return PX, PY


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="overtake_blind")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arm", default="ltc")
    ap.add_argument("--tag", default="_ra10conf"); ap.add_argument("--skip-gif", action="store_true")
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    z = np.load(OUT / f"sensor_{a.case}_{a.arm}_{a.seed}{a.tag}.npz")
    EX, EY, OX, OY, DET = z["EX"], z["EY"], z["OX"], z["OY"], z["DET"].astype(bool)
    PX, PY = predict_opponent(track, OX, OY, DET)
    err = np.hypot(PX - OX, PY - OY); t = np.arange(len(EX)) * DT
    blindmean = float(err[~DET].mean()) if (~DET).any() else 0.0
    print(f"{a.case} {a.arm} s{a.seed}: blind {100*(1-DET.mean()):.0f}% | mean predicted-vs-true error "
          f"{err.mean():.3f} m (blind-only {blindmean:.3f} m, max {err.max():.3f} m)")

    fig, (axT, axE) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    axT.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); axT.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
    pts = np.array([EX, EY]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    axT.add_collection(LineCollection(segs, colors=[SEEN if d else BLIND for d in DET[:-1]], linewidths=2.0, zorder=3))
    axT.plot(OX, OY, color=TRUE, lw=1.4, alpha=.8, zorder=4, label="opponent TRUE")
    axT.plot(PX[~DET], PY[~DET], ".", color=PRED, ms=3, zorder=5, label="opponent PREDICTED (while blind)")
    axT.set_aspect("equal"); axT.set_xticks([]); axT.set_yticks([]); axT.legend(fontsize=8, loc="best")
    axT.set_title(f"{a.case.replace('_',' ')} (seed {a.seed}): ego blue=sees/grey=blind;\ntrue opp (red) vs drives-like-us prediction (orange, blind)", fontsize=9.5)
    axE.fill_between(t, 0, err.max() * 1.05, where=~DET, color="0.9", zorder=0, label="blind")
    axE.plot(t, err, color=PRED, lw=1.3, label="|predicted - true| (m)")
    axE.set_xlabel("time (s)"); axE.set_ylabel("prediction error (m)"); axE.grid(alpha=.3); axE.legend(fontsize=8)
    axE.set_title(f"predicted-vs-true error: mean {err.mean():.2f} m, max {err.max():.2f} m", fontsize=9.5)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_opp_prediction_{a.case}_{a.arm}_{a.seed}.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_opp_prediction_{a.case}_{a.arm}_{a.seed}.pdf (+.png)")

    TZ.mkdir(parents=True, exist_ok=True)
    with open(TZ / f"opp_pred_err_{a.case}_{a.arm}_{a.seed}.dat", "w") as fh:
        ds = max(1, len(t) // 1500); fh.write("t err d\n")
        [fh.write(f"{t[i]:.3f} {err[i]:.4f} {int(DET[i])}\n") for i in range(0, len(t), ds)]
    (TZ / f"fig_opp_prediction_{a.case}_{a.arm}_{a.seed}.tex").write_text(
        "% Auto-generated by tools/paper_opp_prediction.py\n\\begin{tikzpicture}\n"
        "\\begin{axis}[xlabel={time (s)}, ylabel={opponent prediction error (m)}, width=9cm, height=5.5cm, grid=both,\n"
        "    title={drives-like-us opponent prediction vs ground truth}]\n"
        f"  \\addplot[orange,thick] table[x=t,y=err] {{tikz/opp_pred_err_{a.case}_{a.arm}_{a.seed}.dat}};\n"
        "\\end{axis}\n\\end{tikzpicture}\n")
    print(f"wrote {TZ}/fig_opp_prediction_{a.case}_{a.arm}_{a.seed}.tex")

    if not a.skip_gif:
        step = max(1, len(EX) // 150); idx = list(range(0, len(EX), step))
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        trail, = ax.plot([], [], color=SEEN, lw=1.4, alpha=.5, zorder=3)
        ego = ax.scatter([], [], s=70, zorder=6); tru = ax.scatter([], [], s=80, marker="s", color=TRUE, zorder=5)
        prd = ax.scatter([], [], s=90, facecolor="none", edgecolor=PRED, lw=2.0, zorder=7)
        ttl = ax.set_title("", fontsize=10)
        ax.scatter([], [], s=80, marker="s", color=TRUE, label="opponent TRUE")
        ax.scatter([], [], s=90, facecolor="none", edgecolor=PRED, lw=2, label="PREDICTED")
        ax.legend(fontsize=8, loc="upper right")
        def upd(k):
            i = idx[k]; trail.set_data(EX[:i + 1], EY[:i + 1])
            ego.set_offsets([[EX[i], EY[i]]]); ego.set_color(SEEN if DET[i] else BLIND)
            tru.set_offsets([[OX[i], OY[i]]]); prd.set_offsets([[PX[i], PY[i]]])
            prd.set_alpha(0.25 if DET[i] else 1.0)   # prediction pops OUT when blind (that's when it's doing work)
            ttl.set_text(f"{a.case.replace('_',' ')}  t={i*DT:4.1f}s  {'SEES opp' if DET[i] else 'BLIND -> predicting'}")
            return trail, ego, tru, prd, ttl
        anim = manim.FuncAnimation(fig, upd, frames=len(idx), interval=60, blit=False)
        p = OUT / f"opp_prediction_{a.case}_{a.arm}_{a.seed}.gif"
        anim.save(str(p), writer=manim.PillowWriter(fps=16)); plt.close(fig); print(f"wrote {p}")


if __name__ == "__main__":
    main()
