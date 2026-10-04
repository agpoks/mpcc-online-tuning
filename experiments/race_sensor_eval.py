"""Phase-2 step B EVAL -- partial observability, LTC vs MLP ablation. Drives each SENSOR-GATED net
(results/race_phase2/nets/race_<arm>_<seed>.npz) FROZEN (deployed: no explore, no learn) with the same
range+FOV gate it trained under, on the two cases the forward sensor raises:

  A) OVERTAKE-then-BLIND: a SLOWER car ahead -> ego passes it -> it drops behind, out of the FOV cone ->
     ego goes blind. Must stay CLEAN and not phantom-swerve at the car it can no longer see.
  B) FASTER RE-APPROACH: a FASTER car starts behind, unseen -> closes -> ego should react only once it
     enters the cone (detected flips on), not before.

Reproducible/seeded. Emits EVERY format:
  data  results/race_phase2/sensor_<case>_<arm>_<seed>.npz   (ego+opp traj, detected, gap, weights)
  csv   results/race_phase2/sensor_eval.csv                  (arm x case: laps/clean/passes/det%/cross-track)
  png   results/race_phase2/fig_sensor_{tracks,timeline}.{pdf,png}
  tikz  results/race_phase2/tikz/fig_sensor_{tracks,timeline}.tex (+ .dat)
  gif   results/race_phase2/sensor_<case>_<arm>_<seed>.gif    (unless --skip-gif)

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_sensor_eval.py --seed 0 --detect-range 18 --fov-deg 120
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as manim

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import RacelineOpponent, ObstacleTracker
from mpcc_tuning.ltc import LTCCell, MLPCell, WeightPolicy
from experiments.race_mode import (race_features, FAIR_PACE, signed_gap, N_RACE_FEATURES, KEEPOUT_R,
                                    CONTACT_R, PACE_KINDS, LR_VEH, A_LAT_RACE, CORRIDOR_KW,
                                    KV_FLOOR, KV_FLOOR_CLASSES, KV_CEIL, KV_CEIL_CLASSES,
                                    RA_FLOOR, RA_FLOOR_CLASSES, measure_pace, _visible, FOV_DEG)

OUT = ROOT / "results/race_phase2"; TZ = OUT / "tikz"; L = Track.icra_t2_smooth().length
ARMS = ["ltc", "mlp"]
# (name, opponent class, start gap vs ego [+ahead / -behind]). The two sensor cases.
CASES = [("overtake_blind", "slower", 4.0), ("faster_reapproach", "faster", -8.0)]
SEEN, BLIND = "#1f4e8c", "#cfcfcf"


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -L / 2] += L; dS[dS > L / 2] -= L
    return float(dS[dS > 0].sum() / L)


def _cte(EX, EY, ref):
    """cross-track = distance to the nearest raceline point (ego's lateral activity)."""
    rx, ry = ref; return np.array([np.sqrt((rx - x) ** 2 + (ry - y) ** 2).min() for x, y in zip(EX, EY)])


def _load_pol(arm, seed):
    f = OUT / "nets" / f"race_{arm}_{seed}.npz"
    if not f.exists():
        return None
    d = np.load(f); Cell = LTCCell if arm == "ltc" else MLPCell
    cell = Cell(N_RACE_FEATURES, int(d["n_hidden"]), seed=seed); ncls = int(d["D_class"].shape[0])
    pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls, delta_log=float(d["delta_log"]),
                       kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES, kv_ceil=KV_CEIL, kv_ceil_classes=KV_CEIL_CLASSES,
                       ra_floor=RA_FLOOR, ra_floor_classes=RA_FLOOR_CLASSES)
    pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.D_class[...] = d["D_class"]
    return pol


def run_case(m, ego_pace, pol, seed, name, kind, gap0, detect_range, steps, fov_deg, arm):
    track = Track.icra_t2_smooth(); half = np.radians(fov_deg) / 2.0
    pol.reset(); pol.cls = PACE_KINDS.index(kind)
    v_opp = FAIR_PACE[kind] * ego_pace
    s0 = (seed % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (seed % 3)
    opp = RacelineOpponent(track, s0=(s0 + gap0) % track.length, pace=v_opp, offset=0.0, radius=KEEPOUT_R, a_lat=2.5)
    tracker = ObstacleTracker(dt=0.05); P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=s0, v0=v0); m.reset(); opp.reset(); tracker.update(opp.pose()[:2]); seen_age = 0.0

    def emit(slip, gap_rate=None, det=True, age=0.0):
        feat = race_features(track, P.state5(), ([opp] if det else []), opp_speed_est=(tracker.speed if det else None),
                             slip=slip, gap_rate=gap_rate, detected=1.0 if det else 0.0, seen_age=age)
        return np.asarray(pol.step(feat), float)

    _b = float(P._x[6]); _r = float(P._x[5]); _v = float(P._x[3])
    _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
    _d0 = _visible(float(P._x[0]), float(P._x[1]), float(P._x[2]), opp.pose()[0], opp.pose()[1], detect_range, half)
    m.set_obstacles([opp.keepout()] if _d0 else [])
    theta = emit((_ar, _b, _r), det=_d0, age=0.0); u = m.value(P.state_dyn(), theta)["u0"]
    LOGK = ["EX", "EY", "S", "V", "GAP", "DET", "AGE", "OX", "OY"]; log = {k: [] for k in LOGK}; logT = []
    passes = 0; seen_pass = False; prev_g = None; contact = False; off = tr = False
    for _ in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); opp.step(0.05, ego=(ex, ey, float(P._x[3])))
        ox, oy, rad = opp.keepout(); s_ego = track.project(ex, ey); g = signed_gap(track, s_ego, opp.s)
        if np.hypot(ex - ox, ey - oy) < CONTACT_R: contact = True
        _v = float(P._x[3]); _r = float(P._x[5]); _b = float(P._x[6])
        _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
        vo = float(getattr(opp, "speed", v_opp))
        if g < 0 and abs(g) < L / 4 and not seen_pass and _v > vo:
            passes += 1; seen_pass = True
        elif g > 0.5:
            seen_pass = False
        det = _visible(ex, ey, float(P._x[2]), ox, oy, detect_range, half); seen_age = 0.0 if det else min(seen_age + 0.05, 5.0)
        for k, val in zip(LOGK, [ex, ey, s_ego, _v, g, 1.0 if det else 0.0, seen_age, ox, oy]):
            log[k].append(val)
        logT.append(np.exp(theta))
        m.set_obstacles([opp.keepout()] if det else [])
        if det: tracker.update(opp.pose()[:2])
        gr = (g - prev_g) / 0.05 if prev_g is not None else 0.0; prev_g = g
        theta = emit((_ar, _b, _r), gap_rate=gr, det=det, age=seen_age); u = m.value(P.state_dyn(), theta)["u0"]
        if off or tr:
            break
    arr = {k: np.asarray(v, float) for k, v in log.items()}; arr["THETA"] = np.asarray(logT, float)
    arr["off"] = np.asarray([off]); arr["contact"] = np.asarray([contact])
    laps = _laps(arr["S"]); clean = (not off) and (not contact)
    det = arr["DET"].astype(bool)
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    cte = _cte(arr["EX"], arr["EY"], (bnd["ref_x"], bnd["ref_y"]))
    cte_blind = float(np.std(cte[~det])) if (~det).any() else float("nan")
    cte_seen = float(np.std(cte[det])) if det.any() else float("nan")
    OUT.mkdir(parents=True, exist_ok=True)   # store summary scalars too -> figures regenerable from data alone
    np.savez(OUT / f"sensor_{name}_{arm}_{seed}.npz", kind=kind, detect_range=detect_range, fov_deg=fov_deg,
             arm=arm, laps=laps, clean=clean, passes=passes, det_frac=float(det.mean()),
             cte_blind=cte_blind, cte_seen=cte_seen, **arr)
    print(f"  [{arm}] {name:18s} vs {kind:7s}: laps {laps:.2f}  "
          f"{'CLEAN' if clean else ('OFF' if off else 'CONTACT')}  passes {passes}  det {100*det.mean():3.0f}%  "
          f"cross-track blind {cte_blind:.3f} / seen {cte_seen:.3f}", flush=True)
    return dict(arm=arm, name=name, kind=kind, laps=laps, clean=clean, passes=passes, off=bool(off),
                contact=bool(contact), det_frac=float(det.mean()), cte_blind=cte_blind, cte_seen=cte_seen, arr=arr)


# ---- outputs ---------------------------------------------------------------------------------------
def write_csv(results, seed):
    p = OUT / "sensor_eval.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "case", "opponent", "laps", "clean", "passes", "detected_pct", "cross_track_blind", "cross_track_seen"])
        for R in results:
            w.writerow([R["arm"], R["name"], R["kind"], f"{R['laps']:.2f}", int(R["clean"]), R["passes"],
                        f"{100*R['det_frac']:.0f}", f"{R['cte_blind']:.4f}", f"{R['cte_seen']:.4f}"])
    print(f"wrote {p}")


def fig_tracks(results, seed):
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    fig, axs = plt.subplots(len(CASES), len(ARMS), figsize=(5.2 * len(ARMS), 4.6 * len(CASES)), squeeze=False)
    for R in results:
        ri = [c[0] for c in CASES].index(R["name"]); ci = ARMS.index(R["arm"]); ax = axs[ri][ci]
        a = R["arr"]; det = a["DET"].astype(bool)
        ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
        pts = np.array([a["EX"], a["EY"]]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=[SEEN if d else BLIND for d in det[:-1]], linewidths=2.3, zorder=4))
        ax.plot(a["OX"], a["OY"], color="#c23b3b", lw=1.0, alpha=.5, zorder=3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{R['arm'].upper()} — {R['name']} vs {R['kind']}\n"
                     f"{'CLEAN' if R['clean'] else 'OFF' if R['off'] else 'CONTACT'}, passes {R['passes']}, "
                     f"blind {100*(1-R['det_frac']):.0f}%", fontsize=9.5)
    fig.suptitle(f"Phase-2 sensor gate: ego path blue=SEES / grey=BLIND, opp red (seed {seed})", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sensor_tracks.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_sensor_tracks.pdf (+.png)")


def fig_timeline(results, seed):
    fig, axs = plt.subplots(len(CASES), len(ARMS), figsize=(5.6 * len(ARMS), 3.6 * len(CASES)), squeeze=False)
    for R in results:
        ri = [c[0] for c in CASES].index(R["name"]); ci = ARMS.index(R["arm"]); ax = axs[ri][ci]
        a = R["arr"]; det = a["DET"].astype(bool); t = np.arange(len(a["GAP"])) * 0.05
        ax.fill_between(t, -100, 100, where=~det, color="0.9", zorder=0, label="blind")
        ax.plot(t, a["GAP"], color="k", lw=1.3, label="signed gap (m)")
        ax.axhline(0, color="0.5", lw=.7); ax.set_ylim(np.nanmin(a["GAP"]) - 2, np.nanmax(a["GAP"]) + 2)
        ax.set_xlabel("time (s)"); ax.set_ylabel("gap (m)"); ax.grid(alpha=.3)
        if ri == 0 and ci == 0: ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{R['arm'].upper()} — {R['name']}  (det {100*R['det_frac']:.0f}%)", fontsize=9.5)
    fig.suptitle(f"Gap to opponent + blind windows (grey) (seed {seed})", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sensor_timeline.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT}/fig_sensor_timeline.pdf (+.png)")


def make_gif(R, seed):
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    a = R["arr"]; det = a["DET"].astype(bool); EX, EY, OX, OY = a["EX"], a["EY"], a["OX"], a["OY"]
    step = max(1, len(EX) // 150); idx = list(range(0, len(EX), step))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); ax.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    trail, = ax.plot([], [], color=SEEN, lw=1.6, alpha=.6, zorder=3)
    ego = ax.scatter([], [], s=70, zorder=6); opp = ax.scatter([], [], s=70, marker="s", color="#c23b3b", zorder=5)
    ttl = ax.set_title("", fontsize=10)
    def upd(k):
        i = idx[k]; trail.set_data(EX[:i + 1], EY[:i + 1])
        ego.set_offsets([[EX[i], EY[i]]]); ego.set_color(SEEN if det[i] else BLIND)
        opp.set_offsets([[OX[i], OY[i]]])
        ttl.set_text(f"{R['arm'].upper()} {R['name']}  t={i*0.05:4.1f}s  {'SEES opp' if det[i] else 'BLIND'}")
        return trail, ego, opp, ttl
    anim = manim.FuncAnimation(fig, upd, frames=len(idx), interval=60, blit=False)
    p = OUT / f"sensor_{R['name']}_{R['arm']}_{seed}.gif"
    anim.save(str(p), writer=manim.PillowWriter(fps=16)); plt.close(fig); print(f"wrote {p}")


def emit_tikz(results, seed):
    TZ.mkdir(parents=True, exist_ok=True)
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    for s, xk, yk in [("left", "left_x", "left_y"), ("right", "right_x", "right_y")]:
        with open(TZ / f"sensor_{s}.dat", "w") as fh:
            fh.write("x y\n"); [fh.write(f"{x:.4f} {y:.4f}\n") for x, y in zip(b[xk], b[yk])]
    tcells, gcells = [], []
    for R in results:
        a = R["arr"]; det = a["DET"].astype(bool); tag = f"{R['name']}_{R['arm']}"
        nm = R["name"].replace("_", " ")   # LaTeX titles: no bare underscores
        ds = max(1, len(a["EX"]) // 1200)
        with open(TZ / f"sensor_{tag}.dat", "w") as fh:
            fh.write("x y d ox oy\n")
            for i in range(0, len(a["EX"]), ds):
                fh.write(f"{a['EX'][i]:.4f} {a['EY'][i]:.4f} {int(det[i])} {a['OX'][i]:.4f} {a['OY'][i]:.4f}\n")
        t = np.arange(len(a["GAP"])) * 0.05
        with open(TZ / f"sensor_gap_{tag}.dat", "w") as fh:
            fh.write("t g d\n")
            for i in range(0, len(t), ds):
                fh.write(f"{t[i]:.3f} {a['GAP'][i]:.4f} {int(det[i])}\n")
        tcells.append(rf"""\nextgroupplot[title={{{R['arm'].upper()} {nm}}}]
  \addplot[gray,thin] table[x=x,y=y]{{tikz/sensor_left.dat}};
  \addplot[gray,thin] table[x=x,y=y]{{tikz/sensor_right.dat}};
  \addplot[scatter,only marks,scatter src=explicit,mark size=0.6pt,
      colormap={{bw}}{{rgb(0cm)=(0.81,0.81,0.81); rgb(1cm)=(0.12,0.31,0.55)}}] table[x=x,y=y,meta=d]{{tikz/sensor_{tag}.dat}};
  \addplot[red,opacity=0.5,thin] table[x=ox,y=oy]{{tikz/sensor_{tag}.dat}};""")
        gcells.append(rf"""\nextgroupplot[title={{{R['arm'].upper()} {nm}}}]
  \addplot[black] table[x=t,y=g]{{tikz/sensor_gap_{tag}.dat}};""")
    rows, cols = len(CASES), len(ARMS)
    tracks = (r"% Auto-generated by experiments/race_sensor_eval.py.  requires \usepgfplotslibrary{groupplots}" "\n"
              r"\begin{tikzpicture}\begin{groupplot}[group style={group size=" + f"{cols} by {rows}"
              + r", horizontal sep=0.4cm, vertical sep=1cm}, width=7cm, axis equal image, hide axis]" "\n"
              + "\n".join(tcells) + "\n" r"\end{groupplot}\end{tikzpicture}" "\n")
    (TZ / "fig_sensor_tracks.tex").write_text(tracks)
    time = (r"% Auto-generated by experiments/race_sensor_eval.py.  requires \usepgfplotslibrary{groupplots}" "\n"
            r"\begin{tikzpicture}\begin{groupplot}[group style={group size=" + f"{cols} by {rows}"
            + r", horizontal sep=1.4cm, vertical sep=1.4cm}, width=7cm, height=4.4cm, xlabel={time (s)}, ylabel={gap (m)}, grid=both]" "\n"
            + "\n".join(gcells) + "\n" r"\end{groupplot}\end{tikzpicture}" "\n")
    (TZ / "fig_sensor_timeline.tex").write_text(time)
    print(f"wrote {TZ}/fig_sensor_tracks.tex + fig_sensor_timeline.tex")


def load_saved(seed, arms):
    """Rebuild the result dicts from the saved .npz -- regenerate every figure/csv/gif WITHOUT re-driving
    (no acados). Faithful: all scalars were stored at run time."""
    out = []
    for arm in arms:
        for (n, k, g) in CASES:
            f = OUT / f"sensor_{n}_{arm}_{seed}.npz"
            if not f.exists():
                continue
            z = np.load(f); arr = {kk: z[kk] for kk in z.files}
            det = arr["DET"].astype(bool)
            scal = lambda key, fn: float(z[key]) if key in z.files else fn()  # recompute if an older npz lacks it
            cte = _cte(arr["EX"], arr["EY"], (lambda b: (b["ref_x"], b["ref_y"]))(np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")))
            out.append(dict(arm=arm, name=n, kind=k,
                            laps=scal("laps", lambda: _laps(arr["S"])),
                            off=bool(arr["off"][0]), contact=bool(arr["contact"][0]),
                            clean=(not bool(arr["off"][0])) and (not bool(arr["contact"][0])),
                            passes=int(z["passes"]) if "passes" in z.files else int(np.sum((arr["GAP"][:-1] > 0) & (arr["GAP"][1:] < 0) & (np.abs(arr["GAP"][1:]) < L / 4))),
                            det_frac=scal("det_frac", lambda: float(det.mean())),
                            cte_blind=scal("cte_blind", lambda: float(np.std(cte[~det])) if (~det).any() else float("nan")),
                            cte_seen=scal("cte_seen", lambda: float(np.std(cte[det])) if det.any() else float("nan")),
                            arr=arr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--detect-range", type=float, default=18.0)
    ap.add_argument("--steps", type=int, default=3600); ap.add_argument("--fov-deg", type=float, default=FOV_DEG)
    ap.add_argument("--arms", nargs="*", default=ARMS); ap.add_argument("--skip-gif", action="store_true")
    ap.add_argument("--from-saved", action="store_true", help="regenerate csv/figs/tikz/gif from the saved .npz (no re-drive, no acados)")
    a = ap.parse_args()
    if a.from_saved:
        results = load_saved(a.seed, a.arms)
        if not results:
            print("no saved sensor_*.npz found -- run without --from-saved first"); return
        print(f"regenerating outputs from {len(results)} saved runs (no re-drive)", flush=True)
    else:
        track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
        m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                       max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"sensor_{a.seed}")
        ego_pace = measure_pace(m, track, th0, 2000)   # measured ONCE, shared across arms+cases
        print(f"ego solo pace {ego_pace:.2f} m/s ; R={a.detect_range:.0f} m, FOV={a.fov_deg:.0f} deg", flush=True)
        results = []
        for arm in a.arms:
            pol = _load_pol(arm, a.seed)
            if pol is None:
                print(f"  [{arm}] no net at {OUT}/nets/race_{arm}_{a.seed}.npz -- skipped", flush=True); continue
            for (n, k, g) in CASES:
                results.append(run_case(m, ego_pace, pol, a.seed, n, k, g, a.detect_range, a.steps, a.fov_deg, arm))
        if not results:
            print("no nets found -- nothing to eval"); return
    write_csv(results, a.seed); fig_tracks(results, a.seed); fig_timeline(results, a.seed); emit_tikz(results, a.seed)
    if not a.skip_gif:
        for R in results:
            make_gif(R, a.seed)
    print("\n=== sensor eval done: data(.npz) + csv + png/pdf + tikz" + ("" if a.skip_gif else " + gif") + " in results/race_phase2/ ===")


if __name__ == "__main__":
    main()
