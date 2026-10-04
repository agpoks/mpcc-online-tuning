"""Phase-2 step B EVAL -- partial observability. Drives the SENSOR-GATED net (results/race_phase2/)
FROZEN (deployed: no explore, no learn) with the same detect-range gate it trained under, on the two
cases the sensor raises:

  A) OVERTAKE-then-BLIND: a SLOWER car ahead -> ego passes it -> it falls >R behind -> ego goes blind.
     Must stay CLEAN and NOT react to the car it can no longer see (no phantom swerve).
  B) FASTER RE-APPROACH: a FASTER car starts behind, out of range -> closes -> ego should react only
     once it ENTERS range (detected flips on), not before.

Logs ego+opp trajectory, the detected flag and time-since-seen, emitted weights. Prints laps / clean /
passes / %time-detected and the cross-track activity BLIND vs SEEN (a phantom-reaction check: if the ego
only reacts when it sees, blind-window cross-track activity ~ its clean solo line). Figure per case:
track (ego coloured by detected, opp trail) + a gap/detected timeline. Reproducible/seeded.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_sensor_eval.py --seed 0 --detect-range 18
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import RacelineOpponent, ObstacleTracker
from mpcc_tuning.ltc import LTCCell, WeightPolicy
from experiments.race_mode import (race_features, FAIR_PACE, signed_gap, N_RACE_FEATURES, KEEPOUT_R,
                                    CONTACT_R, PACE_KINDS, LR_VEH, A_LAT_RACE, CORRIDOR_KW,
                                    KV_FLOOR, KV_FLOOR_CLASSES, KV_CEIL, KV_CEIL_CLASSES,
                                    RA_FLOOR, RA_FLOOR_CLASSES, measure_pace)

OUT = ROOT / "results/race_phase2"; L = Track.icra_t2_smooth().length
# (name, opponent class, start gap vs ego [+ahead / -behind]). The two sensor cases.
CASES = [("overtake_blind", "slower", 4.0), ("faster_reapproach", "faster", -8.0)]


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -L / 2] += L; dS[dS > L / 2] -= L
    return float(dS[dS > 0].sum() / L)


def _cte(EX, EY, ref):
    """cross-track = distance to the nearest raceline point (ego's lateral activity)."""
    rx, ry = ref; out = np.empty(len(EX))
    for i, (x, y) in enumerate(zip(EX, EY)):
        out[i] = np.sqrt((rx - x) ** 2 + (ry - y) ** 2).min()
    return out


def run_case(seed, name, kind, gap0, detect_range, steps):
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"sensor_{seed}")
    d = np.load(OUT / "nets" / f"race_ltc_{seed}.npz")
    cell = LTCCell(N_RACE_FEATURES, int(d["n_hidden"]), seed=seed); ncls = int(d["D_class"].shape[0])
    pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls, delta_log=float(d["delta_log"]),
                       kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES, kv_ceil=KV_CEIL, kv_ceil_classes=KV_CEIL_CLASSES,
                       ra_floor=RA_FLOOR, ra_floor_classes=RA_FLOOR_CLASSES)
    pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.D_class[...] = d["D_class"]
    pol.reset(); pol.cls = PACE_KINDS.index(kind)
    ego_pace = measure_pace(m, track, th0, 2000); v_opp = FAIR_PACE[kind] * ego_pace
    s0 = (seed % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (seed % 3)
    opp = RacelineOpponent(track, s0=(s0 + gap0) % track.length, pace=v_opp, offset=0.0, radius=KEEPOUT_R, a_lat=2.5)
    tracker = ObstacleTracker(dt=0.05); P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=s0, v0=v0); m.reset(); opp.reset(); tracker.update(opp.pose()[:2]); seen_age = 0.0

    def emit(slip, gap_rate=None, det=True, age=0.0):
        vis = [opp] if det else []
        feat = race_features(track, P.state5(), vis, opp_speed_est=(tracker.speed if det else None),
                             slip=slip, gap_rate=gap_rate, detected=1.0 if det else 0.0, seen_age=age)
        return np.asarray(pol.step(feat), float)

    _b = float(P._x[6]); _r = float(P._x[5]); _v = float(P._x[3])
    _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
    _d0 = (np.hypot(float(P._x[0]) - opp.pose()[0], float(P._x[1]) - opp.pose()[1]) <= detect_range)
    m.set_obstacles([opp.keepout()] if _d0 else [])
    theta = emit((_ar, _b, _r), det=_d0, age=0.0); u = m.value(P.state_dyn(), theta)["u0"]
    LOGK = ["EX", "EY", "S", "V", "GAP", "DET", "AGE", "OX", "OY"]; log = {k: [] for k in LOGK}; logT = []
    passes = 0; seen_pass = False; prev_g = None; contact = False; off = tr = False
    for _ in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); opp.step(0.05, ego=(ex, ey, float(P._x[3])))
        ox, oy, rad = opp.keepout(); s_ego = track.project(ex, ey); g = signed_gap(track, s_ego, opp.s)
        dist = np.hypot(ex - ox, ey - oy)
        if dist < CONTACT_R: contact = True
        _v = float(P._x[3]); _r = float(P._x[5]); _b = float(P._x[6])
        _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
        vo = float(getattr(opp, "speed", v_opp))
        if g < 0 and abs(g) < L / 4 and not seen_pass and _v > vo:
            passes += 1; seen_pass = True
        elif g > 0.5:
            seen_pass = False
        det = (dist <= detect_range); seen_age = 0.0 if det else min(seen_age + 0.05, 5.0)
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
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"sensor_{name}_ltc_{seed}.npz", kind=kind, detect_range=detect_range, **arr)
    laps = _laps(arr["S"]); clean = (not off) and (not contact)
    det = arr["DET"].astype(bool)
    ref = (np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")["ref_x"],
           np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")["ref_y"])
    cte = _cte(arr["EX"], arr["EY"], ref)
    cte_blind = float(np.std(cte[~det])) if (~det).any() else float("nan")
    cte_seen = float(np.std(cte[det])) if det.any() else float("nan")
    print(f"\n=== {name}  (vs {kind}, start gap {gap0:+.0f} m, R={detect_range:.0f} m, seed {seed}) ===", flush=True)
    print(f"laps {laps:.2f}  {'CLEAN' if clean else ('OFF' if off else 'CONTACT')}  passes {passes}  "
          f"detected {100*det.mean():.0f}% of ticks  |  cross-track std blind {cte_blind:.3f} / seen {cte_seen:.3f}", flush=True)
    return dict(name=name, kind=kind, laps=laps, clean=clean, passes=passes, det_frac=float(det.mean()),
                cte_blind=cte_blind, cte_seen=cte_seen, arr=arr)


def figure(results, seed):
    b = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    fig, axs = plt.subplots(len(results), 2, figsize=(12, 4.6 * len(results)),
                            gridspec_kw={"width_ratios": [1.4, 1]})
    axs = np.atleast_2d(axs)
    for row, R in enumerate(results):
        a = R["arr"]; det = a["DET"].astype(bool); axT, axG = axs[row]
        axT.plot(b["left_x"], b["left_y"], color="0.6", lw=.8); axT.plot(b["right_x"], b["right_y"], color="0.6", lw=.8)
        pts = np.array([a["EX"], a["EY"]]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=["#1f4e8c" if d else "#cfcfcf" for d in det[:-1]], linewidths=2.3, zorder=4)
        axT.add_collection(lc)
        axT.plot(a["OX"], a["OY"], color="#c23b3b", lw=1.0, alpha=.5, zorder=3)
        axT.set_aspect("equal"); axT.set_xticks([]); axT.set_yticks([])
        axT.set_title(f"{R['name']} vs {R['kind']}: ego (blue=SEES opp, grey=blind) + opp (red)", fontsize=9.5)
        t = np.arange(len(a["GAP"])) * 0.05
        axG.fill_between(t, -100, 100, where=~det, color="0.9", zorder=0, label="blind (out of range)")
        axG.plot(t, a["GAP"], color="k", lw=1.3, label="signed gap (m)")
        axG.axhline(0, color="0.5", lw=.7)
        axG.set_ylim(np.nanmin(a["GAP"]) - 2, np.nanmax(a["GAP"]) + 2)
        axG.set_xlabel("time (s)"); axG.set_ylabel("gap to opp (m)"); axG.grid(alpha=.3); axG.legend(fontsize=8, loc="best")
        axG.set_title(f"laps {R['laps']:.2f}  {'CLEAN' if R['clean'] else 'NOT CLEAN'}  passes {R['passes']}  "
                      f"det {100*R['det_frac']:.0f}%", fontsize=9.5)
    fig.suptitle(f"Phase-2 sensor gate: deployed gated net under partial observability (seed {seed})", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sensor_cases.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"\nwrote {OUT}/fig_sensor_cases.pdf (+.png)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detect-range", type=float, default=18.0); ap.add_argument("--steps", type=int, default=3600)
    a = ap.parse_args()
    res = [run_case(a.seed, n, k, g, a.detect_range, a.steps) for (n, k, g) in CASES]
    figure(res, a.seed)


if __name__ == "__main__":
    main()
