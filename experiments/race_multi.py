"""Phase-2 step 2 -- MULTI-OPPONENT race eval. The band net races N opponents of MIXED relative pace
in one race: the MPCC avoids ALL N (max_obstacles=N, the step-1 _p fix), and the policy reacts to the
NEAREST-AHEAD opponent (features + class + tracker keyed on it). This is the BASELINE: a net trained on
one opponent at a time, driven multi-opponent with a nearest-ahead heuristic -- it establishes what the
single-opponent policy already does before we add nearest-behind / continuous-pace features + retraining.

Reproducible/seeded; logs ego + all opponents to results/race/paper/states_multi_ltc_<seed>.npz.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_multi.py --seed 0 --steps 9500
"""
import sys, argparse
from pathlib import Path
import numpy as np
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

OUT = ROOT / "results/race/paper"; L = Track.icra_t2_smooth().length
# (class, pace x ego, start gap ahead of ego). Mixed: slower, equal, faster -- all in ONE race.
MIX = [("slower", 0.80, 4.0), ("equal", 1.00, 9.0), ("faster", 1.05, 14.0)]


def _laps(S):
    S = np.asarray(S, float); dS = np.diff(S); dS[dS < -L / 2] += L; dS[dS > L / 2] -= L
    return float(dS[dS > 0].sum() / L)


def race_multi(seed=0, steps=9500, n_opp=3):
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=n_opp, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"multi_{seed}")
    d = np.load(ROOT / "results/race/nets" / f"race_ltc_{seed}.npz")
    cell = LTCCell(N_RACE_FEATURES, int(d["n_hidden"]), seed=seed); ncls = int(d["D_class"].shape[0])
    pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls, delta_log=float(d["delta_log"]),
                       kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES, kv_ceil=KV_CEIL, kv_ceil_classes=KV_CEIL_CLASSES,
                       ra_floor=RA_FLOOR, ra_floor_classes=RA_FLOOR_CLASSES)
    pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.D_class[...] = d["D_class"]; pol.reset()
    ego_pace = measure_pace(m, track, th0, 2000)
    s0 = (seed % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (seed % 3)
    mix = MIX[:n_opp]
    opps = [RacelineOpponent(track, s0=(s0 + g) % track.length, pace=fp * ego_pace, offset=0.0,
                             radius=KEEPOUT_R, a_lat=2.5) for (_, fp, g) in mix]
    trk = [ObstacleTracker(dt=0.05) for _ in opps]
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps; P.reset(s0=s0, v0=v0); m.reset()
    for o in opps: o.reset()
    for i, o in enumerate(opps): trk[i].update(o.pose()[:2])

    def nearest_ahead(s_ego):
        gaps = [signed_gap(track, s_ego, o.s) for o in opps]
        ahead = [(g, i) for i, g in enumerate(gaps) if g > 0]
        return (min(ahead)[1] if ahead else int(np.argmin([abs(g) for g in gaps]))), gaps

    passes = [0] * len(opps); seen = [False] * len(opps); contact_any = False; off = tr = False
    log = {k: [] for k in ("EX", "EY", "V", "S", "THETA")}
    log.update({f"O{i}X": [] for i in range(len(opps))}); log.update({f"O{i}Y": [] for i in range(len(opps))})
    prev_na_gap = None

    def step_once():
        nonlocal contact_any, prev_na_gap
        ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey)
        na, gaps = nearest_ahead(s_ego)
        pol.cls = PACE_KINDS.index(mix[na][0])                       # condition on the ENGAGED (nearest-ahead) opp
        _b = float(P._x[6]) if P._x.size > 6 else 0.0; _r = float(P._x[5]); _v = float(P._x[3])
        _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
        gr = (gaps[na] - prev_na_gap) / 0.05 if prev_na_gap is not None else 0.0; prev_na_gap = gaps[na]
        feat = race_features(track, P.state5(), [opps[na]], opp_speed_est=trk[na].speed,
                             slip=(_ar, _b, _r), gap_rate=gr)
        theta = np.asarray(pol.step(feat), float)
        m.set_obstacles([o.keepout() for o in opps])               # avoid ALL N
        return theta, m.value(P.state_dyn(), theta)["u0"], gaps

    theta, u, gaps = step_once()
    for stepi in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1])
        for i, o in enumerate(opps):
            o.step(0.05, ego=(ex, ey, float(P._x[3]))); trk[i].update(o.pose()[:2])
            ox, oy, _ = o.keepout()
            if np.hypot(ex - ox, ey - oy) < CONTACT_R: contact_any = True
            g = signed_gap(track, track.project(ex, ey), o.s)
            if g < 0 and abs(g) < L / 4 and not seen[i] and float(P._x[3]) > getattr(o, "speed", 0):
                passes[i] += 1; seen[i] = True
            elif g > 0.5:
                seen[i] = False
        log["EX"].append(ex); log["EY"].append(ey); log["V"].append(float(P._x[3]))
        log["S"].append(track.project(ex, ey)); log["THETA"].append(list(np.exp(theta)))
        for i, o in enumerate(opps):
            oxy = o.pose()[:2]; log[f"O{i}X"].append(float(oxy[0])); log[f"O{i}Y"].append(float(oxy[1]))
        if off or tr or contact_any:
            break
        theta, u, gaps = step_once()
    laps = _laps(log["S"]); clean = (not off) and (not contact_any)
    np.savez(OUT / f"states_multi_ltc_{seed}.npz", off=[off], contact=[contact_any],
             mix=[k for k, _, _ in mix], **{k: np.asarray(v) for k, v in log.items()})
    print(f"\n=== MULTI-OPPONENT (band net, seed {seed}): {len(opps)} opponents {[k for k,_,_ in mix]} ===", flush=True)
    print(f"laps {laps:.2f}  {'CLEAN' if clean else ('OFF-TRACK' if off else 'CONTACT')}  v_mean {np.mean(log['V']):.2f}", flush=True)
    for i, (k, fp, _) in enumerate(mix):
        print(f"  vs {k:7s} (x{fp:.2f} pace): passes {passes[i]}", flush=True)
    return dict(laps=laps, clean=clean, off=off, contact=contact_any, passes=passes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=9500); ap.add_argument("--n-opp", type=int, default=3)
    a = ap.parse_args()
    race_multi(a.seed, a.steps, a.n_opp)
