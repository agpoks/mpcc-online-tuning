"""Phase-A multi-car ONLINE training (branch race-multicar). A VARIABLE number of opponents (N varied in
{1..MAX_OPP} per episode, mixed pace) under the real 270-deg + occlusion sensor; the count-invariant
aggregate observation (race_features field=...: the FASTEST detected rival in the opponent slots + the
field aggregates crowding/avg-closing/count); the r_a/k_v bands; PolicyTuner (explore + RTRL + keep-best).
Both arms (ltc/mlp -- memory may finally help juggling several occluded opponents).

Reward = the tuned single-opponent shaping (race_reward) applied to the FASTEST rival (ego speed/progress/
off/edge + the fastest's pass/contact) + extra credit for additional opponents passed. Nets -> results/
race_multicar/nets/. Mirrors experiments/race_mode.run()'s loop; seeded/deterministic.

NOTE: needs a validation smoke-run once the box is idle (acados); the design is reviewable here first.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_multi_train.py --arms ltc mlp --seeds 2
"""
import sys, argparse, time, os
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import (race_features, field_summary, race_reward, FAIR_PACE, signed_gap,
                                    measure_pace, N_RACE_FEATURES_MULTI, MAX_OPP, KEEPOUT_R, CONTACT_R,
                                    PACE_KINDS, LR_VEH, A_LAT_RACE, CORRIDOR_KW, FOV_DEG, _visible,
                                    opponent_keepouts, KV_FLOOR, KV_FLOOR_CLASSES, KV_CEIL, KV_CEIL_CLASSES,
                                    RA_FLOOR, RA_FLOOR_CLASSES, QC_FLOOR, QC_FLOOR_CLASSES)
OUT = ROOT / "results" / "race_multicar"
RACE_KINDS = ("slower", "equal", "faster")   # the field is drawn from these (static = no opponent)
PASS_BONUS_EXTRA = 0.5                         # credit per ADDITIONAL opponent passed beyond the fastest


def run(arm, seed=0, n_ep=18, steps=4000, n_hidden=12, ego_pace=1.6, detect_range=18.0, fov_deg=FOV_DEG,
        occlude=True):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import LTCCell, MLPCell, THETA_HI, THETA_LO, PolicyTuner, WeightPolicy
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    lo, hi = np.asarray(THETA_LO, float), np.asarray(THETA_HI, float)
    half_fov = np.radians(fov_deg) / 2.0
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"multi_{arm}_{seed}")
    cell = (LTCCell if arm == "ltc" else MLPCell)(N_RACE_FEATURES_MULTI, n_hidden, seed=seed)
    pol = WeightPolicy(cell, th0, lo, hi, seed=seed, n_classes=len(PACE_KINDS), delta_log=0.6,
                       kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES, kv_ceil=KV_CEIL,
                       kv_ceil_classes=KV_CEIL_CLASSES, ra_floor=RA_FLOOR, ra_floor_classes=RA_FLOOR_CLASSES,
                       qc_floor=QC_FLOOR, qc_floor_classes=QC_FLOOR_CLASSES)
    tuner = PolicyTuner(m, pol, alpha=3e-3, explore=0.06, delta_clip=1.0, seed=seed, trust_region=0.01,
                        theta_prior=0.15, theta_explore=0.15, entropy=0.03, critic="return",
                        alpha_delta=0.01, trust_region_delta=0.03)
    rng = np.random.default_rng(seed); rows = []

    def slip_of(P):
        b = float(P._x[6]) if P._x.size > 6 else 0.0; r = float(P._x[5]); v = float(P._x[3])
        ar = -np.arctan2(v * np.sin(b) - LR_VEH * r, v * np.cos(b)) if v * np.cos(b) > 0.05 else 0.0
        return ar, b, r

    for ep in range(n_ep):
        N = int(rng.integers(1, MAX_OPP + 1))                      # varied opponent count per episode
        kinds = list(rng.choice(RACE_KINDS, N))
        _scn = seed + ep; s0 = ((_scn * 3) % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (_scn % 3)
        gaps0 = [3.0 + 4.0 * i + 1.5 * (ep % 3) for i in range(N)]  # spread ahead, varied
        opps = [RacelineOpponent(track, s0=(s0 + gaps0[i]) % track.length, pace=FAIR_PACE[kinds[i]] * ego_pace,
                                 offset=0.0, radius=KEEPOUT_R, a_lat=2.5) for i in range(N)]
        trk = [ObstacleTracker(dt=0.05) for _ in range(N)]
        P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps; P.reset(s0=s0, v0=v0); m.reset()
        for o in opps:
            o.reset()
        tuner.reset()
        # condition the class head on the FASTEST rival at the start (the overtake target)
        fast0 = int(np.argmax([o.pace for o in opps])); tuner.set_class(PACE_KINDS.index(kinds[fast0]))
        base = float(P.state5()[4]); seen_age = 0.0; prev_g = [None] * N
        last_s = [None] * N; blind_age = [0.0] * N     # per-opponent last-seen arc-length + time-since-seen
        passed = [False] * N; passes = 0; contact = False; off = tr = False
        ar, b, r_ = slip_of(P); ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey)
        gaps = [signed_gap(track, s_ego, o.s) for o in opps]
        det = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], detect_range, half_fov, occlude=occlude) for o in opps]
        for i, o in enumerate(opps):
            if det[i]:
                trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
            else:
                blind_age[i] += 0.05
        vis = [o for i, o in enumerate(opps) if det[i]]
        fastest, fld = field_summary(track, s_ego, vis, [gaps[i] for i in range(N) if det[i]])
        m.set_obstacles(opponent_keepouts(track, opps, det, last_s, blind_age, [trk[i].speed for i in range(N)]))
        fi = opps.index(fastest) if fastest is not None else None
        feat = race_features(track, P.state5(), ([fastest] if fastest else []),
                             opp_speed_est=(trk[fi].speed if fi is not None else None),
                             slip=(ar, b, r_), detected=1.0 if vis else 0.0, seen_age=0.0, field=fld)
        theta, u = tuner.act(feat, P.state_dyn())
        for _step in range(steps):
            s5n, r, off, tr = P.step(u)
            ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey); v = float(P._x[3])
            _lat = float(track.lateral(ex, ey)); _wl, _wr = track.width(track.wrap(s_ego))
            edge = min(float(_wr) - 0.12 - _lat, _lat + float(_wl) - 0.12)
            ar, b, r_ = slip_of(P); n_passed_now = 0
            for i, o in enumerate(opps):
                o.step(0.05, ego=(ex, ey, v)); ox, oy, _ = o.keepout()
                if np.hypot(ex - ox, ey - oy) < CONTACT_R:
                    contact = True
                g = signed_gap(track, s_ego, o.s); gaps[i] = g
                if g < 0 and abs(g) < track.length / 4 and not passed[i] and v > float(getattr(o, "speed", 0.0)):
                    passed[i] = True; passes += 1; n_passed_now += 1
                elif g > 0.5:
                    passed[i] = False
            det = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], detect_range, half_fov, occlude=occlude) for o in opps]
            seen_age = 0.0 if any(det) else min(seen_age + 0.05, 5.0)
            clos = []
            for i, o in enumerate(opps):
                if det[i]:
                    trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
                else:
                    blind_age[i] += 0.05
                if prev_g[i] is not None:
                    clos.append((gaps[i] - prev_g[i]) / 0.05)
                prev_g[i] = gaps[i]
            vis = [o for i, o in enumerate(opps) if det[i]]
            vis_gaps = [gaps[i] for i in range(N) if det[i]]
            vis_clos = [(gaps[i] - (prev_g[i] if prev_g[i] is not None else gaps[i])) for i in range(N) if det[i]]
            fastest, fld = field_summary(track, s_ego, vis, vis_gaps, vis_clos)
            # avoid DETECTED opponents + a bounded last-seen prediction for the just-occluded ones (<= MAX_OPP)
            m.set_obstacles(opponent_keepouts(track, opps, det, last_s, blind_age, [trk[i].speed for i in range(N)]))
            fi = opps.index(fastest) if fastest is not None else None
            gr_fast = ((gaps[fi] - prev_g[fi]) / 0.05) if (fi is not None and prev_g[fi] is not None) else 0.0
            fn = race_features(track, s5n, ([fastest] if fastest else []),
                               opp_speed_est=(trk[fi].speed if fi is not None else None),
                               slip=(ar, b, r_), gap_rate=gr_fast, detected=1.0 if vis else 0.0,
                               seen_age=seen_age, field=fld)
            fk = kinds[fi] if fi is not None else "equal"
            r_shaped = race_reward(fk, r, n_passed_now > 0, contact, v, float(getattr(fastest, "speed", v)),
                                   gaps[fi] if fi is not None else 99.0, alpha_r=ar, beta=b, off=off,
                                   edge_dist=edge, opp_pace=FAIR_PACE[fk])
            r_shaped += PASS_BONUS_EXTRA * max(0, n_passed_now - 1)  # credit the rest of the field
            out = tuner.learn(r_shaped, P.state_dyn(), fn, off or contact)
            if out[0] is None:
                break
            theta, u = out
            if off or tr or contact:
                break
        laps = (float(P.state5()[4]) - base) / track.length
        rows.append(dict(ep=ep, N=N, kinds=kinds, laps=round(laps, 3), passes=int(passes),
                         contact=bool(contact), off=bool(off), clean=bool((not off) and (not contact))))
        print(f"  [{arm}] ep {ep:2d}  N={N} {kinds}  laps {laps:5.2f}  passes {passes}  "
              f"{'CLEAN' if rows[-1]['clean'] else ('OFF' if off else 'CONTACT')}", flush=True)
    OUT.joinpath("nets").mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT / "nets" / f"race_{arm}_{seed}.npz"), G=pol.G, cell_p=pol.cell.p, th0=th0,
             n_hidden=pol.cell.n, arm=arm, seed=seed, lo=lo, hi=hi,
             D_class=(pol.D_class if pol.D_class is not None else np.zeros((0, 8))), delta_log=pol.delta_log)
    last = rows[-8:] if len(rows) >= 8 else rows
    return dict(arm=arm, seed=seed, clean=float(np.mean([r["clean"] for r in last])),
                passes=float(np.mean([r["passes"] for r in last])), rows=rows)


def one(job):
    arm, seed, ego_pace, n_ep, steps = job
    t0 = time.perf_counter()
    out = run(arm, seed=seed, n_ep=n_ep, steps=steps, ego_pace=ego_pace)
    out["wall_s"] = round(time.perf_counter() - t0, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["ltc", "mlp"]); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=18); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers (0 = sequential)")
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    mp = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                    max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="multi_pace")
    ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; MAX_OPP={MAX_OPP} ; varied N per episode", flush=True)
    jobs = [(arm, s, ego_pace, a.episodes, a.steps) for s in range(a.seeds) for arm in a.arms]
    done = lambda o: print(f"done {o['arm']} seed {o['seed']}: last-8 clean {o['clean']:.0%} "
                           f"passes {o['passes']:.2f} ({o.get('wall_s', 0):.0f}s)", flush=True)
    n_proc = a.jobs or 1
    if n_proc <= 1:
        for j in jobs:
            done(one(j))
    else:
        # SPAWN (not fork): fork-ing a Pool from a parent that has already touched acados/JAX (measure_pace)
        # deadlocks the workers (threads + fork). spawn starts fresh interpreters -> each worker builds its
        # own solver + net cleanly. (A fork pool hangs here at 0% CPU after the first build.)
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(n_proc, len(jobs))) as pool:
            for o in pool.imap_unordered(one, jobs):
                done(o)


if __name__ == "__main__":
    main()
