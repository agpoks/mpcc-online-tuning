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
                                    RA_FLOOR, RA_FLOOR_CLASSES, QC_FLOOR, QC_FLOOR_CLASSES,
                                    QC_CROWD_LO, QC_CROWD_HI, QC_CROWD_N)
OUT = ROOT / "results" / "race_multicar"
RACE_KINDS = ("slower", "equal", "faster")   # the field is drawn from these (static = no opponent)
PASS_BONUS_EXTRA = 0.5                         # credit per ADDITIONAL opponent passed beyond the fastest


def run(arm, seed=0, n_ep=18, steps=4000, n_hidden=12, ego_pace=1.6, detect_range=18.0, fov_deg=FOV_DEG,
        occlude=True, loose_rails=False, physical_rails=False, recover=False, net_dir=None):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import LTCCell, MLPCell, THETA_HI, THETA_LO, PolicyTuner, WeightPolicy
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    half_fov = np.radians(fov_deg) / 2.0
    if loose_rails or physical_rails:
        # ADAPTIVE rails. The learner EXPLORES the factor-2 adaptation box; the limits are PHYSICAL crash
        # edges encoded straight into the box (lo/hi) so there are NO behavioural band clamps overriding the
        # learning. q_c's lower edge opens to the line-COLLAPSE rail (0.30) so the line can widen to weave;
        # k_v's upper edge is the grip-OVER-CLAIM rail (0.70) with NO floor (free to go safe-slow). Inside
        # that box the online RTRL + keep-best + crash-recovery find the weights (fine if it explores values
        # that "make no sense" -- it learns from them). The box is saved in the net, so the eval reconstructs
        # the identical rails with NO extra clamps. q_c idx 0, r_a idx 4, k_v idx 7 (WEIGHT_NAMES).
        #   --loose-rails    : ONLY those two edges -> the learner drove r_a to its box floor (3) and that
        #                      OVER-DRIVES past the ~2.5 m/s the corridor holds (ran wide -> N=1,2 regressed).
        #   --physical-rails : ALSO floor r_a at the corridor OVER-DRIVE rail (10, loose run proved <10 runs
        #                      wide) with a little room above (13) -- the one PHYSICAL limit --loose dropped.
        lo, hi = B.adaptation_box("icra_t2_smooth", 2.0); lo = lo.copy(); hi = hi.copy()
        lo[0] = float(np.log(0.30)); hi[7] = float(np.log(0.70))
        kv_fl = kv_ce = qc_fl = 0.0; kv_fl_c = kv_ce_c = qc_fl_c = (); qc_c_lo = qc_c_hi = 0.0
        if physical_rails:
            # r_a's baseline (6) sits BELOW the corridor-overdrive floor (10), so it cannot be a box edge --
            # the anchor must be interior. Keep it as the proven BAND clamp (r_a >= RA_FLOOR), the one
            # physical rail the box cannot encode; the loose run (r_a->3) proved <10 over-drives off the line.
            ra_fl, ra_fl_c = RA_FLOOR, tuple(range(len(PACE_KINDS)))
        else:
            ra_fl, ra_fl_c = 0.0, ()
    else:
        lo, hi = np.asarray(THETA_LO, float), np.asarray(THETA_HI, float)
        kv_fl, kv_fl_c = KV_FLOOR, KV_FLOOR_CLASSES; kv_ce, kv_ce_c = KV_CEIL, KV_CEIL_CLASSES
        ra_fl, ra_fl_c = RA_FLOOR, RA_FLOOR_CLASSES
        qc_fl, qc_fl_c = (QC_FLOOR if arm == "ltc" else 0.0), QC_FLOOR_CLASSES
        qc_c_lo = (QC_CROWD_LO if arm == "ltc" else 0.0); qc_c_hi = (QC_CROWD_HI if arm == "ltc" else 0.0)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"multi_{arm}_{seed}")
    cell = (LTCCell if arm == "ltc" else MLPCell)(N_RACE_FEATURES_MULTI, n_hidden, seed=seed)
    pol = WeightPolicy(cell, th0, lo, hi, seed=seed, n_classes=len(PACE_KINDS), delta_log=0.6,
                       kv_floor=kv_fl, kv_floor_classes=kv_fl_c, kv_ceil=kv_ce, kv_ceil_classes=kv_ce_c,
                       ra_floor=ra_fl, ra_floor_classes=ra_fl_c, qc_floor=qc_fl, qc_floor_classes=qc_fl_c,
                       qc_crowd_lo=qc_c_lo, qc_crowd_hi=qc_c_hi, qc_crowd_n=QC_CROWD_N)
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
        n_crash = 0; laps_acc = 0.0; seg_base = base   # crash-recovery bookkeeping (progress banked per segment)
        gaps = [signed_gap(track, track.project(float(P._x[0]), float(P._x[1])), o.s) for o in opps]

        def observe(seen_age_val):
            """Build the policy observation at the CURRENT plant+opponent state (and set the solver's
            keep-outs + the memory-crowd estimate). Used at episode start and after every crash-recovery
            reset, so the net always re-acts on a fresh, consistent view. Mutates trk/last_s/blind_age --
            the race memory -- in place, which is what we want: it carries across a recovery."""
            ex, ey = float(P._x[0]), float(P._x[1]); s_e = track.project(ex, ey)
            a_, b_, r2_ = slip_of(P)
            dts = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], detect_range, half_fov, occlude=occlude) for o in opps]
            for i, o in enumerate(opps):
                if dts[i]:
                    trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
                else:
                    blind_age[i] += 0.05
            vis_ = [o for i, o in enumerate(opps) if dts[i]]
            gp = [signed_gap(track, s_e, o.s) for o in opps]
            fastest_, fld_ = field_summary(track, s_e, vis_, [gp[i] for i in range(N) if dts[i]])
            m.set_obstacles(opponent_keepouts(track, opps, dts, last_s, blind_age, [trk[i].speed for i in range(N)]))
            fi_ = opps.index(fastest_) if fastest_ is not None else None
            ft = race_features(track, P.state5(), ([fastest_] if fastest_ else []),
                               opp_speed_est=(trk[fi_].speed if fi_ is not None else None),
                               slip=(a_, b_, r2_), detected=1.0 if vis_ else 0.0, seen_age=seen_age_val, field=fld_)
            pol.set_crowd(sum(ls is not None for ls in last_s))      # memory-based crowd estimate (distinct seen)
            return ft

        theta, u = tuner.act(observe(0.0), P.state_dyn())
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
            pol.set_crowd(sum(ls is not None for ls in last_s))      # memory-based crowd estimate (distinct seen)
            crashed = bool(off or tr or contact)
            out = tuner.learn(r_shaped, P.state_dyn(), fn, crashed and not recover)
            if out[0] is None:
                break
            theta, u = out
            if crashed:
                if not recover:
                    break
                # REAL-RACE RECOVERY. The crash is ALREADY penalised in r_shaped (contact -15/-8, off
                # -OFF_PENALTY), so the net LEARNS this corner/opponent situation -- then, like a marshal,
                # the car is placed back on the CLOSEST point of the raceline to where it crashed (the arc
                # projection of the crash pose, lateral 0) and drives on. The episode does NOT die, so the
                # learner keeps accumulating experience and learns the recovery from its own mistake instead
                # of the keep-best silently reverting a dead run. The net's MEMORY is kept (it remembers the
                # incident); only the plant + solver reset. Progress is banked per segment so the reset's
                # arc jump does not corrupt the lap count.
                n_crash += 1
                s_now = track.project(float(P._x[0]), float(P._x[1]))   # closest raceline arc to the crash
                laps_acc += (float(P.s) - seg_base) / track.length
                P.reset(s0=s_now, v0=1.0); m.reset()
                seg_base = float(P.s); contact = False; off = tr = False
                theta, u = tuner.act(observe(seen_age), P.state_dyn())
        laps = laps_acc + (float(P.s) - seg_base) / track.length
        rows.append(dict(ep=ep, N=N, kinds=kinds, laps=round(laps, 3), passes=int(passes), crashes=int(n_crash),
                         contact=bool(contact), off=bool(off),
                         clean=bool((not off) and (not contact) and n_crash == 0)))
        if recover:
            tag = f"crashes {n_crash} ({n_crash / max(laps, 1e-6):.2f}/lap)"
        else:
            tag = 'CLEAN' if rows[-1]['clean'] else ('OFF' if off else 'CONTACT')
        print(f"  [{arm}] ep {ep:2d}  N={N} {kinds}  laps {laps:5.2f}  passes {passes}  {tag}", flush=True)
    nd = Path(net_dir) if net_dir else (OUT / "nets"); nd.mkdir(parents=True, exist_ok=True)
    np.savez(str(nd / f"race_{arm}_{seed}.npz"), G=pol.G, cell_p=pol.cell.p, th0=th0,
             n_hidden=pol.cell.n, arm=arm, seed=seed, lo=lo, hi=hi,
             D_class=(pol.D_class if pol.D_class is not None else np.zeros((0, 8))), delta_log=pol.delta_log)
    last = rows[-8:] if len(rows) >= 8 else rows
    return dict(arm=arm, seed=seed, clean=float(np.mean([r["clean"] for r in last])),
                passes=float(np.mean([r["passes"] for r in last])),
                crashes_per_lap=float(np.mean([r["crashes"] / max(r["laps"], 1e-6) for r in last])), rows=rows)


def one(job):
    arm, seed, ego_pace, n_ep, steps, loose_rails, physical_rails, recover, net_dir = job
    t0 = time.perf_counter()
    out = run(arm, seed=seed, n_ep=n_ep, steps=steps, ego_pace=ego_pace,
              loose_rails=loose_rails, physical_rails=physical_rails, recover=recover, net_dir=net_dir)
    out["wall_s"] = round(time.perf_counter() - t0, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["ltc", "mlp"]); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=18); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers (0 = sequential)")
    ap.add_argument("--loose-rails", action="store_true",
                    help="ADAPTIVE: explore the factor-2 box, only the q_c/k_v physical crash edges (no behavioural bands)")
    ap.add_argument("--physical-rails", action="store_true",
                    help="ADAPTIVE + the r_a corridor-overdrive floor (the one physical rail --loose drops); no behavioural bands")
    ap.add_argument("--recover", action="store_true",
                    help="crash -> penalise in the reward + place back on the CLOSEST raceline point + keep driving (no terminate)")
    ap.add_argument("--net-dir", default=None, help="where to save the nets (default results/race_multicar/nets)")
    a = ap.parse_args()
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    mp = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                    max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="multi_pace")
    ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; MAX_OPP={MAX_OPP} ; varied N per episode", flush=True)
    jobs = [(arm, s, ego_pace, a.episodes, a.steps, a.loose_rails, a.physical_rails, a.recover, a.net_dir)
            for s in range(a.seeds) for arm in a.arms]
    done = lambda o: print(f"done {o['arm']} seed {o['seed']}: last-8 clean {o['clean']:.0%} "
                           f"passes {o['passes']:.2f} crashes/lap {o.get('crashes_per_lap', 0):.2f} "
                           f"({o.get('wall_s', 0):.0f}s)", flush=True)
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
