"""ROS-agnostic online-tuned-MPCC controller core for deployment.

One class, `OnlineMPCCController`:
  build it once (acados MPCC + trained weight-policy net + Track), then call `.control(...)` every tick
  with the car's estimated state -> returns the drive command (steering angle + longitudinal accel) plus
  the online-tuned weights and the solve time. It reuses mpcc_tuning.* unchanged; the only deploy-specific
  logic is the odom->state mapping, which lives in `.control` and is documented for your frame conventions.

State convention (from mpcc_tuning.plant_scuderia):
  state5    = [x, y, psi, v, s]                     (features)
  state_dyn = [x, y, psi, vx, s, vy, r, delta]      (acados solver)
where x,y = map position (m), psi = heading (rad), vx/vy = BODY-frame velocity (m/s), r = yaw rate (rad/s),
delta = actual steering angle (rad), s = arc length on the track (track.project).
"""
from __future__ import annotations
import time
import numpy as np

from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.ltc import LTCCell, N_FEATURES, WeightPolicy, features
from mpcc_tuning.mpcc import WEIGHT_NAMES

# two-layer qualifying corridor (copied from experiments/race_mode.CORRIDOR_KW to avoid importing the
# race harness on the car); matches how the solo qualifier net was trained.
QUALIFY_CORRIDOR = dict(two_layer_corridor=True, corridor_slack_scale=1.0, corridor_safety=0.07,
                        corridor_edge_safety=0.01, corridor_outer_scale=60.0)


def build_track(cfg: dict) -> Track:
    """The track the car runs on. Default = the validated icra_t2_smooth; for a real deploy map pass
    cfg['raceline_npz'] = an npz with cx,cy,wl,wr,raceline,vref (same schema as the geom-narrow files)."""
    rl = cfg.get("raceline_npz")
    if rl:
        g = np.load(rl)
        t = Track(g["cx"], g["cy"], ds=float(cfg.get("ds", 0.1)), w_left=g["wl"], w_right=g["wr"])
        t.raceline = g["raceline"]; t.v_ref = g["vref"]
        t.width_vehicle_adjusted = False
        if "kv_max" in cfg:
            t.kv_max = float(cfg["kv_max"])
        return t
    return Track.icra_t2_smooth()


class OnlineMPCCController:
    def __init__(self, net_npz: str, cfg: dict | None = None):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.track = build_track(cfg)
        self.track_name = cfg.get("track_name", "icra_t2_smooth")
        st = B.start(self.track_name)
        self.dt = float(cfg.get("dt", 0.05))
        self.mode = str(cfg.get("mode", "qualify"))      # "qualify" (solo fastest lap) | "race" (vs a field)
        if self.mode == "race":
            self._build_race(net_npz, cfg, st)
        else:
            _use_vref = bool(getattr(self.track, "use_optimiser_vref", False))
            _alat = getattr(self.track, "a_lat_ref", None)
            self.mpcc = AcadosMPCC(
                self.track, horizon=st.horizon, dt=self.dt, vehicle="dynamic",
                q_vref=st.q_vref, theta_global=True, discrete=True, use_track_vref=_use_vref,
                a_lat_sectors=([float(_alat)] * 4 if _alat else None), **QUALIFY_CORRIDOR,
                name=cfg.get("solver_name", "mpcc_online_deploy"))
            d = np.load(net_npz, allow_pickle=True)
            seed = int(d["seed"]); hidden = int(d["hidden"])
            self.pol = WeightPolicy(LTCCell(N_FEATURES, hidden, seed=seed), np.asarray(d["theta0"], float),
                                    np.asarray(d["lo"], float), np.asarray(d["hi"], float), seed=seed)
            self.pol.G[...] = d["G"]; self.pol.cell.p[...] = d["cell_p"]
            self._net_seed = seed
            self.last_theta = np.exp(np.asarray(d["theta0"], float))
        self.pol.reset(); self.mpcc.reset()
        # launch guard: cap grip claim for the first lap (cold-start k_v over-claim -> infeasible QP).
        self._ikv = WEIGHT_NAMES.index("k_v")
        self._kv_cap = float(np.log(cfg["kv_launch"])) if cfg.get("kv_launch") else None
        self._lap = 0; self._s = 0.0        # MPCC virtual progress, integrated from u2 (see control())
        # ----- ONLINE LEARNING (optional): wrap PolicyTuner (RTRL + keep-best), the on-car learner -----
        # Qualify mode only. Race online-learning (class-residual PolicyTuner + per-class keep-best) is a
        # later increment; the shipping race path is the FROZEN physical2 net (validated 100/100/67).
        self.learn = bool(cfg.get("learn")) and self.mode != "race"
        if cfg.get("learn") and self.mode == "race":
            self._learn_disabled = "online learning is not wired for race mode yet; running the race net FROZEN"
        if self.learn:
            from mpcc_tuning.ltc import PolicyTuner
            # skip_unused_q=True: the return critic never reads action_value's Q, so PolicyTuner.act's
            # SECOND acados solve/tick is pure waste -- skipping it halves per-tick solve time (p95
            # ~60->48ms, under the 50ms/20Hz budget) with BYTE-IDENTICAL learning. We keep explore at the
            # trained 0.05: turning action-dither OFF instead also fixed the time, but the net then
            # over-drove and ran off-track ~17% of ticks (vs 0% at 0.05) -- the dither is what keeps the
            # online learner safe. theta_explore=0.15 drives the policy-gradient.
            self.tuner = PolicyTuner(self.mpcc, self.pol, alpha=float(cfg.get("alpha", 2e-3)),
                                     explore=float(cfg.get("explore", 0.05)), delta_clip=1.0,
                                     seed=self._net_seed, trust_region=0.01, clock="time", critic="return",
                                     theta_explore=float(cfg.get("theta_explore", 0.15)), skip_unused_q=True)
            self._seed = self._net_seed
        self._learn_reset()

    def _learn_reset(self):
        self._started = False; self._prev_proj = None; self._lap_tick0 = 0; self._tick = 0
        self._lap_off = False; self._best_banked = False
        if self.learn:
            self.tuner.reset()

    def reset(self):
        self.pol.reset(); self.mpcc.reset(); self._lap = 0; self._s = 0.0; self._learn_reset()
        if self.mode == "race":
            self._race_reset()

    def control(self, x, y, psi, vx, vy=0.0, r=0.0, delta=0.0, opponents=()):
        """One control tick. Inputs are the car's estimated state (map pose + BODY-frame velocities +
        yaw rate + steering angle). `opponents` is used in race mode (see _control_race). Returns
        dict(steer, accel, speed_cmd, theta, s, lateral, solve_ms[, race fields])."""
        x, y, psi, vx, vy, r, delta = map(float, (x, y, psi, vx, vy, r, delta))
        if self.mode == "race":
            return self._control_race(x, y, psi, vx, vy, r, delta, opponents)
        if self.learn:
            return self._control_learn(x, y, psi, vx, vy, r, delta, opponents)
        L = self.track.length
        # s is the MPCC's VIRTUAL progress coordinate, integrated from its own progress-rate command
        # (plant: self.s += u2*dt) -- NOT a position projection (which is quantized to the track grid and
        # desynced from the solver's internal progress). The contouring/lag cost keeps s aligned with the
        # physical pose. Start s at 0 (car at the start line). See ScuderiaPlant.step line ~244.
        s = self._s
        self._lap = int(s // L)
        v = float(np.hypot(vx, vy))
        state5 = np.array([x, y, psi, v, s])
        state_dyn = np.array([x, y, psi, vx, s, vy, r, delta])
        th = np.asarray(self.pol.step(features(self.track, state5, opponents)), float)
        # launch k_v cap for the first lap of travel (cold-start grip over-claim guard)
        if self._kv_cap is not None and self._lap == 0:
            th[self._ikv] = min(th[self._ikv], self._kv_cap)
        t0 = time.perf_counter()
        u = np.asarray(self.mpcc.value(state_dyn, th)["u0"], float).ravel()
        solve_ms = (time.perf_counter() - t0) * 1e3
        if u.size < 3 or not np.isfinite(u).all():          # failed/NaN solve -> let the node brake
            return dict(steer=0.0, accel=0.0, speed_cmd=v, u0=None, theta=self.last_theta,
                        s=s, lateral=float(self.track.lateral(x, y)), solve_ms=round(solve_ms, 2))
        self._s += float(u[2]) * self.dt                   # integrate the MPCC progress-rate, like the plant
        self.last_theta = np.exp(th)
        steer, accel = float(u[0]), float(u[1])
        # u0 is the full MPCC input (steer, accel, progress-rate); the real car actuates only steer+accel
        # (progress s is MEASURED from localization), the sim plant needs the whole vector.
        return dict(steer=steer, accel=accel, speed_cmd=v + accel * float(self.cfg.get("dt", 0.05)),
                    u0=u, theta=self.last_theta, s=s, lateral=float(self.track.lateral(x, y)),
                    solve_ms=round(solve_ms, 2))

    def _off_track(self, x, y, s):
        """Off-corridor test against the (variable) usable half-width, like ScuderiaPlant.step."""
        lat = float(self.track.lateral(x, y))
        try:
            wl, wr = self.track.width(self.track.wrap(s))
            off = (lat > float(wr) - 0.12) or (-lat > float(wl) - 0.12)
        except Exception:
            off = abs(lat) > float(getattr(self.track, "half_width", 1.0)) - 0.12
        return bool(off), lat

    def _control_learn(self, x, y, psi, vx, vy, r, delta, opponents=()):
        """Online-learning tick: PolicyTuner RTRL (act/learn) + per-lap keep-best. The weights ADAPT on
        the car. Reward = physical progress - 5*off (plant convention, clock='time', critic='return').
        Call after each new pose; returns the same dict as control() + 'off'/'banked'."""
        L = self.track.length; s = self._s; self._lap = int(s // L); self._tick += 1
        v = float(np.hypot(vx, vy))
        state5 = np.array([x, y, psi, v, s]); state_dyn = np.array([x, y, psi, vx, s, vy, r, delta])
        feat = features(self.track, state5, opponents)
        off, lat = self._off_track(x, y, s)
        # physical progress since the previous tick (projected), for the reward of the previous action
        proj = float(self.track.project(x, y))
        if self._prev_proj is None:
            dprog = 0.0
        else:
            dprog = proj - self._prev_proj
            dprog += L if dprog < -0.5 * L else (-L if dprog > 0.5 * L else 0.0)
        self._prev_proj = proj
        self._lap_off = self._lap_off or off
        reward = dprog - (5.0 if off else 0.0)
        t0 = time.perf_counter()
        if not self._started:
            theta, u = self.tuner.act(feat, state_dyn); self._started = True
        else:
            theta, u = self.tuner.learn(reward, state_dyn, feat, off)   # off -> terminated; clock='time'
        solve_ms = (time.perf_counter() - t0) * 1e3
        banked = None
        if theta is None or u is None:
            # OFF-TRACK: the tuner TERMINATED the episode and returned no action (critic='return' emits
            # (None, None) on a terminated tick). Do NOT index a null action -- that was a 0-d-array crash
            # that would kill the control loop on the first off-track tick. Instead: don't bank the
            # excursion (revert to the banked-best net = safety), reset the episode, and RECOVER this tick
            # by solving once with the reverted net (the MPCC contouring cost pulls the car back to the
            # line). Matches the crash->penalize->revert->recover->continue rule.
            lap_time = (self._tick - self._lap_tick0) * self.dt
            banked = self.tuner.end_episode(-lap_time, crashed=True, validate=False)   # crashed -> revert
            self.tuner.reset(); self._started = False
            self._lap_off = False; self._lap_tick0 = self._tick
            th = np.asarray(self.pol.step(feat), float)
            u = np.asarray(self.mpcc.value(state_dyn, th)["u0"], float).ravel()
            theta = th
        theta = np.asarray(theta, float).ravel(); u = np.asarray(u, float).ravel()
        if u.size < 3 or not np.isfinite(u).all():        # failed/NaN solve -> let the node brake
            return dict(steer=0.0, accel=0.0, speed_cmd=v, u0=None, theta=self.last_theta,
                        s=s, lateral=lat, off=off, banked=banked, solve_ms=round(solve_ms, 2))
        self._s += float(u[2]) * self.dt
        # ---- per-lap keep-best: bank the net if this lap was faster & clean, else revert ----
        if banked is None and int(self._s // L) > self._lap:   # crossed the start line into a new lap
            lap_ticks = self._tick - self._lap_tick0; self._lap_tick0 = self._tick
            lap_time = lap_ticks * self.dt
            # validate=False: direct keep-best (no frozen validation lap, which we can't run mid-drive on
            # the car). Banks a faster CLEAN lap, REVERTS a slower/crashed one -> the on-car safety guard.
            banked = self.tuner.end_episode(-lap_time, crashed=self._lap_off, validate=False)
            self._best_banked = self._best_banked or (banked == "banked")
            self._lap_off = False
        self.last_theta = np.exp(theta)
        return dict(steer=float(u[0]), accel=float(u[1]), speed_cmd=v + float(u[1]) * self.dt, u0=u,
                    theta=self.last_theta, s=s, lateral=lat, off=off, banked=banked,
                    solve_ms=round(solve_ms, 2))

    # ===================== RACE MODE (frozen physical2 net vs a field of opponents) =====================
    def _build_race(self, net_npz, cfg, st):
        """Race build: race-corridor acados (keep-outs + wider a_lat) + the FROZEN physical2 race net
        (WeightPolicy with a class-residual head + k_v/r_a/q_c safety bands). Mirrors the experiments/
        race_mode build and race_multi_eval._load_pol so the deployed net is byte-identical to the
        validated multi-car eval (LTC 100/100/67). Online race-learning is a later increment -- here the
        race net runs FROZEN (pol.step, no RTRL), exactly as the eval that produced the paper numbers."""
        import sys as _sys
        from pathlib import Path as _Path
        _root = str(_Path(__file__).resolve().parents[2])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)                     # repo root, so `experiments.*` imports (same repo)
        import experiments.race_mode as RM                 # race_mode has no module-level heavy/plot imports
        from mpcc_tuning.ltc import LTCCell, MLPCell
        from mpcc_tuning.opponents import Opponent, ObstacleTracker
        self._RM = RM; self._Opponent = Opponent; self._ObstacleTracker = ObstacleTracker
        self._MAX_OPP = RM.MAX_OPP
        self._pred_r = float(cfg["pred_r"]) if cfg.get("pred_r") else RM.PRED_R
        self.ego_pace = float(cfg.get("ego_pace", 1.77))   # ego solo pace, for opponent pace-class (physical2 ~1.77)
        self.mpcc = AcadosMPCC(
            self.track, horizon=st.horizon, dt=self.dt, vehicle="dynamic", q_vref=st.q_vref,
            discrete=True, max_obstacles=RM.MAX_OPP, a_lat_sectors=[RM.A_LAT_RACE] * 4, **RM.CORRIDOR_KW,
            name=cfg.get("solver_name", "mpcc_online_race"))
        d = np.load(net_npz, allow_pickle=True)
        seed = int(d["seed"]); hidden = int(d["n_hidden"]); arm = str(d["arm"]); ncls = int(d["D_class"].shape[0])
        cell = (LTCCell if arm == "ltc" else MLPCell)(RM.N_RACE_FEATURES_MULTI, hidden, seed=seed)
        qc_on = (arm == "ltc")                             # q_c floor + crowd band are LTC-only (they break MLP)
        qcf, qcc = (RM.QC_FLOOR, RM.QC_FLOOR_CLASSES) if qc_on else (0.0, ())
        qlo, qhi = (RM.QC_CROWD_LO, RM.QC_CROWD_HI) if qc_on else (0.0, 0.0)
        self.pol = WeightPolicy(
            cell, np.asarray(d["th0"], float), np.asarray(d["lo"], float), np.asarray(d["hi"], float),
            seed=seed, n_classes=ncls, delta_log=float(d["delta_log"]),
            kv_floor=RM.KV_FLOOR, kv_floor_classes=RM.KV_FLOOR_CLASSES, kv_ceil=RM.KV_CEIL,
            kv_ceil_classes=RM.KV_CEIL_CLASSES, ra_floor=RM.RA_FLOOR, ra_floor_classes=RM.RA_FLOOR_CLASSES,
            qc_floor=qcf, qc_floor_classes=qcc, qc_crowd_lo=qlo, qc_crowd_hi=qhi, qc_crowd_n=RM.QC_CROWD_N)
        self.pol.G[...] = d["G"]; self.pol.cell.p[...] = d["cell_p"]; self.pol.D_class[...] = d["D_class"]
        self._net_seed = seed
        self.last_theta = np.exp(np.asarray(d["th0"], float))
        self._race_reset()

    def _race_reset(self):
        """Per-enable race state: opponent tracks keyed by id, crowd memory, closing-rate history."""
        self._opp = {}            # id -> dict(obj, trk, last_s, blind_age, prev_gap)
        self._seen_ids = set()    # distinct opponents ever seen this race (crowd memory, matches the eval)
        self._seen_age = 0.0      # seconds since ANY opponent was last detected
        self._next_oid = 0        # for synthesising ids when perception provides none

    def _classify_pace(self, speed):
        """Nearest relative-pace class (static/slower/equal/faster) of an opponent, by speed/ego_pace."""
        RM = self._RM; ratio = float(speed) / max(self.ego_pace, 1e-3)
        return int(np.argmin([abs(RM.FAIR_PACE[k] - ratio) for k in RM.PACE_KINDS]))

    def _assoc_id(self, info, x, y):
        """Opponent identity across ticks: a perception-provided id if present, else the nearest previous
        track within a 1.5 m gate, else a fresh id (greedy association for the no-id perception case)."""
        if info.get("id") is not None:
            return str(info["id"])
        best, bd = None, 1.5
        for oid, rec in self._opp.items():
            ox, oy, _ = rec["obj"].keepout()
            dd = float(np.hypot(x - ox, y - oy))
            if dd < bd:
                best, bd = oid, dd
        if best is not None:
            return best
        self._next_oid += 1
        return f"a{self._next_oid}"

    def _control_race(self, x, y, psi, vx, vy, r, delta, opponents):
        """One RACE tick (FROZEN net). Mirrors race_multi_eval.run_eval.emit: ingest the detected field,
        pick the fastest rival + field aggregates, set MPCC keep-outs (detected + bounded blind prediction
        for a just-occluded car), emit class/crowd-conditioned weights, solve. `opponents` = iterable of
        per-opponent dicts: x, y [required, MAP frame]; v (speed m/s) or vx,vy; id (stable, optional);
        s,d (Frenet arc-length/offset, optional -- used directly when given, else projected)."""
        RM = self._RM; L = self.track.length; s = self._s; self._lap = int(s // L)
        v = float(np.hypot(vx, vy))
        state5 = np.array([x, y, psi, v, s]); state_dyn = np.array([x, y, psi, vx, s, vy, r, delta])
        s_ego = float(self.track.project(x, y))
        # ---- ingest the detected field: update per-id tracks ----
        cur = set()
        for info in (opponents or ()):
            ox, oy = float(info["x"]), float(info["y"])
            oid = self._assoc_id(info, ox, oy)
            if info.get("s") is not None:
                os_, od = float(info["s"]) % L, float(info.get("d", 0.0))
            else:
                os_, od = float(self.track.project(ox, oy)) % L, float(self.track.lateral(ox, oy))
            spd = info.get("v")
            if spd is None:
                spd = float(np.hypot(float(info.get("vx", 0.0)), float(info.get("vy", 0.0))))
            rec = self._opp.get(oid)
            if rec is None:
                rec = dict(obj=self._Opponent(self.track, s0=os_, speed=float(spd), offset=od, radius=RM.KEEPOUT_R),
                           trk=self._ObstacleTracker(dt=self.dt), last_s=None, blind_age=0.0, prev_gap=None)
                self._opp[oid] = rec
            rec["obj"].s = os_; rec["obj"].offset = od; rec["obj"].speed = float(spd)
            kx, ky, _ = rec["obj"].keepout(); rec["trk"].update((kx, ky))
            rec["last_s"] = os_; rec["blind_age"] = 0.0
            self._seen_ids.add(oid); cur.add(oid)
        # age-out opponents not reported this tick: keep for a bounded blind prediction, then forget
        for oid in list(self._opp):
            if oid in cur:
                continue
            self._opp[oid]["blind_age"] += self.dt
            if self._opp[oid]["last_s"] is None or self._opp[oid]["blind_age"] >= RM.BLIND_CAP:
                del self._opp[oid]
        # cap to MAX_OPP nearest by |gap| (the solver carries at most MAX_OPP keep-outs)
        ids = sorted(self._opp, key=lambda i: abs(RM.signed_gap(self.track, s_ego, self._opp[i]["obj"].s)))[:self._MAX_OPP]
        opps = [self._opp[i]["obj"] for i in ids]
        det = [i in cur for i in ids]
        last_s = [self._opp[i]["last_s"] for i in ids]
        blind_age = [self._opp[i]["blind_age"] for i in ids]
        spd_est = [self._opp[i]["trk"].speed for i in ids]
        gaps = [RM.signed_gap(self.track, s_ego, o.s) for o in opps]
        vis = [opps[k] for k in range(len(ids)) if det[k]]
        vis_gaps = [gaps[k] for k in range(len(ids)) if det[k]]
        vis_clos = [((gaps[k] - self._opp[ids[k]]["prev_gap"]) if self._opp[ids[k]]["prev_gap"] is not None else 0.0)
                    for k in range(len(ids)) if det[k]]
        fastest, fld = RM.field_summary(self.track, s_ego, vis, vis_gaps, vis_clos)
        # MPCC keep-outs: detected opponents + a bounded last-seen prediction for just-occluded ones
        self.mpcc.set_obstacles(RM.opponent_keepouts(self.track, opps, det, last_s, blind_age, spd_est, r=self._pred_r))
        # class-conditioned head: key on the fastest DETECTED rival's relative pace (hold last when blind)
        if fastest is not None:
            self.pol.cls = self._classify_pace(fastest.speed)
        elif self.pol.cls is None:
            self.pol.cls = 0
        self._seen_age = 0.0 if vis else min(self._seen_age + self.dt, 5.0)
        fi = opps.index(fastest) if fastest is not None else None
        gr = ((gaps[fi] - self._opp[ids[fi]]["prev_gap"]) / self.dt) \
            if (fi is not None and self._opp[ids[fi]]["prev_gap"] is not None) else 0.0
        b = float(np.arctan2(vy, vx)) if abs(vx) > 0.05 else 0.0
        a_r = float(-np.arctan2(vy - RM.LR_VEH * r, vx)) if abs(vx) > 0.05 else 0.0
        feat = RM.race_features(self.track, state5, ([fastest] if fastest is not None else []),
                                opp_speed_est=(spd_est[fi] if fi is not None else None),
                                slip=(a_r, b, r), gap_rate=gr, detected=1.0 if vis else 0.0,
                                seen_age=self._seen_age, field=fld)
        self.pol.set_crowd(len(self._seen_ids))            # memory-based distinct-seen crowd estimate
        for k, i in enumerate(ids):                        # roll closing-rate history
            self._opp[i]["prev_gap"] = gaps[k]
        th = np.asarray(self.pol.step(feat), float)
        if self._kv_cap is not None and self._lap == 0:    # cold-start grip-claim cap (first lap)
            th[self._ikv] = min(th[self._ikv], self._kv_cap)
        t0 = time.perf_counter()
        u = np.asarray(self.mpcc.value(state_dyn, th)["u0"], float).ravel()
        solve_ms = (time.perf_counter() - t0) * 1e3
        if u.size < 3 or not np.isfinite(u).all():         # failed/NaN solve -> let the node brake
            return dict(steer=0.0, accel=0.0, speed_cmd=v, u0=None, theta=self.last_theta, s=s,
                        lateral=float(self.track.lateral(x, y)), solve_ms=round(solve_ms, 2))
        self._s += float(u[2]) * self.dt
        self.last_theta = np.exp(th)
        return dict(steer=float(u[0]), accel=float(u[1]), speed_cmd=v + float(u[1]) * self.dt, u0=u,
                    theta=self.last_theta, s=s, lateral=float(self.track.lateral(x, y)),
                    n_opp=len(ids), n_det=int(sum(det)), crowd=len(self._seen_ids),
                    cls=int(self.pol.cls if self.pol.cls is not None else -1), solve_ms=round(solve_ms, 2))

    def save_ckpt(self, path):
        """Persist the current (or banked-best) net + keep-best bank, per scale-mpcc's per-lap resume."""
        best = getattr(self.tuner, "_best", None) if self.learn else None
        G, cp = (best[1], best[2]) if best is not None else (self.pol.G, self.pol.cell.p)
        np.savez(str(path), G=G, cell_p=cp, th0=self.pol.theta0, lo=self.pol.lo, hi=self.pol.hi,
                 hidden=self.pol.cell.n, seed=getattr(self, "_seed", 0),
                 best_score=(best[0] if best is not None else np.nan))

    def load_ckpt(self, path):
        d = np.load(str(path), allow_pickle=True)
        self.pol.G[...] = d["G"]; self.pol.cell.p[...] = d["cell_p"]
        if self.learn and "best_score" in d and np.isfinite(d["best_score"]):
            self.tuner._best = (float(d["best_score"]), d["G"].copy(), d["cell_p"].copy(), None)
