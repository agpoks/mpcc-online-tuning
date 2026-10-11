"""Comprehensive race telemetry for an online net -- so you can SEE what the learner actually does, not one
muddled plot. Over ONE race it logs, versus lap distance s:
  * ALL 9 MPCC weights, EACH drawn inside its min/max RAILS (shaded) so you can tell a learned move from a
    bound it is sitting on (q_c/q_l/q_v, r_d/r_a/r_dv, d_obs/k_v/d_bound);
  * STATES: speed, lateral offset from the raceline, lateral slip angle, LONGITUDINAL accel a_x (the r_a
    aggression) with the lateral a_y, and yaw rate;
  * INPUTS u: steering cmd, accel cmd, progress rate;
  * REWARD per tick (and cumulative).
Crash-recovery is ON (reset to the CLOSEST raceline point), so a crash shows as a marked recovery, not a
dead run. Deterministic (explore off). Artifacts: PNG + PDF + TikZ (pgfplots) + GIF, and the raw npz so
--from-saved regenerates them with no solver.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/viz_adaptive.py --net-dir results/race_multicar/nets_physical --loose --arm ltc --seed 0 --N 3 --phys 1 --tag physical
    python3 tools/viz_adaptive.py --from-saved --tag physical
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import (race_features, field_summary, race_reward, FAIR_PACE, signed_gap, measure_pace,
                                    N_RACE_FEATURES_MULTI, MAX_OPP, KEEPOUT_R, CONTACT_R, PACE_KINDS, LR_VEH,
                                    A_LAT_RACE, CORRIDOR_KW, FOV_DEG, _visible, opponent_keepouts)
from experiments.race_multi_eval import _load_pol, OUT
from mpcc_tuning.mpcc import WEIGHT_NAMES
VOUT = OUT / "viz"; VOUT.mkdir(parents=True, exist_ok=True)
RACE_KINDS = ("slower", "equal", "faster")
GROUPS = [("cost", ["q_c", "q_l", "q_v"]), ("effort", ["r_d", "r_a", "r_dv"]), ("constraint", ["d_obs", "k_v", "d_bound"])]


def rich_race(arm, seed, N, phys, net_dir, loose, ra_on=False, steps=2600):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"viz_{arm}_{seed}")
    pol = _load_pol(arm, seed, net_dir=net_dir, loose=loose, ra_floor_on=ra_on)
    if pol is None:
        return None
    ego_pace = measure_pace(m, track, th0, 1500)
    half_fov = np.radians(FOV_DEG) / 2.0
    # deterministic field for (N, phys): fixed kinds + spread, phys varies the start pose
    kinds = [RACE_KINDS[(phys + i) % 3] for i in range(N)]
    s0 = (phys % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (phys % 3)
    gaps0 = [3.0 + 4.0 * i for i in range(N)]
    opps = [RacelineOpponent(track, s0=(s0 + gaps0[i]) % track.length, pace=FAIR_PACE[kinds[i]] * ego_pace,
                             offset=0.0, radius=KEEPOUT_R, a_lat=2.5) for i in range(N)]
    trk = [ObstacleTracker(dt=0.05) for _ in range(N)]
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps + 10; P.reset(s0=s0, v0=v0); m.reset()
    for o in opps:
        o.reset()
    pol.reset()
    fast0 = int(np.argmax([o.pace for o in opps])); pol.cls = PACE_KINDS.index(kinds[fast0])
    last_s = [None] * N; blind_age = [0.0] * N; seen_age = 0.0; prev_g = [None] * N
    lo_lin = np.exp(np.asarray(pol.lo, float)); hi_lin = np.exp(np.asarray(pol.hi, float))

    def slip_of():
        b = float(P._x[6]) if P._x.size > 6 else 0.0; r = float(P._x[5]); v = float(P._x[3])
        ar = -np.arctan2(v * np.sin(b) - LR_VEH * r, v * np.cos(b)) if v * np.cos(b) > 0.05 else 0.0
        return ar, b, r

    def observe():
        ex, ey = float(P._x[0]), float(P._x[1]); s_e = track.project(ex, ey); a_, b_, r2_ = slip_of()
        dts = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], 18.0, half_fov, occlude=True) for o in opps]
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
                           slip=(a_, b_, r2_), detected=1.0 if vis_ else 0.0, seen_age=seen_age, field=fld_)
        pol.set_crowd(sum(ls is not None for ls in last_s))
        return ft, fastest_, fi_, dts

    L = {k: [] for k in ["S", "EX", "EY", "V", "LATOFF", "SLIP", "AY", "YAWRATE", "R", "DET"]}
    L["THETA"] = []; L["U"] = []; L["OPP"] = []; crashes = []
    feat, fastest, fi, det = observe()
    theta = pol.step(feat); u = m.value(P.state_dyn(), theta)["u0"]
    passed = [False] * N; v_prev = float(P._x[3])
    for _step in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey); v = float(P._x[3])
        lat = float(track.lateral(ex, ey)); _wl, _wr = track.width(track.wrap(s_ego))
        edge = min(float(_wr) - 0.12 - lat, lat + float(_wl) - 0.12)
        ar, b, r_ = slip_of(); contact = False
        for i, o in enumerate(opps):
            o.step(0.05, ego=(ex, ey, v)); ox, oy, _ = o.keepout()
            if np.hypot(ex - ox, ey - oy) < CONTACT_R:
                contact = True
            g = signed_gap(track, s_ego, o.s)
            if g < 0 and abs(g) < track.length / 4 and not passed[i] and v > float(getattr(o, "speed", 0.0)):
                passed[i] = True
            elif g > 0.5:
                passed[i] = False
        det = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], 18.0, half_fov, occlude=True) for o in opps]
        seen_age = 0.0 if any(det) else min(seen_age + 0.05, 5.0)
        for i, o in enumerate(opps):
            if det[i]:
                trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
            else:
                blind_age[i] += 0.05
        vis = [o for i, o in enumerate(opps) if det[i]]
        vis_gaps = [signed_gap(track, s_ego, o.s) for i, o in enumerate(opps) if det[i]]
        fastest, fld = field_summary(track, s_ego, vis, vis_gaps)
        m.set_obstacles(opponent_keepouts(track, opps, det, last_s, blind_age, [trk[i].speed for i in range(N)]))
        fi = opps.index(fastest) if fastest is not None else None
        fk = kinds[fi] if fi is not None else "equal"
        gap_f = signed_gap(track, s_ego, opps[fi].s) if fi is not None else 99.0
        r_shaped = race_reward(fk, r, False, contact, v, float(getattr(fastest, "speed", v)), gap_f,
                               alpha_r=ar, beta=b, off=off, edge_dist=edge, opp_pace=FAIR_PACE[fk])
        for k, val in zip(["S", "EX", "EY", "V", "LATOFF", "SLIP", "AY", "YAWRATE", "R", "DET"],
                          [s_ego, ex, ey, v, lat, ar, v * float(P._x[5]), float(P._x[5]), r_shaped,
                           1.0 if any(det) else 0.0]):
            L[k].append(val)
        L["THETA"].append(np.exp(np.asarray(theta, float))); L["U"].append(np.asarray(u, float))
        L["OPP"].append([(float(o.keepout()[0]), float(o.keepout()[1]), bool(det[i])) for i, o in enumerate(opps)])
        if off or tr or contact:                                   # recover to the closest raceline point
            crashes.append(len(L["S"]) - 1)
            P.reset(s0=track.project(ex, ey), v0=1.0); m.reset()
        feat, fastest, fi, det = observe()
        theta = pol.step(feat); u = m.value(P.state_dyn(), theta)["u0"]; v_prev = v
    arr = {k: np.asarray(L[k], float) for k in L if k not in ("THETA", "U", "OPP")}
    arr["THETA"] = np.asarray(L["THETA"], float); arr["U"] = np.asarray(L["U"], float)
    arr["OPP"] = np.asarray(L["OPP"], float); arr["CRASH"] = np.asarray(crashes, int)
    arr["LO"] = lo_lin; arr["HI"] = hi_lin; arr["AX"] = np.gradient(arr["V"], 0.05)
    arr["meta"] = np.asarray([N, phys, len(crashes)], float)
    return arr


# ----------------------------------------------------------------------------- figures
def _crash_spans(d):
    T = len(d["S"])
    return [c * 0.05 for c in d["CRASH"] if 0 <= c < T]      # crash times (s)


def fig_dashboard(tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(VOUT / f"telemetry_{tag}.npz", allow_pickle=True)
    s = np.arange(len(d["S"])) * 0.05      # race TIME (s): monotonic, so successive laps do not overlay
    TH = d["THETA"]; U = d["U"]; LO, HI = d["LO"], d["HI"]
    cx = _crash_spans(d)
    fig = plt.figure(figsize=(13, 14)); gs = fig.add_gridspec(8, 1, hspace=0.45)
    axw = [fig.add_subplot(gs[r]) for r in range(3)]
    names = ["q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v", "d_bound"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for gi, (gname, group) in enumerate(GROUPS):
        ax = axw[gi]
        for j, n in enumerate(group):
            i = names.index(n); c = palette[j]
            lo, hi = LO[i], HI[i]
            ax.axhspan(lo, hi, color=c, alpha=0.06)
            ax.axhline(lo, color=c, lw=0.7, ls=":"); ax.axhline(hi, color=c, lw=0.7, ls=":")
            pinned = (np.ptp(TH[:, i]) < 0.02 * max(hi - lo, 1e-9))
            ax.plot(s, TH[:, i], color=c, lw=1.5,
                    label=f"{n}  [{lo:.2f},{hi:.2f}]{'  PINNED' if pinned else ''}")
        ax.set_ylabel(f"{gname}\nweights"); ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5))
        ax.grid(alpha=0.2)
    axw[0].set_title(f"[{tag}] weights vs race time -- each inside its min/max rails (shaded); "
                     f"PINNED = sitting on a rail, else the learner is moving it", fontsize=10, loc="left")
    # states
    aS = fig.add_subplot(gs[3]); aS.plot(s, d["V"], color="0.2", lw=1.4, label="speed v (m/s)")
    aS.plot(s, d["AX"], color="#8c564b", lw=1.0, label="long. accel a_x (m/s$^2$)")
    aS.plot(s, d["AY"], color="#e377c2", lw=1.0, label="lat. accel a_y (m/s$^2$)")
    aS.set_ylabel("speed /\naccel"); aS.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5)); aS.grid(alpha=0.2)
    aL = fig.add_subplot(gs[4]); aL.plot(s, d["LATOFF"], color="#17becf", lw=1.2, label="lateral offset (m)")
    aL.axhline(0, color="0.7", lw=0.6); aL.set_ylabel("line\noffset"); aL.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5)); aL.grid(alpha=0.2)
    aP = fig.add_subplot(gs[5]); aP.plot(s, np.degrees(d["SLIP"]), color="#d62728", lw=1.2, label="lateral slip angle (deg)")
    aP.plot(s, np.degrees(d["YAWRATE"]), color="#9467bd", lw=0.9, label="yaw rate (deg/s)")
    aP.set_ylabel("slip /\nyaw"); aP.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5)); aP.grid(alpha=0.2)
    # inputs (all three controls on one panel)
    aU = fig.add_subplot(gs[6]); ulab = ["u0 steering cmd", "u1 accel cmd", "u2 progress rate"]
    ucol = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(min(3, U.shape[1])):
        aU.plot(s, U[:, k], color=ucol[k], lw=1.0, label=ulab[k])
    aU.set_ylabel("inputs u"); aU.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5)); aU.grid(alpha=0.2)
    aR = fig.add_subplot(gs[7])
    aR.plot(s, d["R"], color="0.2", lw=1.0, label="reward / tick")
    aR.set_ylabel("reward"); aR.set_xlabel("race time (s)")
    aR.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5)); aR.grid(alpha=0.2)
    for ax in fig.axes:
        for xc in cx:
            ax.axvline(xc, color="red", lw=0.8, alpha=0.5)
    if cx:
        axw[0].plot([], [], color="red", lw=0.8, alpha=0.5, label="crash+recover")
    fig.savefig(VOUT / f"telemetry_{tag}.png", dpi=120, bbox_inches="tight")
    fig.savefig(VOUT / f"telemetry_{tag}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {VOUT}/telemetry_{tag}.{{png,pdf}}  ({len(cx)} crash-recoveries)")


def tikz_weights(tag):
    d = np.load(VOUT / f"telemetry_{tag}.npz", allow_pickle=True)
    s = np.arange(len(d["S"])) * 0.05; TH = d["THETA"]; LO, HI = d["LO"], d["HI"]
    names = ["q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v", "d_bound"]
    st = max(1, len(s) // 400)
    lines = [r"% weights vs lap distance; each normalised into its [min,max] rail -> 0=floor, 1=ceiling.",
             r"\begin{tikzpicture}", r"\begin{axis}[width=12cm,height=7cm,xlabel={race time (s)},",
             r"  ylabel={weight position in its rail (0=min, 1=max)},ymin=-0.05,ymax=1.05,",
             r"  legend style={font=\tiny,at={(1.02,1)},anchor=north west},legend cell align=left]"]
    for i, n in enumerate(names):
        rng = max(HI[i] - LO[i], 1e-9); yv = (TH[:, i] - LO[i]) / rng
        pts = " ".join(f"({s[k]:.2f},{yv[k]:.3f})" for k in range(0, len(s), st))
        lines.append(rf"\addplot+[mark=none,thick] coordinates {{{pts}}};")
        lines.append(rf"\addlegendentry{{{n.replace('_','-')}}}")
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    (VOUT / f"weights_{tag}.tikz").write_text("\n".join(lines))
    print(f"wrote {VOUT}/weights_{tag}.tikz")


def make_gif(tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    d = np.load(VOUT / f"telemetry_{tag}.npz", allow_pickle=True)
    EX, EY, OPP, TH, V, R = d["EX"], d["EY"], d["OPP"], d["THETA"], d["V"], d["R"]
    LO, HI = d["LO"], d["HI"]; s = d["S"] - d["S"][0]
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    T = len(EX); N = OPP.shape[1] if OPP.ndim == 3 and OPP.shape[0] else 0
    stride = max(1, T // 200)
    names = ["q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v", "d_bound"]
    norm = (TH - LO) / np.maximum(HI - LO, 1e-9)                 # each weight as 0..1 in its rail
    fig = plt.figure(figsize=(12, 6))
    axT = fig.add_subplot(1, 2, 1); axT.set_aspect("equal"); axT.axis("off")
    axT.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.8); axT.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.8)
    axT.plot(bnd["ref_x"], bnd["ref_y"], color="0.85", lw=0.6, ls=":")
    (ego,) = axT.plot([], [], "o", color="#1f77b4", ms=10); (trail,) = axT.plot([], [], "-", color="#1f77b4", lw=1.0, alpha=0.5)
    opp_s = [axT.plot([], [], "s", color="#d62728", ms=8)[0] for _ in range(N)]
    opp_b = [axT.plot([], [], "s", mfc="none", mec="#d62728", ms=8)[0] for _ in range(N)]
    axB = fig.add_subplot(1, 2, 2)
    bars = axB.bar(range(9), norm[0], color=["#1f77b4"] * 3 + ["#ff7f0e"] * 3 + ["#2ca02c"] * 3)
    axB.set_ylim(0, 1); axB.set_xticks(range(9)); axB.set_xticklabels([n.replace("_", "-") for n in names], rotation=45, fontsize=8)
    axB.set_ylabel("weight position in its rail (0=min, 1=max)"); axB.axhline(0, color="0.5", lw=0.6); axB.axhline(1, color="0.5", lw=0.6)
    ttl = axB.set_title("")

    def frame(k):
        i = min(k * stride, T - 1)
        ego.set_data([EX[i]], [EY[i]]); trail.set_data(EX[:i + 1], EY[:i + 1])
        for j in range(N):
            ox, oy, det = OPP[i, j]
            (opp_s[j] if det > 0.5 else opp_b[j]).set_data([ox], [oy])
            (opp_b[j] if det > 0.5 else opp_s[j]).set_data([], [])
        for j in range(9):
            bars[j].set_height(norm[i, j])
        ttl.set_text(f"[{tag}] s={s[i]:5.1f}m  v={V[i]:.2f}m/s  reward={R[i]:+.1f}   (filled=seen, hollow=blind)")
        return [ego, trail, *opp_s, *opp_b, *bars, ttl]

    ani = animation.FuncAnimation(fig, frame, frames=T // stride, blit=False)
    ani.save(VOUT / f"telemetry_{tag}.gif", writer=animation.PillowWriter(fps=12), dpi=80)
    plt.close(fig); print(f"wrote {VOUT}/telemetry_{tag}.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", default=None); ap.add_argument("--loose", action="store_true")
    ap.add_argument("--physical", action="store_true", help="net has the r_a corridor-floor band (nets_physical)")
    ap.add_argument("--arm", default="ltc"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=3); ap.add_argument("--phys", type=int, default=1)
    ap.add_argument("--tag", default="adaptive"); ap.add_argument("--from-saved", action="store_true")
    a = ap.parse_args()
    if not a.from_saved:
        d = rich_race(a.arm, a.seed, a.N, a.phys, a.net_dir, a.loose or a.physical, ra_on=a.physical)
        if d is None:
            print(f"no net {a.arm}/{a.seed} in {a.net_dir}"); return
        np.savez(VOUT / f"telemetry_{a.tag}.npz", **d)
        print(f"raced: N={a.N} phys={a.phys}  {int(d['meta'][2])} crash-recoveries  {len(d['S'])} ticks")
    fig_dashboard(a.tag); tikz_weights(a.tag); make_gif(a.tag)


if __name__ == "__main__":
    main()
