"""The ONLINE method in action (NOT the fixed net): PolicyTuner = explore + RTRL weight update EVERY tick +
keep-best. Runs the deploy LTC net ONLINE for N=2 and N=3, logs the 9 MPCC weights as they LEARN, the RTRL
signal (TD delta), d_obs (keep-distance) + q_v (progress), and the lateral SPACE (corridor room + nearest gap)
-- so you can see (a) which weights the learning MOVES vs which the safety bands CLAMP, and (b) whether the
car backs off / follows when there is no space to pass. Mirrors experiments/race_multi_train.run's online loop.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/paper_online_learning.py        # run N=2,3 + plot
    python3 tools/paper_online_learning.py --from-saved                                   # re-plot from logs
"""
import sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import (race_features, field_summary, race_reward, FAIR_PACE, signed_gap, measure_pace,
                                    MAX_OPP, KEEPOUT_R, CONTACT_R, PACE_KINDS, LR_VEH, A_LAT_RACE, CORRIDOR_KW,
                                    FOV_DEG, _visible, opponent_keepouts)
from experiments.race_multi_eval import _load_pol, OUT
OUTO = OUT / "online_learning"; OUTO.mkdir(parents=True, exist_ok=True)
RACE_KINDS = ("slower", "equal", "faster")
WEIGHTS = ["q_c", "q_l", "q_v", "r_d", "r_a", "r_dv", "d_obs", "k_v", "d_bound"]
BANDED = {"q_c", "k_v", "r_a"}                         # clamped by the deterministic safety bands
FREE = ["q_v", "d_obs", "q_l", "r_d", "r_dv", "d_bound"]   # the RTRL/online learning can move these


def _mpcc(name):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    return AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                      max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=name)


def run_online(seed, N, phys, ego_pace, steps=2600):
    from mpcc_tuning.ltc import PolicyTuner
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    track = Track.icra_t2_smooth(); L = track.length; half = np.radians(FOV_DEG) / 2.0
    m = _mpcc(f"ol_{N}_{seed}"); pol = _load_pol("ltc", seed)          # deploy net + bands
    tuner = PolicyTuner(m, pol, alpha=3e-3, explore=0.06, delta_clip=1.0, seed=seed, trust_region=0.01,
                        theta_prior=0.15, theta_explore=0.15, entropy=0.03, critic="return",
                        alpha_delta=0.01, trust_region_delta=0.03)
    kinds = [RACE_KINDS[(phys + i) % 3] for i in range(N)]
    s0 = (phys % 4) * L / 4.0; v0 = 1.3 + 0.1 * (phys % 3)
    gaps0 = [3.0 + 4.0 * i + 1.5 * phys for i in range(N)]
    opps = [RacelineOpponent(track, s0=(s0 + gaps0[i]) % L, pace=FAIR_PACE[kinds[i]] * ego_pace, offset=0.0,
                             radius=KEEPOUT_R, a_lat=2.5) for i in range(N)]
    trk = [ObstacleTracker(dt=0.05) for _ in range(N)]
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps; P.reset(s0=s0, v0=v0); m.reset()
    for o in opps:
        o.reset()
    tuner.reset(); tuner.set_class(PACE_KINDS.index(kinds[int(np.argmax([o.pace for o in opps]))]))
    last_s = [None] * N; blind_age = [0.0] * N; prev_g = [None] * N; passed = [False] * N
    passes = 0; contact = off = tr = False

    def slip():
        b = float(P._x[6]) if P._x.size > 6 else 0.0; r = float(P._x[5]); v = float(P._x[3])
        ar = -np.arctan2(v * np.sin(b) - LR_VEH * r, v * np.cos(b)) if v * np.cos(b) > 0.05 else 0.0
        return ar, b, r

    def observe(state5, first=False):
        ar, b, r_ = slip(); ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey)
        det = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], 18.0, half, occlude=True) for o in opps]
        gaps = [signed_gap(track, s_ego, o.s) for o in opps]
        for i, o in enumerate(opps):
            if det[i]:
                trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
            else:
                blind_age[i] += 0.05
        vis = [o for i, o in enumerate(opps) if det[i]]
        fastest, fld = field_summary(track, s_ego, vis, [gaps[i] for i in range(N) if det[i]])
        m.set_obstacles(opponent_keepouts(track, opps, det, last_s, blind_age, [trk[i].speed for i in range(N)]))
        fi = opps.index(fastest) if fastest is not None else None
        gr = ((gaps[fi] - prev_g[fi]) / 0.05) if (fi is not None and prev_g[fi] is not None) else (0.0 if first else None)
        feat = race_features(track, state5, ([fastest] if fastest else []),
                             opp_speed_est=(trk[fi].speed if fi is not None else None), slip=(ar, b, r_),
                             gap_rate=gr, detected=1.0 if vis else 0.0, seen_age=(0.0 if any(det) else 1.0), field=fld)
        pol.set_crowd(sum(ls is not None for ls in last_s))
        return feat, det, gaps, fi, fastest, kinds, (ar, b, r_)

    feat, det, gaps, fi, fastest, kinds, sl = observe(P.state5(), first=True)
    for i in range(N):
        prev_g[i] = gaps[i]
    theta, u = tuner.act(feat, P.state_dyn())
    rec = {k: [] for k in ("EX", "EY", "S", "V", "GAP", "SPACE", "WIDTH", "DET", "DELTA")}
    recW = []; recO = []
    for _ in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey); v = float(P._x[3])
        n_now = 0
        for i, o in enumerate(opps):
            o.step(0.05, ego=(ex, ey, v)); ox, oy, _ = o.keepout()
            if np.hypot(ex - ox, ey - oy) < CONTACT_R:
                contact = True
            g = signed_gap(track, s_ego, o.s)
            if g < 0 and abs(g) < L / 4 and not passed[i] and v > float(getattr(o, "speed", 0.0)):
                passed[i] = True; passes += 1; n_now += 1
            elif g > 0.5:
                passed[i] = False
        lat = float(track.lateral(ex, ey)); wl, wr = track.width(track.wrap(s_ego))
        edge = min(float(wr) - 0.12 - lat, lat + float(wl) - 0.12)
        gmin = min((abs(gg) for gg in gaps), default=99.0)
        rec["EX"].append(ex); rec["EY"].append(ey); rec["S"].append(s_ego); rec["V"].append(v)
        rec["GAP"].append(gmin); rec["SPACE"].append(edge); rec["WIDTH"].append(float(wl + wr))
        rec["DET"].append(1.0 if any(det) else 0.0); rec["DELTA"].append(float(getattr(tuner, "stats", {}).get("delta", 0.0)))
        recW.append(np.exp(np.asarray(theta, float)))
        recO.append([(float(o.keepout()[0]), float(o.keepout()[1]), bool(det[i])) for i, o in enumerate(opps)])
        feat, det, gaps, fi, fastest, kinds, (ar, b, r_) = observe(s5n)
        fk = kinds[fi] if fi is not None else "equal"
        r_shaped = race_reward(fk, r, n_now > 0, contact, v, float(getattr(fastest, "speed", v)),
                               gaps[fi] if fi is not None else 99.0, alpha_r=ar, beta=b, off=off,
                               edge_dist=edge, opp_pace=FAIR_PACE[fk])
        out = tuner.learn(r_shaped, P.state_dyn(), feat, off or contact)
        for i in range(N):
            prev_g[i] = gaps[i]
        if out[0] is None or off or tr or contact:
            break
        theta, u = out
    arr = {k: np.asarray(v, float) for k, v in rec.items()}
    arr["W"] = np.asarray(recW, float); arr["OPP"] = np.asarray(recO, float) if recO else np.zeros((0, N, 3))
    res = "CLEAN" if (not off and not contact) else ("OFF" if off else "CONTACT")
    print(f"  ONLINE N={N} phys={phys}: {res} passes {passes} reverts {getattr(tuner,'reverts',0)} steps {len(arr['EX'])}", flush=True)
    np.savez(OUTO / f"online_N{N}.npz", N=N, phys=phys, passes=passes, result=res,
             reverts=int(getattr(tuner, "reverts", 0)), **arr)
    return arr


def fig_online(N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(OUTO / f"online_N{N}.npz", allow_pickle=True)
    S, V, W, GAP, SPACE = d["S"], d["V"], d["W"], d["GAP"], d["SPACE"]
    s = S - S[0]
    idx = {w: i for i, w in enumerate(WEIGHTS)}
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True, height_ratios=[2, 2, 1, 1])
    # free (online-learned) weights, each normalised to its own range so the LEARNING is visible
    for w in FREE:
        x = W[:, idx[w]]; rng = x.max() - x.min()
        ax[0].plot(s, (x - x.min()) / (rng if rng > 1e-9 else 1), lw=1.4, label=f"{w}")
    ax[0].set_ylabel("free weights\n(norm.)"); ax[0].legend(ncol=3, fontsize=8, loc="upper right"); ax[0].grid(alpha=0.25)
    ax[0].set_title(f"ONLINE method (RTRL + keep-best), N={N}: FREE weights the learning MOVES "
                    f"(d_obs = keep-distance, q_v = progress)  [{str(d['result'])}, reverts {int(d['reverts'])}]")
    # banded weights (actual values) -- show they are clamped
    for w in ["q_c", "k_v", "r_a"]:
        ax[1].plot(s, W[:, idx[w]], lw=1.5, label=f"{w} (band-held)")
    ax[1].set_ylabel("banded weights\n(actual)"); ax[1].legend(fontsize=8, loc="upper right"); ax[1].grid(alpha=0.25)
    ax[1].text(0.01, 0.9, "these are CLAMPED by the safety bands -> the learning cannot move them",
               transform=ax[1].transAxes, fontsize=8, color="0.3", va="top")
    # d_obs vs q_v -- the follow-vs-overtake weights -- against the nearest gap + space
    ax[2].plot(s, W[:, idx["d_obs"]], color="#d62728", lw=1.5, label="d_obs (avoid)")
    ax[2].plot(s, W[:, idx["q_v"]], color="#9467bd", lw=1.5, label="q_v (progress)")
    ax[2].set_ylabel("follow\nweights"); ax[2].legend(fontsize=8, loc="upper right"); ax[2].grid(alpha=0.25)
    ax[3].plot(s, GAP, color="0.3", lw=1.3, label="nearest |gap| (m)")
    ax[3].plot(s, SPACE, color="#2ca02c", lw=1.3, label="lateral space (m)")
    ax[3].axhline(0.0, color="0.7", lw=0.6)
    ax[3].set_ylabel("space"); ax[3].set_xlabel("lap distance s (m)"); ax[3].legend(fontsize=8, loc="upper right"); ax[3].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTO / f"online_learning_N{N}.png", dpi=130); fig.savefig(OUTO / f"online_learning_N{N}.pdf"); plt.close(fig)
    print(f"wrote {OUTO}/online_learning_N{N}.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from-saved", action="store_true"); a = ap.parse_args()
    if not a.from_saved:
        track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
        mp = _mpcc("ol_pace"); ego_pace = measure_pace(mp, track, th0, 1500)
        print(f"ego solo pace {ego_pace:.2f} m/s ; ONLINE method (explore+RTRL+keep-best) N=2 and N=3", flush=True)
        for N in (2, 3):
            run_online(0, N, 1, ego_pace)
    for N in (2, 3):
        if (OUTO / f"online_N{N}.npz").exists():
            fig_online(N)


if __name__ == "__main__":
    main()
