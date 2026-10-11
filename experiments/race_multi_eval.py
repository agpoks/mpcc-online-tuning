"""Phase-A multi-car EVAL (branch race-multicar): the trained keep-best nets
(results/race_multicar/nets/race_<arm>_<seed>.npz) run FIXED (explore off, deterministic forward pass --
`pol.step`, no RTRL update) vs CONTROLLED fields of N=1/2/3 opponents. This is the count-FLEXIBILITY test
(ONE net, any N, no retraining) + the LTC-vs-MLP comparison. Real 270 deg + occlusion sensor; the MPCC
avoids all DETECTED opponents. Metrics: clean %, passes, contact %, laps.

The deployed method adds online keep-best adaptation ON TOP of this net (phase B); this eval isolates what
the net learned, deterministically + reproducibly. Fields are fixed per (N, phys) -- opponents differ
PHYSICALLY (start s0 / gaps / pace mix), not by an RNG seed (the plant is deterministic).

Artifacts (reproducible): data npz (with scalars) + eval csv + flexibility png/pdf + tikz + a race gif.
Regen plots/csv/tikz from saved data only with --from-saved.
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_multi_eval.py --jobs 4
    python3 experiments/race_multi_eval.py --from-saved
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from experiments.race_mode import (race_features, field_summary, FAIR_PACE, signed_gap, measure_pace,
                                    N_RACE_FEATURES_MULTI, MAX_OPP, KEEPOUT_R, CONTACT_R, PACE_KINDS,
                                    LR_VEH, A_LAT_RACE, CORRIDOR_KW, FOV_DEG, _visible, opponent_keepouts,
                                    KV_FLOOR, KV_FLOOR_CLASSES, KV_CEIL, KV_CEIL_CLASSES,
                                    RA_FLOOR, RA_FLOOR_CLASSES, QC_FLOOR, QC_FLOOR_CLASSES,
                                    QC_CROWD_LO, QC_CROWD_HI, QC_CROWD_N)
OUT = ROOT / "results" / "race_multicar"
RACE_KINDS = ("slower", "equal", "faster")
NSET = (1, 2, 3)            # opponent counts to test the ONE net against
PHYS = (0, 1, 2)           # physical start variations per (arm, net-seed, N)
ARMS = ("ltc", "mlp")
SEEDS = (0, 1)


def _load_pol(arm, seed, net_dir=None, qc_floor_on=True, crowd_on=True, loose=False, ra_floor_on=False):
    from mpcc_tuning.ltc import LTCCell, MLPCell, WeightPolicy
    f = (Path(net_dir) if net_dir else OUT / "nets") / f"race_{arm}_{seed}.npz"
    if not f.exists():
        return None
    d = np.load(f); Cell = LTCCell if arm == "ltc" else MLPCell
    cell = Cell(N_RACE_FEATURES_MULTI, int(d["n_hidden"]), seed=seed); ncls = int(d["D_class"].shape[0])
    if loose:
        # ADAPTIVE net: the q_c/k_v physical rails are baked into the SAVED box (d["lo"]/d["hi"]). NO
        # behavioural band clamps -- reconstruct the exact training-time policy so the eval sees the learned
        # weights. ra_floor_on=True for the PHYSICAL-rails net: r_a's corridor floor is a band (it cannot be
        # a box edge, since r_a's baseline sits below it), so it must be re-applied here.
        rfl, rflc = (RA_FLOOR, tuple(range(ncls))) if ra_floor_on else (0.0, ())
        pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls,
                           delta_log=float(d["delta_log"]), kv_floor=0.0, kv_floor_classes=(), kv_ceil=0.0,
                           kv_ceil_classes=(), ra_floor=rfl, ra_floor_classes=rflc, qc_floor=0.0,
                           qc_floor_classes=(), qc_crowd_lo=0.0, qc_crowd_hi=0.0, qc_crowd_n=QC_CROWD_N)
    else:
        qcf, qcc = (QC_FLOOR, QC_FLOOR_CLASSES) if qc_floor_on else (0.0, ())  # off for nets trained WITHOUT the floor
        qlo, qhi = (QC_CROWD_LO, QC_CROWD_HI) if (qc_floor_on and crowd_on) else (0.0, 0.0)  # crowd band: LTC + opt-in
        pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls, delta_log=float(d["delta_log"]),
                           kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES, kv_ceil=KV_CEIL,
                           kv_ceil_classes=KV_CEIL_CLASSES, ra_floor=RA_FLOOR, ra_floor_classes=RA_FLOOR_CLASSES,
                           qc_floor=qcf, qc_floor_classes=qcc, qc_crowd_lo=qlo, qc_crowd_hi=qhi, qc_crowd_n=QC_CROWD_N)
    pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.D_class[...] = d["D_class"]
    return pol


def _slip(P):
    b = float(P._x[6]) if P._x.size > 6 else 0.0; r = float(P._x[5]); v = float(P._x[3])
    ar = -np.arctan2(v * np.sin(b) - LR_VEH * r, v * np.cos(b)) if v * np.cos(b) > 0.05 else 0.0
    return ar, b, r


def run_eval(m, ego_pace, pol, N, phys, detect_range=18.0, steps=2600, fov_deg=FOV_DEG, occlude=True, log=False,
             keepout="gentle", pred_r=None):
    """One deterministic race vs a fixed field of N opponents (explore off). Returns (metrics, log|None).
    keepout: 'gentle' = the bounded predicted keep-out (opponent_keepouts); 'none' = drop blind opponents
    (the pre-fix behaviour) -- for the net-vs-constraint ablation. pred_r overrides the predicted keep-out
    radius (None = race_mode.PRED_R 0.28); a bigger one adds clearance from occluded cars (the N=3 clip)."""
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    track = Track.icra_t2_smooth(); half = np.radians(fov_deg) / 2.0; L = track.length
    kinds = [RACE_KINDS[(phys + i) % 3] for i in range(N)]            # deterministic mixed field
    s0 = (phys % 4) * L / 4.0; v0 = 1.3 + 0.1 * (phys % 3)
    gaps0 = [3.0 + 4.0 * i + 1.5 * phys for i in range(N)]
    opps = [RacelineOpponent(track, s0=(s0 + gaps0[i]) % L, pace=FAIR_PACE[kinds[i]] * ego_pace, offset=0.0,
                             radius=KEEPOUT_R, a_lat=2.5) for i in range(N)]
    trk = [ObstacleTracker(dt=0.05) for _ in range(N)]
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps; P.reset(s0=s0, v0=v0); m.reset()
    for o in opps:
        o.reset()
    pol.reset()
    fast0 = int(np.argmax([o.pace for o in opps])); pol.cls = PACE_KINDS.index(kinds[fast0])
    base = float(P.state5()[4]); seen_age = 0.0; prev_g = [None] * N
    last_s = [None] * N; blind_age = [0.0] * N        # per-opponent last-seen arc-length + time-since-seen
    passed = [False] * N; passes = 0; contact = False; off = tr = False
    LOGK = ["EX", "EY", "S", "V", "DET", "YAW", "SLIP", "YAWRATE"]
    rec = {k: [] for k in LOGK}; recO = []; recTH = []   # recO: (ox,oy,det)/opp; recTH: exp(theta) MPCC weights

    def emit(state5, s_ego, first=False):
        ar, b, r_ = _slip(P)
        ex, ey = float(P._x[0]), float(P._x[1])
        det = [_visible(ex, ey, float(P._x[2]), *o.keepout()[:2], detect_range, half, occlude=occlude) for o in opps]
        gaps = [signed_gap(track, s_ego, o.s) for o in opps]
        for i, o in enumerate(opps):
            if det[i]:
                trk[i].update(o.pose()[:2]); last_s[i] = o.s; blind_age[i] = 0.0
            else:
                blind_age[i] += 0.05
        vis = [o for i, o in enumerate(opps) if det[i]]
        vis_gaps = [gaps[i] for i in range(N) if det[i]]
        vis_clos = [((gaps[i] - prev_g[i]) if prev_g[i] is not None else 0.0) for i in range(N) if det[i]]
        fastest, fld = field_summary(track, s_ego, vis, vis_gaps, vis_clos)
        # avoid DETECTED opponents + a BOUNDED last-seen prediction for the just-occluded ones (<= MAX_OPP)
        if keepout == "none":
            m.set_obstacles([o.keepout() for o in vis])          # pre-fix: drop blind opponents
        else:
            m.set_obstacles(opponent_keepouts(track, opps, det, last_s, blind_age, [trk[i].speed for i in range(N)], r=pred_r))
        fi = opps.index(fastest) if fastest is not None else None
        gr = ((gaps[fi] - prev_g[fi]) / 0.05) if (fi is not None and prev_g[fi] is not None) else (0.0 if first else None)
        feat = race_features(track, state5, ([fastest] if fastest else []),
                             opp_speed_est=(trk[fi].speed if fi is not None else None),
                             slip=(ar, b, r_), gap_rate=gr, detected=1.0 if vis else 0.0,
                             seen_age=seen_age, field=fld)
        pol.set_crowd(sum(ls is not None for ls in last_s))      # memory-based crowd estimate (distinct seen)
        return np.asarray(pol.step(feat), float), det, gaps

    ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey)
    theta, det, gaps = emit(P.state5(), s_ego, first=True); prev_g = list(gaps)
    u = m.value(P.state_dyn(), theta)["u0"]
    for _step in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); s_ego = track.project(ex, ey); v = float(P._x[3])
        for i, o in enumerate(opps):
            o.step(0.05, ego=(ex, ey, v)); ox, oy, _ = o.keepout()
            if np.hypot(ex - ox, ey - oy) < CONTACT_R:
                contact = True
            g = signed_gap(track, s_ego, o.s)
            if g < 0 and abs(g) < L / 4 and not passed[i] and v > float(getattr(o, "speed", 0.0)):
                passed[i] = True; passes += 1
            elif g > 0.5:
                passed[i] = False
        seen_age = 0.0 if any(det) else min(seen_age + 0.05, 5.0)
        if log:
            _ar, _b, _rr = _slip(P)
            for k, val in zip(LOGK, [ex, ey, s_ego, v, 1.0 if any(det) else 0.0, float(P._x[2]), _ar, float(P._x[5])]):
                rec[k].append(val)
            recO.append([(float(o.keepout()[0]), float(o.keepout()[1]), bool(det[i])) for i, o in enumerate(opps)])
            recTH.append(np.exp(np.asarray(theta, float)))
        theta, det, gaps = emit(s5n, s_ego); prev_g = list(gaps)
        u = m.value(P.state_dyn(), theta)["u0"]
        if off or tr or contact:
            break
    laps = (float(P.state5()[4]) - base) / L
    met = dict(N=N, phys=phys, clean=bool((not off) and (not contact)), passes=int(passes),
               contact=bool(contact), off=bool(off), laps=round(float(laps), 3), kinds=",".join(kinds))
    lg = None
    if log:
        arr = {k: np.asarray(rec[k], float) for k in LOGK}
        arr["OPP"] = np.asarray(recO, dtype=float) if recO else np.zeros((0, N, 3))  # (T,N,3): ox,oy,det
        arr["THETA"] = np.asarray(recTH, float) if recTH else np.zeros((0, 9))        # (T,9): exp(theta) weights
        lg = arr
    return met, lg


def one(job):
    arm, seed, ego_pace, steps = job
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                   max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name=f"ev_{arm}_{seed}")
    pol = _load_pol(arm, seed, qc_floor_on=(arm == "ltc"))   # q_c floor is LTC-specific (it breaks MLP)
    if pol is None:
        return dict(arm=arm, seed=seed, rows=[], gif=None)
    rows = []; gif = None
    for N in NSET:
        for phys in PHYS:
            want_log = (N == 3 and phys == 0 and seed == 0)       # one representative race per arm for the gif
            met, lg = run_eval(m, ego_pace, pol, N, phys, steps=steps, log=want_log)
            met.update(arm=arm, seed=seed); rows.append(met)
            if want_log:
                gif = lg
            print(f"  [{arm} s{seed}] N={N} phys={phys} ({met['kinds']}): "
                  f"{'CLEAN' if met['clean'] else ('OFF' if met['off'] else 'CONTACT')} "
                  f"passes {met['passes']} laps {met['laps']:.2f}", flush=True)
    return dict(arm=arm, seed=seed, rows=rows, gif=gif)


# ---------------------------------------------------------------------------- artifacts
def save_rows(all_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["arm", "seed", "N", "phys", "clean", "passes", "contact", "off", "laps"]
    cols = {k: np.asarray([r[k] for r in all_rows]) for k in keys}
    cols["kinds"] = np.asarray([r["kinds"] for r in all_rows])
    np.savez(OUT / "multi_eval_rows.npz", **cols)
    return cols


def write_csv(all_rows):
    import csv
    with open(OUT / "multi_eval.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed", "N", "phys", "kinds", "clean", "passes", "contact", "off", "laps"])
        for r in all_rows:
            w.writerow([r["arm"], r["seed"], r["N"], r["phys"], r["kinds"], int(r["clean"]), r["passes"],
                        int(r["contact"]), int(r["off"]), r["laps"]])
    # aggregated-by-(arm,N) summary
    with open(OUT / "multi_eval_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "N", "runs", "clean_pct", "mean_passes", "contact_pct", "off_pct", "mean_laps"])
        for arm in ARMS:
            for N in NSET:
                rs = [r for r in all_rows if r["arm"] == arm and r["N"] == N]
                if not rs:
                    continue
                n = len(rs)
                w.writerow([arm, N, n, round(100 * np.mean([r["clean"] for r in rs]), 1),
                            round(float(np.mean([r["passes"] for r in rs])), 2),
                            round(100 * np.mean([r["contact"] for r in rs]), 1),
                            round(100 * np.mean([r["off"] for r in rs]), 1),
                            round(float(np.mean([r["laps"] for r in rs])), 2)])
    return OUT / "multi_eval_summary.csv"


def _agg(all_rows):
    """-> {arm: {N: dict(clean_pct, passes, contact_pct, off_pct, laps, n)}}"""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for N in NSET:
            rs = [r for r in all_rows if r["arm"] == arm and r["N"] == N]
            if rs:
                out[arm][N] = dict(clean=100 * np.mean([r["clean"] for r in rs]),
                                   passes=float(np.mean([r["passes"] for r in rs])),
                                   contact=100 * np.mean([r["contact"] for r in rs]),
                                   off=100 * np.mean([r["off"] for r in rs]),
                                   laps=float(np.mean([r["laps"] for r in rs])), n=len(rs))
    return out


def fig_flex(all_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ag = _agg(all_rows)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = np.arange(len(NSET)); w = 0.36
    col = {"ltc": "#1f77b4", "mlp": "#d62728"}
    for j, arm in enumerate(ARMS):
        cl = [ag[arm].get(N, {}).get("clean", np.nan) for N in NSET]
        pa = [ag[arm].get(N, {}).get("passes", np.nan) for N in NSET]
        a0.bar(x + (j - 0.5) * w, cl, w, label=arm.upper(), color=col[arm])
        a1.bar(x + (j - 0.5) * w, pa, w, label=arm.upper(), color=col[arm])
    a0.set_xticks(x); a0.set_xticklabels([f"{N} car{'s' if N > 1 else ''}" for N in NSET])
    a0.set_ylabel("clean races (%)"); a0.set_ylim(0, 105); a0.set_title("Clean rate vs field size"); a0.legend()
    a1.set_xticks(x); a1.set_xticklabels([f"{N} car{'s' if N > 1 else ''}" for N in NSET])
    a1.set_ylabel("passes per race"); a1.set_title("Overtakes vs field size"); a1.legend()
    fig.suptitle("One online net, variable field (fixed eval, 270\u00b0+occlusion sensor)")
    fig.tight_layout()
    fig.savefig(OUT / "multi_flexibility.png", dpi=130); fig.savefig(OUT / "multi_flexibility.pdf")
    plt.close(fig)
    return OUT / "multi_flexibility.png"


def make_gif(arm):
    f = OUT / f"multi_eval_gif_{arm}.npz"
    if not f.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    d = np.load(f, allow_pickle=True); EX, EY, OPP = d["EX"], d["EY"], d["OPP"]
    bnd = np.load(ROOT / "mpcc_tuning/tracks/icra_t2_smooth_boundaries.npz")
    T = len(EX); N = OPP.shape[1] if OPP.ndim == 3 and OPP.shape[0] else 0
    stride = max(1, T // 240)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.set_aspect("equal"); ax.axis("off")
    ax.plot(bnd["left_x"], bnd["left_y"], color="0.6", lw=0.8); ax.plot(bnd["right_x"], bnd["right_y"], color="0.6", lw=0.8)
    ax.plot(bnd["ref_x"], bnd["ref_y"], color="0.8", lw=0.6, ls=":")
    (ego,) = ax.plot([], [], "o", color="#1f77b4", ms=9, label="ego")
    trail, = ax.plot([], [], "-", color="#1f77b4", lw=1.2, alpha=0.6)
    opp_seen = [ax.plot([], [], "s", color="#d62728", ms=8)[0] for _ in range(N)]
    opp_blind = [ax.plot([], [], "s", mfc="none", mec="#d62728", ms=8)[0] for _ in range(N)]
    ttl = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)

    def frame(k):
        i = min(k * stride, T - 1)
        ego.set_data([EX[i]], [EY[i]]); trail.set_data(EX[:i + 1], EY[:i + 1])
        for j in range(N):
            ox, oy, det = OPP[i, j]
            if det > 0.5:
                opp_seen[j].set_data([ox], [oy]); opp_blind[j].set_data([], [])
            else:
                opp_blind[j].set_data([ox], [oy]); opp_seen[j].set_data([], [])
        ttl.set_text(f"{arm.upper()} vs {N} cars  (filled=seen, hollow=blind)  t={i*0.05:.1f}s")
        return [ego, trail, *opp_seen, *opp_blind, ttl]

    anim = animation.FuncAnimation(fig, frame, frames=T // stride + 1, interval=60, blit=True)
    gif = OUT / f"multi_race_{arm}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=16)); plt.close(fig)
    return gif


def emit_tikz(all_rows):
    ag = _agg(all_rows); TZ = OUT / "tikz"; TZ.mkdir(parents=True, exist_ok=True)

    def _coords(metric):
        return {arm: " ".join(f"({N},{ag[arm].get(N, {}).get(metric, 0):.1f})" for N in NSET) for arm in ARMS}
    cl = _coords("clean"); pa = _coords("passes")
    tex = []
    tex.append("% Phase-A multi-car flexibility: one online net vs N=1/2/3 cars (fixed eval). Regen: race_multi_eval.py --from-saved")
    tex.append("\\begin{tikzpicture}")
    tex.append("\\begin{axis}[ybar,bar width=7pt,width=7cm,height=5cm,xtick={1,2,3},xlabel={opponents},"
               "ylabel={clean races (\\%)},ymin=0,ymax=105,legend pos=south west,enlarge x limits=0.3]")
    for arm in ARMS:
        tex.append(f"\\addplot coordinates {{{cl[arm]}}}; \\addlegendentry{{{arm.upper()}}}")
    tex.append("\\end{axis}\n\\end{tikzpicture}")
    (TZ / "multi_flexibility_clean.tex").write_text("\n".join(tex) + "\n")
    tex2 = ["% passes vs field size", "\\begin{tikzpicture}",
            "\\begin{axis}[ybar,bar width=7pt,width=7cm,height=5cm,xtick={1,2,3},xlabel={opponents},"
            "ylabel={passes per race},ymin=0,legend pos=north west,enlarge x limits=0.3]"]
    for arm in ARMS:
        tex2.append(f"\\addplot coordinates {{{pa[arm]}}}; \\addlegendentry{{{arm.upper()}}}")
    tex2.append("\\end{axis}\n\\end{tikzpicture}")
    (TZ / "multi_flexibility_passes.tex").write_text("\n".join(tex2) + "\n")
    return TZ


def load_rows_path(p):
    d = np.load(p, allow_pickle=True)
    n = len(d["arm"]); rows = []
    for i in range(n):
        rows.append(dict(arm=str(d["arm"][i]), seed=int(d["seed"][i]), N=int(d["N"][i]), phys=int(d["phys"][i]),
                         clean=bool(d["clean"][i]), passes=int(d["passes"][i]), contact=bool(d["contact"][i]),
                         off=bool(d["off"][i]), laps=float(d["laps"][i]), kinds=str(d["kinds"][i])))
    return rows


def load_saved():
    return load_rows_path(OUT / "multi_eval_rows.npz")


def print_compare(pre, post):
    ap, bp = _agg(pre), _agg(post)
    print(f"\nbefore(pre-fix) -> after(contact-fix)   clean% | contact% | off%   per arm per N")
    for arm in ARMS:
        for N in NSET:
            a, b = ap[arm].get(N), bp[arm].get(N)
            if a and b:
                print(f"{arm:>4} N={N}:  clean {a['clean']:3.0f}->{b['clean']:3.0f}   "
                      f"contact {a['contact']:3.0f}->{b['contact']:3.0f}   off {a['off']:3.0f}->{b['off']:3.0f}   "
                      f"passes {a['passes']:.2f}->{b['passes']:.2f}")


def fig_compare(pre, post):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap, bp = _agg(pre), _agg(post)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(NSET)); w = 0.2
    col = {"ltc": "#1f77b4", "mlp": "#d62728"}
    for metric, ax, lbl in ((("contact"), a0, "contact races (%)"), (("clean"), a1, "clean races (%)")):
        for j, arm in enumerate(ARMS):
            pre_v = [ap[arm].get(N, {}).get(metric, np.nan) for N in NSET]
            post_v = [bp[arm].get(N, {}).get(metric, np.nan) for N in NSET]
            ax.bar(x + (2 * j - 1.5) * w, pre_v, w, color=col[arm], alpha=0.45, label=f"{arm.upper()} pre")
            ax.bar(x + (2 * j - 0.5) * w, post_v, w, color=col[arm], label=f"{arm.upper()} fix")
        ax.set_xticks(x); ax.set_xticklabels([f"{N} car{'s' if N > 1 else ''}" for N in NSET])
        ax.set_ylabel(lbl); ax.set_ylim(0, 105); ax.legend(fontsize=7, ncol=2)
    a0.set_title("Contact vs field size (lower better)"); a1.set_title("Clean vs field size (higher better)")
    fig.suptitle("Occluded-opponent keep-out: before (pale) vs after (solid)")
    fig.tight_layout()
    fig.savefig(OUT / "multi_contactfix.png", dpi=130); fig.savefig(OUT / "multi_contactfix.pdf")
    plt.close(fig)
    return OUT / "multi_contactfix.png"


def _print_table(all_rows):
    ag = _agg(all_rows)
    print(f"\n{'arm':>4} {'N':>2} {'runs':>5} {'clean%':>7} {'passes':>7} {'contact%':>9} {'off%':>6} {'laps':>6}")
    for arm in ARMS:
        for N in NSET:
            if N in ag[arm]:
                a = ag[arm][N]
                print(f"{arm:>4} {N:>2} {a['n']:>5} {a['clean']:>7.0f} {a['passes']:>7.2f} "
                      f"{a['contact']:>9.0f} {a['off']:>6.0f} {a['laps']:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers over (arm,seed) (0 = sequential)")
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--from-saved", action="store_true", help="regen csv/plot/tikz/gif from saved data only")
    ap.add_argument("--compare", nargs="?", const=str(OUT / "multi_eval_rows_prefix.npz"), default=None,
                    help="before/after table + plot vs a baseline rows npz (default: the pre-fix snapshot)")
    a = ap.parse_args()
    if a.compare:
        pre = load_rows_path(a.compare); post = load_saved()
        print_compare(pre, post); fig_compare(pre, post)
        print(f"wrote {OUT}/multi_contactfix.{{png,pdf}}"); return
    if a.from_saved:
        rows = load_saved(); _print_table(rows); write_csv(rows); fig_flex(rows); emit_tikz(rows)
        for arm in ARMS:
            make_gif(arm)
        print("regenerated csv/plot/tikz/gif from saved data"); return

    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth"); th0 = np.asarray(st.theta(), float)
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    mp = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref, discrete=True,
                    max_obstacles=MAX_OPP, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="ev_pace")
    ego_pace = measure_pace(mp, track, th0, 1500)
    print(f"ego solo pace {ego_pace:.2f} m/s ; eval vs N={NSET} x phys={PHYS}", flush=True)
    jobs = [(arm, seed, ego_pace, a.steps) for arm in ARMS for seed in SEEDS]
    results = []
    n_proc = a.jobs or 1
    if n_proc <= 1:
        for j in jobs:
            results.append(one(j))
    else:
        import multiprocessing as _mp
        with _mp.get_context("spawn").Pool(min(n_proc, len(jobs))) as pool:
            for o in pool.imap_unordered(one, jobs):
                results.append(o)
    all_rows = [r for res in results for r in res["rows"]]
    for res in results:                                           # save the representative gif-logs
        if res.get("gif") is not None:
            np.savez(OUT / f"multi_eval_gif_{res['arm']}.npz", **res["gif"])
    save_rows(all_rows); write_csv(all_rows); _print_table(all_rows); fig_flex(all_rows); emit_tikz(all_rows)
    for arm in ARMS:
        make_gif(arm)
    base = OUT / "multi_eval_rows_prefix.npz"                     # auto before/after vs the pre-fix snapshot
    if base.exists():
        print_compare(load_rows_path(base), all_rows); fig_compare(load_rows_path(base), all_rows)
    print(f"\nwrote {OUT}/multi_eval*.{{npz,csv}} + multi_flexibility.{{png,pdf}} + tikz/ + multi_race_*.gif"
          + (" + multi_contactfix.{png,pdf}" if base.exists() else ""))


if __name__ == "__main__":
    main()
