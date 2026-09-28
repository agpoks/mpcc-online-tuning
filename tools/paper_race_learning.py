"""IEEE-publication figures for the race-mode online-tuning result.

Three figures, all from the LTC arm of results/race/race_phase1_reward5.json plus a frozen re-drive
of the banked nets (results/race/nets/race_ltc_<seed>.npz):

  fig_learn_by_opponent   how the emitted weights differ by OPPONENT class (heatmap, log2 vs START)
  fig_learn_over_rounds   how the key weights LEARN over episodes, one line per opponent class
  fig_track_states        the track + our speed-coloured trajectory (faded) + opponent + our states
                          (speed, sideslip, yaw-rate) along the lap

The re-drive uses the FROZEN banked net (deterministic, exploration off) -- the policy the paper
reports -- with the same corridor / a_lat / k_v floor+ceiling as training. State logs are cached to
results/race/paper/states_ltc_<seed>_<kind>.npz so re-running the figures needs no acados re-drive
(pass --redrive to force).

    ACADOS_SOURCE_DIR=... LD_LIBRARY_PATH=... PYTHONPATH=...:. python3 tools/paper_race_learning.py [--seed 0] [--redrive]
"""
import sys, os, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.mpcc import WEIGHT_NAMES
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B

OUT = ROOT / "results/race/paper"; OUT.mkdir(parents=True, exist_ok=True)
TR = ROOT / "mpcc_tuning/tracks"
KINDS = ["static", "slower", "equal", "faster"]
KIND_LABEL = {"static": "static", "slower": "slower", "equal": "equal", "faster": "faster"}
# colourblind-safe (Wong) per opponent class
KCOL = {"static": "#999999", "slower": "#0072B2", "equal": "#009E73", "faster": "#D55E00"}
# pretty math labels for the 9 weights, in WEIGHT_NAMES order
WLAB = {"q_c": r"$q_c$", "q_l": r"$q_\ell$", "q_v": r"$q_v$", "r_d": r"$r_\delta$",
        "r_a": r"$r_a$", "r_dv": r"$r_{\dot v}$", "d_obs": r"$d_\mathrm{obs}$",
        "k_v": r"$k_v$", "d_bound": r"$d_\mathrm{bnd}$"}
KEY = ["k_v", "q_v", "q_c", "d_obs", "d_bound"]      # the interpretable ones for the over-rounds panel


def ieee_style():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "grid.linewidth": 0.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "legend.frameon": False, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name):
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{e}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# --------------------------------------------------------------------------- data
def load_runs(arm="ltc"):
    d = json.load(open(ROOT / "results/race/race_phase1_reward5.json"))
    return [r for r in d["runs"] if r["arm"] == arm], d


def start_theta():
    return np.exp(np.asarray(B.start("icra_t2_smooth").theta(), float))   # linear START weights


def weights_by_kind(runs):
    """mean emitted weights per (kind) over seeds+episodes, and START, both linear."""
    start = start_theta(); M = {}
    for k in KINDS:
        vals = [row["theta_mean"] for r in runs for row in r["rows"] if row["kind"] == k and "theta_mean" in row]
        if vals:
            M[k] = np.mean(vals, axis=0)
    return M, start


def curve_over_encounters(runs):
    """Weights vs the n-th ENCOUNTER of each opponent class (the classes cycle every 4th episode
    and land on different absolute episodes per seed, so absolute-episode averaging is pure noise).
    Aligns by encounter order within each seed, then means over seeds -> {kind: (nenc, 9)} + std."""
    start = start_theta()
    out = {}
    for k in KINDS:
        per_seed = []
        for r in runs:
            seq = [row["theta_mean"] for row in r["rows"] if row["kind"] == k and "theta_mean" in row]
            if seq:
                per_seed.append(np.array(seq))
        if not per_seed:
            out[k] = (np.zeros((0, len(WEIGHT_NAMES))), np.zeros((0, len(WEIGHT_NAMES)))); continue
        nenc = max(len(s) for s in per_seed)
        mean = np.full((nenc, len(WEIGHT_NAMES)), np.nan); std = np.full_like(mean, np.nan)
        for e in range(nenc):
            vals = [s[e] for s in per_seed if len(s) > e]
            if vals:
                mean[e] = np.mean(vals, axis=0); std[e] = np.std(vals, axis=0)
        out[k] = (mean, std)
    return out, start


def one_lap(S):
    """Index slice of the FIRST complete lap, so state-vs-s traces don't draw wrap connectors."""
    S = np.asarray(S)
    if len(S) < 3:
        return slice(0, len(S))
    L = float(Track.icra_t2_smooth().length)
    for i in range(1, len(S)):
        if S[i] - S[i - 1] < -L / 2:          # wrapped 73 -> 0
            return slice(0, i)
    return slice(0, len(S))


# --------------------------------------------------------------------------- re-drive (states)
def redrive(seed, kind, steps=1600):
    """Frozen banked-net re-drive; log trajectory + states + emitted weights. Cached to npz."""
    cache = OUT / f"states_ltc_{seed}_{kind}.npz"
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import RacelineOpponent, ObstacleTracker
    from mpcc_tuning.ltc import LTCCell, WeightPolicy
    from experiments.race_mode import (race_features, FAIR_PACE, signed_gap, N_RACE_FEATURES,
                                        KEEPOUT_R, CONTACT_R, PACE_KINDS, LR_VEH, A_LAT_RACE,
                                        CORRIDOR_KW, KV_FLOOR, KV_FLOOR_CLASSES, KV_CEIL, KV_CEIL_CLASSES)
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
                   discrete=True, max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW,
                   name=f"paper_ltc_{seed}")
    d = np.load(ROOT / "results/race/nets" / f"race_ltc_{seed}.npz")
    cell = LTCCell(N_RACE_FEATURES, int(d["n_hidden"]), seed=seed)
    dcl = d["D_class"]; ncls = int(dcl.shape[0])
    pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed, n_classes=ncls,
                       delta_log=float(d["delta_log"]),
                       kv_floor=KV_FLOOR, kv_floor_classes=KV_FLOOR_CLASSES,
                       kv_ceil=KV_CEIL, kv_ceil_classes=KV_CEIL_CLASSES)
    pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.D_class[...] = dcl
    pol.reset(); pol.cls = PACE_KINDS.index(kind)
    ego_pace = 1.65   # measured ego solo pace in the reward5 run log; opponent speeds scale off it
    v_opp = FAIR_PACE[kind] * ego_pace
    s0 = (seed % 4) * track.length / 4.0; v0 = 1.3 + 0.1 * (seed % 3)
    opp = RacelineOpponent(track, s0=(s0 + 3.0) % track.length, pace=v_opp, offset=0.0,
                           radius=KEEPOUT_R, a_lat=2.5)
    tracker = ObstacleTracker(dt=0.05); P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=s0, v0=v0); m.reset(); opp.reset(); tracker.update(opp.pose()[:2]); m.set_obstacles([opp.keepout()])

    def emit(slip, gap_rate=None, sec_suit=None):
        feat = race_features(track, P.state5(), [opp], opp_speed_est=tracker.speed,
                             slip=slip, gap_rate=gap_rate, sector_suit=sec_suit)
        return np.asarray(pol.step(feat), float)
    _b = float(P._x[6]); _r = float(P._x[5]); _v = float(P._x[3])
    _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
    theta = emit((_ar, _b, _r)); u = m.value(P.state_dyn(), theta)["u0"]
    log = {k: [] for k in ["EX", "EY", "S", "V", "BETA", "R", "ALR", "GAP", "PASS", "OX", "OY"]}
    logT = []
    passes = 0; seen = False; prev_g = None; sec_pass = np.zeros(4); sec_ct = np.zeros(4)
    for _ in range(steps):
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); opp.step(0.05, ego=(ex, ey, float(P._x[3])))
        ox, oy, rad = opp.keepout(); s_ego = track.project(ex, ey)
        g = signed_gap(track, s_ego, opp.s); sec = int(track.sector(track.wrap(s_ego)))
        _v = float(P._x[3]); _r = float(P._x[5]); _b = float(P._x[6])
        _ar = -np.arctan2(_v * np.sin(_b) - LR_VEH * _r, _v * np.cos(_b)) if _v * np.cos(_b) > 0.05 else 0.0
        vo = float(getattr(opp, "speed", v_opp))
        if g < 0 and abs(g) < track.length / 4 and not seen and _v > vo:
            passes += 1; seen = True; sec_pass[sec] += 1
        elif g > 0.5:
            seen = False
        for k, val in zip(["EX", "EY", "S", "V", "BETA", "R", "ALR", "GAP", "PASS", "OX", "OY"],
                          [ex, ey, s_ego, _v, _b, _r, _ar, g, passes, ox, oy]):
            log[k].append(val)
        logT.append(np.exp(theta))
        m.set_obstacles([opp.keepout()]); tracker.update(opp.pose()[:2])
        gr = (g - prev_g) / 0.05 if prev_g is not None else 0.0; prev_g = g
        theta = emit((_ar, _b, _r), gap_rate=gr, sec_suit=(sec_pass - sec_ct))
        u = m.value(P.state_dyn(), theta)["u0"]
        if off or tr:
            break
    arr = {k: np.asarray(v, float) for k, v in log.items()}
    arr["THETA"] = np.asarray(logT, float); arr["off"] = np.asarray([off])
    np.savez(cache, **arr)
    return arr


def get_states(seed, kind, redrive_flag):
    cache = OUT / f"states_ltc_{seed}_{kind}.npz"
    if cache.exists() and not redrive_flag:
        return {k: np.load(cache)[k] for k in np.load(cache).files}
    return redrive(seed, kind)


# --------------------------------------------------------------------------- track drawing
def track_background(ax):
    from PIL import Image
    im = np.array(Image.open(TR / "icra2026_t2.pgm")); H, W = im.shape; res, ox, oy = 0.05, -2.8, -7.25
    ax.imshow(im, cmap="gray", extent=[ox, ox + W * res, oy, oy + H * res], origin="upper", zorder=0, alpha=0.55)
    d0 = np.load(TR / "icra_t2_raceline_ref_corridor.npz")
    tt = Track(d0["cx"], d0["cy"], ds=0.1, w_left=d0["wl"], w_right=d0["wr"])
    ss = np.linspace(0, tt.length, 1400, endpoint=False); L = []; Rr = []
    for si in ss:
        p = np.asarray(tt.pos(float(si))).ravel(); a = float(tt.tangent_angle(float(si)))
        nx, ny = -np.sin(a), np.cos(a); wl, wr = tt.width(float(si))
        L.append([p[0] + nx * float(wr), p[1] + ny * float(wr)])
        Rr.append([p[0] - nx * float(wl), p[1] - ny * float(wl)])
    L = np.array(L); Rr = np.array(Rr)
    ax.plot(L[:, 0], L[:, 1], color="0.35", lw=0.7, zorder=1)
    ax.plot(Rr[:, 0], Rr[:, 1], color="0.35", lw=0.7, zorder=1)
    ax.set_aspect("equal"); ax.axis("off")


# =========================================================================== FIGURES
def fig_learn_by_opponent(runs):
    M, start = weights_by_kind(runs)
    ks = [k for k in KINDS if k in M]
    Z = np.array([np.log2(M[k] / start) for k in ks])            # (nk, 9)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    vmax = float(np.nanmax(np.abs(Z))) or 1.0
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(WEIGHT_NAMES))); ax.set_xticklabels([WLAB[w] for w in WEIGHT_NAMES])
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([KIND_LABEL[k] for k in ks])
    ax.set_ylabel("opponent")
    for i in range(len(ks)):
        for j in range(len(WEIGHT_NAMES)):
            ax.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=5.5,
                    color="k" if abs(Z[i, j]) < 0.6 * vmax else "w")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_2(\mathrm{emitted}/\mathrm{START})$", fontsize=7)
    ax.set_title("Learned weights by opponent class")
    save(fig, "fig_learn_by_opponent")


def fig_learn_over_rounds(runs):
    C, start = curve_over_encounters(runs)
    idx = {w: WEIGHT_NAMES.index(w) for w in KEY}
    fig, axs = plt.subplots(1, len(KEY), figsize=(7.16, 1.8), sharex=True)
    for ax, w in zip(axs, KEY):
        j = idx[w]
        for k in KINDS:
            mean, std = C[k]
            if not len(mean):
                continue
            y = mean[:, j] / start[j]; ep = np.arange(1, len(y) + 1); ok = ~np.isnan(y)
            ax.plot(ep[ok], y[ok], color=KCOL[k], marker="o", ms=2.6, lw=1.1, label=KIND_LABEL[k])
            s = (std[:, j] / start[j])
            ax.fill_between(ep[ok], (y - s)[ok], (y + s)[ok], color=KCOL[k], alpha=0.12, lw=0)
        ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
        ax.set_title(WLAB[w]); ax.set_xlabel("encounter $n$")
        ax.set_xticks([1, 2, 3, 4]); ax.margins(x=0.05)
    axs[0].set_ylabel(r"emitted / START")
    axs[-1].legend(ncol=1, loc="best", fontsize=6, handlelength=1.2)
    fig.suptitle(r"Online weight adaptation over repeated encounters with each opponent (mean$\pm$sd over seeds)", y=1.07)
    save(fig, "fig_learn_over_rounds")


def fig_track_states(seed, kinds, redrive_flag):
    logs = {k: get_states(seed, k, redrive_flag) for k in kinds}
    nk = len(kinds)
    fig = plt.figure(figsize=(7.16, 2.7 * 1))
    gs = fig.add_gridspec(1, nk + 1, width_ratios=[2.1] * nk + [1.35], wspace=0.28)
    # per-kind track panels
    vall = np.concatenate([logs[k]["V"] for k in kinds])
    vmin, vmax = float(vall.min()), float(vall.max())
    from matplotlib import colormaps
    cmap = colormaps["viridis"]
    for c, k in enumerate(kinds):
        ax = fig.add_subplot(gs[0, c]); track_background(ax)
        L = logs[k]; ex, ey, v = L["EX"], L["EY"], L["V"]
        # faded full-lap ghost
        ax.plot(ex, ey, color="0.25", lw=0.5, alpha=0.25, zorder=2)
        # speed-coloured trajectory
        pts = np.column_stack([ex, ey]); seg = np.concatenate([pts[:-1, None], pts[1:, None]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=1.6, zorder=3)
        lc.set_array(v[:-1]); ax.add_collection(lc)
        # opponent path (faint) + pass markers
        ax.plot(L["OX"], L["OY"], color="#D55E00", lw=0.6, alpha=0.35, zorder=2)
        pas = np.asarray(L["PASS"]); jumps = np.where(np.diff(pas) > 0)[0] + 1
        if len(jumps):
            ax.scatter(ex[jumps], ey[jumps], marker="*", s=45, c="#D55E00",
                       edgecolors="k", linewidths=0.4, zorder=6, label="overtake")
        ax.set_title(f"vs {KIND_LABEL[k]} opponent", fontsize=8)
        if c == 0 and len(jumps):
            ax.legend(loc="lower left", fontsize=6, handletextpad=0.2)
    # colourbar for speed
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cax = fig.add_axes([0.045, -0.02, 0.30, 0.03])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("ego speed [m/s]", fontsize=7)
    # state traces panel (use the first kind for clarity, all states vs lap distance)
    axs = gs[0, nk].subgridspec(3, 1, hspace=0.32)
    a0 = fig.add_subplot(axs[0]); a1 = fig.add_subplot(axs[1]); a2 = fig.add_subplot(axs[2])
    for k in kinds:
        L = logs[k]; sl = one_lap(L["S"]); s = L["S"][sl]
        a0.plot(s, L["V"][sl], color=KCOL[k], lw=0.9, label=KIND_LABEL[k])
        a1.plot(s, np.degrees(L["BETA"][sl]), color=KCOL[k], lw=0.9)
        a2.plot(s, np.degrees(L["R"][sl]), color=KCOL[k], lw=0.9)
    a0.set_ylabel(r"$v$ [m/s]", fontsize=7); a1.set_ylabel(r"$\beta$ [deg]", fontsize=7)
    a2.set_ylabel(r"$\dot\psi$ [deg/s]", fontsize=7); a2.set_xlabel("lap distance $s$ [m]", fontsize=7)
    for a in (a0, a1, a2):
        a.grid(True, alpha=0.25); a.tick_params(labelsize=6)
    a0.legend(ncol=1, fontsize=5.5, loc="lower right", handlelength=1.0)
    a0.set_title("our states along the lap", fontsize=8)
    fig.suptitle(f"Track, speed-coloured trajectory and vehicle states (LTC, seed {seed})", y=1.02)
    save(fig, "fig_track_states")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="+", default=["equal", "faster"])
    ap.add_argument("--redrive", action="store_true")
    a = ap.parse_args()
    ieee_style()
    runs, _ = load_runs("ltc")
    print("figures ->", OUT)
    fig_learn_by_opponent(runs)
    fig_learn_over_rounds(runs)
    fig_track_states(a.seed, a.kinds, a.redrive)
