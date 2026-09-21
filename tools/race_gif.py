"""2D GIF of a race-mode episode: ego vs one opponent on the real map.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/race_gif.py \
        --arm ltc --kind slower --seed 0 --out results/race/gif/ltc_slower.gif

Drives ONE episode on Track.icra_t2_smooth against one opponent of the given pace
class (static/slower/equal/faster), then animates it: the corridor + real map, the
ego car (trail coloured by speed) and the opponent car with its keep-out circle, and
a HUD (pace class, gap, passes, contact). --arm const/fixed drives the baseline; --arm
ltc/mlp REPLAYS the trained net saved by experiments/race_mode.py at
results/race/nets/race_{arm}_{seed}.npz. Always writes a .gif so results are shown, not
just tabulated.
"""
import sys, os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.opponents import ObstacleTracker, Opponent
from experiments.race_mode import race_features, PACE, signed_gap, N_RACE_FEATURES, KEEPOUT_R, CONTACT_R


def drive(arm, kind, seed, steps, ego_pace):
    from mpcc_tuning.ltc import (LTCCell, MLPCell, THETA_LO, THETA_HI,
                                 WeightPolicy, fixed_schedule)
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, discrete=True, max_obstacles=1, name=f"gif_{arm}_{seed}")
    pol = None
    if arm in ("ltc", "mlp"):
        d = np.load(ROOT / "results/race/nets" / f"race_{arm}_{seed}.npz")
        cell = (LTCCell if arm == "ltc" else MLPCell)(N_RACE_FEATURES, int(d["n_hidden"]), seed=seed)
        pol = WeightPolicy(cell, d["th0"], d["lo"], d["hi"], seed=seed)
        pol.G[...] = d["G"]; pol.cell.p[...] = d["cell_p"]; pol.reset()
    v_opp = PACE[kind] * ego_pace
    s0 = (seed % 4) * track.length / 4.0; v0 = 1.0 + 0.1 * (seed % 3)
    opp = Opponent(track, s0=(s0 + 3.0) % track.length, speed=v_opp, offset=0.0, radius=KEEPOUT_R)
    tracker = ObstacleTracker(dt=0.05)
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=s0, v0=v0); m.reset(); opp.reset()
    tracker.update(opp.pose()[:2]); m.set_obstacles([opp.keepout()])

    def emit():
        feat = race_features(track, P.state5(), [opp], opp_speed_est=tracker.speed)
        if pol is not None:
            return np.asarray(pol.step(feat), float)
        if arm == "fixed":
            return fixed_schedule(feat, th0)
        return th0
    theta = emit(); u = m.value(P.state_dyn(), theta)["u0"]
    EX, EY, EV, OX, OY, GAP, PASS, CONTACT = [], [], [], [], [], [], [], []
    passes = 0; contact = False; seen = False
    for _ in range(steps):
        s5n, r, off, tr = P.step(u); opp.step(0.05)
        ex, ey = float(P._x[0]), float(P._x[1]); ox, oy, rad = opp.keepout()
        g = signed_gap(track, track.project(ex, ey), opp.s)
        if float(np.hypot(ex - ox, ey - oy)) < CONTACT_R:
            contact = True
        if g < 0 and abs(g) < track.length / 4 and not seen and float(P._x[3]) > v_opp:
            passes += 1; seen = True
        elif g > 0.5:
            seen = False
        EX.append(ex); EY.append(ey); EV.append(float(P._x[3])); OX.append(ox); OY.append(oy)
        GAP.append(g); PASS.append(passes); CONTACT.append(contact)
        m.set_obstacles([opp.keepout()]); tracker.update(opp.pose()[:2])
        theta = emit(); u = m.value(P.state_dyn(), theta)["u0"]
        if off or tr or contact:
            break
    return dict(EX=np.array(EX), EY=np.array(EY), EV=np.array(EV), OX=np.array(OX),
                OY=np.array(OY), GAP=GAP, PASS=PASS, CONTACT=CONTACT, rad=KEEPOUT_R, track=track)


def animate(d, arm, kind, out, stride=4, fps=20):
    track = d["track"]; EX, EY, EV = d["EX"], d["EY"], d["EV"]
    TR = ROOT / "mpcc_tuning/tracks"
    im = np.array(Image.open(TR / "icra2026_t2.pgm")); H, W = im.shape; res, ox, oy = 0.05, -2.8, -7.25
    # corridor edges
    d0 = np.load(TR / "icra_t2_raceline_ref_corridor.npz")
    tt = Track(d0["cx"], d0["cy"], ds=0.1, w_left=d0["wl"], w_right=d0["wr"])
    ss = np.linspace(0, tt.length, 1200, endpoint=False); L = []; Rr = []
    for si in ss:
        p = np.asarray(tt.pos(float(si))).ravel(); a = float(tt.tangent_angle(float(si)))
        nx, ny = -np.sin(a), np.cos(a); wl, wr = tt.width(float(si))
        L.append([p[0] + nx * float(wr), p[1] + ny * float(wr)]); Rr.append([p[0] - nx * float(wl), p[1] - ny * float(wl)])
    L = np.array(L); Rr = np.array(Rr)
    idx = list(range(0, len(EX), stride))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(im, cmap="gray", extent=[ox, ox + W * res, oy, oy + H * res], origin="upper", zorder=0)
    ax.plot(L[:, 0], L[:, 1], color="0.5", lw=0.8, zorder=1); ax.plot(Rr[:, 0], Rr[:, 1], color="0.5", lw=0.8, zorder=1)
    vmin, vmax = float(EV.min()), float(EV.max() + 1e-6)
    trail = LineCollection([], cmap="viridis", zorder=3, lw=3); trail.set_clim(vmin, vmax); ax.add_collection(trail)
    ego_dot, = ax.plot([], [], "o", color="white", mec="k", ms=11, zorder=5)
    opp_dot, = ax.plot([], [], "s", color="tab:red", mec="k", ms=11, zorder=5)
    circ = plt.Circle((0, 0), d["rad"], fill=False, ec="tab:red", ls="--", lw=1.3, zorder=4); ax.add_patch(circ)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=11,
                  family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.85), zorder=6)
    cb = fig.colorbar(trail, ax=ax, fraction=0.03, pad=0.02); cb.set_label("ego speed [m/s]")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"race-mode: {arm} vs {kind} opponent")

    def frame(k):
        j = idx[k]
        pts = np.column_stack([EX[:j + 1], EY[:j + 1]])
        if len(pts) > 1:
            seg = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            trail.set_segments(seg); trail.set_array(EV[:j])
        ego_dot.set_data([EX[j]], [EY[j]]); opp_dot.set_data([d["OX"][j]], [d["OY"][j]])
        circ.center = (d["OX"][j], d["OY"][j])
        st = "CONTACT" if d["CONTACT"][j] else ("AHEAD" if d["GAP"][j] < 0 else "behind")
        hud.set_text(f"{kind:<7} pace\ngap {d['GAP'][j]:+5.1f} m\npasses {d['PASS'][j]}\n{st}")
        return trail, ego_dot, opp_dot, circ, hud

    an = FuncAnimation(fig, frame, frames=len(idx), interval=1000 / fps, blit=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    an.save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    print(f"saved {out}  ({len(idx)} frames, passes={d['PASS'][-1]}, contact={d['CONTACT'][-1]})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="const"); ap.add_argument("--kind", default="slower")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--ego-pace", type=float, default=1.35)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or str(ROOT / "results/race/gif" / f"{a.arm}_{a.kind}_s{a.seed}.gif")
    d = drive(a.arm, a.kind, a.seed, a.steps, a.ego_pace)
    animate(d, a.arm, a.kind, out)


if __name__ == "__main__":
    main()
