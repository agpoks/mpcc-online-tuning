"""Race-mode Phase 1: head-to-head behaviour training against ONE opponent.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_mode.py --seeds 6

The shipping stack (acados + dynamic drift + STD plant, discrete integrator to
match the online-net training), on the canonical Track.icra_t2_smooth. One
opponent per episode, cycled across relative-pace classes -- static / slower /
equal / faster (scaled to the ego's own solo pace) -- placed a few metres ahead,
with the ego start point varied per seed (physical seeds, not an RNG).

What the policy must learn (per sector): whether to STAY BEHIND at a distance or
OVERTAKE, based on relative pace, available space, closing speed, and WHICH SIDE
IS OPEN. A sector that is too tight / crash-prone is a bad place to attempt a
pass, so the net should follow there and pass in the good sectors instead.

Arms, identical except what emits theta each tick:
  const   START weights held fixed (the best-constant baseline)
  fixed   fixed_schedule -- the rule-based relative-pace lookup (baseline to beat)
  ltc     WeightPolicy(LTCCell)  -- the paper's memory tuner (MPCC-critic, keep-best)
  mlp     WeightPolicy(MLPCell)  -- the memoryless ablation

Success is a MIX, not one headline: overtakes completed + no-contact (clean) +
time gained (progress vs the follow baseline), across opponent configs and seeds.
Per-sector attempt/pass/contact are logged so the "good overtake sector" the net
learns can be read out.

The MPCC keep-out carries the opponent as an (x,y,r) circle whose radius is both
cars' half-widths (mpcc_tuning/opponents.py); the learnable berth d_obs (weight 6)
sets how much room the pass leaves. Contact = the two bodies touch (dist < radius).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402
from mpcc_tuning.mpcc import WEIGHT_NAMES  # noqa: E402

OUT = ROOT / "results" / "race"
ARMS = ("const", "fixed", "ltc", "mlp")


def signed_gap(track, s_ego, s_opp):
    """+ opponent ahead, - opponent behind, in metres of arc length."""
    d = (s_opp - s_ego) % track.length
    return d - track.length if d > track.length / 2 else d


def side_open(track, opponents, car_w=0.24):
    """Two features appended to the base 18: WHICH SIDE IS OPEN at the nearest
    opponent, and whether a pass fits there at all.

    The opponent sits at ``offset`` in lateral() convention; the corridor has
    room ``wl`` (-normal) and ``wr`` (+normal) at the opponent's arc length.
    room_right = wr - offset, room_left = wl + offset.
      f0 = tanh(room_right - room_left)  -- sign = which side has more room
      f1 = clip(max(room_right,room_left) - car_w, over one car width)  -- pass fits?
    No opponent -> (0, 1): no side pressure, a pass trivially fits.
    """
    if not len(opponents):
        return [0.0, 1.0]
    o = opponents[0]   # phase 1: one opponent (extend to nearest-ahead for phase 2)
    try:
        wl, wr = track.width(o.s % track.length)
    except Exception:
        wl = wr = track.half_width
    off = float(getattr(o, "offset", 0.0))
    room_r = float(wr) - off
    room_l = float(wl) + off
    return [float(np.tanh(room_r - room_l)),
            float(np.clip((max(room_r, room_l) - car_w) / car_w, -1.0, 1.0))]


def race_features(track, s5, opponents=(), opp_speed_est=None):
    """Base 18 features + 2 side-open features = 20. The base indices are left
    untouched so fixed_schedule (which reads feat[7], feat[8], feat[14:18]) works."""
    from mpcc_tuning.ltc import features
    base = features(track, s5, opponents, opp_speed_est=opp_speed_est)
    return np.concatenate([base, np.array(side_open(track, opponents), float)])


N_RACE_FEATURES = 20


def measure_pace(m, track, th0, steps):
    """Ego solo mean speed, to scale opponent speeds to the ego's own pace."""
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=0.0, v0=1.0); m.reset()
    vs = []
    for _ in range(steps):
        u = m.value(P.state_dyn(), th0)["u0"]; s5, r, off, tr = P.step(u)
        vs.append(float(P._x[3]))
        if off or tr:
            break
    return float(np.mean(vs)) if vs else 1.4


# relative-pace classes: opponent speed as a fraction of the ego's solo pace.
# static=parked; slower=catchable; equal=expensive pass; faster=cannot catch.
PACE = {"static": 0.0, "slower": 0.55, "equal": 0.90, "faster": 1.20}
PACE_KINDS = ("static", "slower", "equal", "faster")

# Pace-DEPENDENT reward, per the intended behaviour for each opponent type:
#   static/slower : overtake, but drive STABLE (not full-aggressive) -- a strong
#                   bonus for a CLEAN pass, no extra speed pressure.
#   equal         : find and hold a good overtaking POSITION, then pass patiently
#                   -- the largest clean-pass bonus, plus a small reward for staying
#                   engaged in a passing window.
#   faster        : cannot out-wait a faster car -- reward CLOSING PACE (go fast to
#                   catch up); a pass, if it comes, still pays.
# Contact is always heavily penalised: a pass that touches is worse than no pass.
PASS_BONUS = {"static": 9.0, "slower": 9.0, "equal": 13.0, "faster": 5.0}
# Bigger keep-out so a pass leaves ROOM (was "nearly touching"): the MPCC treats the
# opponent as a KEEPOUT_R circle, so with the START berth d_obs~0.15 it holds the ego
# centre >~0.51 m off -> ~0.27 m clear body-to-body. This is a fixed safety margin, NOT
# the tunable d_obs (which must stay strictly inside its box). CONTACT_R is the physical
# body-to-body touch distance used only for detecting an actual collision.
KEEPOUT_R = 0.36
CONTACT_R = 0.24


def race_reward(kind, r, just_passed, contact, v_ego, v_opp, gap):
    """Adapted to overtake FASTER: bigger clean-pass bonuses, plus a shaping term
    that rewards CLOSING on a passable opponent (static/slower/equal) so the tuner
    commits to the move sooner instead of trailing. Contact stays heavily penalised."""
    if contact:
        return float(r) - 8.0
    x = float(r)
    if just_passed:
        x += PASS_BONUS.get(kind, 9.0)
    if kind in ("static", "slower", "equal"):
        # engaged and closing on a passable car -> reward committing to the overtake
        if 0.0 < gap < 4.0 and v_ego > v_opp + 0.02:
            x += 0.4 * float(v_ego - v_opp)
        if kind == "equal" and -1.0 < gap < 3.0:
            x += 0.3                   # hold a good overtaking position
    elif kind == "faster":
        x += 0.5 * float(v_ego)        # reward catching up (fastest safe speed)
    return x


def run(arm, seed=0, n_ep=10, steps=5500, n_hidden=12, ego_pace=1.4,
        factor=2.0, box="adapt", dump_traj=False):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import (LTCCell, MLPCell, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, fixed_schedule)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, Opponent

    track = Track.icra_t2_smooth()
    st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, discrete=True, max_obstacles=1,
                   name=f"race_{arm}_{seed}")
    lo, hi = (B.adaptation_box("icra_t2_smooth", factor) if box == "adapt"
              else (THETA_LO, THETA_HI))

    tuner = pol = None
    if arm in ("ltc", "mlp"):
        cell = (LTCCell if arm == "ltc" else MLPCell)(N_RACE_FEATURES, n_hidden, seed=seed)
        pol = WeightPolicy(cell, th0, lo, hi, seed=seed)
        # adapted to overtake faster: prior 0.3 -> 0.2 (let it move toward the
        # overtake posture q_v/q_c~2), alpha 2e-3 -> 3e-3 (learn quicker). Contact
        # is still penalised, so "faster" is bounded by the safety term.
        tuner = PolicyTuner(m, pol, alpha=3e-3, explore=0.06, delta_clip=1.0,
                            seed=seed, trust_region=0.01, theta_prior=0.2)

    rng = np.random.default_rng(seed)
    rows = []
    traj = {}                      # one trajectory per opponent kind (last episode), if dumping
    for ep in range(n_ep):
        kind = PACE_KINDS[(seed + ep) % 4]
        v_opp = PACE[kind] * ego_pace
        s0 = (seed % 4) * track.length / 4.0
        v0 = 1.0 + 0.1 * (seed % 3)
        gap0 = 3.0 + 1.5 * (ep % 3)          # opponent starts a few m ahead
        opp = Opponent(track, s0=(s0 + gap0) % track.length, speed=v_opp,
                       offset=0.0, radius=KEEPOUT_R)
        tracker = ObstacleTracker(dt=0.05)
        P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
        s5 = P.reset(s0=s0, v0=v0); m.reset()
        if tuner is not None:
            tuner.reset()
        opp.reset()
        tracker.update(opp.pose()[:2])
        m.set_obstacles([opp.keepout()])
        feat = race_features(track, P.state5(), [opp], opp_speed_est=tracker.speed)
        if arm in ("ltc", "mlp"):
            theta, u = tuner.act(feat, P.state_dyn())
        elif arm == "fixed":
            theta = fixed_schedule(feat, th0); u = m.value(P.state_dyn(), theta)["u0"]
        else:  # const
            theta = th0; u = m.value(P.state_dyn(), theta)["u0"]

        base = float(P.state5()[4]); off = tr = False
        passes = 0; contact = False; seen = False
        sec_attempt = np.zeros(4); sec_pass = np.zeros(4); sec_contact = np.zeros(4)
        theta_acc = np.zeros(8); n_th = 0        # mean emitted weights this episode
        sec_theta = np.zeros((4, 8)); sec_theta_n = np.zeros(4)   # weights BY SECTOR
        TEX = []; TEY = []; TEV = []; TOX = []; TOY = []; TG = []; TP = []   # trajectory (if dumping)
        for _ in range(steps):
            theta_acc += np.exp(np.asarray(theta, float)); n_th += 1
            s5n, r, off, tr = P.step(u)
            opp.step(0.05)
            ex, ey = float(P._x[0]), float(P._x[1])
            ox, oy, rad = opp.keepout()
            dist = float(np.hypot(ex - ox, ey - oy))
            s_ego = track.project(ex, ey)
            g = signed_gap(track, s_ego, opp.s)
            sec = int(track.sector(track.wrap(s_ego)))
            sec_theta[sec] += np.exp(np.asarray(theta, float)); sec_theta_n[sec] += 1
            # contact = the two bodies actually touch (physical distance, not the keep-out)
            if dist < CONTACT_R and not contact:
                contact = True; sec_contact[sec] += 1
            # engagement: opponent within a car-length ahead and we are closing
            if 0 < g < 2.0 and float(P._x[3]) > v_opp + 0.05:
                sec_attempt[sec] += 1
            # a completed pass: opponent went from ahead to behind, we are faster
            just_passed = False
            if g < 0 and abs(g) < track.length / 4 and not seen and float(P._x[3]) > v_opp:
                passes += 1; seen = True; sec_pass[sec] += 1; just_passed = True
            elif g > 0.5:
                seen = False
            if dump_traj:
                TEX.append(ex); TEY.append(ey); TEV.append(float(P._x[3]))
                TOX.append(ox); TOY.append(oy); TG.append(g); TP.append(passes)
            m.set_obstacles([opp.keepout()])
            tracker.update(opp.pose()[:2])
            fn = race_features(track, s5n, [opp], opp_speed_est=tracker.speed)
            # pace-DEPENDENT shaped reward (race_reward): the pass bonus fires on the
            # tick the pass COMPLETES (just_passed), which the old code missed because
            # `seen` was already set -- so the tuner never saw a pass reward before.
            r_shaped = race_reward(kind, r, just_passed, contact,
                                   float(P._x[3]), v_opp, g)
            if arm in ("ltc", "mlp"):
                out = tuner.learn(r_shaped, P.state_dyn(), fn, off or contact)
                if out[0] is None:
                    break
                theta, u = out
            elif arm == "fixed":
                theta = fixed_schedule(fn, th0); u = m.value(P.state_dyn(), theta)["u0"]
            else:
                theta = th0; u = m.value(P.state_dyn(), theta)["u0"]
            if off or tr or contact:
                break
        laps = (float(P.state5()[4]) - base) / track.length
        clean = (not off) and (not contact)
        rows.append(dict(ep=ep, kind=kind, laps=round(laps, 3), passes=int(passes),
                         contact=bool(contact), off=bool(off), clean=bool(clean),
                         sec_attempt=sec_attempt.tolist(), sec_pass=sec_pass.tolist(),
                         sec_contact=sec_contact.tolist(),
                         theta_mean=(theta_acc / max(n_th, 1)).tolist(),
                         sec_theta=(sec_theta / np.maximum(sec_theta_n[:, None], 1)).tolist()))
        if dump_traj and TEX:
            traj[kind] = dict(EX=TEX, EY=TEY, EV=TEV, OX=TOX, OY=TOY, GAP=TG, PASS=TP)
    # save the trained policy so the GIF/eval can REPLAY the learned behaviour
    if tuner is not None:
        ndir = OUT / "nets"; ndir.mkdir(parents=True, exist_ok=True)
        np.savez(str(ndir / f"race_{arm}_{seed}.npz"), G=pol.G, cell_p=pol.cell.p,
                 th0=th0, n_hidden=pol.cell.n, arm=arm, seed=seed,
                 lo=np.asarray(lo, float), hi=np.asarray(hi, float))
    if dump_traj and traj:
        # per-opponent-kind trajectory (last episode of each kind) for the 2D paper plots,
        # so tools/race_2d.py can render WITHOUT re-driving (no acados).
        tdir = OUT / "traj"; tdir.mkdir(parents=True, exist_ok=True)
        np.savez(str(tdir / f"traj_{arm}_{seed}.npz"),
                 **{f"{k}_{fld}": np.array(v[fld]) for k, v in traj.items() for fld in v})
    last = rows[-8:] if len(rows) >= 8 else rows
    return dict(arm=arm, seed=seed,
                laps=float(np.mean([r["laps"] for r in last])),
                passes=float(np.mean([r["passes"] for r in last])),
                clean=float(np.mean([r["clean"] for r in last])),
                contact=float(np.mean([r["contact"] for r in last])),
                rows=rows)


def one(job):
    arm, seed, n_ep, steps, ego_pace, dump_traj = job
    t0 = time.perf_counter()
    out = run(arm, seed=seed, n_ep=n_ep, steps=steps, ego_pace=ego_pace, dump_traj=dump_traj)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5500,
                    help="~5 laps at START pace (73.6 m track, ~1.35 m/s) so the ego "
                    "meets each opponent several times and can change strategy per lap")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--pilot", action="store_true",
                    help="1 seed, 4 episodes, 600 steps, arms const+ltc -- quick smoke")
    ap.add_argument("--out", default=str(OUT / "race_phase1.json"))
    ap.add_argument("--dump-traj", action="store_true",
                    help="save one trajectory per opponent kind (results/race/traj/) so "
                    "tools/race_2d.py can render 2D paper plots without re-driving")
    a = ap.parse_args(argv)
    if a.pilot:
        a.seeds, a.episodes, a.steps, a.arms = 1, 2, 1200, ["const", "ltc"]

    # measure ego pace once (solo START drive) to scale opponents
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    mp = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                    q_vref=st.q_vref, discrete=True, max_obstacles=1, name="race_pace")
    ego_pace = measure_pace(mp, track, th0, a.steps)
    print(f"  ego solo pace = {ego_pace:.2f} m/s ; opponents: "
          + ", ".join(f"{k}={PACE[k]*ego_pace:.2f}" for k in PACE_KINDS), flush=True)

    jobs = [(arm, s, a.episodes, a.steps, ego_pace, a.dump_traj)
            for s in range(a.seeds) for arm in a.arms]
    n_proc = a.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  {len(jobs)} runs, {a.episodes} episodes, {n_proc} processes\n", flush=True)

    import multiprocessing as mp2
    res = []
    with mp2.get_context("spawn").Pool(n_proc) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o)
            print(f"  done  {o['arm']:<6} seed {o['seed']}  {o['laps']:5.2f} laps"
                  f"  {o['passes']:.2f} passes  {o['clean']:.0%} clean"
                  f"  {o['contact']:.0%} contact  ({o['wall_s']:.0f}s)", flush=True)

    print(f"\n  {'arm':<7}{'laps':>7}{'passes':>8}{'clean':>7}{'contact':>8}")
    S = {}
    for arm in a.arms:
        r = [x for x in res if x["arm"] == arm]
        if not r:
            continue
        S[arm] = dict(laps=float(np.mean([x["laps"] for x in r])),
                      passes=float(np.mean([x["passes"] for x in r])),
                      clean=float(np.mean([x["clean"] for x in r])),
                      contact=float(np.mean([x["contact"] for x in r])), n=len(r))
        print(f"  {arm:<7}{S[arm]['laps']:7.2f}{S[arm]['passes']:8.2f}"
              f"{S[arm]['clean']:7.0%}{S[arm]['contact']:8.0%}")

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(summary=S, ego_pace=ego_pace, runs=res), indent=2) + "\n")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
