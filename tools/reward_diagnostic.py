"""Is it the REWARD or the LEARNING that fails to differentiate weights by opponent class?

    ACADOS_SOURCE_DIR=... PYTHONPATH=/path/to/scuderia_gym_jax:. python3 tools/reward_diagnostic.py

The full run showed the tuner emits the SAME weight-corner for static/slower/equal/faster.
Two possible causes:
  (A) REWARD: the class-dependent reward's argmax in weight-space does NOT move with class
      (class only SCALES reward), so every class prefers the same weights -- no learning change
      would help.
  (B) LEARNING: a different weight IS optimal per class, but the policy (saturated tanh box)
      cannot express/find it.

This tells them apart WITHOUT training: hold the policy FIXED at START, sweep ONE weight
(k_v = grip claim, then q_c = line shape) across its factor-2 box, and for each opponent class
measure the closed-loop RETURN (what the tuner maximises) and every reward COMPONENT. If the
return-vs-knob curve peaks at the SAME knob value for every class -> cause (A), proven. The
component breakdown shows WHY -- in particular whether the class-scaled SLIP risk (the term
meant to create the difference) ever activates before the car hits the off-track corridor edge.

Outputs results/race/stability/reward_diagnostic.{pdf,png,json} + a CSV.
"""
import sys, os, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B

CLASSES = ("slower", "equal", "faster")
KNOBS = {"k_v": (7, [0.30, 0.45, 0.60, 0.75, 0.90]),    # grip claim: box [0.30, 0.90], START 0.45
         "q_c": (0, [0.50, 0.80, 1.00, 1.40, 2.00])}    # line shape: box [0.5, 2.0],  START 1.0
N_EP = 2                    # physically-distinct episodes per cell (vary opponent start gap)
STEPS = 3000               # ~2-3 laps: enough to see passes and off-track
EGO_PACE = 1.49            # matches results/race/race_phase1_fast.json


def cell(job):
    """One (knob, value) cell: build the MPCC once, run every class x episode at a FIXED theta
    with that one weight overridden, and accumulate the reward components."""
    knob, val = job
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, RacelineOpponent
    from experiments.race_mode import (race_features, race_reward, signed_gap, FAIR_PACE,
                                        KEEPOUT_R, CONTACT_R, A_LAT_RACE, LR_VEH,
                                        ALPHA_R_REF, CLASS_SCALE)
    idx = KNOBS[knob][0]
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    theta = th0.copy(); theta[idx] = float(np.log(val))          # override ONE weight (log space)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
                   discrete=True, max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4,
                   name=f"diag_{knob}_{str(val).replace('.', '')}")
    out = []
    for kind in CLASSES:
        v_opp = FAIR_PACE[kind] * EGO_PACE
        for ep in range(N_EP):
            gap0 = 3.0 + 1.5 * ep
            P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = STEPS
            P.reset(s0=0.0, v0=1.4); m.reset()
            opp = RacelineOpponent(track, s0=gap0 % track.length, pace=v_opp, offset=0.0,
                                   radius=KEEPOUT_R, a_lat=2.5)
            opp.reset(); tracker = ObstacleTracker(dt=0.05); tracker.update(opp.pose()[:2])
            m.set_obstacles([opp.keepout()])
            u = m.value(P.state_dyn(), theta)["u0"]
            base_s = float(P.state5()[4]); off = tr = contact = False; passes = 0; seen = False
            acc = dict(base=0.0, speed=0.0, pass_b=0.0, closing=0.0, slip=0.0, contact=0.0)
            R = 0.0; n = 0; slip_active = 0; amax = 0.0; asum = 0.0
            for _ in range(STEPS):
                s5n, r, off, tr = P.step(u); n += 1
                ex, ey = float(P._x[0]), float(P._x[1])
                opp.step(0.05, ego=(ex, ey, float(P._x[3])))
                vo = float(getattr(opp, "speed", v_opp)); ox, oy, _ = opp.keepout()
                dist = float(np.hypot(ex - ox, ey - oy)); s_ego = track.project(ex, ey)
                g = signed_gap(track, s_ego, opp.s)
                _v = float(P._x[3]); _r = float(P._x[5]); _beta = float(P._x[6]) if P._x.size > 6 else 0.0
                _vx = _v * np.cos(_beta); _vy = _v * np.sin(_beta)
                _alpha_r = -np.arctan2(_vy - LR_VEH * _r, _vx) if _vx > 0.05 else 0.0
                amax = max(amax, abs(_alpha_r)); asum += abs(_alpha_r)
                if abs(_alpha_r) > CLASS_SCALE[kind] * ALPHA_R_REF:
                    slip_active += 1
                if dist < CONTACT_R and not contact:
                    contact = True
                just_passed = False
                if g < 0 and abs(g) < track.length / 4 and not seen and _v > vo:
                    passes += 1; seen = True; just_passed = True
                elif g > 0.5:
                    seen = False
                m.set_obstacles([opp.keepout()]); tracker.update(opp.pose()[:2])
                fn = race_features(track, s5n, [opp], opp_speed_est=tracker.speed)
                r_sh, parts = race_reward(kind, r, just_passed, contact, _v, vo, g,
                                          alpha_r=_alpha_r, beta=_beta, return_parts=True)
                R += r_sh
                for k in acc:
                    acc[k] += parts[k]
                u = m.value(P.state_dyn(), theta)["u0"]
                if off or tr or contact:
                    break
            laps = (float(P.state5()[4]) - base_s) / track.length
            out.append(dict(knob=knob, val=val, kind=kind, ep=ep, R=R, ticks=n,
                            laps=laps, off=bool(off), contact_flag=bool(contact), passes=passes,
                            slip_active_frac=slip_active / max(n, 1),
                            mean_abs_alpha=asum / max(n, 1), max_abs_alpha=amax, **acc))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    jobs = [(kn, v) for kn, (_, vals) in KNOBS.items() for v in vals]
    with Pool(min(a.jobs, len(jobs))) as pool:
        rows = [r for sub in pool.map(cell, jobs) for r in sub]
    OUT = ROOT / "results/race/stability"; OUT.mkdir(parents=True, exist_ok=True)
    # CSV
    cols = ["knob", "val", "kind", "ep", "R", "ticks", "laps", "off", "contact_flag", "passes",
            "slip_active_frac", "mean_abs_alpha", "max_abs_alpha",
            "base", "speed", "pass_b", "closing", "slip", "contact"]
    import csv as _csv
    with open(OUT / "reward_diagnostic.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)

    def agg(knob, kind, val, key):
        xs = [r[key] for r in rows if r["knob"] == knob and r["kind"] == kind and r["val"] == val]
        return float(np.mean(xs)) if xs else np.nan

    # figure: for each knob, RETURN vs knob per class + component/off/slip diagnostics
    fig, axs = plt.subplots(2, len(KNOBS), figsize=(7 * len(KNOBS), 10))
    ckind = {"slower": "tab:green", "equal": "tab:orange", "faster": "tab:red"}
    peaks = {}
    for c, (knob, (_, vals)) in enumerate(KNOBS.items()):
        ax = axs[0, c]
        for kind in CLASSES:
            ys = [agg(knob, kind, v, "R") for v in vals]
            ax.plot(vals, ys, "-o", color=ckind[kind], label=kind)
            vpk = vals[int(np.nanargmax(ys))]; peaks[f"{knob}:{kind}"] = vpk
            ax.scatter([vpk], [np.nanmax(ys)], s=180, facecolor="none", edgecolor=ckind[kind], lw=2)
        ax.set_xlabel(f"{knob}  (START {np.exp(B.start('icra_t2_smooth').theta()[KNOBS[knob][0]]):.2f})")
        ax.set_ylabel("total return  (what the tuner maximises)")
        ax.set_title(f"return vs {knob} -- do the class peaks (rings) DIFFER?"); ax.legend(); ax.grid(alpha=.3)
        # bottom: off-track fraction (the binding constraint) + slip-active fraction, vs knob
        ax2 = axs[1, c]
        for kind in CLASSES:
            offs = [agg(knob, kind, v, "off") for v in vals]
            ax2.plot(vals, offs, "-o", color=ckind[kind], label=f"{kind} off-track")
            sl = [agg(knob, kind, v, "slip_active_frac") for v in vals]
            ax2.plot(vals, sl, "--s", color=ckind[kind], alpha=0.6)
        ax2.set_xlabel(knob); ax2.set_ylabel("fraction (solid=off-track, dashed=slip-active)")
        ax2.set_title(f"{knob}: does slip EVER bind before off-track?"); ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig.suptitle("Reward diagnostic: is a different weight optimal per opponent class? "
                 "(fixed policy, class peaks ringed)", fontsize=13)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"reward_diagnostic.{e}", dpi=140)
    json.dump(dict(peaks=peaks, n_rows=len(rows),
                   note="If peaks[knob:slower]==peaks[knob:equal]==peaks[knob:faster] the reward's "
                        "argmax does NOT move with class -> cause (A) reward, not learning."),
              open(OUT / "reward_diagnostic.json", "w"), indent=2)
    print("PEAKS (return-maximising knob value per class):")
    for knob in KNOBS:
        print(f"  {knob}: " + "  ".join(f"{k}={peaks[f'{knob}:{k}']}" for k in CLASSES))
    print("saved results/race/stability/reward_diagnostic.{pdf,png,json,csv}")


if __name__ == "__main__":
    main()
