"""Overtaking, on a controller that can actually drive.

    PYTHONPATH=/path/to/scuderia_gym_jax python3 experiments/overtaking_dynamic.py

Overtaking has been "tested" in this repo before -- experiments/race_matrix.py,
experiments/situation_demands.py -- and every one of those runs used the
KINEMATIC controller before the model was fixed, when the car could not
complete a lap of the oval. A pass measured against a car that crashes is not
a measurement of passing. None of it carries.

This is the first overtaking test on the dynamic model with the corrected
plant, the corrected prediction model and a solver configuration that drives:
IPOPT (5.12 laps) as the reference, and the best acados configuration
(SQP_WITH_FEASIBLE_QP + FUNNEL globalization, 7.12 laps) as the deployable one.

Three opponent speeds, as fractions of the ego's own measured solo pace, so
"slower" means slower than THIS car rather than slower than a guessed number:

* 0.55  -- must be passed, or the lap time is lost
* 0.85  -- passable, but only by committing
* 1.10  -- faster than the ego; the right answer is to follow, not to try

What is recorded is not just whether the car got past, but whether it stayed on
the track and how close it came: a "pass" that leaves the corridor or clips the
opponent is a failure reported as a success by a lap counter alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def solo_pace(track_name, horizon, steps=400):
    """The ego's own mean speed with no opponent, to scale the opponents to."""
    from mpcc_tuning.model import DynamicBicycle
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.track import Track
    t = getattr(Track, track_name)()
    m = MPCC(t, model=DynamicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300)
    th = MPCCWeights(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05).to_log()
    P = ScuderiaPlant(t, model="std", dt=0.05); P.max_steps = steps
    P.reset(); m.reset(); vs = []
    for _ in range(steps):
        o = m.value(P.state_dyn(), th)
        s5, _r, off, tr = P.step(o["u0"])
        vs.append(float(s5[3]))
        if off or tr:
            break
    return float(np.mean(vs)) if vs else 2.0


def run_ipopt_overtake(track_name, horizon, ratio, pace, steps=900, seed=0,
                       d_obs=0.15):
    from mpcc_tuning.model import DynamicBicycle
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.opponents import Opponent
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.track import Track
    t = getattr(Track, track_name)()
    m = MPCC(t, model=DynamicBicycle(dt=0.05), horizon=horizon, dt=0.05,
             max_iter=300, max_obstacles=1)
    th = MPCCWeights(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05,
                     d_obs=d_obs).to_log()
    P = ScuderiaPlant(t, model="std", dt=0.05, seed=seed); P.max_steps = steps
    P.reset(); m.reset()
    # Repeats must perturb something PHYSICAL. Seeding the plant's RNG changes
    # nothing -- the dynamics are deterministic from a start state, so seeded
    # "repeats" are the same run and their std is 0 by construction. The sweep
    # had this exact flaw and it was fixed there; it was not carried here.
    #
    # For an overtake the natural perturbation is WHERE the opponent is: a pass
    # set up on the straight is a different problem from one that arrives at
    # turn-in, and a controller that only manages the easy one has not been
    # shown to overtake.
    s_opp = 2.5 if not seed else float(
        np.random.default_rng(seed).uniform(1.0, t.length - 1.0))
    opp = Opponent(t, s0=s_opp, speed=ratio * pace, radius=0.24)
    s0 = float(P.state5()[4]); passes = 0; gap_min = 9e9
    behind_prev = True; off = tr = False; nok = k = 0; ts = []
    for _ in range(steps):
        m.set_obstacles([opp.keepout()])
        a = time.perf_counter()
        o = m.value(P.state_dyn(), th)
        ts.append(1000 * (time.perf_counter() - a))
        nok += int(bool(o["ok"])); k += 1
        opp.step(0.05)
        s5, _r, off, tr = P.step(o["u0"])
        ox, oy, _r_ = opp.keepout()
        gap = float(np.hypot(float(s5[0]) - ox, float(s5[1]) - oy))
        gap_min = min(gap_min, gap)
        ds = (t.project(float(s5[0]), float(s5[1])) - opp.s) % t.length
        behind = ds > t.length / 2
        if behind_prev and not behind and float(s5[3]) > opp.speed:
            passes += 1          # direction check -- see the acados path
        behind_prev = behind
        if off or tr:
            break
    ts = np.array(ts)
    return dict(backend="IPOPT", ratio=ratio, seed=seed, s_opp=s_opp,
                laps=(float(P.state5()[4]) - s0) / t.length, off=bool(off),
                passes=passes, min_gap=gap_min, ticks=k,
                solve_pct=100.0 * nok / max(k, 1),
                ms_mean=float(ts.mean()), ms_p99=float(np.percentile(ts, 99)))


def run_acados_overtake(track_name, horizon, ratio, pace, steps=900, seed=0,
                        d_obs=0.15, variant_name="fqp_soft_funnel"):
    """The DEPLOYABLE controller: acados with the winning configuration.

    IPOPT is the reference, not the thing that ships -- 126 ms/tick against
    acados' 7. Overtaking has to work on the solver that will actually run on
    the car, so this is the path that matters.
    """
    import casadi as ca
    from acados_template import AcadosOcpSolver
    from mpcc_tuning.acados_ocp import build_ocp, pack_params
    from mpcc_tuning.acados_status import RunLog, evaluate
    from mpcc_tuning.acados_variants import BY_NAME
    from mpcc_tuning.model import (ACCEL_MAX, SPEED_MAX, STEER_MAX,
                                   DynamicBicycle)
    from mpcc_tuning.mpcc import MPCCWeights
    from mpcc_tuning.opponents import Opponent
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.track import Track

    v = BY_NAME[variant_name]
    t = getattr(Track, track_name)()
    d = DynamicBicycle(dt=0.05); N = horizon
    ocp = build_ocp(t, horizon=N, dt=0.05, vehicle="dynamic",
                    spline_mode="spline", max_obstacles=1,
                    name=f"ot_{variant_name}", **v.build_kwargs())
    v.apply(ocp)
    export = ROOT / "c_generated" / f"ot_{variant_name}"
    ocp.code_export_directory = str(export)
    sv = AcadosOcpSolver(ocp, json_file=str(export) + ".json",
                         generate=True, build=True)

    th = MPCCWeights(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05,
                     d_obs=d_obs).to_log()
    P = ScuderiaPlant(t, model="std", dt=0.05, seed=seed); P.max_steps = steps
    P.reset()
    s_opp = 2.5 if not seed else float(
        np.random.default_rng(seed).uniform(1.0, t.length - 1.0))
    opp = Opponent(t, s0=s_opp, speed=ratio * pace, radius=0.24)

    x0 = P.state_dyn()
    xs, us = ca.MX.sym("x", 4 + d.n_dyn), ca.MX.sym("u", 2)
    stp = ca.Function("s", [xs, us], [d.step_sym(xs, us, 0.05)])
    xk = np.concatenate([x0[:4], x0[5:]]); ss = float(x0[4])
    for k in range(N + 1):
        sv.set(k, "x", np.concatenate([xk[:4], [ss], xk[4:]]))
        if k < N:
            sv.set(k, "u", np.array([0.0, 0.0, max(xk[3], 0.5)]))
            xk = np.asarray(stp(xk, [0.0, 0.0])).ravel()
            ss += max(xk[3], 0.5) * 0.05

    def set_p():
        """theta plus the keep-out, per stage. The obstacle block is part of p."""
        pp = pack_params(th, track=t, s_nodes=None, obstacles=[opp.keepout()],
                         max_obstacles=1, obs_margin=d_obs,
                         spline_mode="spline")[0]
        for k in range(N + 1):
            sv.set(k, "p", pp)

    log = RunLog(); margin = t.half_width - 0.12
    s0 = float(P.state5()[4]); passes = 0; gap_min = 9e9
    behind_prev = True; off = tr = False; k = 0
    for _ in range(steps):
        set_p()
        x0 = P.state_dyn()
        for j in range(N):                      # RTI-style shift
            sv.set(j, "x", sv.get(j + 1, "x"))
            if j < N - 1:
                sv.set(j, "u", sv.get(j + 1, "u"))
        sv.set(0, "lbx", x0); sv.set(0, "ubx", x0)
        a = time.perf_counter(); st = sv.solve()
        ms = 1000 * (time.perf_counter() - a)
        u = sv.get(0, "u"); k += 1
        e_c = [float(t.lateral(float(sv.get(j, "x")[0]),
                               float(sv.get(j, "x")[1]))) for j in range(1, N + 1)]
        log.add(evaluate(sv, st, u, ms, u_lb=[-STEER_MAX, -ACCEL_MAX, 0.0],
                         u_ub=[STEER_MAX, ACCEL_MAX, SPEED_MAX],
                         corridor=(e_c, margin)))
        opp.step(0.05)
        s5, _r, off, tr = P.step(u)
        ox, oy, _rr = opp.keepout()
        gap = float(np.hypot(float(s5[0]) - ox, float(s5[1]) - oy))
        gap_min = min(gap_min, gap)
        # A pass is the EGO gaining on the opponent, not merely the relative
        # arc length wrapping. Without the direction check, an opponent that
        # laps the ego registers as the ego passing IT -- which is how the
        # 1.10x row (a faster opponent, where the right answer is to follow)
        # reported a pass. Require the gap to be closing at the crossing.
        ds = (t.project(float(s5[0]), float(s5[1])) - opp.s) % t.length
        behind = ds > t.length / 2
        if behind_prev and not behind and float(s5[3]) > opp.speed:
            passes += 1
        behind_prev = behind
        if off or tr:
            break
    tm = log.timing()
    return dict(backend="acados:" + variant_name, ratio=ratio, seed=seed,
                s_opp=s_opp, laps=(float(P.state5()[4]) - s0) / t.length,
                off=bool(off), passes=passes, min_gap=gap_min, ticks=k,
                solve_pct=log.usable_pct, converged_pct=log.converged_pct,
                corridor_viol_max=log.corridor_violation_max,
                ms_mean=tm["mean"], ms_p99=tm["p99"], ms_max=tm["max"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="oval")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--ratios", nargs="*", type=float,
                    default=[0.55, 0.85, 1.10])
    ap.add_argument("--backend", default="acados",
                    choices=("acados", "ipopt"),
                    help="acados is the deployable one; IPOPT is "
                         "the reference at 126 ms/tick")
    ap.add_argument("--variant", default="fqp_soft_funnel")
    ap.add_argument("--acados-v060", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.acados_v060:
        from mpcc_tuning.acados_variants import ACADOS_V060
        if os.environ.get("_ACADOS_SWITCHED") != "1":
            env = dict(os.environ, _ACADOS_SWITCHED="1")
            env["ACADOS_SOURCE_DIR"] = ACADOS_V060["ACADOS_SOURCE_DIR"]
            for kk in ("LD_LIBRARY_PATH", "PYTHONPATH"):
                env[kk] = ACADOS_V060[kk] + os.pathsep + env.get(kk, "")
            os.execve(sys.executable, [sys.executable, *sys.argv], env)

    pace = solo_pace(a.track, a.horizon)
    print(f"  {a.track}: ego solo pace {pace:.2f} m/s; opponents scaled to it",
          flush=True)
    print("  %-8s %-8s %8s %6s %7s %9s %8s" %
          ("ratio", "opp m/s", "laps", "off", "passes", "min gap", "solve%"),
          flush=True)
    rows = []
    for r in a.ratios:
        for sd in range(a.repeats):
            if a.backend == "acados":
                row = run_acados_overtake(a.track, a.horizon, r, pace,
                                          steps=a.steps, seed=sd,
                                          variant_name=a.variant)
            else:
                row = run_ipopt_overtake(a.track, a.horizon, r, pace,
                                         steps=a.steps, seed=sd)
            rows.append(row)
            print("  %-8.2f %-8.2f %8.2f %6s %7d %9.3f %7.0f%%"
                  % (r, r * pace, row["laps"], "OFF" if row["off"] else "ok",
                     row["passes"], row["min_gap"], row["solve_pct"]),
                  flush=True)
    out = a.out or str(ROOT / "benchmarks" / "results" / "overtaking_dynamic.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(pace=pace, rows=rows), indent=2) + "\n")
    print(f"\n  wrote {out}")
    print("  min gap below the keep-out radius (0.24 m + margin) means the pass")
    print("  was not clean, whatever the pass counter says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
