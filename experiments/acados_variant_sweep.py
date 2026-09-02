"""Compare acados solver variants on the same closed-loop task.

    # on the installed acados
    PYTHONPATH=/path/to/scuderia_gym_jax python3 experiments/acados_variant_sweep.py

    # on a newer acados, without disturbing the installed one
    python3 experiments/acados_variant_sweep.py --acados-v053

The hypothesis under test:

    a newer acados with feasible-QP restoration should allow a HARD physical
    corridor -- no permanent slack on the track boundary -- while the local
    half-space representation may additionally improve real-time performance.

Every variant runs the identical closed loop: same plant, same track, same
weights, same horizon, same warm-start handling. The only thing that changes is
the solver configuration, which is what makes the comparison worth anything.

A variant the installed build cannot run is reported as **skipped**, never
silently downgraded -- an unsupported ``nlp_solver_type`` that quietly falls
back to ``SQP`` would be compared as though it were the thing it is not.
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

#: The reference every variant is measured against, both on the same problem.
REFERENCE = {
    "IPOPT (mpcc.py, converges)": "5.12 laps, 99% solve, 126 ms/tick",
    "acados baseline (v0.1.9)": "3.40 laps, 77% solve, 6.7 ms/tick",
}


def run_variant(variant, track_name="oval", horizon=12, dt=0.05, steps=1200,
                vehicle="dynamic", weights=None, seed=0):
    """One closed-loop run. Returns a result dict, or a skip/failure reason."""
    import numpy as np
    import casadi as ca
    from acados_template import AcadosOcpSolver
    from mpcc_tuning.acados_ocp import build_ocp
    from mpcc_tuning.acados_status import RunLog, evaluate
    from mpcc_tuning.acados_variants import capabilities, supported
    from mpcc_tuning.model import (ACCEL_MAX, SPEED_MAX, STEER_MAX,
                                   DynamicBicycle, KinematicBicycle)
    from mpcc_tuning.mpcc import MPCCWeights
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.track import Track

    caps = capabilities()
    if not supported(variant, caps):
        return dict(name=variant.name, skipped=True,
                    reason=f"{variant.nlp_solver_type} not in this build "
                           f"({caps['solver_types']})")

    t = getattr(Track, track_name)()
    dyn = vehicle == "dynamic"
    mdl = DynamicBicycle(dt=dt) if dyn else KinematicBicycle(dt=dt)
    N = horizon

    ocp = build_ocp(t, horizon=N, dt=dt, vehicle=vehicle, spline_mode="spline",
                    name=f"var_{variant.name}", **variant.build_kwargs())
    unsupported = variant.apply(ocp)
    # Unique per (variant, track, horizon, pid): parallel workers were all
    # writing c_generated/var_<name> and clobbering each other's shared
    # library mid-build, which showed up as "build failed: libacados_ocp
    # _solver_... not found" on runs whose own build had succeeded.
    export = (ROOT / "c_generated" /
              f"var_{variant.name}_{track_name}_{horizon}_{os.getpid()}")
    ocp.code_export_directory = str(export)
    try:
        sv = AcadosOcpSolver(ocp, json_file=str(export) + ".json",
                             generate=True, build=True)
    except Exception as exc:
        return dict(name=variant.name, skipped=True,
                    reason=f"build failed: {str(exc).splitlines()[0][:90]}")

    w = weights or dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05)
    th = MPCCWeights(**w).to_log()
    P = ScuderiaPlant(t, model="std", dt=dt, seed=seed); P.max_steps = steps
    P.reset()
    # Repeats must perturb something PHYSICAL. Seeding the plant's RNG changes
    # nothing here -- the dynamics are deterministic given a start state, so
    # three "repeats" were three identical runs and the reported std was 0.00
    # for every variant, which is not evidence of anything. Vary the starting
    # point and speed instead: same task, different initial condition.
    if seed:
        rng = np.random.default_rng(seed)
        s_start = float(rng.uniform(0.0, t.length))
        v_start = float(rng.uniform(1.0, 2.5))
        pos = np.asarray(t.pos(s_start)).ravel()
        psi = float(t.tangent_angle(s_start))
        xj = P._state.x
        xj = xj.at[:, 0].set(pos[0]).at[:, 1].set(pos[1])
        xj = xj.at[:, 4].set(psi).at[:, 3].set(v_start)
        for _i in range(7, P._state.x.shape[1]):
            xj = xj.at[:, _i].set(v_start / 0.031)
        P._state = P._state.replace(x=xj)
        P._x = np.asarray(P._state.x[0])
        P.s = s_start
    x0 = P.state_dyn() if dyn else P.state5()
    log = RunLog()
    margin = t.half_width - 0.12

    def corridor_params(s_nodes):
        """Per-stage half-space parameters, when the variant asks for them."""
        for k in range(N + 1):
            if not variant.lin_corridor:
                sv.set(k, "p", th); continue
            sk = float(s_nodes[k])
            ref = np.asarray(t.pos(sk)).ravel()
            ph = float(t.tangent_angle(sk))
            sv.set(k, "p", np.concatenate(
                [th, [np.sin(ph), -np.cos(ph), ref[0], ref[1]]]))

    # seed every stage by rolling the model forward: acados starts at x = 0,
    # which for the dynamic model is vx = 0, the worst-conditioned point.
    if dyn:
        xs, us = ca.MX.sym("x", 4 + mdl.n_dyn), ca.MX.sym("u", 2)
        stp = ca.Function("s", [xs, us], [mdl.step_sym(xs, us, dt)])
        xk = np.concatenate([x0[:4], x0[5:]]); ss = float(x0[4])
        for k in range(N + 1):
            sv.set(k, "x", np.concatenate([xk[:4], [ss], xk[4:]]))
            if k < N:
                sv.set(k, "u", np.array([0.0, 0.0, max(xk[3], 0.5)]))
                xk = np.asarray(stp(xk, [0.0, 0.0])).ravel()
                ss += max(xk[3], 0.5) * dt
    else:
        for k in range(N + 1):
            xk = x0.copy(); xk[4] = x0[4] + k * dt * max(x0[3], 0.5)
            sv.set(k, "x", xk)
    corridor_params(x0[4] + np.arange(N + 1) * dt * max(float(x0[3]), 1.0))

    lap = t.length; s0 = float(x0[4])
    nok = k = 0; vmax = bmax = 0.0; ts = []; off = tr = False
    for _ in range(steps):
        # The plant's own progress, passed through UNCHANGED -- exactly what
        # mpcc.py/IPOPT receives, and IPOPT drives 5.12 laps on it.
        #
        # This used to be overwritten with
        #     project(x, y) + lap * floor(s_virtual / lap)
        # to cancel the drift between the virtual progress and the car. It
        # raised the solve rate (60 -> 83%) and cost more than it bought:
        # 1.44 laps against 3.16 without it, measured.
        #
        # The reason is QUANTISATION, not the lap boundary. `project()` snaps
        # to the track's sample grid -- ds = 0.1 m on the oval -- so the
        # rebuilt s is a staircase: the car moves 4 cm and s does not move,
        # then it jumps 0.1 m. At 4 m/s the car covers 0.2 m per tick, so the
        # progress state was quantised at the same scale as its own motion,
        # corrupting the warm start, the lag error and the progress reward on
        # every tick.
        #
        # A full-lap backward jump at the start/finish line was the first
        # hypothesis and it is WRONG: instrumented over 887 ticks and six lap
        # crossings, zero jumps above 1 m. The fault was everywhere, not at the
        # line.
        #
        # The virtual s drifts but is MONOTONIC and continuous, which is what
        # the solver actually needs. IPOPT has always used it as-is.
        x0 = P.state_dyn() if dyn else P.state5()
        # RTI shift, with the terminal node EXTRAPOLATED rather than left
        # stale. Shifting 0..N-1 from 1..N and stopping leaves stage N holding
        # last tick's value, so stages N-1 and N coincide and the terminal cost
        # and constraint act on a degenerate node -- every tick. Rolling the
        # model forward one step gives the horizon end a real state.
        for j in range(N):
            sv.set(j, "x", sv.get(j + 1, "x"))
            if j < N - 1:
                sv.set(j, "u", sv.get(j + 1, "u"))
        xN1 = sv.get(N - 1, "x"); uN1 = sv.get(N - 1, "u")
        xd_ = np.concatenate([xN1[:4], xN1[5:]])
        xd_ = np.asarray(stp(xd_, uN1[:2])).ravel()
        sv.set(N, "x", np.concatenate(
            [xd_[:4], [xN1[4] + uN1[2] * dt], xd_[4:]]))
        if variant.lin_corridor:
            corridor_params(np.array([sv.get(j, "x")[4] for j in range(N + 1)]))
        sv.set(0, "lbx", x0); sv.set(0, "ubx", x0)
        a = time.perf_counter()
        st = sv.solve()
        ms = 1000.0 * (time.perf_counter() - a)
        u_applied = sv.get(0, "u")
        # corridor violation over the solver's OWN plan, in metres
        e_c = []
        for j in range(1, N + 1):
            xj = sv.get(j, "x")
            e_c.append(float(t.lateral(float(xj[0]), float(xj[1]))))
        log.add(evaluate(sv, st, u_applied, ms,
                         u_lb=[-STEER_MAX, -ACCEL_MAX, 0.0],
                         u_ub=[STEER_MAX, ACCEL_MAX, SPEED_MAX],
                         corridor=(e_c, margin)))
        ts.append(ms)
        nok += int(st == 0); k += 1
        s5, _r, off, tr = P.step(u_applied)
        vmax = max(vmax, float(s5[3]))
        if P._x.size > 6:
            bmax = max(bmax, abs(float(P._x[6])))
        if off or tr:
            break
    tm = log.timing()
    return dict(name=variant.name, skipped=False, seed=seed,
                laps=(float(P.state5()[4]) - s0) / lap, off=bool(off),
                ticks=k, peak_v=vmax, max_beta_deg=float(np.degrees(bmax)),
                # a USABLE step produced a finite, in-bounds control from a
                # non-fatal solve; converged_pct is the stricter status == 0
                usable_pct=log.usable_pct, converged_pct=log.converged_pct,
                max_iter_pct=log.max_iter_pct,
                alpha_mean=log.alpha_mean, rejected_pct=log.rejected_pct,
                globalization=variant.globalization,
                nlp_failures=log.nlp_failures, qp_failures=log.qp_failures,
                corridor_viol_max=log.corridor_violation_max,
                corridor_viol_mean=log.corridor_violation_mean,
                status_counts=log.status_counts(),
                ms_mean=tm["mean"], ms_p95=tm["p95"], ms_p99=tm["p99"],
                ms_max=tm["max"],
                hard_corridor=not variant.soft_corridor,
                lin_corridor=variant.lin_corridor,
                solver=variant.nlp_solver_type,
                iters=variant.nlp_solver_max_iter,
                unsupported_options=unsupported, note=variant.note)


def run_ipopt(track_name="oval", horizon=12, dt=0.05, steps=1200, seed=0,
              weights=None):
    """The CasADi/IPOPT reference, through the same measurement code."""
    import numpy as np
    from mpcc_tuning.acados_status import RunLog, StepLog
    from mpcc_tuning.model import (ACCEL_MAX, SPEED_MAX, STEER_MAX,
                                   DynamicBicycle)
    from mpcc_tuning.mpcc import MPCC, MPCCWeights
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.track import Track
    t = getattr(Track, track_name)()
    m = MPCC(t, model=DynamicBicycle(dt=dt), horizon=horizon, dt=dt,
             max_iter=300)
    w = weights or dict(q_c=1.0, q_l=50.0, q_v=1.0, r_d=0.5, r_a=0.05)
    th = MPCCWeights(**w).to_log()
    P = ScuderiaPlant(t, model="std", dt=dt, seed=seed); P.max_steps = steps
    P.reset(); m.reset()
    if seed:                       # same physical perturbation as the variants
        rng = np.random.default_rng(seed)
        s_start = float(rng.uniform(0.0, t.length))
        v_start = float(rng.uniform(1.0, 2.5))
        pos = np.asarray(t.pos(s_start)).ravel()
        psi = float(t.tangent_angle(s_start))
        xj = P._state.x
        xj = xj.at[:, 0].set(pos[0]).at[:, 1].set(pos[1])
        xj = xj.at[:, 4].set(psi).at[:, 3].set(v_start)
        for _i in range(7, P._state.x.shape[1]):
            xj = xj.at[:, _i].set(v_start / 0.031)
        P._state = P._state.replace(x=xj)
        P._x = np.asarray(P._state.x[0]); P.s = s_start
    log = RunLog(); margin = t.half_width - 0.12
    NS = m._NS; N = m.N
    s0 = float(P.state5()[4]); vmax = bmax = 0.0; off = tr = False; k = 0
    for _ in range(steps):
        st = P.state_dyn()
        a = time.perf_counter(); o = m.value(st, th)
        ms = 1000.0 * (time.perf_counter() - a)
        X = o["w"][:NS * (N + 1)].reshape(N + 1, NS)
        e_c = [float(t.lateral(float(X[j, 0]), float(X[j, 1])))
               for j in range(1, N + 1)]
        u = np.asarray(o["u0"], float)
        viol = max(0.0, float(np.max(np.abs(e_c))) - margin) if e_c else 0.0
        log.add(StepLog(status=0 if o["ok"] else 2, qp_status=0, sqp_iters=0,
                        qp_iters=0, u=u, solve_ms=ms, corridor_violation=viol,
                        usable=bool(np.isfinite(u).all())))
        k += 1
        s5, _r, off, tr = P.step(u)
        vmax = max(vmax, float(s5[3]))
        if P._x.size > 6: bmax = max(bmax, abs(float(P._x[6])))
        if off or tr: break
    tm = log.timing()
    return dict(name="IPOPT", skipped=False, seed=seed,
                laps=(float(P.state5()[4]) - s0) / t.length, off=bool(off),
                ticks=k, peak_v=vmax, max_beta_deg=float(np.degrees(bmax)),
                usable_pct=log.usable_pct, converged_pct=log.converged_pct,
                max_iter_pct=log.max_iter_pct, nlp_failures=log.nlp_failures,
                qp_failures=0,
                corridor_viol_max=log.corridor_violation_max,
                corridor_viol_mean=log.corridor_violation_mean,
                status_counts=log.status_counts(),
                ms_mean=tm["mean"], ms_p95=tm["p95"], ms_p99=tm["p99"],
                ms_max=tm["max"], hard_corridor=True, lin_corridor=False,
                solver="IPOPT", iters=300, unsupported_options=[], note="")


def aggregate(runs):
    """mean +- std over repeats, for every metric that varies."""
    import numpy as np
    keep = [r for r in runs if not r.get("skipped")]
    if not keep:
        return dict(skipped=True, reason=runs[0].get("reason", "?"),
                    name=runs[0]["name"], n=0)
    out = dict(name=keep[0]["name"], skipped=False, n=len(keep),
               globalization=keep[0].get("globalization", "FIXED_STEP"),
               solver=keep[0]["solver"], hard_corridor=keep[0]["hard_corridor"],
               lin_corridor=keep[0]["lin_corridor"], iters=keep[0]["iters"],
               note=keep[0]["note"])
    for f in ("laps", "usable_pct", "converged_pct", "max_iter_pct",
              "alpha_mean", "rejected_pct",
              "nlp_failures", "qp_failures", "corridor_viol_max",
              "corridor_viol_mean", "ms_mean", "ms_p95", "ms_p99", "ms_max",
              "peak_v", "ticks"):
        v = np.array([r[f] for r in keep], float)
        out[f + "_mean"] = float(v.mean())
        out[f + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["off_rate"] = float(np.mean([r["off"] for r in keep]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--vehicle", default="dynamic",
                    choices=("dynamic", "kinematic"))
    ap.add_argument("--track", default="oval")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per variant; results are reported mean +- std")
    ap.add_argument("--with-ipopt", action="store_true",
                    help="include the CasADi/IPOPT reference, measured the "
                         "same way rather than quoted from memory")
    ap.add_argument("--acados-v053", action="store_true")
    ap.add_argument("--acados-v060", action="store_true",
                    help="the newest acados; same isolation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.acados_v053 or a.acados_v060:
        from mpcc_tuning.acados_variants import ACADOS_V053, ACADOS_V060
        sel = ACADOS_V060 if a.acados_v060 else ACADOS_V053
        if os.environ.get("_ACADOS_SWITCHED") != "1":
            env = dict(os.environ, _ACADOS_SWITCHED="1")
            env["ACADOS_SOURCE_DIR"] = sel["ACADOS_SOURCE_DIR"]
            for k in ("LD_LIBRARY_PATH", "PYTHONPATH"):
                env[k] = sel[k] + os.pathsep + env.get(k, "")
            os.execve(sys.executable, [sys.executable, *sys.argv], env)

    from mpcc_tuning.acados_variants import BY_NAME, VARIANTS, capabilities
    caps = capabilities()
    print(f"  acados solver types {caps['solver_types']}")
    print(f"  {a.repeats} repeats per variant, {a.steps} steps, "
          f"{a.vehicle} model on {a.track}\n", flush=True)

    picked = ([BY_NAME[n] for n in a.variants] if a.variants else list(VARIANTS))
    rows, raw = [], []
    for v in picked:
        runs = [run_variant(v, track_name=a.track, horizon=a.horizon,
                            steps=a.steps, vehicle=a.vehicle, seed=sd)
                for sd in range(a.repeats)]
        raw += runs
        rows.append(aggregate(runs))
        r = rows[-1]
        if r["skipped"]:
            print("  %-22s SKIPPED %s" % (r["name"], r["reason"]), flush=True)
        else:
            print("  %-22s %5.2f +-%4.2f laps   usable %3.0f%%   conv %3.0f%%"
                  % (r["name"], r["laps_mean"], r["laps_std"],
                     r["usable_pct_mean"], r["converged_pct_mean"]), flush=True)
    if a.with_ipopt:
        runs = [run_ipopt(a.track, a.horizon, 0.05, a.steps, sd)
                for sd in range(a.repeats)]
        raw += runs; rows.append(aggregate(runs))
        r = rows[-1]
        print("  %-22s %5.2f +-%4.2f laps   usable %3.0f%%   conv %3.0f%%"
              % (r["name"], r["laps_mean"], r["laps_std"],
                 r["usable_pct_mean"], r["converged_pct_mean"]), flush=True)

    print()
    print("  %-26s %13s %7s %6s %6s %5s %5s %7s %8s %8s %8s %8s" %
          ("variant", "laps", "usbl%", "conv%", "alpha", "rej%", "NLPf",
           "QPf", "corr m", "ms mean", "ms p99", "ms max"))
    for r in rows:
        if r["skipped"]:
            continue
        print("  %-26s %6.2f +-%4.2f %6.0f%% %5.0f%% %6.3f %4.0f%% %5.1f %7.1f %8.3f %8.2f %8.1f %8.1f"
              % (r["name"], r["laps_mean"], r["laps_std"],
                 r["usable_pct_mean"], r["converged_pct_mean"],
                 r.get("alpha_mean_mean", 1.0), r.get("rejected_pct_mean", 0.0),
                 r["nlp_failures_mean"], r["qp_failures_mean"],
                 r["corridor_viol_max_mean"], r["ms_mean_mean"],
                 r["ms_p99_mean"], r["ms_max_mean"]))
    out = a.out or str(ROOT / "benchmarks" / "results"
                       / f"acados_variants_{a.vehicle}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ran = [r for r in rows if not r["skipped"]]
    Path(out).write_text(json.dumps(
        dict(acados=caps, vehicle=a.vehicle, track=a.track,
             horizon=a.horizon, repeats=a.repeats,
             n_ran=len(ran), n_skipped=len(rows) - len(ran),
             summary=rows, runs=raw), indent=2, default=str) + "\n")
    print(f"\n  wrote {out}")

    # A sweep in which NOTHING ran is a failed sweep, not a finished one.
    # Twice now an entire run has been skipped -- once on a missing
    # link_libs.json, once on an invalid CasADi name -- and both times the
    # process exited 0 and the completion notification read as success. The
    # numbers were absent, not bad, and absent numbers are easy to mistake for
    # "still going". Exit non-zero so that cannot happen quietly again.
    if not ran:
        print("\n  SWEEP FAILED: 0 of %d variants produced results." % len(rows))
        for r in rows:
            print("    %-26s %s" % (r["name"], r.get("reason", "?")))
        print("  This is a harness or build failure, NOT a measurement.")
        return 1
    if len(ran) < len(rows):
        print("\n  NOTE: %d of %d variants skipped -- results below are partial."
              % (len(rows) - len(ran), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
