"""Code-generate the MPCC to C, for running on the car.

    python -m mpcc_tuning.acados_export --vehicle dynamic --track oval \\
        --horizon 20 --dt 0.05 --out c_generated/mpcc_car

This is the deployment path, and it is deliberately separate from
:mod:`mpcc_tuning.acados_ocp`, which only *describes* the problem. Building a
solver has side effects -- it writes C, runs a compiler, and leaves a shared
library on disk -- and none of that belongs in a function the experiments import
at module scope.

Two modes, because the car and this laptop are not the same machine:

``--build`` (default on)
    Generate C *and* compile it here, giving an importable
    ``AcadosOcpSolver``. This is what you want to check that the generated
    problem actually solves before shipping it.

``--no-build``
    Generate C only. The tree under ``--out`` is self-contained and has its own
    ``Makefile``; copy it to the target and build there. This is the
    cross-compilation path, and it is why the generated code must not depend on
    anything in this repo at runtime -- once exported, the C is the controller.

## What the exported solver still does NOT carry

Recorded here rather than discovered on the vehicle:

* **The track.** ``spline_mode="parameter"`` passes the reference point per
  stage as a runtime parameter, so the C has no geometry compiled into it and
  something on the car must sample the path and fill ``p``. That is the mode
  that generates cleanly; ``"spline"`` compiles the path in and has not been
  exercised through code generation here.
* **The tuner.** ``theta`` is a runtime parameter, which is exactly what makes
  online tuning possible on the car -- the weights change without regenerating
  anything. But TD(lambda), the LTC and ``grad_theta`` are Python and stay off
  the generated side. ``grad_theta`` reads ``lam_g`` out of an IPOPT solution
  and has no acados equivalent written yet; see TODO.md.
* **Timing.** This module reports solve time when it builds, because a solver
  that generates cleanly and misses the tick budget is not a working
  controller. Read the worst case, not the mean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpcc_tuning.acados_ocp import build_ocp, pack_params  # noqa: E402
from mpcc_tuning.mpcc import MPCCWeights  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402


def export(track, out_dir, vehicle: str = "dynamic", horizon: int = 20,
           dt: float = 0.05, build: bool = True, max_obstacles: int = 0,
           name: str = "mpcc_car", variant: str = "fqp_soft_funnel",
           **ocp_kwargs):
    """Generate C for the MPCC into ``out_dir``; optionally compile it.

    Returns ``(ocp, solver_or_None)``. ``solver`` is None when ``build`` is
    False, because nothing was compiled to return.
    """
    from acados_template import AcadosOcpSolver

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Export the configuration that actually drives, not the defaults.
    # Measured on the oval: FIXED_STEP globalization gives 2.17 laps and
    # FUNNEL_L1PEN gives 7.12, beating IPOPT's 5.12 at 7.6 ms/tick against
    # 126 ms. Generating C for the default settings would ship the slow one.
    from mpcc_tuning.acados_variants import BY_NAME
    v = BY_NAME[variant]
    kw = dict(v.build_kwargs()); kw.update(ocp_kwargs)
    kw.setdefault("spline_mode", "spline")
    ocp = build_ocp(track, horizon=horizon, dt=dt, vehicle=vehicle,
                    max_obstacles=max_obstacles, name=name, **kw)
    missing = v.apply(ocp)
    if missing:
        print(f"  NOTE: this acados does not support {missing}; "
              f"the exported solver differs from the measured configuration.")
    ocp.code_export_directory = str(out / "c_generated_code")
    json_path = out / f"{name}.json"

    solver = AcadosOcpSolver(ocp, json_file=str(json_path),
                             generate=True, build=build)
    return ocp, solver


def _time_solve(solver, ocp, track, vehicle, horizon, dt, reps=200):
    """Solve from a rolling start and report the tick cost, worst case included."""
    nx = ocp.model.x.shape[0]
    th = MPCCWeights().to_log()
    x0 = np.zeros(nx)
    x0[:2] = np.asarray(track.pos(0.0)).ravel()
    x0[2] = float(track.tangent_angle(0.0))
    x0[3] = 2.0                                    # rolling, not standing
    # Pack to whatever the built solver expects. In spline mode the track is
    # compiled in and p is theta alone (8); in parameter mode the reference
    # point rides along (11). Assuming one gives
    # "trying to set 11 parameters ... has 8" and no timing at all.
    n_p = ocp.model.p.shape[0]
    if n_p == len(th):
        for k in range(horizon + 1):
            solver.set(k, "p", th)
    else:
        s_nodes = np.arange(horizon + 1) * dt * 2.0
        P = pack_params(th, track=track, s_nodes=s_nodes)
        for k in range(horizon + 1):
            solver.set(k, "p", P[k][:n_p])
    ts, oks = [], 0
    for _ in range(reps):
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        t0 = time.perf_counter()
        st = solver.solve()
        ts.append((time.perf_counter() - t0) * 1e3)
        oks += int(st == 0)
    ts = np.array(ts)
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)),
                worst_ms=float(ts.max()), ok_pct=100.0 * oks / reps)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="dynamic",
                    choices=("dynamic", "kinematic"),
                    help="the CONTROLLER's model. 'dynamic' is the one with "
                         "tyres and the one the car needs.")
    ap.add_argument("--track", default="oval")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--obstacles", type=int, default=0)
    ap.add_argument("--name", default="mpcc_car")
    ap.add_argument("--variant", default="fqp_soft_funnel",
                    help="solver configuration from mpcc_tuning.acados_variants")
    ap.add_argument("--out", default=str(ROOT / "c_generated"))
    ap.add_argument("--no-build", action="store_true",
                    help="generate C only, do not compile here -- the "
                         "cross-compile path for the vehicle")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args(argv)

    track = getattr(Track, a.track)()
    print(f"  {a.vehicle} model, {a.track}, N={a.horizon}, dt={a.dt}s "
          f"({a.horizon * a.dt:.2f}s horizon)", flush=True)
    ocp, solver = export(track, a.out, vehicle=a.vehicle, horizon=a.horizon,
                         dt=a.dt, build=not a.no_build, name=a.name,
                         max_obstacles=a.obstacles, variant=a.variant)
    nx, nu, npar = (ocp.model.x.shape[0], ocp.model.u.shape[0],
                    ocp.model.p.shape[0])
    print(f"  nx={nx} nu={nu} np={npar}", flush=True)
    print(f"  C written to {Path(a.out).resolve() / 'c_generated_code'}",
          flush=True)

    report = dict(vehicle=a.vehicle, track=a.track, horizon=a.horizon, dt=a.dt,
                  nx=nx, nu=nu, n_p=npar, built=not a.no_build)
    if a.no_build:
        print("  not compiled here. Copy the tree to the target and run its "
              "own Makefile.", flush=True)
    else:
        t = _time_solve(solver, ocp, track, a.vehicle, a.horizon, a.dt, a.reps)
        report["timing"] = t
        budget = 1000.0 * a.dt
        print(f"  solve: mean {t['mean_ms']:.2f} ms, p95 {t['p95_ms']:.2f} ms, "
              f"worst {t['worst_ms']:.2f} ms, ok {t['ok_pct']:.0f}%", flush=True)
        print(f"  tick budget at dt={a.dt}s is {budget:.0f} ms -- "
              f"{'FITS' if t['worst_ms'] < budget else 'DOES NOT FIT'} "
              f"on worst case", flush=True)
    (Path(a.out) / f"{a.name}_export.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
