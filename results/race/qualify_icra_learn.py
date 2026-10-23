#!/usr/bin/env python3
"""QUALIFICATION online-learning on the ICRA track (icra_t2_smooth): the on-car PolicyTuner
(RTRL + per-lap validated keep-best) learns the fastest CONSECUTIVE clean lap, solo. Starts from
the deployed icra net and refines it live. Logs lap times, weights, 2D trajectory, speed, per-tick
reward and off-track reverts -> results/race/qualify_icra_learn.npz (data-only; figure built by
qualify_icra_fig.py). We do NOT learn from crashing: an off-track ends the episode with a -5 penalty
and the keep-best reverts the lap, so crashes are undone, not trained on.
Reproduce:  python3 results/race/qualify_icra_learn.py 3600   # 3 min at 20 Hz
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
# --- select the acados build that has SQP_WITH_FEASIBLE_QP + p_global (the real deploy solver) ---
# The default ~/acados is v0.1.9: no feasible-QP, and it cannot codegen the theta_global cost (the
# weights come out as free CasADi symbols). The paper pipeline runs on ~/acados-v060, chosen purely
# through the environment (mpcc_tuning/acados_variants.ACADOS_V060). scuderia_gym_jax is the STD drift
# plant (the one allowed external, see no-cross-repo rule). Re-exec once with the env set so a bare
# `python3 results/race/qualify_icra_learn.py 3600` builds the deploy controller regardless of the shell.
_V060 = os.path.expanduser("~/acados-v060")
if os.environ.get("ACADOS_SOURCE_DIR") != _V060:
    _env = dict(os.environ)
    _env["ACADOS_SOURCE_DIR"] = _V060
    _env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (_V060 + "/lib", _env.get("LD_LIBRARY_PATH", "")) if p)
    _env["PYTHONPATH"] = os.pathsep.join(p for p in (
        _V060 + "/interfaces/acados_template", os.path.expanduser("~/github/scuderia_gym_jax"),
        _env.get("PYTHONPATH", "")) if p)
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
from pathlib import Path
import numpy as np
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from mpcc_tuning.mpcc import WEIGHT_NAMES
from node.mpcc_online.controller import OnlineMPCCController

OUT = ROOT / "results/race"
START_NET = str(ROOT / "results/race_multicar/qualification/best_policy_icra_t2_smooth_2_time_return.npz")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3600


# The real deploy build: SQP_WITH_FEASIBLE_QP + two-layer corridor + theta_global weights, on the
# default icra_t2_smooth track (no raceline_npz). The env header above guarantees the acados that
# supports it; no fallback -- if this raises, the env is wrong, not the model.
ctl = OnlineMPCCController(START_NET, cfg=dict(learn=True, kv_launch=0.45))
L = ctl.track.length
P = ScuderiaPlant(ctl.track, model="std", dt=0.05); P.max_steps = STEPS + 20
P.reset(s0=0.0, v0=1.0); ctl.reset()

th_hist, xy, vh, sv, offv, rew = [], [], [], [], [], []
lap_end_tick, lap_banked = [], []
s_prev, total_prog, off_ticks = None, 0.0, 0
for k in range(STEPS):
    x, y, psi, vx, s, vy, r, delta = (float(q) for q in P.state_dyn())
    out = ctl.control(x, y, psi, vx, vy, r, delta)
    th_hist.append(out["theta"]); xy.append((x, y)); vh.append(float(np.hypot(vx, vy))); sv.append(out["solve_ms"])
    offf = int(out["off"]); offv.append(offf); off_ticks += offf
    ds = 0.0 if s_prev is None else ((s - s_prev + L / 2) % L - L / 2)   # wrapped metres of progress
    s_prev = s; total_prog += max(ds, 0.0)
    rew.append(ds - 5.0 * offf)                                          # reward = progress - 5*off (eq. in paper)
    if out["banked"] is not None:
        lap_end_tick.append(k); lap_banked.append(out["banked"])
    u = out["u0"]
    if u is None:
        u = np.array([0.0, -2.0, 0.0])
    P.step(u)

th_hist = np.array(th_hist); xy = np.array(xy); vh = np.array(vh)
sv = np.array(sv); offv = np.array(offv, bool); rew = np.array(rew)
laps = total_prog / L
lt = np.diff([0] + lap_end_tick).astype(float) * ctl.dt if lap_end_tick else np.array([])
banks = sum(b == "banked" for b in lap_banked)

np.savez(OUT / "qualify_icra_learn.npz", th_hist=th_hist, xy=xy, vh=vh, solve=sv, off=offv, reward=rew,
         lap_end_tick=np.array(lap_end_tick), lap_banked=np.array(lap_banked, dtype=object), lap_times=lt,
         laps=laps, banks=banks, weight_names=np.array(WEIGHT_NAMES),
         start_net=Path(START_NET).name, dt=ctl.dt, length=L)
print(f"QUALIFY icra online-learn: steps={STEPS} ({STEPS*ctl.dt/60:.1f} min)  laps={laps:.2f}  "
      f"off_ticks={off_ticks}/{STEPS}  keep-best banks={banks}/{len(lap_banked)}")
print(f"  lap times [s]: {[round(x,1) for x in lt.tolist()]}")
print(f"  solve p50={np.percentile(sv,50):.0f} p95={np.percentile(sv,95):.0f} ms   speed mean={vh.mean():.2f} m/s")
print(f"  wrote {OUT}/qualify_icra_learn.npz")
