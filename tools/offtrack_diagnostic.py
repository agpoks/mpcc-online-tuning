"""Is off-track a PLAN violation (soft corridor traded for progress) or PLANT drift off a good plan?

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/offtrack_diagnostic.py

The racing policy runs 45-57% off-track while driving BELOW the grip limit (0% over-speed), so it is
leaving the corridor laterally, not sliding out of a corner. The corridor is a SOFT constraint kept
0.20 m inside the edge (car_half_width 0.12 + corridor_safety 0.08), MORE conservative than the
plant's off rule (0.12 m). So a well-tracked plan should NOT go off. This drives solo with aggressive
fixed weights (high k_v/q_v) and, each tick, reads the MPCC's PLANNED trajectory (sv.get(k,'x')[:2])
and compares its corridor clearance to the plant's:
  - plan_off_clear = min over horizon of (edge clearance using the plant's 0.12 off-line)
  - plan_keep_clear = same using the MPCC's own 0.20 soft-keep line
  - plant_clear     = the realised clearance after stepping
  - step1_err       = |plant next (x,y) - plan stage-1 (x,y)|  (plan-plant tracking error)
Verdict: at the ticks the plant goes off, was the PLAN already predicting off (plan_off_clear<0 ->
(A) soft-corridor violation) or was the plan inside and the plant drifted (-> (B) model mismatch)?
Outputs results/race/stability/offtrack_diagnostic.{png,json}.
"""
import sys, os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
from mpcc_tuning.track import Track
from mpcc_tuning import baselines as B
from mpcc_tuning.acados_mpcc import AcadosMPCC
from mpcc_tuning.plant_scuderia import ScuderiaPlant
from experiments.race_mode import A_LAT_RACE

OUT = ROOT / "results/race/stability"; OUT.mkdir(parents=True, exist_ok=True)
track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
th0 = np.asarray(st.theta(), float)
m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic", q_vref=st.q_vref,
               discrete=True, max_obstacles=1, a_lat_sectors=[A_LAT_RACE] * 4, name="offtrack_diag")
N = m.N


def clearance(px, py, keep):
    """signed clearance to the nearer corridor edge, kept `keep` m inside (plant off uses 0.12)."""
    s = track.project(float(px), float(py)); lat = float(track.lateral(float(px), float(py)))
    wl, wr = track.width(track.wrap(s))
    return min(float(wr) - keep - lat, lat + float(wl) - keep)


def drive(kv, qv_mult, opp=True, steps=1500, gap0=3.5):
    from mpcc_tuning.opponents import RacelineOpponent
    from experiments.race_mode import FAIR_PACE, KEEPOUT_R, signed_gap
    theta = th0.copy()
    theta[7] = float(np.log(kv))                 # k_v (grip claim)
    theta[2] = th0[2] + float(np.log(qv_mult))   # q_v (progress) up
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=0.0, v0=1.4); m.reset()
    o = None
    if opp:                                       # an EQUAL-pace opponent a few m ahead
        o = RacelineOpponent(track, s0=gap0, pace=FAIR_PACE["equal"] * 1.43, offset=0.0,
                             radius=KEEPOUT_R, a_lat=2.5); o.reset()
        m.set_obstacles([o.keepout()])
    else:
        m.set_obstacles([(1e3, 1e3, 0.36)])
    rows = []
    for _ in range(steps):
        out = m.value(P.state_dyn(), theta); u = out["u0"]
        plan_off = min(clearance(m.sv.get(k, "x")[0], m.sv.get(k, "x")[1], 0.12) for k in range(1, N + 1))
        plan_keep = min(clearance(m.sv.get(k, "x")[0], m.sv.get(k, "x")[1], 0.20) for k in range(1, N + 1))
        p1 = np.asarray(m.sv.get(1, "x"), float)[:2]
        v = float(P._x[3])
        s5n, r, off, tr = P.step(u)
        ex, ey = float(P._x[0]), float(P._x[1]); pn = np.array([ex, ey])
        gap = 9.9; obsd = 9.9
        if o is not None:
            o.step(0.05, ego=(ex, ey, v)); ox, oy, _ = o.keepout()
            gap = signed_gap(track, track.project(ex, ey), o.s)
            obsd = float(np.hypot(ex - ox, ey - oy))          # body-to-keepout-centre distance
            m.set_obstacles([o.keepout()])
        rows.append(dict(v=v, plant_clear=clearance(ex, ey, 0.12), plan_off=plan_off, plan_keep=plan_keep,
                         step1_err=float(np.hypot(*(pn - p1))), gap=gap, obsd=obsd,
                         off=bool(off), status=out["status"]))
        if off or tr:
            break
    return rows


rows = drive(kv=0.70, qv_mult=1.6, opp=True)      # aggressive + an EQUAL opponent to overtake
n = len(rows)
offr = [r for r in rows if r["off"]]
plan_violates = [r for r in rows if r["plan_off"] < 0]
# of the ticks where the plant went off, was the plan already predicting off?
off_from_plan = [r for r in offr if r["plan_off"] < 0]
off_from_drift = [r for r in offr if r["plan_off"] >= 0]
keep_violations = [r for r in rows if r["plan_keep"] < 0]
mean_step1 = float(np.mean([r["step1_err"] for r in rows]))
print(f"solo aggressive drive: {n} ticks, plant went off at tick {n if offr else '-'}")
print(f"  ticks where PLAN predicts off-track (soft-corridor violated): {len(plan_violates)}/{n} ({100*len(plan_violates)/n:.0f}%)")
print(f"  ticks where PLAN violates its own 0.20 soft-keep line:        {len(keep_violations)}/{n} ({100*len(keep_violations)/n:.0f}%)")
print(f"  mean 1-step plan-vs-plant tracking error: {mean_step1*100:.1f} cm")
if offr:
    o_ = offr[-1]
    print(f"  at the OFF tick: plan_off_clear={o_['plan_off']*100:.1f} cm, gap-to-opp={o_['gap']:.2f} m, "
          f"obs-dist={o_['obsd']:.2f} m ({'PLAN planned off (A)' if o_['plan_off'] < 0 else 'plan inside, PLANT drifted (B)'})")
    print(f"  -> {'OFF near the opponent = OVERTAKE-INDUCED' if o_['obsd'] < 1.2 else 'OFF not near opponent'}")

fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
t = np.arange(n)
ax[0].plot(t, [r["plant_clear"] * 100 for r in rows], color="k", lw=1.6, label="PLANT clearance (realised)")
ax[0].plot(t, [r["plan_off"] * 100 for r in rows], color="tab:red", lw=1.2, label="PLAN min clearance (to 0.12 off-line)")
ax[0].plot(t, [r["plan_keep"] * 100 for r in rows], color="tab:orange", lw=1.0, ls="--", label="PLAN min clearance (to 0.20 soft-keep)")
ax[0].axhline(0, color="0.5", lw=1); ax[0].set_ylabel("corridor clearance [cm]")
ax[0].set_title("Off-track mechanism: does the PLAN (red) go below 0 before the PLANT (black)?  "
                "PLAN<0 = soft-corridor violation (A); PLAN>0 but PLANT<0 = drift (B)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(t, [r["v"] for r in rows], color="tab:blue", label="speed [m/s]")
ax[1].plot(t, [r["step1_err"] * 100 for r in rows], color="tab:green", label="1-step plan-vs-plant error [cm]")
ax[1].plot(t, [min(r["gap"], 6) for r in rows], color="tab:purple", lw=1, label="gap to opponent [m] (clipped 6)")
ax[1].plot(t, [min(r["obsd"], 6) for r in rows], color="tab:red", lw=1, ls="--", label="dist to keep-out centre [m]")
ax[1].axhline(0.36, color="tab:red", lw=0.6, alpha=0.5)
ax[1].set_xlabel("tick"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); [fig.savefig(OUT / f"offtrack_diagnostic.{e}", dpi=140) for e in ("png", "pdf")]
json.dump(dict(n_ticks=n, plan_violates_frac=len(plan_violates) / n, keep_violate_frac=len(keep_violations) / n,
               mean_step1_cm=mean_step1 * 100, went_off=bool(offr),
               verdict=("A: soft-corridor violation (plan leaves)" if plan_violates and len(off_from_plan) >= len(off_from_drift)
                        else "B: plant drifts off a valid plan" if offr else "did not go off in this drive")),
          open(OUT / "offtrack_diagnostic.json", "w"), indent=2)
print("saved results/race/stability/offtrack_diagnostic.{png,json}")
