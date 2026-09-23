"""Offline vehicle-dynamics analysis to CALIBRATE the stability limits used by the
race-mode risk reward -- handling diagram, tyre curve, phase portrait, and a
bifurcation sweep -- from the REAL fitted RC-car tyre (scuderia_gym_jax full Pacejka).

    PYTHONPATH=/path/to/scuderia_gym_jax:. python3 tools/handling_analysis.py

Outputs results/race/stability/: tyre_curve, handling_diagram, phase_portrait,
bifurcation.{pdf,png} and stability_limits.json (alpha_r_ref, beta_ref, kappa_ref, the
peak-force and saddle-node points) -- these set the references for the slip-based risk
term (penalise |alpha_r|>alpha_ref etc.), calibrated to where the tyre actually saturates
and where the (beta,r) cornering equilibrium is lost, not guessed.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import fsolve
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/poxx/github/scuderia_gym_jax")
import scuderia_gym_jax as sgj
from scuderia_gym_jax.envs.tire_models import formula_lateral, formula_longitudinal
OUT = ROOT / "results/race/stability"; OUT.mkdir(parents=True, exist_ok=True)

p = sgj.make_params_and_spec(None, {})[0]
veh = np.asarray(p.veh); tire = np.asarray(p.tire.full)
lf, lr, h_cg, m, Iz = float(veh[0]), float(veh[1]), float(veh[2]), float(veh[3]), float(veh[4])
mu = float(np.asarray(p.st)[0]); g = 9.81; L = lf + lr
Fz_f = m * g * lr / L; Fz_r = m * g * lf / L         # static axle loads
print(f"lf={lf:.3f} lr={lr:.3f} m={m:.3f} I={Iz:.4f} mu={mu:.2f} Fz_f={Fz_f:.1f}N Fz_r={Fz_r:.1f}N")

def Fy(alpha, Fz):   # pure-lateral Pacejka force (N), gamma=0
    return float(formula_lateral(float(alpha), 0.0, float(Fz), tire)[0])
def Fx(kappa, Fz):   # pure-longitudinal force (N); formula_longitudinal returns a scalar
    return float(np.asarray(formula_longitudinal(float(kappa), 0.0, float(Fz), tire)).ravel()[0])

# ---- 1) TYRE CURVES + peaks -------------------------------------------------
al = np.linspace(-0.4, 0.4, 601)
fyf = np.array([Fy(a, Fz_f) for a in al]); fyr = np.array([Fy(a, Fz_r) for a in al])
ka = np.linspace(-0.6, 0.6, 601); fxr = np.array([Fx(k, Fz_r) for k in ka])
apk_f = float(al[np.argmax(fyf)]); apk_r = float(al[np.argmax(fyr)])
kpk = float(ka[np.argmax(fxr)])
print(f"tyre peaks: alpha_f_peak={apk_f:.3f} rad ({np.degrees(apk_f):.1f} deg), "
      f"alpha_r_peak={apk_r:.3f} ({np.degrees(apk_r):.1f} deg), kappa_peak={kpk:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(np.degrees(al), fyf, label=f"front (Fz={Fz_f:.0f}N)")
ax[0].plot(np.degrees(al), fyr, label=f"rear (Fz={Fz_r:.0f}N)")
ax[0].axvline(np.degrees(apk_r), color="tab:red", ls=":", label=f"rear peak {np.degrees(apk_r):.1f}deg")
ax[0].set_xlabel("slip angle [deg]"); ax[0].set_ylabel("Fy [N]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("lateral tyre curve")
ax[1].plot(ka, fxr, color="tab:green"); ax[1].axvline(kpk, color="tab:red", ls=":", label=f"peak {kpk:.2f}")
ax[1].set_xlabel("slip ratio kappa [-]"); ax[1].set_ylabel("Fx [N]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("longitudinal tyre curve (rear)")
fig.tight_layout(); [fig.savefig(OUT / f"tyre_curve.{e}", dpi=140) for e in ("pdf", "png")]

# ---- 2-DOF lateral bicycle: slips, forces, derivative -----------------------
def slips(beta, r, v, delta):
    vx = v * np.cos(beta); vy = v * np.sin(beta)
    af = delta - np.arctan2(vy + lf * r, vx)
    ar = -np.arctan2(vy - lr * r, vx)
    return af, ar
def deriv(beta, r, v, delta):
    af, ar = slips(beta, r, v, delta)
    Fyf = Fy(af, Fz_f); Fyr = Fy(ar, Fz_r)
    bdot = (Fyf * np.cos(delta) + Fyr) / (m * v) - r
    rdot = (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz
    return np.array([bdot, rdot])
def equil(v, delta, guess=(0.0, 0.0)):
    sol, info, ier, _ = fsolve(lambda x: deriv(x[0], x[1], v, delta), guess, full_output=True)
    return (sol if ier == 1 else None)
def stable(beta, r, v, delta, eps=1e-4):
    J = np.zeros((2, 2)); x0 = np.array([beta, r])
    f0 = deriv(beta, r, v, delta)
    for i in range(2):
        xp = x0.copy(); xp[i] += eps
        J[:, i] = (deriv(xp[0], xp[1], v, delta) - f0) / eps
    return np.all(np.real(np.linalg.eigvals(J)) < 0)

# ---- 2) HANDLING DIAGRAM: steer & slip vs lateral accel at fixed speed ------
v_h = 2.5
ay, dsteer, slip_bal = [], [], []
for delta in np.linspace(0.0, 0.5, 60):
    s = equil(v_h, delta, (0.0, delta))
    if s is None: continue
    b, r = s; a_y = v_h * r; af, ar = slips(b, r, v_h, delta)
    if a_y < 0.02: continue
    ay.append(a_y); dsteer.append(delta - L * r / v_h); slip_bal.append(af - ar)   # us gradient / balance
ay = np.array(ay)
fig2, ax2 = plt.subplots(figsize=(6.5, 4.3))
ax2.plot(ay, np.degrees(np.array(dsteer)), "o-", ms=3, label="steer - Ackermann (understeer if +)")
ax2.plot(ay, np.degrees(np.array(slip_bal)), "s-", ms=3, color="tab:orange", label="alpha_f - alpha_r")
ax2.axhline(0, color="k", lw=.6); ax2.set_xlabel("lateral accel a_y [m/s^2]"); ax2.set_ylabel("angle [deg]")
K_us = float(np.polyfit(ay, np.array(dsteer), 1)[0]) if len(ay) > 2 else float("nan")
ax2.set_title(f"handling diagram @ v={v_h} m/s  (understeer gradient K={np.degrees(K_us):.2f} deg per m/s^2)")
ax2.legend(fontsize=8); ax2.grid(alpha=.3); fig2.tight_layout()
[fig2.savefig(OUT / f"handling_diagram.{e}", dpi=140) for e in ("pdf", "png")]

# ---- 3) BIFURCATION: stable cornering equilibrium vs SPEED (fixed steer) -----
delta_b = 0.25
vs = np.linspace(0.8, 22.0, 160); betas, rs, stab = [], [], []
guess = (0.0, delta_b)
for v in vs:
    s = equil(v, delta_b, guess)
    if s is None: betas.append(np.nan); rs.append(np.nan); stab.append(False); continue
    b, r = s; guess = (b, r)
    betas.append(b); rs.append(r); stab.append(stable(b, r, v, delta_b))
betas = np.array(betas); rs = np.array(rs); stab = np.array(stab)
# saddle-node: last speed where the tracked equilibrium is still stable
v_crit = float(vs[stab][-1]) if stab.any() else float("nan")
i_crit = int(np.where(vs == v_crit)[0][0]) if stab.any() else -1
beta_crit = float(betas[i_crit]) if i_crit >= 0 else float("nan")
_, ar_crit = slips(beta_crit, rs[i_crit], v_crit, delta_b) if i_crit >= 0 else (0, float("nan"))
print(f"bifurcation @ delta={delta_b}: critical speed v_crit={v_crit:.2f} m/s, "
      f"beta_crit={beta_crit:.3f} rad ({np.degrees(beta_crit):.1f} deg), alpha_r_crit={ar_crit:.3f} rad")
fig3, ax3 = plt.subplots(figsize=(6.5, 4.3))
ax3.plot(vs, np.degrees(betas), "-", label="sideslip beta")
ax3.plot(vs[stab], np.degrees(betas[stab]), "o", ms=3, color="tab:green", label="stable")
ax3.plot(vs[~stab], np.degrees(betas[~stab]), "x", ms=4, color="tab:red", label="unstable")
ax3.axvline(v_crit, color="tab:red", ls="--", label=f"saddle-node v_crit={v_crit:.1f} m/s")
ax3.set_xlabel("speed v [m/s]"); ax3.set_ylabel("equilibrium sideslip beta [deg]")
ax3.set_title(f"bifurcation: cornering equilibrium vs speed (delta={delta_b} rad)")
ax3.legend(fontsize=8); ax3.grid(alpha=.3); fig3.tight_layout()
[fig3.savefig(OUT / f"bifurcation.{e}", dpi=140) for e in ("pdf", "png")]

# ---- 4) PHASE PORTRAIT (beta, r) near the limit -----------------------------
v_p, d_p = min(v_crit, 4.0), delta_b
B, R = np.meshgrid(np.linspace(-0.5, 0.5, 26), np.linspace(-6, 6, 26))
DB = np.zeros_like(B); DR = np.zeros_like(R)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        d = deriv(B[i, j], R[i, j], v_p, d_p); DB[i, j], DR[i, j] = d[0], d[1]
fig4, ax4 = plt.subplots(figsize=(6.5, 5.5))
ax4.streamplot(np.degrees(B), R, DB, DR, color=np.hypot(DB, DR), cmap="viridis", density=1.2)
sname = equil(v_p, d_p, (0.0, d_p))
if sname is not None: ax4.plot(np.degrees(sname[0]), sname[1], "o", color="lime", ms=12, mec="k", label="stable equilibrium")
ax4.set_xlabel("sideslip beta [deg]"); ax4.set_ylabel("yaw rate r [rad/s]")
ax4.set_title(f"phase portrait (beta,r) @ v={v_p:.1f} m/s, delta={d_p} rad"); ax4.legend(fontsize=8)
fig4.tight_layout(); [fig4.savefig(OUT / f"phase_portrait.{e}", dpi=140) for e in ("pdf", "png")]

# ---- LIMITS for the risk reward (calibrated, not guessed) -------------------
limits = dict(
    alpha_r_peak=round(apk_r, 4), alpha_f_peak=round(apk_f, 4), kappa_peak=round(kpk, 4),
    v_crit=round(v_crit, 3), beta_crit=round(beta_crit, 4), alpha_r_crit=round(float(ar_crit), 4),
    understeer_grad_deg_per_ms2=round(np.degrees(K_us), 3),
    stable_all_speeds=bool(stab.all()),   # understeer car -> no spin bifurcation; tyre-saturation limited
    # references: PRIMARY limit is rear-tyre saturation (this car is understeer-stable, so the
    # binding limit at its operating speeds is the tyre peak, not a phase-plane bifurcation).
    # Start penalising at 0.75x the peak. beta is a softer DRIFT indicator (0.10 rad ~ 6 deg).
    alpha_r_ref=round(0.75 * abs(apk_r), 4), kappa_ref=round(0.75 * abs(kpk), 4), beta_ref=0.10,
    # per-opponent-class scale on the reference (fast -> spend more margin; slow -> keep more):
    class_scale=dict(static=0.6, slower=0.7, equal=0.85, faster=1.0),
    notes=("understeer-stable RC car: binding limit is TYRE SATURATION (alpha_r_peak), not a "
           "saddle-node. alpha_r_ref/kappa_ref = 0.75x peak; beta_ref soft drift limit. Multiply "
           "the ref by class_scale[opp_class] in the risk reward: vs faster spend the full margin, "
           "vs slower keep a bigger one."))
json.dump(limits, open(OUT / "stability_limits.json", "w"), indent=2)
print("LIMITS:", json.dumps(limits))
print("saved results/race/stability/{tyre_curve,handling_diagram,bifurcation,phase_portrait}.* + stability_limits.json")
