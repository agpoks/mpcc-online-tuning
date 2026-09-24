"""Online tyre-curve / slip-limit estimation from the car's OWN measurements -- EKF + GP.

    PYTHONPATH=/path/to/scuderia_gym_jax:. python3 tools/tire_estimation.py

The offline handling analysis (tools/handling_analysis.py) computed the slip limit from the
KNOWN Pacejka tyre model. On a real car with unknown friction you do not have that model -- you
have IMU + wheel + steering measurements. This shows the limit can instead be RECOVERED ONLINE:

1. Drive the STD4W plant through ramp-steer manoeuvres at several speeds (excites rear slip from
   0 up toward the peak).
2. Reconstruct the rear lateral tyre force from measurements exactly as a real car would --
   Fyr = (lf*m*a_y - I*r_dot)/L, with a_y = vy_dot + vx*r and r_dot from the IMU/yaw-rate (here
   finite-differenced from the logged state, i.e. noisy on purpose).
3. Fit the curve two ways:
   - EKF: recursive estimation of (cornering stiffness K, peak force D) from the (alpha_r, Fyr)
     stream -- shows the estimate converging tick by tick, as it would online.
   - GP: a Gaussian-Process fit alpha_r -> Fyr with UNCERTAINTY, so the safe limit is the lower
     confidence bound (relax the envelope only where the data supports it).
4. Compare the estimated peak slip to the TRUE peak (0.148 rad) from the real tyre.

Outputs results/race/stability/tyre_estimation.{pdf,png} + tyre_estimation.json.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/poxx/github/mpcc-online-tuning"); sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/poxx/github/scuderia_gym_jax")
from mpcc_tuning.track import Track
from mpcc_tuning.plant_scuderia import ScuderiaPlant
import scuderia_gym_jax as sgj
from scuderia_gym_jax.envs.tire_models import formula_lateral
OUT = ROOT / "results/race/stability"; OUT.mkdir(parents=True, exist_ok=True)

p = sgj.make_params_and_spec(None, {})[0]
veh = np.asarray(p.veh); tire = np.asarray(p.tire.full)
lf, lr, m, Iz = float(veh[0]), float(veh[1]), float(veh[3]), float(veh[4])
mu = float(np.asarray(p.st)[0]); g = 9.81; L = lf + lr
Fz_r = m * g * lf / L
def Fy_true(alpha):  # ground-truth rear tyre curve (for comparison only)
    return float(formula_lateral(float(alpha), 0.0, Fz_r, tire)[0])
al_grid = np.linspace(0.0, 0.35, 400)
fy_true = np.array([Fy_true(a) for a in al_grid])
alpha_peak_true = float(al_grid[np.argmax(fy_true)]); D_true = float(fy_true.max())
print(f"true rear tyre: alpha_peak={alpha_peak_true:.3f} rad, D=mu*Fz_r={D_true:.1f} N")

# ---- 1) drive ramp-steer manoeuvres, log v, r, beta, delta -----------------
track = Track.icra_t2_smooth()
ALr, FYRr, ALf, FYRf = [], [], [], []      # rear and FRONT (slip angle, reconstructed force)
for v0 in (2.5, 3.3, 4.1, 4.8, 5.5):
    P = ScuderiaPlant(track, model="std", dt=0.02); P.max_steps = 500
    P.reset(s0=0.0, v0=v0)
    prev_vy = None; prev_r = None
    for kstep in range(500):
        v = float(P._x[3])
        steer = min(0.6, 0.002 * kstep)              # steeper steering ramp 0 -> 0.6 rad
        acc = 4.0 * (v0 - v) + 1.0                    # hold speed HARD (keep loading the tyre)
        P.step(np.array([steer, acc, v]))            # u = [steer, accel, v_s] (v_s unused here)
        v = float(P._x[3]); r = float(P._x[5]); beta = float(P._x[6]); delta = float(P._x[2])
        vx = v * np.cos(beta); vy = v * np.sin(beta)
        if prev_vy is not None and vx > 0.5:
            a_y = (vy - prev_vy) / 0.02 + vx * r      # IMU lateral specific force
            r_dot = (r - prev_r) / 0.02
            Fyr = (lf * m * a_y - Iz * r_dot) / L     # rear force, reconstructed from measurements
            Fyf = (Iz * r_dot + lr * m * a_y) / (L * max(np.cos(delta), 0.3))   # front force
            alpha_r = -np.arctan2(vy - lr * r, vx)
            alpha_f = delta - np.arctan2(vy + lf * r, vx)
            if 0.0 <= alpha_r < 0.34 and abs(Fyr) < 60:
                ALr.append(alpha_r); FYRr.append(Fyr)
            if 0.0 <= alpha_f < 0.34 and abs(Fyf) < 60:
                ALf.append(alpha_f); FYRf.append(Fyf)
        prev_vy, prev_r = vy, r
        if not np.isfinite(P._x).all():
            break
AL = np.array(ALr); FYR = np.array(FYRr)
ALF = np.array(ALf); FYRF = np.array(FYRf)
print(f"collected rear {len(AL)} samples (alpha_r up to {AL.max():.3f} rad), "
      f"front {len(ALF)} samples (alpha_f up to {ALF.max():.3f} rad)")

C = 1.4                                              # fixed Pacejka shape factor
def h_tyre(alpha, K, D):
    B = K / (C * max(D, 1.0))
    return D * np.sin(C * np.arctan(B * alpha))
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck

def fit_tire(A, F, Fz_axle):
    """EKF (K,D) + GP fit of one tyre's (slip -> force) from reconstructed samples."""
    x = np.array([200.0, 20.0]); Pk = np.diag([1e4, 1e2]); Q = np.diag([1e-1, 1e-2]); Rn = 16.0
    apk_hist = []
    for i in np.argsort(A):                          # feed low->high slip (as a sweep arrives)
        a, z = A[i], F[i]; Pk = Pk + Q; hh = h_tyre(a, x[0], x[1]); eps = 1e-3
        H = np.array([(h_tyre(a, x[0] + eps, x[1]) - hh) / eps, (h_tyre(a, x[0], x[1] + eps) - hh) / eps])
        Kg = Pk @ H / (H @ Pk @ H + Rn)
        x = x + Kg * (z - hh); x[0] = max(x[0], 10.0); x[1] = max(x[1], 5.0)
        Pk = (np.eye(2) - np.outer(Kg, H)) @ Pk
        apk_hist.append(np.tan(np.pi / (2 * C)) / max(x[0] / (C * x[1]), 1e-6))
    K_e, D_e = float(x[0]), float(x[1]); apk_e = float(np.tan(np.pi / (2 * C)) / (K_e / (C * D_e)))
    sub = np.random.default_rng(0).choice(len(A), size=min(250, len(A)), replace=False)
    kern = Ck(20.0, (1.0, 200.0)) * RBF(0.06, (0.01, 0.3)) + WhiteKernel(4.0, (0.5, 50.0))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2).fit(A[sub][:, None], F[sub])
    mu_gp, sd_gp = gp.predict(al_grid[:, None], return_std=True)
    apk_gp = float(al_grid[np.argmax(mu_gp)]); apk_gp_lb = float(al_grid[np.argmax(mu_gp - 2 * sd_gp)])
    fy_tr = np.array([float(formula_lateral(float(a), 0.0, Fz_axle, tire)[0]) for a in al_grid])
    return dict(K=K_e, D=D_e, apk_e=apk_e, mu_gp=mu_gp, sd_gp=sd_gp, apk_gp=apk_gp, apk_gp_lb=apk_gp_lb,
                apk_hist=apk_hist, fy_true=fy_tr, apk_true=float(al_grid[np.argmax(fy_tr)]), a_max=float(A.max()))

Fz_f = m * g * lr / L
RE = fit_tire(AL, FYR, Fz_r); FR = fit_tire(ALF, FYRF, Fz_f)
for nm, r in (("REAR", RE), ("FRONT", FR)):
    print(f"{nm}: true peak {np.degrees(r['apk_true']):.1f}deg | EKF {np.degrees(r['apk_e']):.1f}deg | "
          f"GP {np.degrees(r['apk_gp']):.1f}deg (lb {np.degrees(r['apk_gp_lb']):.1f}) | drove to {np.degrees(r['a_max']):.1f}deg")

# ---- plot: front (reaches its limit) vs rear (does not -> extrapolate) ------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for col, (nm, A, F, r) in enumerate((("FRONT (reaches its limit)", ALF, FYRF, FR),
                                      ("REAR (understeer -> never reaches limit)", AL, FYR, RE))):
    a0, a1 = ax[0, col], ax[1, col]
    a0.scatter(np.degrees(A), F, s=6, alpha=0.2, color="0.5", label="reconstructed (from a_y, r)")
    a0.axvspan(0, np.degrees(r["a_max"]), color="tab:green", alpha=0.06, label="slip range actually driven")
    a0.plot(np.degrees(al_grid), r["fy_true"], "k-", lw=2, label=f"TRUE (peak {np.degrees(r['apk_true']):.1f}deg)")
    a0.plot(np.degrees(al_grid), [h_tyre(a, r["K"], r["D"]) for a in al_grid], "--", color="tab:blue", lw=2,
            label=f"EKF (peak {np.degrees(r['apk_e']):.1f}deg)")
    a0.plot(np.degrees(al_grid), r["mu_gp"], "-", color="tab:orange", lw=2, label=f"GP mean (peak {np.degrees(r['apk_gp']):.1f}deg)")
    a0.fill_between(np.degrees(al_grid), r["mu_gp"] - 2 * r["sd_gp"], r["mu_gp"] + 2 * r["sd_gp"], color="tab:orange", alpha=0.2)
    a0.axvline(np.degrees(r["apk_true"]), color="k", ls=":", alpha=0.6)
    a0.set_xlabel("slip angle [deg]"); a0.set_ylabel("Fy [N]"); a0.set_title(nm); a0.legend(fontsize=7); a0.grid(alpha=.3)
    a1.plot(np.degrees(r["apk_hist"]), color="tab:blue"); a1.axhline(np.degrees(r["apk_true"]), color="k", ls=":", label="true peak")
    a1.set_xlabel("measurement # (as the car drives)"); a1.set_ylabel("EKF est. peak slip [deg]")
    a1.set_title("EKF peak estimate over time"); a1.legend(fontsize=8); a1.grid(alpha=.3)
fig.suptitle("Online tyre-limit estimation from the car's OWN measurements (EKF + GP) vs the true tyre", fontsize=12)
fig.tight_layout(); [fig.savefig(OUT / f"tyre_estimation.{e}", dpi=140) for e in ("pdf", "png")]

res = dict(rear=dict(true=round(RE["apk_true"], 4), ekf=round(RE["apk_e"], 4), gp=round(RE["apk_gp"], 4),
                     gp_lowerbound=round(RE["apk_gp_lb"], 4), drove_to=round(RE["a_max"], 4)),
           front=dict(true=round(FR["apk_true"], 4), ekf=round(FR["apk_e"], 4), gp=round(FR["apk_gp"], 4),
                      gp_lowerbound=round(FR["apk_gp_lb"], 4), drove_to=round(FR["a_max"], 4)),
           note=("FRONT reaches its limit in cornering -> EKF & GP recover the peak. REAR does not (this "
                 "car is understeer, the front saturates first) -> the rear peak is never sampled, so EKF "
                 "EXTRAPOLATES and the GP stays conservative with a wide band -- exactly the one-pull-test "
                 "limitation: you can only estimate a limit you actually approach. Set alpha_ref from the GP "
                 "lower bound and RELAX it as the driven slip range grows."))
json.dump(res, open(OUT / "tyre_estimation.json", "w"), indent=2)
print("saved results/race/stability/tyre_estimation.{pdf,png,json}")
