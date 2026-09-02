"""The vehicle model, in CasADi (for the controller) and NumPy (for the plant).

Two copies on purpose. The controller's copy is what it *believes*; the plant's
copy is what happens. Keeping them separate is the only way to ask the question
this repo exists for -- what does online tuning recover when the controller's
model is wrong? -- and a single shared model quietly makes that question
unaskable.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

WHEELBASE = 0.33
# 0.40. Raising it to 0.50 was tried and REVERTED -- it made things worse.
#
# The geometric argument for raising it is sound as far as it goes: the minimum
# turn radius is wheelbase/tan(delta), so 0.40 gives 0.78 m against an ICRA
# corridor centre that drops to 0.69 m at its hairpins, and 1% of both laps was
# tighter than the car could turn. At 0.50 that figure is 0%.
#
# It is also not what was stopping the car. Measured on both tracks at two
# horizons, same weights, distance covered before leaving the track:
#
#     track   horizon   steer 0.40   steer 0.50
#     T1        12        22.8 m       22.9 m
#     T1        40       125.3 m       31.8 m     <-- 4x worse
#     T2        12        26.6 m       27.1 m
#     T2        40        17.4 m       11.8 m
#
# At the short horizon the two are indistinguishable, which is the tell: extra
# steering authority is worth nothing when the plan is wrong. At the long
# horizon it is actively harmful, because it lets the controller commit harder
# to a line it will not be able to hold. What actually fixed T1 was the horizon
# -- 0.6 s of lookahead cannot see through a 0.7 m-radius hairpin, which is
# 2.2 m of arc against 1.8 m of plan.
STEER_MAX = 0.40
ACCEL_MAX = 4.0
# 8.0, not 4.0. A flat cap does not describe a vehicle, it describes an
# assumption -- and at 4 m/s the cap binds on 45-66% of every track here, so the
# car is speed-limited rather than grip-limited and no weight setting is ever
# punished. That is not a property of the controller, it is the cap
# manufacturing an easy problem. The friction-ellipse profile in
# mpcc_tuning/speed.py allows 7.34 m/s on the ICRA raceline and the team's own
# optimiser reaches 6.09; at 8.0 the tyres set the speed everywhere.
SPEED_MAX = 8.0
DRAG = 0.15
# Measured against the ICRA team's optimised raceline, their peak lateral
# acceleration over a lap is 6.8 m/s^2, so this is about 12% conservative. Kept
# conservative deliberately, and recorded rather than silently raised.
A_LAT_MAX = 6.0


class KinematicBicycle:
    """``x = [X, Y, psi, v]``, ``u = [delta, a]``, plus MPCC's progress state.

    The grip limit (a cap on yaw rate at ``A_LAT_MAX * grip / v``) is in the
    *plant* only. The controller does not model it, which is deliberate: a
    tyre limit the controller does not know about is exactly the kind of
    mismatch that shows up as a cost weight being wrong, and therefore exactly
    what an online tuner should be able to compensate for.
    """

    nx, nu, n_dyn = 4, 2, 0

    def __init__(self, dt: float = 0.05, grip: float = 1.0):
        self.dt, self.grip = float(dt), float(grip)

    # -- CasADi (controller's model) --------------------------------------
    def f_sym(self, x, u):
        """Continuous-time dynamics as CasADi expressions."""
        psi, v = x[2], x[3]
        delta, a = u[0], u[1]
        return ca.vertcat(v * ca.cos(psi), v * ca.sin(psi),
                          v / WHEELBASE * ca.tan(delta), a - DRAG * v)

    def step_sym(self, x, u, dt):
        """One RK4 step, symbolically."""
        k1 = self.f_sym(x, u)
        k2 = self.f_sym(x + dt / 2 * k1, u)
        k3 = self.f_sym(x + dt / 2 * k2, u)
        k4 = self.f_sym(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- NumPy (plant) -----------------------------------------------------
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One plant step, with the grip limit the controller does not know about."""
        X, Y, psi, v = x
        delta = float(np.clip(u[0], -STEER_MAX, STEER_MAX))
        a = float(np.clip(u[1], -ACCEL_MAX, ACCEL_MAX))
        dt = self.dt
        v = float(np.clip(v + (a - DRAG * v) * dt, 0.0, SPEED_MAX))
        psi_dot = v / WHEELBASE * np.tan(delta)
        if v > 1e-3:
            lim = A_LAT_MAX * self.grip / v
            psi_dot = float(np.clip(psi_dot, -lim, lim))
        return np.array([X + v * np.cos(psi) * dt, Y + v * np.sin(psi) * dt,
                         psi + psi_dot * dt, v])


# -- Tyres, in the CONTROLLER ------------------------------------------------
#
# Adapted from MPCC_controller_ipopt/MPCC_controller.cpp::build_dynamics() --
# the Liniger single-track planar model with simplified Pacejka tyres. Two
# things are changed from that source and both matter:
#
#   * **the input.** That controller integrates duty and steering *rates*
#     (u = [dD, dDelta, dVs]) and gets longitudinal force from a motor curve
#     Frx = Cm1*D - Cm2*D*vx. This repo commands a steering *angle* and an
#     *acceleration*, and plant_scuderia routes u[1] straight into
#     accl_constraints, so Frx = m*a is the matching force.
#   * **the parameters.** Those are Liniger's 1:43 car (m = 0.041 kg). The
#     plant here is scuderia_gym_jax's RC10, so the numbers below come from
#     its config/rc10_default.yaml instead -- mass, axle distances and yaw
#     inertia read straight off, tyres converted from its Althoff normalised
#     stiffnesses to Pacejka B/C/D as derived in TYRE below.
#
# Why this class exists at all: every MPCC in this repo predicted with
# KinematicBicycle, including on the STD plant. A kinematic model has no
# sideslip, so it cannot represent the one thing the drift plant does. The
# baseline that "passed" on tyres needed r_delta = 5.0 -- fifty times the
# bicycle's 0.1 -- which is not a tuning result, it is the damping needed to
# stop a blind controller exciting dynamics it has no state for.

# Cross-checked, in order of how much each reference is worth here:
#
#   1. scuderia_gym_jax/envs/dynamic_models.py::vehicle_dynamics_std -- this
#      IS the plant, so it is the only one that can be authoritative.
#   2. MPCC_planner_py/vehiclemodels/vehicle_dynamics_std.py (line 119) --
#      CommonRoad's STD, the same model family the plant implements.
#   3. On-Track-SysID/src/helpers/generate_predictions.py (lines 49-58) --
#      ForzaETH's race stack, a sysid tool for a 1:10 car. Same vehicle class
#      and the same static axle-load split, so it is a fair check on the SHAPE
#      of the equations. Its Pacejka coefficients are fitted to their car and
#      say nothing about the parameters used here.
#
# All three agree on the slip angles and the yaw equation. None of them found
# any of the five faults that actually mattered -- those came out of rolling
# the plant and this model forward from the same state under the same inputs
# and diffing the trajectories. The equations were never what was wrong. The slip angles and the
# yaw equation below agree with both. Two terms they carry and this does not:
#
#   * **the front longitudinal yaw moment**, ``+ F_xf sin(delta) lf / I_z``.
#     It needs Frx split front/rear, which needs wheel states; the plant has
#     them (omega_f, omega_r) and this does not. Zero for a rear-driven car on
#     power, non-zero under brakes, so the gap opens on corner entry.
#     On-Track-SysID omits it too.
#   * **combined slip.** The plant's tyre is ``full_pacejka_combined``: lateral
#     force is scaled by a G_y factor that falls as longitudinal slip rises.
#     There is no G_y here, so this model believes it can have peak Fx and peak
#     Fy at once -- the failure MPCC_controller_ipopt warns about in its own
#     words, "allows simultaneous peak forces in both directions, causing the
#     car to spin out". That is why the friction-ellipse penalty in the OCP
#     cost is not optional.
#
# Load transfer is also absent: Fz_f and Fz_r are static. Left in deliberately
# -- it is the kind of mismatch online tuning is supposed to absorb.

# CoG-to-axle, mass and yaw inertia: scuderia_gym_jax rc10_default.yaml.
LF, LR = 0.1705, 0.1515
MASS, I_Z = 4.251, 0.04696

#: Pacejka Fy = D sin(C atan(B a - E (B a - atan(B a)))), matched to the tyre
#: the PLANT actually runs. rc10_default.yaml sets
#: ``model.std_tire_model: full_pacejka_combined``, so the numbers come from
#: its ``tire_full_pacejka`` block and NOT from ``tire_linear``:
#:
#:     p_cy1 = 1.30   shape      C
#:     p_dy1 = 1.10   peak/Fz    D = 1.10 Fz
#:     p_ky1 = 18.0   K_y/Fz  -> B = p_ky1/(p_cy1 p_dy1) = 12.59
#:     p_ey1 = -1.00  curvature  E  ("digressive lateral")
#:
#: with static axle loads Fz_f = m g lr/L = 19.62 N, Fz_r = m g lf/L = 22.08 N.
#:
#: An earlier version of this file took B from the ``tire_linear`` block
#: instead (C_Sf = 14.0, C_Sr = 15.5, giving B_f = 9.79, B_r = 10.84) and set
#: E = 0. Both were wrong against this plant. The stiffness error made the
#: controller believe the tyre built cornering force ~25% more slowly than it
#: does, and the separate front/rear stiffnesses invented an understeer bias
#: the plant does not have -- it runs the SAME tyre at both ends, and the only
#: asymmetry is axle load. The config's own comment confirms the corrected
#: value: "-> B_y=12.6".
B_F = B_R = 12.5874
C_F = C_R = 1.30
E_F = E_R = -1.00
D_F, D_R = 21.5829, 24.2897

#: The commanded acceleration is not the achieved one. The plant's wheels have
#: rotational inertia, and spinning them up absorbs part of the drive: with
#: I_y_w = 0.001 kg m^2 and R_w = 0.031 m, each wheel adds I/R^2 = 1.04 kg of
#: equivalent mass, so the car accelerates as though it weighed
#: 4.251 + 2 x 1.04 = 6.33 kg. Measured: commanding 2.0 m/s^2 moved the plant
#: at 1.328 m/s^2, against 1.343 predicted here.
#:
#: This is NOT drag. The plant coasts 3.000 -> 3.000 m/s at zero throttle, so
#: there is no speed-proportional loss to model -- there is a gain on the
#: input. Treating it as drag (the module's DRAG = 0.15) invented 0.45 m/s^2 of
#: deceleration; ignoring it entirely over-predicted speed by 14% per horizon.
ACCEL_GAIN = 4.251 / (4.251 + 2 * 0.001 / 0.031 ** 2)

#: The steering servo. The plant carries delta as a STATE: the command goes
#: through a transport delay (steer_delay = 2 ticks at 0.01 s) and is then rate
#: limited by steering_constraint at sv_max = 4 rad/s. The controller applying
#: delta instantly is the largest remaining prediction error -- measured on the
#: first tick after a step to delta = 0.2, the plant reaches r = 0.938 where an
#: instant-steer model predicts 1.945. They agree by the third tick, but the
#: MPCC acts on the first.
#:
#: T_SERVO = 0.030, fitted against the plant over a full horizon at two
#: operating points rather than guessed from the first tick. Mean |error| in
#: yaw rate over k = 1..12:
#:
#:     T        0.015    0.020    0.030    0.060    instant
#:     delta=0.15  .038     .034     .028     .064     ~.05
#:     delta=0.20  .081     .075     .074     .099     ~.12
#:
#: 0.060 was tried first, from a first-order fit to the k=1 point alone, and it
#: is the worst of the four: it fixes tick 1 and then lags for five more. The
#: plant is RATE-LIMIT dominated -- a 0.15 rad step clears sv_max = 4 rad/s in
#: 37 ms -- so T wants to be small enough that the tanh saturates.
#:
#: What is still not modelled is the 20 ms transport delay, which is why tick 1
#: is still over-predicted (1.29 against the plant's 0.88 at delta = 0.2). That
#: needs a delay state; the ipopt controller does not model it either.
SV_MAX = 4.0
T_SERVO = 0.030

#: Longitudinal peak on the driven axle, p_dx1 Fz_r with p_dx1 = 1.05. Used to
#: normalise the friction ellipse.
F_LONG_PEAK = 23.186

#: Below the blend the slip angles are atan(vy/vx) with vx -> 0 and the model is
#: meaningless; above it the tyre model is the honest one. Blending between the
#: two is standard for exactly this reason (Liniger 2015) and is needed here
#: because the plant starts the car rolling at 1.0 m/s, right in the band where
#: an unblended dynamic model is worst. Centre and width of the tanh, replacing
#: an earlier clamped linear ramp -- see the note in ``f_sym``.
#:
#: 1.40, NOT the plant's 0.70. These were set to 0.70/0.15 to match
#: scuderia's own blend constants, on the reasoning that matching the plant is
#: always more accurate. That is wrong here, and measurably so: the acados QP
#: failed on tick 2 at 0.70 against tick 36 at 1.40, an eighteenfold
#: difference.
#:
#: The two blends do different jobs. The plant's exists because the dynamic
#: limb is meaningless as vx -> 0, and it integrates at 2 ms so it can afford a
#: sharp transition. The controller has that reason AND another: it must
#: LINEARISE this model for a QP at 12.5 ms steps, and the slip angles
#: atan((vy +- lr r)/vx) have Jacobian entries of order 1/vx -- 1.27 at
#: vx = 0.79 against 0.33 at vx = 3. Blending out earlier keeps the controller
#: clear of the band where its own linearisation is worst.
#:
#: The cost is real: below ~1.4 m/s the controller predicts with the kinematic
#: limb while the plant is already dynamic. That is a deliberate trade of
#: fidelity for solvability at a speed the car is rarely at.
V_BLEND_MID, V_BLEND_W = 1.40, 0.30

#: Speed bias in the slip-angle denominators, NOT a numerical epsilon.
#:
#: The slip angles are atan((vy +- lr r)/vx), so their Jacobian entries go like
#: 1/vx: 1.27 at vx = 0.79 against 0.33 at vx = 3. Below roughly 1.5 m/s that
#: division is what makes the linearisation ill-conditioned, and it is what the
#: acados QP was reporting as NaN. A guard of 1e-3 is no guard at all at
#: 0.79 m/s -- it only prevents a divide by exactly zero.
#:
#: Biasing the denominator, vxs = sqrt(vx^2 + V_BIAS^2), caps the Jacobian at
#: 1/V_BIAS everywhere. It makes the model deliberately wrong -- at vx = 3 with
#: V_BIAS = 1.0 the denominator reads 3.16 instead of 3.00, a 5% understatement
#: of slip -- and that is the trade: a slightly wrong model that solves beats
#: an exact one that returns NaN.
V_BIAS = 1.0

#: Smoothing width for the soft abs/max below. Small enough that the penalties
#: still bite where they should, large enough that a second derivative exists.
SMOOTH_EPS = 1e-3


def _sabs(z, eps: float = SMOOTH_EPS):
    """``|z|``, differentiable twice at zero."""
    return ca.sqrt(z * z + eps * eps)


def _smax(z, eps: float = SMOOTH_EPS):
    """``max(z, 0)``, differentiable twice at zero.

    The exact form ``(z + |z|)/2`` has no second derivative at the kink, and
    that is where acados' EXACT Hessian produced NaN -- HPIPM returns QP status
    3 on every solve. This is the same function with the corner rounded over a
    width of ``eps``.
    """
    return 0.5 * (z + _sabs(z, eps))


def _pacejka(a, B, C, D, E):
    """Magic formula with the curvature term kept.

    ``E`` is not a refinement here: at ``p_ey1 = -1.0`` the plant's tyre is
    digressive, so lateral force *falls* after its peak. Dropping it (E = 0)
    gives a tyre that merely saturates, which is optimistic in exactly the
    regime where the car is already sliding and the controller most needs to
    be told that pushing harder makes things worse.
    """
    Ba = B * a
    return D * ca.sin(C * ca.atan(Ba - E * (Ba - ca.atan(Ba))))


class DynamicBicycle:
    """``x = [X, Y, psi, vx, vy, r]``, ``u = [delta, a]`` -- a car with tyres.

    Same interface as :class:`KinematicBicycle`, plus ``n_dyn = 2`` extra
    states. The MPCC lays its state out as ``[X, Y, psi, vx, s, vy, r]`` so
    that every index the kinematic model established keeps its meaning and the
    progress variable stays at 4.
    """

    nx, nu = 6, 2

    def __init__(self, dt: float = 0.05, grip: float = 1.0,
                 integrator: str = "rk4", drag: float = 0.0,
                 steer_lag: bool = True):
        self.dt, self.grip = float(dt), float(grip)
        self.integrator = str(integrator)
        # With steer_lag the commanded angle is a SETPOINT and the actual angle
        # is a state, which is how MPCC_controller_ipopt formulates it too
        # (delta is its state 8, and the input is its rate). Off, the model
        # steers instantly and over-predicts first-tick yaw by ~2x.
        self.steer_lag = bool(steer_lag)
        self.n_dyn = 3 if steer_lag else 2
        # DRAG = 0, not the module's 0.15. plant_scuderia runs the car in
        # ctrl_mode="accl", where u[1] goes straight into accl_constraints and
        # the command IS the achieved acceleration -- the plant coasts 3.000 ->
        # 3.000 m/s over a second at zero throttle, measured. Subtracting
        # 0.15*v invented 0.45 m/s^2 of deceleration that does not exist, and
        # it showed up as the model under-predicting speed by 9% over one
        # horizon and, through v tan(delta)/L, under-predicting yaw with it.
        self.drag = float(drag)

    def f_sym(self, x, u):
        psi, vx, vy, r = x[2], x[3], x[4], x[5]
        a = u[1]
        if self.steer_lag:
            # delta is a STATE; u[0] is the setpoint the servo chases.
            delta = x[6]
            # Smooth saturation, not clip: it is a first-order lag near zero
            # error and a rate limit at large error, and it has to be twice
            # differentiable for acados' EXACT Hessian.
            delta_dot = SV_MAX * ca.tanh((u[0] - delta) / (T_SERVO * SV_MAX))
        else:
            delta = u[0]
        vxs = ca.sqrt(vx ** 2 + V_BIAS ** 2)       # biased, see V_BIAS

        # Slip angles, then simplified Pacejka. mu scales the peak only: the
        # cornering stiffness at alpha = 0 is B*C*D, so holding B fixed while
        # scaling D is exactly "same tyre, less grip".
        alpha_f = delta - ca.atan((LF * r + vy) / vxs)
        alpha_r = -ca.atan((vy - LR * r) / vxs)
        a_eff = ACCEL_GAIN * a
        Frx = MASS * (a_eff - self.drag * vx)

        # Combined slip, as a friction ellipse on the DRIVEN axle.
        #
        # The plant's tyre is full_pacejka_combined: its lateral force carries a
        # G_y factor that falls as longitudinal slip rises, so a tyre being
        # asked for drive has less grip left to corner with. Without it this
        # model believes it can have peak Fx and peak Fy at once -- the failure
        # MPCC_controller_ipopt names in its own words, "allows simultaneous
        # peak forces in both directions, causing the car to spin out".
        #
        # The plant's G_y needs longitudinal slip, which needs wheel speeds this
        # model does not carry. The ellipse is the standard stand-in and needs
        # only Frx: capacity left for lateral use is sqrt(1 - (Fx/Fx_peak)^2).
        # Floored at 0.2 so the rear never loses all grip in the prediction,
        # which would make the OCP wildly non-convex for no physical gain.
        #
        # Drive is at the rear, so only the rear axle pays on power. Under
        # brakes the front pays too and this does not model that -- see the
        # missing F_xf term noted above.
        fx_use = Frx / F_LONG_PEAK
        g_y = ca.sqrt(_smax(1.0 - fx_use * fx_use) + 0.04)

        Ffy = self.grip * _pacejka(alpha_f, B_F, C_F, D_F, E_F)
        Fry = self.grip * g_y * _pacejka(alpha_r, B_R, C_R, D_R, E_R)
        vx_dot_d = (Frx - Ffy * ca.sin(delta)) / MASS + vy * r
        vy_dot_d = (Fry + Ffy * ca.cos(delta)) / MASS - vx * r
        r_dot_d = (Ffy * LF * ca.cos(delta) - Fry * LR) / I_Z

        # Kinematic limit, for the blend: no sideslip state, yaw from geometry.
        #
        #     r  = vx tan(delta) / L        vy = r lr
        #
        # These are ALGEBRAIC. An earlier version turned them into derivatives
        # by relaxing onto them, ``(r_kin - r) / self.dt``, which is wrong in a
        # way worth spelling out: it invents a stiff term whose rate is the
        # controller timestep (1/0.05 = 20 per second), so a 0.1 rad/s
        # discrepancy manufactures 2 rad/s^2 of yaw acceleration out of
        # nothing. A stiff term inside an explicit integrator is precisely what
        # destabilises it, and it made the model's behaviour depend on dt.
        #
        # Differentiate the algebraic relations instead. delta is an input held
        # over the step, so its derivative is zero and only vx_dot survives.
        # Non-stiff, dt-free, and it is what the plant's own kinematic limb
        # does analytically (see _ks_cog_std in scuderia's dynamic_models.py).
        tan_d = ca.tan(delta)
        vx_dot_k = a_eff - self.drag * vx
        r_dot_k = vx_dot_k * tan_d / (LF + LR)
        vy_dot_k = r_dot_k * LR

        # tanh, not fmin(fmax(...)). A clamped linear ramp has an undefined
        # second derivative at both kinks. IPOPT tolerates that; acados builds
        # an EXACT Hessian straight through the dynamics, and HPIPM returned
        # NaN (QP status 3) on every solve until this was smooth. Same shape,
        # C-infinity everywhere, and it costs one tanh.
        lam = 0.5 * (1.0 + ca.tanh((vxs - V_BLEND_MID) / V_BLEND_W))
        vx_dot = lam * vx_dot_d + (1 - lam) * vx_dot_k
        vy_dot = lam * vy_dot_d + (1 - lam) * vy_dot_k
        r_dot = lam * r_dot_d + (1 - lam) * r_dot_k

        out = ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi),
                         vx * ca.sin(psi) + vy * ca.cos(psi),
                         r, vx_dot, vy_dot, r_dot)
        return ca.vertcat(out, delta_dot) if self.steer_lag else out

    def tyre_sym(self, x, u):
        """``(Frx, Ffy, Fry)`` at one stage, for the friction-ellipse penalty.

        Same expressions as :meth:`f_sym`; exposed separately so the OCP can
        price combined tyre loading without rebuilding the dynamics.
        """
        vx, vy, r = x[3], x[4], x[5]
        delta = x[6] if self.steer_lag else u[0]
        a = u[1]
        vxs = ca.sqrt(vx ** 2 + V_BIAS ** 2)       # must match f_sym
        alpha_f = delta - ca.atan((LF * r + vy) / vxs)
        alpha_r = -ca.atan((vy - LR * r) / vxs)
        # Same combined-slip ellipse as f_sym -- the two must not drift apart,
        # or the OCP prices a friction limit the dynamics do not obey.
        Frx = MASS * (ACCEL_GAIN * a - self.drag * vx)
        g_y = ca.sqrt(_smax(1.0 - (Frx / F_LONG_PEAK) ** 2) + 0.04)
        return (Frx,
                self.grip * _pacejka(alpha_f, B_F, C_F, D_F, E_F),
                self.grip * g_y * _pacejka(alpha_r, B_R, C_R, D_R, E_R))

    def step_sym(self, x, u, dt, sub: int = 4, method: str | None = None):
        """One step, by ``method`` in ``{"euler", "rk3", "rk4"}``.

        Euler was chosen here when RK4 made the cold solves return
        Maximum_Iterations_Exceeded, on the grounds that four Pacejka
        evaluations per shooting node was a Hessian IPOPT could not get
        through. That measurement was taken BEFORE the hard vx bounds came off,
        the double-counted grip row came out and the blend was smoothed, so it
        no longer stands on its own -- and Euler is a real accuracy cost: at
        sub=1 this model predicts a yaw rate of 3.95 rad/s where sub>=2 gives
        0.48, an eightfold error.

        It also has to match the other backend. acados integrates the same
        model with ERK at ``sim_method_num_stages = 4``; a CasADi problem
        stepping Euler is not the same controller, whatever the comment says.
        """
        m = (method or self.integrator).lower()
        # sub = 4, not 2. The yaw mode's time constant here is 17.8 ms
        # (r_dot = 77.5 rad/s^2 from rest toward an equilibrium of 1.378), so
        # sub = 2 integrates a 17.8 ms mode with 25 ms steps and cannot track
        # it: measured against the plant at delta = 0.15, sub = 2 reaches
        # r = 1.007 where the plant reaches 1.390, a 28% error, while sub = 4
        # gives 1.366 and sub = 8 gives the same to four decimals. The plant
        # itself resolves this at 2 ms.
        h = dt / sub
        if m == "rk4":
            for _ in range(sub):
                k1 = self.f_sym(x, u)
                k2 = self.f_sym(x + h / 2 * k1, u)
                k3 = self.f_sym(x + h / 2 * k2, u)
                k4 = self.f_sym(x + h * k3, u)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return x
        if m == "rk3":
            for _ in range(sub):
                k1 = self.f_sym(x, u)
                k2 = self.f_sym(x + h / 2 * k1, u)
                k3 = self.f_sym(x - h * k1 + 2 * h * k2, u)
                x = x + h / 6 * (k1 + 4 * k2 + k3)
            return x
        h = dt / sub
        for _ in range(sub):
            x = x + h * self.f_sym(x, u)
        return x
