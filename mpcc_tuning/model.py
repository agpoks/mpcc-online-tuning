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

    nx, nu = 4, 2

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
