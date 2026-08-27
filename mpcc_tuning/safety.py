"""A predictive safety filter for the tuner, and the reason it belongs here.

The failure this repo documents is not a bad gradient -- the gradient is exact
to five decimal places. It is that the tuner walks ``theta`` into a region
where the controller crashes, and the only signal saying so is a single
terminal ``-5`` that arrives after the weights are already somewhere no small
step returns from. Episode 5 leaves the track and the next twenty never
recover.

A safety filter changes the shape of that signal. The car stays on the track
whatever ``theta`` does, so:

* the catastrophic, episode-ending, information-poor event becomes a bounded,
  per-step, information-rich one -- the **intervention rate** is a dense
  measure of how bad the current weights are, available every tick rather than
  once per crash;
* the tuner keeps collecting transitions instead of resetting;
* and on real hardware the car survives its own tuning, which is the only
  version of this that can ever run outside a simulator.

## Why this is not the filter in ``rtrrl-playground``

Same idea, different action space, and that is the whole implementation
difference. ``rtrrl_playground.safety`` filters **nine discrete actions**, so
"minimally modify the proposed action subject to a backup existing" is
enumerate-and-check and the argmin is exact.

The MPCC emits a **continuous** ``(delta, a)``. The exact continuous form is
the QP that ``rtrrl-playground``'s docs list as not implemented, and it is not
implemented here either. What this does instead is search a structured
candidate set around the proposal -- ordered by distance from it -- so the
result is the nearest *sampled* safe action rather than the nearest safe
action. With the default grid that is within a few percent of the input range,
and :meth:`PredictiveSafetyFilter.certify` is exposed so the approximation can
be checked rather than trusted.

The candidate ordering is deliberate: deceleration is tried before steering.
A safety filter that swerves is a safety filter that can lose the car; one that
brakes gives up progress, which is exactly the currency the tuner is trading in
and is therefore the intervention it should be able to learn from.

## What it does not fix

**It changes what the tuner learns.** The envelope-theorem gradient is
``dQ/dtheta`` for the action the MPCC *proposed*. If the filter overrode it,
that gradient describes an action that did not happen -- the same off-policy
problem ``rtrrl-playground``'s ``credit`` flag exists to measure, and it has no
better answer here. ``credit`` is implemented with the same two options and the
same admission that neither is correct.

**Agents learn to lean on it.** Measured in ``rtrrl-playground``: a policy
trained behind a filter scored 344 against 194 without, and then
*under-performed* when the filter was removed. There is no reason the tuner
would be different, so a run with the filter should be evaluated without it
before any claim is made about the weights it found.

**It inherits its guarantee from its model.** Same kinematic bicycle the MPCC
predicts with, so on the ``scuderia`` plant the filter is wrong in exactly the
way the controller is wrong. That is not a detail: see the grip sweep in
``rtrrl-playground``, where an optimistic filter crashed 71% of episodes while
intervening *less* than a correct one.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.model import (
    A_LAT_MAX, ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, WHEELBASE,
)


class PredictiveSafetyFilter:
    """Minimal modification of ``(delta, a)`` subject to a stop remaining possible.

    The certificate is the same one as in ``rtrrl-playground``: apply the
    candidate for one step, then run a braking backup for ``horizon`` steps, and
    accept only if the whole path stays inside the corridor **and** ends
    stopped. The terminal condition is what makes this recursively feasible
    rather than an N-step lookahead -- a stopped car inside the corridor can
    stay there forever, so reaching that set certifies that the episode need not
    end badly.
    """

    def __init__(self, track, dt: float = 0.05, horizon: int = 30,
                 wheelbase: float = WHEELBASE, margin: float = 0.18,
                 stop_speed: float = 0.25, credit: str = "executed",
                 assumed_grip: float = 1.0, n_decel: int = 7, n_steer: int = 9):
        if credit not in ("executed", "proposed"):
            raise ValueError("credit must be 'executed' or 'proposed'")
        self.track, self.dt = track, float(dt)
        self.horizon, self.wheelbase = int(horizon), float(wheelbase)
        self.assumed_grip = float(assumed_grip)
        self.margin = float(margin)
        self.stop_speed = float(stop_speed)
        self.credit = credit
        # Candidates: scale the proposed acceleration down towards full braking,
        # and only then start moving the steering. Braking costs progress, which
        # is the quantity being tuned; swerving costs control of the car.
        self._decel = np.linspace(0.0, 1.0, int(n_decel))
        self._steer = np.linspace(-1.0, 1.0, int(n_steer))
        self.reset_stats()

    def reset_stats(self) -> None:
        self.n_steps = 0
        self.n_interventions = 0
        self.n_no_safe_action = 0

    @property
    def intervention_rate(self) -> float:
        return self.n_interventions / max(self.n_steps, 1)

    # -- the model, as a plain recursion ----------------------------------
    def _step(self, x, y, psi, v, delta, a, h):
        """One step of the plant's bicycle -- **including the yaw-rate cap**.

        This must match :meth:`mpcc_tuning.model.KinematicBicycle.step`, not the
        MPCC's internal prediction. The MPCC's model has no lateral-acceleration
        limit at all, and a filter built on it certifies literally every action
        from every state, because a car that can turn on a dime can always save
        itself. That is not a conservative filter, it is a filter that has been
        switched off while still reporting a 0% intervention rate.

        ``A_LAT_MAX * assumed_grip / v`` is the same quantity as ``ay_max`` in
        an acados MPCC's path constraints, and ``assumed_grip`` carries the same
        risk as everywhere else: set it above the truth and the certificate is
        void.
        """
        delta = min(max(delta, -STEER_MAX), STEER_MAX)
        a = min(max(a, -ACCEL_MAX), ACCEL_MAX)
        v = min(max(v + (a - DRAG * v) * h, 0.0), SPEED_MAX)
        psi_dot = v / self.wheelbase * np.tan(delta)
        if v > 1e-3:
            lim = A_LAT_MAX * self.assumed_grip / v
            psi_dot = min(max(psi_dot, -lim), lim)
        # Position uses the heading *before* the update, then the heading
        # advances -- which is the order KinematicBicycle.step uses. Updating
        # psi first and integrating with the new heading looks equivalent and
        # costs 1.4 cm per step whenever the steering is non-zero; over a
        # 30-step backup that is a systematic error comparable to the whole
        # margin, and it certified braking manoeuvres that then left the track.
        return x + v * np.cos(psi) * h, y + v * np.sin(psi) * h, psi + psi_dot * h, v

    def _lateral(self, x, y) -> float:
        return float(self.track.lateral(x, y))

    def _inside(self, x, y) -> bool:
        """Inside the corridor the filter is willing to certify.

        ``margin`` must be **strictly more conservative than the plant's own
        off-track threshold**, which is ``half_width - 0.12``. At margin 0.10
        the filter's corridor was *wider* than the plant's, so it certified an
        action, the action put the car at 0.651 against the plant's 0.63 limit,
        and the filter first refused on the step the car was already off. A
        filter that is less conservative than the thing it is protecting is not
        a filter. The rest of the margin absorbs the difference between this
        model and the plant, which on the ``scuderia`` plant is large.
        """
        return abs(self._lateral(x, y)) <= self.track.half_width - self.margin

    def certify(self, state5, delta: float, a: float) -> bool:
        """Does a full-braking backup from ``(delta, a)`` stay legal and stop?

        The backup steers towards the centreline while braking. It does not have
        to be good, only safe: it has to reach the terminal set from anywhere
        the filter is willing to certify.
        """
        x, y, psi, v = (float(q) for q in state5[:4])
        x, y, psi, v = self._step(x, y, psi, v, delta, a, self.dt)
        if not self._inside(x, y):
            return False
        for _ in range(self.horizon):
            if v <= self.stop_speed:
                return True                     # in the terminal set, and legal
            k = self.track.project(x, y)
            # Steer back towards the path: the backup must not brake in a
            # straight line out of a corner.
            tgt = float(self.track.tangent_angle(k))
            d = float(self.track.lateral(x, y))
            err = np.arctan2(np.sin(tgt - psi), np.cos(tgt - psi))
            steer = float(np.clip(err - 1.2 * d, -STEER_MAX, STEER_MAX))
            x, y, psi, v = self._step(x, y, psi, v, steer, -ACCEL_MAX, self.dt)
            if not self._inside(x, y):
                return False
        return v <= self.stop_speed

    # -- the filter --------------------------------------------------------
    def __call__(self, state5, u):
        """Return ``(u_to_apply, intervened)``. ``u`` is the MPCC's ``[delta, a, v_s]``.

        ``v_s`` is the MPCC's own progress bookkeeping and is passed through
        untouched -- it never reaches the car, so filtering it would only
        desynchronise the controller from itself.
        """
        self.n_steps += 1
        u = np.asarray(u, dtype=float)
        delta, a = float(u[0]), float(u[1])
        if self.certify(state5, delta, a):
            return u, False

        # Nearest safe candidate, deceleration first. `f` scales the proposal
        # towards -ACCEL_MAX; the steering sweep only opens up once braking
        # alone has failed, so the common intervention is "same line, slower".
        for f in self._decel[1:]:
            a_try = (1.0 - f) * a + f * (-ACCEL_MAX)
            if self.certify(state5, delta, a_try):
                self.n_interventions += 1
                return np.array([delta, a_try, u[2]]), True
        for s in sorted(self._steer, key=lambda t: abs(t * STEER_MAX - delta)):
            d_try = float(s * STEER_MAX)
            if self.certify(state5, d_try, -ACCEL_MAX):
                self.n_interventions += 1
                return np.array([d_try, -ACCEL_MAX, u[2]]), True

        # Nothing in the candidate set is certifiable. With a correct model this
        # is unreachable; it means the state should never have been entered.
        # Brake straight and count it, rather than pretend.
        self.n_no_safe_action += 1
        self.n_interventions += 1
        return np.array([delta, -ACCEL_MAX, u[2]]), True
