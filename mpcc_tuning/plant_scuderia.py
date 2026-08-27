"""Swap the plant for `scuderia_gym_jax`'s real vehicle models.

The controller does not change. The MPCC keeps predicting with the kinematic
bicycle in :mod:`mpcc_tuning.model`, and the *plant* becomes an ST / STD /
STD4W model with Pacejka, brush or Dugoff tyres fitted to real RC-car
recordings. That is the point of the swap and the reason it is worth doing:

**the model mismatch stops being a knob and becomes real.** In
``examples/tune_online.py`` the mismatch is one hand-written line -- a yaw-rate
cap the controller does not model. Here it is slip angles, load transfer,
combined-slip saturation and a steering-rate limit, none of which the MPCC has
any representation of. If online tuning of six cost weights can absorb *that*,
the claim is worth something; if it cannot, that is worth knowing before anyone
puts it on a car.

The track does not change either. `scuderia_gym_jax`'s maps are occupancy
images with no centreline, and MPCC needs the path as a differentiable function
of arc length, so the geometry stays with :class:`mpcc_tuning.track.Track` and
only the *dynamics* are borrowed. Clean separation, and it keeps the comparison
against the bicycle plant like-for-like: same track, same controller, same
reward, different physics.

    pip install jax chex
    PYTHONPATH=/path/to/scuderia_gym_jax \\
        python examples/tune_online.py --plant scuderia --model std

## Two things that differ from the bicycle plant

**The state vector is theirs.** `scuderia_gym_jax` uses the CommonRoad ordering
``[X, Y, delta, v, psi, ...]``; this repo's MPCC uses ``[x, y, psi, v]`` plus
the progress variable. The mapping is in :meth:`ScuderiaPlant.state5` and is
the only place the two conventions meet.

**The control is routed to bypass their PID.** By default `scuderia_gym_jax`
takes a speed *setpoint* and closes a PID around it. Stacking that under the
MPCC gave two cascaded loops fighting each other -- the MPCC would ask for
+4 m/s2, get a fraction of it, ask harder, and overshoot into a -4 m/s2
reversal. That is a bug in the *coupling*, not an interesting physical
mismatch, so the plant is built with ``ctrl_mode="accl"``, which routes ``u[1]``
straight into ``accl_constraints`` as a desired acceleration -- exactly what
the MPCC produces.

**The plant runs at its own rate, not the controller's.** `scuderia_gym_jax`'s
config is calibrated for a 100 Hz tick -- ``steer_delay=2`` means two *ticks*,
and the PID's steering gain of 100 is documented as being calibrated for
0.01 s. Building the env with ``timestep=0.05`` to match the MPCC would
silently stretch a 20 ms servo delay into 100 ms and detune the steering loop
by 5x, which is an artefact of the coupling rather than a property of the car.
So the env is built at its native ``PLANT_DT`` and stepped ``dt / PLANT_DT``
times per control tick with the command held -- which is what a 20 Hz
controller on a 100 Hz vehicle actually does.

The steering path is otherwise left alone, and it is the mismatch worth
keeping: the command is an *angle* setpoint that goes through the transport
delay and is then rate-limited by ``steering_constraint``. The MPCC models
neither. An RC servo really does behave like that, and the tuner has a
legitimate lever against it -- raising ``r_d``, the steering-rate penalty,
smooths the command the servo has to follow. Whether it finds that lever is
the experiment.
"""

from __future__ import annotations

import numpy as np

SPEED_MAX = 4.0
#: The vehicle config's native tick. Its steer delay and PID gain are
#: calibrated for this, so the plant is integrated here and not at the
#: controller rate.
PLANT_DT = 0.01


class ScuderiaPlant:
    """A `scuderia_gym_jax` vehicle, behind the same interface as ``Plant``."""

    def __init__(self, track, model: str = "st", dt: float = 0.05,
                 config: str | None = None, seed: int = 0, **make_kwargs):
        try:
            import jax
            import jax.numpy as jnp
            import scuderia_gym_jax as sgj
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "the scuderia plant needs scuderia_gym_jax and its dependencies:\n"
                "    pip install jax chex\n"
                "    pip install -e /path/to/scuderia_gym_jax\n"
                f"(the import failed with: {exc})"
            ) from exc
        self._jax, self._jnp = jax, jnp
        self.track, self.dt, self.model_name = track, float(dt), model
        # No scans and no collision checking: the MPCC is given the state, and
        # the track constraint lives in the NLP, not in the simulator.
        # ctrl_mode has to be set in two places: in the config override, which
        # lands in the packed vehicle array, and on the ModelSpec, which the
        # kernels dispatch on. ``spec.validate_against`` checks the two agree.
        self.substeps = max(1, int(round(self.dt / PLANT_DT)))
        self.env = sgj.make(config, overrides={"ctrl_mode": "accl"},
                            model=model, ctrl_mode="accl", num_agents=1,
                            produce_scans=False, collision_on=False,
                            timestep=PLANT_DT, **make_kwargs)
        # jit the bound method once -- see the note in rtrrl-playground's
        # envs/scuderia.py about what stepping this from Python costs otherwise.
        self._step_env = jax.jit(self.env.step_env)
        self._key = jax.random.key(seed)
        self.margin = track.half_width - 0.12
        self.max_steps = 600

    # -- helpers ----------------------------------------------------------
    def _split(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def state5(self) -> np.ndarray:
        """``[x, y, psi, v, s]`` -- this repo's convention, from theirs.

        Theirs is CommonRoad's ``[X, Y, delta, v, psi, ...]``; the reordering
        happens here and nowhere else.
        """
        x = self._x
        return np.array([x[0], x[1], x[4], x[3], self.s])

    # -- interface --------------------------------------------------------
    def reset(self, s0: float = 0.0):
        p = self.track.center[0]
        nxt = self.track.center[1]
        psi = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
        poses = self._jnp.asarray([[p[0], p[1], psi]])
        _obs, self._state = self.env.reset(self._split(), poses)
        # start rolling, so the first solves are not from a standstill
        self._state = self._state.replace(
            x=self._state.x.at[:, 3].set(1.0))
        self._x = np.asarray(self._state.x[0])
        self.s = 0.0
        self.t = 0
        self.trace = [self._x.copy()]
        return self.state5()

    def step(self, u):
        """``u = [steering angle, acceleration, v_s]`` from the MPCC.

        Both go through unchanged: the steering angle into the delay buffer,
        the acceleration into ``accl_constraints``. ``v_s`` is the MPCC's own
        progress bookkeeping and never reaches the car.
        """
        u = np.asarray(u, dtype=float)
        act = self._jnp.asarray([[u[0], u[1]]])
        prev = self.track.project(self._x[0], self._x[1])
        for _ in range(self.substeps):        # zero-order hold, like a real rig
            _o, self._state, _r, _d, _i = self._step_env(
                self._split(), self._state, act)
        self._x = np.asarray(self._state.x[0])
        now = self.track.project(self._x[0], self._x[1])
        d = (now - prev) % self.track.length
        progress = d - self.track.length if d > self.track.length / 2 else d
        self.s += float(u[2]) * self.dt
        self.t += 1
        self.trace.append(self._x.copy())
        lateral = self.track.lateral(self._x[0], self._x[1])
        off = abs(lateral) > self.margin or not np.isfinite(self._x).all()
        reward = progress - (5.0 if off else 0.0)
        return self.state5(), reward, off, self.t >= self.max_steps
