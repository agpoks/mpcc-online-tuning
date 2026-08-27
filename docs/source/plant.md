# The plant, and swapping it for a real vehicle model

The controller in this repo has always been the interesting half. The *plant*
started as a kinematic bicycle with one hand-written flaw — a yaw-rate cap the
MPCC does not model — which is enough to make the tuning problem non-trivial
and not enough to make it convincing.

`mpcc_tuning/plant_scuderia.py` replaces it with
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax): ST / STD /
STD4W vehicle models with Pacejka, brush or Dugoff tyres fitted to real RC-car
recordings.

```bash
pip install jax chex
PYTHONPATH=/path/to/scuderia_gym_jax \
    python examples/tune_online.py --plant scuderia --model st
```

## What changes, and what deliberately does not

**The controller does not change.** The MPCC still predicts with the kinematic
bicycle in `mpcc_tuning/model.py`. That is the point: the model mismatch stops
being a knob and becomes real — slip angles, load transfer, combined-slip
saturation, a rate-limited steering servo with transport delay, none of which
the MPCC has any representation of.

**The track does not change.** `scuderia_gym_jax`'s maps are occupancy images
with no centreline, and MPCC needs the path as a differentiable function of arc
length, so geometry stays with `mpcc_tuning.track.Track` and only the
*dynamics* are borrowed. That also keeps the comparison against the bicycle
plant like-for-like: same track, same controller, same reward, different
physics.

Three details of the coupling are load-bearing, and each was a bug first.

**The state vectors are different.** `scuderia_gym_jax` uses CommonRoad's
ordering $[X, Y, \delta, v, \psi, \dots]$; this repo's MPCC uses
$[x, y, \psi, v]$ plus the progress variable. The mapping lives in
`ScuderiaPlant.state5` and nowhere else.

**The acceleration must not go through their PID.** By default the env takes a
speed *setpoint* and closes a PID around it. Stacking that under the MPCC gave
two cascaded loops fighting: the MPCC asks for +4 m/s², gets a fraction of it,
asks harder, and overshoots into a −4 m/s² reversal. That is a defect in the
coupling, not a physical mismatch worth studying, so the plant is built with
`ctrl_mode="accl"` and `u[1]` goes straight to `accl_constraints`.

**The plant runs at its own rate.** Their config is calibrated for 100 Hz —
`steer_delay=2` means two *ticks*, and the PID's steering gain of 100 is
documented as being calibrated for 0.01 s. Building the env with
`timestep=0.05` to match the controller silently stretches a 20 ms servo delay
into 100 ms and detunes the steering loop by 5×. So the env is built at 100 Hz
and stepped five times per control tick with the command held — a zero-order
hold, which is what a 20 Hz controller on a 100 Hz vehicle actually is.

## The result: the mismatch is real, and the default weights do not survive it

With the weights that work on the bicycle plant (`q_v=0.05`, `r_d=1.0`), the
car leaves the track in **12 to 22 steps, every time.** It accelerates to
3.5 m/s on the way into a 2.5 m-radius corner because its kinematic model
believes any speed is corner-able, and the real tyres disagree.

That is not a failure of the connection. It is the connection working: this is
what an unmodelled tyre looks like from inside a controller that has never
heard of one.

### Is the task even feasible for this controller?

Before reading anything into a learning curve, the prior question is whether
the six weights can express a policy that survives at all. Swept open-loop, 400
steps, no learning:

| `q_v` | `r_d` | covered | steps | max speed | |
|---|---|---|---|---|---|
| 0.05 | 1.0 | −2.90 m | 28 | 2.94 | off |
| 0.05 | 10.0 | 0.10 m | 75 | 3.28 | off |
| 0.02 | 1.0 | −3.70 m | 21 | 2.42 | off |
| **0.02** | **10.0** | **5.90 m** | **400** | 1.17 | **survives** |
| 0.01 | 1.0 | −3.80 m | 31 | 1.53 | off |
| **0.01** | **10.0** | **5.70 m** | **400** | 1.10 | **survives** |
| 0.005 | 1.0 | −3.10 m | 34 | 2.15 | off |
| **0.005** | **10.0** | **5.90 m** | **400** | 1.05 | **survives** |

*(covered includes the −5 crash penalty, so a negative number is a crash.)*

**Yes, and the lever is `r_d`.** Every run with the steering-rate penalty at 10
completes 400 steps; every run at 1.0 crashes, at every progress weight tried.
That is the lever predicted in the module docstring for the right reason: the
plant has a rate-limited servo behind a transport delay, and a cost that
penalises steering rate produces a command that servo can actually follow. The
MPCC cannot model the servo, but it can be *taught to stop asking it for things
it cannot do* — which is exactly the claim that "tune the cost, not the model"
is supposed to cash out.

### And the tuner does not find it

120 episodes of online tuning from the bicycle defaults: still 100% off-track,
best episode −2.20 m, and the weights ended at

    q_c=94.9  q_l=99.3  q_v=0.647  r_d=84.7  r_a=30.3  r_dv=152.3

Every input-penalty weight has exploded. The tuner moved `r_d` in the right
direction — 1.0 toward 10 is correct — and then straight past it to 84, and
pulled `r_a` and `r_dv` up with it, arriving at a controller that penalises
control so heavily it barely commands anything. With a reward of
*progress minus a crash penalty*, "do not drive" scores ≈ 0 and "drive and
crash" scores −5, so **not moving is a local optimum**, and from an
initialisation that crashes in under a second it is the nearest one.

Three honest readings, in the order they matter:

**The reward is wrong for this plant.** It was designed for a plant where the
default weights already complete laps and the question is how *fast*. Against a
plant where they crash immediately, it rewards standing still. A per-step
survival bonus, or an initialisation inside the feasible set the sweep above
identifies, is the obvious fix and is not yet done.

**Weight tuning is not a substitute for a model.** The feasible region exists,
and the online tuner did not reach it from a starting point one order of
magnitude away in a single weight. This is the same lesson as
[`rtrrl-playground`'s grip sweep](https://agpoks.github.io/rtrrl-playground/safety.html):
what the controller believes about the car dominates what any outer loop can
recover.

**The negative result is the useful part of the connection.** On the bicycle
plant the tuner improves the controller and then destabilises it, which is a
story about learning rates. On real fitted tyres it does not get that far. Any
claim about online MPC tuning that was validated only against a plant whose
mismatch was one hand-written line should be re-run here before it is believed.

## What is not connected

**The occupancy maps.** Geometry comes from `Track`, so the real recorded
tracks in `scuderia_gym_jax` are not usable until MPCC has a centreline
extracted from them.

**The lidar.** Built with `produce_scans=False`. The MPCC is handed the state
directly, exactly as the safety filters in `rtrrl-playground` are, and with the
same caveat: nothing here is tested against a state *estimate*.

**Collisions and multiple agents.** `collision_on=False`, one agent. Overtaking
against `scuderia_gym_jax` traffic is the obvious next step and would need the
obstacle constraint the MPCC does not currently have.
