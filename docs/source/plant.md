# The plant

Every result in this project is measured on a **dynamic single-track drift
model** with tyres fitted to real RC-car recordings — not on the bicycle the
controller predicts with. The mismatch between the two is deliberate, and it is
the thing the online learning exists to absorb.

## The plant and the controller are different models

`mpcc_tuning/plant_scuderia.py` wraps
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax): ST / STD /
STD4W vehicle models with Pacejka, brush or Dugoff tyres fitted to real data. The
evaluation plant is the STD (single-track drift) model.

The **controller** predicts with its *own* dynamic single-track model
(`mpcc_tuning/model.py::DynamicBicycle`, the `vehicle="dynamic"` acados OCP) —
simpler than the plant, and **not sharing the plant's identified tyre
parameters.** So the mismatch is one of fidelity and calibration, not model
class: slip angles and load transfer the controller represents only
approximately, combined-slip saturation it under-models, and a rate-limited
steering servo with transport delay it does not model at all.

```{admonition} Why not just give the controller the plant's model?
:class: note
Because then there is nothing to study. A controller tuned in simulation meets a
real vehicle, surface and tyres it was not tuned for; closing that gap by
adapting the cost online — rather than re-identifying the model offline — is the
whole claim. A shared model would make the question unaskable.
```

## Three details of the coupling, each a bug first

**The state vectors differ.** `scuderia_gym_jax` uses CommonRoad's ordering
$[X,Y,\delta,v,\psi,\dots]$; this repo's state is $[X,Y,\psi,v_x,v_y,r,s]$. The
mapping lives in `ScuderiaPlant.state5` and nowhere else.

**Acceleration must not go through their PID.** By default the env takes a speed
*setpoint* and closes a PID around it. Stacking that under the MPCC gave two
cascaded loops fighting — the MPCC asks for $+4\,\mathrm{m/s^2}$, gets a
fraction, asks harder, overshoots into a reversal. The plant is built with
`ctrl_mode="accl"` so the acceleration command goes straight to the actuator
constraints.

**The plant runs at its own rate.** Its config is calibrated for 100 Hz
(`steer_delay=2` is two *ticks*). Building it at the controller's 50 ms would
silently stretch a 20 ms servo delay to 100 ms. So the env runs at 100 Hz and is
stepped five times per control tick with the command held — a zero-order hold,
which is what a 20 Hz controller on a 100 Hz vehicle actually is.

## The mismatch is real, and the method survives it

With a *naive* setup — the controller's default weights, or an unconstrained
tuner from a bad initialisation — this plant is unforgiving: the car over-drives
the first tight corner because its model believes any speed is corner-able, the
tyres disagree, and it leaves the track in well under a second. That is not a
failure of the connection; it is an unmodelled tyre seen from inside a controller
that has never heard of one.

The shipped method does not meet the plant from a bad initialisation. It starts
from an **anchored base policy** that is already inside the feasible set, bounds
every move to the measured box, and clamps the grip and acceleration weights with
{doc}`class-conditioned bands <safety>`. Under that construction the drift plant
is exactly where the results are measured:

- the {doc}`main comparison <policies_in_the_table>` — baseline, best constant,
  offline-fitted network, and the online-refined network — all on the STD plant;
- a single online-learned policy racing one to three occluded opponents on it,
  without retraining ({doc}`results`);
- the deployment-map transfer and the software-in-the-loop verification.

```{admonition} The historical version of this page
:class: note
An earlier investigation asked whether the *spike's* envelope-critic tuner, run
from the kinematic bicycle's defaults, could tune its way onto this plant. It
could not — 120 episodes, still off-track — and the honest reading at the time
was that weight tuning is not a substitute for a model and the reward rewarded
standing still. That result is what argued for starting anchored inside the
feasible set and bounding the moves, rather than tuning into it from outside. It
is kept in {doc}`spike_results`.
```

## What is still not connected

**The occupancy maps.** Geometry comes from `mpcc_tuning.track.Track`, which
needs the path as a differentiable function of arc length, so
`scuderia_gym_jax`'s raw occupancy images are not used directly — a corridor is
built raceline-referenced from the optimiser's world CSV instead (the same
representation the net trains on, and what makes the lab-map transfer work). Only
the *dynamics* are borrowed from `scuderia_gym_jax`, which keeps the comparison
against any other plant like-for-like: same track, same controller, same reward,
different physics.
