# A safety filter for the tuner

The failure this repo documents is not a bad gradient. The gradient is exact to
[five decimal places](results.md#the-gradient-is-exact). The failure is that the
tuner walks $\theta$ into a region where the controller crashes, and the only
signal saying so is a single terminal $-5$ that arrives after the weights are
already somewhere no small step returns from — [episode 5 leaves the track and
the next twenty never recover](animations.md#the-tuner-tuning).

That is a problem a **predictive safety filter** is shaped for, and
`rtrrl-playground` already has one. `mpcc_tuning/safety.py` is the port.

## What it changes

The car stays on the track whatever $\theta$ does, so:

- the catastrophic, episode-ending, information-poor event becomes a **bounded,
  per-step, information-rich** one — the intervention rate is a dense measure of
  how bad the current weights are, available every tick rather than once per
  crash;
- the tuner keeps collecting transitions instead of resetting;
- and on real hardware the car survives its own tuning, which is the only
  version of this that can run outside a simulator.

The certificate is the same one as in `rtrrl-playground`: apply the candidate
for one step, run a **braking backup** for the horizon, and accept only if the
whole path stays inside the corridor *and* ends stopped. The terminal condition
is what makes it recursively feasible rather than an $N$-step lookahead — see
[that repo's derivation](https://github.com/agpoks/rtrrl-playground/blob/main/docs/source/safety.md)
for the induction.

## Does it work?

Take the weights the tuner collapsed to (episode 25: $q_c = 0.41$,
$q_v = 22$), which crash the car reliably, and run them with and without.

| weights | filter | steps | covered | outcome | overridden |
|---|---|---|---|---|---|
| collapsed (ep 25) | off | 68 | 7.6 m | **off-track** | — |
| collapsed (ep 25) | **on** | 400 | **77.3 m** | **survived** | 8% |
| default | off | 400 | 5.9 m | survived | — |
| default | **on** | 400 | 5.9 m | survived | **0%** |

Two rows matter, and they say opposite-sounding things.

**A controller that crashes in 68 steps completes 400 and covers ten times the
distance**, intervening on 8% of them, with `n_no_safe_action = 0` — recursive
feasibility held the whole way. The filter is not merely preventing the crash;
the weights the tuner found were *good* weights being applied slightly too
aggressively, and 77.3 m is within a few percent of the 79.1 m the tuner
achieved at its peak before collapsing.

**On weights that were never going to crash, the filter is invisible** — 0%
intervention, byte-identical distance. That is the property to check first in
any filter: one that intervenes on a safe controller is not a filter, it is a
controller, and the thing being tuned is then no longer the thing being
measured.

## Why it is not the `rtrrl-playground` filter, exactly

Same idea, different action space. `rtrrl_playground.safety` filters **nine
discrete actions**, so "minimally modify subject to a backup existing" is
enumerate-and-check and the argmin is exact. The MPCC emits a **continuous**
$(\delta, a)$, and the exact continuous form is the QP that repo's docs list as
not implemented — it is not implemented here either.

What this does instead is search a structured candidate set ordered by distance
from the proposal, so the result is the nearest *sampled* safe action. The
ordering is deliberate: **deceleration is tried before steering.** A filter that
swerves can lose the car; one that brakes gives up progress, which is exactly
the currency the tuner is trading in and therefore the intervention it can learn
from.

## Three bugs, because none of them announced themselves

Each of these produced a filter that looked like it was working.

**The model had no lateral-acceleration limit.** The filter was first built on
the MPCC's *internal* bicycle, which has no yaw-rate cap — so a car predicted
with it can turn on a dime and can therefore always save itself. Result: every
action certified from every state, 45 out of 45, at every speed and every point
on the lap. That is not a conservative filter, it is a filter that has been
switched off while still reporting a 0% intervention rate. It now predicts with
the plant's bicycle, including `A_LAT_MAX * assumed_grip / v` — the same
quantity as `ay_max` in an acados MPCC's path constraints.

**The corridor was wider than the plant's.** `margin = 0.10` gives a limit of
0.65, and the plant declares off-track at 0.63. So the filter certified an
action, the action put the car outside, and the filter first refused on the step
the car was *already off*. A filter less conservative than the thing it protects
is not a filter. The margin is now 0.18, and the rest of it absorbs model error.

**The integrator disagreed with the plant.** The filter updated the heading and
*then* integrated position with the new heading; the plant integrates position
with the old heading and then updates. Equivalent-looking, and worth 1.4 cm per
step whenever the steering is non-zero — over a 30-step backup a systematic
error comparable to the whole margin, and it certified braking manoeuvres that
then left the track. `tests/test_safety.py` now asserts the two models agree to
**0.0** over a grid of 45 $(\delta, a)$ pairs, which is the only assertion of
its kind worth making: "close" was exactly the problem.

The general lesson is the one the grip sweep in `rtrrl-playground` makes
quantitatively — **a wrong filter intervenes *less*, not more.** All three bugs
showed up as a reassuringly low intervention rate. Intervention rate is a cost
metric; reading it as a safety metric inverts the sign.

## What it does not fix

**It changes what the tuner learns.** The envelope-theorem gradient is
$\mathrm{d}Q/\mathrm{d}\theta$ for the action the MPCC *proposed*. If the filter
overrode it, that gradient describes an action that did not happen — the same
off-policy problem `rtrrl-playground`'s `credit` flag exists to measure, and
there is no better answer here. Both options are implemented and neither is
correct.

**Controllers learn to lean on it.** Measured in `rtrrl-playground`: a policy
trained behind a filter scored 344 against 194 without, and then
*under-performed* when the filter was removed. A run tuned behind the filter
must be evaluated *without* it before any claim is made about the weights it
found.

**It inherits its guarantee from its model.** It is the same kinematic bicycle
the MPCC predicts with, so on the [`scuderia` plant](plant.md) the filter is
wrong in exactly the way the controller is wrong — slip angles, load transfer
and a rate-limited servo that neither models. Porting it there is not a matter
of passing a flag; it needs a model that can represent what that plant does.
