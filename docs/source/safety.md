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

## Tuning behind the filter

The previous section fixes a *fixed* set of bad weights. The real question is
what happens when the tuner runs behind the filter from the start. Same seed,
same 26 episodes, same everything except whether the filter is in the loop:

| | no filter | **with filter** |
|---|---|---|
| last 8 episodes, mean | 7.6 m | **68.1 m** |
| last 8 episodes, off-track | **8 / 8** | **0 / 8** |
| best episode | 79.1 m (ep 4) | 78.9 m (ep 5) |
| behaviour after ep 5 | collapsed, never recovered | kept running |

Episode by episode, the filter does nothing at all until it is needed:

| episode | no filter | with filter | overridden |
|---|---|---|---|
| 0–3 | 73.1 → 78.1 m | identical | **0%** |
| 4 | 79.1 m | 78.3 m | 6% |
| **5** | **8.5 m, off-track** | **78.9 m** | 2% |
| 10 | 7.9 m, off-track | 76.9 m | 5% |
| 20 | 7.9 m, off-track | 71.9 m | 2% |
| 25 | 7.4 m, off-track | 48.6 m | 0% |

**Nine times the distance and zero crashes.** For the first four episodes the two
runs are bit-identical, because the filter has nothing to do — it only starts
intervening at episode 4, one episode before the unfiltered run drives into a
wall. That is the ordering you want: the filter is silent until the tuner is
about to do something unrecoverable, and then it costs 2–6% of the actions.

### But look at the last row

`q_c` — the weight holding the car near the path — keeps falling anyway: 2.52 at
episode 0, 0.44 at episode 10, **0.136 at episode 20**. And the distance covered
drifts down with it: 78.9 → 76.9 → 71.9 → 48.6 m.

**The filter converted a crash into a slow degradation.** It fixed the symptom
and not the cause. Nothing in the reward now punishes the tuner for pushing
$q_c$ towards zero, because the consequence of doing so — leaving the track — is
being absorbed by the filter and never reaches the return. The weights are free
to drift, and the performance that remains is increasingly being produced by the
*filter*, not by the controller being tuned.

This is the failure mode `rtrrl-playground` measured directly: a policy trained
behind a filter scored 344 against 194 without one, and then under-performed
when the filter was removed. It is not an argument against the filter — 0
crashes against 8 is worth having on its own, and on hardware it is the whole
ballgame — but it means **a filtered run has to be evaluated unfiltered** before
any claim is made about the weights it found, and it means the filter is not a
substitute for the missing trust region on $\theta$.

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
