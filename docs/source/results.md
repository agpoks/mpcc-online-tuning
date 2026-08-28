# Results, including the failure

Everything below is from `examples/tune_online.py` on a 26.7 m oval with 2.5 m
corners, 200 control ticks per episode at 20 Hz, plant grip 1.0, `alpha = 2e-4`.

## The gradient is exact

The claim the whole approach rests on: at the solution of the MPCC's NLP,

$$\frac{\mathrm{d}J^*}{\mathrm{d}\theta} \;=\; \frac{\partial \mathcal{L}}{\partial \theta}\Big|_{w^*,\lambda^*}$$

Checked against central finite differences on the optimal value:

```{image} _static/plots/gradient_check.png
:alt: envelope-theorem gradient against finite differences
:width: 70%
```

54 components, at states the controller actually visits during a run — the
gradient is checked where it is used, not at points placed on the centreline.
Each finite difference costs two extra solves; the envelope-theorem value costs
one evaluation of an expression that CasADi already has.

| | |
|---|---|
| cosine to finite differences | **0.99999** |
| relative error | **< 1e-3** |
| cost, against the solve | 0.079% |

`tests/test_gradient.py` asserts this, and `examples/gradient_check.py` shows
it across states and weight settings. It costs one evaluation of a function
built once, on quantities the solver already returned.

## It tunes itself into a good controller

Starting from `q_l = 200`, `q_v = 0.05` — far too much lag penalty, almost no
reward for progress, so the MPCC crawls:

| episode | metres covered | `q_l` | `q_v` |
|---|---|---|---|
| 0 | 19.3 | 195 | 0.05 |
| 3 | 23.7 | 183 | 0.06 |
| 6 | 30.9 | 144 | 0.10 |
| 8 | 37.1 | 101 | 0.48 |
| 10 | **37.2** | 66 | 3.5 |
| 12 | 37.1 | 43 | 27 |

Distance covered nearly doubles in a dozen episodes, no crashes on the way up.
37.2 m in 200 ticks is 3.7 m/s average against a grip-limited corner speed of
3.9 — it has essentially found the optimum, and nobody touched the weights.

## Then it destroys itself

| episode | metres covered | `q_c` | `q_v` |
|---|---|---|---|
| 12 | 37.1 | 0.36 | 27 |
| **13** | **20.1  OFF-TRACK** | 0.30 | 36 |
| 20 | 5.2  OFF-TRACK | 0.26 | 37 |
| 23 | 5.1  OFF-TRACK | 0.22 | 45 |

The mechanism is worth more than the success. Once performance saturates the TD
error stays slightly positive — the critic under-estimates a policy that is
already at the limit — so `q_v` keeps climbing and `q_c` keeps falling long
after either helps. By episode 13 the MPCC is willing to ride the constraint
boundary, the tyre limit it does not model puts it off the track, and with
`q_c ≈ 0.2` it has no way back.

**The tuner optimised a proxy past the point where the proxy was valid, with
nothing to stop it.** That is the honest result of the spike: the gradient is
exact and cheap, the loop works, and it has no stopping criterion.

Candidate fixes, none tried here:

* a **decaying step size** — the cheapest thing that would work;
* **keep the best `theta` and revert on regression** — cheap and crude, and it
  needs a notion of "the same conditions" to compare across;
* a **trust region on `theta`** — principled, and the right long-term answer;
* a **predictive safety filter around the whole thing** — the interesting one,
  because this failure is exactly what such a filter exists for. That
  construction is in
  [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground)'s
  `safety.py`; on a comparable task it took an unfiltered learner from crashing
  in 61% of training episodes to 0%, with no cost in final performance.

## Situation-dependent weights, over six seeds

The best result on this problem, and the one that needed confirming: one
$\theta$ per curvature segment, the segment read from the path *ahead*
(`experiments/per_segment_weights.py`). It was measured on **one seed**, and the
failure it claims to fix — the tuner walking $\theta$ out of the good region —
is stochastic, so one draw could get it wrong in either direction.

`experiments/per_segment_seeds.py`, six independent seeds, 26 episodes each,
oval, everything else identical:

| | last 8 episodes | spread | seeds ending collapsed |
|---|---|---|---|
| global $\theta$ | 8.9 m | sd 0.9, range 7.6–9.8 | **6 / 6** |
| **per-segment $\theta$** | **77.8 m** | sd 0.2, range 77.5–78.0 | **0 / 6** |

**It reproduces.** The distributions do not overlap — the *worst* per-segment
seed is eight times the *best* global one — and the per-segment spread (0.2 m)
is tighter than the global one despite being nine times the magnitude. Seed 0
returns 7.6 m and 78.0 m, the published pair, to the decimal.

### What it does is not "find better weights"

The row that matters is the best episode, not the last:

| | best episode | first off-track |
|---|---|---|
| global $\theta$ | 77.6 – 79.1 m | episode 3, 3, 4, 5, 5, 5 |
| per-segment $\theta$ | 78.0 – 78.3 m | **never, on any seed** |

The global tuner finds a ~78 m controller on **every** seed, and then destroys
it on **every** seed, within five episodes. Per-segment weights reach the same
peak and *keep* it. So the mechanism is not that per-segment $\theta$ is a
richer parameterisation that reaches a better policy — it reaches the same
policy — it is that **a bad update made in a corner no longer moves the
straight's weights**, so there is no longer a single vector for the collapse to
propagate through.

### The counter-intuitive weights, explained

The single-seed run reported `q_v = 71.2` in the tightest segment against
`29.5` on the straight — progress weighted *higher* in the hairpin, which is
backwards — and the honest note at the time was that reading a mechanism into
one seed would be over-interpretation. Six seeds say it was.

The direction survives weakly (hairpin above straight on 5 of 6 seeds, mean 35.5
against 24.5) but the dramatic figure does not: **71.2 is a seed-0 outlier**, and
the other five sit between 24.7 and 30.8.

More to the point, the comparison is meaningless. Across all six seeds and all
three segments, `q_v` lands between **19.7 and 71.2** — and
`q_v` saturates above ≈2 (`experiments/weights_as_behaviour.py`: mean speed
0.40 → 3.92 m/s over `q_v` 0.02 → 2, and flat thereafter). Every one of
those numbers is in the dead zone, so they are all the *same behaviour*, and
ordering them is reading structure into a flat region.

**The weight that actually carries the schedule is `q_c`**, and it is not
counter-intuitive at all:

| | straight | long curve | hairpin |
|---|---|---|---|
| mean `q_c` | **0.42** | 1.95 | 1.22 |

`q_c` on the straight is the lowest of the three on **6 of 6 seeds**, by a
factor of three to six. Hold the line through the corners; let it run wide on
the straight. That is the schedule, it is stable across seeds, and it was
invisible while `q_v` was being read.

## The same comparison on a different track reverses it

`experiments/per_sector_weights.py`, `Track.circuit()` — a 47.2 m lap with
straights, a chicane of two 90-degree corners, two 180-degree hairpins and a
pair of sweepers, minimum radius 2.59 m. Six seeds, 26 episodes, everything else
identical to the oval runs above.

| | $\theta$ vectors | last 8 episodes | sd | seeds collapsed |
|---|---|---|---|---|
| **global** | 1 | **79.84 m** | 0.07 | **0 / 6** |
| curvature bins | 3 | 78.99 m | 0.19 | 0 / 6 |
| named sectors | 4 | 78.92 m | 0.07 | 0 / 6 |

Two results, and the second is not the one the experiment was built to get.

### The collapse is a property of the track, not of the parameterisation

On the oval a global $\theta$ collapsed on **6 of 6** seeds. On the circuit it
collapses on **0 of 6**, and reaches 79.84 m with a standard deviation of 7 cm.

So the six-seed result above shows that per-segment weights fix **the oval's**
failure. It is not evidence that they are a better parameterisation, and it
should not be read as any. The honest statement of what was demonstrated is
narrower than it first appeared.

The likely mechanism, which the two tracks together suggest: the oval is mostly
straight, so the tuner is rewarded for raising `q_v` over most of the lap and
then meets a 180-degree corner carrying weights tuned for a straight. The
circuit is 82% non-straight, so that feedback arrives constantly and $\theta$
never gets the room to run away. **The variable is how much of the lap rewards
the wrong thing**, not how many weight vectors are held.

### Scheduling is not free

With no collapse to prevent, performance *decreases* monotonically in the number
of parameters, and the gaps are large relative to the seed spread:

| | difference | separation |
|---|---|---|
| named sectors − curvature bins | −0.07 m | 0.8 SE — **not separated** |
| global − curvature bins | +0.85 m | 10.4 SE — separated |
| global − named sectors | +0.92 m | 22.5 SE — separated |

**Named sectors buy nothing over curvature bins.** That is the question the
circuit was built to answer and the answer is no: four labels cost twice the
parameters of three and return a difference of 7 cm against a seed spread that
swamps it.

And the unscheduled controller beats both, by ten and twenty-two standard
errors. Extra weight vectors are extra freedom for the tuner to wander, and on a
track where nothing is going wrong that freedom has nothing to buy.

Taken with the oval, the defensible claim is narrower and more useful than
"situation-dependent weights help":

> **Situation-dependent weights pay when part of the lap rewards the wrong
> thing, and cost a little when it does not.**

### The caveat, which is not small

All three arms sit within 1 m of each other, at 4.0 m/s — the vehicle's speed
cap — for the whole run. This is a **ceiling-limited task**: the car is at its
limit and the grip constraint the controller does not model is active in the
corners (4.0 m/s at a 2.59 m radius demands 6.18 m/s² against a 6.0 limit).

So "scheduling does not help *here*" is measured, and "scheduling does not help"
is not. A task where scheduling has something to win — a track with genuinely
conflicting demands between sectors, or a speed range the cap does not truncate
— would be needed before the null result generalises. That experiment has not
been run.

## Real-time: solved, and measured

This section used to say the solve took ~150 ms against a 50 ms budget and was
"far too slow for a car". That was true of IPOPT solved to convergence. It is
not true of the SQP-RTI in `mpcc_tuning/rti.py` — one full QP per tick,
warm-started, which is what real-time iteration actually means.

`python benchmarks/solve_time.py`, 60 replayed states:

| solver | mean | **worst** | 20 Hz | 50 Hz | 100 Hz |
|---|---|---|---|---|---|
| IPOPT converged, N=12 | 27.3 ms | **53.4 ms** | miss | miss | miss |
| IPOPT converged, N=20 | 51.1 ms | **87.5 ms** | miss | miss | miss |
| **SQP-RTI, N=12** | **1.9 ms** | **3.4 ms** | **ok** | **ok** | **ok** |
| SQP-RTI, N=20 | 3.6 ms | 10.3 ms | ok | ok | miss |

**Read the worst case, not the mean.** IPOPT at N=12 averages 27 ms, inside the
20 Hz budget, and its worst case is 53 ms — outside it. A controller whose mean
fits and whose tail does not is a controller that misses deadlines, and the
distinction is invisible if only the mean is reported.

At N=12 the RTI has **14× headroom at 20 Hz** and fits 100 Hz. This is CasADi's
`qrqp` from Python; acados would be faster again, and
`MPCC_planner_acados` already runs `SQP_RTI`.

The gradient survives the change. The envelope-theorem sensitivity computed at
the RTI solution agrees with the converged one **exactly in direction** (cosine
1.0000) and is 16% smaller in magnitude — see
[Influence through a solver](influence_through_a_solver.md). So the tuning does
not have to be redone for the fast solver.

### What is still not real-time

The **tuner**, not the controller. Each learning step needs `Q(s,a)` at the
executed action, which is a second solve when the action was perturbed for
exploration. And the safety filters cost 0.6–57 ms on top (see
[Safety filters](filters.md)), so the total tick budget has to be counted with
whichever filter is in the loop, not with the controller alone.

Note also that the second solve is usually free. `Q(s, π(s)) = V(s)` by
definition, so the action-value solve is skipped entirely unless the applied
action was perturbed for exploration.
