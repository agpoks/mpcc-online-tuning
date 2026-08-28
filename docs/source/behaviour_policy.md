# A behaviour policy over θ, and whether it should be liquid

*A design note, written after reading the evidence rather than before.*

The idea in [TODO](https://github.com/agpoks/mpcc-online-tuning/blob/main/TODO.md)
item 2d is that a **recurrent or spiking net** should map *situation* → θ,
because a closing gap is a temporal pattern that no threshold on the
instantaneous gap can represent. The specific proposal considered here is a
**Liquid Time-Constant** head (Hasani et al., AAAI 2021), whose selling point —
a learned, input-dependent time constant — looks like exactly the right shape
for "decide once, hold between".

**The evidence is against it, and it is already measured.** That goes first.

## `rtrrl-playground` has already run this experiment

`rtrrl-playground`'s `overtake` task is the same problem in a different action
space: the same 1:10 car, the same oval, two slower cars holding the racing
line, the opponent's speed deliberately hidden. Every cell in that repo was run
on it, 400k steps, 6 seeds:

| cell | return | passes | crashes |
|---|---|---|---|
| `physics_ligru` | 425.7 | 2.07 | **14%** |
| `ligru` | 397.9 | **2.55** | 21% |
| `mlp` *(no memory)* | 374.2 | **3.07** | 35% |
| `liquid_gru` | 344.9 | 2.24 | 31% |
| `ctrnn` | 283.2 | 2.25 | 35% |
| **`ltc`** | **267.5** | **1.29** | 17% |
| `lrcu` | 186.6 | 1.40 | 28% |

**LTC came sixth of seven, made the fewest passes of any cell, and scored below
the memoryless MLP.** And `liquid_gru`'s own docstring in that repo is blunt
about the family: *"Being straight about the outcome: this cell does not win."*

Two honest qualifications, in both directions:

* That page also says **"nothing else here is separated"** — standard deviations
  of 137–296 on means of 187–426. The ranking should not be read. LTC is not
  *demonstrated* worse; it is *not demonstrated better*, which for a component
  one is proposing to add is the same practical answer.
* The one clearly separated result is that **every learned cell roughly halves
  the scripted policy's crash rate** (14–35% against 65%). The win there is
  *learning over scripting*, not *liquid over gated*.

## What does not transfer, and what does

**The task is genuinely different, so this is a prior and not a verdict.** There
the network emits nine discrete actions from eighteen lidar numbers and has to
learn to drive. Here it would emit **six log weights** and the MPCC drives —
with the keep-out constraint holding feasibility, which in fifteen runs never
once failed. That is a far smaller job, and a prior from the harder task does
not settle the easier one.

**One signal transfers, and it is a warning.** LTC is the *cautious* end of that
table: fewest passes, low crashes. In this repo caution has a name and a
measurement — `experiments/overtake_or_follow.py` shows *following* is a stable,
safe, low-progress attractor: 12.0 m covered, 0.73 m behind, 7 mm of lateral
wander, held for the entire run, on every one of six cells that chose it.
Passing pays 36.8 m. An LTC head risks parking in the follow region and
**scoring respectably for never trying**, which is the exact failure
`rtrrl-playground`'s own page warns about: *"An agent that never passes anybody
and never crashes scores respectably, because progress alone pays."*

## The architecture the measurement actually argues for

Today's grid changes the design, and it argues *against* a network doing the
whole job. The behaviour boundary is:

$$\frac{q_v}{q_c} \;\lessgtr\; 1 \qquad \text{15 of 15 cells}$$

and since θ is already the **log** weights, that is a *difference of two
components of θ* — a **linear** boundary. The geometric half of the decision is
linear, and measured.

So the recurrence is not needed for the decision. It is needed for exactly one
quantity: the **closing rate**, which is a derivative of a range and therefore
genuinely unavailable from one frame. That suggests

```
  opponent range (+ curvature preview)
        │
        ▼
  small recurrent estimator  ──►  closing rate, time-to-collision
        │
        ▼
  linear map (measured: the q_v/q_c boundary)
        │
        ▼
  θ, clamped to the q_v ceiling
```

**Recurrence confined to the one quantity that provably needs it**, and
everything downstream linear, bounded, and already measured. That is a smaller
claim than "an LTC chooses the behaviour", and it is the one the evidence
supports.

## The precondition that decides whether any of this is needed

**In this repo the controller is handed the true state, and opponents are passed
to `MPCC.set_obstacles` as exact $(x, y, r)$.** If the policy is also handed the
opponent's *velocity*, there is no hidden state, no temporal pattern, and no
role for recurrence — an LTC head would be decoration, and item 2d's own gate
would correctly kill it.

**The temporal argument requires deliberately withholding the opponent's
velocity.** That is a design decision to be made explicitly and defended, not
inherited by accident. `rtrrl-playground` makes it deliberately and says so
("*The speed of the car in front. Never observed, redrawn every episode.*"), and
it is defensible here for the same reason — a real car downstream of
`obstacle_perception` gets positions, not velocities. But it must be *chosen*,
and the choice is what creates the problem the network is then credited with
solving.

## The τ argument, restated honestly

The structural argument survives the benchmark and is worth keeping. LTC's
effective time constant

$$\tau_{\text{eff}} = \frac{\tau}{1 + \tau f(x, h)}$$

is a **run-time hold duration set by what is happening** — fast when the gate
opens, holding when it closes. That is the principled form of item 2d's
"re-decide on events, hold between", and it is a stronger claim than "recurrent
nets handle sequences". This repo already names τ as the thing one otherwise has
to guess: see
[Influence through a solver](influence_through_a_solver.md), where `leak_max =
0.99` is called an arbitrary cap and `liquid_gru`'s learned τ is what replaces
it.

The flip side is measured too, in `liquid_gru`'s docstring: the τ floor that
guarantees the influence series converges is simultaneously a **ceiling on
memory**, and the price is monotone and visible on a pure-memory task (+0.291
at a leak floor of 0.89, rising to +0.775 at 0.980, against LiGRU's +0.883 with
no floor). τ is not free.

## If it is built

Per the repo's standing rule, **copy in, do not import.** `ltc.py` there is 117
self-contained NumPy lines with analytic derivatives and a `leak` output, which
is the whole file worth taking.

The gate is item 2d's own, unchanged: **only claim the network if it beats (i) a
fixed schedule and (ii) a per-tick MLP.** With a specific prediction attached
now — the fixed schedule should be *strong*, because the boundary is linear —
so the network can only win on the temporal axis, and if it does not beat a
per-tick MLP given identical features, it is decoration. Given the table above,
the honest prior is that it will not.

## Testing it against more than one car

The keep-out is already built for several: `max_obstacles` is a construction
parameter and `Plant(opponents=[...])` takes a list, so two or three opponents
on the bicycle plant work today.

For a realistic multi-car test, `scuderia_gym_jax` has it —
`examples/overtake.py` and `envs/multi_agent_env.py`, where `num_agents` cars
share one `State`, one `step` integrates all of them, and a pairwise SAT check
runs on their actual rectangles, with ST/STD vehicle models underneath. Its
traffic is dumb by the same deliberate choice made here.

Two things gate that, and neither is about the network:

* **The tuner does not yet survive that plant at all** — see
  [the plant page](plant.md) and TODO item 4. It is a reward-design problem
  ("do not drive" scores ≈ 0 against a crash's −5), and it comes first.
* **Its maps have no centreline**, so track geometry stays with
  `mpcc_tuning.track.Track` and only the dynamics are borrowed, exactly as the
  existing bridge does.

So: multiple cars on the bicycle plant now; multiple cars on `scuderia` after
item 4.
