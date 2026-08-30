# Animations

The results page reports that the tuner improves the controller and then
destroys it, in two separate tables. The point of animating it is that the
collapse and the weight that causes it are visible in the same frame.

Everything here comes from `mpcc_tuning.viz`, which is part of the package
rather than a docs-only script:

```python
from mpcc_tuning.viz import animate_run
animate_run(mpcc, plant, theta, "run.gif")
```

To regenerate this page:

```bash
python scripts/make_animations.py               # all of them
python scripts/make_animations.py --only tuning
PYTHONPATH=/path/to/scuderia_gym_jax python scripts/make_animations.py --only scuderia
```

GIF rather than video: matplotlib's Pillow writer needs no ffmpeg, and a GIF
renders on Read the Docs with no player.

## What the controller believes is about to happen

```{image} _static/anim/mpcc_horizon.gif
:alt: MPCC driving the bicycle plant with its predicted horizon drawn each tick
:width: 100%
```

The orange line is the MPCC's predicted state trajectory over the next $N$
steps, re-solved from scratch every 50 ms, and the orange dot is the reference
point $p(s)$ — which the solver *chooses*, rather than obtaining by projecting
the car onto the path. See [the formulation](formulation.md#0-what-the-two-errors-are)
for why that distinction is what the lag term exists to police.

Drawing the horizon is the cheapest available diagnostic for a model-predictive
controller, because it makes model error visible *before* it becomes tracking
error. On a plant the MPCC models correctly, the prediction lies along the path
the car then actually follows. Which brings us to the plant it does not model.

## The same controller, on real tyres

```{image} _static/anim/mpcc_scuderia.gif
:alt: the same controller on scuderia_gym_jax's fitted tyre model, leaving the track
:width: 100%
```

Identical controller, identical weights, identical track. The only change is
the plant: `scuderia_gym_jax`'s ST model with tyres fitted to real RC-car
recordings, in place of a kinematic bicycle.

Watch the horizon through the corner. It keeps predicting a turn the car cannot
take, is wrong, is re-solved, and predicts it again — because the MPCC's
internal model is a kinematic bicycle that believes any speed is corner-able,
and the plant has slip angles. The car accelerates to 3.5 m/s into a 2.5 m
corner and leaves the track in under a second, every time.

This is the mismatch the tuning is supposed to absorb, and
[The plant](plant.md) reports what happens when it is asked to: the feasible
region exists — every open-loop run with `r_d=10` completes 400 steps — and the
online tuner does not reach it from the bicycle plant's initialisation.

## Two cost weights decide whether the car passes

```{image} _static/anim/mpcc_behaviour.gif
:alt: follow and overtake side by side against the same opponent
:width: 100%
```

The same controller, the same opponent, the same track. The only difference is
`q_v/q_c` — below 1 on the left, above it on the right — and the keep-out
circle is drawn rather than implied, because an animation showing a car
swerving without showing what it swerves around has hidden the mechanism.

Measured: **13.0 m following against 40.3 m overtaking**, neither crashing. The
live readout carries the ratio, so the number crossing 1.0 *is* the decision
being made.

## A stopped car is not a slow car

```{image} _static/anim/mpcc_static_vs_dynamic.gif
:alt: the same weights following a moving car, and parking behind a stopped one
:width: 100%
```

**Identical weights**, and the only difference is whether the obstacle is going
anywhere. Following a moving car works (13.0 m). Following a *stopped* one is
not caution, it is stopping — 2.6 m, and the episode simply times out.

This is why "stay behind" has to be conditioned on a classification, and why
that classification cannot come from a single frame: a stopped car and a slow
car are identical in one observation, and only their positions over time
differ. See [the behaviour policy](behaviour_policy.md).

## The tuner, tuning

```{image} _static/anim/mpcc_tuning.gif
:alt: trajectory and the six cost weights across an online tuning session
:width: 100%
```

One frame per episode on the bicycle plant, at
`examples/tune_online.py`'s defaults. Left, the lap; right, the six cost
weights on a log axis — which is the space they are learned in, and the only
scale on which `q_l` at 200 and `r_a` at 0.01 fit on one chart.

| episode | covered | `q_c` | `q_l` | `q_v` |
|---|---|---|---|---|
| *start* | — | 10.0 | 200.0 | 0.05 |
| 0 | 73.1 m | 2.52 | 4.63 | 11.1 |
| 4 | **79.1 m** | 0.92 | 1.33 | 28.6 |
| 5 | **8.5 m  off-track** | 0.94 | 1.45 | 26.0 |
| 11 | 7.6 m  off-track | 0.55 | 1.49 | 29.3 |
| 25 | 7.4 m  off-track | 0.41 | 1.40 | 22.0 |

**Most of the damage is done in episode 0.** Before a single episode boundary,
`q_v` has gone from 0.05 to 11 — a factor of 222 — and `q_l` from 200 to 4.6.
The tuner has correctly worked out that it is paid for progress and can buy
progress with tracking error, and it acts on that at a rate nothing bounds.

**It improves the controller, briefly.** Episodes 0–4 climb from 73.1 m to
79.1 m, which is real: the initial weights are conservative and there is genuine
speed to be found by relaxing them.

**Then it goes over the edge and stays there.** Episode 5 covers 8.5 m and
leaves the track, and the next twenty episodes never recover — 7 to 8 metres,
off-track every time. `q_c`, the weight holding the car near the path, has
fallen from 10 to 0.41 while `q_v` sits near 30. That controller is not tracking
a path any more; it is being paid to advance the progress variable and is
barely penalised for where the car ends up.

Two things worth taking from that, and neither is about the gradient — which is
exact to [five decimal places](results.md#the-gradient-is-exact):

**It is a reward-design failure.** Distance covered with a single terminal $-5$
for leaving the track gives almost no gradient signal about the boundary until
the boundary is crossed, at which point the weights are already somewhere no
small step returns from.

**It is a step-size failure too.** Moving a weight by 222× in one episode is not
learning, it is a divergence that happens to pass through a good region on the
way. A trust region on $\theta$, or a constraint keeping the weights in a set
where the controller is known stable, is the obvious missing piece — and is
exactly the gap that [Influence through a solver](influence_through_a_solver.md)
sets up.

### A note on which learning rate this is

The clip runs at `examples/tune_online.py`'s argparse default, $\alpha = 2\times
10^{-3}$. The table in [Results](results.md#it-tunes-itself-into-a-good-controller)
and the README were measured at $\alpha = 2\times 10^{-4}$ — ten times smaller —
and 200 ticks per episode rather than 400.

Both runs are real and they are the same story at two speeds, which is worth
seeing side by side:

| | $\alpha = 2\times10^{-4}$ | $\alpha = 2\times10^{-3}$ (animated) |
|---|---|---|
| improves for | ~12 episodes, 19 → 37 m | 5 episodes, 73 → 79 m |
| collapses at | episode 13 | episode 5 |
| after collapse | 5 m, off-track | 7–8 m, off-track, **never recovers** |
| `q_v` at the end | 45 | 22 |

The tenfold learning rate does not change *what* happens, only how many
episodes it takes to get there — which is the signature of a divergence rather
than a tuning artefact, and the reason the fix has to be a bound on where
$\theta$ may go rather than a smaller step.
