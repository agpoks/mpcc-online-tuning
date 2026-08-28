# Opponents, and why they had to come first

Item 2 of the plan is that $\theta$ is a **behaviour parameterisation**: `q_v`
sets aggression, `q_c` sets how strictly the racing line is followed, and a
policy mapping *situation* $\to \theta$ should be able to express
**overtake vs stay behind**.

That claim could not be tested, because **`mpcc_tuning/mpcc.py` had no obstacle
constraint at all.** With no opponent in the optimal control problem, "go
around" and "stay behind" are not two behaviours the weights choose between —
they are the *same problem*, and no policy over $\theta$, however expressive,
can distinguish them. This page is the precondition being removed.

## The formulation

Copied from `MPCC_planner_acados/scripts/generate_acados_solver.py` — its
`max_obstacles` block — and adapted from an acados soft path constraint to this
CasADi NLP. Per obstacle $j$ and shooting node $k$:

$$r_{\text{eff}} = r_{\text{raw}} + m, \qquad
  (x_k - o_x)^2 + (y_k - o_y)^2 - r_{\text{eff}}^2 \;\ge\; 0$$

with $m$ = `obs_margin` = 0.15 m. Obstacles are **runtime parameters**, not
structure: the OCP is built once for `max_obstacles` slots and only the
positions and radii change per tick — the same arrangement the acados version
uses, and the reason a keep-out costs nothing to add to a solve that was
happening anyway.

Two quirks are carried over deliberately, because keeping them identical to the
template makes the eventual acados port a swap rather than a rewrite.

**Inactive slots are switched off arithmetically.** An unused obstacle is
passed with $r_{\text{raw}} = -m$, so $r_{\text{eff}}$ is *exactly* zero and the
constraint degenerates to $\text{dist}^2 \ge 0$, which holds everywhere. The
alternative — a $\max(0, \cdot)$ on the radius — would put a kink in the NLP for
no benefit. `tests/test_obstacles.py` asserts the zero is exact rather than
small, because a *nearly* inert keep-out is a live constraint at the origin, and
the origin is on the oval's centreline.

**The slack is in units of squared distance,** because the constraint is. Worth
knowing when reading the penalty weights: $Z = 200$ is a cost per m² of overlap
area, not per metre of intrusion.

### Slacks, written out

acados softens a path constraint by naming it in `idxsh` and giving it `Zl`,
`zl`. A plain NLP has no such machinery, so the slack is an explicit decision
variable:

$$h_{jk} + s_{jk} \ge 0, \qquad s_{jk} \ge 0, \qquad
  J \mathrel{+}= Z s_{jk}^2 + z s_{jk}$$

which is what `Zl` and `zl` *are*. The slacks live at the **end** of the
decision vector, after the states and controls, so `_nx` and the `u0` slice keep
their meaning and `mpcc_tuning/rti.py` needs no change to its layout
assumptions.

Soft, not hard, and for a specific reason: it makes "stay behind" a
**feasible, finite-cost** option. A hard keep-out turns being caught out into an
infeasible solve, and an infeasible solve is not a behaviour — it is a missing
control input. The solver has to be able to *choose* badly for the choice to be
one the weights make.

### The keep-out is not applied at stage 0

Stage 0 is pinned to the measured $x_0$, so a constraint there is a statement
about the past. It would be unsatisfiable exactly when the car is already
touching an opponent — the one moment the solver most needs to still return
something. Constraints run over $k = 1 \dots N$.

### `max_obstacles = 0` is the old problem

The default. No parameters, no slacks, no extra constraint rows, an unchanged
decision vector. Every result measured before this page is still comparable, and
`tests/test_obstacles.py::test_zero_obstacles_changes_nothing` is what keeps it
that way.

## The gradient survives, and this was checked

Adding slacks puts new decision variables in $w$ and new rows in $g$. The
envelope theorem still applies — $\theta$ enters neither the slack penalty nor
the keep-out — but "still applies in principle" is the kind of claim this repo
checks against finite differences rather than asserts.

With a keep-out active, at a state the controller actually visits:

| | |
|---|---|
| cosine to finite differences | **> 0.999** |
| relative error | **< 5e-2** |

the same thresholds as the obstacle-free
[gradient check](results.md#the-gradient-is-exact).

One consequence worth stating rather than discovering later: the slack penalty
is part of $J$, so it is part of $V = -J^*$. **The value function now knows
about obstacle proximity** — being close to an opponent is expensive in the
critic, not only in the constraint. That is the behaviour you want, and it is
not something that had to be added.

## The opponent

`mpcc_tuning/opponents.py`. Deliberately dumb: it drives the centreline at a
constant speed with a fixed lateral offset, does not react, does not defend, and
has no controller. That is the right first opponent for *"is overtake-vs-follow
expressible as a choice of cost weights"*, because a reactive opponent makes the
outcome depend on two policies at once and makes the question harder to read,
not easier.

Its radius is the sum of both cars' half-widths — the MPCC predicts a point
mass, so the opponent's circle has to carry the ego car's body too, exactly as
`obs_margin` does in the acados template.

**One sign convention, and it is not obvious.** The track exposes two lateral
measures with *opposite* signs: `Track.errors` returns the contouring error
$e_c$, and `Track.lateral` returns $-e_c$. Off-track is judged by `lateral`, so
`offset` is in `lateral`'s convention. Getting this backwards puts the opponent
on the wrong side of the track and nothing else complains, which is why it is a
test and not a comment.

In the plant, a collision ends the episode the way leaving the track does and
pays the same $-5$: from the reward's point of view both are "the run is over
and it is your fault", and inventing a second penalty scale would be a reward
design decision made by accident. Which one happened is recorded in
`plant.failure` rather than added to the return, so the four-tuple every caller
unpacks still fits.

## Measured: overtake vs follow is a choice of weights

`experiments/overtake_or_follow.py`. The oval, an opponent 3 m ahead driving the
centreline at 1.0 m/s against an ego car capable of about 3.9, 200 steps. Only
`q_c` and `q_v` vary; everything else is default. Runs are deterministic — no
exploration noise — so each cell is exact, and the caveat is the *geometry*
(one opponent, one speed, one starting gap), not the sampling.

| `q_v` | `q_c` | `q_v/q_c` | covered | closest approach − r | max\|lat\| | outcome |
|---|---|---|---|---|---|---|
| 0.5 | 10.0 | 0.05 | 12.0 m | 0.732 | 0.007 | followed |
| 0.5 | 3.0 | 0.17 | 12.0 m | 0.730 | 0.008 | followed |
| 0.5 | 1.0 | 0.50 | 12.0 m | 0.729 | 0.021 | followed |
| 0.5 | 0.3 | 1.67 | 22.3 m | 0.123 | 0.405 | **passed**, step 99 |
| 0.5 | 0.1 | 5.00 | 26.8 m | 0.139 | 0.398 | **passed**, step 73 |
| 2.0 | 10.0 | 0.20 | 12.0 m | 0.732 | 0.006 | followed |
| 2.0 | 3.0 | 0.67 | 12.0 m | 0.730 | 0.007 | followed |
| 2.0 | 1.0 | 2.00 | **36.6 m** | 0.203 | 0.465 | **passed**, step 33 |
| 2.0 | 0.3 | 6.67 | **36.8 m** | 0.197 | 0.443 | **passed**, step 32 |
| 2.0 | 0.1 | 20.0 | 3.3 m | 0.167 | 0.671 | passed, then **off-track** |
| 10.0 | 10.0 | 1.00 | 12.1 m | 0.691 | 0.021 | followed |
| 10.0 | 3.0 | 3.33 | 2.3 m | 0.197 | 0.632 | passed, then **off-track** |
| 10.0 | 1.0 | 10.0 | 3.1 m | 0.180 | 0.675 | passed, then **off-track** |
| 10.0 | 0.3 | 33.3 | 3.7 m | 0.182 | 0.663 | passed, then **off-track** |
| 10.0 | 0.1 | 100 | 3.5 m | 0.171 | 0.669 | passed, then **off-track** |

Three things, in the order they matter.

### The behaviour is set by the ratio, not by either weight

**Every cell with $q_v/q_c \le 1$ follows. Every cell with $q_v/q_c > 1$
passes.** Fifteen for fifteen, across two decades of `q_v` and two of `q_c`.
The follow behaviour is not indecision — it settles 0.73 m behind the opponent
with a maximum lateral excursion of 7 mm, and stays there for the whole run.

That is a stronger statement than "the weights are a behaviour dial", and it
matters for how a policy over $\theta$ should be built. Since $\theta$ is
already the *log* weights, the ratio is a **difference of two components of
$\theta$** — so this behaviour boundary is a linear function of $\theta$, and
the smallest policy that can express overtake-vs-follow is a linear one. That is
worth knowing before reaching for a network.

### Safety is set by `q_v` alone, and it is not the same axis

The ratio predicts *what the car decides*. It does not predict whether the car
survives deciding it: $q_v/q_c = 5.0$ completes the lap and $3.33$ leaves the
track. Sort by `q_v` instead and it is immediate — **every pass at $q_v = 10$
leaves the track**, one of two at $q_v = 2$ does, and none at $q_v = 0.5$ do.

So the two questions decouple: *the ratio chooses the behaviour, the magnitude
of `q_v` decides whether it is survivable.* A bound on the policy's output is
therefore **not a box in $\theta$**, and not a bound on the ratio either. It is
a ceiling on `q_v` with the ratio left free.

### The `q_v` dead zone is not benign

The existing sweep in `experiments/weights_as_behaviour.py` found `q_v`
saturating above ≈2 — everything above being "the same behaviour", a dead zone
where more `q_v` simply buys nothing because the car is already at its speed
cap. **With an opponent on the track that is false.** Above ≈2 the extra
progress weight stops buying speed and starts costing track: at `q_v = 10` every
attempted pass ends off-track, at every `q_c` tried.

The tuner drives `q_v` to 30 and beyond. On an empty track that was wasted
motion into a flat region. With traffic it is a region where overtaking reliably
crashes, which turns the ceiling on `q_v` from tidiness into the load-bearing
part of the safety argument.

### The keep-out itself never failed

Worth separating from the above: of the fifteen runs, **none** ended in a
collision. All five failures are `off_track`, and the minimum clearance to the
opponent across every step of every run was **+0.123 m** outside `r`. The
constraint did its job in all fifteen; the cars that failed ran wide *while*
going around, which is a track-limit failure and not an obstacle-avoidance one.

That distinction is available only because the plant records *which* failure
happened, and it is the reason `Plant.failure` exists rather than a single
"off" flag.
