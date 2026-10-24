# Opponents

For $\vect\theta$ to be a *behaviour* parameterisation — for the weights to
choose **overtake vs stay behind** — the opponent has to be in the optimal
control problem. It is: the OCP carries oriented keep-out regions as run-time
parameters, and this page is how they work and what the weights then express.

## The keep-out, in the OCP

Per opponent $j$ and shooting node $k$, with the opponent's radius the sum of
both cars' half-widths (the controller predicts a point mass, so the circle
carries the ego body too) plus a margin $m$:

$$r_{\mathrm{eff}} = r_{\mathrm{raw}} + m,\qquad
  (x_k - o_x)^2 + (y_k - o_y)^2 - r_{\mathrm{eff}}^2 \ge 0 .$$

Opponents are **run-time parameters, not structure**: the OCP is built once for
a fixed number of slots and only the positions and radii change per tick. In the
shipped acados controller these are oriented keep-out **ellipses**; the mechanism
below is identical, and the count-invariant aggregate observation lets one policy
face one to three of them without retraining.

**Inactive slots are switched off arithmetically.** An unused slot is passed with
$r_{\mathrm{raw}} = -m$, so $r_{\mathrm{eff}} = 0$ exactly and the row degenerates
to $\mathrm{dist}^2 \ge 0$, which holds everywhere. A *nearly* inert keep-out is a
live constraint at the origin — and the origin is on the oval's centreline — so
the tests assert the zero is exact, not small.

**Soft, not hard, and for a reason.** The keep-out rows are softened with
nonnegative slacks under a quadratic penalty, so "stay behind" is a *feasible,
finite-cost* option. A hard keep-out turns being caught out into an infeasible
solve, and an infeasible solve is not a behaviour — it is a missing control
input. The solver has to be able to *choose* badly for the choice to be one the
weights make.

**Not at stage 0.** Stage 0 is pinned to the measured state, so a constraint
there is a statement about the past; it is unsatisfiable exactly when the car is
already touching an opponent — the moment the solver most needs to still return
something. The keep-out runs over $k = 1\dots N$.

## The opponent

`mpcc_tuning/opponents.py`. Deliberately dumb: it drives the racing line at a
constant speed and does not react or defend. That is the right first opponent for
*"is overtake-vs-follow expressible as a choice of weights"*, because a reactive
opponent makes the outcome depend on two policies at once. A collision ends the
episode the way leaving the track does and pays the same penalty; which one
happened is recorded in `plant.failure` rather than added to the return, so the
reward does not invent a second penalty scale by accident.

## Measured: overtake vs follow is a choice of weights

`experiments/overtake_or_follow.py`. The oval, an opponent 3 m ahead at 1.0 m/s
against an ego car capable of ~3.9, 200 steps, only $q_c$ and $q_v$ varied.

| `q_v` | `q_c` | `q_v/q_c` | covered | outcome |
|---|---|---|---|---|
| 0.5 | 1.0 | 0.50 | 12.0 m | followed |
| 0.5 | 0.3 | 1.67 | 22.3 m | **passed** |
| 2.0 | 1.0 | 2.00 | **36.6 m** | **passed** |
| 2.0 | 0.1 | 20.0 | 3.3 m | passed, then **off-track** |
| 10.0 | 1.0 | 10.0 | 3.1 m | passed, then **off-track** |

Three things, in the order they matter.

**The behaviour is set by the ratio.** Every cell with $q_v/q_c \le 1$ follows;
every cell with $q_v/q_c > 1$ passes — 15 for 15. Since $\vect\theta$ is the log
weights, that boundary is *linear*, which is why the geometric half of the
decision needs no network (see {doc}`behaviour_policy`).

**Safety is set by $q_v$ alone, a different axis.** The ratio chooses *what the
car decides*; the magnitude of $q_v$ decides whether it survives deciding it.
Every pass at $q_v = 10$ leaves the track, one of two at $q_v = 2$ does, none at
$q_v = 0.5$. So the bound that keeps a pass clean is **a ceiling on $q_v$ with the
ratio left free** — not a box on the ratio. This is the measurement behind the
class-conditioned bands in {doc}`safety`.

**The keep-out itself never failed.** Of fifteen runs, none ended in a collision;
every failure was off-track *while* going around, and the minimum clearance to
the opponent was always positive. The constraint did its job in all fifteen —
which is exactly the separation the soft keep-out plus a hard actuator set is
designed to produce.
