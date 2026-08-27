# Influence through a solver

*A formulation note, and the experiment that settles it.*

**Result, up front: the idea is not needed.** For a genuine real-time
iteration the memoryless gradient everyone already uses is exact in direction
(cosine $1.0000$) and $16\%$ off in magnitude. The recursion below is correct
and the term it adds is negligible in practice. What follows is kept because
the *question* was worth asking, because the measurement is the thing nobody
had made, and because the obvious way to test it gives the opposite answer for
a bad reason — see [The trap](#the-trap).

## The gap this addresses

Every MPC-as-function-approximator paper — Gros & Zanon, AD-MPCC, COAT-MPC,
`mpcrl`, MPC4RL — makes the same modelling assumption, usually without stating
it:

> the controller is a **memoryless map** from state to action,
> $u_t = \pi_\theta(s_t)$.

Under that assumption the envelope theorem is all you need: at a *converged*
solution the optimal value's derivative is the partial derivative of the
Lagrangian, the primal-dual solution can be held fixed, and
$\mathrm{d}w^*/\mathrm{d}\theta$ never appears. That is the argument this repo
already implements and verifies (`examples/gradient_check.py`, cosine 1.00000).

**It stops being true the moment you deploy.** Three standard things break it,
and all three are present on any real vehicle:

1. **Warm starting.** The solve at time $t{+}1$ starts from the solution at
   time $t$.
2. **Real-time iteration.** At 20 Hz you do not converge — you take one SQP
   step and apply the result (Diehl et al.). The applied $u_t$ is then a
   function of the *iterate*, not of the optimum.
3. **A moving-horizon estimator, or an online-adapted model.** The state the
   controller acts on is itself the output of a $\theta$-dependent recursion.

In each case $\theta$ influences future solves **through a carried state**. The
memoryless gradient is then not an approximation with a small constant — it is
missing a term that has no reason to be small.

## The formulation

Write the solver as what it actually is: a dynamical system.

$$w_{t+1} = \Phi(w_t,\, s_t,\, \theta), \qquad u_t = g(w_t)$$

with $w$ the primal-dual iterate (or the iterate plus the MHE state), $\Phi$
one RTI/SQP step, and $g$ the projection onto the first control. Compare an
RNN:

$$h_{t+1} = F(h_t,\, x_t,\, \theta), \qquad a_t = \text{head}(h_t)$$

**They are the same object.** The solver is a recurrent cell whose update
happens to be a Newton step.

So the influence obeys the same recursion RTRL does:

$$J_{t+1} \;=\; \underbrace{\frac{\partial\Phi}{\partial w}}_{D_t}\, J_t \;+\; \frac{\partial\Phi}{\partial\theta},
\qquad J_t = \frac{\partial w_t}{\partial\theta}$$

and the policy gradient a DPG-style update needs is

$$\frac{\partial u_t}{\partial\theta} = \frac{\partial g}{\partial w}\, J_t.$$

That is the whole idea. **The envelope theorem handles the converged,
memoryless case; this handles the deployed one.**

## What $D_t$ is, and the part that is genuinely nice

For a Gauss-Newton RTI step on the KKT system with Hessian approximation $B$
and KKT matrix $M$, the iteration near a solution is affine:

$$w^{+} = w - M^{-1} r(w,\theta) \quad\Longrightarrow\quad D = I - M^{-1}\nabla_w r$$

$D$ is the **Newton contraction matrix**. For an RTI scheme that is stable at
all, its spectral radius satisfies $\rho(D) < 1$ — that is precisely the
condition under which real-time iteration tracks the moving optimum
(Diehl, Bock & Schlöder's contraction estimate).

And now the observation that makes this more than a restatement:

> **The solver's contraction rate is the influence trace's decay rate.**

In `rtrrl-playground` the same quantity is called `leak`, it is the thing that
must stay below 1 for the influence series
$P \leftarrow \text{leak}\cdot P + \text{imm}$ to converge, and for a recurrent
network **you have to guess it** — that is what `leak_max = 0.99` is, an
arbitrary cap, and what `liquid_gru`'s learned $\tau$ replaces it with.

Here you do not have to guess. $\rho(D)$ is a property of the QP you are
already solving, it is computable, and it says exactly how far back the
influence has to be carried:

$$\big\|J_t - J_t^{(k\text{-truncated})}\big\| \;=\; O\!\big(\rho^k\big)$$

**A principled truncation horizon, derived rather than tuned.** That is the
piece I have not seen anywhere, and it is the reason this is worth writing up
rather than just implementing.

## Cost, and the approximations that follow

$J$ is $n_w \times n_\theta$ with $n_w = (N{+}1)n_x + N n_u$ — for the MPCC in
this repo, $13\cdot 5 + 12\cdot 3 = 101$, times six parameters. That is small.
Propagating it costs one $D J$ product per tick, and $D$ is the KKT matrix's
contraction — **already factorised**, because you just used it to take the step.

So for a small parameter set the exact recursion is affordable, which is the
opposite of the RNN case, and worth saying plainly: *this is one of the rare
settings where exact RTRL is the cheap option.*

When it is not — many parameters, long horizons — the same three bargains from
`rtrrl-playground` transfer directly, and their names are already familiar:

| approximation | here it means |
|---|---|
| truncate at $k$ | drop the influence older than $k$ ticks; error $O(\rho^k)$, and $\rho$ is known |
| RFLO-style | keep only the block-diagonal of $D$ — the per-shooting-node coupling, dropping the coupling *between* nodes |
| UORO | a rank-1 sketch of $J$; unbiased, and needs only $D$-vector products, which the factorised KKT matrix gives for free |

## Measured, and refuted

`experiments/rti_influence.py`. A recorded state sequence is **replayed**, so
the only path from $\theta$ to the iterate at tick $T$ is the chain of warm
starts — the plant is out of the loop and the solver's memory is isolated.

| how $\partial u_0/\partial\theta$ is obtained | $\|J\|$ | cosine to memoryless |
|---|---|---|
| **SQP-RTI, one full QP per tick, warm-started** | 1.681 | **1.0000** |
| memoryless (converged, cold) | 1.994 | 1 |
| warm-started *and* converged | 1.994 | 1.0000 |

**The direction is identical.** The magnitude differs by 16%, which for a
gradient method whose step size is a tuned hyperparameter anyway is not a
distinction that survives contact with practice.

The warm start *is* memory — a perturbation of it decays geometrically at a
measured $\rho \approx 0.77$ (bicycle) and $0.87$ (fitted tyres), i.e. over
tens of ticks. The recursion above is a correct description of that. It simply
turns out that the component of it which reaches $\partial u_0/\partial\theta$
is small, because one QP step from a good warm start lands very close to the
converged solution: measured mean step $\|\Delta w\| = 0.748$, against a
converged step of $0.748$.

So the honest statement is the one the note said would be worth having:

> The memoryless assumption in MPC-as-function-approximator is **justified for
> real-time iteration**, and this is the measurement that justifies it. Nobody
> currently states it as an assumption at all.

(id trap)=
## The trap

There is an obvious way to test this that gives the opposite answer, and it is
wrong.

Approximating RTI by capping an interior-point solver at one iteration
(`ipopt.max_iter = 1`) produces a gradient that looks **orthogonal** to the
memoryless one — cosine $0.001$ on the bicycle, $-0.36$ on the fitted-tyre
plant. That is not evidence of solver memory. It is the sensitivity of a
*failed solve*:

* **0 of 41** solves report success;
* the iterate moves by $8.19$ on average, an order of magnitude **further**
  than a converged step ($0.748$), because an interior-point method's first
  step is large and not yet meaningful.

Real-time iteration is an **SQP** scheme — linearise once, solve one full QP —
and `mpcc_tuning/rti.py` implements that. The difference is not a detail: it
inverts the result. An earlier version of this note reported the capped-IPOPT
numbers as confirmation, and they are in `tests/test_rti_influence.py` now as a
regression, so the mistake cannot come back quietly.

## What it predicted, and how it was falsified

Three testable claims, in increasing order of how much they would need to be true:

1. **The memoryless gradient is measurably wrong under RTI.** Compare
   $\partial u/\partial\theta$ from the envelope/IFT-at-convergence assumption
   against finite differences *through the actual warm-started RTI loop*. If
   they agree, the whole idea is unnecessary and that is worth knowing in an
   afternoon. This is the first experiment and it is cheap.
2. **Carrying the influence improves online tuning.** Against the memoryless
   baseline (`mpcrl` / MPC4RL), same budget, same task.
3. **$\rho(D)$ predicts the required truncation.** Sweep $k$, show the error
   falls as $\rho^k$ with the *measured* $\rho$.

Claim 1 is the load-bearing one. If it fails, claims 2 and 3 are moot, and the
honest outcome is a note saying the memoryless assumption is fine in practice —
which would itself be worth having written down, since nobody currently states
it as an assumption at all.

## Relation to the rest of this work

* [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground) has the
  influence machinery — RTRL, RFLO, SnAp-1, UORO — with every Jacobian checked
  against finite differences, and the `leak < 1` convergence condition that
  reappears here as $\rho(D) < 1$.
* That repo's `physics_ligru` is the same idea in miniature: reserve part of
  the recurrent state for a model you *know*, and carry its influence exactly
  rather than approximating it. There it was a null result on the task's
  return — the physics was on the wrong half of the problem — but the
  *mechanism* is the one generalised here, with the solver as the extreme case
  where the known part is everything.
* The unifying statement, and the one a paper would be built on:

> **When part of your recurrent state is a model you know, carry its influence
> exactly and approximate only the rest.**

An RNN is the case where you know nothing; an MPC is the case where you know
everything; a physics-structured cell is in between. The estimator should
follow the structure, and at present nobody's does.

## What is missing before this is a paper

- [ ] **acados RTI**, so $\Phi$ is a real one-iteration step rather than IPOPT
      run to convergence. Without it there is no carried state to propagate and
      claim 1 cannot even be tested.
- [ ] **Experiment 1**, above. Cheap, decisive, and it goes first.
- [ ] **The memoryless baseline** (`mpcrl` or MPC4RL), which is the actual
      competition and not "no adaptation".
- [ ] **Hardware.** ICRA will not take this simulation-only; L4DC or CDC might.
- [ ] **A stopping criterion for the tuner** — see the README's "What it does,
      measured". As it stands the loop improves and then destroys itself, and
      that is the first thing a reviewer will find.
