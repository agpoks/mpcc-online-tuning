# The formulation, written out

Notes to myself while building the spike. The point of writing these down is
that three of the four things that went wrong were sign or scaling errors that
looked like tuning problems, and each one is obvious in the algebra.

## 1. The MPC is the function approximator

Let the MPCC's parametrised optimal-control problem be

$$
J^*(s;\theta) \;=\; \min_{w}\; f(w,\theta)
\quad\text{s.t.}\quad g(w, s, \theta) = 0,\; h(w,\theta) \le 0
$$

where $w$ collects the predicted states and controls and $\theta$ the log cost
weights. Then three RL objects come out of *one* problem:

| RL object | MPC object |
|---|---|
| policy $\pi_\theta(s)$ | the first control of the solution, $u_0^*$ |
| state value $V_\theta(s)$ | $-J^*(s;\theta)$ |
| action value $Q_\theta(s,a)$ | $-J^*(s;\theta)$ with $u_0$ pinned to $a$ |

**The minus signs are not cosmetic.** The MPC minimises a cost; RL maximises a
return. Both the value *and its gradient* carry the negation, and applying one
without the other drives every weight the wrong way while looking exactly like
a learning rate that is too high. That is what the first working version did,
and `tests/test_gradient.py::test_tuner_moves_progress_weight_up_when_progress_pays`
exists so it cannot happen twice.

Pinning $u_0$ is not a second problem. It is the same NLP with
$\text{lb} = \text{ub} = a$ on two decision variables — and when $a = \pi(s)$,
$Q = V$ by construction and the solve can be skipped entirely.

## 2. The gradient is free (envelope theorem)

The one that makes this real-time. Write the Lagrangian

$$\mathcal{L}(w,\lambda,\mu,\theta) = f(w,\theta) + \lambda^\top g(w,s,\theta) + \mu^\top h(w,\theta)$$

At a solution $(w^*, \lambda^*, \mu^*)$, under the usual constraint
qualification,

$$\frac{\mathrm{d}J^*}{\mathrm{d}\theta} \;=\; \frac{\partial \mathcal{L}}{\partial \theta}\Big|_{w^*,\lambda^*,\mu^*}$$

The primal and dual variables are held **fixed**. There is no
$\mathrm{d}w^*/\mathrm{d}\theta$ term — it is annihilated by the stationarity
condition. So the gradient of the value with respect to the tuning parameters
costs one evaluation of a function built once at construction, on quantities
the solver already returned.

Contrast with deterministic policy gradient, which needs
$\mathrm{d}u_0^*/\mathrm{d}\theta$: *that* requires differentiating the KKT
system (implicit function theorem, one linear solve with the KKT matrix).
Affordable, but a different and larger piece of machinery. Q-learning is the
cheap door.

Here $\theta$ only enters $f$, so the $\lambda$ and $\mu$ terms vanish — but
they are implemented anyway, because the moment $\theta$ includes a *model*
parameter it enters $g$ and the term is the whole point.

## 3. TD(λ) on top

$$\delta_t = r_t + \gamma V_\theta(s_{t+1}) - Q_\theta(s_t, a_t), \qquad
e \leftarrow \gamma\lambda e + \nabla_\theta Q, \qquad
\theta \leftarrow \theta + \alpha\,\delta\,e$$

$r$ is the **real** objective — metres covered, leaving the track penalised —
and deliberately not the MPCC's internal cost. If they were the same quantity
there would be nothing to learn: the MPC would already be optimal for it by
construction.

Traces because the consequence of a weight being wrong shows up several ticks
after the tick it was wrong on. One number per parameter, six parameters.

## 4. Two scaling traps

**The gradient has no natural scale.** $J^*$ is a sum of weighted squared
errors and can be any magnitude, so $\nabla_\theta J^*$ can too, and a fixed
$\alpha$ means nothing across problems. Normalising by a running RMS of the
gradient makes $\alpha$ a step in *relative* terms — the only version of it
that transfers between tracks.

**The TD error must be clipped.** Leaving the track pays $-5$; an ordinary tick
pays about $+0.07$. The terminal transition therefore produces a $\delta$ two
orders of magnitude larger than any other, arriving multiplied by a trace that
has been accumulating for ten steps. Unclipped, one crash moves every weight
further than the whole preceding episode did and the run never recovers. This
was observed, on episode 2, not anticipated.

**Parametrise in logs.** The weights span orders of magnitude and must stay
positive. $\theta = \log(\text{weight})$ gives both for free, and makes a
gradient step multiplicative, which is what a scale parameter wants.

## 5. What is still missing for a vehicle

* **Solve time.** ~150 ms per NLP with IPOPT; 20 Hz needs < 50 ms for the whole
  tick. The answer is acados with a real-time iteration scheme — one SQP
  iteration per tick, warm-started from the last — which is what
  `MPCC_planner_acados` already does. The envelope gradient is available there
  too: it needs the objective's partial derivative and the multipliers, both of
  which acados returns.
* **Safety during tuning.** The weights are being changed on a moving vehicle.
  This is exactly the case for a predictive safety filter around the whole
  thing — including around the tuner, not just the controller — and
  [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground)'s
  `safety.py` is the same construction.
* **The interesting gap.** The MPC-RL literature treats the controller as a
  memoryless map from state to action. It is not: warm-starting, an MHE state
  estimate, and an online-adapted model all mean $\theta$ influences *future*
  solves through a carried state. That is precisely the situation forward-mode
  influence propagation (RTRL/RFLO) exists for, and I have not found anyone
  doing it. It is the reason these two repos belong next to each other.
