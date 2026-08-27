# Safety filters

Seven implementations, one interface, one comparison. This page is the
reference for all of them: what each one checks, the equations it checks it
with, what it costs, and where it fails.

```{contents}
:local:
:depth: 2
```

## What a safety filter is

A safety filter is a map

$$u_\text{applied} = \Phi\big(x,\; u_\text{proposed}\big)$$

that returns $u_\text{proposed}$ unchanged unless applying it would forfeit the
ability to stay safe, in which case it returns the nearest input that does not.

The interface is deliberately narrow: **nothing in it knows what produced the
proposed input.** The same filter wraps a tuned MPCC, an untuned one, or a
random number generator, and the guarantee does not depend on which. That is
what separates a filter from every other approach to safe learning:

| approach | where it pays | what it guarantees |
|---|---|---|
| reward shaping | everywhere | nothing; the agent must crash to learn |
| constrained policy class | everywhere | usually costs the optimum too |
| Lagrangian / CMDP | everywhere | a constraint *in expectation*, which permits rare violations by construction |
| **safety filter** | **only at the boundary** | **per-step, for any policy** |

## The common vocabulary

Every filter below is written in these terms.

$h(x)$
: The **safety margin**, positive inside the safe set. Here it is how much
  corridor is left: $h = (w - \text{margin}) - |d|$, with $d$ the lateral
  offset and $w$ the half-width.

$\pi_b(x)$
: The **backup controller**. It does not have to be good, only safe. Here:
  full braking with the steering turned back towards the centreline.

$\mathcal{X}$
: The **constraint set** — on the track, with room.

$\mathcal{X}_\text{safe}$
: The **terminal set** — states you can remain in forever. Here, *stopped and
  inside the corridor*. A stopped car under braking stays stopped, so this set
  is control-invariant, which is the property the whole argument rests on.

`margin`
: How much narrower the filter's corridor is than the plant's. This absorbs
  model error and **must be non-zero**. At `margin = 0.10` the filter's
  corridor was 0.65 against the plant's 0.63 — *wider* — so it certified an
  input, the input put the car outside, and the filter first refused on the
  step the car was already off. The default is 0.18.

## The vehicle model they all share

Every filter predicts with the **plant's** kinematic bicycle, not the MPCC's:

$$
\begin{aligned}
v_{k+1} &= \mathrm{clip}\big(v_k + (a - c_d v_k)\Delta t,\; 0,\; v_\max\big) \\
\dot\psi &= \mathrm{clip}\!\left(\frac{v_{k+1}}{L}\tan\delta,\;
   \pm\frac{a_{\text{lat},\max}\, g}{v_{k+1}}\right) \\
x_{k+1} &= x_k + v_{k+1}\cos\psi_k\,\Delta t \\
y_{k+1} &= y_k + v_{k+1}\sin\psi_k\,\Delta t \\
\psi_{k+1} &= \psi_k + \dot\psi\,\Delta t
\end{aligned}
$$

with $L = 0.33$ m, $c_d = 0.15$, $v_\max = 4$ m/s, $a_{\text{lat},\max} = 6$
m/s², $\delta \in [-0.4, 0.4]$, $a \in [-4, 4]$.

Two details in there are load-bearing and both were bugs first.

**The yaw-rate cap.** $a_{\text{lat},\max}\,g / v$ is the tyre limit. The
MPCC's *own* prediction model does not have it, and a filter built on that
model certifies every input from every state — 45 out of 45, at every speed and
every point on the lap — because a car that can turn on a dime can always save
itself. That is a filter switched off while still reporting a 0% intervention
rate. This is the same quantity as `ay_max` in an acados MPCC's path
constraints.

**The integration order.** Position advances with the heading from *before* the
update. Doing it the other way looks equivalent and costs 1.4 cm per step
whenever the steering is non-zero — over a 30-step backup, a systematic error
comparable to the entire margin, and it certified braking manoeuvres that then
left the track. `tests/test_filters.py` asserts agreement with the plant to
exactly `0.0` over a grid of 45 $(\delta, a)$ pairs; "close" was the bug.

$g$ is the **grip**, which is never observed. What each filter does about that
is most of what distinguishes them.

---

## 1. ASIF — certify by exhibiting the manoeuvre

`mpcc_tuning.filters.ASIF`

### The condition

Apply the proposed input for one step, then run the backup for $N$ steps:

$$x_1 = f(x_0, u), \qquad x_{k+1} = f\big(x_k, \pi_b(x_k)\big)$$

Accept $u$ if and only if

$$x_k \in \mathcal{X}\;\;\forall k \le N
  \qquad\text{and}\qquad x_N \in \mathcal{X}_\text{safe}$$

### Why the terminal set is not optional

It is what makes this an **induction** rather than a lookahead.

*Claim.* If $x_1$ is certified, the filter is non-empty at the next step.

*Proof.* Take $u = \pi_b(x_1)$, the first move of the very backup that
certified $x_1$. It leads to $x_2$, from which the same backup continues, stays
in $\mathcal{X}$ for its remaining $N-2$ steps, and reaches
$\mathcal{X}_\text{safe}$. The certificate at $x_1$ asks for $N-1$ steps, one
more than remains — and that is exactly what $\mathcal{X}_\text{safe}$ is for:
it is invariant under $\pi_b$, so the extra steps cost nothing. Hence $u$ is
certified. $\square$

Drop the terminal condition and an $N$-step lookahead will approve full
throttle at a wall $N+1$ steps away, then again at $N$, and again, until every
input is too late. It was feasible at every step and crashed anyway.

### Cost and exactness

$\sim 30$ model steps per certificate — see the measured table for what that
costs in wall-clock. The argmin is **not** exact: the MPCC
emits continuous $(\delta, a)$ and this searches a structured candidate set
ordered by distance from the proposal, so it returns the nearest *sampled* safe
input. §4 is the version where the argmin is exact.

Deceleration is tried before steering, deliberately: a filter that swerves can
lose the car; one that brakes gives up progress, which is the currency the
tuner is trading in and therefore an intervention it can learn from.

### References

Gurriet, Singletary, Reher, Ciarletta, Feron & Ames, *"Towards a Framework for
Realizable Safety Critical Control through Active Set Invariance"*, ICCPS 2018.
Wabersich & Zeilinger, *"A predictive safety filter for learning-based control
of constrained nonlinear dynamical systems"*, Automatica 2021
([arXiv:1812.05506](https://arxiv.org/abs/1812.05506)).

---

## 2. Tube ASIF — certify against a *set* of models

`mpcc_tuning.filters.TubeASIF`

### The problem it solves

Every guarantee §1 makes is a statement about the model it predicts with. The
grip sweep in `rtrrl-playground` measures the cliff precisely:

| `assumed_grip` | episodes off-track | steps overridden |
|---|---|---|
| **0.6** (worst case) | **0%** | 40% |
| 1.0 (the *mean*) | 7% | 36% |
| 1.2 | 71% | 33% |
| 2.4 | **100%** | **17%** |

Two readings. **Assuming the mean is not good enough** — a guarantee that holds
in expectation is not a guarantee. And **a wrong filter intervenes *less***,
which is what makes an optimistic one dangerous rather than merely useless: it
does not announce itself by becoming annoying, it gets quieter.

### The condition

Ask for a *set* the true grip lies in and certify against all of it:

$$\forall g \in [g_\text{lo}, g_\text{hi}]:\quad
  x_k(g) \in \mathcal{X}\;\;\forall k, \qquad x_N(g) \in \mathcal{X}_\text{safe}$$

Grip enters monotonically — less grip means a tighter yaw-rate cap means a
wider swept path — so the worst case for staying on the track is
$g_\text{lo}$ and **one** rollout at the interval's lower end suffices.
`n_samples > 1` rolls out at several points instead, for models where that
monotonicity is not obvious. It is checked in the tests rather than assumed.

### References

Wabersich & Zeilinger, *"Linear model predictive safety certification for
learning-based control"*, CDC 2018.

---

## 3. Adaptive tube — learn the width instead of guessing it

`mpcc_tuning.filters.AdaptiveTubeASIF`

§2 is sound and **pessimistic for the entire run**: if the true grip is 1.3 and
the interval is $[0.6, 1.4]$, the car drives as if it were 0.6 forever.

Grip is observable — it enters through the yaw-rate cap, so whenever the cap
binds the achieved yaw rate reveals it:

$$\dot\psi_\text{obs} = \frac{a_{\text{lat},\max}\, g}{v}
  \quad\Longrightarrow\quad \hat g = \frac{v\,\dot\psi_\text{obs}}{a_{\text{lat},\max}}$$

A recursive mean and variance over those observations gives a lower confidence
bound $\hat g - \kappa\hat\sigma$, clipped to the prior interval. With no
evidence it sits at the prior's worst case; as evidence arrives the tube
narrows towards the truth.

### What this is not

It is **not a GP**. A Gaussian process over the model residual gives a
state-dependent posterior with calibrated uncertainty *everywhere*, including
states never visited. This gives one scalar with a frequentist-flavoured bound,
valid only where data has been collected. It is the right shape of idea at a
hundredth of the machinery, and it inherits the family's central caveat: the
guarantee is now only as good as the confidence bound, and a bound that is too
tight is exactly the optimistic filter from §2.

### References

Wabersich & Zeilinger, *"Probabilistic model predictive safety certification
for learning-based control"*, IEEE TAC 2021
([arXiv:1906.10417](https://arxiv.org/abs/1906.10417)).
Hewing, Kabzan & Zeilinger, *"Cautious Model Predictive Control using Gaussian
Process Regression"*, IEEE TCST 2020.
Berkenkamp, Turchetta, Schoellig & Krause, *"Safe Model-based Reinforcement
Learning with Stability Guarantees"*, NeurIPS 2017.

---

## 4. CBF QP — certify with one inequality

`mpcc_tuning.filters.CBFQP`

### From continuous to discrete time

In continuous time, for an extended class-$\mathcal{K}$ function $\gamma$:

$$\sup_{u \in \mathcal{U}} \dot h(x, u) \;\ge\; -\gamma\big(h(x)\big)$$

With the linear choice $\gamma(h) = \gamma h$, Grönwall gives
$h(t) \ge h(0)e^{-\gamma t}$, so $h$ never reaches zero. The discrete-time form
replaces the derivative with a difference:

$$h(x_{t+1}) \;\ge\; (1-\alpha)\,h(x_t),\qquad 0 < \alpha \le 1$$

and the same induction gives

$$h(x_t) \;\ge\; (1-\alpha)^t\,h(x_0) \;>\; 0 \quad\text{whenever } h(x_0) > 0$$

At $\alpha = 1$ the condition is just $h(x_{t+1}) \ge 0$, the weakest thing that
is still forward invariant; as $\alpha \to 0$ it approaches "$h$ may never
decrease".

**Be clear-eyed about what that buys.** The bound $(1-\alpha)^t h(x_0)$ tends to
zero, so the guarantee is that $h$ is never negative, not that it stays
comfortably positive. The car is permitted to converge on the boundary forever
— which is why §5 exists.

### The QP

The condition is one scalar inequality in $u$. Linearising $h(f(x,u))$ about
the proposal:

$$\min_u \|u - u_L\|^2
  \quad\text{s.t.}\quad
  L^\top(u - u_L) + h\big(f(x,u_L)\big) \ge (1-\alpha)h(x),
  \quad u \in \mathcal{U}$$

with $L = \partial h(f(x,u))/\partial u$ — two variables, one inequality, box
bounds. **This is the exact continuous argmin**, not a sampled approximation.
$L$ is taken by central differences rather than analytically, because the
dynamics contain a clip (the yaw-rate cap) and an analytic derivative would be
wrong exactly where it matters: at the limit.

**No horizon and no backup policy: one model step instead of thirty.**

### The barrier is the design choice, not the method

The obvious barrier is $h = w - |d|$ and it is **myopic** — it permits full
speed straight at a wall until the step before contact, because $h$ is still
positive and still falling slowly. It does not contain $v$ at all. The fix puts
the dynamics into the barrier:

$$h = w - |d| - T_\text{look}\,\big|v \sin e_\psi\big|$$

subtracting the lateral ground covered in $T_\text{look}$ seconds at the current
closing rate. Both are implemented (`h_kind="lateral"` / `"braking"`), because
*"CBFs are unsafe here"* and *"that barrier was unsafe here"* are very different
claims and only the second is ever true. Measured in `rtrrl-playground`: the
naive barrier put **47%** of episodes off the track; the same method with the
closing-rate term, **0%**.

### References

Ames, Xu, Grizzle & Tabuada, *"Control Barrier Function Based Quadratic
Programs for Safety Critical Systems"*, IEEE TAC 2017.
Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada, *"Control Barrier
Functions: Theory and Applications"*, ECC 2019
([arXiv:1903.11199](https://arxiv.org/abs/1903.11199)).
Agrawal & Sreenath, *"Discrete Control Barrier Functions for Safety-Critical
Control of Discrete Systems"*, RSS 2017.

---

## 5. CLF-CBF QP — safety hard, stability relaxed

`mpcc_tuning.filters.CLFCBFQP`

A CBF says what must not happen and nothing about whether the car does anything
useful. Add a control Lyapunov function $V(x) \ge 0$ that should decrease:

$$V(x_{t+1}) \;\le\; (1-\lambda)V(x_t) + \sigma, \qquad \sigma \ge 0$$

Here $V = d^2$, so "decrease $V$" means "return to the centreline".

The relaxation $\sigma$ is the whole design. **Safety is a hard constraint and
stability is a soft one**, because a problem with both hard is routinely
infeasible, and a filter that fails to return an input is worse than one that
gives up on progress for a step. Safety is never traded away.

### References

Ames et al., ECC 2019, §V (CLF-CBF-QPs).

---

## 6. Viability filter — decide offline, look up online

`mpcc_tuning.filters.ViabilityFilter`

Everything above decides safety online, by predicting. This decides it offline,
once, and then looks the answer up.

### The kernel

The **viability kernel** is the set of states from which *some* input sequence
keeps the system inside the constraints forever:

$$\mathrm{Viab}(\mathcal{X}) = \big\{x_0 : \exists\,u(\cdot),\;
  x_k \in \mathcal{X}\;\;\forall k \ge 0\big\}$$

computed by the fixed-point iteration

$$V_0 = \mathcal{X}, \qquad
  V_{i+1} = \big\{x \in V_i : \exists u \in \mathcal{U},\; f(x,u) \in V_i\big\}$$

which shrinks monotonically and therefore converges. The filter is one
membership test per candidate — **no rollout, no horizon, no backup policy** —
and the answer is exact rather than a sufficient condition.

### Why it is tractable here

The full state is $(x, y, \psi, v)$ and gridding that finely is expensive. But
the track is a corridor, and what matters for staying inside it is not *where*
the car is on the lap but where it is *across* it. In path-relative coordinates
the state collapses to

$$(d,\; e_\psi,\; v) \quad\text{— lateral offset, heading error, speed}$$

which is three dimensions and grids comfortably: the kernel builds in **under a
second** at $31\times31\times15$.

The price is that it is computed for a **single curvature**, so it is exact on a
constant-radius corner and conservative-or-wrong elsewhere. The default takes
the tightest curvature on the track, which makes it conservative everywhere —
the only defensible single-curvature choice.

This is a discrete dynamic-programming stand-in for a proper HJ solve: same
fixed point, no numerical Hamiltonian, accuracy set by the grid rather than by
a PDE scheme.

### References

Bansal, Chen, Herbert & Tomlin, *"Hamilton-Jacobi Reachability: A Brief
Overview and Recent Advances"*, CDC 2017.
Mitchell, Bayen & Tomlin, IEEE TAC 2005.
Aubin, *Viability Theory*, Birkhäuser 1991.

---

## 7. MPCC as its own safety filter

`mpcc_tuning.filters.MPCCSafetyFilter`

Every other filter bolts a second model, a second horizon and a second set of
constraints onto a controller that already has all three. An MPCC is *already* a
constrained trajectory optimisation over a prediction horizon. Add a terminal
constraint saying the trajectory can come to a stop inside the corridor, and
solving it **is** the certificate:

$$\min_{u_{0:N-1}} \|u_0 - u_L\|^2
  \quad\text{s.t.}\quad
  x_{k+1} = f(x_k,u_k),\;\;
  |e_c(x_k)| \le w,\;\;
  u_k \in \mathcal{U},\;\;
  v_N \le v_\text{stop}$$

Feasible $\Rightarrow$ a stop exists $\Rightarrow$ $u_0$ is safe. The objective
is minimum modification, so the returned input is the exact continuous argmin —
no linearisation, no sampling.

### The two tensions

**Soft constraints.** Production MPCC formulations soften the path constraints
with slacks so the QP always has a solution — correct for a *controller*,
because a slightly-infeasible plan beats nothing. But a filter's entire output
is the feasibility bit, and a problem that is always feasible carries no
information. The corridor and terminal constraint must be **hard**, with slack
left only on comfort constraints, and a failed solve is the signal rather than a
fault.

**Cost.** An extra NLP solve per tick, on top of the controller's. Measured
below, and it is the most expensive filter here by a wide margin. Whether that
trade is affordable depends entirely on the solver, which is why both are here.
On an acados controller the safety OCP is the same generated solver with one
extra terminal constraint and a different cost, and inherits its real-time
guarantees.

### References

Wabersich & Zeilinger, Automatica 2021 — this is the filter in its original
form.

---

## Not implemented, and why

**Shielding.** Alshiekh, Bloem, Ehlers, Könighofer, Niekum & Topcu, *"Safe
Reinforcement Learning via Shielding"*, AAAI 2018. Synthesises a filter from a
temporal-logic specification over a **discrete abstraction**. It is the right
tool when the specification is logical — orderings, liveness, "never two of
these at once" — and the state is naturally finite. Here the specification is
"stay between two lines", a geometric constraint the continuous methods express
directly and exactly, and building a *sound* discrete abstraction of a car
would introduce more conservatism than the constraint itself contains. Listed
because it is a real branch of the field, not because it was tried and rejected
on this problem.

**A full HJ reachability solve** — a level-set PDE via `helperOC` or
`optimized_dp`. §6 is the discrete stand-in; a real solve would give
grid-independent accuracy and a genuine value function, at the cost of a
dependency and a much longer offline stage.

## Measured, head to head

`python benchmarks/filters.py` — same track, same MPCC, same plant (grip 1.0),
400 steps. **good** = the default weights, which never crash; **bad** = the
weights the tuner collapsed to, which leave the track in 68 steps.

| filter | good: covered | good: overridden | bad: covered | bad: outcome | bad: overridden | µs/tick |
|---|---|---|---|---|---|---|
| *none* | 72.5 m | — | 7.6 m | **off-track** | — | — |
| **ASIF** | **72.5 m** | **0%** | 77.3 m | survived | 8% | 8 000 |
| **TubeASIF** | **72.5 m** | **0%** | 78.4 m | survived | 15% | 9 000 |
| **AdaptiveTube** | **72.5 m** | **0%** | 78.4 m | survived | 15% | 9 100 |
| CBFQP | 68.3 m | 8% | 78.9 m | survived | 28% | **2 500** |
| CLFCBFQP | 68.3 m | 8% | 78.9 m | survived | 28% | 17 600 |
| **ViabilityFilter** | **72.5 m** | **0%** | 76.9 m | survived | 8% | **600** |
| MPCCSafetyFilter | 73.9 m | 43% | **79.2 m** | survived | 99% | 57 000 |

### Every one of them saves the car

No filter let the collapsed controller off the track, and all of them turned
7.6 m into 77–79 m. On the headline question they are indistinguishable, which
is worth saying before any of the differences below are read as important.

### Only four are invisible

ASIF, the two tube variants and the viability filter override **0%** of a
controller that was never going to crash, and return its distance to the metre.
That is the property that separates a filter from a controller.

The **CBF overrides 8%** of the safe controller and costs it 4.2 m. This is not
a bug — it is the price of not having a horizon. A one-step condition cannot
tell that a *plan* exists, only that the next state is acceptable, so it refuses
inputs a rollout certifies. The `n_no_safe_action` column tells the same story:
27 states where no input satisfied the barrier at all, against 0 for the
rollout filters.

The **MPCC filter overrides 43%** of the safe controller and 99% of the bad one.
It achieves the best distance on the bad weights (79.2 m) by *being* the
controller, which is the outcome the docstring warns about, and it is not a
recommendation.

### The cheap one is the precomputed one

`ViabilityFilter` is **4× cheaper than the CBF and 13× cheaper than ASIF**, at
600 µs per tick, because online it is one array index. The cost moved offline:
the kernel takes ~1.5 s to build, and it is computed for a single curvature, so
it is conservative away from the tightest corner. For a fixed track that is an
excellent trade and it is the filter to reach for on hardware.

(These timings are unoptimised Python. The *ratios* are the meaningful part;
the absolute numbers would fall by one to two orders of magnitude in C++.)

### Getting them wrong is quiet

Three separate bugs during development each produced a filter that looked like
it was working, and every one showed up as a **lower** intervention rate: a
missing yaw-rate cap (45/45 inputs certified everywhere, 0% intervention), a
corridor wider than the plant's, and a `searchsorted` lookup that made the
viability kernel systematically permissive — that last one left the filter
overriding 4% and *failing to save the car at all*, which is exactly what a
working filter's numbers look like from a distance.

## Choosing between them

| if you… | use | because |
|---|---|---|
| have a credible backup manoeuvre | **ASIF** | cheapest sound option |
| do not trust your model | **TubeASIF** | the guarantee stops depending on one parameter |
| …and that parameter is observable | **AdaptiveTubeASIF** | the tube narrows instead of staying at the worst case |
| need a hard real-time bound and can write $h$ | **CBFQP** | one model step, no horizon, exact argmin |
| also need the car to *do* something | **CLFCBFQP** | a barrier alone permits sitting on the boundary |
| have a fixed track and offline time | **ViabilityFilter** | exact, and the online cost is an array index |
| already have a tuned MPC | **MPCCSafetyFilter** | no second model to maintain |
| do not trust your model **at all** | *none of them yet* | every guarantee here is a statement about $f$ |

The last row is not a rhetorical flourish. On this problem the difference
between the *criteria* is worth a few percent of intervention rate; the
difference between a correct and an optimistic grip estimate is worth **0%
against 71%** of episodes ending in a wall.
