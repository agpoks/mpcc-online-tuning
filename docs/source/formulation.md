# The controller

The object being tuned is a model predictive contouring controller. This page
states it: the geometry it is built on, the model it predicts with, the cost
whose weights are learned, and the constrained problem acados solves every tick.
The weights enter as an ordinary run-time parameter — the whole safe-learning
argument rests on that, so it is made precise here.

## Contouring, not tracking

```{image} _static/plots/mpcc_geometry.png
:alt: contouring error and lag error against the reference path
:width: 100%
```

MPCC does not track a *trajectory* — a path plus a schedule saying where to be
when. It tracks a **path**, and carries its own progress along it as an
optimisation variable $s$. The reference point $p(s)$ is therefore chosen by the
solver, not obtained by projecting the car onto the path, and the vector from it
to the car splits into two pieces:

$$e_c = \sin\phi\,\Delta x - \cos\phi\,\Delta y \qquad
  e_l = -\cos\phi\,\Delta x - \sin\phi\,\Delta y$$

with $\phi$ the path heading at $s$. The **contouring error** $e_c$ is *across*
the path and is what you want small. The **lag error** $e_l$ is *along* it, and
exists only because $s$ is free.

The lag term is not a refinement. Drop it and the solver finds the cheapest way
to collect the progress reward: run $s$ forward and leave the car behind.
Penalising $e_l$ is what keeps the progress variable honest, and it is why there
are two error weights to tune, not one.

## The prediction model

The controller predicts with a dynamic single-track model augmented by the
steering and progress states. The state $\vect{x}\in\R^7$ and input
$\vect{u}\in\R^3$ are

$$\vect{x}=[\,X\;Y\;\psi\;v_x\;v_y\;r\;s\,]^\top,\qquad
  \vect{u}=[\,a\;\delta\;v_s\,]^\top,$$

with pose $(X,Y,\psi)$, body-frame velocity $(v_x,v_y)$, yaw rate $r$, and
progress $s$ evolving as $\dot s=v_s$. The inputs are a commanded longitudinal
acceleration $a$, a steering angle $\delta$, and the progress rate $v_s$.
Lateral forces follow a combined-slip tyre model with sideslip
$\beta=\arctan(v_y/v_x)$. The model is discretised with an implicit integrator
to $\vect{x}_{k+1}=f_d(\vect{x}_k,\vect{u}_k)$.

This is **not** the plant. The evaluation plant is a different, higher-fidelity
drift model, so model mismatch is part of every test — see {doc}`plant`.

## The cost, and the weights that are learned

The reference is a periodic spline in $s$. With the grip-utilisation factor
$k_v\in(0,1]$ scaling how much lateral-acceleration budget the controller is
willing to claim, the stage cost over the horizon $k=1\dots N$ is

$$J=\sum_{k=1}^{N} q_c\,e_{c,k}^2 + q_l\,e_{l,k}^2 - q_v\,\dot s_k\,\Delta t
      + r_\delta\,\delta_k^2 + r_a\,a_k^2 + r_{\dot s}(\dot s_k - v_{\mathrm{ref},k})^2 .$$

The progress term $-q_v\dot s\,\Delta t$ rewards advancement along the
reference, the speed term couples $\dot s$ to a curvature-derived target
$v_{\mathrm{ref}}$, and the input terms penalise steering and drive effort. The
tunable cost weights are collected **in log space**,

$$\vect\theta=\log\big(q_c,\;q_l,\;q_v,\;r_\delta,\;r_a,\;r_{\dot s}\big),\qquad
  k_v\ \text{carried alongside},$$

so a multiplicative change is an additive step and positivity is automatic.
Together with the two constraint-side quantities — the opponent keep-out radius
$d_{\mathrm{obs}}$ and the boundary margin $d_{\mathrm{bnd}}$ — these make the
**nine** quantities the policy emits. The policy is what {doc}`behaviour_policy`
and {doc}`tutorial/learning_algorithm` describe; here they are just numbers the
OCP reads.

One line each: $q_c$ how expensive it is to leave the line, $q_l$ lag along the
path, $q_v$ how much progress is rewarded, $r_\delta/r_a/r_{\dot s}$ how much
steering, drive and progress-rate effort are damped, $d_{\mathrm{obs}}$ the
berth around an opponent, $k_v$ how much grip the plan claims,
$d_{\mathrm{bnd}}$ the margin kept to the wall.

## Constraints and the optimal control problem

Track boundaries bound the contouring error, and drive acceleration $a_x$ and
quasi-steady lateral demand $a_y=v_x r$ share a grip budget scaled by $k_v$:

$$|e_c|\le w - r_{\mathrm{car}},\qquad a_x^2 + a_y^2 \le (k_v\,a_{\mathrm{lat,max}})^2 .$$

Opponents enter as oriented keep-out ellipses (see {doc}`obstacles`). Because
the geometric, grip and speed demands can conflict, these rows are **softened**
with nonnegative slacks under a linear–quadratic penalty $P$, while actuator
bounds stay hard. Over stages $k=0\dots N-1$,

$$\begin{aligned}
\min_{\vect{x},\vect{u},\vect\epsilon}\;\;
  & \Phi(\vect{x}_N)+\sum_{k=0}^{N-1}\ell(\vect{x}_k,\vect{u}_k;\vect\theta,k_v)+P(\vect\epsilon)\\
\text{s.t.}\;\; & \vect{x}_0=\hat{\vect{x}},\quad \vect{x}_{k+1}=f_d(\vect{x}_k,\vect{u}_k),\\
  & \vect{u}_k\in\mathcal U,\quad h(\vect{x}_k,\vect{u}_k,\vect{p}_k)\le\vect\epsilon_k,\;\;\vect\epsilon_k\ge 0,
\end{aligned}$$

with $\hat{\vect{x}}$ the state estimate, $\mathcal U$ the hard actuator set,
$h$ the softened rows, and $\vect{p}_k$ the stage parameters carrying the
reference, the grip budget, and the opponent ellipses.

```{admonition} The one sentence the safety argument rests on
:class: important
**The cost weights $\vect\theta$ and $k_v$ enter only through $\ell$.** The
dynamics and the constraints are untouched by learning, so whatever weights the
policy emits, every predicted step still satisfies the vehicle's limits. A
learned weight can change the *behaviour* the controller optimises for; it
cannot relax a constraint.
```

The problem is generated as C code and solved with **one SQP real-time
iteration** in acados; on a failed solve the controller holds the last feasible
steering and brakes gently. The code lives in `mpcc_tuning/acados_mpcc.py` and
`mpcc_tuning/acados_ocp.py`; the solver settings and bounds are in the paper's
`tab:controller`. Real-time numbers are in {doc}`results`.
