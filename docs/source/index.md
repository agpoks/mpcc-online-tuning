# mpcc-online-tuning

**Safe online learning of an MPCC's cost weights — refining a stable base
controller *while the car drives*, without ever letting it construct an unsafe
one.**

A model predictive contouring controller (MPCC) drives the car by solving a
constrained optimal-control problem every 50 ms. What it *does* inside those
constraints — how hard it corners, how much grip it claims, how much room it
gives an opponent — is set by a handful of cost weights. This project treats
those weights as the output of a small recurrent policy and **learns them
online**: a base policy is fitted offline, then kept improving on the real
vehicle, surface, and opponents it actually meets. That online refinement is a
direct route to closing the sim-to-real gap.

The hard part is not finding better weights but **moving them safely**. On a car
that must finish the lap, an unconstrained tuner walks a weight to a bound,
saturates, and silently claims more grip or aggression than the tyres have — and
leaves the track. The method here makes that impossible by construction:

- the learned weights are a **run-time parameter of the OCP**, so its hard
  constraints (track, grip, actuators) hold whatever the policy emits;
- the policy is **anchored** at a hand-verified baseline and squashed into a
  **box measured** from where each weight stops helping;
- the weights that carry safety are clamped by **class-conditioned bands**;
- and an update is accepted only through a **validated keep-best** rule — banked
  after a frozen, exploration-free lap confirms it, reverted on any regression.

The learning signal is the **measured racing return**, read by a small
return-fitted critic and a score-function actor, carried online by real-time
recurrent learning (no backpropagation through time). This is the method in the
accompanying RA-L paper, and it runs the same single acados solve whether it is
learning or frozen.

```{admonition} New here? Start with the tutorial.
:class: tip
{doc}`tutorial/learning_algorithm` builds the whole learning rule from the
ground up on toy problems you can run in seconds — what is learned, why the
naive critic walks to a corner, and what replaces it. Then {doc}`formulation`
and {doc}`safety` give the controller and the safe-learning construction.
```

```{toctree}
:maxdepth: 2
:caption: The method

getting_started
tutorial/learning_algorithm
formulation
behaviour_policy
safety
obstacles
plant
code_map
policies_in_the_table
results
```

```{toctree}
:maxdepth: 1
:caption: How we got here — the exploration log

history
spike_formulation
spike_results
safety_filter_for_the_tuner
filters
influence_through_a_solver
```

```{toctree}
:maxdepth: 1
:caption: Reference

animations
references
```

---

This repository began as a **spike** — a throwaway prototype that asked whether
an MPC's weight gradient is available in closed form (the envelope theorem) and
whether the loop is real-time. Both answers were yes, but the method that
shipped is not the spike: the envelope/`V=−J*` critic was *shelved* (it points
the same way in every situation and over-claims grip), the predictive safety
filter was *replaced* by the box-and-bands-and-keep-best construction above, and
the plant became a dynamic single-track drift model on acados. The
{doc}`history` section keeps that path honestly, including the measured dead ends
— they are why the current design looks the way it does.
