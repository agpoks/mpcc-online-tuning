# mpcc-online-tuning

**Tuning an MPCC's cost weights online, every control tick, from one scalar.**

The controller stays an MPCC — same solver, same constraints, same guarantees.
What changes is that its cost weights are treated as the parameters of a
reinforcement-learning policy and updated *while the car drives*, from a single
TD error per tick.

This is a **spike**: a small, deliberately throwaway prototype built to answer
one question — is this real-time feasible, and does the gradient it needs
actually exist in closed form? Both answers are yes, and both are checked here
rather than asserted. It is not a finished tool, and {doc}`results` is explicit
about where it breaks.

Companion to [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground),
where the same TD(λ) outer loop drives a recurrent network instead of a solver.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
formulation
influence_through_a_solver
plant
safety
filters
results
animations
references
```
