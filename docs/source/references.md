# References

## The framework this implements

* **Gros & Zanon**, *Data-driven Economic NMPC using Reinforcement Learning*,
  IEEE Transactions on Automatic Control, 2020 — the MPC-as-function-approximator
  formulation: the MPC supplies the policy, the value function and the
  action-value function at once, and RL tunes its parametrisation.
* **Zanon & Gros**, *Safe Reinforcement Learning Using Robust MPC*, IEEE TAC 2021.
* **Gros & Zanon**, *Reinforcement Learning for MPC: Fundamentals and Current
  Challenges*, IFAC World Congress 2023 — the survey to read first.

## Software doing the same thing, more completely

* [`mpcrl`](https://github.com/FilippoAiraldi/mpc-reinforcement-learning) —
  CasADi/`csnlp`, computes the MPC sensitivities for you. If you want this in
  production rather than in a spike, start here.
* [MPC4RL](https://arxiv.org/html/2501.15897) — the **acados**-based
  counterpart, which matters because acados is what makes the solve fast enough
  for a vehicle.

## Racing-specific, and closest to this

* **Nguyen, Nguyen, Amine, Vo-Duy, Mangharam, Nghiem**, *AD-MPCC: Adaptive
  Differentiable Model Predictive Contouring Control for Autonomous Racing*,
  2026 — [arXiv:2607.00141](https://arxiv.org/abs/2607.00141). Adapts MPCC
  weights per control tick by differentiating through the solver, evaluated in
  F1TENTH-Gym across road surfaces. The nearest neighbour to this repo.
* **COAT-MPC**, *Performance-driven Constrained Optimal Auto-Tuner for MPC*,
  2025 — [arXiv:2503.07127](https://arxiv.org/abs/2503.07127).
* *A Safe Reinforcement Learning driven Weights-varying Model Predictive
  Control*, 2024 — [arXiv:2402.02624](https://arxiv.org/abs/2402.02624);
  selects from a catalogue of Bayesian-optimised weight sets.

## The MPCC formulation itself

* **Liniger, Domahidi & Morari**, *Optimization-based autonomous racing of 1:43
  scale RC cars*, Optimal Control Applications and Methods, 2015.

## The RL machinery

* **Sutton**, *Learning to predict by the methods of temporal differences*,
  Machine Learning 1988 — TD(λ) and eligibility traces.
* **van Seijen, Mahmood, Pilarski, Machado & Sutton**, *True Online
  Temporal-Difference Learning*, JMLR 2016.

## Safety, for the open problem in {doc}`results`

* **Wabersich & Zeilinger**, *A predictive safety filter for learning-based
  control of constrained nonlinear dynamical systems*, Automatica 2021 —
  [arXiv:1812.05506](https://arxiv.org/abs/1812.05506). Implemented from
  scratch in
  [`rtrrl-playground`](https://github.com/agpoks/rtrrl-playground)'s
  `rtrrl_playground/safety.py`.
