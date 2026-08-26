# Getting started

## Install

```bash
git clone https://github.com/agpoks/mpcc-online-tuning.git
cd mpcc-online-tuning
pip install -e .
```

CasADi and IPOPT come with the `casadi` wheel; there is nothing else to build.
No simulator is required — the plant is a plain kinematic bicycle in
`mpcc_tuning/model.py`, so the spike runs out of the box.

## The two things to run

**Check the premise.** The whole approach rests on one claim: that the gradient
the tuner needs is the partial derivative of the Lagrangian at the solution, and
therefore free. This example verifies it against finite differences, across
states and weight settings:

```bash
python examples/gradient_check.py
```

**Watch it tune.** The MPCC starts with deliberately bad weights — far too much
lag penalty, almost no reward for progress, so it crawls — and has to find
better ones from driving:

```bash
python examples/tune_online.py --episodes 30 --plot runs/tuning.png
python examples/tune_online.py --frozen          # the control: no tuning
```

Both have a notebook version (`examples/*.ipynb`), regenerated with
`python scripts/make_notebooks.py`.

## The knobs that matter

| flag | what it does |
|---|---|
| `--alpha` | step size on the log weights. `2e-4` is stable; `5e-3` reaches a good policy in two episodes and then destroys it |
| `--delta-clip` | clip on the TD error. Too small and the crash signal is crushed along with everything else |
| `--grip` | the plant's tyre limit. **The MPCC never models it** — that mismatch is the point |
| `--explore` | actuator noise, as a fraction of full scale. Q-learning needs the applied action to sometimes differ from the argmin, and it costs a second NLP solve when it fires |
| `--frozen` | do not tune. Always run this |

## What is deliberately wrong

The plant has a tyre grip limit — a cap on yaw rate at `A_LAT_MAX·grip/v` — and
the MPCC does not model it. A limit the controller does not know about shows up
as cost weights that are wrong for the real vehicle, and compensating for it is
exactly what an online tuner should be able to do. A shared model between plant
and controller would make the question unaskable.
