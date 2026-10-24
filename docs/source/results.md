# Results

This is what the shipped method does, measured. The detailed per-policy table —
baseline, best constant, offline-fitted network, online-refined — is
{doc}`policies_in_the_table`; this page is the higher-level findings, in the order
the paper makes them, each with the honest caveat.

All numbers are on the acados controller and the dynamic drift plant
({doc}`plant`), from seeded scripts; the plant is deterministic, so a seed is a
draw of the exploration noise and scenario variation is physical.

## Safe online refinement improves pace, and stays clean

Starting from the offline base policy, the validated keep-best tuner refines the
weights during a qualifying session: lap time falls from **54.9 s to 44.3 s**,
the keep-best staircase monotone down, every slower or off-track attempt reverted
after a frozen validation lap. The refinement is localised — it concentrates in
the corners where the base policy left margin — while the settled line carries
speed through them.

Run continuously as the on-vehicle adaptation-and-safety layer on the deployment
map, the tuner drove **14.3 laps through 38 track-departure ticks, every one
recovered by a revert and none terminal.** The bounding is what makes this safe:
the grip band on $k_v$ cuts grip-over-claim departures from 100% to 12% of runs,
and the $r_a$ floor eliminates the over-driven wall-out ({doc}`safety`).

## One policy races a variable, occluded field

A single online-learned policy races one, two and three opponents of mixed pace
under the $270^\circ$ occluding sensor, **no retraining**. Both the recurrent and
memoryless arms solve one- and two-car races cleanly (zero contact, zero
departures); at three occluded cars the recurrent policy is 67% clean (75% at the
firmer sample) and the memoryless one 83%, with passes scaling as the field grows
(LTC $1.0/2.5/3.3$, MLP $1.2/2.5/3.7$).

> **The residual is the hard floor of three cars hidden most of the time.** Track
> departures are zero throughout; the three-car imperfection is occluded-opponent
> *contact* (33% of three-car runs for the recurrent policy, 17% for the
> memoryless one), which calls for better occluded-opponent estimation, not a
> different reward.

## Why recurrent: the ablation

Memory helps specifically where an **equal-pace** opponent must be tracked
through occlusion. The claim rests on **pass completion**, the quantity stable
from learning to deployment:

| | online-learning | deployed (frozen) |
|---|---|---|
| clean rate, LTC vs MLP | 0.91 vs 0.36 | 1.00 vs 0.89 |
| **pass completion, LTC vs MLP** | **0.82 vs 0.64** | **0.89 vs 0.67** |

The large clean-rate gap is mostly the adaptation transient — the memoryless arm
takes contact while it explores — and it closes once both are frozen. The
**pass-completion gap (~0.2) is stable across both regimes**: the recurrent net
keeps tracking the occluded car through the blind stretches and clears it, where
the memoryless net loses it more often (74% vs 60% of the window blind) and
stalls. On static, faster and slower traffic the two arms pass at the same rate,
so the advantage is equal-pace specific — a design choice reported with its
ablation, not the contribution ({doc}`behaviour_policy`).

## Real-time

The single acados solve holds the 20 Hz budget throughout, including with up to
three opponents. On the deployment map, over a 5-minute continuous drive, the
solve runs at **41 / 48 ms (p50 / p95)** in simulation. Recurrence is effectively
free: policy inference is **61 µs (LTC) against 29 µs (MLP)**, both negligible
beside the solve. Online tuning adds only the linear critic and the
score-function update — a handful of microseconds — and, because the
return-fitted critic draws its direction from the measured outcome rather than the
envelope gradient, it needs **no exact Hessian and no second solve.** The deployed
controller runs the same single solve whether it is learning or frozen. The
on-board solve timing is re-measured on the vehicle.

## The cautionary case, kept on purpose

The shelved `V=−J*` (MPCC-critic) policy is genuinely the *fastest* once the car
is moving — 2.79 laps from a flying start — but from a cold standstill it
over-claims grip, over-drives the first hairpin, and the QP goes infeasible. It
is kept in {doc}`policies_in_the_table` because it separates the **policy** (fast)
from the **deployment condition** (needs a warm-up lap), and because it is the
measured reason the project uses the return-fitted critic instead. The full
argument for why the envelope gradient collapses the opponent classes onto one
behaviour is in {doc}`tutorial/learning_algorithm` §5.

## Toward hardware

The method transfers to the SLAM-mapped figure-eight lab track (the deployment
map) when the corridor is built raceline-referenced — the representation the net
trains on — rather than by ray-casting the mapped walls. Driven frozen it laps
the lab map cleanly, and the keep-best tuner refines it there. The ROS 2
deployment node has been exercised end-to-end in software-in-the-loop (it
consumes pose and odometry, publishes the drive command and the live tuned
weights, holds 0% off-track). The on-vehicle trial on the `car21` platform — the
on-board solve budget, live online refinement, and a two-car laboratory
demonstration — is the immediate next step and is set up in the paper's
deployment section; the lab-map and keep-best figures are regenerated by
`results/deploy_labtest/paper_deploy_labmap.py`.
