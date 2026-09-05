# Frozen summary v1 — the paper's result table, archived and reproducible

**Best result (corrected 2026-09-05):** the **grid-fitted network with k_v
capped at 0.50** drives **2.48 / 2.49 / 2.76 laps frozen, all clean**, on ICRA
T2 from three physically different starts -- reproduced cold with
`drive_policy.py`, so it is real and rerunnable. Against the safe baseline
START (1.87 / 2.01 / 2.00) that is +0.53 laps at zero departures, and above the
best *constant* any search found (2.32 / 2.33 / 2.40). The gain over the
constant is the sector-dependent q_v/q_c schedule at the same grip claim.

> **The online-adaptation numbers were wrong.** They were reported as
> 2.72 / 2.78 / 2.99, but that frozen evaluation ran on an acados solver warmed
> by ten preceding learning episodes, and `sv.reset()` does not fully clear
> that state. Re-driven COLD (a fresh solver, `drive_policy.py`) the same
> networks do far worse -- seed 2 gives ~1.0-1.5 and leaves the track. The
> grid-fitted network reproduces cold exactly (2.48 / 2.49 / 2.76), which is
> how the contamination was localized: only the runs whose eval solver was
> warmed by online learning are affected. Online adaptation did NOT beat the
> supervised fit. See TODO 2k.

Git tag: `paper-frozen-summary-v1`. Figure: `frozen_summary.png` in this folder.

## The table (laps in 2500 steps, FROZEN: learning off, noise off; x = left the track)

| policy | seed 0 | seed 1 | seed 2 | geometry |
|---|---|---|---|---|
| START (constant, k_v 0.40) | 1.87 | 2.01 | 2.00 | smoothed |
| best constant by search (k_v 0.50) — context, not target | 2.32 | 2.33 | 2.40 | smoothed |
| grid-fitted network, uncapped | 1.31x | 0.10x | 1.81x | smoothed |
| grid-fitted network, k_v capped 0.50 | 2.48 | 2.49 | 2.76 | smoothed |
| online from START, MPCC critic, keep-best unvalidated | 2.11 | 2.06x | 1.83 | notched |
| online from START, MPCC critic, validated | 2.18 | 2.55 | 2.66 | notched |
| online from START, RETURN critic, validated | 1.76 | 2.42 | 2.30 | notched |
| **online from FITTED network, MPCC critic, validated** | **2.72** | **2.78** | **2.99** | smoothed |
| online from FITTED network, RETURN critic, validated | 2.34 | 2.30 | 2.57 | smoothed |

"notched" rows were evaluated before the corridor-smoothing fix of 2026-09-05;
the reproduction script regenerates them on the current geometry.

## Files

| file | what |
|---|---|
| `best_policy_icra_t2_raceline_{0,1,2}_progress_mpcc_val.npz` | **the three best networks** (G, cell params, box, banked frozen laps) |
| `best_policy_icra_t2_raceline_{0,1,2}_progress_return_val.npz` | the return-critic networks from the same start |
| `fitted_policy_icra_t2_raceline_kv0.50.npz` | the grid-fitted start network (k_v capped) — `--init-policy` input |
| `fitted_policy_icra_t2_raceline.npz` | the uncapped fit (crashes; kept as the negative) |
| `situation_demands_qc.json` | the situation grid on the smoothed corridor (fit target) |
| `online_t2_*.json` | per-episode laps, weights, validations, reverts, frozen evals per run |
| `online_t2_all.json` | all runs merged (input to the figure) |
| `frozen_summary_refs.json`, `fitted_refs.json` | the reference rows with provenance |
| `frozen_summary.png`, `fitted_policy_*_kv0.50.png`, `sectors_*.png` | figures |

## Re-drive a saved network (minutes)

    python3 scripts/drive_policy.py results/paper/frozen_summary_v1/best_policy_icra_t2_raceline_2_progress_mpcc_val.npz \
        --box-from results/paper/frozen_summary_v1/fitted_policy_icra_t2_raceline_kv0.50.npz

drives it frozen on the three standard starts and prints laps and clean/off.

**The v1 `best_policy_*.npz` files do not carry the box they were trained in or
the LTC seed** (files written after 2026-09-05 do). Without `--box-from` the
tool falls back to the factor-2 box, which is the wrong squash for these
networks -- measured: the 2.99 network then drove 1.31x / 2.89 / 0.81x. The
box is the fitted start network's (`fitted_policy_..._kv0.50.npz`, the same
for all six online-from-fitted networks); the seed is the number before
`_progress` in the filename and is inferred automatically.
Use this after ANY change to `mpcc_tuning/track.py`: a policy learned on one
boundary does not transfer to another (the 2.66 network of the notched
corridor crashed on the smoothed one).

## Regenerate the whole table (~6 h on 16 cores)

    bash scripts/reproduce_frozen_summary.sh            # all steps
    bash scripts/reproduce_frozen_summary.sh --from 7   # only the best-result run and onwards

The script lists every command with its runtime. Steps: sectors → grid →
fit (uncapped, capped) → controls → online from START (two critics) → online
from the fitted network (two critics) → merge → figure.

## Method, in one paragraph

Stable baseline (`baselines.START`) → the situation grid drives every
(q_v, k_v, q_c) vector in every (sector, entry-speed) cell for 400 steps and
keeps the winner → the weight policy (LTC cell + readout over [hidden state;
features; 1]) is fitted to those winners with the grip claim capped at the
level shown clean over full laps → online TD(λ) per metre of progress, critic
V = −J*, adaptation inside the data-driven box, starts from that network →
after each episode a candidate network is validated frozen and banked only if
it beats the validated incumbent; reverts return to the incumbent → the banked
network is the deliverable, driven frozen.
