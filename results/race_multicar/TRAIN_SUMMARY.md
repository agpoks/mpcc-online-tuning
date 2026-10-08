# Phase-A multi-car training (first full run)

ltc+mlp x 2 seeds x 18 ep x 4000 steps, --jobs 2 (spawn pool), real 270%+occlusion sensor, varied N in 1..3.

| net | last-8 clean % | last-8 avg passes | wall s |
|---|---|---|---|
| ltc seed 0 | 75 | 3.50 | 3822 |
| ltc seed 1 | 50 | 2.12 | 2812 |
| mlp seed 0 | 75 | 3.38 | 3853 |
| mlp seed 1 | 88 | 3.12 | 2776 |

Nets: results/race_multicar/nets/race_{ltc,mlp}_{0,1}.npz. Full log: train_full.log.
Honest read: MLP more consistent (75/88); LTC bimodal (seed1 collapsed to 50%, like the earlier single-opp LTC-equal bimodality). Passing improved a lot vs the untrained smoke (1 -> 2-3.5). OFF (off-track) persists late = the known MPCC-corridor issue in the multi-car setting.
NEXT: fixed eval of the keep-best nets vs controlled 1/2/3-car fields (the real flexibility + LTC-vs-MLP numbers) + artifacts.
