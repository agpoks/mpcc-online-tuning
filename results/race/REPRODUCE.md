# Reproducing the race-mode results and paper figures

Everything here regenerates from `experiments/race_mode.py` (training) and the `tools/paper_race_*`
scripts (figures), driven by `Makefile.race` at the repo root.

```bash
make -f Makefile.race print-config     # show env + the canonical policy constants
make -f Makefile.race all              # train -> analysis -> figures -> tikz -> gifs (~1 h)
make -f Makefile.race figures-all      # just re-make figures+tikz+gifs from the banked nets (fast)
```

## Pipeline (what produces what)

| stage | command | inputs | outputs |
|---|---|---|---|
| train | `race_mode.py --arms ltc mlp --seeds 3 --episodes 14 --steps 4000 --fair-opp --dump-traj` | policy constants in `race_mode.py` | `nets/race_<arm>_<seed>.npz`, `race_phase1.json`, `traj/traj_<arm>_<seed>.npz` |
| analysis | `race_analysis.py`, `race_table.py` | the JSON | `race_*_*.csv`, `race_phase1_table.md`, learning figures |
| figures | `paper_race_learning.py --seed 0` | banked nets | `paper/fig_*.{pdf,png}`, `paper/states_online_ltc_0_*.npz` |
| tikz | `paper_race_tikz.py --seed 0` | the state caches | `paper/tikz/*.tex` (+ `.dat`) |
| gifs | `paper_race_gifs.py --seed 0` | the state caches | `gif/ltc_<kind>_race_s0.gif` |

## Determinism

The STD plant is deterministic and every RNG is seeded, so `(code, params, seed)` regenerates
identical nets / JSON / figures / GIFs. **Seeds vary the scenario physically** (start `s0`,`v0` and
opponent placement), not just an RNG draw. `ego_pace` is **measured** (a solo START drive), not
hardcoded, so opponent speeds scale reproducibly.

## The policy — single source of truth

All policy constants live in `experiments/race_mode.py` (not duplicated in the Makefile):
`KV_FLOOR`, `KV_CEIL`, `KV_FLOOR_CLASSES`, `A_LAT_RACE`, `FAIR_PACE`, the two-layer corridor
(`CORRIDOR_KW`), the tuner config. Change one there and re-run `make -f Makefile.race all`.

## Frozen vs online figures

This is an ONLINE-RL method: `paper_race_learning.py` defaults to the **online** policy
(`get_states(online=True)` → `redrive_online`: the `PolicyTuner` runs explore+learn each tick from
the banked net, seeded). That is the method as deployed and it overtakes all three opponent classes.
The **frozen** replay (`online=False`) is available as a deterministic lower bound.

## Provenance / consistency note (READ THIS)

The historical iteration runs `race_phase1_reward2..5.json` were produced at **`KV_FLOOR=0.50`**
(reward5 = commit `bfcef13`). The floor was then raised to **`0.56`** (commit `64f80d7`) because
0.50 gave the ego ~1.6 m/s — below the equal opponent — so it could not overtake a matched car.
**The canonical run at the current code is `race_phase1.json` at `KV_FLOOR=0.56`**, produced by
`make -f Makefile.race train`. Re-run it to refresh the headline numbers so JSON, nets and figures
all sit at one floor. (`reward2..5.json` are kept only as history.)
