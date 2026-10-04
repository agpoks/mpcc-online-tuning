# Phase-2 sensor gate — LTC vs MLP under partial observability (seed 0)

**Sensor model:** opponent observed only within **range 18 m AND a forward FOV cone ±60°** (`FOV_DEG`=120
total). A car outside the cone — e.g. directly behind, just overtaken — is unseen. (Range alone never
loses the opponent on this folded track: detected ~99–100%. The FOV is what creates the blind window.)

Both nets trained online under this gate (`make -f Makefile.race sensor-train`), then evaluated FROZEN
(deployed: no explore/learn) on the two cases the forward sensor raises.

| arm | case | result | passes | blind % | cross-track **blind** | cross-track **seen** |
|-----|------|--------|:---:|:---:|:---:|:---:|
| **LTC** | overtake-then-blind (slower ahead) | **CLEAN** | 2 | 69 | 0.022 | 0.041 |
| **LTC** | faster re-approach (faster behind) | **CLEAN** | 1 | 63 | 0.012 | 0.009 |
| MLP | overtake-then-blind | **OFF-TRACK** | 3 | 75 | 0.043 | 0.026 |
| MLP | faster re-approach | CLEAN | 1 | 65 | 0.031 | 0.024 |

**Finding — memory wins, with a mechanism.** With the opponent unseen 63–75 % of the time, the LTC stays
clean on both cases; the MLP runs off-track on overtake-then-blind. The cross-track activity blind-vs-seen
is the tell: the **LTC is calmer when blind** than when it sees (0.022 < 0.041) — it reacts only when it
sees and holds a steady line otherwise — while the **MLP is more erratic when blind** (0.043 > 0.026): with
only instantaneous observations it swerves at the car it can no longer see, and goes off. The LTC's
recurrent state carries the opponent across the blind window; the MLP cannot.

**Caveat:** 1 seed, two scenarios. Training metrics were noisier (MLP slightly better there: 25 % vs 38 %
contact). The clean separation is in the frozen deployed eval on these two deterministic cases — confirm
across more seeds / start-gaps before treating it as a paper claim.

## Artifacts (all in this folder, deterministic/seeded)
- **data** `sensor_<case>_<arm>_0.npz` (ego+opp trajectory, detected flag, gap, emitted weights, scalars)
- **csv** `sensor_eval.csv`
- **png+pdf** `fig_sensor_tracks.*` (ego path blue=sees / grey=blind, opp red), `fig_sensor_timeline.*` (gap + blind windows)
- **tikz** `tikz/fig_sensor_tracks.tex`, `tikz/fig_sensor_timeline.tex` (pgfplots groupplots, compile with tectonic; need `\usepgfplotslibrary{groupplots}`)
- **gif** `sensor_<case>_<arm>_0.gif` (×4)

## Reproduce
```
make -f Makefile.race sensor        # retrain both arms (range+FOV) + eval + all formats
make -f Makefile.race sensor-eval   # eval from the banked nets only
python3 experiments/race_sensor_eval.py --from-saved   # regenerate figs/csv/tikz/gif from the .npz (no acados)
```
