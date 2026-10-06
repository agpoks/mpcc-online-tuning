# Paper notes — LTC vs MLP (online MPCC weight-tuning), + the phase-2 sensor result

Working notes for the paper's LTC-vs-MLP comparison. All numbers/figures are reproducible (seeded,
deterministic plant); figure paths are relative to the repo root. Honest framing is flagged — do NOT
over-claim beyond what the data shows.

## TL;DR suggested narrative (defensible)
1. **Memory is essentially free in compute.** So the recurrent policy carries no real-time penalty.
2. **Under full visibility, memory gives a modest, class-dependent overtaking edge** by integrating the
   *temporal* observations (slip, closing-rate, opponent-speed estimate, per-sector history) that the MLP
   sees only as an instantaneous snapshot.
3. **Under partial observability (realistic 270° + occluding walls, ~80–90% blind), LTC and MLP are
   equally clean**, but the memory yields a **more conservative, line-holding policy** — a behavioural
   difference, not an outcome gap.
4. **The occluded opponent is predictable because it lives on a known racing manifold**, which is *why*
   memory is not decisive for this track's occlusion.

## 1. Cost (both arms, 31-feature obs, n_hidden = 12)
| | LTC | MLP |
|---|---|---|
| parameters (G + cell + class head) | **984** | **816** (+21% for the liquid/recurrent state) |
| inference / control tick | **61 µs** | **29 µs** |
| — vs the 34 ms acados QP solve | <0.2% | <0.1% |
| training | ~30–60 min/net, **solve-dominated** (~72k solves/net); comparable across arms |

Takeaway for the paper: the QP solve dominates the tick by ~500–1000×, so **recurrence adds no meaningful
real-time cost** (RTRL eligibility-trace overhead ≈ the 32 µs inference gap).
Figures/data: measured inline (see `tools/paper_ltc_vs_mlp_compare.py`); solve timing `results/race/solve_timing.md`.

## 2. Full-visibility behaviour (race-mode, pre-sensor)
Current committed by-kind pass rates (`results/race/race_by_kind.csv`):
| opponent | LTC passes | MLP passes | contact |
|---|---|---|---|
| static | 4.38 | 4.24 | both 0 |
| slower | 1.73 | 1.46 | LTC 0.045 / MLP 0 |
| equal  | 1.10 | 0.90 | both 0 |
| faster | 0.75 | **0.95** | both 0 |

**Honest:** a *modest, class-dependent* edge — LTC passes more on static/slower/equal, MLP edges faster,
MLP marginally cleaner on slower. NOT a blowout. The stronger "LTC wins" result is the **enriched-
observation ablation**: with an instantaneous obs LTC ≈ MLP; adding the temporal features (20→29 feat)
flipped it to an LTC win (equal pass-rate 1.0 vs 0.64, 0% vs 5% contact). Cite the ablation as the
*mechanism* (memory integrates temporal cues), not the headline pass number.
Figures: `results/race/race_weights_by_kind_{ltc,mlp}.png`, `results/race/race_behaviour.png`,
`results/race/paper/fig_learn_by_opponent.*`.

## 3. Online weight adaptation (the method's point)
The tuner adapts the MPCC weights online, per situation, during the race.
Figures: `results/race/paper/fig_weight_learning_map.*` (where on the track it adapts most),
`results/race/paper/fig_weight_settling_{curve,equal,faster,slower}.*` (adaptation converging over laps),
`results/race/race_weight_curve_{ltc,mlp}.*`.

## 4. Partial observability — the realistic sensor (phase-2)
**Sensor model (hardware-grounded):** 270° lidar (rear ~90° blind — spoiler) + ~120° camera inside it +
**car-height tube-wall occlusion** (line-of-sight blocked around bends). On the twisty 73.6 m track this
makes the opponent **unseen ~80–90%** of the time. (`_visible(occlude=True)` in `experiments/race_mode.py`.)

**Result (online method, both arms, seeds 0–1):** LTC **4/4 clean**, MLP **4/4 clean**, 0 contact — **tied**.
So the occlusion is handled by the online adaptation + the opponent-on-a-known-line, NOT by memory.
Figure: `results/race_phase2/fig_sensor_compare.*`, data `results/race_phase2/sensor_eval_summary_ra10conf.csv`.

**Same outcome, different policy (the interesting bit):** `results/race_phase2/fig_ltc_vs_mlp_overtake_blind_0.*`
| signal (overtake) | LTC | MLP |
|---|---|---|
| q_c (line-keeping) | **1.78** | 0.87 (+104% for LTC) |
| speed V | **1.77** | 2.01 (LTC slower) |
| q_v (aggression) | 0.27 | 0.32 |
| k_v, r_a | 0.63, 9.45 | identical (band-clamped) |

⇒ the LTC adopts a **conservative hold-the-line** strategy under occlusion (higher q_c, slower); the MLP
is faster/looser (and so happened to *see* the opponent more, 27% vs 13%). On the faster case they differ
less (LTC q_v +22%). k_v and r_a are pinned by the class-conditioned band, so the differentiation lives in
q_c/q_v/speed.

**Two prerequisites for this result (state them):** (a) the deployed method is **online RTRL + keep-best**,
not a frozen banked net — evaluating frozen hid the adaptation; (b) a **class-conditioned r_a-floor band**
(`RA_FLOOR=10`, classes slower/equal/faster) prevents the over-drive that otherwise runs the car wide
chasing a slower car (it was off-track *while seeing* the opponent — an over-drive, not a blindness
failure). Gifs: `results/race_phase2/sensor_{overtake_blind,faster_reapproach}_ltc_0_ra10conf.gif`.

## 5. Opponent prediction under occlusion
A "drives-like-us" predictor (same car ⇒ grip-limited speed profile on the racing line, scaled by the
observed pace, EMA) tracks the occluded opponent:
- **slower (overtake):** predicted-vs-true error mean **1.24 m**, max 3.06 m through 80% blind — good.
- **faster (re-approach):** mean **5.0 m**, max 14.4 m — poor (a maneuvering faster car is not
  drives-like-us). This is why the deployed controller **drops the obstacle when blind** rather than trust
  a 5–14 m-off guess.
Figures: `results/race_phase2/fig_opp_prediction_{overtake_blind,faster_reapproach}_ltc_0.*` + gifs.
Paper point: partial observability is tractable here because the opponent lies on a known 1-D manifold.

## Caveats / do-NOT-claim
- Do **not** claim LTC beats MLP under partial observability — it's tied (4/4 each).
- Do **not** cite the "1.0 vs 0.64" as the current pass rate — it's the enriched-obs ablation run; the
  current by-kind edge is modest (table in §2).
- The phase-2 numbers are 2 seeds (seeds 0–1) under the corrected model; for a hard claim, widen seeds.
- Warm-start caveat: the online runs start from banked nets of slightly different training history.

## Reproduce
```
# phase-2 sensor result (online, both arms):
make -f Makefile.race sensor          # retrain both arms (270°+occlusion, r_a-band) + online eval
python3 experiments/race_sensor_eval.py --online --arms ltc mlp --seeds 0 1 --fov-deg 270 --occlude
python3 experiments/race_sensor_eval.py --from-saved --tag _ra10conf --arms ltc mlp --seeds 0 1   # figs/tikz/gifs from data
python3 tools/paper_ltc_vs_mlp_compare.py --case overtake_blind --seed 0 --tag _ra10conf            # weights/states compare
python3 tools/paper_opp_prediction.py    --case overtake_blind --seed 0 --tag _ra10conf            # prediction plot+gif
```
Full phase-2 arc: `results/race_phase2/SENSOR_EVAL.md`.
