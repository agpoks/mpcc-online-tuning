# Phase A — multi-car online MPCC tuning (branch `race-multicar`)

**Goal.** One online-tuned MPCC (PolicyTuner: explore + RTRL + keep-best) that races a **variable** number of
opponents (1/2/3) of mixed pace under the **real 270° + occlusion sensor**, overtaking the beatable ones — and
does it with **one net, any field size, no retraining**. LTC (memory) vs MLP (memoryless) weight-policy.
acados C-codegen + dynamic drift model + STD plant, `Track.icra_t2_smooth`. Solve ~37–39 ms (fits 20 Hz).

## Headline — deploy configuration (the two best nets)

Fixed eval (explore off), 6 runs/cell (3 physical fields × 2 seeds). Deploy config = **LTC with the q_c floor**
(LTC-specific) + **MLP without it** (gentler keep-out). Figure: `multi_flexibility.{png,pdf}` + `tikz/`.

| net | N=1 | N=2 | N=3 | contact (1/2/3) | off-track |
|---|---|---|---|---|---|
| **LTC** q_c-floor **+ crowd band** | 100 % | 100 % | 67 %* | 0 / 0 / 33 % | 0 everywhere |
| **MLP** gentler | 100 % | 100 % | 83 % | 0 / 0 / 17 % | 0 everywhere |

*LTC N=3 is **75 % at the firmer n=12** (the n=6 cell is coarse). Passes scale with the field (LTC 1.0/2.5/3.3,
MLP 1.2/2.5/3.7). **Both arms solve 1–2 car races (100 % clean, 0 contact, 0 off).** The crowd band lifted
LTC N=3 from 58 % → 67–75 % (see step 5) by opening the line when the field is REMEMBERED; MLP still edges
N=3 (83 %). Both are deploy-viable; MLP is the slightly stronger single net, LTC is the memory arm that now
races the crowd nearly as cleanly.

## The development arc (pre-fix → gentler → q_c-floor)

Figure: `multi_contact_threeway.{png,pdf}` + `tikz/multi_contact_threeway_clean.tex`.

1. **Count-flexibility** (`multi_flexibility`): the count-invariant aggregate obs (fastest-of-field +
   crowding/avg-closing/count) lets ONE net race 1/2/3 cars. Holds.
2. **Occluded contact → bounded predicted keep-out** (`multi_contactfix`): the MPCC was planning *through*
   opponents it could no longer see. Fix = keep a keep-out at a blind opponent's predicted raceline position
   (last-seen s propagated at its tracked speed). HARD (r0.36/1.5 s) cut contact but ran LTC off-track;
   **GENTLER (r0.28/1.0 s) eliminated off-track (0 % everywhere)**.
3. **LTC line collapse → q_c floor** (`ltc_investigate/ltc_trace`, three-way): the gentler retrain collapsed
   LTC's q_c to 0.22 (crash regime) → loose line → wander → contact (CONTACT @0.13 laps vs pre-fix CLEAN 3.15).
   Fix = a class-conditioned **q_c floor (0.5)**, same mechanism as the k_v/r_a bands. q_c settles at a healthy
   **~0.75** (wider-but-safe: not glued/3.74, not collapsed/0.22). **Recovers LTC N=1,2 → 100 %. LTC-specific —
   it breaks MLP (MLP never collapsed q_c), so it is applied to LTC only.**
4. **N=3 crowding diagnosed** (`ltc_n3/`): LTC N=3 (58 %) < MLP (83 %). The q_c floor does NOT bind at N=3
   (the net holds q_c ~0.85, 0 % at the floor) — it keeps a *N=1-tight* line through a 3-car crowd and clips an
   occluded car, because q_c was conditioned on the rival's PACE, not the CROWD. The *looser* gentler net was
   better at N=3 (67 %). So: a conditioning gap, not the floor, not a memory failure.
5. **Crowd-aware q_c** (`ltc_n3/ltc_n3_crowd_test`): make q_c **crowd-aware via memory** — track the number of
   DISTINCT opponents seen this race (persists through occlusion, where LTC memory pays off) and, once a full
   3-car field is REMEMBERED, open the line to a WEAVING band (q_c → 0.30–0.45, overriding the floor); 1–2 cars
   keep the tight line. A **deploy-time rule on the stable floor-net (no retrain needed)**: it lifts LTC N=3
   from 58 % → **67 % (n=6) / 75 % (n=12)**, contact halved, N=1,2 held at 100 %, and the band fires 92 % of an
   N=3 run under occlusion. (Retraining *with* the band only added bimodal instability — the eval-time rule is
   the keeper; `qc_crowd_lo/hi/n` in race_mode.py, LTC-only like the floor.)

## Paper findings
- **One online net generalises across opponent count** (1/2/3, no retrain) under a realistic occluding sensor.
- **Memory vs constraint:** pre-fix, LTC won (its memory implicitly handled occlusion); an explicit predicted
  keep-out (constraint-side) then let the simpler MLP match it — the constraint commoditises what the memory
  did. **But the crowd case re-separates them:** the fix that recovers LTC's 3-car racing is explicitly
  **memory-driven** (remember the crowd through occlusion → open the line), which a memoryless policy cannot do.
- **Deterministic eval flips training-noise reads** — always evaluate the keep-best net explore-off; and some
  fixes (the crowd band) are best as **deploy-time rules**, not retrains (retraining destabilised N=2).
- **The line-collapse failure and its deterministic-floor fix** (q_c floor), weight/policy-specific (helps LTC,
  harms MLP); and the **conditioning axis matters** — q_c keyed on pace vs crowd is the N=3 story.

## Artifacts (all reproducible)
- **Data:** `multi_eval_rows_{prefix,gentler,qcfloor,fix,deploy}.npz` (per-stage scalars) · `multi_eval.csv` /
  `multi_eval_summary.csv` · `ltc_n3/ltc_n3_rows.npz` · `ltc_investigate/{ablation,trace_*}.npz`.
- **Figures (png+pdf):** `multi_flexibility` (LTC-vs-MLP deploy) · `multi_contact_threeway` (arc) ·
  `multi_contactfix` (keep-out before/after) · `ltc_investigate/ltc_trace` (q_c collapse) ·
  `ltc_n3/ltc_n3_contact` (N=3 crowd clip).
- **TikZ (compile via tectonic):** `tikz/multi_flexibility_{clean,passes}.tex` · `tikz/multi_contact_threeway_clean.tex`.
- **GIFs:** `multi_race_{ltc,mlp}.gif` (a 3-car race per arm) · `ltc_investigate/ltc_trace.gif` (collapse).
- **Nets:** `nets/race_ltc_{0,1}.npz` (q_c-floor) + `nets/race_mlp_{0,1}.npz` (gentler).
- **Docs:** this file · `EVAL.md` (full arc tables) · `TRAIN_SUMMARY.md` · `PHASE_A.md`.

## Reproduce (deterministic/seeded)
    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_multi_train.py --arms ltc mlp --seeds 2 --jobs 2   # train (both arms)
    python3 experiments/race_multi_eval.py --jobs 4 --steps 2600         # deploy eval (LTC+floor vs MLP-no-floor) + flexibility fig/tikz/gif
    python3 experiments/race_multi_eval.py --from-saved                  # plots/csv/tikz/gif from saved data only
    python3 tools/multi_contact_threeway.py                              # arc three-way (reads the stage snapshots) + tikz
    python3 tools/ltc_investigate.py --jobs 4                            # q_c-collapse ablation + trace + gif
    python3 tools/ltc_n3_chase.py --jobs 4   (or --from-saved)           # N=3 crowding firmer stats + contact trace
Bands/knobs in `experiments/race_mode.py`: KV/RA floors, `QC_FLOOR=0.5` classes (1,2,3), `PRED_R=0.28`,
`BLIND_CAP=1.0`, `MAX_OPP=3`. q_c floor is LTC-only (`race_multi_eval.one` sets `qc_floor_on=(arm=="ltc")`).
