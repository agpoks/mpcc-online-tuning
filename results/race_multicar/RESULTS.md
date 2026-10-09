# Phase A — multi-car online MPCC tuning (branch `race-multicar`)

**Goal.** One online-tuned MPCC (PolicyTuner: explore + RTRL + keep-best) that races a **variable** number of
opponents (1/2/3) of mixed pace under the **real 270° + occlusion sensor**, overtaking the beatable ones — and
does it with **one net, any field size, no retraining**. LTC (memory) vs MLP (memoryless) weight-policy.
acados C-codegen + dynamic drift model + STD plant, `Track.icra_t2_smooth`. Solve ~37–39 ms (fits 20 Hz).

## Headline — deploy configuration (the two best nets)

Fixed eval (explore off), 6 runs/cell (3 physical fields × 2 seeds). Deploy config = **LTC with the q_c floor**
(LTC-specific) + **MLP without it** (gentler keep-out). Figure: `multi_flexibility.{png,pdf}` + `tikz/`.

| net | N=1 | N=2 | N=3 | mean clean | contact (1/2/3) | off-track |
|---|---|---|---|---|---|---|
| **LTC** q_c-floor | 100 % | 100 % | 50 %* | 83 % | 0 / 0 / 50 % | 0 everywhere |
| **MLP** gentler | 100 % | 100 % | 83 % | 94 % | 0 / 0 / 17 % | 0 everywhere |

*LTC N=3 is 58 % at n=12 (the n=6 cell is coarse; see the N=3 chase). Passes scale with the field
(LTC 1.0/2.5/2.8, MLP 1.2/2.5/3.7). **Both arms solve 1–2 car races (100 % clean, 0 contact, 0 off).** MLP
handles the 3-car crowd better; LTC is right behind and the safer line-holder. **MLP-gentler is the deploy net;
LTC-q_c-floor is the memory arm.**

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
4. **N=3 crowding** (`ltc_n3/ltc_n3_contact`): the remaining LTC N=3 contact (33 % at n=12) is NOT a q_c
   collapse (q_c healthy 0.80, line controlled) — it is a close-quarters crowding clip of an *occluded* car
   (PRED_R 0.28 + prediction error slips inside the 0.24 contact radius). Inherent hard case; MLP clears it
   better. Candidate tune (not done): PRED_R 0.28→0.32 for more crowd clearance (risks a little off-track).

## Paper findings
- **One online net generalises across opponent count** (1/2/3, no retrain) under a realistic occluding sensor.
- **Memory vs constraint:** pre-fix, LTC won (its memory implicitly handled occlusion); adding an explicit
  predicted keep-out (constraint-side occlusion handling) let the **simpler MLP match/beat it** — the constraint
  commoditises what the memory did. A clean result either way.
- **Deterministic eval flips training-noise reads** — always evaluate the keep-best net explore-off.
- **The line-collapse failure and its deterministic-floor fix** (q_c floor), consistent with the proven
  k_v/r_a bands; and that such a floor is weight- and policy-specific (helps LTC, harms MLP).

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
