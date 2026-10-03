# 10-lap race evaluation — r_a-floor BAND (RA_FLOOR=8, classes equal/faster)

Canonical nets (`race_phase1.json`, seed 0) + the class-conditioned r_a floor wired into
`WeightPolicy`. Reproduce: `ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 tools/race_eval_laps.py --seed 0 --frozen`
(the band is applied automatically — `redrive` passes `ra_floor=RA_FLOOR`).

| opponent | laps | outcome | v_max | r_a (emitted) | note |
|---|---:|---|---:|---:|---|
| slower  | 12.98 | CLEAN | 2.38 | 10.00 | floor doesn't bind (already >8) |
| equal   | 12.54 | CLEAN | 2.51 |  9.79 | floor doesn't bind (already >8) |
| faster  | 11.84 | CLEAN | 2.39 |  8.00 | **floored 4.15 → 8.0 — fixes the wall-out** |

Baseline (no band), same eval: faster **3.65 laps OFF-TRACK** (over-drove to 3.13 m/s, ran wide
over the corridor edge); slower/equal already clean. So the band fixes faster with **zero regression**
on the other classes. Commit: `44c33f0`.
