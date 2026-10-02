# Seed-count choice: 5 vs 6 seeds (ltc)

Both are the first-N seeds of the deterministic canonical run (`race_phase1.json`); seeds 0..4 are bit-identical in both, so the only difference is the extra seed(s).

## Overall

| metric | 5-seed | 6-seed | 6-seed 95% CI |
|---|---:|---:|---:|
| passes | 1.97 | 2.00 | ±0.1 |
| clean % | 82.86 | 80.95 | ±10.6 |
| off-track % | 17.14 | 17.86 | ±10.1 |
| contact % | 0.00 | 1.19 | ±2.3 |

## By opponent class (passes / clean% / off%)

| class | 5-seed | 6-seed |
|---|---|---|
| static | 4.22 / 78 / 22 | 4.38 / 81 / 19 |
| slower | 1.67 / 83 / 17 | 1.73 / 77 / 18 |
| equal | 1.12 / 76 / 24 | 1.10 / 71 / 29 |
| faster | 0.76 / 94 / 6 | 0.75 / 95 / 5 |

## Per-seed (off-track %), to show spread / bimodality

| class | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| static | 0 | 0 | 33 | 25 | 50 | 0 |
| slower | 0 | 0 | 67 | 0 | 25 | 25 |
| equal | 0 | 50 | 0 | 0 | 67 | 50 |
| faster | 0 | 33 | 0 | 0 | 0 | 0 |
