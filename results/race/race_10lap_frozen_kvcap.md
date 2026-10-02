# 10-lap race, FROZEN net, k_v capped for slower (inference-time; baseline net)

Only slower changed (its k_v now capped at 0.62 instead of 0.90); static/equal/faster are the
baseline values (their k_v was already <=0.62, so the cap does not touch them).

| opponent | laps | overtakes | outcome | note |
|---|---:|---:|---|---|
| static | 12.53 | 13 | clean | unchanged |
| slower | 15.07 | 6 | clean | FIXED (was 1.6, off) |
| equal | 14.43 | 3 | clean | unchanged |
| faster | 3.65 | 1 | off-track | unchanged (over-drives chasing faster car) |
