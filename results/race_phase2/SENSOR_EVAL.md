# Phase-2 sensor gate — the online method handles occlusion (the fix was r_a, not the sensor)

## Sensor model (the real one, per the hardware)
The ego's perception is **not** 360°: a **270° lidar** (rear ~90° blind behind the spoiler) + a **~120°
camera** inside that arc, both ~18 m, and crucially the **car-height tube walls occlude line-of-sight** —
the opponent is hidden around bends. On this twisty 73.6 m track occlusion dominates: the straight line
ego→opponent crosses a wall most of the time, so the opponent is unseen **~80–90%** of the time
(`_visible(..., occlude=True)` in experiments/race_mode.py: range + 270° coverage + wall line-of-sight).

## What actually broke, and what didn't
Earlier "blind collisions / off-tracks" were two artifacts, **not** a partial-observability failure:
1. **I evaluated the net FIXED** (`WeightPolicy.step`, no learning) instead of running the **online method**
   (`PolicyTuner`: explore + RTRL weight update every tick, keep-best). The method is *online adaptation
   during the race*; a fixed net can't adapt to the blindness, so it looked broken.
2. **`r_a` over-drive on the SLOWER class.** Chasing a slower car the tuner wound up to ~2.9 m/s past the
   ~2.5 m/s geometric corridor ceiling and ran wide — the *same* over-drive as the faster-10-lap wall-out,
   but the `r_a`-floor band only covered equal/faster (classes 2,3), never slower (1). (It was off-track
   while *seeing* the car — definitively not blindness.)

## Result — online method, real sensor (270° + occlusion), seeds 0–1
Fix = two constants: `RA_FLOOR_CLASSES` 2,3 → **1,2,3** (cover slower) and `RA_FLOOR` 8 → **10** (damp the
over-drive). Online eval (`race_sensor_eval.py --online`), obstacle simply dropped when blind, **no
dead-reckon, no blind-caution**:

| case | seed 0 | seed 1 |
|------|--------|--------|
| overtake-then-blind (slower ahead) | CLEAN (2 passes, v_max 2.22) | CLEAN |
| faster re-approach (faster behind) | CLEAN | CLEAN |

**4/4 clean, 0 contact.** The online learner + the LTC memory handle the ~80–90% occlusion blindness on
their own; it still overtakes. The `r_a`=10 floor costs a little straight-line aggression (gentler accel)
but keeps the car under the corridor ceiling.

## Explored and dropped (negative result, kept for the record)
- **Dead-reckon** (predict the occluded opponent as an MPCC keep-out) — unneeded. It only mattered because
  the *fixed* eval couldn't adapt; worse, a growing-radius keep-out made the car swerve *harder* the longer
  it was blind and run off the edge. Not used.
- **Blind-caution** (hold-line + slow when blind, hand-coded override) — a crude patch for the same
  self-inflicted swerve. Not used. The online method needs neither.

## Reproduce (deterministic/seeded)
```
# code has RA_FLOOR=10, RA_FLOOR_CLASSES=(1,2,3), FOV_DEG=270
python3 experiments/race_sensor_eval.py --online --arms ltc --seeds 0 1 --fov-deg 270 --occlude
```
Data: `sensor_<case>_ltc_<seed>_ra10conf.npz` (ego+opp trajectory, detected flag, gap, emitted weights).
