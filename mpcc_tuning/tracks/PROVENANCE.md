# Track data, and where it came from

Copied in rather than depended on, per the repo's standing rule.

## `Spielberg_centerline.csv`

The Red Bull Ring, Spielberg, at the 1:10 scaling used by the **F1TENTH**
community — a 343 m lap over a 100 x 63 m footprint with a constant 1.1 m
half-width. This is the standard `f1tenth_racetracks` centreline format:

    # x_m, y_m, w_tr_right_m, w_tr_left_m

It is public data, published with the F1TENTH gym and used as a benchmark
across the autonomous-racing literature. Taken here from a local checkout of
`f1tenth_gym_jax/maps/Spielberg/`, which is itself a copy of that dataset.

**Why a real circuit matters here.** Every track in `mpcc_tuning/track.py` is
synthetic and built by hand — an oval, a smooth harmonic, and a circuit
assembled to contain one of each corner type. A synthetic track can be built to
suit the hypothesis being tested, and the per-sector result in
`docs/source/results.md` is a demonstration that conclusions move between
tracks. A published circuit that nobody in this repo designed is the control
for that.

The polyline is not closed (0.4 m gap at the seam) and is resampled to uniform
arc length by `Track`, which is where the closure is absorbed.
