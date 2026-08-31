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

## `icra_t1_raceline.csv`

The optimised raceline for ICRA 2026 Track 1, from the team's own trajectory
optimiser. Semicolon-separated, and the columns that matter here are

    x; y; vx_mps; ...; kappa_radpm; ...; s_m; w_left_m; w_right_m

**This is the file that makes a width-dependent experiment possible.** The
corridor varies from \SI{0.69}{} to \SI{3.13}{\meter} round one lap --- a factor
of 4.5 --- where every synthetic track in `mpcc_tuning/track.py` is a
constant-width ribbon. A controller handed one number for the whole track
cannot be asked whether its weights should depend on the width, so until this
was loaded the question was unaskable rather than unanswered.

It also carries the optimiser's speed profile (2.4--\SI{6.1}{\meter\per\second})
and curvature, so "what should the weights be here" has a reference answer that
did not come from us.

## `icra_t2_sectors_reference.yaml`

**Not used by any code here. Kept because it is the ground truth.** The team
already hand-tunes controller weights per sector on Track 2, and this file
records both the values and the reasoning:

    S1   s 1.5-7.8,  FAST (7.1-8.8 m/s), tight corridor (0.87 m), carpet
    S2   s 20.7-26.6, sharp (|k| 0.60), tight (0.98 m), CARPET, drifts
    S4   s 49.4-53.7, tightest/slowest (|k| 0.82), narrowest (0.79 m), CARPET

Two things in it are worth taking seriously rather than rediscovering.

The sectors are justified by **corridor width, surface and speed together**, not
by curvature alone --- which is the case for the feature set in
`mpcc_tuning/ltc.py` carrying width as well as sector.

And `blend_dist: 1.0` exists because hard sector switching **causes wobble**:
"Kills the single-cycle weight steps that cause wobble." `Track.sector()`
switches hard. That is a defect this repo has and the team has already fixed.
