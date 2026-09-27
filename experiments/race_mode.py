"""Race-mode Phase 1: head-to-head behaviour training against ONE opponent.

    ACADOS_SOURCE_DIR=... PYTHONPATH=...:. python3 experiments/race_mode.py --seeds 6

The shipping stack (acados + dynamic drift + STD plant, discrete integrator to
match the online-net training), on the canonical Track.icra_t2_smooth. One
opponent per episode, cycled across relative-pace classes -- static / slower /
equal / faster (scaled to the ego's own solo pace) -- placed a few metres ahead,
with the ego start point varied per seed (physical seeds, not an RNG).

What the policy must learn (per sector): whether to STAY BEHIND at a distance or
OVERTAKE, based on relative pace, available space, closing speed, and WHICH SIDE
IS OPEN. A sector that is too tight / crash-prone is a bad place to attempt a
pass, so the net should follow there and pass in the good sectors instead.

Arms, identical except what emits theta each tick:
  const   START weights held fixed (the best-constant baseline)
  fixed   fixed_schedule -- the rule-based relative-pace lookup (baseline to beat)
  ltc     WeightPolicy(LTCCell)  -- the paper's memory tuner (MPCC-critic, keep-best)
  mlp     WeightPolicy(MLPCell)  -- the memoryless ablation

Success is a MIX, not one headline: overtakes completed + no-contact (clean) +
time gained (progress vs the follow baseline), across opponent configs and seeds.
Per-sector attempt/pass/contact are logged so the "good overtake sector" the net
learns can be read out.

The MPCC keep-out carries the opponent as an (x,y,r) circle whose radius is both
cars' half-widths (mpcc_tuning/opponents.py); the learnable berth d_obs (weight 6)
sets how much room the pass leaves. Contact = the two bodies touch (dist < radius).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mpcc_tuning import baselines as B  # noqa: E402
from mpcc_tuning.track import Track  # noqa: E402
from mpcc_tuning.mpcc import WEIGHT_NAMES  # noqa: E402

OUT = ROOT / "results" / "race"
ARMS = ("const", "fixed", "ltc", "mlp")


def signed_gap(track, s_ego, s_opp):
    """+ opponent ahead, - opponent behind, in metres of arc length."""
    d = (s_opp - s_ego) % track.length
    return d - track.length if d > track.length / 2 else d


def side_open(track, opponents, car_w=0.24):
    """Two features appended to the base 18: WHICH SIDE IS OPEN at the nearest
    opponent, and whether a pass fits there at all.

    The opponent sits at ``offset`` in lateral() convention; the corridor has
    room ``wl`` (-normal) and ``wr`` (+normal) at the opponent's arc length.
    room_right = wr - offset, room_left = wl + offset.
      f0 = tanh(room_right - room_left)  -- sign = which side has more room
      f1 = clip(max(room_right,room_left) - car_w, over one car width)  -- pass fits?
    No opponent -> (0, 1): no side pressure, a pass trivially fits.
    """
    if not len(opponents):
        return [0.0, 1.0]
    o = opponents[0]   # phase 1: one opponent (extend to nearest-ahead for phase 2)
    try:
        wl, wr = track.width(o.s % track.length)
    except Exception:
        wl = wr = track.half_width
    off = float(getattr(o, "offset", 0.0))
    room_r = float(wr) - off
    room_l = float(wl) + off
    return [float(np.tanh(room_r - room_l)),
            float(np.clip((max(room_r, room_l) - car_w) / car_w, -1.0, 1.0))]


def race_features(track, s5, opponents=(), opp_speed_est=None,
                  slip=None, gap_rate=None, sector_suit=None):
    """Base 18 + 2 side-open + 9 TEMPORAL/DYNAMIC = 29. Indices 0..19 are left
    untouched so fixed_schedule (which reads feat[7], feat[8], feat[14:18]) works.

    The 9 appended features are the history-dependent signals the policy previously
    could NOT see (it only observed an instantaneous s5=[x,y,psi,v,s]); without them
    memory has nothing to integrate and the recurrent LTC cannot beat the memoryless
    MLP. They are, in order:
      20  rear slip angle  alpha_r / ALPHA_R_REF          (signed; the drift the reward
                                                           punishes but the policy could not observe)
      21  body sideslip    beta / BETA_REF                (signed)
      22  yaw rate         r / 3.0                        (signed)
      23  gap closing rate d(gap)/dt                      (signed; <0 = closing on the opponent)
      24  opponent speed   opp_speed_est / v_max          (the raw noisy tracked speed, not just
                                                           the coarse class one-hot at 14:18)
      25-28 per-sector overtake suitability  tanh(sec_pass - sec_contact)  (the "where can I
                                                           overtake" map, accumulated over the race)
    Neutral (0) when an arg is None, so the vector is always length 29."""
    from mpcc_tuning.ltc import features
    base = features(track, s5, opponents, opp_speed_est=opp_speed_est)
    a_r, be, yr = slip if slip is not None else (0.0, 0.0, 0.0)
    extra = [np.tanh(float(a_r) / ALPHA_R_REF), np.tanh(float(be) / BETA_REF),
             np.tanh(float(yr) / 3.0),
             np.tanh(float(gap_rate) / 1.0) if gap_rate is not None else 0.0,
             np.tanh(float(opp_speed_est) / 4.0) if opp_speed_est is not None else 0.0]
    ss = sector_suit if sector_suit is not None else (0.0, 0.0, 0.0, 0.0)
    extra += [np.tanh(float(x)) for x in ss]
    return np.concatenate([base, np.array(side_open(track, opponents), float), np.array(extra, float)])


N_RACE_FEATURES = 29


def measure_pace(m, track, th0, steps):
    """Ego solo mean speed, to scale opponent speeds to the ego's own pace."""
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
    P.reset(s0=0.0, v0=1.0); m.reset()
    vs = []
    for _ in range(steps):
        u = m.value(P.state_dyn(), th0)["u0"]; s5, r, off, tr = P.step(u)
        vs.append(float(P._x[3]))
        if off or tr:
            break
    return float(np.mean(vs)) if vs else 1.4


# relative-pace classes: opponent speed as a fraction of the ego's solo pace.
# static=parked; slower=catchable; equal=expensive pass; faster=cannot catch.
PACE = {"static": 0.0, "slower": 0.55, "equal": 0.90, "faster": 1.20}
# For the FAIR (grip-limited) opponent, pace is a STRAIGHT-LINE target; it slows for corners
# AND pays a grip cost for lateral moves, so the realised lap pace is well below the target.
# Recalibrated DOWN (was 0.9/1.3/1.8, which made "faster" ~1.8x the ego -> uncatchable): now
# slower is genuinely slower, equal ~matched, faster only a little quicker so it is catchable.
FAIR_PACE = {"static": 0.0, "slower": 0.80, "equal": 1.00, "faster": 1.10}  # equal = matched; faster = only a TOUCH quicker (was 1.35 = uncatchable, over-pulled the chase and starved equal). Tight realistic spread 0.8/1.0/1.1.
PACE_KINDS = ("static", "slower", "equal", "faster")

# Pace-DEPENDENT reward, per the intended behaviour for each opponent type:
#   static/slower : overtake, but drive STABLE (not full-aggressive) -- a strong
#                   bonus for a CLEAN pass, no extra speed pressure.
#   equal         : find and hold a good overtaking POSITION, then pass patiently
#                   -- the largest clean-pass bonus, plus a small reward for staying
#                   engaged in a passing window.
#   faster        : cannot out-wait a faster car -- reward CLOSING PACE (go fast to
#                   catch up); a pass, if it comes, still pays.
# Contact is always heavily penalised: a pass that touches is worse than no pass.
PASS_BONUS = {"static": 7.0, "slower": 7.0, "equal": 10.0, "faster": 0.0}
# Bigger keep-out so a pass leaves ROOM (was "nearly touching"): the MPCC treats the
# opponent as a KEEPOUT_R circle, so with the START berth d_obs~0.15 it holds the ego
# centre >~0.51 m off -> ~0.27 m clear body-to-body. This is a fixed safety margin, NOT
# the tunable d_obs (which must stay strictly inside its box). CONTACT_R is the physical
# body-to-body touch distance used only for detecting an actual collision.
KEEPOUT_R = 0.36
CONTACT_R = 0.24
# The dynamic-MPCC speed comes from the v_ref reference profile (curvature-limited at a_lat)
# scaled by k_v. The default a_lat=6.0 is only 55% of the real tyre grip (mu*g=10.8), so the car
# was capped well below the limit and no weight could make it faster. Raise the reference profile
# toward the real grip so the car HAS headroom to go quicker; the calibrated slip risk (ALPHA_R_REF,
# from the handling/bifurcation analysis) is the safety bound that keeps it off the tyre-saturation
# edge, and the class-dependent speed reward (W_SPEED_CLASS) decides how much of it to use per opponent.
A_LAT_RACE = 9.0        # ~83% of the real tyre grip (was 6.0 = 55%)
# CORRIDOR: off-track is the SOFT corridor being BOUGHT for progress (tools/offtrack_diagnostic.py):
# the plan dips out and the plant tracks it off (0 cm error, no model mismatch). Uniformly stiffening
# the corridor (scale 6) halved off-track but COST overtakes (the car would not use the width to go
# around an opponent). TWO-LAYER fixes that: a moderate SOFT inner buffer (keep 0.19, scale 1) that
# the car can still push into to overtake, plus a near-hard SOFT-HARD wall 1 cm inside the off-line
# (keep 0.13, huge penalty) that the plan will not cross. Wide 6 cm buffer = reaction runway (not the
# 1-2 cm that would hit the wall abruptly). Both rows soft -> no solve-rate collapse.
CORRIDOR_SLACK_SCALE = 1.0        # inner buffer penalty (moderate -- usable for overtaking)
CORRIDOR_EDGE_SAFETY = 0.01       # outer wall 1 cm inside the 0.12 off-line -> keep_edge 0.13
CORRIDOR_OUTER_SCALE = 60.0       # outer-wall slack penalty (near-hard; raised 30->60 as the learned aggressive policy bought off the 30x wall -> 38-57% off-track)
CORRIDOR_INNER_SAFETY = 0.07      # inner buffer at car_half_width + this = 0.19
CORRIDOR_KW = dict(two_layer_corridor=True, corridor_slack_scale=CORRIDOR_SLACK_SCALE,
                   corridor_safety=CORRIDOR_INNER_SAFETY, corridor_edge_safety=CORRIDOR_EDGE_SAFETY,
                   corridor_outer_scale=CORRIDOR_OUTER_SCALE)


SAFE_FOLLOW = 1.5   # metres: safe following gap behind a faster car (no rear-end)
# SLIP-BASED STABILITY RISK, calibrated offline from the real fitted tyre + geometry
# (tools/handling_analysis.py -> results/race/stability/stability_limits.json): the RC car is
# understeer-stable, so the binding limit is TYRE SATURATION at rear slip alpha_r ~ 0.148 rad
# (8.5 deg). Penalise rear slip beyond 0.75x the peak and body sideslip beyond a soft drift
# limit, EACH TICK -> a certain, graded, physically-grounded cost of running near the limit, so
# the reward optimum is interior. The reference is scaled by the OPPONENT CLASS: vs a faster car
# spend the full grip margin, vs a slower one keep more -> the risk BUDGET differs by opponent,
# giving the tuner a reason to learn different weights per class.
ALPHA_R_REF = 0.111     # rad -- 0.75x the rear-tyre saturation peak (0.148)
BETA_REF = 0.10         # rad -- soft body-sideslip / drift limit
W_SLIP = 8.0            # weight of the slip risk penalty
LR_VEH = 0.1515         # CoG->rear axle [m], for the rear slip angle
# CLASS-DEPENDENT speed/risk trade-off, keyed on the tracked opponent class:
#  - W_SPEED_CLASS: how hard to reward SPEED (less time per metre). Push HARD vs a faster/equal
#    car (must go to the limit to keep up / make the pass), gently vs a slower one (no need).
#    This pushes q_v UP and the damping weights r_a/r_dv DOWN as well as k_v -- all the knobs
#    that make the car quicker -- so the OPTIMAL weights differ by opponent.
#  - CLASS_SCALE: how much of the calibrated slip budget to spend -- looser vs faster (spend the
#    grip margin), tighter vs slower (keep it). The stability analysis still BOUNDS the push.
W_SPEED_CLASS = {"static": 0.3, "slower": 0.3, "equal": 0.7, "faster": 1.0}   # legacy buckets (kept)
CLASS_SCALE = {"static": 0.6, "slower": 0.7, "equal": 0.85, "faster": 1.0}    # -- superseded by diff
# OVERTAKE-COMMITMENT + CONTINUOUS-RISK reward (2026-09-27). Instead of the fixed class buckets, the
# risk/reward rides on the MEASURED difficulty diff = v_opp/v_ego (0 parked .. ~1 matched .. >1 faster):
# a harder pass rewards MORE speed, allows MORE slip, and pays a bigger COMMITMENT reward (the speed
# edge WHILE alongside) so pushing into a pass scores BEFORE it completes -- bridging the "valley of
# death" where only a completed pass (bonus) or a crash (contact) scored, so the tuner learned to
# FOLLOW instead of race (k_v collapsed vs equal). Contact while alongside/ahead is a racing incident,
# not a careless rear-end, so it is penalised less. Diagnosis: results/race/race_phase1_dbound.json.
ENGAGE_WINDOW = 3.0     # m: |gap| within which the overtake-commitment reward applies
W_COMMIT = 1.5          # weight of the speed-edge-while-engaged (commitment) reward

# STAY-IN-THE-CORRIDOR shaping. Measured (tools trajectory read, 2026-09-25): the racing policy is
# NOT over-speeding -- 0% of ticks exceed the grip-limit speed; it peaks at ~20-40% of the cornering
# limit, so it has grip to spare. The 45-57% off-track is LATERAL corridor-departure under aggressive
# weights, not corner over-speed. So we reward staying IN the corridor rather than slowing down --
# and because the grip headroom is there, this should cut off-track WITHOUT costing pace/overtakes.
# A graded near-edge penalty gives a gradient before it leaves; a terminal penalty makes actually
# leaving cost the race (the return then prefers fast-AND-inside over fast-and-wide).
EDGE_MARGIN = 0.20      # m: penalise within this distance of the corridor edge
W_EDGE = 5.5            # penalty weight per metre inside the margin (lightened from 8.0: the 8.0
OFF_PENALTY = 10.0      # run over-braked -- pace 2.0->1.6 and equal passes 0.9->0.73. Lighter penalty
                        # (~-30%) aims to keep mlp's stability gain while recovering the racing edge.


def race_reward(kind, r, just_passed, contact, v_ego, v_opp, gap,
                alpha_r=0.0, beta=0.0, off=False, edge_dist=None, opp_pace=1.0, return_parts=False):
    """Safe-first behaviour reward + a calibrated, class-scaled SLIP stability risk.

    - Contact strongly dominates (-15).
    - static/slower/equal: clean-pass bonus + a bounded closing reward.
    - faster: no pass bonus; catch up from far but hold SAFE_FOLLOW (follow, don't rear-end).
    - SLIP RISK: -W_SLIP*(max(0,|alpha_r|-scale*ALPHA_R_REF) + 0.5*max(0,|beta|-BETA_REF)) where
      scale = CLASS_SCALE[kind]. Grounded in the tyre curve, paid every tick, budget by class.

    With ``return_parts=True`` also returns a dict of the additive components (base/speed/pass/
    closing/slip/contact), so a diagnostic can see WHICH term -- if any -- actually differs by
    class along the closed-loop trajectory (e.g. whether the slip risk ever activates at all).
    """
    v_ego = float(v_ego); v_opp = float(v_opp)
    # difficulty = the opponent's CHARACTERISTIC pace relative to the ego (STABLE per episode, e.g.
    # FAIR_PACE[kind]: slower 0.8, matched 1.0, faster 1.35), NOT the jittery instantaneous v_opp/v_ego
    # (which would drop the speed reward exactly when the ego accelerates to pass). 0 = parked.
    diff = min(max(float(opp_pace), 0.0), 1.4)
    if contact:
        # a committed pass (alongside/ahead) is a racing incident; a rear-end (opponent still ahead) is careless
        pen = -8.0 if gap <= 0.3 else -15.0
        x = float(r) + pen
        return (x, dict(base=float(r), speed=0.0, pass_b=0.0, commit=0.0, closing=0.0, slip=0.0,
                        edge=0.0, off_pen=0.0, contact=pen)) if return_parts else x
    base = float(r)
    speed = (0.3 + 0.7 * min(diff, 1.2)) * v_ego            # reward speed MORE when the pass is harder
    pass_b = (6.0 + 6.0 * diff) if just_passed else 0.0     # a pass vs a faster car is worth more
    # VALLEY-OF-DEATH BRIDGE: reward the speed edge WHILE engaged (committing to the pass) -- continuous,
    # so pushing into a pass pays before it completes, bigger when the opponent is faster.
    commit = W_COMMIT * (v_ego - v_opp) * (0.5 + diff) if (abs(gap) < ENGAGE_WINDOW and v_ego > v_opp) else 0.0
    closing = 0.0
    if 0.0 < gap < 6.0 and v_ego > v_opp - 0.1:
        closing += 0.3 * (6.0 - gap)                       # close on a car ahead you can catch
    if 0.0 < gap < SAFE_FOLLOW and v_ego <= v_opp + 0.1:
        closing -= 1.0 * (SAFE_FOLLOW - gap)               # tailgating one you can't pass -> back off
    # NEAR-MATCHED (equal / barely-faster): reward HOLDING a good overtaking position, so the tuner
    # sets up and commits to the hard pass instead of abandoning it -- recovers ltc's equal-passing,
    # which the pure chase reward had starved (equal dipped to 0.55 while it over-invested in faster).
    if 0.85 <= diff <= 1.2 and -1.0 < gap < 3.0:
        closing += 0.4
    scale = min(max(0.65 + 0.4 * diff, 0.65), 1.10)        # slip budget: LOOSER vs a faster opponent
    slip = -W_SLIP * (max(0.0, abs(float(alpha_r)) - scale * ALPHA_R_REF)
                      + 0.5 * max(0.0, abs(float(beta)) - BETA_REF))
    # STAY-IN-THE-CORRIDOR: graded penalty within EDGE_MARGIN of the edge (a gradient before it
    # leaves) + a terminal penalty for actually leaving. The car has grip to spare, so this trades
    # nothing against pace -- it just stops the lateral wander out of the corridor.
    edge = 0.0
    if edge_dist is not None and float(edge_dist) < EDGE_MARGIN:
        edge = -W_EDGE * (EDGE_MARGIN - float(edge_dist))
    off_pen = -OFF_PENALTY if off else 0.0
    x = base + speed + pass_b + commit + closing + slip + edge + off_pen
    return (x, dict(base=base, speed=speed, pass_b=pass_b, commit=commit, closing=closing, slip=slip,
                    edge=edge, off_pen=off_pen, contact=0.0)) if return_parts else x


def run(arm, seed=0, n_ep=10, steps=5500, n_hidden=12, ego_pace=1.4,
        factor=2.0, box="adapt", dump_traj=False, fair_opp=False):
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    from mpcc_tuning.ltc import (LTCCell, MLPCell, THETA_HI, THETA_LO,
                                 PolicyTuner, WeightPolicy, fixed_schedule)
    from mpcc_tuning.plant_scuderia import ScuderiaPlant
    from mpcc_tuning.opponents import ObstacleTracker, Opponent, RacelineOpponent

    track = Track.icra_t2_smooth()
    st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    m = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                   q_vref=st.q_vref, discrete=True, max_obstacles=1,
                   a_lat_sectors=[A_LAT_RACE] * 4,   # raise the reference speed toward real grip
                   **CORRIDOR_KW,                    # two-layer corridor: usable inner + edge wall
                   name=f"race_{arm}_{seed}")
    lo, hi = (B.adaptation_box("icra_t2_smooth", factor) if box == "adapt"
              else (THETA_LO, THETA_HI))

    tuner = pol = None
    if arm in ("ltc", "mlp"):
        cell = (LTCCell if arm == "ltc" else MLPCell)(N_RACE_FEATURES, n_hidden, seed=seed)
        # CLASS-CONDITIONED RESIDUAL HEAD: theta = theta0 + span*tanh(G u) + Delta(class).
        # The reward diagnostic (tools/reward_diagnostic.py) proved the reward's optimal weight
        # DIFFERS by opponent class (k_v 0.30/0.60/0.75), but the saturated tanh readout collapsed
        # every class to one corner. Delta(class) is a small per-class table the tuner learns
        # directly (indexed by class), so it CAN emit different weights per class. Set each episode
        # via tuner.set_class below.
        pol = WeightPolicy(cell, th0, lo, hi, seed=seed, n_classes=len(PACE_KINDS))
        # THE fix for "we get stuck on the values and don't explore more": the old call set
        # explore=0.06 (CONTROL noise on steering/accel only) but theta_explore=0 and entropy=0,
        # so the WEIGHTS were emitted deterministically and, once the tanh saturated at the box
        # edge, the policy gradient (1-tanh^2 z) vanished and it was stuck. Now:
        #  - theta_explore=0.15 : Gaussian noise ON THE WEIGHTS each tick -> actually try
        #    different weight combinations (incl. higher k_v = faster) and learn from them;
        #  - entropy=0.01 : anti-saturation bonus that pushes theta OFF a saturated bound so it
        #    can settle interior instead of pinning to the corner.
        # theta_prior lowered 0.3 -> 0.15 so the trust region does not fight the exploration.
        # (frozen eval turns exploration off, so the banked net stays deterministic.)
        # RETURN critic, not MPCC. The class signal lives in the MEASURED return (the reward
        # diagnostic showed the optimal weight differs by class), but the MPCC critic's gradient
        # dJ*/dtheta points the SAME way for every class -- only its sign comes from reward -- so
        # under it the class-residual head just drives every class to the same corner (measured:
        # residuals saturated at -0.59 on all weights, class spread ~0). The return critic fits the
        # actual return and moves theta via the theta-exploration that correlated with higher return
        # FOR THIS CLASS, so different classes differentiate. See mpcc_tuning/ltc.py critic docs.
        # alpha_delta / trust_region_delta give the low-dimensional per-class head its own budget.
        tuner = PolicyTuner(m, pol, alpha=3e-3, explore=0.06, delta_clip=1.0,
                            seed=seed, trust_region=0.01, theta_prior=0.15,
                            theta_explore=0.15, entropy=0.03, critic="return",
                            alpha_delta=0.01, trust_region_delta=0.03)

    rng = np.random.default_rng(seed)
    rows = []
    traj = {}                      # one trajectory per opponent kind (last episode), if dumping
    for ep in range(n_ep):
        kind = PACE_KINDS[(seed + ep) % 4]
        v_opp = (FAIR_PACE if fair_opp else PACE)[kind] * ego_pace
        s0 = (seed % 4) * track.length / 4.0
        v0 = 1.3 + 0.1 * (seed % 3)          # start near the ego's own pace, not crawling
        gap0 = 3.0 + 1.5 * (ep % 3)          # opponent starts a few m ahead
        if fair_opp:
            opp = RacelineOpponent(track, s0=(s0 + gap0) % track.length, pace=v_opp,
                                   offset=0.0, radius=KEEPOUT_R, a_lat=2.5)
        else:
            opp = Opponent(track, s0=(s0 + gap0) % track.length, speed=v_opp,
                           offset=0.0, radius=KEEPOUT_R)
        tracker = ObstacleTracker(dt=0.05)
        P = ScuderiaPlant(track, model="std", dt=0.05); P.max_steps = steps
        s5 = P.reset(s0=s0, v0=v0); m.reset()
        if tuner is not None:
            tuner.reset()
            tuner.set_class(PACE_KINDS.index(kind))   # index the class-residual head for this opponent
        opp.reset()
        tracker.update(opp.pose()[:2])
        m.set_obstacles([opp.keepout()])
        # initial slip (near zero at reset) for the enriched observation; no gap/sector history yet
        _b0 = float(P._x[6]) if P._x.size > 6 else 0.0; _r0 = float(P._x[5]); _v0 = float(P._x[3])
        _ar0 = -np.arctan2(_v0 * np.sin(_b0) - LR_VEH * _r0, _v0 * np.cos(_b0)) if _v0 * np.cos(_b0) > 0.05 else 0.0
        feat = race_features(track, P.state5(), [opp], opp_speed_est=tracker.speed,
                             slip=(_ar0, _b0, _r0))
        if arm in ("ltc", "mlp"):
            theta, u = tuner.act(feat, P.state_dyn())
        elif arm == "fixed":
            theta = fixed_schedule(feat, th0); u = m.value(P.state_dyn(), theta)["u0"]
        else:  # const
            theta = th0; u = m.value(P.state_dyn(), theta)["u0"]

        base = float(P.state5()[4]); off = tr = False
        passes = 0; contact = False; seen = False
        sec_attempt = np.zeros(4); sec_pass = np.zeros(4); sec_contact = np.zeros(4)
        nW = th0.shape[0]                        # number of tunable weights (8, or 9 with d_bound)
        theta_acc = np.zeros(nW); n_th = 0       # mean emitted weights this episode
        v_sum = 0.0; v_max = 0.0; n_v = 0        # ON-TRACK PACE (the racing objective, not clean%)
        prev_g = None                            # for the gap-closing-rate feature
        sec_theta = np.zeros((4, nW)); sec_theta_n = np.zeros(4)   # weights BY SECTOR
        TEX = []; TEY = []; TEV = []; TOX = []; TOY = []; TG = []; TP = []   # trajectory (if dumping)
        for _ in range(steps):
            theta_acc += np.exp(np.asarray(theta, float)); n_th += 1
            s5n, r, off, tr = P.step(u)
            ex, ey = float(P._x[0]), float(P._x[1])
            # fair opponent sees the ego (for reactive overtaking); dumb one steps blind
            if fair_opp:
                opp.step(0.05, ego=(ex, ey, float(P._x[3])))
            else:
                opp.step(0.05)
            vo = float(getattr(opp, "speed", v_opp))   # opponent's CURRENT speed (dynamic if fair)
            ox, oy, rad = opp.keepout()
            dist = float(np.hypot(ex - ox, ey - oy))
            s_ego = track.project(ex, ey)
            g = signed_gap(track, s_ego, opp.s)
            sec = int(track.sector(track.wrap(s_ego)))
            # distance to the nearer corridor edge, for the STAY-IN shaping. Aligned to the PLANT's
            # own off-track rule (plant_scuderia.py: off when lateral > wr-0.12 or -lateral > wl-0.12,
            # a 0.12 m body margin per side), so edge_dist is EXACTLY the plant's signed clearance:
            # >0 inside, <=0 at the boundary the plant flags off. Same lateral()/width() convention.
            _lat = float(track.lateral(ex, ey)); _wl, _wr = track.width(track.wrap(s_ego))
            edge_dist = min(float(_wr) - 0.12 - _lat, _lat + float(_wl) - 0.12)
            # slip state for the calibrated stability risk: rear slip angle + body sideslip
            _v = float(P._x[3]); _r = float(P._x[5]); _beta = float(P._x[6]) if P._x.size > 6 else 0.0
            v_sum += _v; v_max = max(v_max, _v); n_v += 1        # pace accumulation
            _vx = _v * np.cos(_beta); _vy = _v * np.sin(_beta)
            _alpha_r = -np.arctan2(_vy - LR_VEH * _r, _vx) if _vx > 0.05 else 0.0
            sec_theta[sec] += np.exp(np.asarray(theta, float)); sec_theta_n[sec] += 1
            # contact = the two bodies actually touch (physical distance, not the keep-out)
            if dist < CONTACT_R and not contact:
                contact = True; sec_contact[sec] += 1
            # engagement: opponent within a car-length ahead and we are closing
            if 0 < g < 2.0 and float(P._x[3]) > vo + 0.05:
                sec_attempt[sec] += 1
            # a completed pass: opponent went from ahead to behind, we are faster
            just_passed = False
            if g < 0 and abs(g) < track.length / 4 and not seen and float(P._x[3]) > vo:
                passes += 1; seen = True; sec_pass[sec] += 1; just_passed = True
            elif g > 0.5:
                seen = False
            if dump_traj:
                TEX.append(ex); TEY.append(ey); TEV.append(float(P._x[3]))
                TOX.append(ox); TOY.append(oy); TG.append(g); TP.append(passes)
            m.set_obstacles([opp.keepout()])
            tracker.update(opp.pose()[:2])
            gr = (g - prev_g) / 0.05 if prev_g is not None else 0.0   # gap closing rate
            prev_g = g
            fn = race_features(track, s5n, [opp], opp_speed_est=tracker.speed,
                               slip=(_alpha_r, _beta, _r), gap_rate=gr,
                               sector_suit=(sec_pass - sec_contact))
            # pace-DEPENDENT shaped reward (race_reward): the pass bonus fires on the
            # tick the pass COMPLETES (just_passed), which the old code missed because
            # `seen` was already set -- so the tuner never saw a pass reward before.
            r_shaped = race_reward(kind, r, just_passed, contact,
                                   float(P._x[3]), vo, g, alpha_r=_alpha_r, beta=_beta,
                                   off=off, edge_dist=edge_dist,
                                   opp_pace=(FAIR_PACE if fair_opp else PACE)[kind])
            if arm in ("ltc", "mlp"):
                out = tuner.learn(r_shaped, P.state_dyn(), fn, off or contact)
                if out[0] is None:
                    break
                theta, u = out
            elif arm == "fixed":
                theta = fixed_schedule(fn, th0); u = m.value(P.state_dyn(), theta)["u0"]
            else:
                theta = th0; u = m.value(P.state_dyn(), theta)["u0"]
            if off or tr or contact:
                break
        laps = (float(P.state5()[4]) - base) / track.length
        clean = (not off) and (not contact)
        rows.append(dict(ep=ep, kind=kind, laps=round(laps, 3), passes=int(passes),
                         contact=bool(contact), off=bool(off), clean=bool(clean),
                         mean_speed=round(v_sum / max(n_v, 1), 3), max_speed=round(v_max, 3),
                         sec_attempt=sec_attempt.tolist(), sec_pass=sec_pass.tolist(),
                         sec_contact=sec_contact.tolist(),
                         theta_mean=(theta_acc / max(n_th, 1)).tolist(),
                         sec_theta=(sec_theta / np.maximum(sec_theta_n[:, None], 1)).tolist()))
        if dump_traj and TEX:
            traj[kind] = dict(EX=TEX, EY=TEY, EV=TEV, OX=TOX, OY=TOY, GAP=TG, PASS=TP)
    # save the trained policy so the GIF/eval can REPLAY the learned behaviour
    if tuner is not None:
        ndir = OUT / "nets"; ndir.mkdir(parents=True, exist_ok=True)
        np.savez(str(ndir / f"race_{arm}_{seed}.npz"), G=pol.G, cell_p=pol.cell.p,
                 th0=th0, n_hidden=pol.cell.n, arm=arm, seed=seed,
                 lo=np.asarray(lo, float), hi=np.asarray(hi, float),
                 D_class=(pol.D_class if pol.D_class is not None else np.zeros((0, 8))),
                 delta_log=pol.delta_log)
    if dump_traj and traj:
        # per-opponent-kind trajectory (last episode of each kind) for the 2D paper plots,
        # so tools/race_2d.py can render WITHOUT re-driving (no acados).
        tdir = OUT / "traj"; tdir.mkdir(parents=True, exist_ok=True)
        np.savez(str(tdir / f"traj_{arm}_{seed}.npz"),
                 **{f"{k}_{fld}": np.array(v[fld]) for k, v in traj.items() for fld in v})
    last = rows[-8:] if len(rows) >= 8 else rows
    return dict(arm=arm, seed=seed,
                laps=float(np.mean([r["laps"] for r in last])),
                passes=float(np.mean([r["passes"] for r in last])),
                clean=float(np.mean([r["clean"] for r in last])),
                contact=float(np.mean([r["contact"] for r in last])),
                rows=rows)


def one(job):
    arm, seed, n_ep, steps, ego_pace, dump_traj, fair_opp = job
    t0 = time.perf_counter()
    out = run(arm, seed=seed, n_ep=n_ep, steps=steps, ego_pace=ego_pace,
              dump_traj=dump_traj, fair_opp=fair_opp)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5500,
                    help="~5 laps at START pace (73.6 m track, ~1.35 m/s) so the ego "
                    "meets each opponent several times and can change strategy per lap")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--pilot", action="store_true",
                    help="1 seed, 4 episodes, 600 steps, arms const+ltc -- quick smoke")
    ap.add_argument("--out", default=str(OUT / "race_phase1.json"))
    ap.add_argument("--dump-traj", action="store_true",
                    help="save one trajectory per opponent kind (results/race/traj/) so "
                    "tools/race_2d.py can render 2D paper plots without re-driving")
    ap.add_argument("--fair-opp", action="store_true",
                    help="use the FAIR opponent (RacelineOpponent): grip-limited speed "
                    "(slows for corners like our car) + reactive side-step overtake, "
                    "instead of the dumb constant-speed centreline ghost")
    a = ap.parse_args(argv)
    if a.pilot:
        a.seeds, a.episodes, a.steps, a.arms = 1, 2, 1200, ["const", "ltc"]

    # measure ego pace once (solo START drive) to scale opponents
    from mpcc_tuning.acados_mpcc import AcadosMPCC
    track = Track.icra_t2_smooth(); st = B.start("icra_t2_smooth")
    th0 = np.asarray(st.theta(), float)
    mp = AcadosMPCC(track, horizon=st.horizon, dt=0.05, vehicle="dynamic",
                    q_vref=st.q_vref, discrete=True, max_obstacles=1,
                    a_lat_sectors=[A_LAT_RACE] * 4, **CORRIDOR_KW, name="race_pace")
    ego_pace = measure_pace(mp, track, th0, a.steps)
    _pace = FAIR_PACE if a.fair_opp else PACE
    print(f"  ego solo pace = {ego_pace:.2f} m/s ; {'FAIR ' if a.fair_opp else ''}opponents "
          + "(straight-line target): " + ", ".join(f"{k}={_pace[k]*ego_pace:.2f}" for k in PACE_KINDS), flush=True)

    jobs = [(arm, s, a.episodes, a.steps, ego_pace, a.dump_traj, a.fair_opp)
            for s in range(a.seeds) for arm in a.arms]
    n_proc = a.jobs or min(len(jobs), os.cpu_count() or 1)
    print(f"  {len(jobs)} runs, {a.episodes} episodes, {n_proc} processes\n", flush=True)

    import multiprocessing as mp2
    res = []
    with mp2.get_context("spawn").Pool(n_proc) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o)
            print(f"  done  {o['arm']:<6} seed {o['seed']}  {o['laps']:5.2f} laps"
                  f"  {o['passes']:.2f} passes  {o['clean']:.0%} clean"
                  f"  {o['contact']:.0%} contact  ({o['wall_s']:.0f}s)", flush=True)

    print(f"\n  {'arm':<7}{'laps':>7}{'passes':>8}{'clean':>7}{'contact':>8}")
    S = {}
    for arm in a.arms:
        r = [x for x in res if x["arm"] == arm]
        if not r:
            continue
        S[arm] = dict(laps=float(np.mean([x["laps"] for x in r])),
                      passes=float(np.mean([x["passes"] for x in r])),
                      clean=float(np.mean([x["clean"] for x in r])),
                      contact=float(np.mean([x["contact"] for x in r])), n=len(r))
        print(f"  {arm:<7}{S[arm]['laps']:7.2f}{S[arm]['passes']:8.2f}"
              f"{S[arm]['clean']:7.0%}{S[arm]['contact']:8.0%}")

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(summary=S, ego_pace=ego_pace, runs=res), indent=2) + "\n")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
