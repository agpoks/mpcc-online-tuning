"""A liquid time-constant policy over the MPCC's cost weights.

The tuner in ``learner.py`` moves **one** theta by TD(lambda). This moves a
*policy* that emits theta from the situation, so the weights become a function
of what is happening rather than a single vector to be found.

## Why a liquid cell rather than any recurrent one

The behaviour a policy has to express here is *overtake or follow*, and two
things about it are awkward for a memoryless map:

* a **closing rate is a derivative of a range**, so it cannot be read from one
  frame at any resolution -- "catching them" and "being caught" look identical
  instantaneously;
* the decision should be taken **once and held**. Re-deciding at 20 Hz invites
  chattering between overtake and follow, which is a real failure and not a
  hypothetical (see ``scuderia_gym_jax``'s ``examples/overtake.py``, whose whole
  overtake logic exists to latch a side).

An LTC's effective time constant

    tau_eff = tau / (1 + tau f(x, h))

is **a hold duration set at run time by the input** -- long when the gate is
shut, short when the input drives it. That is the principled form of "decide on
an event, hold between", rather than a hand-set dwell time.

Cell and derivatives adapted from Hasani, Lechner, Amini, Rus & Grosu,
*Liquid Time-constant Networks*, AAAI 2021, and from the implementation in
``rtrrl-playground/rtrrl_playground/nets/ltc.py`` -- **copied in and adapted,
not imported**, per this repo's standing rule. The fused semi-implicit Euler
step is the paper's, and is stable at any step size because the state appears on
both sides.

## How it is trained, and why the recurrence is not free

The envelope theorem gives ``dQ/dtheta`` for free at the solution. The policy
adds ``dtheta/dphi`` for its own parameters ``phi``, and because ``h`` is
recurrent that factor is **not** a per-tick quantity: ``phi`` influenced ``h``
at every earlier tick too. That is exactly the influence recursion, and here it
is carried in the RFLO form

    P <- leak * P + immediate

with ``leak = 1/den < 1`` guaranteed by ``1/tau > 0``. Two things follow that
are worth stating. The bound is **structural, not a clipped hyperparameter** --
an LTC cannot learn to hold its state perfectly and so cannot make the influence
series diverge. And the decay is *input-dependent*: the trace forgets fast
exactly when the neuron is being driven hard, which is the property that makes
liquid cells and local online rules fit together.

Every parameter belongs to exactly one neuron, so the RFLO update is exact per
neuron and the influence array is the same shape as the parameters.
"""

from __future__ import annotations

import numpy as np

from mpcc_tuning.mpcc import WEIGHT_NAMES


def _sigmoid(z):
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


class LTCCell:
    """One LTC layer with analytic derivatives, RFLO-ready.

    ``theta`` here is the *cell's* parameter block ``[W | A | tau]``, not the
    MPCC's cost weights; the two are kept apart by calling the latter ``theta``
    only outside this class.
    """

    def __init__(self, n_in: int, n: int, dt: float = 1.0, tau_init=(2.0, 30.0),
                 tau_min: float = 1.0, tau_max: float = 60.0, a_max: float = 2.0,
                 input_gain: float = 3.0, seed: int = 0):
        self.n_in, self.n, self.dt = int(n_in), int(n), float(dt)
        self.tau_min, self.tau_max, self.a_max = tau_min, tau_max, a_max
        self.n_xi = n_in + n + 1                     # [x ; h ; bias]
        rng = np.random.default_rng(seed)
        # An LTC needs a bigger input gain than a tanh cell: the pre-activation
        # goes through a sigmoid gate and then only *scales* the reversal
        # potential, so the same spread in W xi produces a fraction of the state
        # variation a CT-RNN would get.
        W = rng.normal(0.0, input_gain / np.sqrt(self.n_xi), (n, self.n_xi))
        # Gaussian A, not +/-1. The state settles at f A / (1/tau + f), so A is
        # the only thing setting a unit's amplitude -- with A = +/-1 the layer is
        # one signal and its negation, the hidden state is near rank one, and a
        # linear head on top has almost nothing to work with.
        A = rng.normal(0.0, 1.0, (n, 1))
        tau = np.exp(rng.uniform(*np.log(tau_init), (n, 1)))
        self.p = np.concatenate([W, A, tau], axis=1)
        self.SL = {"W": (0, self.n_xi), "A": (self.n_xi, self.n_xi + 1),
                   "tau": (self.n_xi + 1, self.n_xi + 2)}
        self.reset()

    def reset(self):
        self.h = np.zeros(self.n)

    def step(self, x, need_D: bool = False):
        """Advance one tick. Returns ``(h, immediate, leak)``, plus ``D`` if asked.

        ``leak`` alone gives the RFLO/SnAp-1 approximation: each neuron carries
        its own influence and the coupling *between* neurons is dropped. ``D``
        is the full recurrent Jacobian, which turns the same recursion into
        exact RTRL. Both are provided so the approximation can be *measured*
        rather than assumed adequate -- see ``experiments/gradient_fidelity.py``.
        """
        nx = self.n_xi
        xi = np.concatenate([np.asarray(x, float), self.h, [1.0]])
        W, A, tau = self.p[:, :nx], self.p[:, nx], self.p[:, nx + 1]
        f = _sigmoid(W @ xi)
        den = 1.0 + self.dt * (1.0 / tau + f)
        h_new = (self.h + self.dt * f * A) / den

        dz = self.dt * f * (1.0 - f) * (A - h_new) / den
        imm = np.empty_like(self.p)
        imm[:, :nx] = dz[:, None] * xi[None, :]
        imm[:, nx] = self.dt * f / den
        imm[:, nx + 1] = h_new * self.dt / (tau ** 2 * den)
        leak = 1.0 / den                 # < 1 for any gate value, since 1/tau > 0
        D = None
        if need_D:
            # dh'/dh: the diagonal leak plus the gate's dependence on h.
            W_h = W[:, self.n_in:self.n_in + self.n]
            D = np.diag(leak) + dz[:, None] * W_h
        self.h = h_new
        return (h_new, imm, leak, D) if need_D else (h_new, imm, leak)

    def clip(self):
        lo, hi = self.SL["tau"]
        self.p[:, lo:hi] = np.clip(self.p[:, lo:hi], self.tau_min, self.tau_max)
        lo, hi = self.SL["A"]
        # A bounded reversal potential is a bounded state. Unbounded, it feeds
        # back through the recurrence and takes the cell to infinity, which over
        # a long online run with no batch to average over is what happens.
        self.p[:, lo:hi] = np.clip(self.p[:, lo:hi], -self.a_max, self.a_max)

    def tau_eff(self):
        """The hold duration each unit is currently choosing, in ticks."""
        nx = self.n_xi
        xi = np.concatenate([np.zeros(self.n_in), self.h, [1.0]])
        f = _sigmoid(self.p[:, :nx] @ xi)
        tau = self.p[:, nx + 1]
        return tau / (1.0 + tau * f)


class MLPCell:
    """The memoryless control for the same interface. One tanh layer, no state.

    Item 2d's gate requires beating this as well as a fixed schedule, so it has
    to exist and be trained by the identical rule -- otherwise a win is a win
    over an untuned baseline, which is not a win.
    """

    def __init__(self, n_in: int, n: int, seed: int = 0, **_):
        self.n_in, self.n, self.dt = int(n_in), int(n), 1.0
        self.n_xi = n_in + 1
        rng = np.random.default_rng(seed)
        self.p = rng.normal(0.0, 1.0 / np.sqrt(self.n_xi), (n, self.n_xi))
        self.SL = {"W": (0, self.n_xi)}
        self.reset()

    def reset(self):
        self.h = np.zeros(self.n)

    def step(self, x, need_D: bool = False):
        xi = np.concatenate([np.asarray(x, float), [1.0]])
        z = self.p @ xi
        self.h = np.tanh(z)
        imm = ((1.0 - self.h ** 2)[:, None] * xi[None, :])
        leak = np.zeros(self.n)                   # no influence carried
        return (self.h, imm, leak, np.zeros((self.n, self.n))) if need_D \
            else (self.h, imm, leak)

    def clip(self):
        pass

    def tau_eff(self):
        return np.zeros(self.n)


class WeightPolicy:
    """situation -> theta, with the output bounded to a measured box.

    ``theta = theta0 + G h``, clipped. **The clip is not cosmetic.** The usual
    safety argument for learning weights rather than steering -- "the MPCC
    enforces the constraints, so any theta is feasible" -- is false as stated:
    ``experiments/weights_as_behaviour.py`` found a q_c that drives off the
    track, and ``experiments/overtake_or_follow.py`` found that every attempted
    pass at q_v = 10 leaves it. The bound is what makes the argument
    recoverable, and it is a *ceiling on q_v* rather than a box, because
    behaviour is set by the ratio q_v/q_c and safety by q_v alone.
    """

    def __init__(self, cell, theta0, lo, hi, out_scale: float = 0.5, seed: int = 0,
                 influence: str = "rflo"):
        self.cell = cell
        self.theta0 = np.asarray(theta0, float)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        rng = np.random.default_rng(seed + 7919)
        self.G = rng.normal(0.0, out_scale / np.sqrt(cell.n),
                            (len(self.theta0), cell.n))
        if influence not in ("rflo", "exact"):
            raise ValueError("influence must be 'rflo' or 'exact'")
        self.influence = influence
        self.P = np.zeros_like(cell.p)      # dh/d(cell params)
        self.reset()

    def reset(self):
        self.cell.reset()
        self.P[:] = 0.0

    def step(self, feat):
        """One tick: advance the cell, emit theta, update the influence trace.

        The output is **squashed** into the box, not clipped. A hard clip has
        zero derivative, so once a weight sits on its bound the learner gets no
        gradient for it and can never bring it back -- the bound added for
        safety silently switches off learning on the axis it bounds. Measured:
        with a clip, q_v pinned at the ceiling for an entire lap while only q_c
        moved, and the episode returns were bimodal because the policy had lost
        an axis. A tanh keeps the same box and stays differentiable inside it.
        """
        if self.influence == "exact":
            h, imm, leak, D = self.cell.step(feat, need_D=True)
            # Exact RTRL: P <- D P + imm, keeping the coupling between neurons
            # that RFLO drops. Same recursion, full Jacobian.
            self.P = D @ self.P + imm
        else:
            h, imm, leak = self.cell.step(feat)
            self.P = leak[:, None] * self.P + imm
        mid, half = 0.5 * (self.hi + self.lo), 0.5 * (self.hi - self.lo)
        z = (self.theta0 - mid) / half + self.G @ h
        self._sq = 1.0 - np.tanh(z) ** 2          # d(theta)/dz, for the chain
        theta = mid + half * np.tanh(z)
        self._h = h
        return theta

    def grads(self, dQ_dtheta):
        """Chain the envelope gradient through the policy.

        ``dQ/dG = dQ/dtheta (x) h`` exactly, and ``dQ/d(cell params)`` is
        ``(dQ/dtheta . G) * P`` -- the RFLO approximation, which keeps only each
        neuron's own influence and drops the coupling between neurons.
        """
        g = np.asarray(dQ_dtheta, float)
        # Through the squash: d(theta)/dz = half * (1 - tanh^2 z).
        g = g * 0.5 * (self.hi - self.lo) * self._sq
        dG = np.outer(g, self._h)
        dcell = (g @ self.G)[:, None] * self.P
        return dG, dcell


#: Opponent classes, by *relative* speed. The distinction is behavioural, not
#: cosmetic: against a static object you must go around or park; against a slow
#: car a pass is worth taking; against an equal car it is expensive and rarely
#: available; against a faster car you are being caught, not catching, and
#: attempting a pass is simply wrong.
#:
#: **This is the distinction that needs the recurrence.** Catching and being
#: caught are the same instantaneous gap with opposite signs of closing rate,
#: and they call for opposite behaviour -- so no function of one frame can tell
#: them apart, however many features it is given.
OPPONENT_CLASSES = ("static", "slower", "equal", "faster")


def classify_opponent(v_ego: float, v_opp: float, eps: float = 0.25,
                      band: float = 0.15) -> int:
    """0 static, 1 slower, 2 equal, 3 faster -- from an *estimated* speed."""
    if v_opp < eps:
        return 0
    r = v_opp / max(v_ego, 1e-3)
    if r < 1.0 - band:
        return 1
    return 2 if r <= 1.0 + band else 3


def features(track, s5, opponents=(), v_max: float = 4.0, a_max: float = 4.0,
             a_lat_max: float = 6.0, preview=(0.5, 1.5, 3.0),
             opp_speed_est=None):
    """Physics-informed features, in units that mean something.

    A policy keyed on coordinates will not transfer between tracks or speeds, so
    every entry here is a ratio of two physical quantities and is dimensionless
    by construction. TODO item 2c.

    Ordered: curvature preview (3), speed fraction, lateral margin fraction,
    gap in braking distances, time-to-collision, pass availability, absolute
    gap, **sector one-hot (4)**, **track width in car widths**, and
    **opponent-class one-hot (4)**.

    The last three are what let the policy condition on the things a driver
    actually conditions on. The sector is the *named* one -- straight, long
    curve, 90-degree, hairpin -- classified by the corner's total heading
    change, because curvature at a point cannot separate a 90 from a 180
    (Section "What the labels can and cannot be"). The width matters because
    whether a pass fits is a question about the corridor, not only about the
    two cars. And the opponent class is graded by *relative* speed, because
    "someone is 2 m ahead" means opposite things depending on whether they are
    parked or pulling away.

    ``opp_speed_est`` is the tracker's estimate, not the opponent's true speed.
    Passing the truth here would remove the hidden state the recurrence exists
    for, so it is threaded in from
    :class:`~mpcc_tuning.opponents.ObstacleTracker` and is late and noisy on
    purpose.
    """
    x, y, psi, v, s = (float(q) for q in s5)
    f = [np.tanh(3.0 * track.curvature(track.wrap(s + d))) for d in preview]
    f.append(v / v_max)
    half = track.half_width - 0.12
    f.append(float(np.clip(track.lateral(x, y) / max(half, 1e-6), -1.5, 1.5)))

    # No opponent is a *state*, not a missing value: saturate the gap features
    # rather than pass zeros, which would read as "someone is right here".
    gap_brake, ttc, avail, gap_m = 1.0, 1.0, 0.0, 1.0
    if len(opponents):
        best = None
        for o in opponents:
            d = (o.s - s) % track.length
            if best is None or d < best[0]:
                best = (d, o)
        d, o = best
        if d < track.length / 2:
            brake = max(v * v / (2.0 * a_max), 1e-3)
            gap_brake = float(np.tanh(d / (3.0 * brake)))
            # An *absolute* gap as well, and it is not redundant. Braking
            # distance vanishes at low speed -- at 1.4 m/s it is 0.26 m, so
            # "within three braking distances" is 0.54 m, while the keep-out
            # holds the car 1.04 m away and its own effective radius is 0.39 m.
            # An engagement test on gap_brake therefore asks for a proximity
            # the safety constraint forbids: measured, it fired on 0 of 400
            # ticks at low aggression and 5 of 400 at neutral, which is why
            # that cell was bimodal. Engagement is a question about the
            # *geometry* of the two cars, so it is measured in car lengths.
            gap_m = float(np.tanh(d / (6.0 * ENGAGE_SCALE)))
            closing = v - o.speed
            ttc = float(np.tanh((d / closing) / 4.0)) if closing > 1e-3 else 1.0
            # Is the pass physically available: the lateral acceleration needed
            # to move one car-width across within the gap, against the limit.
            t_gap = d / max(closing, 1e-3)
            a_need = 2.0 * 0.30 / max(t_gap ** 2, 1e-6)
            avail = float(np.clip(1.0 - a_need / a_lat_max, -1.0, 1.0))
    f += [gap_brake, ttc, avail, gap_m]

    # Named sector ahead, one-hot. Read from the path at the preview distance,
    # so it is observed rather than inferred.
    sec = np.zeros(4)
    sec[track.sector(float(track.wrap(s + preview[1])))] = 1.0
    f += list(sec)

    # Corridor width in car widths: "can two cars fit" is a property of the
    # track, and it is the quantity that decides whether a pass exists at all.
    f.append(float(np.clip(track.half_width / 0.12, 0.0, 12.0) / 12.0))

    # Opponent class, one-hot, from the *estimate*.
    cls = np.zeros(4)
    if len(opponents):
        ve = v if v > 1e-3 else 1e-3
        vo = float(opp_speed_est) if opp_speed_est is not None else 0.0
        cls[classify_opponent(ve, vo)] = 1.0
    f += list(cls)
    return np.array(f, float)


#: Length scale for the engagement test, in metres -- roughly a car length. Not
#: a braking distance: see the note in :func:`features`.
ENGAGE_SCALE = 0.5

N_FEATURES = 18      # 9 + sector(4) + width(1) + opponent class(4)
THETA_LO = np.log(np.array([0.05, 0.05, 0.02, 1e-3, 1e-3, 1e-3]))
#: Ceiling on q_v at 2.0 -- measured: it saturates above ~2 on an empty track
#: and every attempted pass above it leaves the track with an opponent present.
THETA_HI = np.log(np.array([20.0, 200.0, 2.0, 10.0, 10.0, 10.0]))
assert len(WEIGHT_NAMES) == len(THETA_LO) == len(THETA_HI)


class PolicyTuner:
    """TD(lambda) on a *policy* over theta, instead of on theta itself.

    Identical outer loop to :class:`~mpcc_tuning.learner.QLambdaTuner` -- same
    TD error, same trace, same clipping, same RMS normalisation -- with one
    factor inserted. Where that tuner uses ``dQ/dtheta`` directly, this chains
    it through the policy to ``dQ/dphi``. Keeping the rest identical is what
    makes the comparison in ``experiments/ltc_behaviour.py`` a comparison of
    *parameterisations* and not of learning rules.
    """

    def __init__(self, mpcc, policy, gamma=0.98, lam=0.9, alpha=2e-3,
                 clip=1.0, delta_clip=1.0, explore=0.05, seed=0,
                 trust_region: float | None = None, theta_prior: float = 0.0):
        from mpcc_tuning.model import ACCEL_MAX, STEER_MAX
        self.mpcc, self.pol = mpcc, policy
        self.gamma, self.lam, self.alpha = gamma, lam, alpha
        self.clip, self.delta_clip, self.explore = clip, delta_clip, explore
        self._lim = np.array([STEER_MAX, ACCEL_MAX])
        self.rng = np.random.default_rng(seed)
        # A bound on the STEP, not on the output. Bounding the output only says
        # where theta may go; it does not say whether theta should keep going,
        # and the two are different requirements. Measured: with the output
        # bounded and nothing else, the policy walks to the bound and stays
        # there while the return collapses -- the same failure the global tuner
        # shows with an exact gradient.
        self.trust_region = trust_region
        # A weak pull back towards the initial weights, which is the cheapest
        # thing that makes "stop moving" an equilibrium rather than a place the
        # dynamics never reach.
        self.theta_prior = float(theta_prior)
        self._rms_G = self._rms_c = None
        self.reset()

    def reset(self):
        self.pol.reset()
        self.eG = np.zeros_like(self.pol.G)
        self.ec = np.zeros_like(self.pol.cell.p)
        self.prev = None
        self.stats = {}

    def _explore(self, u):
        if self.explore <= 0:
            return u
        u = np.asarray(u, float).copy()
        u[:2] += self.rng.normal(0.0, self.explore, 2) * self._lim
        u[:2] = np.clip(u[:2], -self._lim, self._lim)
        return u

    def _norm(self, g, which):
        a = np.abs(g)
        if which == "G":
            self._rms_G = a if self._rms_G is None else 0.99 * self._rms_G + 0.01 * a
            sc = np.maximum(self._rms_G, 1e-8)
        else:
            self._rms_c = a if self._rms_c is None else 0.99 * self._rms_c + 0.01 * a
            sc = np.maximum(self._rms_c, 1e-8)
        return np.clip(g / sc, -self.clip, self.clip)

    def act(self, feat, state5):
        """Emit theta for this tick, solve, and return ``(theta, action)``."""
        theta = self.pol.step(feat)
        out = self.mpcc.value(state5, theta)
        action = self._explore(out["u0"])
        q = self.mpcc.action_value(state5, theta, action, v_out=out)
        self._pending = (state5, theta, q)
        return theta, action

    def learn(self, reward, next_state5, next_feat, terminated):
        """One TD update, then emit the next tick's theta and action."""
        s, theta, q = self._pending
        gQ = self.mpcc.grad_theta(q, s, theta)
        if self.prev is not None:
            pg, pq = self.prev
            v_next = 0.0 if terminated else -q["value"]
            delta = float(np.clip(reward + self.gamma * v_next - (-pq),
                                  -self.delta_clip, self.delta_clip))
            dG, dc = pg
            self.eG = self.gamma * self.lam * self.eG + self._norm(-dG, "G")
            self.ec = self.gamma * self.lam * self.ec + self._norm(-dc, "c")
            dG = self.alpha * delta * self.eG
            dc = self.alpha * delta * self.ec
            if self.trust_region is not None:
                n = float(np.sqrt((dG ** 2).sum() + (dc ** 2).sum()))
                if n > self.trust_region:
                    f = self.trust_region / max(n, 1e-12)
                    dG, dc = dG * f, dc * f
            self.pol.G += dG
            self.pol.cell.p += dc
            if self.theta_prior > 0:
                self.pol.G *= (1.0 - self.alpha * self.theta_prior)
            self.pol.cell.clip()
            self.stats = {"delta": delta}
        self.prev = (self.pol.grads(gQ), q["value"])
        if terminated:
            return None, None
        return self.act(next_feat, next_state5)


def fixed_schedule(feat, theta0):
    """The hand-written control the gate requires beating.

    A lookup, not a learner: if an opponent is close and the pass is physically
    available, use the measured *aggressive overtake* weights; otherwise the
    *follow* ones. Both come from ``experiments/overtake_or_follow.py``, where
    the behaviour boundary is the ratio q_v/q_c and the safe pass is
    (q_v, q_c) = (2.0, 1.0) against a follow at (0.5, 10.0).

    It commits on a single frame and has no idea how fast it is closing, which
    is precisely the defect a recurrent policy is supposed to fix.
    """
    gap_brake, avail = feat[6], feat[7]
    th = np.array(theta0, float)
    if gap_brake < 0.6 and avail > 0.0:
        th[0], th[2] = np.log(1.0), np.log(2.0)     # q_c, q_v -- overtake
    else:
        th[0], th[2] = np.log(10.0), np.log(0.5)    # follow
    return th


# --------------------------------------------------------------------------
# Named behaviours
# --------------------------------------------------------------------------
#: The behaviours the weights are supposed to express, as (q_c, q_v) pairs at
#: neutral aggression. Read them against the two axes measured in
#: ``experiments/overtake_or_follow.py``: the *ratio* q_v/q_c decides whether
#: the car goes around or sits behind (15/15 cells, boundary at 1), and the
#: *magnitude* of q_v decides whether it survives doing so.
#:
#: So FOLLOW is not "slow" -- it is a ratio below 1. And OVERTAKE is not "fast"
#: -- it is a ratio above 1. Aggression is the separate axis.
BEHAVIOURS = {
    "follow":          dict(q_c=10.0, q_v=0.5),   # ratio 0.05 -- sits behind
    "overtake":        dict(q_c=1.0,  q_v=2.0),   # ratio 2.0  -- goes around
}

#: Aggression scales how hard a behaviour is pursued. It does **not** move the
#: behaviour across the ratio boundary, and an earlier version did: with
#: ``q_c / sqrt(g)`` and ``q_v * g`` the "cautious overtake" cell landed at
#: q_v/q_c = 0.71, below the measured boundary of 1, so it *followed*.
#: Measured, it was byte-identical to ``stay_behind``: 33.6 m and 0.00 passes.
#: A cautious overtake that does not overtake is not a cautious overtake.
AGGRESSION = {"cautious": 0.5, "neutral": 1.0, "aggressive": 2.0}

#: How far a behaviour is allowed to sit from the boundary, as a ratio. Follow
#: must stay below 1 and overtake above it, whatever the aggression.
_RATIO_MARGIN = 1.35


def behaviour_theta(name, aggression="neutral", theta0=None):
    """A named behaviour at a named aggression, as log weights.

    ``theta0`` supplies the four weights this does not touch (q_l, r_d, r_a,
    r_dv); only q_c and q_v carry behaviour.

    Aggression scales q_v, and q_c is then set so the *ratio* stays on the
    behaviour's own side of 1 -- the boundary measured in
    ``experiments/overtake_or_follow.py``. Crossing it is a change of
    behaviour, not a change of intensity, and aggression must not do that.
    """
    th = np.array(theta0, float).copy()
    b = BEHAVIOURS[name]
    g = AGGRESSION[aggression] if isinstance(aggression, str) else float(aggression)
    q_v = float(np.clip(b["q_v"] * g, np.exp(THETA_LO[2]), np.exp(THETA_HI[2])))
    ratio = b["q_v"] / b["q_c"]
    if ratio > 1.0:
        # An overtaking behaviour. Aggression also lowers q_c, because q_v is
        # capped by the *measured* ceiling (above ~2 every attempted pass left
        # the track) and would otherwise saturate: without this, "aggressive
        # overtake" and "neutral overtake" are the same weights. So above
        # neutral, aggression is expressed as caring less about the racing line
        # rather than as wanting more speed -- which is what the ceiling leaves
        # available, and is arguably the more honest reading of aggression.
        q_c = min(b["q_c"] / g, q_v / _RATIO_MARGIN)
    else:                                # a following behaviour
        q_c = max(b["q_c"], q_v * _RATIO_MARGIN)
    th[0], th[2] = np.log(q_c), np.log(q_v)
    return np.clip(th, THETA_LO, THETA_HI)


#: The three overtaking postures a driver actually chooses between. The middle
#: one is the interesting case: it is the only one that consults whether the
#: pass is *physically available* rather than merely desirable.
POSTURES = ("stay_behind", "overtake_when_safe", "always_try")


def pass_is_available(feat, track=None, s=None, v=None, a_lat_max: float = 6.0,
                      grip: float = 1.0, margin: float = 0.35):
    """Is the pass physically available, against the **friction ellipse**?

    An earlier version tested only the lateral acceleration needed to change
    lane, ``2 w / t_gap**2``, against the full grip limit. That is satisfied
    almost always, and the consequence was measured: ``overtake_when_safe`` and
    ``always_try`` came out **identical in all nine cells** -- same distance,
    same passes, same switch count. The gate the whole safety argument rests on
    was doing nothing.

    Two things were missing. The car is already **using grip to corner**, so the
    lateral acceleration available for a lane change is what is left over, not
    the whole budget: ``a_free**2 = (a_lat_max mu)**2 - (v**2 kappa)**2``. And a
    pass needs a **closing speed** at all -- with none, ``t_gap`` is unbounded
    and the test passes trivially while the manoeuvre never completes.
    """
    ttc, lane, gap_m = float(feat[6]), float(feat[7]), float(feat[8])
    if ttc >= 1.0 - 1e-9:            # no closing speed: nothing to time the pass by
        return False
    budget = a_lat_max * grip
    if track is not None and s is not None and v is not None:
        a_corner = abs(v * v * track.curvature(track.wrap(s)))
        budget = float(np.sqrt(max(budget ** 2 - a_corner ** 2, 0.0)))
    # ``lane`` is 1 - a_need/a_lat_max, so a_need = (1 - lane) * a_lat_max.
    a_need = (1.0 - lane) * a_lat_max
    return bool(a_need < (1.0 - margin) * budget and gap_m < 0.85)


def posture_theta(posture, feat, aggression="neutral", theta0=None,
                  gap_close=0.6, track=None, s=None, v=None, is_dynamic=True):
    """Choose weights from a posture and the current situation.

    ``feat`` is :func:`features`' output; index 5 is the gap in braking
    distances and index 7 is whether the lateral acceleration the pass needs is
    inside the grip limit.

    ``is_dynamic`` says whether the obstacle has been *observed* to move. It is
    an estimate from :class:`~mpcc_tuning.opponents.ObstacleTracker`, not ground
    truth, because on a vehicle it is one too.

    ``overtake_when_safe`` is the posture the safety argument is about: it
    commits only when the opponent is close enough to matter **and** the
    manoeuvre is available, and follows otherwise. ``always_try`` drops the
    availability test, which is what a driver with more ambition than grip does
    and is included so that the cost of dropping it can be measured rather than
    asserted.
    """
    # Index 8, the absolute gap -- not index 5, the braking-distance gap.
    close = float(feat[8]) < gap_close
    if posture == "stay_behind":
        # Following is only a behaviour against something that is *going
        # somewhere*. Behind a static obstacle it is not caution, it is
        # stopping -- so the posture falls through to passing. That distinction
        # is what makes "stay behind" a choice rather than a refusal, and it
        # needs the obstacle *classified*, which cannot be done from a single
        # frame: a stopped car and a slow one are identical in one observation.
        # See mpcc_tuning.opponents.ObstacleTracker.
        want = close and not is_dynamic
    elif posture == "always_try":
        want = close
    else:
        want = close and pass_is_available(feat, track=track, s=s, v=v)
    return behaviour_theta("overtake" if want else "follow", aggression, theta0)
