# Unified phase-plane motion planning (topp-ra-v3)
#
# One reachability pass over a move batch that folds every velocity/acceleration
# constraint into a single feasible boundary-velocity solution, plus one
# invariant-guaranteed emitter. Pure (only `math`) and toolhead-decoupled so it
# is unit-testable in isolation; the toolhead adapter supplies the constraint
# oracle and per-move caps.
#
# See docs/topp-ra-v3-design.md. This replaces the distributed
# calc_junction/flush/plan_move + jerk-slice + input_shaper composition on v2.
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
import math


class Constraints:
    """Phase-plane limits.

    a_of_v(v) -> max |acceleration| (mm/s^2) available at speed v (mm/s), or
                 None if the curve does not cover v. `a_const` is the constant
                 fallback used when a_of_v is None (stock behaviour / no curve).
    v_ceil     -> hard speed ceiling (mm/s), e.g. the curve's top speed or the
                 machine max_velocity; the reachability integration stops here.
    dv_slice   -> velocity step (mm/s) for the a_max(v) reachability quadrature.
    max_jerk   -> max |da/dt| (mm/s^3). None/0 = no jerk limiting (sharp
                 constant-accel ladders). When set, emit_profile ramps the
                 acceleration (Stage 3, approach A) so a(t) is continuous.
    jerk_dt    -> integration time step (s) for the jerk-limited ramp.
    """

    def __init__(self, a_of_v=None, a_const=None, v_ceil=1e9, dv_slice=25.0,
                 max_jerk=None, jerk_dt=0.001):
        self._a_of_v = a_of_v
        self.a_const = a_const
        self.v_ceil = v_ceil
        self.dv_slice = dv_slice
        self.max_jerk = max_jerk
        self.jerk_dt = jerk_dt

    def a_max(self, v):
        if self._a_of_v is not None:
            a = self._a_of_v(v)
            if a is not None and a > 0.0:
                return a
        return self.a_const


def reach_v2(u0, dist, cons):
    # Max u = v^2 reachable from u0 over path-distance `dist`, riding
    # |du/ds| = 2*a_max(sqrt(u)). Direction-symmetric: serves forward accel and
    # backward decel identically.
    if dist <= 0.0:
        return u0
    dv = cons.dv_slice
    v_ceil = cons.v_ceil
    v = math.sqrt(max(u0, 0.0))
    s = 0.0
    while s < dist and v < v_ceil:
        v_next = min(v + dv, v_ceil)
        a = cons.a_max(v_next)
        if a is None or a <= 0.0:
            break
        ds = (v_next * v_next - v * v) / (2.0 * a)
        if s + ds >= dist:
            return v * v + 2.0 * a * (dist - s)
        v, s = v_next, s + ds
    return v * v


def dist_change(v_lo, v_hi, cons):
    # Path distance to change speed v_lo -> v_hi (v_hi >= v_lo) under a_max(v),
    # upper-edge (conservative) accel per slice.
    if v_hi <= v_lo:
        return 0.0
    dv = cons.dv_slice
    v = v_lo
    d = 0.0
    while v < v_hi - 1e-12:
        v_next = min(v + dv, v_hi)
        a = cons.a_max(v_next)
        if a is None or a <= 0.0:
            return d
        d += (v_next * v_next - v * v) / (2.0 * a)
        v = v_next
    return d


def peak_velocity(start_v, end_v, dist, cons):
    # Highest velocity a move of length `dist` can peak at (accelerate up then
    # back down to end_v) under a_max(v). Returns max(start_v,end_v) floor when
    # the move is too short to peak higher (triangle collapses to a ramp).
    lo = max(start_v, end_v)
    hi = cons.v_ceil
    if dist_change(start_v, lo, cons) + dist_change(end_v, lo, cons) >= dist:
        return lo
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        need = dist_change(start_v, mid, cons) + dist_change(end_v, mid, cons)
        if need > dist:
            hi = mid
        else:
            lo = mid
    return lo


def plan_batch(move_d, v_cap, v_junction, cons):
    """Solve feasible boundary velocities for a batch.

    move_d[i]      -> path length of move i (mm), i in [0, n).
    v_cap[i]       -> cruise speed ceiling of move i (mm/s): min of requested
                      feedrate, machine max, melt-flow cap, kinematic caps.
    v_junction[k]  -> speed ceiling at boundary k (mm/s), len n+1. v_junction[0]
                      and v_junction[n] are the batch entry/exit speeds (0.0 for
                      a full start/stop; higher when chaining flushes).
    Returns (vs, vc, ve) lists (len n): start, cruise, end speed per move,
    guaranteed jerk-free-agnostic feasible under a_max(v).
    """
    n = len(move_d)
    if n == 0:
        return [], [], []
    # Boundary speed ceilings: junction cap AND both adjacent cruise caps.
    vb = [0.0] * (n + 1)
    for k in range(n + 1):
        cap = v_junction[k]
        if k > 0:
            cap = min(cap, v_cap[k - 1])
        if k < n:
            cap = min(cap, v_cap[k])
        vb[k] = cap
    # Backward pass: a boundary can be no faster than what lets us decelerate to
    # the next boundary over the intervening move.
    for i in range(n - 1, -1, -1):
        reach = math.sqrt(reach_v2(vb[i + 1] * vb[i + 1], move_d[i], cons))
        vb[i] = min(vb[i], reach)
    # Forward pass: nor faster than what we can accelerate to from the previous.
    for i in range(n):
        reach = math.sqrt(reach_v2(vb[i] * vb[i], move_d[i], cons))
        vb[i + 1] = min(vb[i + 1], reach)
    vs = [vb[i] for i in range(n)]
    ve = [vb[i + 1] for i in range(n)]
    vc = [min(v_cap[i], peak_velocity(vs[i], ve[i], move_d[i], cons))
          for i in range(n)]
    return vs, vc, ve


def _ramp_up_jerk(v0, v1, cons, collect=True):
    # Jerk-limited acceleration ramp taking velocity v0 -> v1 (v1 >= v0). The
    # acceleration starts at ~0, rises toward a_max(v) bounded by |da/dt| <=
    # max_jerk, then falls back to ~0 landing on v1 -- so a(t) is continuous
    # (no hard accel step). Integrated at fixed jerk_dt; the last slice lands
    # exactly on v1 (velocity continuity is exact). Returns (slices, total_d);
    # slices is None when collect=False (distance-only, for peak bisection).
    #
    # The taper is enforced by a "brake" cap a_brake = sqrt(2*J*(v1-v)): along
    # it da/dt = -J exactly, so following min(a_curve, a_brake, a+J*dt)
    # guarantees a bounded rise (a+J*dt) and a jerk-feasible fall to 0 at v1.
    J = cons.max_jerk
    dt0 = cons.jerk_dt
    slices = [] if collect else None
    total = 0.0
    if v1 <= v0 + 1e-12 or J is None or J <= 0.0:
        return slices, total
    v = v0
    a = 0.0
    guard = 0
    while v < v1 - 1e-9:
        guard += 1
        if guard > 500000:
            break
        rem = v1 - v
        a_curve = cons.a_max(v)
        if a_curve is None or a_curve <= 0.0:
            a_curve = cons.a_const if cons.a_const else 1e30
        a_brake = math.sqrt(2.0 * J * rem)
        a_new = min(a_curve, a_brake, a + J * dt0)
        if a_new <= 0.0:
            a_new = min(a_curve, a_brake)
            if a_new <= 0.0:
                break
        v_next = v + a_new * dt0
        this_dt = dt0
        if v_next >= v1:
            v_next = v1
            this_dt = (v1 - v) / a_new
        dist = 0.5 * (v + v_next) * this_dt
        if collect:
            slices.append((this_dt, 0.0, 0.0, v, v_next, a_new, dist))
        total += dist
        v, a = v_next, a_new
    return slices, total


def _decel_from_accel(acc_slices):
    # Time-reverse an increasing-velocity jerk ramp into a decel slice list:
    # an accel slice (dt,0,0, v_lo, v_hi, a, dist) becomes decel
    # (0,0,dt, v_hi, v_hi, a, dist) (velocity v_hi -> v_lo). Reversed order so
    # the chain runs vc -> ve and stays velocity-continuous.
    dec = []
    for (at, ct, dt, sv, cv, a, dist) in reversed(acc_slices):
        dec.append((0.0, 0.0, at, cv, cv, a, dist))
    return dec


def _peak_velocity_jerk(vs, ve, move_d, cons):
    # Highest cruise a jerk-limited move can reach: bisection on the ramp
    # distance (accel vs->vp plus decel vp->ve). Returns max(vs,ve) when even
    # that connecting ramp overfills move_d (caller then falls back to sharp).
    lo = max(vs, ve)
    hi = cons.v_ceil
    need_lo = (_ramp_up_jerk(vs, lo, cons, collect=False)[1]
               + _ramp_up_jerk(ve, lo, cons, collect=False)[1])
    if need_lo >= move_d:
        return lo
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        need = (_ramp_up_jerk(vs, mid, cons, collect=False)[1]
                + _ramp_up_jerk(ve, mid, cons, collect=False)[1])
        if need > move_d:
            hi = mid
        else:
            lo = mid
    return lo


def _emit_jerk(vs, vc, ve, move_d, cons):
    # Jerk-limited profile: ramp vs->vc, cruise, ramp vc->ve. Returns None when
    # the move is too short to jerk-limit even at the connecting peak, so the
    # caller falls back to the sharp emit (which uses full a_max and always fits
    # what the lookahead already approved).
    vc = max(vc, vs, ve)
    acc, d_acc = _ramp_up_jerk(vs, vc, cons)
    dec_acc, d_dec = _ramp_up_jerk(ve, vc, cons)
    cruise_d = move_d - d_acc - d_dec
    if cruise_d < -1e-9:
        vc = max(_peak_velocity_jerk(vs, ve, move_d, cons), vs, ve)
        acc, d_acc = _ramp_up_jerk(vs, vc, cons)
        dec_acc, d_dec = _ramp_up_jerk(ve, vc, cons)
        cruise_d = move_d - d_acc - d_dec
        if cruise_d < -1e-6 * max(1.0, move_d):
            return None
        cruise_d = max(0.0, cruise_d)
    segs = list(acc)
    if cruise_d > 1e-9 and vc > 1e-9:
        segs.append((0.0, cruise_d / vc, 0.0, vc, vc, 0.0, cruise_d))
    segs.extend(_decel_from_accel(dec_acc))
    return segs


def emit_profile(vs, vc, ve, move_d, cons):
    """Emit ONE monotonic constant-accel segment list for a single move.

    Each segment is (accel_t, cruise_t, decel_t, start_v, cruise_v, accel, dist)
    -- the tuple the toolhead/extruder trapq consume. Invariant-guaranteed by
    construction: non-negative times/speeds, no decel past zero, per-segment
    trapq-implied distance == dist, inter-segment velocity continuity, and
    sum(dist) == move_d. (This is the jl-stepguard check made structural.)

    When cons.max_jerk is set, the acceleration is ramped (Stage 3, approach A)
    so a(t) is continuous; on a move too short to jerk-limit, it falls back to
    the sharp constant-accel profile (which always fits what the a_max
    lookahead approved).
    """
    if move_d <= 0.0:
        return []
    if getattr(cons, "max_jerk", None):
        segs = _emit_jerk(vs, vc, ve, move_d, cons)
        if segs is not None:
            return segs
        # else: jerk-infeasible for this short move -> sharp fallback below
    vc = max(vc, vs, ve)
    d_acc = dist_change(vs, vc, cons)
    d_dec = dist_change(ve, vc, cons)
    cruise_d = move_d - d_acc - d_dec
    if cruise_d < -1e-9:
        vc = peak_velocity(vs, ve, move_d, cons)
        vc = max(vc, vs, ve)
        d_acc = dist_change(vs, vc, cons)
        d_dec = dist_change(ve, vc, cons)
        cruise_d = max(0.0, move_d - d_acc - d_dec)
    dv = cons.dv_slice
    segs = []
    # Accel ladder vs -> vc (represented in the accel phase: cruise_v = v_next)
    v = vs
    while v < vc - 1e-9:
        v_next = min(v + dv, vc)
        a = cons.a_max(v_next)
        if a is None or a <= 0.0:
            break
        dt = (v_next - v) / a
        dist = (v_next * v_next - v * v) / (2.0 * a)
        segs.append((dt, 0.0, 0.0, v, v_next, a, dist))
        v = v_next
    # Cruise (single constant-velocity segment)
    if cruise_d > 1e-9 and vc > 1e-9:
        segs.append((0.0, cruise_d / vc, 0.0, vc, vc, 0.0, cruise_d))
    # Decel ladder vc -> ve (represented in the decel phase: cruise_v = v_hi)
    ladder = []
    v = ve
    while v < vc - 1e-9:
        v_next = min(v + dv, vc)
        ladder.append((v, v_next))
        v = v_next
    for v_lo, v_hi in reversed(ladder):
        a = cons.a_max(v_hi)
        if a is None or a <= 0.0:
            break
        dt = (v_hi - v_lo) / a
        dist = (v_hi * v_hi - v_lo * v_lo) / (2.0 * a)
        segs.append((0.0, 0.0, dt, v_hi, v_hi, a, dist))
    return segs
