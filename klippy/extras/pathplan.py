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
    """

    def __init__(self, a_of_v=None, a_const=None, v_ceil=1e9, dv_slice=25.0):
        self._a_of_v = a_of_v
        self.a_const = a_const
        self.v_ceil = v_ceil
        self.dv_slice = dv_slice

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


def emit_profile(vs, vc, ve, move_d, cons):
    """Emit ONE monotonic constant-accel segment list for a single move.

    Each segment is (accel_t, cruise_t, decel_t, start_v, cruise_v, accel, dist)
    -- the tuple the toolhead/extruder trapq consume. Invariant-guaranteed by
    construction: non-negative times/speeds, no decel past zero, per-segment
    trapq-implied distance == dist, inter-segment velocity continuity, and
    sum(dist) == move_d. (This is the jl-stepguard check made structural.)
    """
    if move_d <= 0.0:
        return []
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
