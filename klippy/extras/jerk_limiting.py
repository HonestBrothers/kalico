# Jerk-limited motion: smooth acceleration transitions and corner rounding.
#
# This is an OPTIONAL, opt-in layer on top of Kalico's constant-acceleration
# trapezoidal motion. It is gated entirely behind the [jerk_limiting] config
# section and is OFF by default; when disabled the motion pipeline behaves
# exactly as stock.
#
# Three independently toggleable phases:
#   smooth_ramps    (Phase 1) -- replace each constant-accel ramp with a
#                                jerk-limited (S-curve) ramp.
#   blend_junctions (Phase 2) -- carry acceleration across move boundaries so a
#                                continuous acceleration is not forced back to
#                                zero at every gcode move seam.
#   round_corners   (Phase 3) -- replace sharp corners with a curvature-
#                                continuous blend (cubic B-spline) within a
#                                deviation tolerance.
#
# EXPERIMENTAL. Validated by unit tests (test/test_jerk_limiting.py); not yet
# validated on hardware.
#
# Copyright (C) 2026
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math

# ---------------------------------------------------------------------------
# Pure math (no klippy dependencies) -- unit-tested directly.
#
# Symbols: v velocity (mm/s), a acceleration (mm/s^2), J jerk (mm/s^3),
# A acceleration cap (mm/s^2), d path distance (mm), u = v^2.
# ---------------------------------------------------------------------------


def ramp_time(dv, A, J):
    # Duration of a symmetric jerk-limited velocity change |dv| with peak accel
    # capped at A and jerk |J|. The accel profile ramps 0 -> a_peak -> 0; for
    # any such symmetric profile the mean velocity is (v0+v1)/2 exactly.
    dv = abs(dv)
    if dv <= 0.0:
        return 0.0
    if dv <= A * A / J:
        # triangular accel profile (cap never reached)
        return 2.0 * math.sqrt(dv / J)
    # trapezoidal accel profile
    return A / J + dv / A


def dist_jerk(v0, v1, A, J):
    # Path distance to change velocity v0 -> v1 under a jerk-limited ramp.
    return 0.5 * (v0 + v1) * ramp_time(v1 - v0, A, J)


def reach_v2(u0, dist, A, J, vmax):
    # Max reachable u = v^2 accelerating from u0 = v0^2 over `dist`, jerk
    # limited. Symmetric, so it also bounds backward deceleration. Monotonic in
    # the target velocity, solved by bisection.
    if dist <= 0.0:
        return u0
    v0 = math.sqrt(max(u0, 0.0))
    if v0 >= vmax:
        return vmax * vmax
    if dist_jerk(v0, vmax, A, J) <= dist:
        return vmax * vmax
    lo, hi = v0, vmax
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if dist_jerk(v0, mid, A, J) <= dist:
            lo = mid
        else:
            hi = mid
    return lo * lo


def is_travel_move(move):
    # A travel move is a kinematic (XYZ) move that deposits no filament. The
    # codebase treats bool(move.axes_d[3]) as "is extruding" (see toolhead.py),
    # so a kinematic move with zero extruder delta is a pure travel. Jerk
    # limiting exists to keep extrusion smooth (pressure advance / stepcompress);
    # there is nothing to smooth on a travel, and travels want full speed.
    return move.is_kinematic_move and not move.axes_d[3]


def _accel_profile(dv, A, J):
    # Returns (a_peak, t1, t2): jerk-ramp time t1, const-accel time t2.
    dv = abs(dv)
    if dv <= A * A / J:
        a_peak = math.sqrt(J * dv)
        return a_peak, a_peak / J, 0.0
    return A, A / J, (dv - A * A / J) / A


def _vel_integral(t, t1, t2, a_peak, J):
    # Integral of |a| from 0..t for the 0->a_peak->0 profile (t1 ramp, t2 hold).
    if t <= t1:
        return 0.5 * J * t * t
    if t <= t1 + t2:
        return 0.5 * a_peak * t1 + a_peak * (t - t1)
    td = t - t1 - t2
    return (0.5 * a_peak * t1 + a_peak * t2
            + a_peak * td - 0.5 * J * td * td)


def ramp_slices(v0, v1, A, J, dt, max_slices=400):
    # Emit a jerk-limited ramp v0 -> v1 as constant-accel micro-slices. Each
    # slice is (accel_t, cruise_t, decel_t, start_v, cruise_v, accel, dist).
    # Velocities at slice boundaries come from the analytic profile so the ramp
    # hits v1 exactly and per-slice distances are consistent.
    dv = v1 - v0
    if abs(dv) < 1e-12:
        return []
    sgn = 1.0 if dv > 0.0 else -1.0
    a_peak, t1, t2 = _accel_profile(dv, A, J)
    T = 2.0 * t1 + t2
    n = max(2, min(max_slices, int(math.ceil(T / dt))))
    h = T / n
    slices = []
    v_prev = v0
    for i in range(1, n + 1):
        t = i * h
        adv = _vel_integral(t, t1, t2, a_peak, J) if i < n else abs(dv)
        v_cur = v0 + sgn * adv
        a = abs(v_cur - v_prev) / h
        dist = 0.5 * (v_prev + v_cur) * h
        if sgn > 0.0:
            slices.append((h, 0.0, 0.0, v_prev, v_cur, a, dist))
        else:
            slices.append((0.0, 0.0, h, v_prev, v_prev, a, dist))
        v_prev = v_cur
    return slices


def peak_velocity(vs, ve, dist, A, J, vmax):
    # Highest cruise velocity a move of length `dist` can peak at (accelerate
    # vs -> vp, decelerate vp -> ve) under the jerk limit. None if too short to
    # even connect vs -> ve.
    lo = max(vs, ve)
    if dist_jerk(vs, lo, A, J) + dist_jerk(ve, lo, A, J) > dist + 1e-9:
        return None
    hi = vmax
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        need = dist_jerk(vs, mid, A, J) + dist_jerk(ve, mid, A, J)
        if need > dist:
            hi = mid
        else:
            lo = mid
    return lo


def plan_segments(vs, vc, ve, move_d, A, J, dt, entry_accel=0.0):
    # Build the jerk-limited slice list for one move given its planned
    # start/cruise/end velocities. `entry_accel` (Phase 2) is reserved for
    # carrying acceleration in from the previous move; 0 means start at rest
    # acceleration (the Phase 1 behaviour). Returns a slice list or None to
    # fall back to stock single-trapezoid emission.
    if move_d <= 0.0:
        return None
    d_acc = dist_jerk(vs, vc, A, J)
    d_dec = dist_jerk(ve, vc, A, J)
    cruise_d = move_d - d_acc - d_dec
    if cruise_d < -1e-9:
        vp = peak_velocity(vs, ve, move_d, A, J, max(vc, vs, ve))
        if vp is None:
            return None
        vc = vp
        d_acc = dist_jerk(vs, vc, A, J)
        d_dec = dist_jerk(ve, vc, A, J)
        cruise_d = max(0.0, move_d - d_acc - d_dec)
    segs = ramp_slices(vs, vc, A, J, dt)
    if cruise_d > 1e-9 and vc > 1e-9:
        segs.append((0.0, cruise_d / vc, 0.0, vc, vc, 0.0, cruise_d))
    segs.extend(ramp_slices(vc, ve, A, J, dt))
    return segs if segs else None


def split_slice(s, d0):
    # Split one constant-accel slice at sub-distance d0 (0 < d0 < dist) into two
    # slices with continuous velocity. s = (at, ct, dct, sv, cv, a, dist).
    at, ct, dct, sv, cv, a, dist = s
    if ct > 0.0:  # cruise
        t0 = d0 / sv
        return ((0.0, t0, 0.0, sv, sv, 0.0, d0),
                (0.0, ct - t0, 0.0, sv, sv, 0.0, dist - d0))
    if at > 0.0:  # accelerating sv -> cv
        vmid = math.sqrt(max(sv * sv + 2.0 * a * d0, 0.0))
        t0 = (vmid - sv) / a
        return ((t0, 0.0, 0.0, sv, vmid, a, d0),
                (at - t0, 0.0, 0.0, vmid, cv, a, dist - d0))
    # decelerating sv -> sv - a*dct
    vmid = math.sqrt(max(sv * sv - 2.0 * a * d0, 0.0))
    t0 = (sv - vmid) / a
    return ((0.0, 0.0, t0, sv, sv, a, d0),
            (0.0, 0.0, dct - t0, vmid, vmid, a, dist - d0))


def distribute_slices(slices, dlist):
    # Distribute a flat slice list across consecutive moves with path lengths
    # `dlist`, splitting slices that straddle a move boundary. Returns a list of
    # per-move slice lists. Total slice distance must equal sum(dlist).
    out = [[] for _ in dlist]
    k = 0
    rem = dlist[0] if dlist else 0.0
    for s in slices:
        while True:
            sd = s[6]
            if sd <= rem + 1e-9 or k >= len(dlist) - 1:
                out[k].append(s)
                rem -= sd
                break
            first, second = split_slice(s, rem)
            out[k].append(first)
            k += 1
            rem = dlist[k]
            s = second
        while rem <= 1e-9 and k < len(dlist) - 1:
            k += 1
            rem = dlist[k]
    return out


# --- Phase 3 corner geometry: cubic B-spline subdivision (N-dimensional) ----
def _avg(p, q, wp, wq, s):
    return tuple((wp * p[i] + wq * q[i]) / s for i in range(len(p)))


def _subdivide_cubic(pts, rounds):
    # Cubic B-spline subdivision (vertex mask (1,6,1)/8, edge mask (4,4)/8).
    # Dimension-agnostic. Repeat-endpoint padding keeps the open-curve ends
    # near the originals.
    for _ in range(rounds):
        n = len(pts)
        padded = [pts[0]] + list(pts) + [pts[-1]]
        out = []
        for i in range(1, n + 1):
            pm, pc, pn = padded[i - 1], padded[i], padded[i + 1]
            out.append(tuple((pm[d] + 6.0 * pc[d] + pn[d]) / 8.0
                             for d in range(len(pc))))
            out.append(tuple(0.5 * (pc[d] + pn[d]) for d in range(len(pc))))
        pts = out
    return pts


def corner_blend(a, vertex, b, rounds=3):
    # Round the corner a -> vertex -> b with a cubic B-spline through the three
    # control points (any dimension). Returns interior blend points.
    return _subdivide_cubic([tuple(a), tuple(vertex), tuple(b)], rounds)[1:-1]


def _seg_dist(p, s0, s1):
    ll = sum((s1[d] - s0[d]) ** 2 for d in range(len(p)))
    if ll <= 1e-18:
        return math.sqrt(sum((p[d] - s0[d]) ** 2 for d in range(len(p))))
    t = sum((p[d] - s0[d]) * (s1[d] - s0[d]) for d in range(len(p))) / ll
    t = max(0.0, min(1.0, t))
    return math.sqrt(sum((p[d] - (s0[d] + t * (s1[d] - s0[d]))) ** 2
                         for d in range(len(p))))


def max_deviation(points, a, vertex, b):
    # Max distance from blend `points` to the original corner polyline a-V-b.
    worst = 0.0
    for p in points:
        worst = max(worst, min(_seg_dist(p, a, vertex), _seg_dist(p, vertex, b)))
    return worst


def _unit(p, q):
    d = [q[i] - p[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    if n <= 1e-12:
        return None
    return [x / n for x in d]


def _decimate(pts, max_dev, min_len):
    # Adaptive (Douglas-Peucker) resampling of a dense blend curve: keep only
    # the vertices needed to stay within max_dev of the dense curve -- i.e.
    # refine where discrete curvature is high -- then enforce a hard minimum
    # segment length so no sub-step micro-moves are emitted. Replaces fixed
    # uniform subdivision, which over-samples shallow corners ~10-20x and is
    # what produced the un-rampable, pressure-advance-spiking micro-segments.
    n = len(pts)
    if n <= 2:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        worst, wk = -1.0, -1
        for k in range(i + 1, j):
            d = _seg_dist(pts[k], pts[i], pts[j])
            if d > worst:
                worst, wk = d, k
        if worst > max_dev:
            keep[wk] = True
            stack.append((i, wk))
            stack.append((wk, j))
    kept = [pts[k] for k in range(n) if keep[k]]
    # Min-length floor: drop interior vertices closer than min_len to the last
    # emitted one (endpoints always kept). Very sharp corners may then exceed
    # max_dev slightly -- a deliberate trade vs. emitting micro-segments; the
    # trim shrink in round_corner already bounds the curve's own deviation.
    out = [kept[0]]
    for k in range(1, len(kept) - 1):
        dx = [out[-1][d] - kept[k][d] for d in range(3)]
        if math.sqrt(sum(x * x for x in dx)) >= min_len:
            out.append(kept[k])
    out.append(kept[-1])
    return out


def corner_blend_adaptive(a, vertex, b, max_dev, min_len, rounds=3):
    # Same limit curve as corner_blend, but sampled with only as many points as
    # the turn needs (point count scales with turn angle, not a fixed depth).
    dense = [tuple(a)] + corner_blend(a, vertex, b, rounds) + [tuple(b)]
    return _decimate(dense, max_dev, min_len)[1:-1]


# ---------------------------------------------------------------------------
# Klipper integration
# ---------------------------------------------------------------------------
class JerkLimiting:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.enabled = config.getboolean("enabled", False)
        # Phase toggles
        self.smooth_ramps = config.getboolean("smooth_ramps", True)
        self.blend_junctions = config.getboolean("blend_junctions", True)
        self.round_corners = config.getboolean("round_corners", False)
        # Parameters
        self.max_jerk = config.getfloat("max_jerk", 100000.0, above=0.0)
        # Per-axis jerk ceilings (mm/s^3). X and Y have different moving mass and
        # resonance, so each gets its own limit; each defaults to the scalar
        # max_jerk. A move's jerk ceiling is set by its most binding axis (see
        # _move_jerk), exactly as limited_cartesian derives per-move accel from
        # per-axis caps. Z defaults to max_jerk and only binds pure-Z moves.
        self.max_jerk_x = config.getfloat(
            "max_jerk_x", self.max_jerk, above=0.0
        )
        self.max_jerk_y = config.getfloat(
            "max_jerk_y", self.max_jerk, above=0.0
        )
        self.max_jerk_z = config.getfloat(
            "max_jerk_z", self.max_jerk, above=0.0
        )
        self.resolution = config.getfloat("resolution", 0.002, above=0.0)
        self.corner_max_deviation = config.getfloat(
            "corner_max_deviation", 0.05, above=0.0
        )
        self.corner_min_angle = config.getfloat(
            "corner_min_angle", 5.0, minval=0.0, maxval=179.0
        )
        self.corner_blend_ratio = config.getfloat(
            "corner_blend_ratio", 0.25, above=0.0, maxval=0.5
        )
        # Adaptive blend: hard floor on emitted blend segment length so the
        # corner is never sampled into sub-step micro-moves.
        self.corner_min_seg_len = config.getfloat(
            "corner_min_seg_len", 0.1, above=0.0
        )
        # Per-corner jerk allowance (mm/s^3). A sub-mm blend segment cannot ramp
        # a velocity change at max_jerk, so any residual decel/accel left in the
        # blend (after the velocity cap pushes the bulk into the straight legs)
        # would fall back to constant max-accel and, through pressure advance,
        # spike the extruder. Permitting higher jerk on blend moves lets that
        # residual ramp at near-minimum accel instead. 0 -> use max_jerk.
        self.corner_max_jerk = config.getfloat(
            "corner_max_jerk", 0.0, minval=0.0
        )
        # Auto-derive the scalar max_jerk from the input shaper's measured
        # resonant frequency. A jerk-limited ramp lasts T_j = a/J, with its
        # first spectral null near 1/T_j; keeping energy out of the lowest mode
        # f_n wants T_j >= 1/f_n, i.e. J <= a * f_n. So set
        #   max_jerk = auto_jerk_ratio * max_accel * f_n   (ratio <= 1 = margin)
        # from the lowest measured axis frequency (most excitation-prone),
        # replacing the folklore "max_jerk ~ 20-50 * max_accel" with a value
        # tied to the machine's actual resonance. Falls back to the configured
        # max_jerk when no shaper frequency is available; recompute after a
        # fresh SHAPER_CALIBRATE with SET_JERK_LIMIT AUTO=1.
        self.auto_jerk = config.getboolean("auto_jerk", False)
        self.auto_jerk_ratio = config.getfloat(
            "auto_jerk_ratio", 1.0, above=0.0
        )
        self.toolhead = None
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )
        self.printer.register_event_handler(
            "klippy:ready", self._handle_ready
        )
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SET_JERK_LIMIT", self.cmd_SET_JERK_LIMIT,
            desc=self.cmd_SET_JERK_LIMIT_help,
        )
        gcode.register_command(
            "JERK_LIMIT_STATUS", self.cmd_JERK_LIMIT_STATUS,
            desc=self.cmd_JERK_LIMIT_STATUS_help,
        )

    def _handle_connect(self):
        self.toolhead = self.printer.lookup_object("toolhead")
        if getattr(self.toolhead, "jerk_limiting", None) is not None:
            raise self.printer.config_error(
                "Only one [jerk_limiting] section is allowed"
            )
        self.toolhead.jerk_limiting = self

    def _handle_ready(self):
        # Apply auto-derived jerk once everything (incl. input_shaper) is up.
        if self.auto_jerk:
            self._apply_auto_jerk()

    def _measured_fn_per_axis(self):
        # {axis: input-shaper frequency (Hz)} for the axes that have one. Each
        # axis's own measured resonance sets that axis's jerk ceiling.
        ins = self.printer.lookup_object("input_shaper", None)
        if ins is None:
            return {}
        out = {}
        for sh in ins.get_shapers():
            f = getattr(getattr(sh, "params", None), "shaper_freq", 0.0)
            if f and f > 0.0:
                out[sh.get_axis()] = f
        return out

    def _axis_max_accel(self, axis_idx):
        # Per-axis max accel (limited_cartesian exposes max_accels), else the
        # toolhead's scalar max_accel.
        if hasattr(self.toolhead, "get_kinematics"):
            accels = getattr(self.toolhead.get_kinematics(), "max_accels", None)
            if accels is not None and axis_idx < len(accels):
                return accels[axis_idx]
        return self.toolhead.max_accel

    def _apply_auto_jerk(self, gcmd=None):
        # Per axis: max_jerk_<axis> = auto_jerk_ratio * max_accel_axis * f_n_axis
        # from that axis's measured input-shaper frequency. Axes without a
        # frequency keep their configured value.
        freqs = self._measured_fn_per_axis()
        if not freqs:
            msg = ("auto_jerk: no input_shaper frequency available; keeping "
                   "configured jerk limits (x=%.0f y=%.0f)"
                   % (self.max_jerk_x, self.max_jerk_y))
        else:
            parts = []
            for ax, idx in (("x", 0), ("y", 1)):
                f = freqs.get(ax)
                if not f:
                    continue
                a = self._axis_max_accel(idx)
                jv = self.auto_jerk_ratio * a * f
                setattr(self, "max_jerk_" + ax, jv)
                parts.append("%s=%.0f (a=%.0f, f=%.1fHz)" % (ax, jv, a, f))
            msg = ("auto_jerk: %.2f * max_accel * f_n -> %s"
                   % (self.auto_jerk_ratio, ", ".join(parts)))
        if gcmd is not None:
            gcmd.respond_info(msg)
        else:
            logging.info(msg)

    # -- queried by the toolhead lookahead/emitter (all no-ops when off) --
    def retime_active(self):
        # True when ramps are reshaped -> lookahead must use jerk reachability.
        return self.enabled and self.smooth_ramps

    def active_for(self, move):
        return (self.retime_active() and move.is_kinematic_move
                and not is_travel_move(move))

    def accel_limit(self, move, v):
        # Acceleration ceiling (mm/s^2) for `move` at speed `v`. This is the
        # single seam the jerk/corner logic queries instead of a static accel,
        # so the ceiling can be a function of speed. When a TOPP-RA torque curve
        # is active for this move, return its a_max(v) (already safety-margined)
        # -- the jerk-limited S-curve then rides the velocity-dependent torque
        # limit. Falls back to the move's static accel when no curve applies.
        # NB: this is the *motion* torque curve (a_max vs. speed), unrelated to
        # any TMC stepper driver register feature.
        tc = getattr(self.toolhead, "topp_ra", None)
        if tc is not None and tc.active_for(move):
            a = tc.get_max_accel_for_speed(v)
            if a is not None and a > 0.0:
                return a
        return move.accel

    def _move_jerk(self, move):
        # Directional jerk ceiling from the per-axis limits. Each axis sees jerk
        # j*|axes_r[axis]|, so to keep every axis within its own limit the move
        # jerk is min over axes of (max_jerk_axis / |axes_r[axis]|) -- the same
        # construction limited_cartesian uses for per-move accel. A move with no
        # X/Y/Z component falls back to the scalar max_jerk.
        j = None
        for axis, jmax in ((0, self.max_jerk_x), (1, self.max_jerk_y),
                           (2, self.max_jerk_z)):
            r = abs(move.axes_r[axis])
            if r > 1e-12:
                cand = jmax / r
                j = cand if j is None else min(j, cand)
        return self.max_jerk if j is None else j

    def _jerk_for(self, move):
        # Blend (corner) moves get a higher jerk allowance so the small residual
        # velocity change left after the velocity cap can ramp instead of
        # falling back to constant max-accel. See corner_max_jerk.
        if self.corner_max_jerk and getattr(move, "_jl_blend", False):
            return self.corner_max_jerk
        return self._move_jerk(move)

    def reach(self, move, u0, dist):
        # Max reachable u = v^2 over `dist`, jerk-limited, using this move's
        # (possibly speed-dependent) accel ceiling. Called by the toolhead
        # lookahead in place of the constant-accel delta_v2. Routing accel
        # through accel_limit is what makes lookahead reachability honor the
        # torque curve when TOPP-RA is active.
        #
        # NB: reach() runs during lookahead (calc_junction / flush) BEFORE
        # set_junction(), so move.cruise_v does not exist yet -- evaluate the
        # accel ceiling at the move's top speed (max_cruise_v2, set at move
        # creation), which is also the conservative point on a falling curve.
        v = math.sqrt(move.max_cruise_v2)
        accel = self.accel_limit(move, v)
        return reach_v2(u0, dist, accel, self._jerk_for(move),
                        self.toolhead.max_velocity)

    def plan_move(self, move):
        if not self.active_for(move):
            return None
        return plan_segments(
            move.start_v, move.cruise_v, move.end_v, move.move_d,
            self.accel_limit(move, move.cruise_v), self._jerk_for(move),
            self.resolution,
        )

    def plan_moves(self, moves):
        # Per-move jerk-limited slices (Phase 1). With blend_junctions (Phase 2)
        # maximal *collinear* runs whose ramps are already tight (no cruise
        # filler) are coalesced into a single continuous ramp so acceleration is
        # not forced to zero at every interior seam. Coalescing is restricted to
        # collinear runs because non-collinear interior junctions carry a
        # centripetal velocity limit that a merged ramp could violate.
        n = len(moves)
        out = [None] * n
        i = 0
        while i < n:
            m = moves[i]
            if not self.active_for(m):
                i += 1
                continue
            j = self._run_end(moves, i) if self.blend_junctions else i
            if j > i:
                self._plan_run(moves, i, j, out)
            else:
                out[i] = self.plan_move(m)
            i = j + 1
        return out

    def _is_tight(self, m, sign):
        # sign +1: pure accel ramp filling its length; -1: pure decel.
        if sign > 0:
            if not (m.cruise_v > m.start_v + 1e-9
                    and abs(m.cruise_v - m.end_v) < 1e-6):
                return False
            d = dist_jerk(m.start_v, m.cruise_v, m.accel, self._jerk_for(m))
        else:
            if not (m.cruise_v > m.end_v + 1e-9
                    and abs(m.cruise_v - m.start_v) < 1e-6):
                return False
            d = dist_jerk(m.end_v, m.cruise_v, m.accel, self._jerk_for(m))
        return d >= m.move_d - 1e-6

    def _collinear(self, a, b):
        return all(abs(a.axes_r[k] - b.axes_r[k]) < 1e-9 for k in range(3))

    def _run_end(self, moves, i):
        m = moves[i]
        sign = 1 if m.cruise_v > m.start_v else (-1 if m.cruise_v > m.end_v
                                                 else 0)
        if sign == 0 or not self._is_tight(m, sign):
            return i
        j = i
        while (j + 1 < len(moves) and self.active_for(moves[j + 1])
               and self._collinear(moves[j], moves[j + 1])
               and abs(moves[j].end_v - moves[j + 1].start_v) < 1e-6
               and self._is_tight(moves[j + 1], sign)):
            j += 1
        return j

    def _plan_run(self, moves, i, j, out):
        v0 = moves[i].start_v
        vend = moves[j].end_v
        total_d = sum(moves[k].move_d for k in range(i, j + 1))
        # Run accel ceiling through the same seam as single moves, so a
        # coalesced run rides the torque curve too (a_max at each move's cruise
        # speed); the run uses the most conservative of them.
        A = min(self.accel_limit(moves[k], moves[k].cruise_v)
                for k in range(i, j + 1))
        # The run is collinear, so all moves share a direction and one jerk
        # ceiling -- take it from the first move.
        J = self._jerk_for(moves[i])
        # Clamp the run's end velocity to what is actually reachable from v0
        # over the run length under the jerk limit. The per-move velocities come
        # from independent symmetric-ramp reachability, so the single coalesced
        # ramp v0 -> vend can need more distance than the moves provide. Without
        # this clamp `covered` exceeds total_d, distribute_slices() overflows the
        # excess into the final move (k >= len-1), and that endpoint overshoot
        # becomes a trapq position discontinuity -> stepcompress error. This
        # mirrors the peak_velocity() guard plan_segments() already applies to
        # single moves.
        reach = reach_v2(v0 * v0, total_d, A, J,
                         self.toolhead.max_velocity)
        if vend * vend > reach:
            vend = math.sqrt(max(reach, 0.0))
        slices = ramp_slices(v0, vend, A, J, self.resolution)
        covered = sum(s[6] for s in slices)
        filler = total_d - covered
        if filler > 1e-9 and vend > 1e-9:
            slices.append((0.0, filler / vend, 0.0, vend, vend, 0.0, filler))
        dlist = [moves[k].move_d for k in range(i, j + 1)]
        per_move = distribute_slices(slices, dlist)
        for k in range(i, j + 1):
            out[k] = per_move[k - i]

    def corners_active(self):
        return self.enabled and self.round_corners

    def _corner_cruise_v2(self, pts, accel):
        # Slowest centripetal-limited speed^2 across the discrete blend a..b,
        # via Klipper's junction-deviation model. Capping the blend moves to
        # this pushes the bulk decel/accel into the straight legs (which are
        # long enough to jerk-ramp it) instead of the sub-mm blend segments,
        # so the corner is traversed at ~constant speed. None -> no limit.
        jd = self.toolhead.junction_deviation
        best = None
        for i in range(1, len(pts) - 1):
            d0 = _unit(pts[i - 1], pts[i])
            d1 = _unit(pts[i], pts[i + 1])
            if d0 is None or d1 is None:
                continue
            jcos = -(d0[0] * d1[0] + d0[1] * d1[1] + d0[2] * d1[2])
            jcos = max(jcos, -0.999999)
            sin_d2 = math.sqrt(0.5 * (1.0 - jcos))
            if sin_d2 >= 1.0 - 1e-9:
                continue  # essentially straight at this vertex
            R = sin_d2 / (1.0 - sin_d2)
            v2 = accel * jd * R
            if best is None or v2 < best:
                best = v2
        return best

    def round_corner(self, prev, move):
        # Phase 3: replace the sharp corner prev->move with a curvature-
        # continuous cubic B-spline blend inside corner_max_deviation. Returns a
        # list [trimmed_prev, *blend_moves, trimmed_move] of new Move objects, or
        # None to leave the corner unchanged. Extrusion is distributed by length
        # so total filament is conserved and endpoints are preserved.
        if not (move.is_kinematic_move and prev.is_kinematic_move):
            return None
        if is_travel_move(move) or is_travel_move(prev):
            # Don't reshape a corner that involves a travel leg; jerk limiting is
            # only applied to extruding motion.
            return None
        Move = type(move)
        cos_t = sum(prev.axes_r[k] * move.axes_r[k] for k in range(3))
        cos_t = max(-1.0, min(1.0, cos_t))
        turn = math.acos(cos_t)
        if turn < math.radians(self.corner_min_angle) or cos_t < -0.99:
            return None  # too straight to matter, or a near-reversal
        th = self.toolhead
        trim = min(self.corner_blend_ratio * prev.move_d,
                   self.corner_blend_ratio * move.move_d)
        # Cap trim so the worst-case deviation stays within tolerance (deviation
        # is ~linear in trim); shrink and re-check a couple of times.
        vtx = prev.end_pos[:3]
        for _ in range(4):
            if trim < 1e-3:
                return None
            a = tuple(vtx[k] - prev.axes_r[k] * trim for k in range(3))
            b = tuple(vtx[k] + move.axes_r[k] * trim for k in range(3))
            blend = corner_blend(a, vtx, b, rounds=3)
            dev = max_deviation(blend, a, vtx, b)
            if dev <= self.corner_max_deviation:
                break
            trim *= self.corner_max_deviation / dev * 0.95
        else:
            return None
        # Emit the blend adaptively: the trim shrink above used the dense curve
        # to choose how far to round; here we sample only as many points as the
        # turn needs and never below corner_min_seg_len.
        blend = corner_blend_adaptive(
            a, vtx, b, self.corner_max_deviation, self.corner_min_seg_len
        )
        # Centripetal speed cap for the curved interior, so the straight legs
        # absorb the velocity change rather than the tiny blend segments.
        corner_v2 = self._corner_cruise_v2(
            [a] + list(blend) + [b], min(prev.accel, move.accel)
        )
        # Build the chain of points (x,y,z) and per-point cumulative extrusion.
        pa = prev.start_pos
        pb_end = move.end_pos
        pts = [pa[:3], a] + list(blend) + [b, pb_end[:3]]
        # Extrusion: prev contributes e over its full length, move over its full
        # length; distribute each move's e across its share of the new chain by
        # length so totals are conserved.
        prev_e = prev.axes_d[3]
        move_e = move.axes_d[3]
        seg_len = [math.sqrt(sum((pts[s + 1][k] - pts[s][k]) ** 2
                                 for k in range(3)))
                   for s in range(len(pts) - 1)]
        # Index of the vertex region split: segments [0] belong to prev body;
        # the corner (segments touching a..b) splits prev/move e by trim.
        # Simpler exact rule: assign e per segment proportional to how much of
        # prev vs move that segment covers. prev body = first segment, move body
        # = last segment, corner segments split prev_trim_e + move_trim_e.
        prev_body = seg_len[0]
        move_body = seg_len[-1]
        corner_len = sum(seg_len[1:-1])
        prev_full = prev.move_d
        move_full = move.move_d
        prev_body_e = prev_e * (prev_body / prev_full) if prev_full else 0.0
        move_body_e = move_e * (move_body / move_full) if move_full else 0.0
        corner_e = (prev_e - prev_body_e) + (move_e - move_body_e)
        e_seg = [prev_body_e]
        for s in range(1, len(seg_len) - 1):
            frac = (seg_len[s] / corner_len) if corner_len > 1e-12 else 0.0
            e_seg.append(corner_e * frac)
        e_seg.append(move_body_e)
        # Assemble Move objects with cumulative absolute extruder coordinate.
        prev_speed = math.sqrt(prev.max_cruise_v2)
        move_speed = math.sqrt(move.max_cruise_v2)
        new_moves = []
        e_abs = pa[3]
        for s in range(len(seg_len)):
            if seg_len[s] <= 1e-9:
                e_abs += e_seg[s]
                continue
            start = (pts[s][0], pts[s][1], pts[s][2], e_abs)
            e_abs += e_seg[s]
            end = (pts[s + 1][0], pts[s + 1][1], pts[s + 1][2], e_abs)
            spd = prev_speed if s == 0 else move_speed
            nm = Move(th, start, end, spd)
            # Interior (curved) blend segments: tag them so they get the corner
            # jerk allowance, and cap them to the centripetal corner speed so
            # the velocity change stays in the straight legs.
            if 0 < s < len(seg_len) - 1:
                nm._jl_blend = True
                if corner_v2 is not None and corner_v2 < nm.max_cruise_v2:
                    nm.limit_speed(math.sqrt(corner_v2), nm.accel)
            new_moves.append(nm)
        if len(new_moves) < 2:
            return None
        return new_moves

    def get_status(self, eventtime):
        return {
            "enabled": self.enabled,
            "smooth_ramps": self.smooth_ramps,
            "blend_junctions": self.blend_junctions,
            "round_corners": self.round_corners,
            "max_jerk": self.max_jerk,
            "max_jerk_x": self.max_jerk_x,
            "max_jerk_y": self.max_jerk_y,
            "max_jerk_z": self.max_jerk_z,
            "corner_max_deviation": self.corner_max_deviation,
            "corner_min_seg_len": self.corner_min_seg_len,
            "corner_max_jerk": self.corner_max_jerk,
        }

    cmd_SET_JERK_LIMIT_help = "Enable/disable jerk limiting and set parameters"

    def cmd_SET_JERK_LIMIT(self, gcmd):
        self.toolhead.flush_step_generation()
        en = gcmd.get_int("ENABLE", None)
        if en is not None:
            self.enabled = bool(en)
        for name, attr in (("SMOOTH_RAMPS", "smooth_ramps"),
                           ("BLEND_JUNCTIONS", "blend_junctions"),
                           ("ROUND_CORNERS", "round_corners")):
            v = gcmd.get_int(name, None)
            if v is not None:
                setattr(self, attr, bool(v))
        j = gcmd.get_float("MAX_JERK", None, above=0.0)
        if j is not None:
            # Convenience: set the base and all axes at once.
            self.max_jerk = j
            self.max_jerk_x = self.max_jerk_y = self.max_jerk_z = j
        jx = gcmd.get_float("MAX_JERK_X", None, above=0.0)
        if jx is not None:
            self.max_jerk_x = jx
        jy = gcmd.get_float("MAX_JERK_Y", None, above=0.0)
        if jy is not None:
            self.max_jerk_y = jy
        jz = gcmd.get_float("MAX_JERK_Z", None, above=0.0)
        if jz is not None:
            self.max_jerk_z = jz
        dev = gcmd.get_float("CORNER_MAX_DEVIATION", None, above=0.0)
        if dev is not None:
            self.corner_max_deviation = dev
        cmj = gcmd.get_float("CORNER_MAX_JERK", None, minval=0.0)
        if cmj is not None:
            self.corner_max_jerk = cmj
        msl = gcmd.get_float("CORNER_MIN_SEG_LEN", None, above=0.0)
        if msl is not None:
            self.corner_min_seg_len = msl
        ratio = gcmd.get_float("AUTO_JERK_RATIO", None, above=0.0)
        if ratio is not None:
            self.auto_jerk_ratio = ratio
        auto = gcmd.get_int("AUTO", None)
        if auto is not None:
            self.auto_jerk = bool(auto)
        # Re-derive max_jerk from the current shaper frequency when AUTO/ratio
        # is touched (e.g. after a fresh SHAPER_CALIBRATE) and auto_jerk is on.
        if self.auto_jerk and (auto is not None or ratio is not None):
            self._apply_auto_jerk(gcmd)
        self.cmd_JERK_LIMIT_STATUS(gcmd)

    cmd_JERK_LIMIT_STATUS_help = "Report jerk limiting configuration"

    def cmd_JERK_LIMIT_STATUS(self, gcmd):
        gcmd.respond_info(
            "jerk_limiting: enabled=%s smooth_ramps=%s blend_junctions=%s "
            "round_corners=%s max_jerk x/y/z=%.0f/%.0f/%.0f "
            "corner_max_deviation=%.3f corner_max_jerk=%.0f "
            "corner_min_seg_len=%.3f"
            % (self.enabled, self.smooth_ramps, self.blend_junctions,
               self.round_corners, self.max_jerk_x, self.max_jerk_y,
               self.max_jerk_z, self.corner_max_deviation,
               self.corner_max_jerk, self.corner_min_seg_len)
        )


def load_config(config):
    return JerkLimiting(config)
