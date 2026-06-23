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
        self.toolhead = None
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
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

    # -- queried by the toolhead lookahead/emitter (all no-ops when off) --
    def retime_active(self):
        # True when ramps are reshaped -> lookahead must use jerk reachability.
        return self.enabled and self.smooth_ramps

    def active_for(self, move):
        return self.retime_active() and move.is_kinematic_move

    def reach(self, u0, dist, accel):
        return reach_v2(u0, dist, accel, self.max_jerk,
                        self.toolhead.max_velocity)

    def plan_move(self, move):
        if not self.active_for(move):
            return None
        return plan_segments(
            move.start_v, move.cruise_v, move.end_v, move.move_d,
            move.accel, self.max_jerk, self.resolution,
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
            d = dist_jerk(m.start_v, m.cruise_v, m.accel, self.max_jerk)
        else:
            if not (m.cruise_v > m.end_v + 1e-9
                    and abs(m.cruise_v - m.start_v) < 1e-6):
                return False
            d = dist_jerk(m.end_v, m.cruise_v, m.accel, self.max_jerk)
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
        A = min(moves[k].accel for k in range(i, j + 1))
        slices = ramp_slices(v0, vend, A, self.max_jerk, self.resolution)
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

    def round_corner(self, prev, move):
        # Phase 3: replace the sharp corner prev->move with a curvature-
        # continuous cubic B-spline blend inside corner_max_deviation. Returns a
        # list [trimmed_prev, *blend_moves, trimmed_move] of new Move objects, or
        # None to leave the corner unchanged. Extrusion is distributed by length
        # so total filament is conserved and endpoints are preserved.
        if not (move.is_kinematic_move and prev.is_kinematic_move):
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
            "corner_max_deviation": self.corner_max_deviation,
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
            self.max_jerk = j
        dev = gcmd.get_float("CORNER_MAX_DEVIATION", None, above=0.0)
        if dev is not None:
            self.corner_max_deviation = dev
        self.cmd_JERK_LIMIT_STATUS(gcmd)

    cmd_JERK_LIMIT_STATUS_help = "Report jerk limiting configuration"

    def cmd_JERK_LIMIT_STATUS(self, gcmd):
        gcmd.respond_info(
            "jerk_limiting: enabled=%s smooth_ramps=%s blend_junctions=%s "
            "round_corners=%s max_jerk=%.0f corner_max_deviation=%.3f"
            % (self.enabled, self.smooth_ramps, self.blend_junctions,
               self.round_corners, self.max_jerk, self.corner_max_deviation)
        )


def load_config(config):
    return JerkLimiting(config)
