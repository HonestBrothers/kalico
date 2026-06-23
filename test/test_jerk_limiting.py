"""Unit tests for the jerk-limiting motion math (klippy/extras/jerk_limiting).

These cover the pure trajectory math (no hardware): jerk-limited ramp distance,
reachability inversion, slice emission, per-move distance conservation, slice
distribution across move boundaries, and cubic B-spline corner blending. A final
test drives emitted slices through the real C trapezoid queue to confirm the C
motion integrator accepts them and yields continuous, conservation-correct
motion.
"""
import math
import pathlib
import typing

from klippy_testing import PrinterShim

import klippy.chelper
from klippy.extras import jerk_limiting as jl

A = 3000.0   # accel cap (mm/s^2)
J = 100000.0  # jerk (mm/s^3)
DT = 0.002   # slice resolution (s)
VMAX = 400.0


def _ramp_endpoints(slices):
    """Reconstruct (final velocity, total distance) from a slice list."""
    v = slices[0][3]
    total = 0.0
    for at, ct, dct, sv, cv, a, dist in slices:
        h = at + ct + dct
        v = cv if (at > 0.0 or ct > 0.0) else cv - a * dct
        total += dist
    return v, total


def test_dist_jerk_at_least_constant_accel():
    for v0, v1 in [(0, 100), (50, 200), (0, 30), (300, 10), (0, 5)]:
        d = jl.dist_jerk(v0, v1, A, J)
        d_const = abs(v1 * v1 - v0 * v0) / (2 * A)
        assert d >= d_const - 1e-9


def test_reach_inverts_dist_jerk():
    for v0 in [0.0, 50.0, 150.0]:
        d = jl.dist_jerk(v0, 250.0, A, J)
        v = math.sqrt(jl.reach_v2(v0 * v0, d, A, J, VMAX))
        assert abs(v - 250.0) < 0.5


def test_ramp_slices_properties():
    for v0, v1 in [(0, 250), (250, 0), (40, 300), (300, 40), (0, 8)]:
        segs = jl.ramp_slices(v0, v1, A, J, DT)
        # velocity continuity
        v = v0
        max_a = prev_a = max_jk = 0.0
        for at, ct, dct, sv, cv, a, dist in segs:
            assert abs(sv - v) < 1e-6
            h = at + ct + dct
            max_a = max(max_a, a)
            max_jk = max(max_jk, abs(a - prev_a) / h)
            prev_a = a
            v = cv if at > 0.0 else cv - a * dct
        assert abs(v - v1) < 0.5                      # exact endpoint
        assert max_a <= A * 1.02                      # accel cap respected
        assert max_jk <= J * 2.5                      # jerk roughly bounded
        _, total = _ramp_endpoints(segs)
        assert abs(total - jl.dist_jerk(v0, v1, A, J)) < 0.05 * max(total, 1.0)


def test_plan_segments_conserves_distance_and_endpoints():
    cases = [(0, 250, 0, 60), (0, 300, 0, 5), (50, 200, 100, 40),
             (0, 400, 0, 200), (100, 100, 100, 30)]
    for vs, vc, ve, d in cases:
        segs = jl.plan_segments(vs, vc, ve, d, A, J, DT)
        assert segs is not None
        v_end, total = _ramp_endpoints(segs)
        assert abs(total - d) < 0.02 * max(d, 1.0)
        assert abs(v_end - ve) < 1.0
        assert abs(segs[0][3] - vs) < 1e-6


def test_distribute_slices_conserves_per_move_distance():
    slices = jl.ramp_slices(0.0, 300.0, A, J, DT)
    total = sum(s[6] for s in slices)
    dlist = [total * 0.2, total * 0.5, total * 0.3]
    per = jl.distribute_slices(slices, dlist)
    for k, dl in enumerate(dlist):
        assert abs(sum(s[6] for s in per[k]) - dl) < 1e-6
    # velocity stays continuous across the reassembled chain
    v = 0.0
    for s in (s for pm in per for s in pm):
        assert abs(s[3] - v) < 1e-6
        v = s[4] if (s[0] > 0 or s[1] > 0) else s[4] - s[5] * s[2]
    assert abs(v - 300.0) < 0.5


def test_corner_blend_deviation_bounded_and_linear():
    vtx = (40.0, 0.0, 0.0)
    devs = []
    for trim in [16.0, 8.0, 4.0, 2.0]:
        a = (vtx[0] - trim, 0.0, 0.0)
        b = (vtx[0], trim, 0.0)
        pts = jl.corner_blend(a, vtx, b, rounds=4)
        devs.append(jl.max_deviation(pts, a, vtx, b))
    # strictly shrinking and ~linear in trim (halving trim halves deviation)
    for i in range(len(devs) - 1):
        assert devs[i + 1] < devs[i]
        assert abs(devs[i] / devs[i + 1] - 2.0) < 0.1


def test_split_slice_conserves():
    s = jl.ramp_slices(0.0, 200.0, A, J, DT)[5]
    first, second = jl.split_slice(s, 0.4 * s[6])
    assert abs((first[6] + second[6]) - s[6]) < 1e-9
    # velocity continuity at the split point
    v_first_end = first[4] if first[0] > 0 else first[4] - first[5] * first[2]
    assert abs(v_first_end - second[3]) < 1e-6


class _MockTH:
    max_velocity = 400.0


class _MockMove:
    """Minimal Move stand-in for exercising plan_moves / round_corner."""

    def __init__(self, th, start, end, speed):
        self.toolhead = th
        self.start_pos = tuple(start)
        self.end_pos = tuple(end)
        d = [end[i] - start[i] for i in range(4)]
        self.axes_d = d
        self.move_d = math.sqrt(sum(d[i] * d[i] for i in range(3))) or abs(d[3])
        inv = 1.0 / self.move_d if self.move_d else 0.0
        self.axes_r = [x * inv for x in d]
        self.is_kinematic_move = self.move_d > 1e-9 and any(
            d[i] for i in range(3)
        )
        self.accel = A
        vel = min(speed, th.max_velocity)
        self.max_cruise_v2 = vel * vel
        self.start_v = self.cruise_v = self.end_v = vel


def _mk_jl():
    obj = jl.JerkLimiting.__new__(jl.JerkLimiting)
    obj.enabled = True
    obj.smooth_ramps = True
    obj.blend_junctions = True
    obj.round_corners = True
    obj.max_jerk = J
    obj.resolution = DT
    obj.corner_max_deviation = 0.05
    obj.corner_min_angle = 5.0
    obj.corner_blend_ratio = 0.25
    obj.toolhead = _MockTH()
    return obj


def test_phase2_coalesces_collinear_accel_run():
    th = _MockTH()
    obj = _mk_jl()
    v1 = math.sqrt(jl.reach_v2(0.0, 20.0, A, J, 400.0))
    d1 = jl.dist_jerk(0.0, v1, A, J)
    v2 = math.sqrt(jl.reach_v2(v1 * v1, 20.0, A, J, 400.0))
    d2 = jl.dist_jerk(v1, v2, A, J)
    m1 = _MockMove(th, (0, 0, 0, 0), (d1, 0, 0, 0), 400)
    m1.start_v, m1.cruise_v, m1.end_v = 0.0, v1, v1
    m2 = _MockMove(th, (d1, 0, 0, 0), (d1 + d2, 0, 0, 0), 400)
    m2.start_v, m2.cruise_v, m2.end_v = v1, v2, v2
    out = obj.plan_moves([m1, m2])
    assert out[0] and out[1]
    # Acceleration is carried across the seam (does not dip to zero).
    assert out[0][-1][5] > 100.0 and out[1][0][5] > 100.0
    assert abs(sum(s[6] for s in out[0]) - d1) < 1e-4
    assert abs(sum(s[6] for s in out[1]) - d2) < 1e-4


def test_phase2_does_not_coalesce_across_corner():
    th = _MockTH()
    obj = _mk_jl()
    v1 = math.sqrt(jl.reach_v2(0.0, 20.0, A, J, 400.0))
    d1 = jl.dist_jerk(0.0, v1, A, J)
    m1 = _MockMove(th, (0, 0, 0, 0), (d1, 0, 0, 0), 400)
    m1.start_v, m1.cruise_v, m1.end_v = 0.0, v1, v1
    m2 = _MockMove(th, (d1, 0, 0, 0), (d1, d1, 0, 0), 400)  # 90 deg turn
    m2.start_v, m2.cruise_v, m2.end_v = v1, v1, v1
    out = obj.plan_moves([m1, m2])
    # Still correct per move, just not merged.
    assert abs(sum(s[6] for s in out[0]) - d1) < 1e-4


def test_phase2_run_clamps_unreachable_end_velocity():
    # Regression: a coalesced collinear accel run whose per-move velocities
    # imply a single ramp v0 -> vend longer than the moves' summed length (the
    # nominal end velocity is not reachable from v0 within the run distance).
    # Before the _plan_run reachability clamp, `covered` exceeded total_d, the
    # cruise-filler branch (filler > 0) was skipped, and distribute_slices()
    # dumped the leftover slices into the final move -- overshooting its
    # endpoint. On hardware that endpoint overshoot is a trapq position
    # discontinuity that the step compressor rejects ("Internal error in MCU
    # stepcompress"). Each move must emit exactly its own length and the run
    # must conserve total distance.
    th = _MockTH()
    obj = _mk_jl()
    # Three short, collinear, accelerating "tight" moves (each move_d is well
    # below its dist_jerk, as happens when a long move is split into small
    # collinear sub-moves, e.g. by bed_mesh).
    vels = [(0.0, 100.0), (100.0, 160.0), (160.0, 200.0)]
    move_d = 2.0
    moves = []
    x = 0.0
    for sv, ev in vels:
        m = _MockMove(th, (x, 0, 0, 0), (x + move_d, 0, 0, 0), 400)
        m.start_v, m.cruise_v, m.end_v = sv, ev, ev
        moves.append(m)
        x += move_d
    total_d = move_d * len(moves)
    # Precondition: the single ramp 0 -> 200 genuinely needs more than the run.
    assert jl.dist_jerk(0.0, 200.0, A, J) > total_d
    out = obj.plan_moves(moves)
    assert all(o for o in out)
    # No move overshoots its own length...
    for k, m in enumerate(moves):
        assert abs(sum(s[6] for s in out[k]) - m.move_d) < 1e-3
    # ...and the run conserves total distance exactly.
    emitted = sum(s[6] for pm in out for s in pm)
    assert abs(emitted - total_d) < 1e-3
    # End velocity is clamped to what is reachable over the run distance.
    v_end, _ = _ramp_endpoints([s for pm in out for s in pm])
    assert v_end < 200.0
    assert abs(jl.dist_jerk(0.0, v_end, A, J) - total_d) < 1e-2


def test_phase3_round_corner_conserves_and_bounds():
    th = _MockTH()
    obj = _mk_jl()
    prev = _MockMove(th, (0, 0, 0, 0), (40, 0, 0, 2.0), 200)
    mv = _MockMove(th, (40, 0, 0, 2.0), (40, 40, 0, 4.0), 200)
    chain = obj.round_corner(prev, mv)
    assert chain is not None and len(chain) >= 3
    # endpoints preserved
    assert chain[0].start_pos[:3] == (0, 0, 0)
    assert all(abs(chain[-1].end_pos[k] - mv.end_pos[k]) < 1e-9
               for k in range(3))
    # extrusion conserved
    assert abs(sum(c.axes_d[3] for c in chain) - 4.0) < 1e-6
    assert abs(chain[-1].end_pos[3] - 4.0) < 1e-6
    # deviation within tolerance
    blend = [c.start_pos[:3] for c in chain[1:]]
    dev = jl.max_deviation(blend, chain[0].end_pos[:3], (40, 0, 0),
                           chain[-1].start_pos[:3])
    assert dev <= 0.05 + 1e-6
    # contiguous chain
    assert all(chain[s].end_pos == chain[s + 1].start_pos
               for s in range(len(chain) - 1))


def test_phase3_skips_gentle_corner():
    th = _MockTH()
    obj = _mk_jl()
    prev = _MockMove(th, (0, 0, 0, 0), (40, 0, 0, 0), 200)
    mv = _MockMove(th, (40, 0, 0, 0), (80, 1, 0, 0), 200)  # ~1.4 deg
    assert obj.round_corner(prev, mv) is None


def test_config_section_parses(
    config_root: typing.Annotated[pathlib.Path, "test_configs/jerk_limiting"],
):
    """The [jerk_limiting] section and all its options parse from a real config."""
    start_args = {"config_file": str(config_root / "printer.cfg")}
    with PrinterShim(start_args) as printer:
        config = printer.load_config()
        sec = config.getsection("jerk_limiting")
        assert sec.getboolean("enabled") is True
        assert sec.getboolean("round_corners") is True
        assert sec.getfloat("max_jerk") == 80000.0
        assert abs(sec.getfloat("resolution") - 0.0015) < 1e-12
        assert abs(sec.getfloat("corner_max_deviation") - 0.08) < 1e-12
        assert abs(sec.getfloat("corner_blend_ratio") - 0.3) < 1e-12


def test_c_trapq_accepts_jerk_slices():
    ffi_main, ffi_lib = klippy.chelper.get_ffi()
    segs = jl.plan_segments(0.0, 250.0, 0.0, 60.0, A, J, DT)
    tq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
    t = 10.0
    pos = 0.0
    for at, ct, dct, sv, cv, a, dist in segs:
        ffi_lib.trapq_append(tq, t, at, ct, dct,
                             pos, 0.0, 0.0, 1.0, 0.0, 0.0, sv, cv, a)
        t += at + ct + dct
        pos += dist
    ffi_lib.trapq_finalize_moves(tq, t + 1.0, 0.0)
    pm = ffi_main.new("struct pull_move[]", 4096)
    cnt = ffi_lib.trapq_extract_old(tq, pm, 4096, 0.0, t + 1.0)
    assert cnt > 0
    moves = sorted([pm[i] for i in range(cnt)], key=lambda m: m.print_time)
    prev_pos = prev_v = None
    total = 0.0
    for m in moves:
        seg_d = m.start_v * m.move_t + 0.5 * m.accel * m.move_t * m.move_t
        ex = m.start_x + seg_d * m.x_r
        ev = m.start_v + m.accel * m.move_t
        if prev_pos is not None:
            assert abs(m.start_x - prev_pos) < 1e-6   # position continuous
            assert abs(m.start_v - prev_v) < 1e-3     # velocity continuous
        prev_pos, prev_v = ex, ev
        total += seg_d
    assert abs(total - 60.0) < 0.02
    assert abs(moves[0].start_v) < 1e-6
    assert abs(prev_v) < 0.5
