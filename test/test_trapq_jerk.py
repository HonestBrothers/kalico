import math

import pytest

from klippy import chelper
from klippy.extras import motion_report


@pytest.fixture
def ffi():
    ffi_main, ffi_lib = chelper.get_ffi()
    tq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
    return ffi_main, ffi_lib, tq


def _finalize(ffi_lib, tq):
    ffi_lib.trapq_finalize_moves(tq, 1000.0, 0.0)


def _history(ffi_main, ffi_lib, tq):
    data = ffi_main.new("struct pull_move[16]")
    count = ffi_lib.trapq_extract_old(tq, data, len(data), 0.0, 1000.0)
    return [data[i] for i in range(count)]


def _append_jerk(ffi_lib, tq, print_time, move_t, start_v, accel, jerk):
    ffi_lib.trapq_append_jerk(
        tq, print_time, move_t, 1.0, 2.0, 3.0, 0.6, 0.8, 0.0,
        start_v, accel, jerk,
    )  # fmt: skip


def test_extract_reports_jerk(ffi):
    ffi_main, ffi_lib, tq = ffi
    _append_jerk(ffi_lib, tq, 1.0, 0.01, 20.0, 500.0, 2.5e5)
    _finalize(ffi_lib, tq)
    (move,) = _history(ffi_main, ffi_lib, tq)
    assert move.print_time == 1.0
    assert move.move_t == 0.01
    assert move.start_v == 20.0
    assert move.accel == pytest.approx(500.0)
    assert move.jerk == pytest.approx(2.5e5)
    assert (move.start_x, move.start_y, move.start_z) == (1.0, 2.0, 3.0)
    assert (move.x_r, move.y_r, move.z_r) == (0.6, 0.8, 0.0)


def test_zero_jerk_matches_trapq_append(ffi):
    # A constant-accel segment appended through either entry point must be
    # stored identically, so existing moves are untouched by the jerk term.
    ffi_main, ffi_lib, tq = ffi
    ffi_lib.trapq_append(
        tq, 1.0, 0.02, 0.0, 0.0, 1.0, 2.0, 3.0, 0.6, 0.8, 0.0,
        10.0, 30.0, 1000.0,
    )  # fmt: skip
    _append_jerk(ffi_lib, tq, 2.0, 0.02, 10.0, 1000.0, 0.0)
    _finalize(ffi_lib, tq)
    newer, older = _history(ffi_main, ffi_lib, tq)
    for field in ("move_t", "start_v", "accel", "jerk", "x_r", "y_r"):
        assert getattr(older, field) == getattr(newer, field), field
    assert older.jerk == 0.0


def test_history_keeps_jerk_only_segment(ffi):
    # A ramp that starts from rest with zero accel moves only through jerk;
    # it must not be mistaken for an idle segment and dropped from history.
    ffi_main, ffi_lib, tq = ffi
    _append_jerk(ffi_lib, tq, 1.0, 0.01, 0.0, 0.0, 1.0e5)
    _finalize(ffi_lib, tq)
    (move,) = _history(ffi_main, ffi_lib, tq)
    assert move.jerk == pytest.approx(1.0e5)


def test_zero_duration_segment_ignored(ffi):
    ffi_main, ffi_lib, tq = ffi
    _append_jerk(ffi_lib, tq, 1.0, 0.0, 10.0, 100.0, 1.0e5)
    _finalize(ffi_lib, tq)
    assert _history(ffi_main, ffi_lib, tq) == []


def test_motion_report_position_follows_cubic(ffi):
    ffi_main, ffi_lib, tq = ffi
    start_v, accel, jerk, move_t = 15.0, -300.0, 4.0e5, 0.012
    _append_jerk(ffi_lib, tq, 1.0, move_t, start_v, accel, jerk)
    _finalize(ffi_lib, tq)
    dump = object.__new__(motion_report.DumpTrapQ)
    dump.trapq = tq
    for frac in (0.01, 0.25, 0.5, 0.9, 1.0):
        t = frac * move_t
        pos, velocity = dump.get_trapq_position(1.0 + t)
        dist = start_v * t + accel * t * t / 2.0 + jerk * t**3 / 6.0
        assert pos[0] == pytest.approx(1.0 + 0.6 * dist, abs=1e-12)
        assert pos[1] == pytest.approx(2.0 + 0.8 * dist, abs=1e-12)
        assert pos[2] == 3.0
        assert velocity == pytest.approx(
            start_v + accel * t + jerk * t * t / 2.0, abs=1e-12
        )
    # Past the end the position holds at the segment's final point.
    pos, _ = dump.get_trapq_position(2.0)
    end = start_v * move_t + accel * move_t**2 / 2.0 + jerk * move_t**3 / 6.0
    assert math.isclose(pos[0], 1.0 + 0.6 * end, abs_tol=1e-12)
