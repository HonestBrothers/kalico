# Verifies the toolhead._process_moves unified-emit adapter logic in isolation:
# the constant-accel (no torque curve) path must reproduce the stock trapezoid
# that Move.set_junction would have emitted, plus the emit_profile invariant.
# (The torque-curve path is covered by test_pathplan.py's curve mode.)
# Run: klippy-env/bin/python klippy/extras/test_unified_adapter.py
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathplan  # noqa: E402
from test_pathplan import check_segs  # noqa: E402


class FakeMove:
    # Minimal stand-in with the fields _pathplan_cons + emit_profile read.
    def __init__(self, start_v, cruise_v, end_v, move_d, accel):
        self.start_v, self.cruise_v, self.end_v = start_v, cruise_v, end_v
        self.move_d, self.accel = move_d, accel
        self.is_kinematic_move = True


def cons_no_curve(move):
    # Mirrors ToolHead._pathplan_cons when TOPP-RA is inactive.
    v_ceil = max(move.cruise_v, move.start_v, move.end_v) + 1.0
    return pathplan.Constraints(a_of_v=None, a_const=move.accel,
                                v_ceil=v_ceil, dv_slice=1e18)


def stock_trapezoid(move):
    # What Move.set_junction / stock trapq_append would produce.
    a = move.accel
    half_inv = 0.5 / a
    accel_d = (move.cruise_v**2 - move.start_v**2) * half_inv
    decel_d = (move.cruise_v**2 - move.end_v**2) * half_inv
    cruise_d = move.move_d - accel_d - decel_d
    return accel_d, cruise_d, decel_d


def emitted_phase_distances(segs):
    # Collapse the emit_profile ladder back to (accel_d, cruise_d, decel_d).
    ad = cd = dd = 0.0
    for (at, ct, dt, sv, cv, a, dist) in segs:
        if at > 0.0:
            ad += dist
        elif dt > 0.0:
            dd += dist
        else:
            cd += dist
    return ad, cd, dd


def test_reproduces_stock_trapezoid():
    cases = [  # (start_v, cruise_v, end_v, move_d, accel)
        (0.0, 100.0, 0.0, 20.0, 3000.0),      # symmetric trapezoid
        (40.0, 180.0, 90.0, 30.0, 2500.0),    # asymmetric
        (120.0, 120.0, 120.0, 15.0, 3000.0),  # pure cruise
        (0.0, 150.0, 100.0, 25.0, 3000.0),    # accel then partial decel
    ]
    for sv, cv, ev, d, a in cases:
        m = FakeMove(sv, cv, ev, d, a)
        segs = pathplan.emit_profile(sv, cv, ev, d, cons_no_curve(m))
        # invariant (same guard the stepguard enforced)
        check_segs(segs, d, sv, ev, "trap(%.0f,%.0f,%.0f)" % (sv, cv, ev))
        # geometry matches the stock trapezoid to tight tolerance
        want = stock_trapezoid(m)
        got = emitted_phase_distances(segs)
        for lbl, w, g in zip(("accel_d", "cruise_d", "decel_d"), want, got):
            assert abs(w - g) <= 1e-6 + 1e-4 * abs(w), (
                "stock mismatch", sv, cv, ev, lbl, w, g)
        # constant-accel case should ladder to a single accel + single decel
        n_acc = sum(1 for s in segs if s[0] > 0.0)
        n_dec = sum(1 for s in segs if s[2] > 0.0)
        assert n_acc <= 1 and n_dec <= 1, ("not collapsed", n_acc, n_dec)
    print("  const-accel path reproduces stock trapezoid (%d cases) OK"
          % len(cases))


def test_triangle_peak_clamps():
    # Move too short to reach the requested cruise_v: emit_profile must clamp
    # the peak and still conserve move_d and hit the endpoints.
    m = FakeMove(0.0, 400.0, 0.0, 3.0, 3000.0)
    segs = pathplan.emit_profile(m.start_v, m.cruise_v, m.end_v, m.move_d,
                                 cons_no_curve(m))
    check_segs(segs, m.move_d, m.start_v, m.end_v, "triangle")
    peak = max(cv for (_, _, _, _, cv, _, _) in segs)
    assert peak < 400.0, ("peak not clamped", peak)
    print("  short-move triangle clamps peak to %.1f mm/s OK" % peak)


def main():
    test_reproduces_stock_trapezoid()
    test_triangle_peak_clamps()
    print("ALL PASS")


if __name__ == "__main__":
    main()
