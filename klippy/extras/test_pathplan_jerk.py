# Tests for Stage 3 jerk limiting inside pathplan.emit_profile (approach A).
# Verifies: the stepguard invariant still holds; acceleration is continuous
# (bounded |da/dt| on the controllable side and tapers to ~0 at the phase
# boundaries -- the precondition for the model-inverse FF accel term); distance
# is conserved and endpoints are hit; and short moves fall back to sharp.
# Run: klippy-env/bin/python klippy/extras/test_pathplan_jerk.py
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathplan  # noqa: E402
from test_pathplan import check_segs  # noqa: E402


def max_abs_step(segs):
    # Largest |a_i - a_{i-1}| across ALL consecutive slices, including the
    # accel<->cruise<->decel seams. Bounded by the discrete taper floor 2*J*dt0:
    # the brake cap a=sqrt(2*J*rem) lands with residual accel <= 2*J*dt0 (a
    # partial slice triggers at rem <= a*dt0 => a <= 2*J*dt0). Everything
    # smaller is a genuine bounded-jerk transition (<= J*dt0 within a ramp).
    worst = 0.0
    prev_a = 0.0
    for (at, ct, dt, sv, cv, a, dist) in segs:
        worst = max(worst, abs(a - prev_a))
        prev_a = a
    worst = max(worst, prev_a)  # final drop to rest
    return worst


# Representative FF accel coefficient b2 = 1/wn^2 for the measured 77 Hz Y mode.
B2_77HZ = 1.0 / (2.0 * math.pi * 77.0) ** 2
ONE_STEP_MM = 1.0 / 80.0  # ~80 steps/mm axis


def make_cons(curve=False, jerk=1.0e5, jerk_dt=0.001):
    if curve:
        return pathplan.Constraints(
            a_of_v=lambda v: max(1000.0, 20000.0 - 30.0 * v),
            a_const=1000.0, v_ceil=400.0, dv_slice=10.0,
            max_jerk=jerk, jerk_dt=jerk_dt)
    return pathplan.Constraints(
        a_of_v=None, a_const=8000.0, v_ceil=400.0, dv_slice=1e18,
        max_jerk=jerk, jerk_dt=jerk_dt)


def test_invariant_and_taper(cons, tag):
    cases = [  # (vs, vc, ve, move_d)
        (0.0, 200.0, 0.0, 60.0),
        (0.0, 250.0, 120.0, 50.0),
        (80.0, 300.0, 40.0, 80.0),
        (150.0, 150.0, 150.0, 20.0),   # pure cruise (no ramp)
    ]
    worst_jump = 0.0
    for vs, vc, ve, d in cases:
        segs = pathplan.emit_profile(vs, vc, ve, d, cons)
        assert segs, (tag, "empty", vs, vc, ve)
        check_segs(segs, d, vs, ve, "%s:%g,%g,%g" % (tag, vs, vc, ve))
        J, dtj = cons.max_jerk, cons.jerk_dt
        # No hard accel step: every accel discontinuity (including the cruise
        # seams) is bounded by the discrete taper floor 2*J*dt0.
        step = max_abs_step(segs)
        assert step <= 2.1 * J * dtj + 1e-6, (tag, "accel step exceeded",
                                              step, 2.0 * J * dtj)
        # THE property the FF needs: the worst accel jump, mapped through the
        # FF accel coefficient b2, is a sub-step motor-position jump -> no
        # stepcompress 'Invalid sequence' even with the accel term on.
        ff_jump = B2_77HZ * step
        assert ff_jump < ONE_STEP_MM, (tag, "FF jump not sub-step",
                                       ff_jump, ONE_STEP_MM)
        worst_jump = max(worst_jump, ff_jump)
    print("  [%s] invariant + accel step<=2*J*dt + worst FF jump %.5fmm "
          "(<%.5f) OK" % (tag, worst_jump, ONE_STEP_MM))


def test_jerk_widens_ramp_vs_sharp():
    # A jerk-limited accel ramp reaches a lower peak accel -> uses MORE distance
    # than the sharp constant-accel change for the same dv. Sanity that jerk is
    # actually doing something.
    sharp = pathplan.Constraints(a_of_v=None, a_const=8000.0, v_ceil=400.0,
                                 dv_slice=1e18)
    jerky = make_cons(curve=False, jerk=5.0e4)
    d_sharp = pathplan.dist_change(0.0, 200.0, sharp)
    _, d_jerk = pathplan._ramp_up_jerk(0.0, 200.0, jerky)
    assert d_jerk > d_sharp, (d_jerk, d_sharp)
    print("  jerk ramp uses more distance than sharp (%.2f > %.2f mm) OK"
          % (d_jerk, d_sharp))


def test_short_move_falls_back_to_sharp():
    # A move too short to jerk-limit must fall back to the sharp profile (not
    # crash, not return empty) and still satisfy the invariant + endpoints.
    cons = make_cons(curve=False, jerk=1.0e4, jerk_dt=0.001)  # very low jerk
    vs, vc, ve, d = 0.0, 300.0, 0.0, 1.5  # tiny move, high requested cruise
    segs = pathplan.emit_profile(vs, vc, ve, d, cons)
    assert segs, "fallback returned empty"
    check_segs(segs, d, vs, ve, "shortfallback")
    # sharp fallback -> few segments, peak clamped below requested cruise
    peak = max(s[4] for s in segs)
    assert peak < 300.0, ("peak not clamped on fallback", peak)
    print("  short move falls back to sharp (peak=%.1f, invariant holds) OK"
          % peak)


def test_disabled_matches_sharp():
    # max_jerk=None must be byte-for-byte the sharp path.
    sharp = pathplan.Constraints(a_of_v=None, a_const=8000.0, v_ceil=400.0,
                                 dv_slice=1e18)
    withoff = pathplan.Constraints(a_of_v=None, a_const=8000.0, v_ceil=400.0,
                                   dv_slice=1e18, max_jerk=None)
    a = pathplan.emit_profile(0.0, 200.0, 50.0, 30.0, sharp)
    b = pathplan.emit_profile(0.0, 200.0, 50.0, 30.0, withoff)
    assert a == b, "max_jerk=None diverged from sharp"
    print("  max_jerk=None identical to sharp path OK")


def test_ff_short_move_no_hard_step():
    # Regression for the 2026-07-09 hardware crash: short print-junction moves
    # (large velocity change over a short distance) fell back to a SHARP profile
    # (hard accel step); the model-inverse FF's p2*a term turned that into a
    # multi-step position jump -> stepcompress 'Invalid sequence'. With max_da
    # set (FF active), the emitter must keep every accel step small enough that
    # p2*da stays below one motor step, on every feasible move.
    p2 = 1.0 / (2.0 * math.pi * 77.0) ** 2   # 77 Hz Y mode
    step = 1.0 / 80.0
    max_da = 0.3 * step / p2
    cons = pathplan.Constraints(a_of_v=None, a_const=15000.0, v_ceil=500.0,
                                dv_slice=1e18, max_jerk=1.0e5, jerk_dt=0.001,
                                max_da=max_da)
    worst = 0.0
    n = 0
    for d in (0.2, 0.4, 0.6, 1.0, 1.5, 2.0, 3.0):
        for vs, ve, vc in [(10, 80, 200), (0, 120, 300), (5, 150, 250),
                           (30, 30, 120), (80, 10, 200)]:
            # skip inputs the lookahead could never emit (unreachable even at
            # constant accel over the move length)
            if abs(ve * ve - vs * vs) / (2.0 * 15000.0) > d + 1e-9:
                continue
            segs = pathplan.emit_profile(vs, vc, ve, d, cons)
            check_segs(segs, d, vs, ve, "ffshort %g,%g,%g,%g" % (vs, vc, ve, d))
            prev = 0.0
            mda = 0.0
            for s in segs:
                mda = max(mda, abs(s[5] - prev))
                prev = s[5]
            mda = max(mda, prev)
            ff_jump = p2 * mda
            assert ff_jump < step, ("FF jump exceeds a step", vs, ve, d,
                                    ff_jump, step)
            worst = max(worst, ff_jump)
            n += 1
    print("  FF short-move: %d feasible moves, worst FF jump %.5fmm (<%.5f) OK"
          % (n, worst, step))


def test_maxda_off_matches_no_maxda():
    # max_da=None must not change the profile (non-FF path unaffected).
    base = pathplan.Constraints(a_of_v=None, a_const=8000.0, v_ceil=400.0,
                                dv_slice=1e18, max_jerk=1.0e5, jerk_dt=0.001)
    withn = pathplan.Constraints(a_of_v=None, a_const=8000.0, v_ceil=400.0,
                                 dv_slice=1e18, max_jerk=1.0e5, jerk_dt=0.001,
                                 max_da=None)
    a = pathplan.emit_profile(0.0, 200.0, 50.0, 30.0, base)
    b = pathplan.emit_profile(0.0, 200.0, 50.0, 30.0, withn)
    assert a == b, "max_da=None changed the profile"
    print("  max_da=None identical to unset OK")


def main():
    for curve in (False, True):
        test_invariant_and_taper(make_cons(curve=curve),
                                 "curve" if curve else "const")
    test_jerk_widens_ramp_vs_sharp()
    test_short_move_falls_back_to_sharp()
    test_disabled_matches_sharp()
    test_ff_short_move_no_hard_step()
    test_maxda_off_matches_no_maxda()
    print("ALL PASS")


if __name__ == "__main__":
    main()
