# Tests for torque-curve (TOPP-RA) ramps in the notch planner.
#
# Each notched ramp gets its own constant plateau accel from an AccelCurve:
# the largest A for which the whole ramp stays under the curve. Verifies the
# envelope only ever lowers the curve, that rendered ramps (with and without
# the spectral-null solve) never exceed the RAW curve at any slice, that the
# lookahead's reach is always renderable, that a curve changes nothing when it
# does not bind, and that slow ramps get more accel than one per-move cap.
#
# Run: klippy-env/bin/python test/test_pathplan_torque.py
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "klippy", "extras"))
import pathplan  # noqa: E402
from test_pathplan import check_segs  # noqa: E402

# A stepper-like curve: flat to 100 mm/s, then falling.
SPEEDS = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 650.0]
ACCELS = [30000.0, 30000.0, 24000.0, 17000.0, 12000.0, 8500.0, 5000.0]


def raw_at(v, speeds=SPEEDS, accels=ACCELS):
    # The curve exactly as measured (no envelope), clamped outside the knots.
    if v <= speeds[0]:
        return accels[0]
    if v >= speeds[-1]:
        return accels[-1]
    for i in range(len(speeds) - 1):
        if speeds[i] <= v <= speeds[i + 1]:
            t = (v - speeds[i]) / (speeds[i + 1] - speeds[i])
            return accels[i] + t * (accels[i + 1] - accels[i])
    raise AssertionError(v)


def make_cons(curve, f1=50.0, f2=None, a_const=100000.0, null=False):
    return pathplan.Constraints(
        a_const=a_const,
        v_ceil=700.0,
        jerk_dt=0.0005,
        notch_freq=f1,
        notch_freq2=f2,
        spectral_null=null,
        curve=curve,
    )


def assert_under_raw(slices, speeds, accels, tol=1e-6):
    for s in slices:
        a = s[5]
        for v in (s[3], s[4]):
            lim = raw_at(v, speeds, accels)
            assert a <= lim * (1.0 + tol) + 1e-6, (
                "slice accel %.1f over curve %.1f at v=%.2f" % (a, lim, v)
            )


def random_curve(rng):
    n = rng.randint(2, 8)
    speeds = sorted(rng.uniform(0.0, 700.0) for _ in range(n))
    speeds[0] = 0.0
    speeds = sorted(set(speeds))
    # Mostly falling, with the odd bump the envelope must flatten.
    a = rng.uniform(8000.0, 40000.0)
    accels = []
    for _ in speeds:
        accels.append(a)
        a = max(500.0, a * rng.uniform(0.4, 1.08))
    return speeds, accels


def test_envelope_only_lowers_and_limits_fall():
    rng = random.Random(1)
    for _ in range(200):
        speeds, accels = random_curve(rng)
        curve = pathplan.AccelCurve(speeds, accels)
        fall = rng.uniform(20.0, 90.0)
        prev = None
        for i in range(400):
            v = 710.0 * i / 399
            e = curve.at(v, fall)
            assert e <= raw_at(v, speeds, accels) + 1e-6
            if prev is not None:
                dv = 710.0 / 399
                assert e <= prev + 1e-6, "envelope rose with speed"
                assert prev - e <= fall * dv * (1 + 1e-9) + 1e-6
            prev = e
    print("  envelope <= curve, non-increasing, fall-rate limited OK")


def test_rendered_ramps_stay_under_the_curve():
    rng = random.Random(2)
    count = 0
    for _ in range(150):
        speeds, accels = random_curve(rng)
        curve = pathplan.AccelCurve(speeds, accels)
        f1 = rng.uniform(35.0, 90.0)
        f2 = rng.choice([None, f1 * rng.uniform(1.05, 1.5)])
        for null in (False, True):
            cons = make_cons(curve, f1, f2, null=null)
            for _ in range(6):
                v0 = rng.uniform(0.0, 600.0)
                v1 = v0 + rng.uniform(1.0, 650.0 - v0 + 1.0)
                slices, dist = pathplan._ramp_up_jerk(v0, v1, cons)
                assert slices, (v0, v1)
                assert abs(slices[-1][4] - v1) < 1e-6
                slices = pathplan._apply_spectral_null(slices, cons)
                assert_under_raw(slices, speeds, accels)
                count += 1
    print(
        "  %d rendered ramps (sampled + nulled) under the raw curve OK" % count
    )


def test_curve_that_never_binds_changes_nothing():
    # 1e6 everywhere: every ramp is the plain notch ramp, bit for bit.
    flat = pathplan.AccelCurve([0.0, 700.0], [1e6, 1e6])
    for f2 in (None, 70.0):
        a = make_cons(None, 50.0, f2, a_const=30000.0)
        b = make_cons(flat, 50.0, f2, a_const=30000.0)
        for v0, v1 in ((0.0, 100.0), (20.0, 500.0), (0.0, 650.0)):
            assert pathplan._ramp_up_jerk(v0, v1, a) == pathplan._ramp_up_jerk(
                v0, v1, b
            )
            assert pathplan.notch_dist(
                v0, v1, 30000.0, 50.0, f2
            ) == pathplan.notch_dist(v0, v1, 30000.0, 50.0, f2, flat, a.jerk_dt)
    print("  non-binding curve is a no-op OK")


def test_reach_is_always_renderable():
    # The lookahead's reach must never promise a speed the emitter cannot
    # render: plan exit speeds with notch_reach_v2, then emit the move.
    rng = random.Random(3)
    n = 0
    for _ in range(80):
        speeds, accels = random_curve(rng)
        curve = pathplan.AccelCurve(speeds, accels)
        f1 = rng.uniform(35.0, 90.0)
        f2 = rng.choice([None, f1 * rng.uniform(1.05, 1.5)])
        for null in (False, True):
            cons = make_cons(curve, f1, f2, null=null)
            for _ in range(5):
                vs = rng.uniform(0.0, 300.0)
                d = rng.uniform(0.5, 400.0)
                ve2 = pathplan.notch_reach_v2(
                    vs * vs, d, 100000.0, 700.0, f1, f2, curve, cons.jerk_dt
                )
                ve = math.sqrt(ve2)
                segs = pathplan.emit_profile(vs, max(vs, ve), ve, d, cons)
                check_segs(segs, d, vs, ve, "torque reach")
                assert_under_raw(
                    [s for s in segs if s[0] > 0.0 or s[2] > 0.0],
                    speeds,
                    accels,
                )
                n += 1
    print("  %d reach-planned moves render feasibly under the curve OK" % n)


def test_slow_ramps_get_more_than_a_per_move_cap():
    curve = pathplan.AccelCurve(SPEEDS, ACCELS)
    cons = make_cons(curve, 50.0)
    # 0 -> 150 inside a move whose feedrate is 650: a per-move cap would hold
    # the whole move to a(650) = 5000.
    slices, _d = pathplan._ramp_up_jerk(0.0, 150.0, cons)
    slow = max(s[5] for s in slices)
    assert slow > 5000.0 * 1.4, slow
    # And the long ramp to 650 saturates below the low-speed accel.
    slices, _d = pathplan._ramp_up_jerk(0.0, 650.0, cons)
    fast = max(s[5] for s in slices)
    assert fast < 30000.0, fast
    assert_under_raw(slices, SPEEDS, ACCELS)
    print(
        "  ramp peaks: 0->150 %.0f, 0->650 %.0f (per-move cap: 5000) OK"
        % (slow, fast)
    )


def test_notch_dist_bounds_emitter_with_curve():
    # A curved ramp IS the plain notch ramp at its own accel A, in both the
    # planner and the emitter -- so it inherits every property the notch law
    # already has, rather than needing new ones. And the planner's distance
    # never undercuts what the emitter integrates (the lookahead may only be
    # conservative).
    rng = random.Random(4)
    n = 0
    for _ in range(80):
        speeds, accels = random_curve(rng)
        curve = pathplan.AccelCurve(speeds, accels)
        f1 = rng.uniform(35.0, 90.0)
        f2 = rng.choice([None, f1 * rng.uniform(1.05, 1.5)])
        cons = make_cons(curve, f1, f2)
        for _ in range(6):
            v0 = rng.uniform(0.0, 600.0)
            v1 = v0 + rng.uniform(1.0, 651.0 - v0)
            _s, d_emit = pathplan._ramp_up_jerk(v0, v1, cons)
            d_plan = pathplan.notch_dist(
                v0, v1, 100000.0, f1, f2, curve, cons.jerk_dt
            )
            assert d_plan >= d_emit - 1e-8, (v0, v1, d_emit, d_plan)
            a = cons.ramp_accel(v0, v1)
            plain = make_cons(None, f1, f2, a_const=a)
            assert pathplan._ramp_up_jerk(v0, v1, plain)[1] == d_emit
            assert pathplan.notch_dist(v0, v1, a, f1, f2) == d_plan
            n += 1
    print(
        "  %d curved ramps == plain notch ramps at their A; plan >= emit OK" % n
    )


def main():
    test_envelope_only_lowers_and_limits_fall()
    test_rendered_ramps_stay_under_the_curve()
    test_curve_that_never_binds_changes_nothing()
    test_reach_is_always_renderable()
    test_slow_ramps_get_more_than_a_per_move_cap()
    test_notch_dist_bounds_emitter_with_curve()
    print("ALL PASS")


if __name__ == "__main__":
    main()
