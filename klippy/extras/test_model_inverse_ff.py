# Standalone tests for model_inverse_ff.py (topp-ra-v3 Stage 4 math).
# Run: klippy-env/bin/python klippy/extras/test_model_inverse_ff.py
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_inverse_ff as mff  # noqa: E402

TWO_PI = 2.0 * math.pi
# The measured Y mode on red: lightly damped ~77 Hz.
F0 = 77.0
Z0 = 0.019
WN0 = TWO_PI * F0


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol + 1e-9 * abs(b)


def test_coefficients():
    p = mff.FFParams(F0, Z0, robustness=0.0)
    assert approx(p.wn, WN0)
    assert approx(p.c1, 2.0 * Z0 / WN0)
    assert approx(p.c2, 1.0 / (WN0 * WN0))
    # r=0 -> single inverse: b1=c1, b2=c2, no higher order
    assert approx(p.b1, p.c1) and approx(p.b2, p.c2)
    assert p.b3 == 0.0 and p.b4 == 0.0
    # r=1 -> squared inverse F2 = F1^2
    q = mff.FFParams(F0, Z0, robustness=1.0)
    assert approx(q.b1, 2.0 * p.c1)
    assert approx(q.b2, p.c1 * p.c1 + 2.0 * p.c2)
    assert approx(q.b3, 2.0 * p.c1 * p.c2)
    assert approx(q.b4, p.c2 * p.c2)
    print("  coefficients: r=0 single / r=1 squared inverse OK")


def test_disabled_is_identity():
    p = mff.FFParams(0.0, Z0, robustness=1.0)  # freq 0 = mode off
    assert p.b1 == 0.0 and p.b2 == 0.0 and p.b3 == 0.0 and p.b4 == 0.0
    assert p.augment(5.0, 100.0, 20000.0, 1e6, 1e9) == 5.0
    print("  disabled mode: augment is identity OK")


def test_exact_cancellation():
    # Matched model, r=0: residual at the true pole must be ~0.
    p = mff.FFParams(F0, Z0, robustness=0.0)
    res = p.residual_gain(WN0, Z0)
    assert res < 1e-9, ("exact cancellation not ~0", res)
    # On the jw-axis the notch bottoms at 2*zeta (the zero is at the COMPLEX
    # pole -zeta*wn + j*wd, not on the imaginary axis) -- this is the real
    # cancellation depth against a lightly damped mode.
    ng = p.notch_gain(F0)
    assert approx(ng, 2.0 * Z0, tol=1e-3), (ng, 2.0 * Z0)
    print("  exact cancellation: residual@pole=%.2e notch@f0=%.4f(=2zeta) OK"
          % (res, ng))


def test_robustness_monotone():
    # b1 (and the higher-order content) grow with robustness.
    prev = -1.0
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = mff.FFParams(F0, Z0, robustness=r)
        hi = p.b3 + p.b4  # higher-order (jerk/snap) content
        assert hi >= prev - 1e-18, ("higher-order not monotone", r, hi, prev)
        prev = hi
    assert mff.FFParams(F0, Z0, 1.0).b1 > mff.FFParams(F0, Z0, 0.0).b1
    print("  robustness monotonicity: higher-order content grows with r OK")


def test_robustness_flattens_sensitivity():
    # THE robustness proof: sweep the TRUE natural frequency over a drift band
    # around the model. The derivative-matched inverse (r=1) has a lower
    # WORST-CASE residual across the band than exact cancellation (r=0),
    # despite r=0 being deeper exactly at the center. (Flat-bottomed notch.)
    exact = mff.FFParams(F0, Z0, robustness=0.0)
    robust = mff.FFParams(F0, Z0, robustness=1.0)
    band = [1.0 + d for d in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)]
    res_exact = [exact.residual_gain(k * WN0, Z0) for k in band]
    res_robust = [robust.residual_gain(k * WN0, Z0) for k in band]
    worst_exact = max(res_exact)
    worst_robust = max(res_robust)
    # both zero at the matched center
    assert res_exact[3] < 1e-9 and res_robust[3] < 1e-9
    # robust wins worst-case over the drift band
    assert worst_robust < worst_exact, (worst_robust, worst_exact)
    # and dominates pointwise at every off-center probe
    for i, k in enumerate(band):
        if abs(k - 1.0) > 1e-12:
            assert res_robust[i] <= res_exact[i] + 1e-12, (k, res_robust[i],
                                                           res_exact[i])
    print("  robustness sensitivity: worst-case residual over +/-15%% band "
          "exact=%.3f robust=%.3f OK" % (worst_exact, worst_robust))


def test_headroom_and_inflation():
    p = mff.FFParams(F0, Z0, robustness=0.0)
    # position excursion from the correction is small at r=0
    exc = p.excursion(vel=300.0, accel=20000.0)
    assert exc == p.b1 * 300.0 + p.b2 * 20000.0
    assert 0.0 < exc < 0.5, exc  # sub-mm
    # motor accel inflation at r=0 is just b1*jerk (b2*snap etc = via extra)
    infl = p.motor_accel_extra(jerk=1e5, snap=0.0)
    assert approx(infl, p.b1 * 1e5)
    # position discontinuity from an accel jump = b2*da  (the crash quantity)
    da = 20000.0
    jump = p.pos_discontinuity(accel_jump=da)
    assert approx(jump, p.b2 * da)
    # r=1 makes the crash quantity depend on jerk jumps too (b3 term)
    q = mff.FFParams(F0, Z0, robustness=1.0)
    j2 = q.pos_discontinuity(accel_jump=da, jerk_jump=1e6)
    assert approx(j2, q.b2 * da + q.b3 * 1e6)
    print("  headroom/inflation: excursion=%.4fmm accel_jump_disc=%.4fmm OK"
          % (exc, jump))


def test_augment_reduces_to_shift_at_constant_v():
    # At constant velocity (a=0) the FF is a pure position lead b1*v -- an
    # identity on constant-speed motion up to a fixed spatial offset, exactly
    # like a shift-corrected shaper. No accel term contribution.
    p = mff.FFParams(F0, Z0, robustness=0.0)
    v = 120.0
    x = p.augment(pos=10.0, vel=v, accel=0.0)
    assert approx(x, 10.0 + p.b1 * v)
    print("  constant-velocity: pure lead b1*v (no distortion) OK")


def main():
    test_coefficients()
    test_disabled_is_identity()
    test_exact_cancellation()
    test_robustness_monotone()
    test_robustness_flattens_sensitivity()
    test_headroom_and_inflation()
    test_augment_reduces_to_shift_at_constant_v()
    print("ALL PASS")


if __name__ == "__main__":
    main()
