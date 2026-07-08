# Standalone tests for pathplan.py (topp-ra-v3 unified planner).
# Run: klippy-env/bin/python klippy/extras/test_pathplan.py
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathplan  # noqa: E402


def check_segs(segs, move_d, want_vs, want_ve, tag):
    # The jl-stepguard invariant, enforced as a test on emit_profile output.
    eps_v = 1e-4
    total = 0.0
    prev_ev = None
    first_sv = None
    for k, (at, ct, dt, sv, cv, a, dist) in enumerate(segs):
        assert at >= -1e-9 and ct >= -1e-9 and dt >= -1e-9, (tag, "neg t", k)
        assert sv >= -eps_v and cv >= -eps_v, (tag, "neg v", k, sv, cv)
        ev = cv - a * dt
        assert ev >= -eps_v, (tag, "end vel<0", k, ev, cv, a, dt)
        d_impl = 0.5 * (sv + cv) * at + cv * ct + (cv * dt - 0.5 * a * dt * dt)
        assert abs(d_impl - dist) <= 1e-6 + 1e-4 * abs(dist), (
            tag, "dist", k, d_impl, dist)
        if prev_ev is not None:
            assert abs(sv - prev_ev) <= 1e-3, (tag, "discont", k, sv, prev_ev)
        else:
            first_sv = sv
        prev_ev = ev
        total += dist
    assert abs(total - move_d) <= 1e-3 + 1e-4 * move_d, (
        tag, "total", total, move_d)
    if segs:
        assert abs(first_sv - want_vs) <= 1e-3, (tag, "start", first_sv, want_vs)
        assert abs(prev_ev - want_ve) <= 1e-3, (tag, "end", prev_ev, want_ve)


def test_emit(cons):
    cases = [  # (vs, vc, ve, move_d, label)
        (0.0, 200.0, 0.0, 50.0, "trapezoid from rest"),
        (0.0, 300.0, 0.0, 2.0, "short triangle (peak clamps)"),
        (50.0, 250.0, 80.0, 40.0, "asymmetric start/end"),
        (100.0, 100.0, 100.0, 10.0, "pure cruise"),
        (0.0, 150.0, 120.0, 30.0, "accel then partial decel"),
    ]
    for vs, vc, ve, d, label in cases:
        segs = pathplan.emit_profile(vs, vc, ve, d, cons)
        # peak may reduce vc; recover the actual end/start we asked for.
        check_segs(segs, d, vs, ve, "emit:" + label)
    print("  emit_profile: %d cases OK" % len(cases))


def test_batch(cons):
    # 4 collinear-ish moves; interior junctions allow some speed, ends stop.
    move_d = [30.0, 5.0, 30.0, 20.0]
    v_cap = [300.0, 300.0, 120.0, 300.0]  # move 2 is a slow (e.g. melt) cap
    v_junction = [0.0, 250.0, 250.0, 250.0, 0.0]
    vs, vc, ve = pathplan.plan_batch(move_d, v_cap, v_junction, cons)
    n = len(move_d)
    for i in range(n):
        # caps respected
        assert vc[i] <= v_cap[i] + 1e-6, ("vcap", i, vc[i], v_cap[i])
        assert vs[i] <= v_cap[i] + 1e-6 and ve[i] <= v_cap[i] + 1e-6
        # boundary continuity: end of i == start of i+1
        if i + 1 < n:
            assert abs(ve[i] - vs[i + 1]) <= 1e-6, ("cont", i, ve[i], vs[i + 1])
        # feasible: can accel vs->vc and decel vc->ve within move_d
        need = pathplan.dist_change(vs[i], vc[i], cons) + \
            pathplan.dist_change(ve[i], vc[i], cons)
        assert need <= move_d[i] + 1e-6, ("infeasible", i, need, move_d[i])
        # and the emitted profile satisfies the invariant
        segs = pathplan.emit_profile(vs[i], vc[i], ve[i], move_d[i], cons)
        check_segs(segs, move_d[i], vs[i], ve[i], "batch move %d" % i)
    assert vs[0] == 0.0 and ve[-1] == 0.0, ("ends not stopped", vs[0], ve[-1])
    # the slow middle move must throttle the boundaries around it
    assert ve[1] <= 120.0 + 1e-6 and vs[2] <= 120.0 + 1e-6, "melt cap ignored"
    print("  plan_batch: %d moves, caps+continuity+feasibility OK" % n)
    print("    vs=%s" % ["%.1f" % x for x in vs])
    print("    vc=%s" % ["%.1f" % x for x in vc])
    print("    ve=%s" % ["%.1f" % x for x in ve])


def main():
    # (a) constant-accel (stock fallback, no curve)
    const = pathplan.Constraints(a_const=3000.0, v_ceil=400.0, dv_slice=25.0)
    # (b) torque curve: accel falls with speed (back-EMF), floored
    curve = pathplan.Constraints(
        a_of_v=lambda v: max(400.0, 20000.0 - 40.0 * v),
        a_const=400.0, v_ceil=400.0, dv_slice=10.0)
    for name, cons in (("const-accel", const), ("torque-curve", curve)):
        print("[%s]" % name)
        test_emit(cons)
        test_batch(cons)
    print("ALL PASS")


if __name__ == "__main__":
    main()
