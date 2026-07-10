# Regression: jerk-aware lookahead makes every emitted move jerk-feasible, so
# the model-inverse FF's p2*a term never becomes a multi-step position jump.
#
# This reproduces the 2026-07-10 crash class end-to-end: a string of short fast
# moves (the polygon-circle in ff_stress.gcode) that, planned with the STOCK
# constant-accel reach, are jerk-INfeasible -> emitter jerk-raise truncates ->
# ~4-step FF jump -> stepcompress. With the jerk-aware reach the same string is
# planned at feasible boundary speeds and emits cleanly jerk-limited.
#
# Run: /home/brandon/klippy-env/bin/python klippy/extras/test_jerk_lookahead.py
import math
import pathplan as pp


def _cons(A, J, max_da):
    c = pp.Constraints(a_of_v=None, a_const=float(A), v_ceil=400.0,
                       dv_slice=1e18, max_jerk=float(J), jerk_dt=0.001,
                       max_da=max_da)
    return c


def _lookahead(move_d, v_cap, v_junction, A, J):
    # Mirror ToolHead.LookAheadQueue.flush()/calc_junction with the jerk-aware
    # reach (pathplan.jerk_reach_v2): backward decel pass then forward accel
    # pass over squared boundary speeds, then a jerk peak per move.
    n = len(move_d)
    vb = [0.0] * (n + 1)
    for k in range(n + 1):
        cap = v_junction[k]
        if k > 0:
            cap = min(cap, v_cap[k - 1])
        if k < n:
            cap = min(cap, v_cap[k])
        vb[k] = cap
    for i in range(n - 1, -1, -1):
        r = math.sqrt(pp.jerk_reach_v2(vb[i + 1] ** 2, move_d[i], A, J, 400.0))
        vb[i] = min(vb[i], r)
    for i in range(n):
        r = math.sqrt(pp.jerk_reach_v2(vb[i] ** 2, move_d[i], A, J, 400.0))
        vb[i + 1] = min(vb[i + 1], r)
    vs = [vb[i] for i in range(n)]
    ve = [vb[i + 1] for i in range(n)]
    vc = []
    for i in range(n):
        # jerk peak reachable from each end over the move (matches flush clamp)
        pk = min(pp.jerk_reach_v2(vs[i] ** 2, move_d[i], A, J, 400.0),
                 pp.jerk_reach_v2(ve[i] ** 2, move_d[i], A, J, 400.0))
        vc.append(min(v_cap[i], math.sqrt(pk)))
    return vs, vc, ve


def _emit_chain_stats(vs, vc, ve, move_d, cons, p2):
    # Emit the whole chain; return (used_sharp_or_raise, worst_ff_jump_mm).
    used_bad = False
    prev_sa = 0.0
    worst_ff = 0.0
    for i in range(len(move_d)):
        # base-jerk profile MUST exist (no jerk-raise, no sharp fallback)
        base = pp._emit_jerk_core(vs[i], vc[i], ve[i], move_d[i], cons)
        if base is None:
            used_bad = True
        segs = pp.emit_profile(vs[i], vc[i], ve[i], move_d[i], cons)
        assert segs, "empty emit"
        # invariant: velocity continuity + exact distance
        pend = vs[i]
        dsum = 0.0
        for (at, ct, dt, sv, cvv, a, dist) in segs:
            assert abs(sv - pend) < 1e-6, "velocity gap within move"
            sa = (-a if dt > 0.0 else (a if at > 0.0 else 0.0))
            worst_ff = max(worst_ff, abs(sa - prev_sa) * p2)
            prev_sa = sa
            # end velocity: accel -> cruise_v; decel -> start_v - a*decel_t;
            # cruise -> cruise_v.
            pend = (cvv - a * dt) if dt > 0.0 else cvv
            dsum += dist
        assert abs(dsum - move_d[i]) < 1e-6, "distance mismatch"
    return used_bad, worst_ff


def test_polygon_circle_no_sharp_fallback():
    # The ff_stress polygon circle: 0.628mm segs, requested v climbing to 266,
    # Y accel 15000, jerk 100000. Junctions request full speed (worst case).
    A, J = 15000.0, 100000.0
    p2 = 1.0 / (2.0 * math.pi * 77.0) ** 2   # 77 Hz Y mode
    step = 0.0125
    max_da = 0.1 * step / p2
    cons = _cons(A, J, max_da)
    n = 40
    move_d = [0.628] * n
    v_cap = [266.7] * n
    v_junction = [0.0] + [266.7] * (n - 1) + [0.0]
    vs, vc, ve = _lookahead(move_d, v_cap, v_junction, A, J)
    used_bad, worst_ff = _emit_chain_stats(vs, vc, ve, move_d, cons, p2)
    assert not used_bad, "a move needed jerk-raise/sharp fallback"
    assert worst_ff < step, "FF jump %.5f >= step %.5f" % (worst_ff, step)
    print("  polygon circle: peak cruise=%.1f mm/s (capped from 266.7), "
          "no sharp fallback, worst FF jump=%.5fmm (<%.5f) OK"
          % (max(vc), worst_ff, step))


def test_stock_reach_would_have_crashed():
    # Contrast: plan the SAME chain with the stock constant-accel reach and show
    # the emitter is forced into the jerk-raise/sharp path (the old crash).
    A, J = 15000.0, 100000.0
    p2 = 1.0 / (2.0 * math.pi * 77.0) ** 2
    step = 0.0125
    cons = _cons(A, J, 0.1 * step / p2)
    n = 40
    move_d = [0.628] * n
    # stock reach: v^2 + 2*A*d (no jerk awareness)
    vb = [0.0] * (n + 1)
    cap = 266.7
    for i in range(n - 1, -1, -1):
        vb[i] = min(cap, math.sqrt(vb[i + 1] ** 2 + 2 * A * move_d[i]))
    for i in range(n):
        vb[i + 1] = min(vb[i + 1], math.sqrt(vb[i] ** 2 + 2 * A * move_d[i]))
    vs = [vb[i] for i in range(n)]
    ve = [vb[i + 1] for i in range(n)]
    vc = [cap] * n
    raised = 0
    for i in range(n):
        if pp._emit_jerk_core(vs[i], vc[i], ve[i], move_d[i], cons) is None:
            raised += 1
    assert raised > 0, "expected stock plan to be jerk-infeasible somewhere"
    print("  stock reach: %d/%d moves jerk-infeasible -> would jerk-raise/sharp "
          "(the crash) OK" % (raised, n))


def test_jerk_reach_conservative_vs_emitter():
    # jerk_dist (used by the lookahead) must be >= the emitter's ramp distance
    # so the lookahead never over-plans past what the emitter can render.
    J = 100000.0
    bad = 0
    for A in (5000, 10000, 15000, 20000):
        cons = _cons(A, J, 293.0)
        for v0 in range(0, 300, 15):
            for dv in range(2, 160, 5):
                _, d = pp._ramp_up_jerk(float(v0), float(v0 + dv), cons,
                                        collect=False)
                if d == float("inf"):
                    continue
                if pp.jerk_dist(float(v0), float(v0 + dv), float(A), J) < d - 1e-9:
                    bad += 1
    assert bad == 0, "%d optimistic jerk_dist cases" % bad
    print("  jerk_dist conservative vs emitter ramp (all cases) OK")


if __name__ == "__main__":
    test_jerk_reach_conservative_vs_emitter()
    test_stock_reach_would_have_crashed()
    test_polygon_circle_no_sharp_fallback()
    print("ALL PASS")
