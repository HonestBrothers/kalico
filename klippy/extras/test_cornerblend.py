# Bead-bounded corner blending geometry tests.
# Run: /home/brandon/klippy-env/bin/python klippy/extras/test_cornerblend.py
import math
import cornerblend as cb


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _point_to_seg(p, a, b):
    # distance from point p to segment a-b (3D)
    ab = [b[i] - a[i] for i in range(3)]
    L2 = sum(c * c for c in ab)
    if L2 <= 1e-18:
        return _dist(p, a)
    t = max(0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / L2))
    proj = [a[i] + t * ab[i] for i in range(3)]
    return _dist(p, proj)


def test_bead_deviation():
    assert abs(cb.bead_deviation(0.45, 0.2) - 0.09) < 1e-12
    assert cb.bead_deviation(0.45, 0.0) == 0.0
    assert cb.bead_deviation(0.0, 0.2) == 0.0
    print("  bead_deviation = ratio*width OK")


def test_corner_geometry_angles():
    g = cb.corner_geometry((0, 0, 0), (10, 0, 0), (20, 0, 0))
    assert abs(g["phi"]) < 1e-9, "straight -> phi 0"
    g = cb.corner_geometry((0, 0, 0), (10, 0, 0), (10, 10, 0))
    assert abs(math.degrees(g["phi"]) - 90.0) < 1e-6, "right angle -> 90"
    assert abs(g["cos_half"] - math.cos(math.radians(45))) < 1e-9
    g = cb.corner_geometry((0, 0, 0), (10, 0, 0), (0, 0, 0))
    assert abs(math.degrees(g["phi"]) - 180.0) < 1e-6, "reversal -> 180"
    print("  corner_geometry angles (0/90/180) OK")


def test_blend_radius_matches_deviation():
    # 90 deg: delta = r*(1/cos45 - 1) = r*0.41421
    cos_half = math.cos(math.radians(45))
    r = cb.blend_radius(cos_half, 0.1)
    assert abs(r * (1.0 / cos_half - 1.0) - 0.1) < 1e-12
    # near-straight -> very large radius (shallow corner needs little rounding)
    assert cb.blend_radius(math.cos(math.radians(0.1)), 0.1) > 1e3
    print("  blend_radius inverts the deviation formula OK")


def test_plan_corner_90_geometry():
    delta = 0.1
    P = cb.plan_corner((0, 0, 0), (10, 0, 0), (10, 10, 0), delta,
                       blend_ratio=0.5, chord_len=0.05)
    assert P is not None
    assert P["dev"] <= delta + 1e-9, "deviation within bound"
    # tangent points lie on the legs, trimmed by t
    assert abs(_dist(P["p_in"], (10, 0, 0)) - P["t"]) < 1e-9
    assert abs(_dist(P["p_out"], (10, 0, 0)) - P["t"]) < 1e-9
    assert P["p_in"][0] < 10.0 and abs(P["p_in"][1]) < 1e-9  # on incoming leg
    assert abs(P["p_out"][0] - 10.0) < 1e-9 and P["p_out"][1] > 0.0
    # every arc point (and endpoints) is exactly r from the arc center
    cos_half = math.cos(math.radians(45))
    bis = [(-1) / math.sqrt(2), 1 / math.sqrt(2), 0.0]
    vertex = (10.0, 0.0, 0.0)
    C = [vertex[i] + (P["r"] / cos_half) * bis[i] for i in range(3)]
    for q in [P["p_in"]] + P["pts"] + [P["p_out"]]:
        assert abs(_dist(q, C) - P["r"]) < 1e-6, "arc point off-radius"
    # the full blended polyline never deviates from the vertex more than delta
    poly = [(0, 0, 0), P["p_in"]] + P["pts"] + [P["p_out"], (10, 10, 0)]
    worst = min(_point_to_seg((10, 0, 0), poly[i], poly[i + 1])
                for i in range(len(poly) - 1))
    # worst (closest approach of vertex to the path) == achieved deviation
    assert abs(worst - P["dev"]) < 2e-3, "polyline deviation %.4f vs %.4f" % (
        worst, P["dev"])
    print("  plan_corner 90deg: r=%.4f t=%.4f dev=%.4f nchord=%d, arc on-radius,"
          " polyline within delta OK" % (P["r"], P["t"], P["dev"],
                                         len(P["pts"]) + 1))


def test_plan_corner_skips():
    # near-straight -> None
    assert cb.plan_corner((0, 0, 0), (10, 0, 0), (20, 0.5, 0), 0.1,
                          min_turn_deg=8.0) is None
    # near-reversal -> None (real corner / seam, keep the stop)
    assert cb.plan_corner((0, 0, 0), (10, 0, 0), (0.5, 0.2, 0), 0.1,
                          max_turn_deg=150.0) is None
    # zero deviation budget -> None
    assert cb.plan_corner((0, 0, 0), (10, 0, 0), (10, 10, 0), 0.0) is None
    print("  plan_corner skips straight / reversal / zero-budget OK")


def test_trim_capped_by_leg():
    # short legs: trim must not exceed blend_ratio * shorter leg, r shrinks.
    delta = 5.0  # huge budget so the leg cap binds, not delta
    P = cb.plan_corner((0, 0, 0), (1.0, 0, 0), (1.0, 1.0, 0), delta,
                       blend_ratio=0.25, chord_len=0.1)
    assert P is not None
    assert P["t"] <= 0.25 * 1.0 + 1e-9, "trim exceeds leg cap"
    assert P["dev"] <= delta + 1e-9
    print("  trim capped by blend_ratio*leg (t=%.4f <= 0.25) OK" % P["t"])


def test_chord_length_reasonable():
    P = cb.plan_corner((0, 0, 0), (10, 0, 0), (10, 10, 0), 0.2,
                       blend_ratio=0.5, chord_len=0.4)
    pts = [P["p_in"]] + P["pts"] + [P["p_out"]]
    for i in range(len(pts) - 1):
        assert _dist(pts[i], pts[i + 1]) <= 0.4 + 1e-6, "chord too long"
    print("  arc chords <= target chord_len OK")


if __name__ == "__main__":
    test_bead_deviation()
    test_corner_geometry_angles()
    test_blend_radius_matches_deviation()
    test_plan_corner_90_geometry()
    test_plan_corner_skips()
    test_trim_capped_by_leg()
    test_chord_length_reasonable()
    print("ALL PASS")
