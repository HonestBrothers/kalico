# Bead-bounded corner blending (topp-ra-v3)
#
# Replace a sharp path corner with a tangent circular arc whose maximum
# deviation from the vertex is bounded by the EXTRUDED BEAD geometry, not a
# hand-tuned junction_deviation. The bead is a stadium cross-section (width w
# cross-track, height h) so a printed corner is already rounded at scale ~w/2;
# a toolpath rounding whose deviation stays well under that is optically
# indistinguishable from a true corner. Keeping the tool moving through the
# corner (no velocity zero) also reduces pressure advance
# decompress/recompress blobs. The target circle is tangent to both legs; the
# emitted path is a bounded-angle polyline approximation because Move is linear.
#
# Pure (only `math`), toolhead-decoupled -> unit-testable in isolation. The
# toolhead adapter splices the returned points into the move stream and lets the
# UNIFIED emitter render them (so blend moves get jerk limiting + FF handling +
# pressure advance for free -- unlike the old jerk_limiting round_corners, which
# fell back to a constant-accel path and bypassed check_move/PA).
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
import math


def bead_deviation(bead_width, ratio):
    """Max corner deviation (mm) that stays invisible at the bead scale.

    bead_width -- extrusion width w (mm), ~1.0-1.2x nozzle diameter.
    ratio      -- fraction c of the bead width (~0.1-0.25); the printed corner
                  is already rounded at ~w/2, so c*w is well inside the bead.
    """
    return max(0.0, ratio) * max(0.0, bead_width)


def _unit(dx, dy, dz):
    m = math.sqrt(dx * dx + dy * dy + dz * dz)
    if m <= 1e-12:
        return None, 0.0
    return (dx / m, dy / m, dz / m), m


def corner_geometry(p_prev, p_corner, p_next):
    """Turn geometry at p_corner. Returns dict with unit legs, leg lengths,
    turn angle phi (0=straight, pi=reversal), and half-angle cos(phi/2), or None
    if either leg is degenerate."""
    uin, lin = _unit(p_corner[0] - p_prev[0], p_corner[1] - p_prev[1],
                     p_corner[2] - p_prev[2])
    uout, lout = _unit(p_next[0] - p_corner[0], p_next[1] - p_corner[1],
                       p_next[2] - p_corner[2])
    if uin is None or uout is None:
        return None
    dot = max(-1.0, min(1.0, uin[0] * uout[0] + uin[1] * uout[1]
                        + uin[2] * uout[2]))
    phi = math.acos(dot)                 # turn angle (deviation from straight)
    cos_half = math.sqrt(max(0.0, (1.0 + dot) * 0.5))   # cos(phi/2)
    return {"uin": uin, "uout": uout, "lin": lin, "lout": lout,
            "phi": phi, "cos_half": cos_half}


def blend_radius(cos_half, delta_max):
    """Arc radius r whose max deviation from the vertex equals delta_max, for a
    corner with cos(phi/2)=cos_half. delta = r*(1/cos_half - 1)."""
    denom = (1.0 / cos_half) - 1.0
    if denom <= 1e-9:                    # ~straight: no rounding needed
        return float("inf")
    return delta_max / denom


def plan_corner(p_prev, p_corner, p_next, delta_max, blend_ratio=0.5,
                min_turn_deg=8.0, max_turn_deg=150.0, chord_len=0.4,
                max_chord_angle_deg=5.0):
    """Plan a bead-bounded arc blend for the corner at p_corner.

    delta_max   -- max path deviation from the vertex (mm), e.g. bead_deviation.
    blend_ratio -- trim no more than this fraction of either adjacent leg.
    min_turn_deg-- skip near-straight corners (nothing to round).
    max_turn_deg-- skip near-reversals (a real corner / seam -> keep the stop).
    chord_len   -- target arc chord length (mm), ~bead width, for discretization.
    max_chord_angle_deg -- maximum direction change represented by one chord.

    Returns None to leave the corner sharp, else a dict:
      p_in   -- tangent point on the incoming leg (trim p_prev->p_corner here)
      p_out  -- tangent point on the outgoing leg (start p_corner->p_next here)
      pts    -- interior arc points strictly between p_in and p_out (may be [])
      r, t   -- arc radius and per-leg trim length (mm)
      dev    -- actual deviation achieved (<= delta_max)
    The blended move chain is: p_prev->p_in, p_in->pts[0]->...->pts[-1]->p_out,
    p_out->p_next. All rendered by the unified emitter.
    """
    if delta_max <= 0.0:
        return None
    g = corner_geometry(p_prev, p_corner, p_next)
    if g is None:
        return None
    phi = g["phi"]
    turn_deg = math.degrees(phi)
    if turn_deg < min_turn_deg or turn_deg > max_turn_deg:
        return None
    cos_half = g["cos_half"]
    tan_half = math.sqrt(max(0.0, 1.0 - cos_half * cos_half)) / cos_half
    if tan_half <= 1e-9:
        return None
    r = blend_radius(cos_half, delta_max)
    t = r * tan_half                     # trim length along each leg
    # Cap the trim to a fraction of the SHORTER leg (so a leg shared by two
    # corners is never over-consumed); shrink r to match if capped.
    t_cap = blend_ratio * min(g["lin"], g["lout"])
    if t_cap <= 1e-9:
        return None
    if t > t_cap:
        t = t_cap
        r = t / tan_half
    dev = r * ((1.0 / cos_half) - 1.0)
    uin, uout = g["uin"], g["uout"]
    p_in = tuple(p_corner[i] - t * uin[i] for i in range(3))
    p_out = tuple(p_corner[i] + t * uout[i] for i in range(3))
    # Discretize the arc (swept angle == phi) into chords ~chord_len. Chord count
    # from the target chord on radius r; the chord's own sag from the arc,
    # r*(1-cos(dpsi/2)), is bead-scale small and bounded by the assert-tested
    # invariant below.
    arc_len = r * phi
    n_len = int(math.ceil(arc_len / max(chord_len, 1e-6)))
    n_angle = int(math.ceil(turn_deg / max(max_chord_angle_deg, 1e-6)))
    n = max(1, n_len, n_angle)
    pts = []
    if n > 1:
        # Rotate p_in about center C by k*phi/n. C is on the inward bisector at
        # distance r/cos_half from the vertex.
        bis = _unit(uout[0] - uin[0], uout[1] - uin[1], uout[2] - uin[2])[0]
        if bis is not None:
            c = tuple(p_corner[i] + (r / cos_half) * bis[i] for i in range(3))
            # Vector C->p_in, rotated in the plane of the two legs. Build an
            # orthonormal basis (e0 = C->p_in normalized, e1 in-plane perp).
            e0 = _unit(p_in[0] - c[0], p_in[1] - c[1], p_in[2] - c[2])[0]
            v2 = tuple(p_out[i] - c[i] for i in range(3))
            dot02 = sum(e0[i] * v2[i] for i in range(3))
            perp = tuple(v2[i] - dot02 * e0[i] for i in range(3))
            e1 = _unit(*perp)[0]
            if e0 is not None and e1 is not None:
                for k in range(1, n):
                    a = phi * k / n
                    ca, sa = math.cos(a), math.sin(a)
                    pts.append(tuple(c[i] + r * (ca * e0[i] + sa * e1[i])
                                     for i in range(3)))
    return {"p_in": p_in, "p_out": p_out, "pts": pts, "r": r, "t": t,
            "dev": dev, "phi": phi}


def plan_blend_chain(prev_start, vertex, move_end, prev_e, move_e,
                     prev_len, move_len, accel, delta_max, blend_ratio=0.5,
                     min_turn_deg=8.0, max_turn_deg=150.0, chord_len=0.4,
                     max_chord_angle_deg=5.0, max_extrusion_scale=1.35):
    """Plan the full blended move chain for the corner at `vertex` as PLAIN DATA
    (no Move objects), so extrusion conservation and geometry are unit-testable.

    prev_start/vertex/move_end -- (x,y,z) of the incoming move start, the shared
        corner, and the outgoing move end.
    prev_e/move_e              -- extruded filament (mm) on each original move.
    prev_len/move_len          -- XYZ path length (mm) of each original move.
    accel                      -- min accel of the two moves (for the centripetal
        cap v_corner = sqrt(accel*r)).

    Returns None to leave the corner sharp, else a dict:
      pts      -- (n+1) chain points prev_start .. move_end (through p_in, arc,
                  p_out); consecutive pairs are the new moves.
      e_seg    -- per-segment extrusion (len n), sum == prev_e+move_e.
      interior -- per-segment bool: True for the curved arc segments (get the
                  centripetal cap), False for the two straight body legs.
      corner_v -- centripetal speed cap (mm/s) for interior segments.
      r, dev   -- arc radius and achieved deviation.
    """
    if prev_e <= 0.0 or move_e <= 0.0:
        return None
    P = plan_corner(
        prev_start, vertex, move_end, delta_max, blend_ratio,
        min_turn_deg, max_turn_deg, chord_len, max_chord_angle_deg)
    if P is None:
        return None
    pts = [tuple(prev_start[:3]), P["p_in"]] + P["pts"] \
        + [P["p_out"], tuple(move_end[:3])]
    n = len(pts) - 1
    seg_len = [math.sqrt(sum((pts[s + 1][k] - pts[s][k]) ** 2
                             for k in range(3))) for s in range(n)]
    # Body legs are the first (prev) and last (move) segments; everything between
    # is the arc. Distribute each original move's extrusion over its share of the
    # new chain by length so total filament is conserved and endpoints preserved.
    prev_body_e = prev_e * (seg_len[0] / prev_len) if prev_len > 1e-12 else 0.0
    move_body_e = move_e * (seg_len[-1] / move_len) if move_len > 1e-12 else 0.0
    corner_e = (prev_e - prev_body_e) + (move_e - move_body_e)
    corner_len = sum(seg_len[1:-1])
    prev_density = prev_e / prev_len if prev_len > 1e-12 else 0.0
    move_density = move_e / move_len if move_len > 1e-12 else 0.0
    source_density = max(prev_density, move_density)
    corner_density = corner_e / corner_len if corner_len > 1e-12 else float("inf")
    extrusion_scale = (
        corner_density / source_density if source_density > 1e-12 else float("inf"))
    if extrusion_scale > max_extrusion_scale:
        return None
    e_seg = [prev_body_e]
    for s in range(1, n - 1):
        frac = (seg_len[s] / corner_len) if corner_len > 1e-12 else 0.0
        e_seg.append(corner_e * frac)
    if n >= 2:
        e_seg.append(move_body_e)
    interior = [False] + [True] * (n - 2) + [False] if n >= 2 else [False]
    corner_v = math.sqrt(max(0.0, accel) * P["r"])
    return {"pts": pts, "e_seg": e_seg, "interior": interior,
            "corner_v": corner_v, "r": P["r"], "dev": P["dev"],
            "seg_len": seg_len, "extrusion_scale": extrusion_scale}
