# Bead-bounded corner blending
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
# The geometry half of this file is pure (only `math`) and printer-decoupled, so
# it is unit-testable in isolation. The CornerBlend object below owns the config
# section and the printer-dependent parts (bead width from the nozzle, layer
# height from the move stream) and hands the toolhead a chain of ordinary Move
# objects. Those go through the normal lookahead, so they get check_move,
# junction planning and pressure advance -- unlike the old jerk_limiting
# round_corners, which rendered blends on a private constant-accel path that
# bypassed check_move and spiked PA.
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
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
    uin, lin = _unit(
        p_corner[0] - p_prev[0],
        p_corner[1] - p_prev[1],
        p_corner[2] - p_prev[2],
    )
    uout, lout = _unit(
        p_next[0] - p_corner[0],
        p_next[1] - p_corner[1],
        p_next[2] - p_corner[2],
    )
    if uin is None or uout is None:
        return None
    dot = max(
        -1.0, min(1.0, uin[0] * uout[0] + uin[1] * uout[1] + uin[2] * uout[2])
    )
    phi = math.acos(dot)  # turn angle (deviation from straight)
    cos_half = math.sqrt(max(0.0, (1.0 + dot) * 0.5))  # cos(phi/2)
    return {
        "uin": uin,
        "uout": uout,
        "lin": lin,
        "lout": lout,
        "phi": phi,
        "cos_half": cos_half,
    }


def blend_radius(cos_half, delta_max):
    """Arc radius r whose max deviation from the vertex equals delta_max, for a
    corner with cos(phi/2)=cos_half. delta = r*(1/cos_half - 1)."""
    denom = (1.0 / cos_half) - 1.0
    if denom <= 1e-9:  # ~straight: no rounding needed
        return float("inf")
    return delta_max / denom


def plan_corner(
    p_prev,
    p_corner,
    p_next,
    delta_max,
    blend_ratio=0.5,
    min_turn_deg=8.0,
    max_turn_deg=150.0,
    chord_len=0.4,
    max_chord_angle_deg=5.0,
):
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
    t = r * tan_half  # trim length along each leg
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
                    pts.append(
                        tuple(
                            c[i] + r * (ca * e0[i] + sa * e1[i])
                            for i in range(3)
                        )
                    )
    return {
        "p_in": p_in,
        "p_out": p_out,
        "pts": pts,
        "r": r,
        "t": t,
        "dev": dev,
        "phi": phi,
    }


def plan_blend_chain(
    prev_start,
    vertex,
    move_end,
    prev_e,
    move_e,
    prev_len,
    move_len,
    accel,
    delta_max,
    blend_ratio=0.5,
    min_turn_deg=8.0,
    max_turn_deg=150.0,
    chord_len=0.4,
    max_chord_angle_deg=5.0,
    max_extrusion_scale=1.35,
):
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
        prev_start,
        vertex,
        move_end,
        delta_max,
        blend_ratio,
        min_turn_deg,
        max_turn_deg,
        chord_len,
        max_chord_angle_deg,
    )
    if P is None:
        return None
    pts = (
        [tuple(prev_start[:3]), P["p_in"]]
        + P["pts"]
        + [P["p_out"], tuple(move_end[:3])]
    )
    n = len(pts) - 1
    seg_len = [
        math.sqrt(sum((pts[s + 1][k] - pts[s][k]) ** 2 for k in range(3)))
        for s in range(n)
    ]
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
    corner_density = (
        corner_e / corner_len if corner_len > 1e-12 else float("inf")
    )
    extrusion_scale = (
        corner_density / source_density
        if source_density > 1e-12
        else float("inf")
    )
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
    return {
        "pts": pts,
        "e_seg": e_seg,
        "interior": interior,
        "corner_v": corner_v,
        "r": P["r"],
        "dev": P["dev"],
        "seg_len": seg_len,
        "extrusion_scale": extrusion_scale,
    }


class CornerBlend:
    """The [corner_blend] config section.

    Owns the tunables plus the two printer-dependent inputs the geometry needs:
    the bead width (from the nozzle, or inferred per move from that move's own
    extrusion) and the layer height (pinned, or derived from the move stream).
    """

    def __init__(self, config):
        self.printer = config.get_printer()
        self.enabled = config.getboolean("enable", True)
        # Bead width: 0 (default) derives it from the extruder's
        # nozzle_diameter * bead_ratio at first use (a typical extrusion width
        # is 1.0-1.2x the nozzle). Set bead_width > 0 to pin a literal width.
        self.bead_width = config.getfloat("bead_width", 0.0, minval=0.0)
        self.bead_ratio = config.getfloat("bead_ratio", 1.125, above=0.0)
        self._bead_w = None  # resolved lazily from the nozzle
        # Layer height for per-move extrusion-width inference: 0 (default)
        # DERIVES it from the Z rise between extruding moves; set > 0 to pin it.
        # Given a layer height, each move's real bead width follows from its own
        # extrusion as filament_area*E/(move_d*h), so thin gap-fill beads get a
        # proportionally tighter deviation bound with no extra tuning.
        self.layer_height = config.getfloat("layer_height", 0.0, minval=0.0)
        self._layer_height = 0.0  # derived from the move stream
        self._last_extrude_z = None
        self.deviation_ratio = config.getfloat(
            "deviation_ratio", 0.15, above=0.0, maxval=0.5
        )
        self.blend_ratio = config.getfloat(
            "blend_ratio", 0.4, above=0.0, maxval=0.5
        )
        self.min_turn = config.getfloat(
            "min_turn", 8.0, minval=0.0, below=180.0
        )
        self.max_turn = config.getfloat(
            "max_turn", 150.0, minval=0.0, below=180.0
        )
        if self.min_turn > self.max_turn:
            raise config.error("min_turn must not exceed max_turn")
        self.chord_ratio = config.getfloat("chord_ratio", 1.0, above=0.0)
        self.max_chord_angle = config.getfloat(
            "max_chord_angle", 5.0, above=0.0, maxval=45.0
        )
        self.max_extrusion_scale = config.getfloat(
            "max_extrusion_scale", 1.35, minval=1.0
        )
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SET_CORNER_BLEND",
            self.cmd_SET_CORNER_BLEND,
            desc=self.cmd_SET_CORNER_BLEND_help,
        )

    # Live-tunable fields, as (gcode parameter, attribute, getter kwargs). The
    # bounds match the config parser's, so a value rejected at startup is also
    # rejected here.
    _TUNABLES = (
        ("BEAD_WIDTH", "bead_width", {"minval": 0.0}),
        ("BEAD_RATIO", "bead_ratio", {"above": 0.0}),
        ("LAYER_HEIGHT", "layer_height", {"minval": 0.0}),
        ("DEVIATION_RATIO", "deviation_ratio", {"above": 0.0, "maxval": 0.5}),
        ("BLEND_RATIO", "blend_ratio", {"above": 0.0, "maxval": 0.5}),
        ("MIN_TURN", "min_turn", {"minval": 0.0, "below": 180.0}),
        ("MAX_TURN", "max_turn", {"minval": 0.0, "below": 180.0}),
        ("CHORD_RATIO", "chord_ratio", {"above": 0.0}),
        ("MAX_CHORD_ANGLE", "max_chord_angle", {"above": 0.0, "maxval": 45.0}),
        ("MAX_EXTRUSION_SCALE", "max_extrusion_scale", {"minval": 1.0}),
    )

    def _state(self):
        return (self.enabled,) + tuple(
            getattr(self, attr) for _, attr, _ in self._TUNABLES
        )

    def _restore(self, state):
        self.enabled = state[0]
        for (_, attr, _), value in zip(self._TUNABLES, state[1:]):
            setattr(self, attr, value)

    cmd_SET_CORNER_BLEND_help = "Set bead-bounded corner blending parameters"

    def cmd_SET_CORNER_BLEND(self, gcmd):
        en = gcmd.get_int("ENABLE", None, minval=0, maxval=1)
        values = [
            (attr, gcmd.get_float(name, None, **kw))
            for name, attr, kw in self._TUNABLES
        ]
        # Flush pending moves so the change only affects moves planned after
        # this point (same live-mutation contract as SET_VELOCITY_LIMIT).
        self.printer.lookup_object("toolhead").flush_step_generation()
        old = self._state()
        try:
            if en is not None:
                self.enabled = bool(en)
            for attr, value in values:
                if value is not None:
                    setattr(self, attr, value)
            if self.min_turn > self.max_turn:
                raise gcmd.error("MIN_TURN must not exceed MAX_TURN")
        except:
            self._restore(old)
            raise
        # The nozzle-derived width is memoized, so a changed bead setting has to
        # drop it or the old width would outlive the command that replaced it.
        if self._state() != old:
            self._bead_w = None
        gcmd.respond_info(
            "corner_blend: enable=%d bead_width=%.4f bead_ratio=%.3f"
            " layer_height=%.4f deviation_ratio=%.4f blend_ratio=%.3f"
            " min_turn=%.1f max_turn=%.1f chord_ratio=%.3f"
            " max_chord_angle=%.1f max_extrusion_scale=%.3f"
            " [effective bead=%.4f]"
            % (
                self.enabled,
                self.bead_width,
                self.bead_ratio,
                self.layer_height or self._layer_height,
                self.deviation_ratio,
                self.blend_ratio,
                self.min_turn,
                self.max_turn,
                self.chord_ratio,
                self.max_chord_angle,
                self.max_extrusion_scale,
                self._static_bead_width(),
            )
        )

    def get_status(self, eventtime):
        return {
            "enabled": self.enabled,
            "bead_width": self.bead_width,
            "bead_ratio": self.bead_ratio,
            "layer_height": self.layer_height or self._layer_height,
            "deviation_ratio": self.deviation_ratio,
            "blend_ratio": self.blend_ratio,
            "min_turn": self.min_turn,
            "max_turn": self.max_turn,
            "chord_ratio": self.chord_ratio,
            "max_chord_angle": self.max_chord_angle,
            "max_extrusion_scale": self.max_extrusion_scale,
        }

    def _get_extruder(self):
        toolhead = self.printer.lookup_object("toolhead", None)
        if toolhead is None:
            return None
        return toolhead.get_extruder()

    def note_layer_height(self, move):
        # Derive the layer height as the Z rise between EXTRUDING moves. Z-hops
        # are travel moves, so they are skipped without a special case. Runs per
        # move, before any splice reshapes the stream.
        if not (move.is_kinematic_move and move.axes_d[3]):
            return
        z = move.end_pos[2]
        lz = self._last_extrude_z
        if lz is not None and z > lz + 1e-6:
            self._layer_height = z - lz
        if lz is None or abs(z - lz) > 1e-6:
            self._last_extrude_z = z

    def _extrusion_width(self, move, h):
        # Deposited bead width (mm) of an extruding move: its cross-section area
        # (filament_area * E / XY_len) divided by the layer height h. None when
        # the move carries no usable extrusion.
        e = move.axes_d[3]
        if e <= 0.0 or move.move_d <= 1e-9 or h <= 0.0:
            return None
        fd = getattr(self._get_extruder(), "filament_diameter", None)
        if not fd:
            return None
        fil_area = math.pi * (fd * 0.5) ** 2
        return fil_area * e / move.move_d / h

    def _static_bead_width(self):
        # Fallback width (mm): the literal bead_width when set, else
        # nozzle_diameter * bead_ratio, read from the extruder once.
        if self.bead_width > 0.0:
            return self.bead_width
        if self._bead_w is None:
            nd = getattr(self._get_extruder(), "nozzle_diameter", None)
            self._bead_w = nd * self.bead_ratio if nd else 0.45
            logging.info(
                "corner_blend: static bead_width=%.3f (nozzle=%s*%.3f)"
                % (self._bead_w, nd, self.bead_ratio)
            )
        return self._bead_w

    def corner_bead_width(self, prev, move):
        # Effective bead width for this corner: inferred per move from each
        # leg's own extrusion (the min of the two, so the tighter bead governs)
        # using the pinned or derived layer height, falling back to the static
        # nozzle-based width when neither is available.
        h = self.layer_height or self._layer_height
        if h > 0.0:
            ws = [
                w
                for w in (
                    self._extrusion_width(prev, h),
                    self._extrusion_width(move, h),
                )
                if w is not None
            ]
            if ws:
                return min(ws)
        return self._static_bead_width()

    def plan_chain(self, prev, move):
        # Bead-bounded corner blend for the sharp corner prev->move. Returns
        # [trimmed_prev, *arc_chords, trimmed_move] as ordinary Move objects, or
        # None to leave the corner sharp.
        if not (move.is_kinematic_move and prev.is_kinematic_move):
            return None
        # The bead bound is a print-geometry argument, so it only licenses
        # moving the path where a bead is actually being laid. Travels,
        # retracting wipes and mixed-extrusion transitions keep the exact
        # commanded corner.
        if move.axes_d[3] <= 0.0 or prev.axes_d[3] <= 0.0:
            return None
        # A timing callback marks the original endpoint as a semantic boundary
        # for a synchronized pin/fan/etc. change. Trimming that endpoint away
        # would move the event, so leave the corner sharp.
        if prev.timing_callbacks:
            return None
        toolhead = move.toolhead
        # The chain carries one extruder coordinate (index 3). With further
        # extra axes queued, a synthesized position tuple would be narrower than
        # commanded_pos and those axes would silently stop tracking, so leave
        # those corners sharp rather than guess how to split them.
        if len(toolhead.extra_axes) != 1:
            return None
        bead_w = self.corner_bead_width(prev, move)
        ch = plan_blend_chain(
            prev.start_pos,
            prev.end_pos[:3],
            move.end_pos,
            prev.axes_d[3],
            move.axes_d[3],
            prev.move_d,
            move.move_d,
            min(prev.accel, move.accel),
            bead_deviation(bead_w, self.deviation_ratio),
            blend_ratio=self.blend_ratio,
            min_turn_deg=self.min_turn,
            max_turn_deg=self.max_turn,
            chord_len=self.chord_ratio * bead_w,
            max_chord_angle_deg=self.max_chord_angle,
            max_extrusion_scale=self.max_extrusion_scale,
        )
        if ch is None:
            return None
        Move = type(move)
        pts, e_seg = ch["pts"], ch["e_seg"]
        interior, seg_len = ch["interior"], ch["seg_len"]
        corner_v = ch["corner_v"]
        prev_speed = math.sqrt(prev.max_cruise_v2)
        move_speed = math.sqrt(move.max_cruise_v2)
        new_moves = []
        e_abs = prev.start_pos[3]
        for s in range(len(e_seg)):
            if seg_len[s] <= 1e-9:
                e_abs += e_seg[s]
                continue
            start = (pts[s][0], pts[s][1], pts[s][2], e_abs)
            e_abs += e_seg[s]
            end = (pts[s + 1][0], pts[s + 1][1], pts[s + 1][2], e_abs)
            nm = Move(
                toolhead, start, end, prev_speed if s == 0 else move_speed
            )
            if interior[s]:
                nm._corner_blend = True
                # Cap every chord to the circular path's centripetal speed.
                # Junction handling applies the tighter polyline-turn limits.
                if corner_v > 0.0 and corner_v * corner_v < nm.max_cruise_v2:
                    nm.limit_speed(corner_v, nm.accel)
            new_moves.append(nm)
        if len(new_moves) < 2:
            return None
        return new_moves


def load_config(config):
    return CornerBlend(config)
