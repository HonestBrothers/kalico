# Model-inverse feedforward vibration compensation (topp-ra-v3, Stage 4)
#
# Replaces the time-domain input_shaper (a convolution / FIR notch) with a
# pointwise phase-plane correction applied to the motor command:
#
#     x_motor = y + b1*y' + b2*y'' + b3*y''' + b4*y''''      (per axis)
#
# where y is the desired tool-tip trajectory and the b_k are the Taylor
# coefficients of the inverse of the flexible mode
#
#     G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2).
#
# The exact inverse is F1(s) = (s^2 + 2*zeta*wn*s + wn^2)/wn^2 = 1 + c1*s +
# c2*s^2 with c1 = 2*zeta/wn, c2 = 1/wn^2 -- a pure differentiator, legal only
# because Klipper plans offline (the acausal stable inverse needs the full
# future). See docs/topp-ra-v3-design.md.
#
# ROBUSTNESS KNOB (r in [0,1]): F1 is a single zero on the mode pole -- deep
# but fragile (the classic "pole peeks out from behind the zero" when wn
# drifts). The frequency-robust form is the derivative-matched inverse
# F2(s) = F1(s)^2, which enforces BOTH F(jwn)=0 and dF/dw|wn = 0 (a flat-
# bottomed notch, ZVD-style). We interpolate F_r = (1-r)*F1 + r*F2, so r trades
# nominal depth for tolerance to wn error. The cost of r>0 is higher-order
# derivative content (jerk & snap) -> more motor a_max headroom and a smoother
# (C^3) trajectory. That is why r>0 rides on the jerk-limiting stage.
#
# Copyright (C) 2026
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math

TWO_PI = 2.0 * math.pi


class FFParams:
    """Pure-math model-inverse feedforward coefficients for one flexible mode.

    freq_hz     -- free-decay / natural frequency of the mode (Hz). For the
                   light damping we see (zeta ~ 0.02) the damped and natural
                   frequencies coincide to <0.05%, so freq_hz is used as wn/2pi.
    zeta        -- damping ratio of the mode (from the ring-down estimator).
    robustness  -- r in [0,1]: 0 = exact cancellation (F1), 1 = derivative-
                   matched robust inverse (F2 = F1^2). Interpolated linearly.

    Coefficients (x_motor = y + b1 y' + b2 y'' + b3 y''' + b4 y''''):
        c1 = 2 zeta / wn        c2 = 1 / wn^2
        F1 = 1 + c1 s + c2 s^2
        F2 = F1^2 = 1 + 2c1 s + (c1^2 + 2c2) s^2 + 2 c1 c2 s^3 + c2^2 s^4
        b1 = (1-r) c1 + r (2 c1)            = c1 (1 + r)
        b2 = (1-r) c2 + r (c1^2 + 2 c2)     = c2 (1 + r) + r c1^2
        b3 = r (2 c1 c2)
        b4 = r (c2^2)
    """

    def __init__(self, freq_hz, zeta, robustness=0.0, zeta_wide=0.15):
        self.freq_hz = float(freq_hz)
        self.zeta = float(zeta)
        self.robustness = min(1.0, max(0.0, float(robustness)))
        # Target zero damping at robustness=1 for the pointwise (2nd-order)
        # realization: a wider, shallower notch tolerant of parameter error.
        self.zeta_wide = float(zeta_wide)
        self._update()

    def _update(self):
        self.wn = TWO_PI * self.freq_hz
        if self.wn <= 0.0:
            # Disabled mode: identity transform.
            self.c1 = self.c2 = 0.0
            self.b1 = self.b2 = self.b3 = self.b4 = 0.0
            self.zeta_eff = self.zeta
            self.p1 = self.p2 = 0.0
            return
        r = self.robustness
        c1 = self.c1 = 2.0 * self.zeta / self.wn
        c2 = self.c2 = 1.0 / (self.wn * self.wn)
        # 4th-order derivative-matched (ZVD-style) inverse -- analysis / the
        # target once a C^3 trajectory carries jerk & snap. F_r = (1-r)F1+r F2.
        self.b1 = c1 * (1.0 + r)
        self.b2 = c2 * (1.0 + r) + r * c1 * c1
        self.b3 = r * (2.0 * c1 * c2)
        self.b4 = r * (c2 * c2)
        # 2nd-order pointwise inverse ACTUALLY applied in the C seam:
        #   x = y + p1*v + p2*a,  p1 = 2*zeta_eff/wn,  p2 = 1/wn^2.
        # Robustness widens the zero's damping zeta_eff (clean 2nd-order notch
        # at wn, wider/shallower) -- realizable from v,a alone. It does NOT give
        # the ZVD frequency-insensitivity (that needs b3,b4 == jerk,snap); wn
        # drift is instead tracked by re-identifying the mode from ring-down.
        self.zeta_eff = self.zeta + r * (self.zeta_wide - self.zeta)
        self.p1 = 2.0 * self.zeta_eff / self.wn
        self.p2 = c2

    # -- application -------------------------------------------------------
    def augment(self, pos, vel, accel, jerk=0.0, snap=0.0):
        """Motor command for one axis from the tool-tip motion state.

        jerk/snap are only needed when robustness>0 (b3,b4 != 0); on a
        constant-accel trapq they are 0 within a segment and impulsive at
        segment boundaries -- which is why r>0 requires a C^3 (jerk-limited)
        trajectory. At r=0 only pos/vel/accel are consulted.
        """
        return (pos + self.b1 * vel + self.b2 * accel
                + self.b3 * jerk + self.b4 * snap)

    def excursion(self, vel, accel, jerk=0.0, snap=0.0):
        """Extra motor displacement from the correction (headroom sizing)."""
        return (self.b1 * vel + self.b2 * accel
                + self.b3 * jerk + self.b4 * snap)

    def motor_accel_extra(self, jerk, snap, crackle=0.0, pop=0.0):
        """Extra MOTOR acceleration beyond tool accel (= d^2/dt^2 of the
        correction terms): b1*jerk + b2*snap + b3*crackle + b4*pop. Used to
        reserve a_max(v) headroom for the augmented motor trajectory."""
        return (self.b1 * jerk + self.b2 * snap
                + self.b3 * crackle + self.b4 * pop)

    def pos_discontinuity(self, accel_jump, jerk_jump=0.0):
        """Motor-position jump produced by a discontinuity in the tool accel
        (and jerk, for r>0) across a trapq segment boundary: b2*da + b3*dj.
        This is the quantity that must stay below a step to avoid a
        stepcompress 'Invalid sequence'. On coarse constant-accel trapq it is
        b2*da; jerk-limiting drives da->0 per boundary and makes it safe."""
        return self.b2 * accel_jump + self.b3 * jerk_jump

    # -- frequency-domain analysis (for tuning/tests) ---------------------
    def _F(self, s_re, s_im):
        """F_r(s) evaluated at complex s = s_re + j s_im, returned as
        (re, im). F_r = 1 + b1 s + b2 s^2 + b3 s^3 + b4 s^4."""
        re, im = 1.0, 0.0
        # accumulate b_k * s^k using running power of s
        p_re, p_im = 1.0, 0.0  # s^0
        for bk in (self.b1, self.b2, self.b3, self.b4):
            # p = p * s
            p_re, p_im = p_re * s_re - p_im * s_im, p_re * s_im + p_im * s_re
            re += bk * p_re
            im += bk * p_im
        return re, im

    def notch_gain(self, freq_hz):
        """|F_r(j w)| at a probe frequency -- the shaper's excitation gain
        (how much a sinusoid at freq_hz drives the mode). 0 = fully notched."""
        w = TWO_PI * freq_hz
        re, im = self._F(0.0, w)
        return math.hypot(re, im)

    def residual_gain(self, wn_true, zeta_true):
        """Leftover excitation at the ACTUAL mode pole after applying this FF:
        |F_r(s_pole)| with s_pole = -zeta_true wn_true + j wn_true
        sqrt(1-zeta_true^2). ~0 when the model matches truth and r small;
        rises with wn/zeta mismatch. Robustness (r) flattens its variation
        with wn_true at the cost of a higher value at the matched center.
        wn_true is the true natural frequency in rad/s."""
        wt = wn_true
        wd = wt * math.sqrt(max(0.0, 1.0 - zeta_true * zeta_true))
        re, im = self._F(-zeta_true * wt, wd)
        return math.hypot(re, im)


# ---------------------------------------------------------------------------
# Klipper printer object
# ---------------------------------------------------------------------------
class ModelInverseFF:
    """[model_inverse_ff] config object.

    Computes per-axis FFParams and (best-effort) pushes the coefficients to the
    C kinematic seam via input_shaper_set_ff_params. If the C symbol is not yet
    present (chelper not rebuilt with the Stage-4 seam), it stays in
    planner-only mode: the coefficients and the a_max headroom are still exposed
    for the reachability adapter, but no per-step correction is applied.
    """

    def __init__(self, config):
        self.printer = config.get_printer()
        self.robustness = config.getfloat("robustness", 0.0,
                                          minval=0.0, maxval=1.0)
        self.zeta_wide = config.getfloat("robust_zeta", 0.15,
                                         minval=0.0, maxval=0.5)
        self.axes = {}
        for axis in ("x", "y"):
            freq = config.getfloat("freq_" + axis, 0.0, minval=0.0)
            zeta = config.getfloat("damping_ratio_" + axis, 0.05,
                                   minval=0.0, maxval=0.5)
            self.axes[axis] = FFParams(freq, zeta, self.robustness,
                                       self.zeta_wide)
        # a_max(v) fraction reserved for the FF correction on the motor
        # trajectory; consumed by the reachability adapter (plan against
        # (1-headroom)*a_max). Scales with robustness by default.
        self.headroom = config.getfloat("headroom", 0.15,
                                        minval=0.0, maxval=0.9)
        # Enabled only when an axis actually has a mode frequency; off by
        # default (freq defaults to 0) so a bare [model_inverse_ff] is inert.
        self.enabled = config.getboolean("enabled", True)
        self._wrapped = {}  # id(stepper) -> gc-held is_sk wrapper
        self.max_da = None  # per-segment accel-change cap for the emitter
        self.printer.register_event_handler("klippy:connect", self._connect)
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command("SET_MODEL_FF", self.cmd_SET_MODEL_FF,
                               desc=self.cmd_SET_MODEL_FF_help)

    def _connect(self):
        # Expose ourselves to the toolhead so its pathplan adapter can read
        # max_da (the per-segment accel-change cap that keeps our p2*a term
        # sub-step) without a per-move lookup.
        self.printer.lookup_object("toolhead").model_inverse_ff = self
        self._push()

    # Safety factor vs one step. The worst accel jump the FF sees is ~4*max_da:
    # at a TRIANGLE PEAK (accel ramp straight into a decel ramp, no cruise) the
    # discrete taper floor (~2*max_da) on the accel side flips sign against the
    # ~2*max_da on the decel side. So the FF position jump is p2*4*max_da; 0.1
    # keeps it ~0.4 step with robust margin. (Only short fallback moves get the
    # resulting finer slicing; normal moves are unaffected since max_da/J > dt.)
    _DA_SAFETY = 0.1

    def _compute_max_da(self, kin):
        # Tightest per-segment path accel-change (mm/s^2) that keeps p2*da below
        # a fraction of a motor step on every FF-active axis. None if no axis is
        # active (then the emitter is unconstrained -- bare steppers / PA are
        # fine with velocity-continuous hard accel steps; only the FF isn't).
        # Worst-case accel jump at a seam is ~4*max_da (a triangle-peak / corner
        # where the accel taper floor (~2*max_da) flips sign), so _DA_SAFETY
        # already accounts for the factor of 4.
        best = None
        matched = []
        for stepper in kin.get_steppers():
            nm = stepper.get_name()
            for axis in ("x", "y"):
                p = self.axes[axis]
                if not (self.enabled and p.wn > 0.0 and p.p2 > 0.0):
                    continue
                if nm == "stepper_" + axis or nm.endswith("_" + axis):
                    lim = self._DA_SAFETY * stepper.get_step_dist() / p.p2
                    best = lim if best is None else min(best, lim)
                    matched.append((nm, axis, stepper.get_step_dist()))
        logging.info("model_inverse_ff: max_da=%s matched_steppers=%s"
                     % (best, matched))
        return best

    def plan_accel_scale(self):
        """Fraction of a_max(v) the tool planner may use; the rest is FF
        headroom. Consumed by the pathplan Constraints adapter."""
        if not self.enabled:
            return 1.0
        return 1.0 - self.headroom

    def get_params(self, axis):
        return self.axes.get(axis)

    def _get_ffi_setter(self):
        try:
            from klippy import chelper
            ffi_main, ffi_lib = chelper.get_ffi()
            setter = getattr(ffi_lib, "input_shaper_set_ff_params", None)
            return ffi_main, ffi_lib, setter
        except Exception:
            return None, None, None

    def _ensure_wrapped(self, stepper, ffi_main, ffi_lib):
        # Wrap the stepper's kinematics with an input_shaper struct (the same
        # wrapper the C FF lives in) exactly like [input_shaper] does, so the
        # sk we push FF params into is guaranteed an input_shaper struct.
        # Idempotent: once wrapped, get_stepper_kinematics() returns our is_sk.
        # NOTE: mutually exclusive with [input_shaper] on the same axis -- the
        # FF replaces the shaper. Do not configure both for one axis.
        sk = stepper.get_stepper_kinematics()
        prev = self._wrapped.get(id(stepper))
        if prev is not None and prev == sk:
            return sk
        is_sk = ffi_main.gc(ffi_lib.input_shaper_alloc(), ffi_lib.free)
        if ffi_lib.input_shaper_set_sk(is_sk, sk) < 0:
            stepper.set_stepper_kinematics(sk)
            return None
        stepper.set_stepper_kinematics(is_sk)
        self._wrapped[id(stepper)] = is_sk
        return is_sk

    def _push(self):
        ffi_main, ffi_lib, setter = self._get_ffi_setter()
        if setter is None:
            logging.info("model_inverse_ff: C seam absent; planner-only mode "
                         "(coefficients+headroom active, no per-step FF)")
            return
        active = self.enabled and any(p.wn > 0.0 for p in self.axes.values())
        toolhead = self.printer.lookup_object("toolhead")
        toolhead.flush_step_generation()
        kin = toolhead.get_kinematics()
        # Recompute the emitter's accel-change cap for the current coeffs. Set
        # BEFORE any moves are planned so the pathplan adapter honors it.
        self.max_da = self._compute_max_da(kin) if active else None
        if not active:
            # Disable on anything we previously wrapped; never wrap just to off.
            for is_sk in self._wrapped.values():
                for axis in ("x", "y"):
                    setter(is_sk, axis.encode(), 0, 0.0, 0.0)
            return
        for stepper in kin.get_steppers():
            if stepper.get_trapq() is None:
                continue
            is_sk = self._ensure_wrapped(stepper, ffi_main, ffi_lib)
            if is_sk is None:
                continue
            for axis in ("x", "y"):
                p = self.axes[axis]
                en = 1 if p.wn > 0.0 else 0
                # 2nd-order pointwise inverse: x = y + p1*v + p2*a
                setter(is_sk, axis.encode(), en, p.p1, p.p2)

    cmd_SET_MODEL_FF_help = ("Tune model-inverse feedforward "
                             "(FREQ_X/Y, DAMPING_RATIO_X/Y, ROBUSTNESS, "
                             "HEADROOM, ENABLE)")

    def cmd_SET_MODEL_FF(self, gcmd):
        self.enabled = bool(gcmd.get_int("ENABLE", 1 if self.enabled else 0))
        self.robustness = gcmd.get_float("ROBUSTNESS", self.robustness,
                                         minval=0.0, maxval=1.0)
        self.headroom = gcmd.get_float("HEADROOM", self.headroom,
                                       minval=0.0, maxval=0.9)
        for axis in ("x", "y"):
            p = self.axes[axis]
            f = gcmd.get_float("FREQ_" + axis.upper(), p.freq_hz, minval=0.0)
            z = gcmd.get_float("DAMPING_RATIO_" + axis.upper(), p.zeta,
                               minval=0.0, maxval=0.5)
            p.freq_hz, p.zeta = f, z
            p.robustness, p.zeta_wide = self.robustness, self.zeta_wide
            p._update()
        self._push()
        gcmd.respond_info(self._status_str())

    def _status_str(self):
        parts = ["model_inverse_ff enabled=%d robustness=%.3f headroom=%.3f"
                 " (plan_accel_scale=%.3f)"
                 % (self.enabled, self.robustness, self.headroom,
                    self.plan_accel_scale())]
        for axis in ("x", "y"):
            p = self.axes[axis]
            parts.append("  %s: f=%.1fHz zeta=%.4f zeta_eff=%.4f  applied: "
                         "p1=%.3e p2=%.3e" % (axis, p.freq_hz, p.zeta,
                                              p.zeta_eff, p.p1, p.p2))
        return "\n".join(parts)

    def get_status(self, eventtime):
        st = {"enabled": self.enabled, "robustness": self.robustness,
              "headroom": self.headroom,
              "plan_accel_scale": self.plan_accel_scale(),
              "max_da": self.max_da}
        for axis in ("x", "y"):
            p = self.axes[axis]
            st[axis] = {"freq": p.freq_hz, "zeta": p.zeta,
                        "zeta_eff": p.zeta_eff, "p1": p.p1, "p2": p.p2}
        return st


def load_config(config):
    return ModelInverseFF(config)
