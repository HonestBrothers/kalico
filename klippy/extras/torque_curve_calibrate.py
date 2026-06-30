# Automated calibration for stepper motor torque-speed curves
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math
import os
import subprocess


class TorqueCurveCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]

        # Test configuration
        self.test_axis = config.get("axis", "x").lower()
        if self.test_axis not in ("x", "y"):
            raise config.error("axis must be 'x' or 'y'")

        # Speed range to test (mm/s). speed_end is optional: when omitted, the
        # sweep runs up to the theoretical back-EMF top speed (see below); when
        # given, it is still capped at that ceiling.
        self.speed_start = config.getfloat("speed_start", 50.0, above=0.0)
        self.speed_end = config.getfloat("speed_end", None, above=0.0)
        self.speed_step = config.getfloat("speed_step", 25.0, above=0.0)

        # --- Theoretical top-speed model (optional) --------------------------
        # A stepper is a fixed-voltage device: past the speed where its back-EMF
        # equals the supply voltage there is no headroom left to drive current,
        # so torque -> 0 and the carriage physically cannot go faster. That is a
        # hard kinematic ceiling, so we cap speed_end at it (or, if speed_end is
        # unset, sweep right up to it). Needs the supply voltage -- which is NOT
        # in printer.cfg and cannot be read from a TMC2209 (no voltage
        # telemetry) -- plus the motor's back-EMF constant Ke, given directly or
        # derived from datasheet holding torque / rated current.
        self.supply_voltage = config.getfloat(
            "supply_voltage", 0.0, minval=0.0
        )
        # Ke in V/(rad/s). If 0, derived from holding_torque / rated_current.
        self.motor_back_emf = config.getfloat(
            "motor_back_emf", 0.0, minval=0.0
        )
        self.motor_holding_torque = config.getfloat(
            "motor_holding_torque", 0.0, minval=0.0
        )  # N*m (e.g. 0.59 for a 17HS19-2004S1)
        self.motor_rated_current = config.getfloat(
            "motor_rated_current", 0.0, minval=0.0
        )  # A peak, from the datasheet -- NOT run_current

        # --- Peak-velocity dwell (current-saturation hold) -------------------
        # Phase current settles with the winding time constant tau = L/R; a
        # pure-triangle apex is shorter than tau, so the motor is judged before
        # current (hence true torque at speed) has saturated. Each test move
        # therefore holds at peak velocity for ~SETTLE_TAUS*tau (+ margin) of
        # cruise. Both the inductance and DC phase resistance must be given (from
        # the datasheet); without both, the dwell is disabled (pure triangles).
        self.motor_inductance = config.getfloat(
            "motor_inductance", 0.0, minval=0.0
        )  # henries (e.g. 0.0028 for a 17HS19-2004S1)
        self.motor_resistance = config.getfloat(
            "motor_resistance", 0.0, minval=0.0
        )  # ohms, DC phase resistance (e.g. 1.1 for a 17HS19-2004S1)
        # Extra margin added to the computed dwell (0.2 = +20%).
        self.dwell_margin = config.getfloat("dwell_margin", 0.2, minval=0.0)

        # Acceleration range to test (mm/s^2). The search climbs geometrically
        # (x accel_growth) from accel_start up to accel_max; accel_step is the
        # final resolution -- the search stops once the known-good/known-skip
        # bracket is narrower than this -- not a linear increment.
        self.accel_start = config.getfloat("accel_start", 1000.0, above=0.0)
        self.accel_max = config.getfloat("accel_max", 50000.0, above=0.0)
        self.accel_step = config.getfloat("accel_step", 1000.0, above=0.0)

        # Multiplicative step for the exponential-from-below search. Each
        # overshoot (skip) halves this ratio toward 1, so the climb refines onto
        # the real limit. 2.0 = double each step until the first skip.
        self.accel_growth = config.getfloat("accel_growth", 1.5, above=1.0)

        # Move distance for testing (mm). This caps the *largest* triangular
        # move (the low-accel end); each probe's actual distance is sized to
        # v^2/a so the profile is a pure accel/decel triangle with no cruise.
        self.test_move_distance = config.getfloat(
            "test_move_distance", 50.0, above=10.0
        )

        # Floor for the triangular (no-cruise) move distance. v^2/a shrinks to
        # microns at low speed + very high accel; too short a move is dominated
        # by transients and step quantization rather than torque, so clamp up to
        # this. The clamp reintroduces a little cruise, but only in the regime
        # where the motor has ample torque headroom anyway.
        self.min_move_distance = config.getfloat(
            "min_move_distance", 2.0, above=0.0
        )

        # Position tolerance for detecting lost steps (in mm)
        self.position_tolerance = config.getfloat(
            "position_tolerance", 0.1, above=0.0
        )

        # Number of back-and-forth passes per (speed, accel) test. Each pass is
        # a hard apex slam; many passes accumulate winding heat (R rises ->
        # torque sags) and stack up mechanical/resonance stress. The peak dwell
        # (above) handles per-pass current saturation; pass count handles the
        # cumulative thermal/mechanical side. Homing dominates per-probe cost,
        # so a generous count is nearly free.
        self.test_cycles = config.getint("test_cycles", 30, minval=1)

        # Output file for calibration results
        self.output_file = config.get("output_file", "torque_curve.csv")

        # Target torque_curve module to update
        self.target_curve = config.get("target_curve", None)

        # Raise the gantry to a safe height after homing, before the high-speed
        # sweeps, so the nozzle cannot scrape or crash the bed. 0 disables.
        self.safe_z = config.getfloat("safe_z", 20.0, minval=0.0)
        self.z_lift_speed = config.getfloat("z_lift_speed", 25.0, above=0.0)

        # --- Vibration instrumentation (accelerometer) ----------------------
        # When True, capture the accelerometer during each probe and log a
        # vibration metric per (speed, accel) point alongside the skip data --
        # the raw material for the excitation map a TOPP-RA constraint can use
        # to notch out resonant speed/accel bands. Measures with input shaping
        # ON by default (real-world, shaped vibration); set
        # vibration_shaper_off True to disable shaping and capture raw modes.
        self.measure_vibration = config.getboolean("measure_vibration", False)
        self.accel_chip_name = config.get("accel_chip", None)
        self.vibration_shaper_off = config.getboolean(
            "vibration_shaper_off", False
        )
        # Ring-down window (s) captured while stationary right after the stress
        # burst. Commanded accel is 0 there, so it's the *pure* decaying
        # resonance -- the cleanest "commanded factored out" metric (and it
        # measures f_n directly, no input_shaper needed). 0 disables it.
        self.ring_down_time = config.getfloat(
            "ring_down_time", 0.15, minval=0.0
        )
        # Stationary baseline (s) captured once before the sweep with the motors
        # energized -- the noise floor (sensor + ambient + holding-current buzz)
        # that's quadrature-subtracted from every probe's metrics.
        self.baseline_time = config.getfloat(
            "baseline_time", 0.25, minval=0.0
        )
        # When True, apply the shaper the real-move sweep recommends (live
        # SET_INPUT_SHAPER) and stage it into SAVE_CONFIG. Off by default so a
        # sweep never silently changes motion; flip per-run with APPLY_SHAPER=1.
        # Applying it also closes the loop for auto_jerk, which derives per-axis
        # max_jerk from the input_shaper frequency.
        self.apply_shaper = config.getboolean("apply_shaper", False)

        # Resonance-extraction band + peak prominence. The structural mode is a
        # fixed property of the axis, so a probe's dominant residual peak should
        # land in this band and stand clearly above the rest of the spectrum.
        # Degenerate probes (microscopically short or near-skip high-accel moves)
        # produce a flat/edge spectrum with no real peak; requiring prominence
        # and rejecting band-edge peaks keeps that junk -- which would otherwise
        # default to the search-window floor (~20 Hz) and drag the recommended
        # frequency low -- out of the summary and the shaper graph.
        self.resonance_min_freq = config.getfloat(
            "resonance_min_freq", 25.0, minval=1.0)
        self.resonance_max_freq = config.getfloat(
            "resonance_max_freq", 150.0, above=0.0)
        self.resonance_prominence = config.getfloat(
            "resonance_prominence", 4.0, minval=1.0)
        # Only fold probes accelerating below this fraction of the speed's
        # measured skip boundary into the resonance spectrum. Right at the skip
        # threshold the motor stutters (a strong low-frequency lurch) that isn't
        # a structural mode; excluding the near-skip band keeps it out of the
        # max-accumulated PSD that feeds the shaper graph.
        self.resonance_accel_frac = config.getfloat(
            "resonance_accel_frac", 0.85, above=0.0, maxval=1.0)

        # Internal state
        self.calibration_running = False
        self.calibration_results = []
        self._skip_accel = {}
        self._recommended_shaper = None  # (name, freq) from the last sweep
        self._resume = False
        self._speed_resid_buffer = []  # per-speed (accel, (t, resid)) buffer
        self.vibration_rows = []
        self._vib_file = None
        self._vib_path = None
        self._progress_file = None
        self._baseline = None
        # Running Klipper-format resonance spectrum built from the residual
        # ringing of the real test moves -- fed to the stock shaper fitter at
        # the end of the sweep to produce the standard input-shaper graph.
        self._vib_caldata = None
        self._vib_sc = None

        # Register event handler
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )

        # Register gcode commands
        gcode = self.printer.lookup_object("gcode")
        gcode.register_mux_command(
            "TORQUE_CURVE_CALIBRATE", "NAME", self.name,
            self.cmd_TORQUE_CURVE_CALIBRATE,
            desc=self.cmd_TORQUE_CURVE_CALIBRATE_help,
        )

    def _handle_connect(self):
        self.toolhead = self.printer.lookup_object("toolhead")
        self.gcode = self.printer.lookup_object("gcode")
        self.kin = self.toolhead.get_kinematics()
        self.accel_chip = None
        if self.measure_vibration:
            self.accel_chip = self._lookup_accel_chip()

    def _lookup_accel_chip(self):
        """Find the accelerometer chip object, by config name or common ones."""
        names = ([self.accel_chip_name] if self.accel_chip_name
                 else ["lis2dw", "adxl345"])
        for n in names:
            chip = self.printer.lookup_object(n, None)
            if chip is not None and hasattr(chip, "start_internal_client"):
                return chip
        return None

    def _shaper_freq(self):
        """Input-shaper frequency (Hz) for the test axis, or None."""
        ins = self.printer.lookup_object("input_shaper", None)
        if ins is None:
            return None
        for sh in ins.get_shapers():
            if sh.get_axis() == self.test_axis:
                f = getattr(getattr(sh, "params", None), "shaper_freq", 0.0)
                return f or None
        return None

    def _get_axis_index(self):
        """Get axis index (0=X, 1=Y, 2=Z)."""
        return {"x": 0, "y": 1, "z": 2}[self.test_axis]

    def _get_axis_limits(self):
        """Get min/max positions for the test axis."""
        axis_idx = self._get_axis_index()
        rails = self.kin.rails
        for rail in rails:
            steppers = rail.get_steppers()
            for stepper in steppers:
                if stepper.is_active_axis(self.test_axis):
                    pos_min, pos_max = rail.get_range()
                    return pos_min, pos_max
        raise self.printer.command_error(
            "Could not find axis limits for %s" % self.test_axis
        )

    def _home_axis(self):
        """Home the test axis."""
        self.gcode.run_script_from_command("G28 %s" % self.test_axis.upper())

    def _raise_z(self, gcmd):
        """Home Z if needed, then lift the gantry to safe_z.

        The sweep only moves the test axis, leaving Z wherever it started, so
        without this the high-speed moves run at the current gantry height and
        could scrape/crash the bed. Runs after the initial home; no-op when
        safe_z is 0.
        """
        if self.safe_z <= 0.0:
            return
        systime = self.printer.get_reactor().monotonic()
        if "z" not in self.toolhead.get_status(systime)["homed_axes"]:
            self.gcode.run_script_from_command("G28 Z")
        self.gcode.run_script_from_command(
            "G90\nG1 Z%.3f F%.0f" % (self.safe_z, self.z_lift_speed * 60.0)
        )
        self.toolhead.wait_moves()
        gcmd.respond_info(
            "Raised Z to %.1f mm (safe height) before calibration"
            % self.safe_z
        )

    def _supports_kinematic_limits(self):
        """True for kinematics exposing per-axis caps (Kalico limited_*)."""
        return (
            hasattr(self.kin, "max_accels")
            and hasattr(self.kin, "max_velocities")
            and hasattr(self.kin, "scale_per_axis")
        )

    def _save_kinematic_limits(self):
        """Snapshot per-axis kinematic limits so they can be restored.

        limited_cartesian (and friends) enforce per-axis accel/velocity caps,
        and with scale_xy_accel the commanded accel is rescaled. The sweep
        drives accel via SET_VELOCITY_LIMIT and needs commanded == actual, so
        we widen the caps and disable scaling for the run, then put them back.
        """
        if not self._supports_kinematic_limits():
            return None
        return {
            "max_accels": list(self.kin.max_accels),
            "max_velocities": list(self.kin.max_velocities),
            "scale_per_axis": self.kin.scale_per_axis,
        }

    def _apply_kinematic_limits(self, gcmd):
        """Open up per-axis caps to cover the sweep and turn off scaling."""
        if not self._supports_kinematic_limits():
            return
        self.gcode.run_script_from_command(
            "SET_KINEMATICS_LIMIT X_ACCEL=%.1f Y_ACCEL=%.1f"
            " X_VELOCITY=%.1f Y_VELOCITY=%.1f SCALE=0"
            % (self.accel_max, self.accel_max, self.speed_end, self.speed_end)
        )
        gcmd.respond_info(
            "Kinematic limits raised to ACCEL=%.0f VELOCITY=%.0f (SCALE off) "
            "for calibration" % (self.accel_max, self.speed_end)
        )

    def _restore_kinematic_limits(self, saved, gcmd):
        """Restore the snapshot taken by _save_kinematic_limits."""
        if not saved:
            return
        xa, ya, za = saved["max_accels"]
        xv, yv, zv = saved["max_velocities"]
        self.gcode.run_script_from_command(
            "SET_KINEMATICS_LIMIT X_ACCEL=%.1f Y_ACCEL=%.1f Z_ACCEL=%.1f"
            " X_VELOCITY=%.1f Y_VELOCITY=%.1f Z_VELOCITY=%.1f SCALE=%d"
            % (xa, ya, za, xv, yv, zv, 1 if saved["scale_per_axis"] else 0)
        )
        gcmd.respond_info("Kinematic limits restored")

    def _disable_reshapers(self, gcmd):
        """Turn off motion-profile reshapers for the sweep; return saved state.

        The sweep must command raw constant-accel moves. A reshaper would
        invalidate that: TOPP-RA would clamp the commanded accel to its existing
        curve (so the search could never climb past it -- silently corrupting
        the measured limit), and jerk limiting would replace the trapezoid with
        an S-curve. Both are disabled here and restored in the finally block.
        (Jerk already skips travel moves, which the sweep uses, but it's
        disabled anyway so the guard doesn't depend on that detail.)
        """
        self.toolhead.flush_step_generation()
        saved = {}
        tc = getattr(self.toolhead, "topp_ra", None)
        if tc is not None and getattr(tc, "enabled", False):
            saved["topp"] = tc
            tc.enabled = False
            gcmd.respond_info("Disabled TOPP-RA reshaping for calibration")
        jl = getattr(self.toolhead, "jerk_limiting", None)
        if jl is not None and getattr(jl, "enabled", False):
            saved["jerk"] = jl
            jl.enabled = False
            gcmd.respond_info("Disabled jerk limiting for calibration")
        # Optional raw-mode: drop input shaping so the accelerometer sees the
        # unshaped resonance (default keeps shaping on = real-world vibration).
        if (self.measure_vibration and self.vibration_shaper_off
                and self.printer.lookup_object("input_shaper", None)):
            self.gcode.run_script_from_command("DISABLE_INPUT_SHAPER")
            saved["shaper"] = True
            gcmd.respond_info(
                "Input shaping disabled for raw vibration measurement"
            )
        return saved

    def _restore_reshapers(self, saved, gcmd):
        """Re-enable whatever _disable_reshapers turned off."""
        if not saved:
            return
        self.toolhead.flush_step_generation()
        if "topp" in saved:
            saved["topp"].enabled = True
        if "jerk" in saved:
            saved["jerk"].enabled = True
        if "shaper" in saved:
            self.gcode.run_script_from_command("ENABLE_INPUT_SHAPER")
        gcmd.respond_info("Restored motion reshaping (TOPP-RA / jerk / shaper)")

    def _get_current_position(self):
        """Get current position on test axis."""
        axis_idx = self._get_axis_index()
        return self.toolhead.get_position()[axis_idx]

    def _get_stepper_position(self):
        """Get raw stepper position for precise lost step detection."""
        axis_idx = self._get_axis_index()
        self.toolhead.flush_step_generation()

        for rail in self.kin.rails:
            for stepper in rail.get_steppers():
                if stepper.is_active_axis(self.test_axis):
                    return stepper.get_mcu_position()
        return None

    def _calculate_test_positions(self):
        """Calculate safe test positions that allow full acceleration."""
        pos_min, pos_max = self._get_axis_limits()
        range_size = pos_max - pos_min

        # Use center of travel for testing
        center = (pos_min + pos_max) / 2
        half_move = min(self.test_move_distance / 2, range_size / 4)

        start_pos = center - half_move
        end_pos = center + half_move

        return start_pos, end_pos

    def _calculate_required_distance(self, speed, accel):
        """
        Calculate minimum distance required to reach target speed.
        Using: v^2 = 2*a*d  ->  d = v^2 / (2*a)
        Need 2x this for accel + decel.
        """
        accel_dist = (speed ** 2) / (2 * accel)
        return accel_dist * 2  # Need to accelerate and decelerate

    def _perform_test_move(self, center, max_half, speed, accel, dwell_time,
                           capture=False):
        """Stress the axis at (speed, accel) with test_cycles passes,
        centered on `center`.

        Each pass is a triangle sized to v^2/a (accelerate to exactly `speed`
        at the midpoint, then decelerate) plus a minimum cruise segment of
        `dwell_time` seconds held at peak velocity. The triangle puts the
        hardest mechanical instant (peak velocity + full accel) at the target
        speed; the dwell holds there long enough (~5*L/R) for phase current --
        and thus the real torque at that speed -- to saturate before a skip is
        judged. With dwell_time = 0 it degenerates to the pure triangle.

        D = v^2/a + v*dwell_time, clamped to [min_move_distance, 2*max_half]:
        the upper bound keeps the move on the bed (the caller's per-speed
        start_accel accounts for the cruise so the lowest-accel probe still
        fits), and the lower bound avoids degenerate sub-resolution micro-moves.

        Lost-step detection is done by the caller via _home_and_measure /
        _check_for_lost_steps, so nothing is returned."""
        axis_idx = self._get_axis_index()

        # Triangle (v^2/a) plus a minimum cruise hold of dwell_time at peak.
        tri_dist = self._calculate_required_distance(speed, accel)  # v^2/a
        half = (tri_dist + speed * dwell_time) / 2.0
        half = max(self.min_move_distance / 2.0, min(half, max_half))
        start_pos = center - half
        end_pos = center + half

        # Move to start position at a safe speed/accel
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=1000 VELOCITY=100"
        )
        pos = list(self.toolhead.get_position())
        pos[axis_idx] = start_pos
        self.toolhead.move(pos, 100)
        self.toolhead.wait_moves()

        # Set the test acceleration/speed, then slam the axis back and forth
        # test_cycles times. The passes are queued with NO wait_moves between
        # them, so they pipeline into a continuous zigzag that keeps the coils
        # loaded; with SCV=0 each 180-degree reversal still brings the axis to
        # a stop and re-accelerates at full accel. A single wait_moves at the
        # end lets the whole burst complete before the lost-step check.
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.1f VELOCITY=%.1f" % (accel, speed)
        )
        # Capture the accelerometer over just the stress burst (the slow
        # positioning move above is excluded), then -- while the axis sits still
        # -- a short ring-down window where commanded accel is 0. Returns a dict
        # the caller reduces to vibration metrics.
        aclient = None
        if capture and self.accel_chip is not None:
            aclient = self.accel_chip.start_internal_client()
        for _ in range(self.test_cycles):
            pos[axis_idx] = end_pos
            self.toolhead.move(pos, speed)
            pos[axis_idx] = start_pos
            self.toolhead.move(pos, speed)
        self.toolhead.wait_moves()
        if aclient is None:
            return None
        aclient.finish_measurements()
        cap = {
            "burst": aclient.get_samples(),
            "t0": aclient.request_start_time,
            "t1": aclient.request_end_time,
        }
        # Ring-down: hold still and capture the pure decaying resonance.
        if self.ring_down_time > 0.0:
            rd = self.accel_chip.start_internal_client()
            self.toolhead.dwell(self.ring_down_time)
            self.toolhead.wait_moves()
            rd.finish_measurements()
            cap["ringdown"] = rd.get_samples()
        return cap

    def _axis_ac(self, np, samples):
        """(times, test-axis AC acceleration), DC/gravity removed, or None.

        Everything is measured on this one signal so RMS, peak and spectrum all
        refer to the same thing (the test axis), which keeps the metrics and the
        at-f_n spectral lines mutually consistent.
        """
        if not samples or len(samples) < 16:
            return None
        idx = 1 + self._get_axis_index()
        t = np.asarray([s[0] for s in samples], dtype=float)
        sig = np.asarray([s[idx] for s in samples], dtype=float)
        return t, sig - sig.mean()

    def _spectrum(self, np, times, sig):
        """(freqs, single-sided magnitude spectrum), or (None, None)."""
        if len(times) < 8:
            return None, None
        dt = (times[-1] - times[0]) / (len(times) - 1)
        if dt <= 0:
            return None, None
        win = np.hanning(len(sig))
        spec = np.abs(np.fft.rfft(sig * win)) * (2.0 / win.sum())
        return np.fft.rfftfreq(len(sig), dt), spec

    def _mag_at(self, np, freqs, spec, fn):
        """Spectrum magnitude at frequency fn (nearest bin), or 0."""
        if freqs is None or not fn or len(spec) < 2:
            return 0.0
        return float(spec[int(np.argmin(np.abs(freqs - fn)))])

    def _peak_freq(self, np, freqs, spec, lo=0.0, hi=1e9):
        """Dominant frequency of spec within [lo, hi] (skipping DC), or 0."""
        if freqs is None or len(spec) < 2:
            return 0.0
        band = (freqs >= max(lo, freqs[1])) & (freqs <= hi)
        if not band.any():
            return 0.0
        return float(freqs[int(np.argmax(np.where(band, spec, 0.0)))])

    def _dominant_mode(self, np, freqs, spec):
        """Frequency of the dominant residual peak, or 0.0 if there isn't a
        clean one. A real structural mode is a sharp interior peak that stands
        well above the rest of the band; a degenerate or near-skip probe gives
        a flat/edge spectrum with no such peak. Rejecting band-edge maxima and
        requiring the peak to exceed resonance_prominence x the band median
        keeps that junk from defaulting to the search-window floor (~the lo
        bound) and biasing the recommended frequency low."""
        lo, hi = self.resonance_min_freq, self.resonance_max_freq
        if freqs is None or spec is None or len(spec) < 2:
            return 0.0
        idx = np.where((freqs >= max(lo, freqs[1])) & (freqs <= hi))[0]
        if len(idx) < 5:
            return 0.0
        sb = spec[idx]
        k = int(np.argmax(sb))
        # No real interior peak -> the maximum sits at the band edge.
        if k == 0 or k == len(sb) - 1:
            return 0.0
        med = float(np.median(sb))
        if med <= 0.0 or float(sb[k]) < self.resonance_prominence * med:
            return 0.0
        return float(freqs[idx[k]])

    def _commanded_axis_accel(self, np, times):
        """Commanded test-axis acceleration (mm/s^2) at each print_time.

        Read straight from the toolhead trapq: each segment has a constant
        accel along its unit direction axes_r, so the axis component is
        accel * axes_r[axis]. Samples already carry print_time, so this *is*
        the clock alignment -- evaluate the commanded profile at the sample
        times and subtract to factor out the bulk motion.
        """
        import chelper
        ffi_main, ffi_lib = chelper.get_ffi()
        trapq = self.toolhead.get_trapq()
        axis = self._get_axis_index()
        data = ffi_main.new("struct pull_move[1024]")
        count = ffi_lib.trapq_extract_old(
            trapq, data, 1024, times[0] - 0.05, times[-1] + 0.05
        )
        if not count:
            return None
        starts = np.empty(count); ends = np.empty(count); accs = np.empty(count)
        for i in range(count):
            m = data[i]
            ar = (m.x_r, m.y_r, m.z_r)[axis]
            starts[i] = m.print_time
            ends[i] = m.print_time + m.move_t
            accs[i] = m.accel * ar
        idx = np.searchsorted(starts, times, side="right") - 1
        cmd = np.zeros(len(times))
        valid = (idx >= 0) & (idx < count)
        ii = np.clip(idx, 0, count - 1)
        inside = valid & (times < ends[ii])
        cmd[inside] = accs[ii][inside]
        return cmd

    def _capture_baseline(self, gcmd):
        """Capture the stationary noise floor (sensor + ambient + motor hold).

        Motors are energized but idle, so this is the floor sitting *under*
        every probe. Its RMS and spectrum are quadrature-subtracted from the
        ring-down and residual so a flat near-noise reading can't masquerade as
        resonance (the ring-down RMS was doing exactly that).
        """
        self._baseline = None
        if not (self.measure_vibration and self.accel_chip is not None):
            return
        self.toolhead.wait_moves()
        client = self.accel_chip.start_internal_client()
        self.toolhead.dwell(max(self.baseline_time, 0.05))
        self.toolhead.wait_moves()
        client.finish_measurements()
        import numpy as np
        sig = self._axis_ac(np, client.get_samples())
        if sig is None:
            return
        t, s = sig
        freqs, spec = self._spectrum(np, t, s)
        self._baseline = {
            "rms": float(np.sqrt((s * s).mean())),
            "freqs": freqs, "spec": spec,
        }
        gcmd.respond_info(
            "Vibration noise floor: rms=%.0f mm/s^2 "
            "(quadrature-subtracted from every probe)"
            % self._baseline["rms"]
        )

    def _vibration_metrics(self, cap, lost=False):
        """Reduce a capture dict to noise-floor-corrected vibration metrics.

        Three views of the probe, all de-floored against the baseline by
        subtracting the uncorrelated noise floor in *power*
        (clean = sqrt(meas^2 - floor^2)):
          raw_*   -- burst, commanded NOT removed (contaminated; for contrast)
          resid_* -- burst with the commanded accel subtracted (#3)
          rd_*    -- stationary ring-down (commanded == 0)
        f_n is the dominant mode of the residual spectrum when it's a clean,
        prominent peak; otherwise 0 (no mode found) and the configured shaper
        frequency is used only as a fallback for the a_fn measurement. A probe
        that lost steps, or whose residual has no prominent interior peak, is
        kept out of the resonance spectrum (it's near-skip/degenerate junk).
        """
        if not cap or not cap.get("burst"):
            return None
        import numpy as np
        base = self._baseline
        base_rms = base["rms"] if base else 0.0

        def defloor(x):
            return float(np.sqrt(max(0.0, x * x - base_rms * base_rms)))

        braw = self._axis_ac(np, cap["burst"])
        if braw is None:
            return None
        bt, bsig = braw
        bfreqs, bspec = self._spectrum(np, bt, bsig)
        raw_rms = float(np.sqrt((bsig * bsig).mean()))
        raw_fpeak = self._peak_freq(np, bfreqs, bspec)

        # Residual: subtract the trapq-aligned commanded accel.
        resid_rms = 0.0
        rfreqs = rspec = None
        resid = None
        cmd = self._commanded_axis_accel(np, bt)
        if cmd is not None:
            resid = bsig - (cmd - cmd.mean())
            resid_rms = float(np.sqrt((resid * resid).mean()))
            rfreqs, rspec = self._spectrum(np, bt, resid)

        # The MEASURED dominant mode of the residual (commanded removed) is the
        # truth -- but only when it's a clean, prominent peak; a degenerate or
        # near-skip probe has no real peak and returns 0. The configured shaper
        # frequency is the fallback used purely to place the a_fn measurement.
        resid_fpeak = self._dominant_mode(np, rfreqs, rspec)
        fn = resid_fpeak or self._shaper_freq() or 0.0

        # Hand the clean residual back for deferred, accel-gated accumulation:
        # only in-sync probes (not lost) with a prominent mode are eligible;
        # the per-speed accel cap (applied once the skip boundary is known) then
        # drops the near-skip stutter band before it reaches the shaper PSD.
        clean_resid = ((bt, resid)
                       if resid is not None and not lost and resid_fpeak > 0.0
                       else None)

        rd_rms = 0.0
        rdfreqs = rdspec = None
        rd = self._axis_ac(np, cap.get("ringdown"))
        if rd is not None:
            rdt, rdsig = rd
            rd_rms = float(np.sqrt((rdsig * rdsig).mean()))
            rdfreqs, rdspec = self._spectrum(np, rdt, rdsig)

        # De-floor the spectral line at f_n (quadrature vs the baseline at f_n)
        # -- this is the mode-isolated, noise-corrected ring amplitude.
        base_afn = (self._mag_at(np, base["freqs"], base["spec"], fn)
                    if base else 0.0)

        def defloor_line(x):
            return float(np.sqrt(max(0.0, x * x - base_afn * base_afn)))

        return {
            "raw_rms": raw_rms, "raw_fpeak": raw_fpeak,
            "resid_rms": defloor(resid_rms),
            "resid_fpeak": resid_fpeak,
            "resid_afn": defloor_line(self._mag_at(np, rfreqs, rspec, fn)),
            "rd_rms": defloor(rd_rms),
            "rd_afn": defloor_line(self._mag_at(np, rdfreqs, rdspec, fn)),
            "fn": fn or 0.0, "base_rms": base_rms,
            "_clean_resid": clean_resid,
        }

    def _shaper_calibrate(self):
        """Lazily build a stock ShaperCalibrate helper (or None if missing)."""
        if self._vib_sc is None:
            try:
                from . import shaper_calibrate
                self._vib_sc = shaper_calibrate.ShaperCalibrate(self.printer)
            except Exception:
                logging.exception(
                    "torque_curve_calibrate: shaper_calibrate unavailable")
                self._vib_sc = False
        return self._vib_sc or None

    def _accumulate_psd(self, np, times, resid):
        """Fold one probe's residual ringing into the running PSD.

        Each probe's commanded-removed signal is run through Klipper's own
        calc_freq_response (same windowing/PSD as a resonance test), then
        combined with add_data (element-wise max across bins) -- exactly how
        the stock tester builds a spectrum across measurement points. The
        result is a Klipper-format CalibrationData the shaper fitter can plot,
        but excited by real moves instead of the canonical pulse sweep.
        """
        sc = self._shaper_calibrate()
        if sc is None or times is None or len(times) < 8:
            return
        try:
            n = len(times)
            data = np.zeros((n, 4))
            data[:, 0] = times
            data[:, 1 + self._get_axis_index()] = resid
            cd = sc.calc_freq_response(data)  # None if shorter than the window
            if cd is None:
                return
            cd.set_numpy(sc.numpy)
            if self._vib_caldata is None:
                self._vib_caldata = cd
            else:
                self._vib_caldata.add_data(cd)
        except Exception:
            logging.exception(
                "torque_curve_calibrate: PSD accumulation failed")

    def _accumulate_speed_psd(self, last_good):
        """Fold a completed speed's buffered clean residuals into the PSD,
        dropping probes within the near-skip stutter band (accel above
        resonance_accel_frac x the speed's skip boundary)."""
        if not self._speed_resid_buffer:
            return
        import numpy as np
        cap = (self.resonance_accel_frac * last_good
               if last_good > 0.0 else float("inf"))
        for accel, (bt, resid) in self._speed_resid_buffer:
            if accel <= cap:
                self._accumulate_psd(np, bt, resid)
        self._speed_resid_buffer = []

    def _emit_shaper_graph(self, gcmd):
        """Fit shapers to the accumulated real-move spectrum and write the
        standard resonances CSV + a PNG graph (same as SHAPER_CALIBRATE)."""
        cd = self._vib_caldata
        sc = self._shaper_calibrate()
        if cd is None or sc is None:
            return
        try:
            cd.set_numpy(sc.numpy)
            cd.normalize_to_frequencies()
            systime = self.printer.get_reactor().monotonic()
            scv = self.toolhead.get_status(systime)["square_corner_velocity"]
            best, all_shapers = sc.find_best_shaper(cd, scv=scv, logger=None)
        except Exception:
            logging.exception("torque_curve_calibrate: shaper fit failed")
            return
        out_dir, stem = self._output_dir_stem("input_shaping")
        csv_path = os.path.join(
            out_dir, "%s_shaper_%s.csv" % (stem, self.test_axis))
        try:
            # accel_per_hz is a pulse-sweep concept we don't have; 0.0 is just a
            # placeholder column so the stock grapher can read the CSV.
            sc.save_calibration_data(csv_path, cd, all_shapers, accel_per_hz=0.0)
            gcmd.respond_info("Resonance spectrum -> %s" % csv_path)
        except Exception:
            logging.exception("torque_curve_calibrate: CSV write failed")
        if best is not None:
            gcmd.respond_info(
                "Recommended %s shaper (from real moves): %s @ %.1f Hz "
                "(vibr=%.1f%%, smoothing~=%.2f, accel<=%.0f)"
                % (self.test_axis.upper(), best.name, best.freq,
                   best.vibrs * 100.0, best.smoothing,
                   round(best.max_accel / 100.0) * 100.0))
            self._recommended_shaper = (best.name, best.freq)
            self._apply_recommended_shaper(gcmd, sc, best)
        png_path = self._plot_shaper(cd, all_shapers, best, out_dir, stem)
        if png_path:
            gcmd.respond_info("Resonance graph -> %s" % png_path)

    def _apply_recommended_shaper(self, gcmd, sc, best):
        """Apply the recommended shaper live and stage it into SAVE_CONFIG.

        Mirrors what SHAPER_CALIBRATE does. Gated behind apply_shaper /
        APPLY_SHAPER (default off) so a sweep never silently changes motion.
        Applying it also re-points auto_jerk, which reads the input_shaper
        frequency, at this measured resonance.
        """
        if not self.apply_shaper:
            gcmd.respond_info(
                "    (APPLY_SHAPER=1 to set shaper_type_%s=%s shaper_freq_%s"
                "=%.1f and stage SAVE_CONFIG)"
                % (self.test_axis, best.name, self.test_axis, best.freq))
            return
        ins = self.printer.lookup_object("input_shaper", None)
        if ins is None:
            gcmd.respond_info(
                "    APPLY_SHAPER requested but no [input_shaper] configured")
            return
        try:
            # Live apply (SET_INPUT_SHAPER on the test axis).
            sc.apply_params(ins, self.test_axis, best.name, best.freq)
            # Stage into SAVE_CONFIG so it survives a restart once saved.
            configfile = self.printer.lookup_object("configfile")
            sc.save_params(configfile, self.test_axis, best.name, best.freq)
        except Exception:
            logging.exception(
                "torque_curve_calibrate: applying shaper failed")
            gcmd.respond_info("    failed to apply shaper (see klippy.log)")
            return
        gcmd.respond_info(
            "    applied shaper_type_%s=%s shaper_freq_%s=%.1f (live); run "
            "SAVE_CONFIG to persist."
            % (self.test_axis, best.name, self.test_axis, best.freq))
        # Close the loop: re-derive auto_jerk from the freq we just applied.
        jl = self.printer.lookup_object("jerk_limiting", None)
        if jl is not None and getattr(jl, "auto_jerk", False):
            try:
                jl._apply_auto_jerk(gcmd)
            except Exception:
                logging.exception(
                    "torque_curve_calibrate: auto_jerk recompute failed")

    def _plot_shaper(self, cd, shapers, best, out_dir, stem):
        """Render a SHAPER_CALIBRATE-style PNG (PSD + shaper response curves).

        Mirrors scripts/calibrate_shaper.plot_freq_response but headless (Agg)
        and trimmed to the single test axis.
        """
        try:
            import matplotlib
            matplotlib.rcParams.update({"figure.autolayout": True})
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager
            import matplotlib.ticker
        except Exception:
            logging.exception("torque_curve_calibrate: matplotlib unavailable")
            return None
        try:
            max_freq = 200.0
            freqs = cd.freq_bins
            freqs = freqs[freqs <= max_freq]
            psd = cd.get_psd(self.test_axis)[:len(freqs)]
            # find_best_shaper already trims shaper.vals to <= max_freq; align
            # every series to that common length.
            if shapers:
                m = min(len(freqs), len(shapers[0].vals))
                freqs, psd = freqs[:m], psd[:m]

            fontP = matplotlib.font_manager.FontProperties()
            fontP.set_size("x-small")
            fig, ax = plt.subplots()
            ax.set_xlabel("Frequency, Hz")
            ax.set_xlim([0, max_freq])
            ax.set_ylabel("Power spectral density")
            ax.plot(freqs, psd, label=self.test_axis.upper(), color="purple")
            shaper_f = self._shaper_freq()
            if shaper_f:
                ax.axvline(shaper_f, color="orange", linestyle=":",
                           label="configured %.1f Hz" % shaper_f)
            ax.set_title(
                "Torque-sweep resonance and shapers (%s axis)"
                % self.test_axis.upper())
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            ax.grid(which="major", color="grey")
            ax.grid(which="minor", color="lightgrey")

            ax2 = ax.twinx()
            ax2.set_ylabel("Shaper vibration reduction (ratio)")
            best_vals = None
            best_name = best.name if best is not None else None
            for shaper in shapers:
                label = "%s (%.1f Hz, vibr=%.1f%%, sm~=%.2f, accel<=%.0f)" % (
                    shaper.name.upper(), shaper.freq, shaper.vibrs * 100.0,
                    shaper.smoothing, round(shaper.max_accel / 100.0) * 100.0)
                ls = "dotted" if shaper.name.startswith("smooth") else "dashed"
                lw = 1.0
                vals = shaper.vals[:len(freqs)]
                if shaper.name == best_name:
                    ls, lw, best_vals = "dashdot", 2.0, vals
                ax2.plot(freqs, vals, label=label,
                         linestyle=ls, linewidth=lw)
            if best_vals is not None:
                ax.plot(freqs, psd * best_vals, label="After shaper",
                        color="cyan")
                ax2.plot([], [], " ",
                         label="Recommended: %s" % best_name.upper())
            ax.legend(loc="upper left", prop=fontP)
            ax2.legend(loc="upper right", prop=fontP)
            ax.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)

            png_path = os.path.join(
                out_dir, "%s_shaper_%s.png" % (stem, self.test_axis))
            fig.savefig(png_path)
            plt.close(fig)
            return png_path
        except Exception:
            logging.exception("torque_curve_calibrate: plotting failed")
            return None

    def _step_dist(self):
        """Step distance (mm) of the test axis stepper."""
        for rail in self.kin.rails:
            for stepper in rail.get_steppers():
                if stepper.is_active_axis(self.test_axis):
                    return stepper.get_step_dist()
        return None

    def _home_and_measure(self):
        """Home the test axis and return the at-home stepper position.

        Lost steps are only observable against the endstop: the MCU counts
        commanded steps, so a balanced there-and-back move returns the same
        count whether or not the motor skipped. Re-homing drives the carriage
        back to the physical endstop, so the step count *at home* shifts by
        exactly the steps lost since the previous home. The caller compares two
        at-home readings -- the same principle TEST_SPEED uses with
        GET_POSITION before and after a re-home.
        """
        self._home_axis()
        return self._get_stepper_position()

    def _check_for_lost_steps(self, ref_home_pos):
        """Re-home and report drift from the reference at-home position.

        Returns (lost_steps_detected, drift_mm). ref_home_pos must have been
        captured by a prior _home_and_measure() (i.e. at the endstop), NOT at
        an arbitrary position -- otherwise the difference is just the homing
        travel distance.
        """
        after = self._home_and_measure()
        if ref_home_pos is None or after is None:
            return False, 0.0
        step_dist = self._step_dist() or 0.0
        diff_mm = abs(after - ref_home_pos) * step_dist
        return diff_mm > self.position_tolerance, diff_mm

    def _axis_rotation_distance(self):
        """Belt travel per motor revolution (mm) for the test axis."""
        for rail in self.kin.rails:
            for stepper in rail.get_steppers():
                if stepper.is_active_axis(self.test_axis):
                    return stepper.get_rotation_distance()[0]
        return None

    def _back_emf_constant(self):
        """Motor back-EMF constant Ke (V/(rad/s)), or None if unknown.

        Ke is a winding constant, numerically equal (SI) to the torque constant
        Kt. For a 2-phase hybrid the holding torque is ~2*Kt*I_rated_peak, so
        Ke ~ holding_torque / (2 * rated_current). A direct motor_back_emf
        overrides this estimate.
        """
        if self.motor_back_emf > 0.0:
            return self.motor_back_emf
        if self.motor_holding_torque > 0.0 and self.motor_rated_current > 0.0:
            return self.motor_holding_torque / (2.0 * self.motor_rated_current)
        return None

    def _theoretical_max_speed(self):
        """Back-EMF-limited top speed in mm/s, or None if under-specified.

        At this speed back-EMF == supply voltage, leaving nothing to drive
        current (zero torque), so the carriage cannot physically go faster:
            v = V_supply * C / (2*pi*Ke)
        with C the belt travel per rev and Ke the back-EMF constant.
        """
        if self.supply_voltage <= 0.0:
            return None
        ke = self._back_emf_constant()
        rot_dist = self._axis_rotation_distance()
        if not ke or not rot_dist:
            return None
        return self.supply_voltage * rot_dist / (2.0 * math.pi * ke)

    # Time constants of cruise to reach steady-state phase current. ~5 tau is
    # >99% settled.
    SETTLE_TAUS = 5.0

    def _winding_resistance(self):
        """DC phase resistance (ohms) from config, or None if unset."""
        if self.motor_resistance > 0.0:
            return self.motor_resistance
        return None

    def _peak_dwell_time(self):
        """Cruise-at-peak hold time (s) for current to saturate, or 0.

        tau = L/R; hold ~SETTLE_TAUS*tau plus dwell_margin so the current
        waveform (and thus the real torque at that speed) settles before a skip
        is judged. 0 when inductance/resistance are unknown.
        """
        if self.motor_inductance <= 0.0:
            return 0.0
        r = self._winding_resistance()
        if not r:
            return 0.0
        tau = self.motor_inductance / r
        return (1.0 + self.dwell_margin) * self.SETTLE_TAUS * tau

    def _resolve_speed_end(self, gcmd):
        """Set self.speed_end, honoring/cap-ing against the theoretical max."""
        v_theory = self._theoretical_max_speed()
        if self.speed_end is None:
            if v_theory is None:
                raise gcmd.error(
                    "speed_end is not set and the theoretical top speed is "
                    "unavailable. Set supply_voltage plus motor_back_emf (or "
                    "motor_holding_torque + motor_rated_current), or pass "
                    "SPEED_END."
                )
            self.speed_end = v_theory
            gcmd.respond_info(
                "speed_end unset -> sweeping to theoretical back-EMF max "
                "%.0f mm/s" % v_theory
            )
        elif v_theory is not None and self.speed_end > v_theory:
            gcmd.respond_info(
                "speed_end %.0f mm/s is above the theoretical back-EMF max "
                "%.0f mm/s; capping there" % (self.speed_end, v_theory)
            )
            self.speed_end = v_theory
        elif v_theory is not None:
            gcmd.respond_info(
                "Theoretical back-EMF max %.0f mm/s; testing to speed_end "
                "%.0f mm/s" % (v_theory, self.speed_end)
            )
        if self.speed_end < self.speed_start:
            raise gcmd.error(
                "speed_end (%.0f) is below speed_start (%.0f)"
                % (self.speed_end, self.speed_start)
            )

    def _run_calibration(self, gcmd):
        """Main calibration routine."""
        self.calibration_running = True
        self.calibration_results = []
        self.vibration_rows = []
        self._skip_accel = {}  # {speed: first accel that lost steps, or None}
        self._recommended_shaper = None
        self._speed_resid_buffer = []
        self._vib_file = None
        self._progress_file = None
        self._vib_caldata = None

        # Resume: reload completed speeds + the saved PSD; otherwise clear any
        # stale PSD checkpoint so a later RESUME can't pick up an old run's data.
        completed = set()
        if self._resume:
            completed = self._load_progress()
            self._load_psd_checkpoint()
            gcmd.respond_info(
                "RESUME: %d speeds already done (%s); skipping them"
                % (len(completed),
                   ", ".join("%.0f" % s for s in sorted(completed))
                   or "none"))
        else:
            try:
                if os.path.exists(self._psd_path()):
                    os.remove(self._psd_path())
            except OSError:
                pass
        self._completed_speeds = completed

        gcmd.respond_info("Starting torque curve calibration on %s axis"
                         % self.test_axis.upper())
        gcmd.respond_info(
            "If the axis crashes or skips, STOP IT NOW: hit Mainsail's "
            "Emergency Stop button, or cut printer power. A g-code abort "
            "cannot interrupt a running test."
        )

        # Save original settings
        systime = self.printer.get_reactor().monotonic()
        toolhead_info = self.toolhead.get_status(systime)
        orig_max_accel = toolhead_info["max_accel"]
        orig_max_velocity = toolhead_info["max_velocity"]
        orig_scv = toolhead_info["square_corner_velocity"]
        orig_min_cruise_ratio = toolhead_info["minimum_cruise_ratio"]
        saved_kin_limits = self._save_kinematic_limits()
        saved_reshapers = {}

        try:
            # Resolve speed_end now (may set it from / cap it at the theoretical
            # back-EMF max) before it is used to widen kinematic limits below.
            self._resolve_speed_end(gcmd)

            # Widen per-axis kinematic caps so the commanded accel is the accel
            # the motor actually sees (see _save_kinematic_limits).
            self._apply_kinematic_limits(gcmd)

            # Make every test move a clean, isolated accel/decel: no junction
            # carry-over (SCV=0) and no forced cruise fraction (min ratio=0), so
            # the result can't be skewed by short-move top-speed capping. Moves
            # already fully stop between each (wait_moves), but this keeps the
            # test self-contained against config edits.
            self.gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT SQUARE_CORNER_VELOCITY=0"
                " MINIMUM_CRUISE_RATIO=0"
            )

            # Disable motion-profile reshapers (TOPP-RA / jerk) so the sweep
            # commands raw constant-accel moves; restored in the finally block.
            saved_reshapers = self._disable_reshapers(gcmd)

            # Initial full home so the test never depends on the user having
            # homed first, and so the safe-Z lift (and safe_z_home, which needs
            # X/Y homed to reach its probe point) have everything they need.
            gcmd.respond_info("Homing all axes...")
            self.gcode.run_script_from_command("G28")

            # Lift the gantry clear of the bed before any high-speed sweeping.
            self._raise_z(gcmd)

            # Capture the stationary noise floor (motors energized, idle) once,
            # before any moves, for per-probe quadrature subtraction.
            self._capture_baseline(gcmd)

            # Test region: a center and the maximum half-travel. Each probe's
            # actual move is a triangle sized to v^2/a plus a peak dwell (see
            # _perform_test_move); max_distance bounds the largest one.
            start_pos, end_pos = self._calculate_test_positions()
            center = 0.5 * (start_pos + end_pos)
            max_half = 0.5 * (end_pos - start_pos)
            max_distance = end_pos - start_pos

            gcmd.respond_info(
                "Test region: center %.1f mm, up to %.1f mm travel"
                % (center, max_distance)
            )

            # Peak-velocity dwell so phase current saturates before judging.
            dwell_time = self._peak_dwell_time()
            if dwell_time > 0.0:
                r = self._winding_resistance()
                gcmd.respond_info(
                    "Peak dwell %.1f ms/pass (tau=L/R=%.2f ms, %d tau, +%.0f%%)"
                    % (dwell_time * 1e3,
                       (self.motor_inductance / r) * 1e3,
                       int(self.SETTLE_TAUS), self.dwell_margin * 100.0)
                )
            else:
                gcmd.respond_info(
                    "No peak dwell (set motor_inductance to enable the "
                    "current-saturation hold); using pure triangular moves"
                )

            # Generate speed list
            speeds = []
            speed = self.speed_start
            while speed <= self.speed_end:
                speeds.append(speed)
                speed += self.speed_step

            gcmd.respond_info(
                "Testing %d speeds from %.0f to %.0f mm/s"
                % (len(speeds), self.speed_start, speeds[-1])
            )

            # Test each speed
            for speed_idx, test_speed in enumerate(speeds):
                if not self.calibration_running:
                    gcmd.respond_info("Calibration aborted by user")
                    break

                # RESUME: already checkpointed -> reuse, don't re-run.
                if test_speed in self._completed_speeds:
                    gcmd.respond_info(
                        "[%d/%d] Speed %.0f mm/s: from checkpoint"
                        % (speed_idx + 1, len(speeds), test_speed))
                    continue

                gcmd.respond_info(
                    "\n[%d/%d] Testing speed: %.0f mm/s"
                    % (speed_idx + 1, len(speeds), test_speed)
                )

                # Lowest accel whose triangle+dwell still fits the bed:
                # v^2/a + v*dwell <= max_distance, so a >= v^2/(max_distance -
                # v*dwell). Below this the move can't reach the target speed, so
                # start the search here.
                usable_distance = max_distance - test_speed * dwell_time
                if usable_distance <= 0.0:
                    gcmd.respond_info(
                        "  Speed %.0f mm/s: peak dwell alone exceeds travel, "
                        "skipping" % test_speed
                    )
                    continue
                min_accel_for_speed = (test_speed ** 2) / usable_distance
                start_accel = max(self.accel_start, min_accel_for_speed)

                if start_accel > self.accel_max:
                    gcmd.respond_info(
                        "  Speed %.0f mm/s requires accel > %.0f, skipping"
                        % (test_speed, self.accel_max)
                    )
                    continue

                # Exponential-from-below search with refinement on overshoot.
                #
                # Climb the accel geometrically (x accel_growth) from
                # start_accel until the motor skips. Multiplicative steps cover
                # a huge range in a handful of moves (1000 -> 1e6 is ~10 moves
                # at x2 vs. ~1000 for an additive ramp), and because we always
                # approach from below, the first skip sits just above a
                # known-good value -- we never slam the axis at a wildly-high
                # accel the way a [start, max] bisection's first midpoint probe
                # would.
                #
                # On a skip we don't stop: record it as the upper bound, drop
                # back to the last good accel, halve the growth ratio (finer
                # climb), and resume the exponential from there. Each overshoot
                # shrinks the step, so last_good converges up onto the true
                # threshold from underneath. A probe is never commanded at or
                # above a known skip -- if a finer climb would cross an older
                # skip bound, we split that bracket instead. Stop once the
                # [last_good, skip] bracket is tighter than accel_step (the
                # resolution), or at accel_max if it never skips.
                last_good_accel = 0.0
                skip_accel = None  # lowest accel known to skip (upper bound)
                growth = self.accel_growth
                test_accel = start_accel
                self._speed_resid_buffer = []  # this speed's clean residuals
                while self.calibration_running:
                    # Clamp: never above accel_max, never at/above a known skip.
                    if test_accel > self.accel_max:
                        test_accel = self.accel_max
                    if skip_accel is not None and test_accel >= skip_accel:
                        test_accel = 0.5 * (last_good_accel + skip_accel)

                    ref_home = self._home_and_measure()
                    cap = self._perform_test_move(
                        center, max_half, test_speed, test_accel, dwell_time,
                        capture=self.measure_vibration
                    )
                    lost, diff = self._check_for_lost_steps(ref_home)

                    if self.measure_vibration:
                        vm = self._vibration_metrics(cap, lost)
                        if vm is not None:
                            row = (
                                test_speed, test_accel, 1 if lost else 0, diff,
                                vm["base_rms"], vm["raw_rms"], vm["raw_fpeak"],
                                vm["resid_rms"], vm["resid_fpeak"],
                                vm["resid_afn"], vm["rd_rms"], vm["rd_afn"],
                            )
                            self.vibration_rows.append(row)
                            self._write_vibration_row(row)  # flushed to disk
                            # Buffer the clean residual for deferred, accel-gated
                            # accumulation once this speed's skip edge is known.
                            cr = vm.get("_clean_resid")
                            if cr is not None:
                                self._speed_resid_buffer.append((test_accel, cr))
                            # a@fn is measured at the detected mode (resid_fpeak)
                            # -- the de-floored amplitude of the real resonance.
                            gcmd.respond_info(
                                "    mode=%.0fHz resid a@mode=%.1f | ringdown "
                                "a@mode=%.1f | resid rms=%.0f floor=%.0f"
                                % (vm["resid_fpeak"], vm["resid_afn"],
                                   vm["rd_afn"], vm["resid_rms"], vm["base_rms"])
                            )

                    if lost:
                        skip_accel = test_accel
                        gcmd.respond_info(
                            "  Accel %.0f: FAILED (lost %.3f mm)"
                            % (test_accel, diff)
                        )
                        if last_good_accel <= 0.0:
                            gcmd.respond_info(
                                "  Skipped at the starting accel -- real limit "
                                "is below %.0f" % test_accel
                            )
                            break
                        # Overshot: refine. Halve the growth ratio toward 1 and
                        # resume the climb from the last known-good accel.
                        growth = 1.0 + 0.5 * (growth - 1.0)
                        test_accel = last_good_accel * growth
                    else:
                        gcmd.respond_info("  Accel %.0f: OK" % test_accel)
                        last_good_accel = test_accel
                        if test_accel >= self.accel_max:
                            gcmd.respond_info(
                                "  Reached accel_max %.0f without skipping"
                                % self.accel_max
                            )
                            break
                        test_accel = last_good_accel * growth

                    # Converged: bracket tighter than the resolution.
                    if (skip_accel is not None
                            and skip_accel - last_good_accel <= self.accel_step):
                        break

                # Record result (0 means it skipped at the very first accel)
                self.calibration_results.append((test_speed, last_good_accel))
                # The lowest accel that lost steps at this speed (None if it
                # never skipped) -- the measured upper bound the mode margins
                # are derived against.
                self._skip_accel[test_speed] = skip_accel
                # Now the skip boundary is known: fold this speed's clean,
                # sub-skip-threshold residuals into the resonance spectrum
                # (drops the near-skip stutter band).
                self._accumulate_speed_psd(last_good_accel)
                # Checkpoint this speed (result + PSD) so an interruption only
                # costs the in-progress speed, not the whole sweep.
                self._write_progress_row(
                    test_speed, last_good_accel, skip_accel)
                self._save_psd_checkpoint()
                gcmd.respond_info(
                    "  Result: %.0f mm/s -> max accel %.0f mm/s^2"
                    % (test_speed, last_good_accel)
                )

        finally:
            # Restore original settings
            self.gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT ACCEL=%.1f VELOCITY=%.1f"
                " SQUARE_CORNER_VELOCITY=%.4f MINIMUM_CRUISE_RATIO=%.4f"
                % (orig_max_accel, orig_max_velocity,
                   orig_scv, orig_min_cruise_ratio)
            )
            self._restore_kinematic_limits(saved_kin_limits, gcmd)
            self._restore_reshapers(saved_reshapers, gcmd)
            # Vibration rows are written to disk as they're collected; just
            # close the handle here so even an aborted/shut-down sweep keeps
            # everything captured up to the failure point.
            self._close_vibration_csv(gcmd)
            self._close_progress()
            self.calibration_running = False

        # Filter out failed results (accel = 0); sort by speed so a resumed run
        # (loaded checkpoint rows + newly-run speeds) is always ascending.
        valid_results = sorted(
            (r for r in self.calibration_results if r[1] > 0),
            key=lambda r: r[0])

        if len(valid_results) >= 2:
            # Save results
            self._save_results(gcmd, valid_results)
            self._emit_motion_modes(gcmd, valid_results)
        else:
            gcmd.respond_info(
                "Calibration incomplete - not enough valid data points"
            )

    def _compute_mode_margins(self, valid_results):
        """Mode SAFETY_MARGINs derived from the measured skip boundary.

        The stored torque curve is the last accel that did NOT lose steps (the
        measured boundary); SAFETY_MARGIN scales it. Skip data legitimately
        pins only one thing -- how close to the edge it's safe to run -- so the
        Speed margin is derived from `band`, the median bracket width between
        last-good and first-skip accel ((As-Ag)/Ag): a crisp edge (small band)
        lets Speed sit right under the boundary, a fuzzy/coarse one backs it
        off. The calmer modes step down from Speed by fixed reliability offsets
        (these are ride-quality choices, not skip-derived):

            speed    = clamp(0.97 - band/2, 0.90, 0.97)
            balanced = speed - 0.07
            quality  = speed - 0.12
            safe     = speed - 0.25   (all floored at 0.50)

        Returns (margins_dict, band).
        """
        bands = []
        for speed_v, ag in valid_results:
            as_ = self._skip_accel.get(speed_v)
            if as_ and ag > 0.0 and as_ > ag:
                bands.append((as_ - ag) / ag)
        if bands:
            bands.sort()
            band = bands[len(bands) // 2]  # median bracket width
        else:
            band = 0.06  # never skipped (accel_max-limited): nominal backoff
        band = min(0.15, max(0.02, band))
        speed = min(0.97, max(0.90, 0.97 - 0.5 * band))
        margins = {
            "speed": speed,
            "balanced": speed - 0.07,
            "quality": speed - 0.12,
            "safe": speed - 0.25,
        }
        return ({k: round(max(0.5, v), 2) for k, v in margins.items()}, band)

    def _emit_motion_modes(self, gcmd, valid_results):
        """Write end_game/motion_modes.cfg: switchable gcode_macros that layer
        the three calibrated knobs (TOPP-RA torque curve, input shaper, jerk
        limiting) into named profiles from one sweep."""
        margins, band = self._compute_mode_margins(valid_results)
        ax = self.test_axis
        AX = ax.upper()
        curve = self.target_curve or "my_curve"
        rec = self._recommended_shaper  # (name, freq) or None

        def shaper_on():
            if rec:
                return ("    SET_INPUT_SHAPER SHAPER_TYPE_%s=%s "
                        "SHAPER_FREQ_%s=%.1f" % (AX, rec[0], AX, rec[1]))
            return ("    # no recommended shaper (run MEASURE_VIBRATION=1); "
                    "leaving shaper as-is")

        def macro(name, desc, margin, shaper_line, jerk_line, tag):
            return "\n".join([
                "[gcode_macro %s]" % name,
                "description: %s" % desc,
                "gcode:",
                "    TORQUE_CURVE_SET NAME=%s ENABLE=1 SAFETY_MARGIN=%.2f"
                % (curve, margin),
                shaper_line,
                jerk_line,
                "    { action_respond_info(\"Motion mode: %s\") }" % tag,
                "",
            ])

        rows = "\n".join(
            "#   %8.0f  %12.0f  %s"
            % (s, ag, ("%.0f" % self._skip_accel[s]
                       if self._skip_accel.get(s) else "(no skip)"))
            for s, ag in valid_results)
        header = "\n".join([
            "# Motion modes -- generated by TORQUE_CURVE_CALIBRATE (axis %s)" % AX,
            "#",
            "# Each macro layers the three calibrated knobs into a profile:",
            "#   TOPP-RA torque curve (velocity-dependent accel ceiling),",
            "#   input shaper (ringing cancellation), jerk limiting (S-curve).",
            "# Switch any time, or call one from PRINT_START.",
            "#",
            "# SAFETY_MARGIN values are fractions of the MEASURED skip boundary",
            "# (the last accel that did not lose steps). Speed is derived from",
            "# band=%.0f%% (median last-good->first-skip bracket); the calmer modes"
            % (band * 100.0),
            "# step down from it by fixed reliability offsets -- edit to taste:",
            "#   BATSHIT_BENCHY=%.2f OG=%.2f ENDGAME=%.2f QUIET=%.2f"
            % (margins["speed"], margins["balanced"],
               margins["quality"], margins["safe"]),
            "#",
            "# Measured boundary per speed:",
            "#     speed       last_good     first_skip",
            rows,
            "",
            "",
        ])
        speed_shaper = ("    SET_INPUT_SHAPER SHAPER_FREQ_%s=0" % AX) if rec \
            else ("    # (shaper left as-is; nothing to disable)")
        # Every ENABLE=1 mode sets all three phase flags explicitly -- they
        # persist across SET_JERK_LIMIT calls, so a mode must fully define the
        # jerk state or the previous mode's flags leak in. Benchy keeps ramps
        # bare (smooth_ramps off) but rounds corners to carry speed through them
        # (round_corners is independent of smooth_ramps and check_move-guarded).
        jerk_benchy = ("    SET_JERK_LIMIT ENABLE=1 SMOOTH_RAMPS=0 "
                       "BLEND_JUNCTIONS=0 ROUND_CORNERS=1")
        jerk_endgame = ("    SET_JERK_LIMIT ENABLE=1 SMOOTH_RAMPS=1 "
                        "BLEND_JUNCTIONS=1 ROUND_CORNERS=0 AUTO=1")
        jerk_quiet = ("    SET_JERK_LIMIT ENABLE=1 SMOOTH_RAMPS=1 "
                      "BLEND_JUNCTIONS=1 ROUND_CORNERS=0 AUTO=1 "
                      "AUTO_JERK_RATIO=0.7")
        body = "\n".join([
            macro("BATSHIT_BENCHY",
                  "Batshit benchy: bare TOPP-RA + corner rounding, no "
                  "shaping (%s)" % AX,
                  margins["speed"], speed_shaper, jerk_benchy,
                  "BATSHIT BENCHY (bare TOPP-RA + rounded corners)"),
            macro("ENDGAME",
                  "Endgame: TOPP-RA + shaper + auto jerk (%s)" % AX,
                  margins["quality"], shaper_on(), jerk_endgame, "ENDGAME"),
            macro("OG",
                  "OG: shaped, no jerk smoothing (%s)" % AX,
                  margins["balanced"], shaper_on(),
                  "    SET_JERK_LIMIT ENABLE=0", "OG"),
            macro("QUIET",
                  "Quiet: TOPP-RA + shaper + gentle jerk (%s)" % AX,
                  margins["safe"], shaper_on(), jerk_quiet, "QUIET"),
        ])
        out_dir, _ = self._output_dir_stem("")  # end_game/ root
        path = os.path.join(out_dir, "motion_modes.cfg")
        try:
            with open(path, "w") as f:
                f.write(header + body)
        except IOError:
            logging.exception(
                "torque_curve_calibrate: motion_modes write failed")
            return
        if not self.target_curve:
            gcmd.respond_info(
                "  note: set NAME= in the macros to your [torque_curve] (no "
                "target_curve configured; used placeholder 'my_curve')")
        gcmd.respond_info(
            "Motion modes -> %s" % path)
        gcmd.respond_info(
            "  BATSHIT_BENCHY=%.2f OG=%.2f ENDGAME=%.2f QUIET=%.2f "
            "(x measured skip boundary, band=%.0f%%)"
            % (margins["speed"], margins["balanced"], margins["quality"],
               margins["safe"], band * 100.0))

    def _output_dir_stem(self, subdir="torque_curve"):
        """(<config_dir>/end_game/<subdir>, output-file stem); makes the dir.

        Everything this module produces lives under an `end_game/` parent, split
        into `torque_curve/` (sweep results + vibration data) and
        `input_shaping/` (resonance spectrum + shaper graph).
        """
        config_file = self.printer.get_start_args().get("config_file")
        base_dir = (os.path.dirname(os.path.abspath(config_file))
                    if config_file else os.getcwd())
        out_dir = os.path.join(base_dir, "end_game", subdir)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.output_file))[0]
        return out_dir, stem

    # --- Resume checkpoints ---------------------------------------------------
    def _progress_path(self):
        out_dir, stem = self._output_dir_stem()
        return os.path.join(
            out_dir, "%s_%s_progress.csv" % (stem, self.test_axis))

    def _psd_path(self):
        out_dir, stem = self._output_dir_stem()
        return os.path.join(
            out_dir, "%s_%s_psd.npz" % (stem, self.test_axis))

    def _write_progress_row(self, speed, last_good, skip):
        """Append one completed speed's result, flushed to disk, so RESUME can
        skip it. Lazily opens (append on resume, truncate+header otherwise)."""
        if self._progress_file is None:
            path = self._progress_path()
            self._progress_file = open(path, "a" if self._resume else "w")
            if self._progress_file.tell() == 0:
                self._progress_file.write(
                    "# Per-speed checkpoint for RESUME (axis %s)\n"
                    "speed,last_good_accel,first_skip_accel\n"
                    % self.test_axis.upper())
        self._progress_file.write(
            "%.2f,%.2f,%s\n"
            % (speed, last_good, "" if skip is None else "%.2f" % skip))
        self._progress_file.flush()
        os.fsync(self._progress_file.fileno())

    def _close_progress(self):
        if self._progress_file is not None:
            try:
                self._progress_file.close()
            finally:
                self._progress_file = None

    def _load_progress(self):
        """Read the checkpoint into calibration_results + _skip_accel and
        return the set of completed speeds (empty if none / no file)."""
        path = self._progress_path()
        done = set()
        if not os.path.exists(path):
            return done
        with open(path) as f:
            for line in f:
                if line[:1] == "#" or line.startswith("speed"):
                    continue
                p = line.strip().split(",")
                if len(p) < 2:
                    continue
                speed, last_good = float(p[0]), float(p[1])
                skip = float(p[2]) if len(p) > 2 and p[2] != "" else None
                self.calibration_results.append((speed, last_good))
                self._skip_accel[speed] = skip
                done.add(speed)
        return done

    def _save_psd_checkpoint(self):
        """Persist the accumulated resonance spectrum so the shaper graph
        survives an interruption. Cheap enough to write once per speed."""
        cd = self._vib_caldata
        if cd is None:
            return
        try:
            import numpy as np
            np.savez(self._psd_path(), freq_bins=cd.freq_bins,
                     psd_x=cd.psd_x, psd_y=cd.psd_y, psd_z=cd.psd_z,
                     psd_sum=cd.psd_sum)
        except Exception:
            logging.exception(
                "torque_curve_calibrate: PSD checkpoint save failed")

    def _load_psd_checkpoint(self):
        """Rebuild the PSD accumulator from a saved checkpoint, if present."""
        path = self._psd_path()
        if not os.path.exists(path):
            return
        sc = self._shaper_calibrate()
        if sc is None:
            return
        try:
            import numpy as np
            from . import shaper_calibrate
            d = np.load(path)
            cd = shaper_calibrate.CalibrationData(
                d["freq_bins"], d["psd_sum"], d["psd_x"], d["psd_y"],
                d["psd_z"])
            cd.set_numpy(sc.numpy)
            self._vib_caldata = cd
        except Exception:
            logging.exception(
                "torque_curve_calibrate: PSD checkpoint load failed")

    def _write_vibration_row(self, row):
        """Append one probe's metrics to the CSV, flushing immediately.

        Written incrementally (not batched at the end) so a mid-sweep
        accelerometer dropout / shutdown can't wipe everything collected so far
        -- the data is on disk after every probe.
        """
        if self._vib_file is None:
            out_dir, stem = self._output_dir_stem()
            self._vib_path = os.path.join(
                out_dir, "%s_%s_vibration.csv" % (stem, self.test_axis)
            )
            self._vib_file = open(self._vib_path, "a" if self._resume else "w")
        if self._vib_file.tell() == 0:
            fn = self._shaper_freq()
            self._vib_file.write(
                "# Vibration sweep on %s axis (input shaping %s)\n"
                "# Input shaper f_n: %s Hz "
                "(rd_freq column is the measured ring-down frequency)\n"
                "# All metrics quadrature-subtract base_rms (the noise floor).\n"
                "# resid_* = burst with commanded accel subtracted; rd_* = "
                "stationary ring-down. resid_fpeak = MEASURED resonance (mode); "
                "*_afn = de-floored amplitude at that measured mode (the clean "
                "numbers); raw_* shown for contrast.\n"
                "speed,accel,lost,drift_mm,base_rms,raw_rms,raw_fpeak,"
                "resid_rms,resid_fpeak,resid_afn,rd_rms,rd_afn\n"
                % (self.test_axis.upper(),
                   "OFF/raw" if self.vibration_shaper_off else "ON",
                   "%.1f" % fn if fn else "n/a")
            )
        self._vib_file.write(
            "%.2f,%.2f,%d,%.4f,%.4f,%.4f,%.2f,%.4f,%.2f,%.4f,%.4f,%.4f\n"
            % row
        )
        self._vib_file.flush()
        os.fsync(self._vib_file.fileno())

    def _close_vibration_csv(self, gcmd):
        """Close the incremental CSV (safe to call when never opened)."""
        if self._vib_file is None:
            return
        try:
            self._vib_file.close()
        finally:
            self._vib_file = None
        gcmd.respond_info(
            "Vibration data -> %s (%d points)"
            % (self._vib_path, len(self.vibration_rows))
        )
        # Measured resonance across the sweep: the MEDIAN of each probe's clean
        # dominant mode. Median, not amplitude-weighted -- the structural mode
        # is speed/accel-invariant and recurs across probes, while artifacts
        # (near-skip stutter, residual leakage) are sparse but can be high
        # amplitude; a weighted mean lets a few loud outliers hijack the number,
        # the median ignores them. row col 8 = resid_fpeak (0 when no clean mode).
        modes = sorted(r[8] for r in self.vibration_rows if r[8] > 0.0)
        if modes:
            mid = modes[len(modes) // 2]
            shaper = self._shaper_freq()
            gcmd.respond_info(
                "Measured %s resonance: %.1f Hz (median of %d clean probes)%s"
                % (self.test_axis.upper(), mid, len(modes),
                   "" if not shaper
                   else " -- configured input shaper is %.1f Hz" % shaper)
            )
        # Fit shapers to the real-move spectrum and emit the input-shaper graph.
        self._emit_shaper_graph(gcmd)

    def _save_results(self, gcmd, results):
        """Save calibration results to CSV file."""
        out_dir, stem = self._output_dir_stem()
        output_path = os.path.join(
            out_dir, "%s_%s.csv" % (stem, self.test_axis)
        )

        # Write CSV
        with open(output_path, "w") as f:
            f.write("# Torque curve calibration results\n")
            f.write("# Axis: %s\n" % self.test_axis.upper())
            f.write("# Test distance: %.1f mm\n" % self.test_move_distance)
            f.write("# Position tolerance: %.3f mm\n" % self.position_tolerance)
            f.write("speed,max_accel\n")
            for speed, accel in results:
                f.write("%.2f,%.2f\n" % (speed, accel))

        gcmd.respond_info("\nCalibration complete!")
        gcmd.respond_info("Results saved to: %s" % output_path)
        gcmd.respond_info("Data points: %d" % len(results))

        # Show summary
        gcmd.respond_info("\nSummary:")
        for speed, accel in results:
            gcmd.respond_info("  %.0f mm/s -> %.0f mm/s^2" % (speed, accel))

        # Plot the curve to a PNG alongside the CSV
        self._plot_curve(gcmd, output_path)

        # Update target torque_curve if specified
        if self.target_curve:
            try:
                tc = self.printer.lookup_object(
                    "torque_curve %s" % self.target_curve
                )
                speeds = [r[0] for r in results]
                accels = [r[1] for r in results]
                tc.set_curve_data(speeds, accels)
                gcmd.respond_info(
                    "\nUpdated torque_curve '%s' with calibration data"
                    % self.target_curve
                )
            except Exception as e:
                gcmd.respond_info(
                    "\nWarning: Could not update torque_curve: %s" % str(e)
                )

    def _plot_curve(self, gcmd, csv_path):
        """Render the curve to a PNG next to the CSV (non-blocking).

        Spawns scripts/plot_torque_curve.py with the system python3 so it can
        use matplotlib without pulling it into klippy-env. Fire-and-forget:
        the calibration is already done and the toolhead idle, so the fork
        cannot disturb motion timing.
        """
        png_path = os.path.splitext(csv_path)[0] + ".png"
        klipper_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        script = os.path.join(klipper_root, "scripts", "plot_torque_curve.py")
        try:
            subprocess.Popen(
                ["python3", script, csv_path, png_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            gcmd.respond_info(
                "Plotting torque curve -> %s (needs python3-matplotlib)"
                % png_path
            )
        except Exception as e:
            gcmd.respond_info("Could not start plot: %s" % str(e))

    cmd_TORQUE_CURVE_CALIBRATE_help = (
        "Run automated torque curve calibration"
    )
    def cmd_TORQUE_CURVE_CALIBRATE(self, gcmd):
        if self.calibration_running:
            raise gcmd.error("Calibration already in progress")

        # Allow parameter overrides
        self.speed_start = gcmd.get_float(
            "SPEED_START", self.speed_start, above=0.0
        )
        self.speed_end = gcmd.get_float(
            "SPEED_END", self.speed_end, above=0.0
        )
        self.speed_step = gcmd.get_float(
            "SPEED_STEP", self.speed_step, above=0.0
        )
        self.accel_start = gcmd.get_float(
            "ACCEL_START", self.accel_start, above=0.0
        )
        self.accel_max = gcmd.get_float(
            "ACCEL_MAX", self.accel_max, above=0.0
        )
        self.accel_step = gcmd.get_float(
            "ACCEL_STEP", self.accel_step, above=0.0
        )
        self.accel_growth = gcmd.get_float(
            "ACCEL_GROWTH", self.accel_growth, above=1.0
        )
        self.test_cycles = gcmd.get_int(
            "TEST_CYCLES", self.test_cycles, minval=1
        )
        self.output_file = gcmd.get("OUTPUT_FILE", self.output_file)
        self.measure_vibration = bool(gcmd.get_int(
            "MEASURE_VIBRATION", 1 if self.measure_vibration else 0,
            minval=0, maxval=1
        ))
        self.vibration_shaper_off = bool(gcmd.get_int(
            "VIBRATION_SHAPER_OFF", 1 if self.vibration_shaper_off else 0,
            minval=0, maxval=1
        ))
        self.ring_down_time = gcmd.get_float(
            "RING_DOWN_TIME", self.ring_down_time, minval=0.0
        )
        self.baseline_time = gcmd.get_float(
            "BASELINE_TIME", self.baseline_time, minval=0.0
        )
        self.apply_shaper = bool(gcmd.get_int(
            "APPLY_SHAPER", 1 if self.apply_shaper else 0, minval=0, maxval=1
        ))
        # RESUME=1 continues an interrupted sweep: speeds already checkpointed
        # to <stem>_<axis>_progress.csv are skipped, their results + the saved
        # PSD accumulator are reloaded, and new data is appended.
        self._resume = bool(gcmd.get_int("RESUME", 0, minval=0, maxval=1))
        # Resolve the accel chip if vibration was just enabled at runtime.
        if self.measure_vibration and self.accel_chip is None:
            self.accel_chip = self._lookup_accel_chip()
            if self.accel_chip is None:
                raise gcmd.error(
                    "MEASURE_VIBRATION requested but no accelerometer found; "
                    "set accel_chip (e.g. lis2dw / adxl345)"
                )

        self._run_calibration(gcmd)


def load_config_prefix(config):
    return TorqueCurveCalibrate(config)
