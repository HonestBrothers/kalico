# Automated calibration for stepper motor torque-speed curves
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math
import os
import subprocess


class _ModeConvergence:
    """Welch-averaged mode-frequency estimator with a per-burst SNR gate and a
    running convergence test (Pass 1 of the unified calibration).

    Fed one residual PSD per burst. Each burst is gated on its peak-to-median
    ratio (SNR); accepted bursts are averaged onto a shared frequency grid, the
    averaged peak is refit by parabolic interpolation, and the frequency is
    declared converged once the last `window` refits all sit within `tol` Hz of
    their mean. Pure and numpy-only (numpy handed in) so it can be exercised on
    captured spectra with no hardware. See calibration_flow.md, Pass 1.
    """

    def __init__(self, snr_min, tol, window, max_bursts, band_lo, band_hi):
        self.snr_min = snr_min
        self.tol = tol
        self.window = max(2, int(window))
        self.max_bursts = max(1, int(max_bursts))
        self.band_lo = band_lo
        self.band_hi = band_hi
        self._grid = None        # reference freq grid (first accepted burst)
        self._psd_sum = None     # running sum of accepted PSDs, on _grid
        self.n_accepted = 0
        self.n_seen = 0          # total bursts fed (accepted + rejected)
        self.per_burst = []      # single-burst peak fits (the convergence input)
        self.history = []        # averaged-PSD peak after each accept (reported)
        self.converged = False

    def add_burst(self, np, freqs, spec):
        """Fold one burst PSD (freqs, magnitude) in. Returns a status dict:
        {accepted, snr, f_hat, f_burst, converged, n_accepted}.

        Convergence is decided on the spread of the recent *single-burst* peaks
        (do the individual measurements agree?), NOT on the averaged-PSD peak.
        The averaged peak is 1/n-damped, so it stops moving from estimator
        inertia alone -- a scattered mode (the ring-down failure this guards
        against) would false-converge on it. The averaged peak is still the
        reported f_n: lowest variance once the inputs are known to agree."""
        self.n_seen += 1
        band = (freqs >= self.band_lo) & (freqs <= self.band_hi)
        snr = 0.0
        if band.any():
            sb = spec[band]
            med = float(np.median(sb))
            snr = float(sb.max()) / med if med > 0.0 else 0.0
        if not band.any() or snr < self.snr_min:
            return {"accepted": False, "snr": snr, "f_hat": self._last(),
                    "f_burst": 0.0, "converged": self.converged,
                    "n_accepted": self.n_accepted}
        # Accepted. This burst's own peak feeds the convergence test...
        f_burst = self._fit_peak(np, freqs, spec)
        self.per_burst.append(f_burst)
        # ...and its PSD averages (onto a shared grid; interp handles bursts of
        # differing length) into the low-variance reported estimate.
        if self._grid is None:
            self._grid = freqs.copy()
            self._psd_sum = spec.copy()
        else:
            self._psd_sum = self._psd_sum + np.interp(self._grid, freqs, spec)
        self.n_accepted += 1
        f_hat = self._fit_peak(np, self._grid, self._psd_sum / self.n_accepted)
        self.history.append(f_hat)
        # Converged when the last `window` single-burst peaks all sit within tol
        # of their mean (they agree) AND the averaged estimate is consistent
        # with them (no bimodal split hiding inside a tight window).
        if len(self.per_burst) >= self.window:
            win = self.per_burst[-self.window:]
            m = sum(win) / len(win)
            if (all(v > 0.0 and abs(v - m) <= self.tol for v in win)
                    and f_hat > 0.0 and abs(f_hat - m) <= self.tol):
                self.converged = True
        return {"accepted": True, "snr": snr, "f_hat": f_hat,
                "f_burst": f_burst, "converged": self.converged,
                "n_accepted": self.n_accepted}

    def _fit_peak(self, np, freqs, spec):
        """Sub-bin peak of spec within the band by parabolic interpolation on
        the log-magnitude around the max bin, or 0.0 if there's no usable peak."""
        idx = np.where((freqs >= self.band_lo) & (freqs <= self.band_hi))[0]
        if len(idx) < 3:
            return 0.0
        k = int(idx[int(np.argmax(spec[idx]))])
        if k <= 0 or k >= len(spec) - 1:
            return float(freqs[k])
        y0 = math.log(max(float(spec[k - 1]), 1e-300))
        y1 = math.log(max(float(spec[k]), 1e-300))
        y2 = math.log(max(float(spec[k + 1]), 1e-300))
        denom = y0 - 2.0 * y1 + y2
        # denom >= 0 means the "peak" isn't concave (flat/edge) -> no sub-bin fit
        delta = 0.5 * (y0 - y2) / denom if denom < 0.0 else 0.0
        delta = max(-0.5, min(0.5, delta))
        return float(freqs[k]) + delta * float(freqs[k + 1] - freqs[k])

    def _last(self):
        return self.history[-1] if self.history else 0.0

    @property
    def exhausted(self):
        """True once the burst budget is spent without converging."""
        return not self.converged and self.n_seen >= self.max_bursts

    def verdict(self):
        """(status, f_n, detail). status in
        converged / not_converged / no_signal."""
        if self.converged:
            return ("converged", self._last(),
                    "%d/%d bursts accepted; last %d refits within %.2f Hz"
                    % (self.n_accepted, self.n_seen, self.window, self.tol))
        if self.n_accepted == 0:
            return ("no_signal", 0.0,
                    "no burst cleared SNR>=%.1f (%d seen)"
                    % (self.snr_min, self.n_seen))
        return ("not_converged", self._last(),
                "%d/%d bursts accepted; estimate still moving (last=%.1f Hz)"
                % (self.n_accepted, self.n_seen, self._last()))


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
        # Immutable config value. self.speed_end is the *per-run* working copy:
        # _resolve_speed_end() writes the resolved ceiling back into it, and the
        # PASS presets narrow it, so it must be restored from this at the start
        # of every run or one run's resolution leaks into the next.
        self._cfg_speed_end = config.getfloat("speed_end", None, above=0.0)
        self.speed_end = self._cfg_speed_end
        self.speed_step = config.getfloat("speed_step", 25.0, above=0.0)

        # Pass 1 (mode ID) sweeps only the RESONANCE-limited band; above it the
        # axis is torque-limited (back-EMF) and the mode barely rings, so those
        # bursts are low-SNR noise that drags the clustered f_n off the true
        # mode. Measured on this machine: at 300mm/s the axis tolerates a mode
        # amplitude of ~8800 at its accel limit (SNR 138-268, f_peak locked
        # 74.2-74.9); by 500mm/s it skips at a@mode ~290 with SNR 30-39 and
        # f_peak scattered 54-105Hz. The knee sits at ~400-450 (back-EMF is
        # ~10.4V of the 24V rail there, and L/R=1.9ms swamps the 444us step
        # period). So cap Pass 1 at the knee. Pass 2 (torque) still needs the
        # full range -- that band IS the torque curve.
        self.modeid_speed_end = config.getfloat(
            "modeid_speed_end", 400.0, above=0.0
        )

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

        # Minimum length of the acceleration ramp, in MICROSTEPS. This is the
        # real ceiling on a torque sweep -- accel_max is only a backstop.
        #
        # The ramp spans n = (v^2 / 2a) * steps_per_mm microsteps. Requiring
        # n >= min_ramp_steps gives a per-speed accel ceiling
        #     a_max(v) = v^2 * steps_per_mm / (2 * min_ramp_steps)
        # Above it the "ramp" is shorter than a few microsteps, so:
        #   * the motor never physically experiences the accel (it just starts
        #     stepping at the cruise rate) -- it CANNOT skip, so the search
        #     never brackets and climbs forever, and
        #   * the commanded profile becomes a ~instantaneous velocity step,
        #     which segfaults host step generation.
        # e.g. 100mm/s @ 2.5e6 on a 80 step/mm axis = a 0.002mm / 0.16-microstep
        # "ramp" -- that exact move killed the host.
        #
        # a_max grows as v^2 while the motor's skip accel falls with speed
        # (back-EMF), so below their crossing speed the torque limit is simply
        # not measurable -- the sweep now says so instead of crashing.
        self.min_ramp_steps = config.getfloat(
            "min_ramp_steps", 4.0, minval=0.0
        )


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

        # Lost-step detection threshold, expressed in FULL STEPS (not mm).
        # A 2-phase hybrid only loses sync in whole electrical cycles (4 full
        # steps), so a genuine skip lands >= ~4 full steps off at the verifying
        # re-home. The endstop's own re-home scatter is ~1 full step and must
        # NOT be flagged. Working in full steps puts the gate on that physics
        # and auto-scales with rotation_distance / microsteps across machines.
        # Default 2.5 sits between the ~1.4-step re-home phantom and the 4-step
        # pole slip. See _lost_step_tol_mm().
        self.position_tolerance_steps = config.getfloat(
            "position_tolerance_steps", 2.5, above=0.0
        )
        # Optional hard mm override. None -> derive the threshold from
        # position_tolerance_steps * full-step distance (the default path).
        self.position_tolerance = config.getfloat(
            "position_tolerance", None, above=0.0
        )
        # Microsteps of the test-axis stepper, read from its own config section
        # so we can convert the raw MCU step count (microsteps) into full steps.
        self._axis_microsteps = None
        try:
            sec = config.getsection("stepper_" + self.test_axis)
            self._axis_microsteps = sec.getint("microsteps", None, minval=1)
        except Exception:
            self._axis_microsteps = None

        # Settle dwell held AFTER the stress burst, BEFORE the verification
        # re-home. The FF is suspended for the sweep (raw trapezoids), so the
        # lightly-damped Y mode rings freely after the burst; if the re-home
        # fires while it is still oscillating, the endstop triggers at a
        # displaced point and reports a phantom ~1-full-step "lost" drift that
        # is NOT a real skip. Waiting ~5 mode time-constants (tau = 1/(zeta*wn),
        # ~0.11s for red's 74.8Hz/zeta=0.019 -> ~0.55s) lets it ring down first
        # so only genuine multi-mm stalls are flagged. 0 = disable.
        self.settle_time = config.getfloat("settle_time", 0.6, minval=0.0)

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

        # --- Pass 1: mode-ID convergence test -------------------------------
        # A single burst's PSD peak has real variance (this is what sank the
        # ring-down -- its zero-crossings landed in the noise). So we Welch-
        # average the residual PSD across bursts that clear an SNR gate, refit
        # the averaged peak each time, and declare the frequency converged only
        # once the running estimate stops moving. Runs passively alongside the
        # sweep (each in-sync probe's clean residual is a burst); P1_ABORT=1
        # raises if it never pins the mode down, so a bad mode ID can't quietly
        # feed Pass 2. Band = [resonance_min_freq, resonance_max_freq].
        self.p1_snr_min = config.getfloat("p1_snr_min", 6.0, minval=1.0)
        # Convergence gate: the last p1_converge_window per-burst peaks must
        # agree within this tol (Hz). Must be looser than the FFT bin (~0.78 Hz
        # for a ~1.3 s / 1600 Hz capture) or independent burst peaks can never
        # repeat tightly enough to latch. On red the mode locks in a ~0.4 Hz
        # band while the pre-lock scatter jumps >4 Hz/burst, so 2 Hz sits in
        # that gap: loose enough to latch, tight enough to reject a wanderer.
        self.p1_converge_tol = config.getfloat(
            "p1_converge_tol", 2.0, above=0.0)
        self.p1_converge_window = config.getint(
            "p1_converge_window", 3, minval=2)
        self.p1_max_bursts = config.getint("p1_max_bursts", 20, minval=1)
        # Default disposition of the abort gate; overridable per-run with
        # P1_ABORT=. Passive (report-only) by default so it never fails a sweep
        # that isn't being run as a dedicated Pass-1 mode ID.
        self.p1_abort = config.getboolean("p1_abort", False)

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
        # Pass-1 convergence tracker (built lazily on the first fed burst) and
        # its final verdict tuple (status, f_n, detail), set at finalization.
        self._p1 = None
        self._p1_verdict = None
        # Which PASS preset the current run selected ("" = plain sweep).
        self._pass_name = ""

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
        # topp-ra-v3: the UNIFIED emitter's jerk limiting also replaces the raw
        # trapezoid with an S-curve, so zero it for the sweep (0 = sharp
        # constant-accel ladders). Restored (or re-derived by the FF) after.
        if getattr(self.toolhead, "unified_max_jerk", 0.0):
            saved["ujerk"] = self.toolhead.unified_max_jerk
            self.toolhead.unified_max_jerk = 0.0
            gcmd.respond_info("Zeroed unified_max_jerk for calibration")
        # The model-inverse FF rewrites the trapezoid too: it adds p1*v + p2*a to
        # the commanded position. It MUST be suspended for the sweep -- always,
        # not just for raw-vibration capture -- for the SAME reason jerk limiting
        # is zeroed above. With jerk limiting off the sweep commands raw
        # trapezoids whose accel steps instantaneously; the FF's p2*a term then
        # injects a p2*da position jump in that single step (at the sweep's
        # extreme accels, p2=4.5e-6 * 10e6 = ~45mm) -> a step rate no MCU can
        # emit -> "Timer too close" shutdown. It also biases the measured limit.
        # Restored in _restore_reshapers.
        ff = getattr(self.toolhead, "model_inverse_ff", None)
        if ff is not None and getattr(ff, "enabled", False):
            saved["ff"] = ff
            ff.enabled = False
            try:
                ff._push()              # clear the C seam; max_da -> None
            except Exception:
                logging.exception(
                    "torque_curve_calibrate: FF disable push failed")
            gcmd.respond_info("Suspended model-inverse FF for calibration")
        # Optional raw-mode: also drop the legacy INPUT SHAPER so the
        # accelerometer sees the unshaped resonance (default keeps it on =
        # real-world). The FF above is already unconditionally off.
        if self.measure_vibration and self.vibration_shaper_off:
            if self.printer.lookup_object("input_shaper", None):
                self.gcode.run_script_from_command("DISABLE_INPUT_SHAPER")
                saved["shaper"] = True
                gcmd.respond_info(
                    "Input shaping disabled for raw vibration measurement")
        return saved

    def _restore_reshapers(self, saved, gcmd):
        """Re-enable whatever _disable_reshapers turned off."""
        if not saved:
            return
        self.toolhead.flush_step_generation()
        if "topp" in saved:
            saved["topp"].enabled = True
        if "shaper" in saved:
            self.gcode.run_script_from_command("ENABLE_INPUT_SHAPER")
        if "ujerk" in saved:
            self.toolhead.unified_max_jerk = saved["ujerk"]
        if "ff" in saved:
            saved["ff"].enabled = True
            try:
                # Re-push restores the C seam + max_da; if unified_auto_jerk is
                # on, apply_auto_jerk here re-derives unified_max_jerk from the
                # (possibly newly measured) FF frequency, overriding the restore.
                saved["ff"]._push()
            except Exception:
                logging.exception(
                    "torque_curve_calibrate: FF restore push failed")
        gcmd.respond_info(
            "Restored motion reshaping (unified jerk / FF / shaper)")

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
        rd_fn = rd_zeta = 0.0
        rd = self._axis_ac(np, cap.get("ringdown"))
        if rd is not None:
            rdt, rdsig = rd
            rd_rms = float(np.sqrt((rdsig * rdsig).mean()))
            rdfreqs, rdspec = self._spectrum(np, rdt, rdsig)
            rd_fn, rd_zeta, _ = self._ringdown_decay(np, rdt, rdsig, fn)

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
            "rd_fn": rd_fn, "rd_zeta": rd_zeta,
            "fn": fn or 0.0, "base_rms": base_rms,
            "_clean_resid": clean_resid,
        }

    def _ringdown_decay(self, np, rdt, rdsig, fn):
        """Damping ratio + free-decay frequency from the stationary ring-down.

        rdt, rdsig come straight from _axis_ac(cap["ringdown"]). Returns
        (f_free, zeta, r2); zeta is 0.0 when there is no clean exponential decay
        (fit rejected). Pure post-processing of samples already captured -- no
        change to the probe motion or capture.
        """
        if rdt is None or len(rdsig) < 32 or not fn:
            return 0.0, 0.0, 0.0
        n = len(rdsig)                        # analytic-signal envelope (numpy-only Hilbert)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        env = np.abs(np.fft.ifft(np.fft.fft(rdsig) * h))
        floor = (self._baseline or {}).get("rms", 0.0)
        m = env > max(env.max() * 0.1, floor)  # fit only above the noise floor
        if m.sum() < 16:
            return 0.0, 0.0, 0.0
        t = rdt[m] - rdt[m][0]
        y = np.log(env[m])
        A = np.vstack([t, np.ones_like(t)]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]   # slope = -zeta*2*pi*fn (log-decrement)
        zeta = float(-coef[0] / (2.0 * np.pi * fn))
        zc = np.count_nonzero(np.diff(np.signbit(rdsig[m])))
        dur = float(t[-1] - t[0])
        f_free = float(zc / (2.0 * dur)) if dur > 0.0 else 0.0
        yhat = A.dot(coef)
        ss = float(((y - yhat) ** 2).sum())
        tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss / tot if tot > 0.0 else 0.0
        if not (0.0 < zeta < 0.5) or r2 < 0.8:
            return f_free, 0.0, r2             # keep free-decay freq, flag zeta unreliable
        return f_free, zeta, r2

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

    def _p1_feed(self, np, bt, resid, gcmd):
        """Feed one in-sync probe's clean residual to the Pass-1 convergence
        tracker (built lazily). Reports the convergence transition once. The
        SNR gate inside the tracker is a second, tunable quality filter on top
        of the upstream clean-residual selection, so junk bursts that slip
        through don't move the estimate."""
        if bt is None or resid is None or len(bt) < 8:
            return
        freqs, spec = self._spectrum(np, bt, resid)
        if freqs is None:
            return
        if self._p1 is None:
            self._p1 = _ModeConvergence(
                self.p1_snr_min, self.p1_converge_tol, self.p1_converge_window,
                self.p1_max_bursts, self.resonance_min_freq,
                self.resonance_max_freq)
        was = self._p1.converged
        st = self._p1.add_burst(np, freqs, spec)
        if st["accepted"]:
            # f_peak is this burst's own peak (what the convergence gate
            # compares); f_n is the Welch-averaged peak (the reported estimate).
            # Showing both makes a non-latch diagnosable: if f_n is steady but
            # f_peak jitters wider than p1_converge_tol, the gate is starved.
            gcmd.respond_info(
                "    [P1] burst %d accepted (snr=%.1f) -> "
                "f_peak=%.2f f_n=%.2f Hz%s"
                % (st["n_accepted"], st["snr"], st["f_burst"], st["f_hat"],
                   " CONVERGED" if st["converged"] and not was else ""))

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
        # (topp-ra-v3: the old [jerk_limiting] auto_jerk closer was removed with
        # that module. The unified planner's auto_jerk is FF-driven and
        # re-derives itself on SET_MODEL_FF -- see toolhead.apply_auto_jerk.)

    def _learn_ff(self, gcmd):
        """Pass 3: learn the model-inverse FF from the residual measured across
        the skip test (input shaper on).

        The residual's dominant mode across speeds (clustered resid_fpeak) is
        the FF frequency; the ring-down fits near it give its damping ratio.
        Set live via SET_MODEL_FF and staged into SAVE_CONFIG (freq_<ax> /
        damping_ratio_<ax> in [model_inverse_ff]) so it persists.
        """
        res = self._structural_resonance()  # clusters resid_fpeak across speeds
        if res is None:
            gcmd.respond_info(
                "  [FF] no clean residual mode across speeds; FF not set")
            return
        freq, nspeeds, nprobes = res
        # zeta: median ring-down damping among clean fits near the residual mode
        # (rd_fn = col 12, rd_zeta = col 13).
        zetas = sorted(r[13] for r in self.vibration_rows
                       if r[13] > 0.0 and r[12] > 0.0
                       and abs(r[12] - freq) <= 8.0)
        if not zetas:
            zetas = sorted(r[13] for r in self.vibration_rows if r[13] > 0.0)
        if not zetas:
            gcmd.respond_info(
                "  [FF] residual mode %.2f Hz found but no ring-down zeta; FF "
                "not set (need clean ring-down data)" % freq)
            return
        zeta = zetas[len(zetas) // 2]
        ax = self.test_axis
        gcmd.respond_info(
            "  [FF] residual mode %.2f Hz (clustered across %d speeds / %d "
            "probes), zeta=%.4f (median of %d ring-downs)"
            % (freq, nspeeds, nprobes, zeta, len(zetas)))
        try:
            self.gcode.run_script_from_command(
                "SET_MODEL_FF FREQ_%s=%.3f DAMPING_RATIO_%s=%.4f ENABLE=1"
                % (ax.upper(), freq, ax.upper(), zeta))
            configfile = self.printer.lookup_object("configfile")
            configfile.set("model_inverse_ff", "freq_" + ax, "%.3f" % freq)
            configfile.set("model_inverse_ff", "damping_ratio_" + ax,
                           "%.4f" % zeta)
        except Exception:
            logging.exception("torque_curve_calibrate: FF learn/apply failed")
            gcmd.respond_info("  [FF] failed to set FF (see klippy.log)")
            return
        gcmd.respond_info(
            "  [FF] model-inverse FF set live (freq_%s=%.2f damping_ratio_%s"
            "=%.4f); run SAVE_CONFIG to persist." % (ax, freq, ax, zeta))

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

    def _full_step_dist(self):
        """Belt travel (mm) per motor FULL step on the test axis.

        get_step_dist() is mm per *micro*step; a full step is that times the
        configured microsteps. Returns None if either is unknown.
        """
        sd = self._step_dist()
        if not sd or not self._axis_microsteps:
            return None
        return sd * self._axis_microsteps

    def _max_meaningful_accel(self, speed):
        """Highest accel whose ramp still spans >= min_ramp_steps microsteps.

        The accel phase covers d = v^2 / 2a mm, i.e.
            n_ramp = (v^2 / 2a) * steps_per_mm   microsteps
        Solving n_ramp >= min_ramp_steps for a:
            a <= v^2 * steps_per_mm / (2 * min_ramp_steps)

        Beyond this the ramp is sub-microstep: the motor never physically sees
        the acceleration (so it can never skip -- the search would climb to
        accel_max forever), and the commanded profile degenerates into a
        velocity step that crashes host step generation.

        Returns None when unknown/disabled (-> fall back to accel_max alone).
        """
        sd = self._step_dist()
        if not sd or self.min_ramp_steps <= 0.0:
            return None
        steps_per_mm = 1.0 / sd
        return (speed * speed * steps_per_mm) / (2.0 * self.min_ramp_steps)

    def _ramp_steps(self, speed, accel):
        """Microsteps spanned by the accel ramp (for reporting)."""
        sd = self._step_dist()
        if not sd or accel <= 0.0:
            return 0.0
        return (speed * speed) / (2.0 * accel) / sd

    def _lost_step_tol_mm(self):
        """Lost-step threshold in mm.

        Derived from position_tolerance_steps (full steps) x the full-step
        distance so the gate tracks the physics -- a real skip is a whole
        electrical cycle (>= 4 full steps), re-home scatter is ~1 full step. An
        explicit position_tolerance (mm) overrides. Falls back to a fixed mm if
        the full-step size can't be resolved.
        """
        if self.position_tolerance is not None:
            return self.position_tolerance
        fs = self._full_step_dist()
        if fs:
            return self.position_tolerance_steps * fs
        return 0.4

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
        # Let the freely-ringing (FF-suspended) mode decay before re-homing, so
        # the endstop triggers at rest and we don't record a phantom ~1-step
        # "loss" from homing mid-oscillation. See settle_time in __init__.
        if self.settle_time > 0.0:
            self.toolhead.dwell(self.settle_time)
            self.toolhead.wait_moves()
        after = self._home_and_measure()
        if ref_home_pos is None or after is None:
            return False, 0.0
        step_dist = self._step_dist() or 0.0
        diff_mm = abs(after - ref_home_pos) * step_dist
        return diff_mm > self._lost_step_tol_mm(), diff_mm

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
        self._p1 = None
        self._p1_verdict = None

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

                # Per-speed ceiling: the accel above which the ramp spans fewer
                # than min_ramp_steps microsteps and stops being a ramp at all.
                # This -- not accel_max -- is what actually bounds the search;
                # accel_max is just a backstop. Grows as v^2, so it only binds
                # at low speed, exactly where the motor cannot skip anyway.
                a_degen = self._max_meaningful_accel(test_speed)
                accel_ceiling = self.accel_max
                ceiling_is_degen = False
                if a_degen is not None and a_degen < accel_ceiling:
                    accel_ceiling = a_degen
                    ceiling_is_degen = True

                if start_accel > accel_ceiling:
                    if ceiling_is_degen:
                        gcmd.respond_info(
                            "  Speed %.0f mm/s: not measurable -- even the "
                            "starting accel %.0f gives a ramp under %.1f "
                            "microsteps (ceiling %.0f); skipping"
                            % (test_speed, start_accel, self.min_ramp_steps,
                               accel_ceiling)
                        )
                    else:
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
                    # Clamp: never above the ceiling (degeneracy or accel_max),
                    # never at/above a known skip.
                    if test_accel > accel_ceiling:
                        test_accel = accel_ceiling
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
                                vm["rd_fn"], vm["rd_zeta"],
                            )
                            self.vibration_rows.append(row)
                            self._write_vibration_row(row)  # flushed to disk
                            # Buffer the clean residual for deferred, accel-gated
                            # accumulation once this speed's skip edge is known.
                            cr = vm.get("_clean_resid")
                            if cr is not None:
                                self._speed_resid_buffer.append((test_accel, cr))
                                # Pass-1 mode-ID convergence runs live off the
                                # same in-sync residuals (SNR-gated + Welch-
                                # averaged); an independent verdict from the raw
                                # per-probe modes for a bad mode ID to trip on.
                                import numpy as _np
                                self._p1_feed(_np, cr[0], cr[1], gcmd)
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
                        if test_accel >= accel_ceiling:
                            if ceiling_is_degen:
                                # Not a torque limit -- we ran out of ramp. The
                                # motor never saw the accel, so it could never
                                # skip. Say so rather than reporting a number
                                # that looks like a measurement.
                                gcmd.respond_info(
                                    "  No measurable torque limit at %.0f mm/s:"
                                    " ramp degenerates first (accel %.0f spans"
                                    " %.1f microsteps, min %.1f)"
                                    % (test_speed, test_accel,
                                       self._ramp_steps(test_speed, test_accel),
                                       self.min_ramp_steps)
                                )
                            else:
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

        # Pass 3: learn + set the model-inverse FF from the residual this run
        # captured. Runs after all sweep outputs are written and motion is
        # restored (via the finally block above).
        if self._pass_name == "ff":
            self._learn_ff(gcmd)

        # Abort gate: only when this run is being treated as a dedicated Pass-1
        # mode ID (P1_ABORT=1 / p1_abort). Raised last, after every output is
        # written and motion is restored, so a failed mode ID can't silently
        # feed Pass 2 but the operator keeps all captured data.
        if self.p1_abort and self.measure_vibration:
            status = (self._p1_verdict or ("no_signal", 0.0, ""))[0]
            if status != "converged":
                raise gcmd.error(
                    "Pass-1 mode ID failed to converge (%s); aborting before "
                    "the frequency is trusted. Loosen p1_converge_tol, raise "
                    "p1_max_bursts, or lower p1_snr_min, then re-run." % status)

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
                "resid_rms,resid_fpeak,resid_afn,rd_rms,rd_afn,rd_fn,rd_zeta\n"
                % (self.test_axis.upper(),
                   "OFF/raw" if self.vibration_shaper_off else "ON",
                   "%.1f" % fn if fn else "n/a")
            )
        self._vib_file.write(
            "%.2f,%.2f,%d,%.4f,%.4f,%.4f,%.2f,%.4f,%.2f,%.4f,%.4f,%.4f,%.2f,%.4f\n"
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
        # Measured resonance by speed-consistency (see _structural_resonance).
        res = self._structural_resonance()
        if res is not None:
            mid, nspeeds, n = res
            shaper = self._shaper_freq()
            gcmd.respond_info(
                "Measured %s resonance: %.1f Hz (recurs across %d speeds, "
                "%d probes)%s"
                % (self.test_axis.upper(), mid, nspeeds, n,
                   "" if not shaper
                   else " -- configured input shaper is %.1f Hz" % shaper)
            )
        # Pass-1 convergence verdict (Welch-averaged, SNR-gated mode ID). This
        # is the authoritative f_n for the FF/shaper; _structural_resonance
        # above is a cross-check by speed-consistency.
        self._report_p1_verdict(gcmd)
        # Fit shapers to the real-move spectrum and emit the input-shaper graph.
        self._emit_shaper_graph(gcmd)

    def _report_p1_verdict(self, gcmd):
        """Emit the Pass-1 convergence verdict; store it for the abort gate."""
        if self._p1 is None:
            self._p1_verdict = ("no_signal", 0.0, "no residual bursts captured")
            return
        self._p1_verdict = self._p1.verdict()
        status, fn, detail = self._p1_verdict
        if status == "converged":
            gcmd.respond_info(
                "Pass-1 mode ID CONVERGED: f_n = %.2f Hz (%s)"
                % (fn, detail))
        else:
            gcmd.respond_info(
                "Pass-1 mode ID did NOT converge [%s]: %s%s"
                % (status, detail,
                   "" if fn <= 0.0 else " -- best estimate %.1f Hz" % fn))

    def _structural_resonance(self, window=8.0):
        """Resonance frequency by speed-consistency, returned as
        (freq, n_speeds, n_probes) or None.

        A structural mode is a property of the axis, so it recurs at the same
        frequency across the whole speed range; artifacts are confined to a
        regime (near-skip stutter at high accel; degenerate sub-mm impulse moves
        at never-skip speeds where the search ran to accel_max). So the right
        estimator isn't the median or an amplitude-weighted mean -- both of
        which a loud but speed-localized artifact can hijack -- but the
        frequency where the most DISTINCT speeds pile up. Pick that center
        (tie-break by probe count), then report its cluster's median.
        Per-probe modes are row col 8 (resid_fpeak; 0 when no clean mode).
        """
        pr = [(r[0], r[8]) for r in self.vibration_rows if r[8] > 0.0]
        if not pr:
            return None
        best_center, best_score = None, (-1, -1)
        for _, fc in pr:
            inwin = [f for s, f in pr if abs(f - fc) <= window]
            nspeeds = len(set(s for s, f in pr if abs(f - fc) <= window))
            score = (nspeeds, len(inwin))
            if score > best_score:
                best_score, best_center = score, fc
        cluster = sorted(f for s, f in pr if abs(f - best_center) <= window)
        return (cluster[len(cluster) // 2], best_score[0], len(cluster))

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
            f.write("# Position tolerance: %.3f mm (%.2f full steps)\n"
                    % (self._lost_step_tol_mm(),
                       self._lost_step_tol_mm() / (self._full_step_dist() or 1.0)))
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

        # PASS selector: preset the flag combo for one step of the three-pass
        # flow. Each pass is the SAME sweep with different damping/capture, and
        # every preset flag is still individually overridable below.
        #   modeid  - raw ringing (shaper off): cluster the axis resonance and
        #             apply it as the input-shaper freq. That mode is what damps
        #             the axis for Pass 2 and parameterizes the FF for Pass 3.
        #   torque  - input shaper ON (Pass-1 freq -> mode damped): the accel
        #             search now reaches the REAL skip limits, not the resonance
        #             floor an undamped axis trips super early.
        #   ff      - (simple) enable the model-inverse FF at the identified mode
        #             and report the residual it cancels.
        # PASS omitted -> the plain configured sweep (back-compat).
        pass_name = gcmd.get("PASS", "").strip().lower()
        if pass_name not in ("", "modeid", "torque", "ff"):
            raise gcmd.error("PASS must be modeid, torque, or ff")
        self._pass_name = pass_name
        # Restore the per-run working copy from config BEFORE the presets touch
        # it: _resolve_speed_end() writes its resolved ceiling back into
        # self.speed_end, so without this an unset speed_end would stay pinned
        # to the first run's theoretical max, and Pass 1's cap would leak into
        # the torque/ff passes that follow it in CALIBRATE_Y_ALL.
        self.speed_end = self._cfg_speed_end
        if pass_name == "modeid":
            self.measure_vibration = True
            self.vibration_shaper_off = True
            self.apply_shaper = True
            # Only the resonance-limited band carries mode signal (see
            # modeid_speed_end). Never widen a tighter configured speed_end.
            if self.speed_end is None:
                self.speed_end = self.modeid_speed_end
            else:
                self.speed_end = min(self.speed_end, self.modeid_speed_end)
            gcmd.respond_info(
                "[PASS 1/3 MODE ID] Raw sweep to %.0f mm/s, input shaper OFF so "
                "the axis rings freely; cluster the resonance across speeds and "
                "set it as the input-shaper frequency. Reason: that mode is what "
                "damps the axis so Pass 2 can push accel to the real skip limit "
                "-- undamped, the ringing trips skips far too early. Capped at "
                "the back-EMF knee: above it the axis is torque-limited, the "
                "mode barely rings, and those low-SNR bursts drag f_n off the "
                "true mode." % self.speed_end)
        elif pass_name == "torque":
            self.measure_vibration = False
            self.vibration_shaper_off = False
            self.apply_shaper = False
            gcmd.respond_info(
                "[PASS 2/3 TORQUE CURVE] Input shaper ON at the Pass-1 mode so "
                "the axis is damped; run the accel search to the true skip "
                "limit. Reason: with the resonance suppressed the measured "
                "curve is the motor's real torque envelope, not a mode floor.")
        elif pass_name == "ff":
            self.measure_vibration = True
            self.vibration_shaper_off = False
            self.apply_shaper = False
            gcmd.respond_info(
                "[PASS 3/3 FF] Same skip test, input shaper ON, vibration "
                "capture ON. Reason: measure the RESIDUAL the shaper leaves "
                "behind across the full sweep, learn the FF mode(s) from it, "
                "and set the model-inverse FF (FREQ_Y + DAMPING_RATIO_Y).")

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
        # 0 disables the ramp-degeneracy ceiling (accel_max alone bounds the
        # search again) -- only sane if you know the axis skips first.
        self.min_ramp_steps = gcmd.get_float(
            "MIN_RAMP_STEPS", self.min_ramp_steps, minval=0.0
        )
        self.test_cycles = gcmd.get_int(
            "TEST_CYCLES", self.test_cycles, minval=1
        )
        self.position_tolerance_steps = gcmd.get_float(
            "POSITION_TOLERANCE_STEPS", self.position_tolerance_steps, above=0.0
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
        # Pass-1 convergence overrides.
        self.p1_snr_min = gcmd.get_float(
            "P1_SNR_MIN", self.p1_snr_min, minval=1.0)
        self.p1_converge_tol = gcmd.get_float(
            "P1_CONVERGE_TOL", self.p1_converge_tol, above=0.0)
        self.p1_converge_window = gcmd.get_int(
            "P1_CONVERGE_WINDOW", self.p1_converge_window, minval=2)
        self.p1_max_bursts = gcmd.get_int(
            "P1_MAX_BURSTS", self.p1_max_bursts, minval=1)
        self.p1_abort = bool(gcmd.get_int(
            "P1_ABORT", 1 if self.p1_abort else 0, minval=0, maxval=1))
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
