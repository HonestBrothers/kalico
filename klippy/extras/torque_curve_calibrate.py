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

        # Internal state
        self.calibration_running = False
        self.calibration_results = []

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
        gcmd.respond_info("Restored motion reshaping (TOPP-RA / jerk)")

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

    def _perform_test_move(self, center, max_half, speed, accel, dwell_time):
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
        for _ in range(self.test_cycles):
            pos[axis_idx] = end_pos
            self.toolhead.move(pos, speed)
            pos[axis_idx] = start_pos
            self.toolhead.move(pos, speed)
        self.toolhead.wait_moves()

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
                while self.calibration_running:
                    # Clamp: never above accel_max, never at/above a known skip.
                    if test_accel > self.accel_max:
                        test_accel = self.accel_max
                    if skip_accel is not None and test_accel >= skip_accel:
                        test_accel = 0.5 * (last_good_accel + skip_accel)

                    ref_home = self._home_and_measure()
                    self._perform_test_move(
                        center, max_half, test_speed, test_accel, dwell_time
                    )
                    lost, diff = self._check_for_lost_steps(ref_home)

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
            self.calibration_running = False

        # Filter out failed results (accel = 0)
        valid_results = [r for r in self.calibration_results if r[1] > 0]

        if len(valid_results) >= 2:
            # Save results
            self._save_results(gcmd, valid_results)
        else:
            gcmd.respond_info(
                "Calibration incomplete - not enough valid data points"
            )

    def _save_results(self, gcmd, results):
        """Save calibration results to CSV file."""
        # Resolve output path: <config_dir>/torque_curve/<stem>_<axis>.csv
        config_file = self.printer.get_start_args().get("config_file")
        if config_file:
            base_dir = os.path.dirname(os.path.abspath(config_file))
        else:
            base_dir = os.getcwd()
        out_dir = os.path.join(base_dir, "torque_curve")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.output_file))[0]
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

        self._run_calibration(gcmd)


def load_config_prefix(config):
    return TorqueCurveCalibrate(config)
