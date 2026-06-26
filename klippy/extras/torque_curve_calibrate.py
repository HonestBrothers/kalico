# Automated calibration for stepper motor torque-speed curves
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math
import os


class TorqueCurveCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]

        # Test configuration
        self.test_axis = config.get("axis", "x").lower()
        if self.test_axis not in ("x", "y"):
            raise config.error("axis must be 'x' or 'y'")

        # Speed range to test (mm/s)
        self.speed_start = config.getfloat("speed_start", 50.0, above=0.0)
        self.speed_end = config.getfloat("speed_end", 300.0, above=0.0)
        self.speed_step = config.getfloat("speed_step", 25.0, above=0.0)

        # Acceleration range to test (mm/s^2)
        self.accel_start = config.getfloat("accel_start", 1000.0, above=0.0)
        self.accel_max = config.getfloat("accel_max", 50000.0, above=0.0)
        self.accel_step = config.getfloat("accel_step", 1000.0, above=0.0)

        # Move distance for testing (mm)
        self.test_move_distance = config.getfloat(
            "test_move_distance", 50.0, above=10.0
        )

        # Position tolerance for detecting lost steps (in mm)
        self.position_tolerance = config.getfloat(
            "position_tolerance", 0.1, above=0.0
        )

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
        gcode.register_mux_command(
            "TORQUE_CURVE_CALIBRATE_ABORT", "NAME", self.name,
            self.cmd_TORQUE_CURVE_CALIBRATE_ABORT,
            desc=self.cmd_TORQUE_CURVE_CALIBRATE_ABORT_help,
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

    def _perform_test_move(self, start_pos, end_pos, speed, accel):
        """
        Perform a test move at given speed and acceleration.
        Returns True if move completed without issues.
        """
        # Set acceleration
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.1f VELOCITY=%.1f" % (accel, speed)
        )

        # Get initial stepper position
        initial_stepper_pos = self._get_stepper_position()

        # Move to start position at safe speed/accel
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=1000 VELOCITY=100"
        )
        axis_idx = self._get_axis_index()
        pos = list(self.toolhead.get_position())
        pos[axis_idx] = start_pos
        self.toolhead.move(pos, 100)
        self.toolhead.wait_moves()

        # Set test acceleration
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.1f VELOCITY=%.1f" % (accel, speed)
        )

        # Record position before test
        pre_test_stepper_pos = self._get_stepper_position()

        # Perform test move
        pos[axis_idx] = end_pos
        self.toolhead.move(pos, speed)
        self.toolhead.wait_moves()

        # Move back
        pos[axis_idx] = start_pos
        self.toolhead.move(pos, speed)
        self.toolhead.wait_moves()

        # Record position after test
        post_test_stepper_pos = self._get_stepper_position()

        return pre_test_stepper_pos, post_test_stepper_pos

    def _check_for_lost_steps(self):
        """
        Check for lost steps by re-homing and comparing position.
        Returns (lost_steps_detected, step_difference)
        """
        # Record expected position
        expected_stepper_pos = self._get_stepper_position()

        # Re-home
        self._home_axis()

        # Get new position
        actual_stepper_pos = self._get_stepper_position()

        # Calculate difference
        if expected_stepper_pos is None or actual_stepper_pos is None:
            return False, 0

        diff = abs(actual_stepper_pos - expected_stepper_pos)

        # Get step distance to convert to mm
        for rail in self.kin.rails:
            for stepper in rail.get_steppers():
                if stepper.is_active_axis(self.test_axis):
                    step_dist = stepper.get_step_dist()
                    diff_mm = diff * step_dist
                    lost = diff_mm > self.position_tolerance
                    return lost, diff_mm

        return False, 0

    def _run_calibration(self, gcmd):
        """Main calibration routine."""
        self.calibration_running = True
        self.calibration_results = []

        gcmd.respond_info("Starting torque curve calibration on %s axis"
                         % self.test_axis.upper())

        # Save original settings
        systime = self.printer.get_reactor().monotonic()
        toolhead_info = self.toolhead.get_status(systime)
        orig_max_accel = toolhead_info["max_accel"]
        orig_max_velocity = toolhead_info["max_velocity"]
        saved_kin_limits = self._save_kinematic_limits()

        try:
            # Widen per-axis kinematic caps so the commanded accel is the accel
            # the motor actually sees (see _save_kinematic_limits).
            self._apply_kinematic_limits(gcmd)

            # Initial home
            gcmd.respond_info("Homing %s axis..." % self.test_axis.upper())
            self._home_axis()

            # Lift the gantry clear of the bed before any high-speed sweeping.
            self._raise_z(gcmd)

            # Calculate test positions
            start_pos, end_pos = self._calculate_test_positions()
            test_distance = abs(end_pos - start_pos)

            gcmd.respond_info(
                "Test positions: %.1f to %.1f mm (%.1f mm travel)"
                % (start_pos, end_pos, test_distance)
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

                # Check if we can reach this speed in the available distance
                min_accel_for_speed = (test_speed ** 2) / test_distance
                start_accel = max(self.accel_start, min_accel_for_speed * 1.5)

                if start_accel > self.accel_max:
                    gcmd.respond_info(
                        "  Speed %.0f mm/s requires accel > %.0f, skipping"
                        % (test_speed, self.accel_max)
                    )
                    continue

                # Binary search for maximum acceleration
                accel_low = start_accel
                accel_high = self.accel_max
                last_good_accel = accel_low
                test_count = 0

                # First, verify the starting acceleration works
                self._home_axis()
                self._perform_test_move(start_pos, end_pos, test_speed, accel_low)
                lost, diff = self._check_for_lost_steps()

                if lost:
                    gcmd.respond_info(
                        "  FAILED at starting accel %.0f (lost %.3f mm)"
                        % (accel_low, diff)
                    )
                    # Can't even do the minimum, record as failed
                    self.calibration_results.append((test_speed, 0))
                    continue

                last_good_accel = accel_low
                gcmd.respond_info("  Accel %.0f: OK" % accel_low)

                # Binary search for maximum working acceleration
                while accel_high - accel_low > self.accel_step:
                    if not self.calibration_running:
                        break

                    test_accel = (accel_low + accel_high) / 2
                    test_count += 1

                    self._home_axis()
                    self._perform_test_move(
                        start_pos, end_pos, test_speed, test_accel
                    )
                    lost, diff = self._check_for_lost_steps()

                    if lost:
                        gcmd.respond_info(
                            "  Accel %.0f: FAILED (lost %.3f mm)"
                            % (test_accel, diff)
                        )
                        accel_high = test_accel
                    else:
                        gcmd.respond_info("  Accel %.0f: OK" % test_accel)
                        accel_low = test_accel
                        last_good_accel = test_accel

                # Record result
                self.calibration_results.append((test_speed, last_good_accel))
                gcmd.respond_info(
                    "  Result: %.0f mm/s -> max accel %.0f mm/s^2"
                    % (test_speed, last_good_accel)
                )

        finally:
            # Restore original settings
            self.gcode.run_script_from_command(
                "SET_VELOCITY_LIMIT ACCEL=%.1f VELOCITY=%.1f"
                % (orig_max_accel, orig_max_velocity)
            )
            self._restore_kinematic_limits(saved_kin_limits, gcmd)
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
        # Resolve output path
        config_file = self.printer.get_start_args().get("config_file")
        if config_file:
            config_dir = os.path.dirname(os.path.abspath(config_file))
            output_path = os.path.join(config_dir, self.output_file)
        else:
            output_path = os.path.abspath(self.output_file)

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
        self.output_file = gcmd.get("OUTPUT_FILE", self.output_file)

        self._run_calibration(gcmd)

    cmd_TORQUE_CURVE_CALIBRATE_ABORT_help = "Abort running calibration"
    def cmd_TORQUE_CURVE_CALIBRATE_ABORT(self, gcmd):
        if not self.calibration_running:
            gcmd.respond_info("No calibration in progress")
            return
        self.calibration_running = False
        gcmd.respond_info("Calibration abort requested")


def load_config_prefix(config):
    return TorqueCurveCalibrate(config)
