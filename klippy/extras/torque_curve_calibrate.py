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

    def _run_calibration(self, gcmd):
        """Main calibration routine."""
        self.calibration_running = True
        self.calibration_results = []

        gcmd.respond_info("Starting torque curve calibration on %s axis"
                         % self.test_axis.upper())
        gcmd.respond_info(
            "To STOP mid-run press EMERGENCY STOP (M112) -- it is the only "
            "command Klipper runs while a test is in progress."
        )

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

            # Initial full home so the test never depends on the user having
            # homed first, and so the safe-Z lift (and safe_z_home, which needs
            # X/Y homed to reach its probe point) have everything they need.
            gcmd.respond_info("Homing all axes...")
            self.gcode.run_script_from_command("G28")

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

                # Linear ramp-up search (gentle). Step the acceleration up from
                # start_accel by accel_step until the first skip. This never
                # commands more than one accel_step above a known-good value, so
                # a skip overshoots the motor's real limit by at most one step.
                # A binary search converges faster but its first probe jumps to
                # the midpoint of [start_accel, accel_max] -- e.g. ~50000 when
                # accel_max is 100000 -- which can be far above the real limit
                # and crash the axis hard. Trade time for safety here.
                last_good_accel = 0
                test_accel = start_accel
                while test_accel <= self.accel_max:
                    if not self.calibration_running:
                        break

                    ref_home = self._home_and_measure()
                    self._perform_test_move(
                        start_pos, end_pos, test_speed, test_accel
                    )
                    lost, diff = self._check_for_lost_steps(ref_home)

                    if lost:
                        gcmd.respond_info(
                            "  Accel %.0f: FAILED (lost %.3f mm) -> limit found"
                            % (test_accel, diff)
                        )
                        break

                    gcmd.respond_info("  Accel %.0f: OK" % test_accel)
                    last_good_accel = test_accel
                    test_accel += self.accel_step

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
        self.output_file = gcmd.get("OUTPUT_FILE", self.output_file)

        self._run_calibration(gcmd)

    cmd_TORQUE_CURVE_CALIBRATE_ABORT_help = (
        "Emergency-stop (M112). NOTE: cannot interrupt a test already running"
    )
    def cmd_TORQUE_CURVE_CALIBRATE_ABORT(self, gcmd):
        # Fire an emergency stop. Caveat: g-code is serialized, so if a
        # calibration is mid-run this command is queued behind it and only
        # runs once the test ends -- it cannot interrupt. Only the literal
        # M112 string is handled out-of-order, so to halt a *running* test
        # send M112 / press Emergency Stop directly.
        self.calibration_running = False
        gcmd.respond_info("Emergency stop (M112) - calibration abort")
        self.gcode.run_script_from_command("M112")


def load_config_prefix(config):
    return TorqueCurveCalibrate(config)
