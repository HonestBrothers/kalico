# Dynamic acceleration limiting based on stepper motor torque-speed curves
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math
import os
import bisect


class TorqueCurve:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]

        # CSV file path - can be absolute or relative to config directory
        self.curve_file = config.get("curve_file", None)

        # Fallback limits when no curve is loaded
        self.fallback_max_accel = config.getfloat(
            "fallback_max_accel", None, above=0.0
        )

        # Interpolation method: 'linear' or 'step'
        self.interpolation = config.get("interpolation", "linear")
        if self.interpolation not in ("linear", "step"):
            raise config.error(
                "Invalid interpolation method '%s', must be 'linear' or 'step'"
                % self.interpolation
            )

        # Safety margin - multiply looked-up accel by this factor (0.0-1.0)
        self.safety_margin = config.getfloat(
            "safety_margin", 0.9, minval=0.1, maxval=1.0
        )

        # Enable/disable the dynamic limiting
        self.enabled = config.getboolean("enabled", True)

        # --- TOPP-RA mode -------------------------------------------------
        # When True, the curve is used as a velocity-dependent acceleration
        # constraint *inside* the lookahead (reachability analysis) and the
        # move is emitted as a sequence of constant-accel slices honoring
        # a_max(v), instead of a single conservative per-move accel cap.
        self.topp_ra = config.getboolean("enable_topp_ra", False)
        # Only reshape travel (non-extruding) moves. Extruding moves keep the
        # stock single-trapezoid so the extruder / pressure-advance sync (which
        # is tied to one accel per move) stays correct. Leave True until the
        # extruder sync is reworked for multi-segment moves.
        self.travel_only = config.getboolean("topp_ra_travel_only", True)
        # Velocity-space integration step for reach()/segment emission (mm/s).
        # Smaller = closer to the continuous optimum, more segments/cost.
        self.dv_slice = config.getfloat("topp_ra_resolution", 25.0, above=1.0)

        # Internal data structures for the curve
        # speeds and accels are parallel arrays, sorted by speed
        self.speeds = []
        self.accels = []
        self.curve_loaded = False

        # Load curve file if specified
        if self.curve_file is not None:
            self._load_curve_file(config)

        # Register with printer
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )

        # Register gcode commands
        gcode = self.printer.lookup_object("gcode")
        gcode.register_mux_command(
            "TORQUE_CURVE_LOAD", "NAME", self.name,
            self.cmd_TORQUE_CURVE_LOAD,
            desc=self.cmd_TORQUE_CURVE_LOAD_help,
        )
        gcode.register_mux_command(
            "TORQUE_CURVE_STATUS", "NAME", self.name,
            self.cmd_TORQUE_CURVE_STATUS,
            desc=self.cmd_TORQUE_CURVE_STATUS_help,
        )
        gcode.register_mux_command(
            "TORQUE_CURVE_SET", "NAME", self.name,
            self.cmd_TORQUE_CURVE_SET,
            desc=self.cmd_TORQUE_CURVE_SET_help,
        )

    def _resolve_file_path(self, filepath, config=None):
        """Resolve file path, handling relative paths from config directory."""
        if os.path.isabs(filepath):
            return filepath
        # Try relative to config file directory
        if config is not None:
            config_file = self.printer.get_start_args().get("config_file")
            if config_file:
                config_dir = os.path.dirname(os.path.abspath(config_file))
                resolved = os.path.join(config_dir, filepath)
                if os.path.exists(resolved):
                    return resolved
        # Try current working directory
        return os.path.abspath(filepath)

    def _load_curve_file(self, config=None):
        """Load torque curve from CSV file."""
        if self.curve_file is None:
            return False

        filepath = self._resolve_file_path(self.curve_file, config)

        if not os.path.exists(filepath):
            if config is not None:
                raise config.error(
                    "Torque curve file '%s' not found" % filepath
                )
            logging.warning("torque_curve: File '%s' not found", filepath)
            return False

        speeds = []
        accels = []

        try:
            with open(filepath, "r") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Skip header line if present
                    if line_num == 1 and not line[0].isdigit():
                        continue

                    parts = line.split(",")
                    if len(parts) < 2:
                        logging.warning(
                            "torque_curve: Skipping malformed line %d: %s",
                            line_num, line
                        )
                        continue

                    try:
                        speed = float(parts[0].strip())
                        accel = float(parts[1].strip())
                        if speed < 0 or accel <= 0:
                            logging.warning(
                                "torque_curve: Invalid values at line %d "
                                "(speed=%f, accel=%f)", line_num, speed, accel
                            )
                            continue
                        speeds.append(speed)
                        accels.append(accel)
                    except ValueError as e:
                        logging.warning(
                            "torque_curve: Parse error at line %d: %s",
                            line_num, str(e)
                        )
                        continue
        except IOError as e:
            if config is not None:
                raise config.error(
                    "Error reading torque curve file '%s': %s"
                    % (filepath, str(e))
                )
            logging.error("torque_curve: Error reading file: %s", str(e))
            return False

        if len(speeds) < 2:
            msg = "Torque curve file must contain at least 2 data points"
            if config is not None:
                raise config.error(msg)
            logging.error("torque_curve: %s", msg)
            return False

        # Sort by speed and store
        sorted_pairs = sorted(zip(speeds, accels), key=lambda x: x[0])
        self.speeds = [p[0] for p in sorted_pairs]
        self.accels = [p[1] for p in sorted_pairs]
        self.curve_loaded = True

        logging.info(
            "torque_curve: Loaded %d data points from '%s' "
            "(speed range: %.1f - %.1f mm/s)",
            len(self.speeds), filepath, self.speeds[0], self.speeds[-1]
        )
        return True

    def _handle_connect(self):
        """Called when printer connects."""
        self.toolhead = self.printer.lookup_object("toolhead")
        # Register ourselves with toolhead for move limiting
        if not hasattr(self.toolhead, "torque_curves"):
            self.toolhead.torque_curves = []
        self.toolhead.torque_curves.append(self)
        # Publish ourselves as the single active TOPP-RA constraint provider
        # that the lookahead (Move.calc_junction / LookAheadQueue.flush /
        # ToolHead._process_moves) consults via getattr(toolhead,"topp_ra",None).
        if self.topp_ra:
            if getattr(self.toolhead, "topp_ra", None) is not None:
                raise self.printer.config_error(
                    "Only one [torque_curve] may set enable_topp_ra"
                )
            self.toolhead.topp_ra = self

    def get_max_accel_for_speed(self, speed):
        """
        Look up maximum acceleration for a given speed.

        Args:
            speed: Current or target speed in mm/s

        Returns:
            Maximum acceleration in mm/s^2, or None if no limit applies
        """
        if not self.enabled or not self.curve_loaded:
            return self.fallback_max_accel

        speed = abs(speed)

        # Handle speeds outside the curve range
        if speed <= self.speeds[0]:
            max_accel = self.accels[0]
        elif speed >= self.speeds[-1]:
            max_accel = self.accels[-1]
        else:
            # Find interpolation position
            idx = bisect.bisect_right(self.speeds, speed) - 1

            if self.interpolation == "step":
                # Step interpolation: use the lower speed's accel
                max_accel = self.accels[idx]
            else:
                # Linear interpolation
                speed_low = self.speeds[idx]
                speed_high = self.speeds[idx + 1]
                accel_low = self.accels[idx]
                accel_high = self.accels[idx + 1]

                # Linear interpolation factor
                t = (speed - speed_low) / (speed_high - speed_low)
                max_accel = accel_low + t * (accel_high - accel_low)

        # Apply safety margin
        return max_accel * self.safety_margin

    def limit_move(self, move):
        """
        Apply torque curve limits to a move.
        Called from toolhead during move planning.

        Args:
            move: Move object being planned
        """
        if not self.enabled or not self.curve_loaded:
            return

        # Use the move's current max cruise velocity to determine accel limit
        # This is the maximum speed the move might reach
        import math
        cruise_speed = math.sqrt(move.max_cruise_v2)

        max_accel = self.get_max_accel_for_speed(cruise_speed)
        if max_accel is not None and max_accel < move.accel:
            move.limit_speed(cruise_speed, max_accel)

    # ------------------------------------------------------------------
    # TOPP-RA: velocity-dependent reachability in u = v^2 space.
    #
    # du/ds = 2*a, so the accel constraint a <= a_max(v) becomes a bound on
    # how fast u may grow/shrink along the path. All helpers use the
    # *upper-edge* (smaller, conservative) accel within each velocity slice,
    # so planned velocities are always achievable by the emitter and braking
    # never exceeds the torque limit. a_max(v) is assumed symmetric for
    # accel/decel (stepper pull-out torque is ~direction-independent).
    # ------------------------------------------------------------------
    def active_for(self, move):
        # True if TOPP-RA should reshape this move.
        if not (self.topp_ra and self.enabled and self.curve_loaded):
            return False
        if self.travel_only and move.axes_d[3]:
            return False
        return True

    def _accel_at(self, v):
        a = self.get_max_accel_for_speed(v)
        if a is None or a <= 0.0:
            return None
        return a

    def reach(self, u0, dist):
        # Max u = v^2 reachable from u0 over path-distance `dist`, riding
        # |du/ds| = 2*a_max(sqrt(u)). Direction-symmetric, so this serves both
        # forward-accel (calc_junction) and backward-decel (flush).
        if not self.curve_loaded or dist <= 0.0:
            return u0
        v_ceil = self.speeds[-1]
        v = math.sqrt(max(u0, 0.0))
        s = 0.0
        while s < dist and v < v_ceil:
            v_next = min(v + self.dv_slice, v_ceil)
            a = self._accel_at(v_next)
            if a is None:
                break
            ds = (v_next * v_next - v * v) / (2.0 * a)
            if s + ds >= dist:
                return v * v + 2.0 * a * (dist - s)
            v, s = v_next, s + ds
        return v * v

    def _dist_up(self, v_lo, v_hi):
        # Path distance to accelerate from v_lo to v_hi (>= v_lo) under the
        # curve, using upper-edge (conservative) accel per slice.
        if v_hi <= v_lo:
            return 0.0
        v = v_lo
        d = 0.0
        while v < v_hi:
            v_next = min(v + self.dv_slice, v_hi)
            a = self._accel_at(v_next)
            if a is None:
                return d
            d += (v_next * v_next - v * v) / (2.0 * a)
            v = v_next
        return d

    def _peak_velocity(self, start_v, end_v, dist):
        # Highest velocity a move of length `dist` can peak at (accelerate up
        # to v_p then back down to end_v). Bisection; None if too short to even
        # connect start_v -> end_v under the curve.
        lo = max(start_v, end_v)
        hi = self.speeds[-1]
        if self._dist_up(start_v, lo) + self._dist_up(end_v, lo) > dist:
            return None
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            need = self._dist_up(start_v, mid) + self._dist_up(end_v, mid)
            if need > dist:
                hi = mid
            else:
                lo = mid
        return lo

    def plan_move(self, move):
        # Build the TOPP-RA segment list for `move`, honoring a_max(v). Each
        # segment is (accel_t, cruise_t, decel_t, start_v, cruise_v, accel,
        # distance) -- ready to feed both the toolhead and (scaled) extruder
        # trapq. Returns None to fall back to stock single-trapezoid emission.
        if not self.active_for(move):
            return None
        start_v, cruise_v, end_v = move.start_v, move.cruise_v, move.end_v
        move_d = move.move_d
        d_acc = self._dist_up(start_v, cruise_v)
        d_dec = self._dist_up(end_v, cruise_v)
        cruise_d = move_d - d_acc - d_dec
        if cruise_d < -1e-9:
            # Too short to reach cruise_v: solve the triangle peak.
            v_p = self._peak_velocity(start_v, end_v, move_d)
            if v_p is None:
                return None
            cruise_v = v_p
            d_acc = self._dist_up(start_v, cruise_v)
            d_dec = self._dist_up(end_v, cruise_v)
            cruise_d = max(0.0, move_d - d_acc - d_dec)
        segs = []
        # Accel slices: start_v -> cruise_v (accel-only)
        v = start_v
        while v < cruise_v - 1e-9:
            v_next = min(v + self.dv_slice, cruise_v)
            a = self._accel_at(v_next)
            if a is None:
                break
            dt = (v_next - v) / a
            dist = (v_next * v_next - v * v) / (2.0 * a)
            segs.append((dt, 0.0, 0.0, v, v_next, a, dist))
            v = v_next
        # Cruise (single constant-velocity segment)
        if cruise_d > 1e-9 and cruise_v > 1e-9:
            ct = cruise_d / cruise_v
            segs.append((0.0, ct, 0.0, cruise_v, cruise_v, 0.0, cruise_d))
        # Decel slices: cruise_v -> end_v (decel-only, high->low)
        ladder = []
        v = end_v
        while v < cruise_v - 1e-9:
            v_next = min(v + self.dv_slice, cruise_v)
            ladder.append((v, v_next))
            v = v_next
        for v_lo, v_hi in reversed(ladder):
            a = self._accel_at(v_hi)
            if a is None:
                break
            dt = (v_hi - v_lo) / a
            dist = (v_hi * v_hi - v_lo * v_lo) / (2.0 * a)
            segs.append((0.0, 0.0, dt, v_hi, v_hi, a, dist))
        return segs if segs else None

    def set_curve_data(self, speeds, accels):
        """
        Programmatically set curve data (used by calibration).

        Args:
            speeds: List of speeds in mm/s
            accels: List of max accelerations in mm/s^2
        """
        if len(speeds) != len(accels):
            raise ValueError("speeds and accels must have same length")
        if len(speeds) < 2:
            raise ValueError("Need at least 2 data points")

        sorted_pairs = sorted(zip(speeds, accels), key=lambda x: x[0])
        self.speeds = [p[0] for p in sorted_pairs]
        self.accels = [p[1] for p in sorted_pairs]
        self.curve_loaded = True

    def save_curve_to_file(self, filepath):
        """Save current curve data to a CSV file."""
        if not self.curve_loaded:
            raise ValueError("No curve data to save")

        filepath = self._resolve_file_path(filepath)

        with open(filepath, "w") as f:
            f.write("# Torque curve: speed (mm/s), max_accel (mm/s^2)\n")
            f.write("# Generated by torque_curve calibration\n")
            f.write("speed,max_accel\n")
            for speed, accel in zip(self.speeds, self.accels):
                f.write("%.2f,%.2f\n" % (speed, accel))

        logging.info("torque_curve: Saved curve to '%s'", filepath)
        return filepath

    def get_status(self, eventtime):
        return {
            "enabled": self.enabled,
            "curve_loaded": self.curve_loaded,
            "curve_file": self.curve_file,
            "interpolation": self.interpolation,
            "safety_margin": self.safety_margin,
            "num_points": len(self.speeds) if self.curve_loaded else 0,
            "speed_min": self.speeds[0] if self.curve_loaded else 0,
            "speed_max": self.speeds[-1] if self.curve_loaded else 0,
        }

    # Gcode commands
    cmd_TORQUE_CURVE_LOAD_help = "Load a torque curve from a CSV file"
    def cmd_TORQUE_CURVE_LOAD(self, gcmd):
        filepath = gcmd.get("FILE", self.curve_file)
        if filepath is None:
            raise gcmd.error("No FILE specified and no default curve_file set")

        self.curve_file = filepath
        if self._load_curve_file():
            gcmd.respond_info(
                "Loaded torque curve with %d points (%.1f - %.1f mm/s)"
                % (len(self.speeds), self.speeds[0], self.speeds[-1])
            )
        else:
            raise gcmd.error("Failed to load torque curve file")

    cmd_TORQUE_CURVE_STATUS_help = "Show torque curve status and sample values"
    def cmd_TORQUE_CURVE_STATUS(self, gcmd):
        msg = ["Torque curve '%s':" % self.name]
        msg.append("  Enabled: %s" % self.enabled)
        msg.append("  Curve loaded: %s" % self.curve_loaded)
        msg.append("  Interpolation: %s" % self.interpolation)
        msg.append("  Safety margin: %.1f%%" % (self.safety_margin * 100))

        if self.curve_loaded:
            msg.append("  Data points: %d" % len(self.speeds))
            msg.append("  Speed range: %.1f - %.1f mm/s"
                      % (self.speeds[0], self.speeds[-1]))
            msg.append("  Sample lookups:")
            for speed in [50, 100, 150, 200, 250, 300]:
                if speed <= self.speeds[-1]:
                    accel = self.get_max_accel_for_speed(speed)
                    msg.append("    %d mm/s -> %.0f mm/s^2" % (speed, accel))

        gcmd.respond_info("\n".join(msg))

    cmd_TORQUE_CURVE_SET_help = "Enable/disable torque curve or set parameters"
    def cmd_TORQUE_CURVE_SET(self, gcmd):
        enable = gcmd.get_int("ENABLE", None)
        if enable is not None:
            self.enabled = bool(enable)
            gcmd.respond_info(
                "Torque curve %s" % ("enabled" if self.enabled else "disabled")
            )

        margin = gcmd.get_float("SAFETY_MARGIN", None, minval=0.1, maxval=1.0)
        if margin is not None:
            self.safety_margin = margin
            gcmd.respond_info("Safety margin set to %.1f%%" % (margin * 100))

        interp = gcmd.get("INTERPOLATION", None)
        if interp is not None:
            if interp.lower() not in ("linear", "step"):
                raise gcmd.error("INTERPOLATION must be 'linear' or 'step'")
            self.interpolation = interp.lower()
            gcmd.respond_info("Interpolation set to '%s'" % self.interpolation)


def load_config_prefix(config):
    return TorqueCurve(config)
