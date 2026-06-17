# Dynamic pressure advance from a measured pa-vs-flow-speed curve
#
# Companion to [torque_curve]: just as the torque curve makes max_accel a
# function of velocity, this makes pressure_advance a function of extruder
# flow speed, pa(v). It is delivered per TOPP-RA velocity slice through the
# extruder's per-move trapq pressure-advance slot (see Extruder.move_segment),
# so no C changes are required. Without TOPP-RA segmentation it is inert.
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import os
import bisect


class PressureAdvanceCurve:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        # Extruder this curve attaches to (its extruder_stepper).
        self.extruder_name = config.get("extruder", "extruder")
        # CSV of "flow_speed_mm_s,pressure_advance" rows.
        self.curve_file = config.get("curve_file", None)
        self.interpolation = config.get("interpolation", "linear")
        if self.interpolation not in ("linear", "step"):
            raise config.error(
                "pressure_advance_curve: interpolation must be"
                " 'linear' or 'step'"
            )
        self.enabled = config.getboolean("enabled", True)
        # speeds (extruder flow, mm/s of filament) and pas are parallel,
        # sorted by speed.
        self.speeds = []
        self.pas = []
        self.curve_loaded = False
        if self.curve_file is not None:
            self._load_curve_file()
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )
        gcode = self.printer.lookup_object("gcode")
        gcode.register_mux_command(
            "PA_CURVE_SET", "NAME", self.name, self.cmd_PA_CURVE_SET,
            desc="Set pressure-advance curve points (SPEEDS=.. PAS=..)",
        )
        gcode.register_mux_command(
            "PA_CURVE_STATUS", "NAME", self.name, self.cmd_PA_CURVE_STATUS,
            desc="Report pressure-advance curve status",
        )

    def _resolve_path(self, filepath):
        if os.path.isabs(filepath):
            return filepath
        cfg = self.printer.get_start_args().get("config_file", "")
        return os.path.join(os.path.dirname(cfg), filepath)

    def _load_curve_file(self):
        path = self._resolve_path(self.curve_file)
        speeds, pas = [], []
        try:
            with open(path, "r") as f:
                for n, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 2:
                        logging.warning(
                            "pressure_advance_curve: skip malformed line %d", n
                        )
                        continue
                    try:
                        speeds.append(float(parts[0]))
                        pas.append(float(parts[1]))
                    except ValueError:
                        logging.warning(
                            "pressure_advance_curve: bad values line %d", n
                        )
        except OSError as e:
            logging.error("pressure_advance_curve: %s", e)
            return
        if len(speeds) >= 2:
            self.set_curve_data(speeds, pas)

    def set_curve_data(self, speeds, pas):
        pairs = sorted(zip(speeds, pas), key=lambda p: p[0])
        self.speeds = [p[0] for p in pairs]
        self.pas = [p[1] for p in pairs]
        self.curve_loaded = len(self.speeds) >= 2

    def _handle_connect(self):
        extruder = self.printer.lookup_object(self.extruder_name, None)
        if extruder is None:
            raise self.printer.config_error(
                "pressure_advance_curve: extruder '%s' not found"
                % self.extruder_name
            )
        es = getattr(extruder, "extruder_stepper", None)
        if es is None:
            raise self.printer.config_error(
                "pressure_advance_curve: '%s' has no extruder_stepper"
                % self.extruder_name
            )
        # Extruder.move_segment consults es.pa_curve per slice.
        es.pa_curve = self

    def active(self):
        return self.enabled and self.curve_loaded

    def get_pa_for_speed(self, speed):
        # speed: extruder flow speed (mm/s of filament) for the slice.
        if not self.curve_loaded:
            return 0.0
        speed = abs(speed)
        if speed <= self.speeds[0]:
            return self.pas[0]
        if speed >= self.speeds[-1]:
            return self.pas[-1]
        idx = bisect.bisect_right(self.speeds, speed) - 1
        if self.interpolation == "step":
            return self.pas[idx]
        s0, s1 = self.speeds[idx], self.speeds[idx + 1]
        p0, p1 = self.pas[idx], self.pas[idx + 1]
        t = (speed - s0) / (s1 - s0)
        return p0 + t * (p1 - p0)

    cmd_PA_CURVE_SET_help = "Set pressure-advance curve points"

    def cmd_PA_CURVE_SET(self, gcmd):
        speeds = [float(x) for x in gcmd.get("SPEEDS").split(",")]
        pas = [float(x) for x in gcmd.get("PAS").split(",")]
        if len(speeds) != len(pas) or len(speeds) < 2:
            raise gcmd.error("SPEEDS and PAS must match and have >= 2 points")
        self.set_curve_data(speeds, pas)
        gcmd.respond_info(
            "pressure_advance_curve '%s': %d points loaded"
            % (self.name, len(self.speeds))
        )

    cmd_PA_CURVE_STATUS_help = "Report pressure-advance curve status"

    def cmd_PA_CURVE_STATUS(self, gcmd):
        gcmd.respond_info(
            "pressure_advance_curve '%s': enabled=%s loaded=%s points=%d"
            % (self.name, self.enabled, self.curve_loaded, len(self.speeds))
        )


def load_config(config):
    return PressureAdvanceCurve(config)


def load_config_prefix(config):
    return PressureAdvanceCurve(config)
