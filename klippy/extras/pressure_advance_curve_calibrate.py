# Semi-automated calibration for pressure-advance vs flow-speed curves
#
# Unlike acceleration (where lost steps give an objective pass/fail that can be
# binary-searched -- see torque_curve_calibrate.py), pressure advance is judged
# visually. So this does NOT pretend to auto-measure it. Instead it automates
# the procedure and bookkeeping around the standard PA-tower workflow:
#
#   1) For each flow speed you care about, slice/print a PA test at that speed
#      and run a pressure_advance tuning tower. PA_CURVE_TUNE emits the right
#      TUNING_TOWER command for you.
#   2) Read off the best PA for that print and record it:
#         PA_CURVE_RECORD NAME=<curve> FLOW=<mm/s_of_filament> PA=<value>
#   3) PA_CURVE_CALIBRATE_SAVE assembles pa(flow), writes the CSV consumed by
#      [pressure_advance_curve], and loads it into the live curve object.
#
# FLOW is extruder flow speed (mm/s of filament = toolhead_speed * extrusion
# ratio), matching how [pressure_advance_curve] indexes the curve.
#
# Second order (tau1/tau2) is intentionally NOT calibrated here: measuring the
# two melt/filament time constants needs a hotend pressure sensor or inference
# from extrusion-width ringing after acceleration steps -- instrumentation this
# routine doesn't have. See the module docstring note at the bottom.
#
# Copyright (C) 2024
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import os


class PressureAdvanceCurveCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        # Curve object to populate (a [pressure_advance_curve <name>] section).
        self.target_curve = config.get("target_curve", None)
        self.output_file = config.get("output_file", "pa_curve.csv")
        # Tuning-tower defaults (override per command).
        self.tower_start = config.getfloat("tower_start", 0.0, minval=0.0)
        self.tower_factor = config.getfloat("tower_factor", 0.005, above=0.0)
        self.tower_band = config.getfloat("tower_band", 0.0, minval=0.0)
        # Recorded (flow_speed, pa) points.
        self.points = []
        gcode = self.printer.lookup_object("gcode")
        gcode.register_mux_command(
            "PA_CURVE_TUNE", "NAME", self.name, self.cmd_PA_CURVE_TUNE,
            desc=self.cmd_PA_CURVE_TUNE_help,
        )
        gcode.register_mux_command(
            "PA_CURVE_RECORD", "NAME", self.name, self.cmd_PA_CURVE_RECORD,
            desc=self.cmd_PA_CURVE_RECORD_help,
        )
        gcode.register_mux_command(
            "PA_CURVE_CALIBRATE_LIST", "NAME", self.name,
            self.cmd_PA_CURVE_CALIBRATE_LIST,
            desc="List recorded pa(flow) points",
        )
        gcode.register_mux_command(
            "PA_CURVE_CALIBRATE_CLEAR", "NAME", self.name,
            self.cmd_PA_CURVE_CALIBRATE_CLEAR,
            desc="Clear recorded pa(flow) points",
        )
        gcode.register_mux_command(
            "PA_CURVE_CALIBRATE_SAVE", "NAME", self.name,
            self.cmd_PA_CURVE_CALIBRATE_SAVE,
            desc=self.cmd_PA_CURVE_CALIBRATE_SAVE_help,
        )

    cmd_PA_CURVE_TUNE_help = (
        "Emit a pressure_advance TUNING_TOWER for the current print"
    )

    def cmd_PA_CURVE_TUNE(self, gcmd):
        # Convenience wrapper: start a pressure_advance tuning tower so the
        # current (already-sliced-at-some-flow) print sweeps PA over Z.
        start = gcmd.get_float("START", self.tower_start, minval=0.0)
        factor = gcmd.get_float("FACTOR", self.tower_factor, above=0.0)
        band = gcmd.get_float("BAND", self.tower_band, minval=0.0)
        flow = gcmd.get_float("FLOW", None, above=0.0)
        cmd = (
            "TUNING_TOWER COMMAND=SET_PRESSURE_ADVANCE PARAMETER=ADVANCE"
            " START=%.6f FACTOR=%.6f" % (start, factor)
        )
        if band > 0.0:
            cmd += " BAND=%.3f" % band
        self.printer.lookup_object("gcode").run_script_from_command(cmd)
        msg = "PA tuning tower started: %s" % cmd
        if flow is not None:
            msg += ("\nSlice/print this at flow ~%.1f mm/s; after printing, read"
                    " the best layer and run:\n  PA_CURVE_RECORD NAME=%s"
                    " FLOW=%.1f PA=<value>" % (flow, self.name, flow))
        gcmd.respond_info(msg)

    cmd_PA_CURVE_RECORD_help = (
        "Record a measured best PA for a flow speed: FLOW=<mm/s> PA=<value>"
    )

    def cmd_PA_CURVE_RECORD(self, gcmd):
        flow = gcmd.get_float("FLOW", above=0.0)
        pa = gcmd.get_float("PA", minval=0.0)
        # Replace an existing point at the same flow, else append.
        self.points = [p for p in self.points if abs(p[0] - flow) > 1e-6]
        self.points.append((flow, pa))
        self.points.sort(key=lambda p: p[0])
        gcmd.respond_info(
            "Recorded flow=%.1f mm/s -> pa=%.6f (%d points)"
            % (flow, pa, len(self.points))
        )

    def cmd_PA_CURVE_CALIBRATE_LIST(self, gcmd):
        if not self.points:
            gcmd.respond_info("No pa(flow) points recorded")
            return
        lines = ["Recorded pa(flow) points:"]
        for flow, pa in self.points:
            lines.append("  %.1f mm/s -> %.6f" % (flow, pa))
        gcmd.respond_info("\n".join(lines))

    def cmd_PA_CURVE_CALIBRATE_CLEAR(self, gcmd):
        self.points = []
        gcmd.respond_info("Cleared recorded pa(flow) points")

    cmd_PA_CURVE_CALIBRATE_SAVE_help = (
        "Write pa(flow) CSV and load it into the target curve"
    )

    def cmd_PA_CURVE_CALIBRATE_SAVE(self, gcmd):
        if len(self.points) < 2:
            raise gcmd.error("Need >= 2 recorded points (PA_CURVE_RECORD)")
        output_file = gcmd.get("OUTPUT_FILE", self.output_file)
        config_file = self.printer.get_start_args().get("config_file")
        if config_file:
            out = os.path.join(
                os.path.dirname(os.path.abspath(config_file)), output_file
            )
        else:
            out = os.path.abspath(output_file)
        with open(out, "w") as f:
            f.write("# Pressure advance curve calibration results\n")
            f.write("# flow speed is mm/s of filament (toolhead_v * extr_ratio)\n")
            f.write("flow_speed,pressure_advance\n")
            for flow, pa in self.points:
                f.write("%.2f,%.6f\n" % (flow, pa))
        gcmd.respond_info("Wrote %d points -> %s" % (len(self.points), out))
        # Load into the live curve object if specified.
        target = gcmd.get("TARGET", self.target_curve)
        if target:
            try:
                pac = self.printer.lookup_object(
                    "pressure_advance_curve %s" % target
                )
                pac.set_curve_data(
                    [p[0] for p in self.points], [p[1] for p in self.points]
                )
                gcmd.respond_info(
                    "Loaded into pressure_advance_curve '%s'" % target
                )
            except Exception as e:
                gcmd.respond_info("Could not update curve: %s" % e)


def load_config_prefix(config):
    return PressureAdvanceCurveCalibrate(config)
