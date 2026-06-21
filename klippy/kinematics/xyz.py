# Code for handling the kinematics of CoreXYZ robots
#
# CoreXYZ uses four motors (A, B, C, D) that each move on all three
# toolhead axes.  Z is a common-mode of all four belts while X and Y are
# differential:
#     A =  x + y + z
#     B =  x - y + z
#     C = -x - y + z
#     D = -x + y + z
# The (over-determined) forward transform is:
#     x = (A + B - C - D) / 4
#     y = (A - B - C + D) / 4
#     z = (A + B + C + D) / 4
#
# Motor A is configured via [stepper_x], B via [stepper_y], C via
# [stepper_z] (these three carry the axis endstops and ranges), and the
# fourth motor D via [stepper_w] (a bare stepper without its own endstop).
#
# Copyright (C) 2026  Kalico contributors
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from klippy import stepper


class XYZKinematics:
    def __init__(self, toolhead, config):
        # Setup the three axis rails (motors A, B, C) plus the fourth motor
        self.rails = [
            stepper.LookupMultiRail(config.getsection("stepper_" + n))
            for n in "xyz"
        ]
        self.motor_d = stepper.PrinterStepper(config.getsection("stepper_w"))
        # All four motors move on every axis, so register every stepper with
        # every axis endstop so homing any axis stops all carriages.
        all_steppers = self.get_steppers()
        for rail in self.rails:
            endstop = rail.get_endstops()[0][0]
            for s in all_steppers:
                endstop.add_stepper(s)
        # Assign the coupled itersolve to each motor
        self.rails[0].setup_itersolve("xyz_stepper_alloc", b"a")
        self.rails[1].setup_itersolve("xyz_stepper_alloc", b"b")
        self.rails[2].setup_itersolve("xyz_stepper_alloc", b"c")
        self.motor_d.setup_itersolve("xyz_stepper_alloc", b"d")
        for s in all_steppers:
            s.set_trapq(toolhead.get_trapq())
            toolhead.register_step_generator(s.generate_steps)
        config.get_printer().register_event_handler(
            "stepper_enable:motor_off", self._motor_off
        )
        # Setup boundary checks
        max_velocity, max_accel = toolhead.get_max_velocity()
        self.max_z_velocity = config.getfloat(
            "max_z_velocity", max_velocity, above=0.0, maxval=max_velocity
        )
        self.max_z_accel = config.getfloat(
            "max_z_accel", max_accel, above=0.0, maxval=max_accel
        )
        self.limits = [(1.0, -1.0)] * 3
        ranges = [r.get_range() for r in self.rails]
        self.axes_min = toolhead.Coord(*[r[0] for r in ranges], e=0.0)
        self.axes_max = toolhead.Coord(*[r[1] for r in ranges], e=0.0)
        self.supports_dual_carriage = False

    def get_steppers(self):
        steppers = [s for rail in self.rails for s in rail.get_steppers()]
        steppers.append(self.motor_d)
        return steppers

    def calc_position(self, stepper_positions):
        a, b, c = [stepper_positions[rail.get_name()] for rail in self.rails]
        d = stepper_positions[self.motor_d.get_name()]
        # Forward transform (inverse of the coupling matrix)
        return [
            0.25 * (a + b - c - d),
            0.25 * (a - b - c + d),
            0.25 * (a + b + c + d),
        ]

    def set_position(self, newpos, homing_axes):
        for i, rail in enumerate(self.rails):
            rail.set_position(newpos)
            if i in homing_axes:
                self.limits[i] = rail.get_range()
        self.motor_d.set_position(newpos)

    def note_z_not_homed(self):
        self.clear_homing_state([2])

    def clear_homing_state(self, axes):
        for i, _ in enumerate(self.limits):
            if i in axes:
                self.limits[i] = (1.0, -1.0)

    def home(self, homing_state):
        # Each axis is homed independently and in order
        for axis in homing_state.get_axes():
            rail = self.rails[axis]
            # Determine movement
            position_min, position_max = rail.get_range()
            hi = rail.get_homing_info()
            homepos = [None, None, None, None]
            homepos[axis] = hi.position_endstop
            forcepos = list(homepos)
            if hi.positive_dir:
                forcepos[axis] -= 1.5 * (hi.position_endstop - position_min)
            else:
                forcepos[axis] += 1.5 * (position_max - hi.position_endstop)
            # Perform homing
            homing_state.home_rails([rail], forcepos, homepos)

    def _motor_off(self, print_time):
        self.clear_homing_state((0, 1, 2))

    def _check_endstops(self, move):
        end_pos = move.end_pos
        for i in (0, 1, 2):
            if move.axes_d[i] and (
                end_pos[i] < self.limits[i][0] or end_pos[i] > self.limits[i][1]
            ):
                if self.limits[i][0] > self.limits[i][1]:
                    raise move.move_error("Must home axis first")
                raise move.move_error()

    def check_move(self, move):
        limits = self.limits
        xpos, ypos = move.end_pos[:2]
        if (
            xpos < limits[0][0]
            or xpos > limits[0][1]
            or ypos < limits[1][0]
            or ypos > limits[1][1]
        ):
            self._check_endstops(move)
        if not move.axes_d[2]:
            # Normal XY move - use defaults
            return
        # Move with Z - update velocity and accel for slower Z axis
        self._check_endstops(move)
        z_ratio = move.move_d / abs(move.axes_d[2])
        move.limit_speed(
            self.max_z_velocity * z_ratio, self.max_z_accel * z_ratio
        )

    def get_status(self, eventtime):
        axes = [a for a, (l, h) in zip("xyz", self.limits) if l <= h]
        return {
            "homed_axes": "".join(axes),
            "axis_minimum": self.axes_min,
            "axis_maximum": self.axes_max,
        }


def load_kinematics(toolhead, config):
    return XYZKinematics(toolhead, config)
