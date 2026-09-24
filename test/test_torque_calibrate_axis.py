"""TORQUE_CURVE_CALIBRATE axis selection: X, Y and XY (both at once) on
cartesian and CoreXY kinematics."""
import importlib.util
import math
import os
import types

import pytest

KLIPPY = os.path.join(os.path.dirname(__file__), "..", "klippy")


def _load_calibrate():
    spec = importlib.util.spec_from_file_location(
        "_tcc_axis_test",
        os.path.join(KLIPPY, "extras", "torque_curve_calibrate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib = _load_calibrate()


class FakeStepper:
    # Linear kinematics: position = cx*x + cy*y. Cartesian X is (1, 0),
    # CoreXY A is (1, 1) and B is (1, -1).
    def __init__(self, name, cx, cy, step_dist=0.0125, rot=40.0):
        self.name, self.cx, self.cy = name, cx, cy
        self.step_dist, self.rot = step_dist, rot
        self.mcu_pos = 0

    def get_name(self):
        return self.name

    def is_active_axis(self, axis):
        return {"x": self.cx, "y": self.cy}.get(axis, 0.0) != 0.0

    def calc_position_from_coord(self, coord):
        return self.cx * coord[0] + self.cy * coord[1]

    def get_step_dist(self):
        return self.step_dist

    def get_rotation_distance(self):
        return self.rot, 3200

    def get_mcu_position(self):
        return self.mcu_pos


class FakeRail:
    def __init__(self, stepper, rng):
        self.stepper, self.rng = stepper, rng

    def get_steppers(self):
        return [self.stepper]

    def get_range(self):
        return self.rng


def make(kind, axis):
    c = object.__new__(calib.TorqueCurveCalibrate)
    if kind == "cartesian":
        sx, sy = FakeStepper("stepper_x", 1, 0), FakeStepper("stepper_y", 0, 1)
    else:  # corexy: A = x + y, B = x - y
        sx, sy = FakeStepper("stepper_x", 1, 1), FakeStepper("stepper_y", 1, -1)
    c.kin = types.SimpleNamespace(rails=[FakeRail(sx, (0.0, 250.0)),
                                         FakeRail(sy, (0.0, 200.0))])
    c.toolhead = types.SimpleNamespace(flush_step_generation=lambda: None)
    c._stepper_microsteps = {"stepper_x": 16, "stepper_y": 16}
    c.position_tolerance = None
    c.position_tolerance_steps = 2.5
    c.settle_time = 0.0
    c.test_axis = calib.TorqueCurveCalibrate._parse_axis(axis, ValueError)
    return c, sx, sy


def ratios(c):
    return {st.get_name(): r for st, r in c._test_steppers()}


@pytest.mark.parametrize("value,want", [
    ("x", "x"), ("Y", "y"), ("XY", "xy"), ("yx", "xy"), ("both", "xy")])
def test_parse_axis(value, want):
    assert calib.TorqueCurveCalibrate._parse_axis(value, ValueError) == want


def test_parse_axis_rejects_z():
    with pytest.raises(ValueError):
        calib.TorqueCurveCalibrate._parse_axis("z", ValueError)


def test_cartesian_single_axis_unchanged():
    c, sx, sy = make("cartesian", "x")
    assert ratios(c) == {"stepper_x": 1.0}
    assert c._get_axis_limits() == (0.0, 250.0)
    assert c._set_path_pos([1.0, 2.0, 3.0], 42.0) == [42.0, 2.0, 3.0]
    assert c._step_dist() == pytest.approx(0.0125)
    assert c._axis_rotation_distance() == pytest.approx(40.0)
    assert c._full_step_dist() == pytest.approx(0.2)
    c, sx, sy = make("cartesian", "y")
    assert ratios(c) == {"stepper_y": 1.0}
    assert c._set_path_pos([1.0, 2.0, 3.0], 42.0) == [1.0, 42.0, 3.0]


def test_cartesian_xy_diagonal_drives_both_motors():
    c, sx, sy = make("cartesian", "xy")
    r = ratios(c)
    assert r["stepper_x"] == pytest.approx(math.sqrt(0.5))
    assert r["stepper_y"] == pytest.approx(math.sqrt(0.5))
    lo, hi = c._get_axis_limits()
    # Centred at (125, 100); Y (200 mm) is the tighter range: +-100/sqrt(.5)
    assert c._xy_center == (125.0, 100.0)
    assert lo == pytest.approx(-100.0 / math.sqrt(0.5))
    assert hi == pytest.approx(100.0 / math.sqrt(0.5))
    for s in (lo, hi):
        x, y, _ = c._set_path_pos([0.0, 0.0, 5.0], s)
        assert -1e-9 <= x <= 250.0 + 1e-9 and -1e-9 <= y <= 200.0 + 1e-9
    # Each motor moves 1/sqrt(2) mm per path mm -> path step is longer, and
    # the back-EMF top speed in path mm/s is higher.
    assert c._step_dist() == pytest.approx(0.0125 / math.sqrt(0.5))
    assert c._axis_rotation_distance() == pytest.approx(40.0 / math.sqrt(0.5))


def test_corexy_single_axis_uses_both_motors():
    c, sa, sb = make("corexy", "x")
    assert ratios(c) == {"stepper_x": 1.0, "stepper_y": 1.0}
    c, sa, sb = make("corexy", "y")
    assert ratios(c) == {"stepper_x": 1.0, "stepper_y": 1.0}


def test_corexy_xy_diagonal_loads_one_motor():
    c, sa, sb = make("corexy", "xy")
    # +X+Y: A = x + y moves sqrt(2) per path mm, B = x - y stays still.
    r = ratios(c)
    assert list(r) == ["stepper_x"]
    assert r["stepper_x"] == pytest.approx(math.sqrt(2.0))
    assert c._primary_stepper()[0] is sa
    assert c._step_dist() == pytest.approx(0.0125 / math.sqrt(2.0))
    assert c._axis_rotation_distance() == pytest.approx(40.0 / math.sqrt(2.0))


def test_lost_steps_checked_on_every_driven_motor():
    c, sx, sy = make("cartesian", "xy")
    c._home_axis = lambda: None
    ref = c._get_stepper_position()
    assert ref == {"stepper_x": 0, "stepper_y": 0}
    # tolerance: 2.5 full steps * 0.2 mm = 0.5 mm = 40 microsteps
    sy.mcu_pos = 39
    lost, diff = c._check_for_lost_steps(ref)
    assert not lost and diff == pytest.approx(39 * 0.0125)
    # X clean, Y skipped: still a skip.
    sy.mcu_pos = 64
    lost, diff = c._check_for_lost_steps(ref)
    assert lost and diff == pytest.approx(64 * 0.0125)


def test_home_axis_homes_both_for_xy():
    sent = []
    c, _, _ = make("cartesian", "xy")
    c.gcode = types.SimpleNamespace(run_script_from_command=sent.append)
    c._home_axis()
    c.test_axis = "y"
    c._home_axis()
    assert sent == ["G28 X Y", "G28 Y"]
