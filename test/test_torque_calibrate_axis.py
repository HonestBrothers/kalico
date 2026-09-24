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


# --- Notch kept on during the sweep ------------------------------------------

def _load_pathplan():
    spec = importlib.util.spec_from_file_location(
        "_tcc_pathplan", os.path.join(KLIPPY, "extras", "pathplan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_notch(f_x=56.0, f_y=48.0, dt=0.0005, keep=True):
    c, _, _ = make("cartesian", "x")
    c.keep_notch = keep
    c.toolhead = types.SimpleNamespace(
        flush_step_generation=lambda: None, unified_emit=True,
        unified_notch_freq_x=f_x, unified_notch_freq_y=f_y,
        unified_jerk_dt=dt)
    return c


def test_notch_params_follow_keep_notch():
    assert make_notch()._notch_params() == (48.0, 56.0, 0.0005)
    assert make_notch(keep=False)._notch_params() is None
    c = make_notch()
    c.toolhead.unified_emit = False
    assert c._notch_params() is None


def test_notch_ceiling_is_speed_times_f_lo():
    c = make_notch()
    assert c._notch_peak_ceiling(100.0) == pytest.approx(4800.0)
    assert make_notch(keep=False)._notch_peak_ceiling(100.0) is None


def test_stress_point_without_notch_is_identity():
    c = make_notch(keep=False)
    assert c._stress_point(300.0, 20000.0) == (300.0, 20000.0)


def test_stress_point_caps_peak_and_moves_speed_down():
    c = make_notch()
    k = 0.5 / 56.0 - 2 * 0.0005
    # Saturated: plateau at the commanded accel, left k*A below the target.
    v, a = c._stress_point(400.0, 10000.0)
    assert a == 10000.0 and v == pytest.approx(400.0 - k * 10000.0)
    # Unsaturated: the ramp only reaches speed*f_lo.
    v, a = c._stress_point(100.0, 20000.0)
    assert a == pytest.approx(4800.0)
    assert v == pytest.approx(100.0 - k * 4800.0)


def test_notched_move_is_long_enough():
    c = make_notch()
    pp = _load_pathplan()
    for v, a in ((100.0, 20000.0), (400.0, 10000.0), (500.0, 3000.0)):
        need = 2 * pp.notch_dist(0.0, v, a, 48.0, 56.0)
        assert c._calculate_required_distance(v, a) >= need
    # Without the notch it is the plain triangle.
    c.keep_notch = False
    assert c._calculate_required_distance(400.0, 10000.0) == pytest.approx(
        400.0 ** 2 / 10000.0)


def test_recorded_points_admit_the_tested_ramps():
    """The planner, given a curve built from the recorded stress points,
    lets a 0 -> v ramp use the accel that passed at v."""
    c = make_notch()
    pp = _load_pathplan()
    probes = [(200.0, 9000.0), (300.0, 8000.0), (400.0, 7000.0),
              (500.0, 6000.0)]
    pts = sorted(c._stress_point(v, a) for v, a in probes)
    curve = pp.AccelCurve([p[0] for p in pts], [p[1] for p in pts])
    for v, a in probes:
        got = curve.ramp_accel(0.0, v, 1e9, 48.0, 56.0, 0.0005)
        assert got == pytest.approx(min(a, v * 48.0), rel=1e-6)


def test_sweep_keeps_notch_when_asked():
    c = make_notch()
    c.measure_vibration = False
    c.vibration_shaper_off = False
    msgs = []
    gcmd = types.SimpleNamespace(respond_info=msgs.append)
    saved = c._disable_reshapers(gcmd)
    assert c.toolhead.unified_emit and "unified" not in saved
    assert any("stays ON" in m for m in msgs)
