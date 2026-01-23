"""Unit tests for torque_curve module."""
import os
import tempfile
import pytest


class MockConfig:
    """Mock configuration object for testing."""

    def __init__(self, options=None):
        self._options = options or {}
        self._printer = MockPrinter()

    def get_printer(self):
        return self._printer

    def get_name(self):
        return "torque_curve test_curve"

    def get(self, option, default=None):
        return self._options.get(option, default)

    def getfloat(self, option, default=None, above=None, minval=None, maxval=None):
        val = self._options.get(option, default)
        if val is None:
            return default
        return float(val)

    def getboolean(self, option, default=None):
        val = self._options.get(option, default)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return val.lower() in ("true", "yes", "1")

    def error(self, msg):
        raise ValueError(msg)


class MockPrinter:
    """Mock printer object for testing."""

    def __init__(self):
        self._event_handlers = {}
        self._objects = {}
        self._start_args = {}

    def register_event_handler(self, event, callback):
        self._event_handlers[event] = callback

    def lookup_object(self, name, default=None):
        return self._objects.get(name, default)

    def get_start_args(self):
        return self._start_args


class MockGcode:
    """Mock gcode object for testing."""

    def __init__(self):
        self._commands = {}

    def register_mux_command(self, name, key, value, callback, desc=None):
        self._commands[(name, key, value)] = callback


class TestTorqueCurveInterpolation:
    """Test the torque curve interpolation logic."""

    def test_linear_interpolation_midpoint(self):
        """Test linear interpolation at midpoint between two values."""
        # Import the module
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        # Create mock config
        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,  # No safety margin for testing
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)

        # Set up test data: speed 100 -> accel 10000, speed 200 -> accel 5000
        tc.set_curve_data([100, 200], [10000, 5000])

        # Test midpoint (150 mm/s should give 7500 mm/s^2)
        result = tc.get_max_accel_for_speed(150)
        assert result == 7500.0, f"Expected 7500, got {result}"

    def test_linear_interpolation_quarter(self):
        """Test linear interpolation at quarter point."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # 125 mm/s should give 8750 mm/s^2 (75% of 10000 + 25% of 5000)
        result = tc.get_max_accel_for_speed(125)
        assert result == 8750.0, f"Expected 8750, got {result}"

    def test_step_interpolation(self):
        """Test step interpolation uses lower value."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "step",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # Any speed between 100-200 should use 10000 (step uses lower point)
        assert tc.get_max_accel_for_speed(150) == 10000.0
        assert tc.get_max_accel_for_speed(199) == 10000.0

    def test_below_min_speed(self):
        """Test speeds below the curve minimum use first value."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # Speed below minimum should use first value
        assert tc.get_max_accel_for_speed(50) == 10000.0
        assert tc.get_max_accel_for_speed(0) == 10000.0

    def test_above_max_speed(self):
        """Test speeds above the curve maximum use last value."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # Speed above maximum should use last value
        assert tc.get_max_accel_for_speed(250) == 5000.0
        assert tc.get_max_accel_for_speed(500) == 5000.0

    def test_safety_margin(self):
        """Test safety margin is applied correctly."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 0.8,  # 80% safety margin
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # 100 mm/s -> 10000 * 0.8 = 8000
        result = tc.get_max_accel_for_speed(100)
        assert result == 8000.0, f"Expected 8000, got {result}"

    def test_disabled_returns_fallback(self):
        """Test disabled curve returns fallback value."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": False,
            "fallback_max_accel": 3000,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # Should return fallback when disabled
        result = tc.get_max_accel_for_speed(150)
        assert result == 3000.0, f"Expected 3000 (fallback), got {result}"

    def test_negative_speed_uses_absolute(self):
        """Test negative speeds are converted to absolute value."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([100, 200], [10000, 5000])

        # Negative speed should give same result as positive
        assert tc.get_max_accel_for_speed(-150) == tc.get_max_accel_for_speed(150)


class TestTorqueCurveCSV:
    """Test CSV file loading and saving."""

    def test_load_csv_file(self):
        """Test loading a torque curve from CSV file."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("# Comment line\n")
            f.write("speed,max_accel\n")
            f.write("50,15000\n")
            f.write("100,12000\n")
            f.write("150,9000\n")
            f.write("200,6000\n")
            temp_path = f.name

        try:
            config = MockConfig({
                "curve_file": temp_path,
                "interpolation": "linear",
                "safety_margin": 1.0,
                "enabled": True,
            })
            config._printer._objects["gcode"] = MockGcode()

            tc = TorqueCurve(config)

            assert tc.curve_loaded is True
            assert len(tc.speeds) == 4
            assert tc.speeds == [50, 100, 150, 200]
            assert tc.accels == [15000, 12000, 9000, 6000]
        finally:
            os.unlink(temp_path)

    def test_save_csv_file(self):
        """Test saving a torque curve to CSV file."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data([50, 100, 150], [15000, 12000, 9000])

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            tc.save_curve_to_file(temp_path)

            # Read back and verify
            with open(temp_path, 'r') as f:
                content = f.read()

            assert "50.00,15000.00" in content
            assert "100.00,12000.00" in content
            assert "150.00,9000.00" in content
        finally:
            os.unlink(temp_path)

    def test_csv_unsorted_input(self):
        """Test that unsorted CSV data is sorted by speed."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        # Create CSV with unsorted data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("speed,max_accel\n")
            f.write("200,6000\n")
            f.write("50,15000\n")
            f.write("150,9000\n")
            f.write("100,12000\n")
            temp_path = f.name

        try:
            config = MockConfig({
                "curve_file": temp_path,
                "interpolation": "linear",
                "safety_margin": 1.0,
                "enabled": True,
            })
            config._printer._objects["gcode"] = MockGcode()

            tc = TorqueCurve(config)

            # Should be sorted by speed
            assert tc.speeds == [50, 100, 150, 200]
            assert tc.accels == [15000, 12000, 9000, 6000]
        finally:
            os.unlink(temp_path)


class TestTorqueCurveMultiPoint:
    """Test multi-point interpolation scenarios."""

    def test_five_point_curve(self):
        """Test interpolation with 5 data points."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "klippy"))
        from extras.torque_curve import TorqueCurve

        config = MockConfig({
            "interpolation": "linear",
            "safety_margin": 1.0,
            "enabled": True,
        })
        config._printer._objects["gcode"] = MockGcode()

        tc = TorqueCurve(config)
        tc.set_curve_data(
            [0, 50, 100, 150, 200],
            [20000, 18000, 14000, 9000, 5000]
        )

        # Test exact points
        assert tc.get_max_accel_for_speed(0) == 20000
        assert tc.get_max_accel_for_speed(50) == 18000
        assert tc.get_max_accel_for_speed(100) == 14000
        assert tc.get_max_accel_for_speed(150) == 9000
        assert tc.get_max_accel_for_speed(200) == 5000

        # Test interpolated points
        # Between 100 and 150: 14000 -> 9000, at 125 should be 11500
        assert tc.get_max_accel_for_speed(125) == 11500.0
