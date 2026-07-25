import math
import types

from klippy import toolhead
from klippy.extras import cornerblend


def test_default_arc_has_bounded_chord_angle():
    plan = cornerblend.plan_corner(
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0),
        delta_max=0.1,
        chord_len=10.0,
    )

    assert plan is not None
    assert len(plan["pts"]) + 1 == 18
    assert plan["dev"] <= 0.1


def test_chain_conserves_e_with_bounded_density():
    chain = cornerblend.plan_blend_chain(
        (0.0, 0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0, 0.0),
        prev_e=0.5,
        move_e=0.5,
        prev_len=10.0,
        move_len=10.0,
        accel=15000.0,
        delta_max=0.1,
    )

    assert chain is not None
    assert math.isclose(sum(chain["e_seg"]), 1.0)
    assert chain["extrusion_scale"] <= 1.35


def test_chain_rejects_retraction_and_excessive_density():
    points = (
        (0.0, 0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0, 0.0),
    )
    assert cornerblend.plan_blend_chain(
        *points, -0.5, -0.5, 10.0, 10.0, 15000.0, 0.1
    ) is None


def test_toolhead_preserves_callback_boundaries():
    prev = types.SimpleNamespace(
        is_kinematic_move=True,
        axes_d=[10.0, 0.0, 0.0, 0.5],
        timing_callbacks=[lambda print_time: None],
    )
    move = types.SimpleNamespace(
        is_kinematic_move=True,
        axes_d=[0.0, 10.0, 0.0, 0.5],
    )

    assert toolhead.ToolHead.plan_corner_blend(object(), prev, move) is None

    sharp_points = (
        (0.0, 0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (1.34, 5.0, 0.0, 0.0),
    )
    assert cornerblend.plan_blend_chain(
        *sharp_points,
        0.5,
        0.5,
        10.0,
        10.0,
        15000.0,
        0.1,
        max_turn_deg=160.0,
        max_extrusion_scale=1.35,
    ) is None
