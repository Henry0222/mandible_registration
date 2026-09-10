import numpy as np
import pytest

from mandible_registration.condyles import displacement_description, rotation_in_minus_x_view


@pytest.mark.parametrize("side,vector,expected", [
    ("left", [1, -2, 3], "外移 1.000 mm · 前移 2.000 mm · 上移 3.000 mm"),
    ("right", [-1, -2, 3], "外移 1.000 mm · 前移 2.000 mm · 上移 3.000 mm"),
    ("left", [-1, 2, -3], "内移 1.000 mm · 后移 2.000 mm · 下移 3.000 mm"),
    ("right", [1, 2, -3], "内移 1.000 mm · 后移 2.000 mm · 下移 3.000 mm"),
    ("right", [0, 0, 0], "无明显位移"),
])
def test_user_side_specific_axis_convention(side, vector, expected):
    assert displacement_description(side, vector) == expected


def rotation_x(angle):
    t = np.eye(4)
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    t[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return t


@pytest.mark.parametrize("angle,label", [(12, "下颌骨顺旋"), (-12, "下颌骨逆旋"), (0, "无明显顺逆旋")])
def test_clockwise_observed_from_minus_x_is_positive_x(angle, label):
    result = rotation_in_minus_x_view(rotation_x(angle))
    assert result["label"] == label
    assert result["signed_degrees"] == pytest.approx(angle)
    # Screen right = -Y, screen up = +Z viewed from -X. A positive X
    # rotation moves the top point to screen right, i.e. clockwise.
    top = rotation_x(angle)[:3, :3] @ [0, 0, 1]
    assert -top[1] == pytest.approx(np.sin(np.deg2rad(angle)))


def test_non_x_rotation_is_not_mislabeled_as_opening():
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    assert rotation_in_minus_x_view(matrix)["label"] == "无明显顺逆旋"
    matrix[:3, :3] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    assert rotation_in_minus_x_view(matrix)["signed_degrees"] is None
    assert "方向不定" in rotation_in_minus_x_view(rotation_x(180))["label"]
