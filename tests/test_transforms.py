import numpy as np
import pytest

from mandible_registration.transforms import (
    TransformValidationError,
    compose,
    rotation_degrees,
    translation_mm,
    validated_rigid_transform,
)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def test_compose_applies_ct_pose_then_delta() -> None:
    ct_pose = _translation(2.0, 0.0, 0.0)
    delta = _translation(0.0, 3.0, 0.0)

    combined = compose(delta, ct_pose)

    assert np.allclose(combined, delta @ ct_pose)
    assert translation_mm(combined) == pytest.approx(np.sqrt(13.0))
    assert rotation_degrees(combined) == pytest.approx(0.0)


def test_scaling_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 1.01
    with pytest.raises(TransformValidationError, match="缩放"):
        validated_rigid_transform(matrix)
