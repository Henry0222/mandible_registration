from __future__ import annotations

import math

import numpy as np
import open3d as o3d


class TransformValidationError(ValueError):
    pass


def validated_rigid_transform(value: object, *, tolerance: float = 1e-5) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise TransformValidationError("变换必须是有限的 4x4 矩阵。")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=tolerance, rtol=0.0):
        raise TransformValidationError("齐次变换最后一行无效。")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0.0):
        raise TransformValidationError("变换包含缩放或非正交旋转。")
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > tolerance:
        raise TransformValidationError("变换包含镜像或非法旋转。")
    return matrix.copy()


def compose(*matrices: object) -> np.ndarray:
    """Compose column-vector transforms in written left-to-right order.

    ``compose(delta, baseline)`` returns ``delta @ baseline`` so the baseline
    pose is applied first and the delta pose second.
    """

    result = np.eye(4, dtype=float)
    for matrix in matrices:
        result = result @ validated_rigid_transform(matrix)
    return validated_rigid_transform(result, tolerance=1e-4)


def transformed_mesh(mesh: o3d.geometry.TriangleMesh, matrix: object) -> o3d.geometry.TriangleMesh:
    result = o3d.geometry.TriangleMesh(mesh)
    result.transform(validated_rigid_transform(matrix, tolerance=1e-4))
    result.compute_triangle_normals()
    result.compute_vertex_normals()
    return result


def rotation_degrees(matrix: object) -> float:
    rotation = validated_rigid_transform(matrix, tolerance=1e-4)[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def translation_vector_mm(matrix: object) -> tuple[float, float, float]:
    translation = validated_rigid_transform(matrix, tolerance=1e-4)[:3, 3]
    return tuple(float(value) for value in translation)


def translation_mm(matrix: object) -> float:
    return float(np.linalg.norm(translation_vector_mm(matrix)))
