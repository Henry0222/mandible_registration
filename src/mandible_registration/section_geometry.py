"""Local, reproducible section geometry in the common T0 world frame (mm).

No anatomical orientation is inferred from STL axes. ROI selection retains every
triangle intersecting the sphere, including triangles whose vertices all lie
outside it. Cut segments are clipped analytically to that sphere, without caps
or fabricated bone surfaces. This module has no Qt, VTK, or file I/O dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

ROI_RADIUS_MM = 20.0
INITIAL_ORIENTATION = "世界 YZ 平面 · 法向 +X；非解剖方位"


def vector3(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("需要有限的三维坐标。")
    return result.copy()


def unit(value):
    result = vector3(value)
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError("方向不能为零。")
    return result / length


def _triangle_distance_squared(corners):
    """Squared distance from the origin to triangles, including degenerate ones."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    edge_distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        denominator = np.einsum("ij,ij->i", edge, edge)
        t = np.clip(np.divide(-np.einsum("ij,ij->i", start, edge), denominator,
                              out=np.zeros(len(a)), where=denominator > 1e-24), 0, 1)
        closest = start + t[:, None] * edge
        edge_distances.append(np.einsum("ij,ij->i", closest, closest))
    distance = np.min(edge_distances, axis=0)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = np.einsum("ij,ij->i", normal, normal)
    height = np.einsum("ij,ij->i", a, normal)
    projection = normal * np.divide(height, n2, out=np.zeros(len(a)), where=n2 > 1e-24)[:, None]
    inside = n2 > 1e-24
    for start, end in ((a, b), (b, c), (c, a)):
        inside &= np.einsum("ij,ij->i", np.cross(end - start, projection - start), normal) >= -1e-12 * n2
    distance[inside] = height[inside] ** 2 / n2[inside]
    return distance


@dataclass
class RoiMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    source_face_ids: np.ndarray
    source_triangle_count: int


def sphere_roi(vertices, triangles, center, radius=ROI_RADIUS_MM):
    center = vector3(center)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("ROI 半径必须为正数。")
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    # Vertex outcodes reject triangles wholly beyond any of the six ROI box
    # planes. This scans indices rather than materializing every full triangle.
    # Unlike a vertex-in-sphere test it also keeps large triangles crossing ROI.
    codes = np.zeros(len(vertices), dtype=np.uint8)
    for axis in range(3):
        codes |= (vertices[:, axis] < center[axis] - radius).astype(np.uint8) << (2 * axis)
        codes |= (vertices[:, axis] > center[axis] + radius).astype(np.uint8) << (2 * axis + 1)
    broad = np.flatnonzero((codes[triangles[:, 0]] & codes[triangles[:, 1]] & codes[triangles[:, 2]]) == 0)
    selected = []
    for start in range(0, len(broad), 65536):
        candidates = broad[start:start + 65536]
        corners = vertices[triangles[candidates]] - center
        distances = _triangle_distance_squared(corners)
        selected.append(candidates[distances <= radius ** 2 + 1e-9])
    ids = np.concatenate(selected).astype(np.int64) if selected else np.empty(0, dtype=np.int64)
    used, inverse = np.unique(triangles[ids].ravel(), return_inverse=True)
    return RoiMesh(np.asarray(vertices[used], dtype=float), inverse.reshape(-1, 3), ids, len(triangles))


def clip_segments_to_sphere(segments, center, radius=ROI_RADIUS_MM):
    """Return clipped Nx2x3 segments and their input indices, never closing gaps."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    start = segments[:, 0] - vector3(center)
    edge = segments[:, 1] - segments[:, 0]
    aa = np.einsum("ij,ij->i", edge, edge)
    bb = np.einsum("ij,ij->i", start, edge)
    cc = np.einsum("ij,ij->i", start, start) - radius ** 2
    disc = bb ** 2 - aa * cc
    valid = (aa > 1e-20) & (disc >= 0)
    root = np.sqrt(np.maximum(disc, 0))
    lo = np.maximum(0, np.divide(-bb - root, aa, out=np.zeros_like(aa), where=valid))
    hi = np.minimum(1, np.divide(-bb + root, aa, out=np.zeros_like(aa), where=valid))
    ids = np.flatnonzero(valid & (hi - lo > 1e-10))
    result = segments[ids, :1] + np.stack((lo[ids], hi[ids]), axis=1)[:, :, None] * edge[ids, None]
    return result, ids


def _rotate(vector, axis, angle):
    axis = unit(axis)
    return vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle) + axis * np.dot(axis, vector) * (1 - np.cos(angle))


@dataclass
class SectionState:
    center: np.ndarray
    normal: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    up: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    offset_mm: float = 0.0
    pan_mm: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.center = vector3(self.center)
        self.normal = unit(self.normal)
        self.up = unit(self.up - self.normal * np.dot(self.up, self.normal))

    @property
    def origin(self):
        # Positive scroll moves into the scene, along camera direction of projection.
        return self.center - self.normal * self.offset_mm

    @property
    def right(self):
        return np.cross(self.up, self.normal)

    @property
    def disk_radius(self):
        return float(np.sqrt(max(0, ROI_RADIUS_MM ** 2 - self.offset_mm ** 2)))

    def rotate(self, dx, dy):
        yaw, pitch = np.deg2rad([-dx * .4, -dy * .4])
        self.normal = unit(_rotate(self.normal, self.up, yaw))
        right = self.right
        self.normal = unit(_rotate(self.normal, right, pitch))
        self.up = unit(_rotate(self.up, right, pitch))

    def scroll(self, steps, step_mm=.5):
        if not np.isfinite(steps) or not np.isfinite(step_mm) or step_mm <= 0:
            raise ValueError("剖面步长必须是有限正数。")
        self.offset_mm = float(np.clip(self.offset_mm + steps * step_mm, -ROI_RADIUS_MM, ROI_RADIUS_MM))

    def snapshot(self):
        return {"roi_center_t0_mm": self.center.tolist(), "roi_radius_mm": ROI_RADIUS_MM,
                "origin_mm": self.origin.tolist(), "normal_xyz": self.normal.tolist(),
                "view_up_xyz": self.up.tolist(), "offset_along_view_mm": self.offset_mm,
                "orientation_reference": INITIAL_ORIENTATION}
