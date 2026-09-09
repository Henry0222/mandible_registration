"""Auditable selected-surface centroids and rigid motion; no anatomical axis inference."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from .transforms import validated_rigid_transform, rotation_degrees
from .registration_profiles import condyle_profile_path


REGIONS = {"left": "左侧髁突", "right": "右侧髁突"}


def displacement_description(side, vector):
    """Explicit user convention in common world XYZ; no inferred anatomy."""
    x, y, z = np.asarray(vector, dtype=float)
    values = [(x if side == "left" else -x, "外移", "内移"),
              (-y, "前移", "后移"), (z, "上移", "下移")]
    return " · ".join(f"{positive if value > 0 else negative} {abs(value):.3f} mm"
                      for value, positive, negative in values if abs(value) >= .0005) or "无明显位移"


def rotation_in_minus_x_view(matrix):
    """Closest YZ-plane rotation, clockwise as seen from -X (screen up +Z).

    Equivalent to the quaternion's signed X twist. The 3D principal angle is a
    separate quantity: non-X rotations must not be reported as jaw opening.
    """
    r = validated_rigid_transform(matrix, tolerance=1e-4)[:3, :3]
    sine, cosine = r[2, 1] - r[1, 2], r[1, 1] + r[2, 2]
    if np.hypot(sine, cosine) < 1e-10:
        return {"signed_degrees": None, "label": "顺逆旋方向无法判定"}
    angle = float(np.degrees(np.arctan2(sine, cosine)))
    if abs(abs(angle) - 180) < 1e-6:
        return {"signed_degrees": angle, "label": "顺逆旋方向不定（180°）"}
    return {"signed_degrees": angle,
            "label": "无明显顺逆旋" if abs(angle) < .005 else "下颌骨顺旋" if angle > 0 else "下颌骨逆旋"}


def profile_path(mesh_path):
    return condyle_profile_path(mesh_path)


def surface_centroid(mesh, selected):
    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (len(mesh.triangles),) or not mask.any():
        raise ValueError("请先选取髁突面片。")
    corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[mask]]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("选区没有有效表面积。")
    center = np.sum(corners.mean(axis=1) * areas[:, None], axis=0) / total
    return center, total


def build_profile(mesh, mesh_path, mesh_sha256, masks):
    from .core_bridge import ensure_registration_core
    ensure_registration_core()
    from auto_alignment.integration.selection import encode_face_ranges

    occupied = np.zeros(len(mesh.triangles), dtype=bool)
    regions = {}
    for key in REGIONS:
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape != occupied.shape:
            raise ValueError("选区面片数与颌骨不一致。")
        if np.any(occupied & mask):
            raise ValueError("左右髁突选区不能包含相同面片，请取消重叠选区。")
        occupied |= mask
        if not mask.any():
            continue
        center, area = surface_centroid(mesh, mask)
        regions[key] = {"name": REGIONS[key], "selected_ranges": encode_face_ranges(np.flatnonzero(mask)),
                        "triangle_count": int(mask.sum()), "area_mm2": area, "center_ct_mm": center.tolist()}
    return {"schema_version": 1, "kind": "mandible_condyle_regions", "mesh_path": str(Path(mesh_path).resolve()),
            "mesh_sha256": mesh_sha256, "mesh_triangle_count": len(mesh.triangles),
            "center_definition": "selected_surface_area_weighted_centroid",
            "updated_at": datetime.now().astimezone().isoformat(), "regions": regions}


def validate_profile(data, expected_sha256):
    if (data.get("kind") != "mandible_condyle_regions" or data.get("schema_version") != 1
            or not expected_sha256 or data.get("mesh_sha256") != expected_sha256):
        raise ValueError("髁突选区不属于本次配准使用的颌骨 STL，请重新选区。")
    if data.get("center_definition") != "selected_surface_area_weighted_centroid":
        raise ValueError("髁突中心定义不受支持。")
    if not isinstance(data.get("regions"), dict) or set(data["regions"]) - REGIONS.keys():
        raise ValueError("髁突选区名称无效。")
    for region in data["regions"].values():
        center = np.asarray(region["center_ct_mm"], dtype=float)
        area = float(region["area_mm2"])
        if (center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(area)
                or area <= 0 or int(region["triangle_count"]) <= 0):
            raise ValueError("髁突选区中心或表面积无效。")
    return data


def read_profile(path, expected_sha256):
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")), expected_sha256)


def analyze_motion(profile, t0, t1):
    first = validated_rigid_transform(t0, tolerance=1e-4)
    second = validated_rigid_transform(t1, tolerance=1e-4)
    delta = second @ np.linalg.inv(first)
    regions = {}
    for key, region in profile["regions"].items():
        center = np.asarray(region["center_ct_mm"], dtype=float)
        p0 = first[:3, :3] @ center + first[:3, 3]
        p1 = second[:3, :3] @ center + second[:3, 3]
        vector = p1 - p0
        distance = float(np.linalg.norm(vector))
        direction = vector / distance if distance > 1e-9 else None
        regions[key] = {"name": REGIONS[key], "center_t0_mm": p0.tolist(), "center_t1_mm": p1.tolist(),
                        "displacement_xyz_mm": vector.tolist(), "distance_mm": distance,
                        "direction_unit_xyz": direction.tolist() if direction is not None else None,
                        "direction_angles_to_positive_xyz_deg": np.degrees(np.arccos(np.clip(direction, -1, 1))).tolist() if direction is not None else None}
    return {"regions": regions, "rigid_rotation_degrees": rotation_degrees(delta),
            "rotation_minus_x_view": rotation_in_minus_x_view(delta),
            "coordinate_reference": "上颌口扫.1 坐标系 XYZ，不代表解剖方向",
            "rotation_definition": "整个颌骨的刚体总旋转角（0–180°），不是单个髁突独立旋转角",
            "selection": profile}


def project_analysis(project_path):
    data = json.loads(Path(project_path).read_text(encoding="utf-8"))
    record = data.get("inputs", {}).get("ct_mandible", {})
    mesh_path = Path(record.get("path", ""))
    if not mesh_path.is_absolute():
        mesh_path = Path(project_path).parent / mesh_path
    current = profile_path(mesh_path)
    profile = read_profile(current, record.get("sha256")) if current.is_file() else data.get("condyle_selection")
    if profile is None:
        return None
    profile = validate_profile(profile, record.get("sha256"))
    matrices = data["transforms"]
    t0 = matrices.get("T_MANDIBLE_T0", matrices.get("T_CT", {}))["matrix"]
    delta = validated_rigid_transform(matrices["T_DELTA"]["matrix"], tolerance=1e-4)
    t1 = matrices.get("T_MANDIBLE_T1", {}).get("matrix", delta @ np.asarray(t0))
    if not np.allclose(t1, delta @ np.asarray(t0), atol=1e-4, rtol=0):
        raise ValueError("项目中的颌骨变换矩阵不一致，不能计算髁突位移。")
    return analyze_motion(profile, t0, t1)
