"""Common-frame scene loading and geometry calculations, independent of the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import open3d as o3d

from .core_bridge import ensure_registration_core
from .ct_coordinates import ct_frame_compatibility
from .transforms import validated_rigid_transform

ensure_registration_core()
def load_mesh_arrays(path):
    """Read viewer geometry into owned NumPy arrays without creating Open3D objects."""
    from vtkmodules.vtkIOGeometry import vtkSTLReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if reader.GetErrorCode() or not data.GetNumberOfCells() or not data.GetNumberOfPoints():
        raise ValueError("STL 不包含可读取的三角网格")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Setting the shape on a NumPy array has been deprecated"
        )
        vertices = np.array(vtk_to_numpy(data.GetPoints().GetData()), dtype=np.float64, copy=True)
        offsets = vtk_to_numpy(data.GetPolys().GetOffsetsArray())
        faces = np.array(
            vtk_to_numpy(data.GetPolys().GetConnectivityArray()), dtype=np.int32, copy=True
        )
    if not np.isfinite(vertices).all() or not np.all(np.diff(offsets) == 3):
        raise ValueError("STL 坐标或三角面无效")
    return vertices, faces.reshape(-1, 3), None


def load_mesh(path):
    """Viewer-only STL reader: no repeated registration cleanup of exported meshes."""
    vertices, faces, _ = load_mesh_arrays(path)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces.reshape(-1, 3)))
    mesh.compute_vertex_normals()
    return mesh, None


MODEL_SPECS = (
    ("ct_mandible_t0", "颌骨.1（T0）", (0.22, 0.58, 0.90)),
    ("ct_mandible_t1", "颌骨.2（T1）", (1.00, 0.58, 0.20)),
    ("ct_maxilla_t0", "上颌骨.1（固定参考）", (0.86, 0.79, 0.62)),
    ("baseline_lower", "下颌口扫.1", (0.20, 0.82, 0.48)),
    ("baseline_upper", "上颌口扫.1（固定参考）", (0.86, 0.88, 0.94)),
    ("ct_dentition_t0", "全牙列.1（按下颌定位）", (0.70, 0.65, 0.86)),
    ("ct_dentition_t1", "全牙列.2（随下颌位姿）", (0.94, 0.54, 0.78)),
    ("followup_upper_in_t0", "上颌口扫.2（同系）", (0.28, 0.83, 0.88)),
    ("followup_lower_in_t0", "下颌口扫.2（同系）", (0.97, 0.36, 0.43)),
)

BONE_KEYS = ("ct_mandible_t0", "ct_mandible_t1")
COMPARISON_KEYS = (
    *BONE_KEYS,
    "ct_maxilla_t0",
    "baseline_lower",
    "baseline_upper",
    "followup_upper_in_t0",
    "followup_lower_in_t0",
    "ct_dentition_t0",
    "ct_dentition_t1",
)
CT_INSPECTION_KEYS = ("ct_dentition_t0", "baseline_lower")
JOINT_VIEW_KEYS = (*BONE_KEYS, "ct_maxilla_t0")
OPTIONAL_MODEL_KEYS = frozenset({"ct_maxilla_t0"})


@dataclass
class SceneModel:
    key: str
    title: str
    path: Path
    mesh: o3d.geometry.TriangleMesh
    color: tuple[float, float, float]
    sha256: str = ""


@dataclass
class SceneData:
    project_path: Path
    models: list[SceneModel]
    delta: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    coordinate_reference: str = "上颌口扫.1 坐标系；单位 mm"
    deferred_keys: set[str] = field(default_factory=set)


def load_scene(project_path: str | Path, *, keys=None, array_meshes: bool = False) -> SceneData:
    project_path = Path(project_path).resolve()
    data = json.loads(project_path.read_text(encoding="utf-8"))
    if data.get("workflow") != "mandibular_pose_transfer":
        raise ValueError("请选择本软件结果目录内的 project.json。")
    scene = SceneData(project_path, [], coordinate_reference=data.get("coordinate_reference", "T0 / mm"))
    input_records = data.get("inputs", {})
    dentition_facts = input_records.get("ct_dentition", {}).get("mesh_facts")
    mandible_facts = input_records.get("ct_mandible", {}).get("mesh_facts")
    if isinstance(dentition_facts, dict) and isinstance(mandible_facts, dict):
        try:
            frame_report = ct_frame_compatibility(dentition_facts, mandible_facts)
        except (KeyError, TypeError, ValueError):
            frame_report = None
        if frame_report is not None and not frame_report["compatible"]:
            scene.warnings.append(
                "此结果的原始全牙列与颌骨不在同一 CT 坐标系"
                f"（包围盒交叠 {100 * float(frame_report['aabb_overlap_fraction']):.1f}%，"
                f"中心相距 {float(frame_report['center_distance_mm']):.1f} mm）；"
                "牙列与颌骨相互分离，因此本结果不能用于髁突变化判断，请换用未经单独移动的全牙列重新配准。"
            )
    maxilla_facts = input_records.get("ct_maxilla", {}).get("mesh_facts")
    if isinstance(dentition_facts, dict) and isinstance(maxilla_facts, dict):
        try:
            frame_report = ct_frame_compatibility(dentition_facts, maxilla_facts)
        except (KeyError, TypeError, ValueError):
            frame_report = None
        if frame_report is not None and not frame_report["compatible"]:
            scene.warnings.append(
                "此结果的原始全牙列与上颌骨不在同一 CT 坐标系"
                f"（包围盒交叠 {100 * float(frame_report['aabb_overlap_fraction']):.1f}%，"
                f"中心相距 {float(frame_report['center_distance_mm']):.1f} mm）；"
                "上颌骨.1不能作为关节窝参考，请换用与全牙列同时导出的原始 STL。"
            )
    try:
        scene.delta = validated_rigid_transform(data["transforms"]["T_DELTA"]["matrix"], tolerance=1e-4)
    except (KeyError, ValueError, TypeError) as exc:
        scene.warnings.append(f"T_DELTA 不可用，不能生成第二位置模型：{exc}")
    for key, title, color in MODEL_SPECS:
        is_input = key in ("baseline_lower", "baseline_upper")
        record = data.get("inputs" if is_input else "outputs", {}).get(key)
        legacy_dentition_t1 = False
        if key == "ct_dentition_t1" and not isinstance(record, dict):
            fallback = data.get("outputs", {}).get("ct_dentition_t0")
            if isinstance(fallback, dict) and scene.delta is not None:
                record = fallback
                legacy_dentition_t1 = True
        if not isinstance(record, dict) or not record.get("path"):
            if key not in OPTIONAL_MODEL_KEYS and (keys is None or key in keys):
                scene.warnings.append(f"未收录：{title}")
            continue
        if keys is not None and key not in keys:
            scene.deferred_keys.add(key)
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            path = project_path.parent / path
        # Exported models travel with the result folder. Original scans do not.
        local = project_path.parent / "meshes" / path.name
        if not is_input and local.is_file():
            path = local
        try:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if record.get("sha256") and digest != record["sha256"]:
                raise ValueError("文件内容与配准时的哈希不符，请恢复原文件")
            mesh = load_mesh_arrays(path) if array_meshes else load_mesh(path)[0]
            if legacy_dentition_t1:
                if array_meshes:
                    vertices, triangles, normals = mesh
                    vertices = vertices @ scene.delta[:3, :3].T + scene.delta[:3, 3]
                    if normals is not None:
                        normals = normals @ scene.delta[:3, :3].T
                    mesh = vertices, triangles, normals
                else:
                    mesh = o3d.geometry.TriangleMesh(mesh)
                    mesh.transform(scene.delta)
        except (OSError, ValueError, RuntimeError) as exc:
            scene.warnings.append(f"未加载 {title}：{exc}")
            continue
        scene.models.append(SceneModel(key, title, path.resolve(), mesh, color, digest))
    if not scene.models:
        raise ValueError("没有可读取的模型。\n" + "\n".join(scene.warnings))
    return scene


def point_array(points, count: int | None = None) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("点坐标必须是有限的 N×3 数组。")
    if count is not None and len(result) != count:
        raise ValueError(f"需要 {count} 个点。")
    return result


def length_mm(points) -> float:
    p = point_array(points, 2)
    return float(np.linalg.norm(p[1] - p[0]))


def angle_degrees(points) -> float:
    p = point_array(points, 3)
    a, b = p[0] - p[1], p[2] - p[1]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        raise ValueError("角度的两条边必须有长度；第二个点是角的顶点。")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))))


def segment_angle_degrees(points) -> float:
    """Angle between two ordered segments, one drawn on each mandible."""
    p = point_array(points, 4)
    a, b = p[1] - p[0], p[3] - p[2]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        raise ValueError("两条线都必须有长度，请重新选择当前线段的终点。")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))))


def bone_pick_keys(mode: str, anchors: list[dict]) -> tuple[str, ...]:
    """The first click chooses a bone; subsequent clicks enforce cross-bone pairs."""
    if mode not in ("point", "length", "angle"):
        return ()
    if not anchors:
        return BONE_KEYS
    first = anchors[0]["model"]
    if first not in BONE_KEYS:
        return ()
    other = next(key for key in BONE_KEYS if key != first)
    if mode == "length" and len(anchors) == 1:
        return (other,)
    if mode == "angle":
        if len(anchors) == 1:
            return (first,)
        if len(anchors) in (2, 3):
            return (other,)
    return ()
