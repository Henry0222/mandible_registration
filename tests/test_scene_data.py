import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from mandible_registration import scene_data
from mandible_registration.scene_data import (
    angle_degrees, length_mm, BONE_KEYS, bone_pick_keys, segment_angle_degrees,
)


def test_measurement_geometry():
    assert length_mm([[0, 0, 0], [3, 4, 0]]) == 5
    assert angle_degrees([[1, 0, 0], [0, 0, 0], [0, 1, 0]]) == 90
    assert angle_degrees([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]) == 180
    with pytest.raises(ValueError):
        angle_degrees([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        length_mm([[0, 0, 0], [np.nan, 0, 0]])


def test_scene_load_only_common_frame_models_and_relocated_outputs(tmp_path, monkeypatch):
    folder = tmp_path / "meshes"
    folder.mkdir()
    path = folder / "ct_mandible_T0.stl"
    path.write_bytes(b"fake model for loader")
    record = {"path": str(tmp_path / "old" / path.name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    project = tmp_path / "project.json"
    payload = {"workflow": "mandibular_pose_transfer", "inputs": {"ct_mandible": record},
               "outputs": {"ct_mandible_t0": record}, "transforms": {"T_DELTA": {"matrix": np.eye(4).tolist()}}}
    project.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(scene_data, "load_mesh", lambda path: (o3d.geometry.TriangleMesh.create_box(), None))
    scene = scene_data.load_scene(project)
    assert [m.key for m in scene.models] == ["ct_mandible_t0"]
    assert scene.models[0].path == path
    assert len(scene.warnings) == 7
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        scene_data.load_scene(project)


def test_legacy_project_derives_second_dentition_with_delta(tmp_path, monkeypatch):
    folder = tmp_path / "meshes"
    folder.mkdir()
    path = folder / "ct_dentition_T0.stl"
    path.write_bytes(b"legacy dentition")
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    delta = np.eye(4)
    delta[:3, 3] = [5, -2, 3]
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "workflow": "mandibular_pose_transfer",
        "outputs": {"ct_dentition_t0": record},
        "transforms": {"T_DELTA": {"matrix": delta.tolist()}},
    }), encoding="utf-8")
    monkeypatch.setattr(scene_data, "load_mesh", lambda _path: (o3d.geometry.TriangleMesh.create_box(), None))
    scene = scene_data.load_scene(project, keys={"ct_dentition_t0", "ct_dentition_t1"})
    models = {model.key: model for model in scene.models}
    assert set(models) == {"ct_dentition_t0", "ct_dentition_t1"}
    np.testing.assert_allclose(models["ct_dentition_t1"].mesh.get_min_bound(), [5, -2, 3])
    np.testing.assert_allclose(models["ct_dentition_t0"].mesh.get_min_bound(), [0, 0, 0])


def test_old_result_warns_when_ct_input_relative_coordinates_were_lost(tmp_path, monkeypatch):
    folder = tmp_path / "meshes"
    folder.mkdir()
    path = folder / "ct_mandible_T0.stl"
    path.write_bytes(b"mesh")
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    inputs = {
        "ct_dentition": {"mesh_facts": {
            "bounds_min": [-28.38, -23.52, -30.94], "bounds_max": [41.74, 41.74, 29.50],
            "diagonal_mm": 113.26,
        }},
        "ct_mandible": {"mesh_facts": {
            "bounds_min": [24.98, 36.48, 10.23], "bounds_max": [157.50, 120.02, 101.27],
            "diagonal_mm": 181.19,
        }},
    }
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "workflow": "mandibular_pose_transfer",
        "inputs": inputs,
        "outputs": {"ct_mandible_t0": record},
        "transforms": {"T_DELTA": {"matrix": np.eye(4).tolist()}},
    }), encoding="utf-8")
    monkeypatch.setattr(scene_data, "load_mesh", lambda _path: (o3d.geometry.TriangleMesh.create_box(), None))
    scene = scene_data.load_scene(project, keys={"ct_mandible_t0"})
    assert any("不能用于髁突变化判断" in warning for warning in scene.warnings)


def test_fast_reader_preserves_triangle_geometry_and_unicode_path(tmp_path):
    path = tmp_path / "颌骨.stl"
    mesh = o3d.geometry.TriangleMesh.create_box(2, 3, 4)
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(path), mesh)
    loaded, _ = scene_data.load_mesh(path)
    np.testing.assert_allclose(loaded.get_min_bound(), mesh.get_min_bound())
    np.testing.assert_allclose(loaded.get_max_bound(), mesh.get_max_bound())
    assert len(loaded.triangles) == len(mesh.triangles)
    assert loaded.has_vertex_normals()


def test_cross_bone_angle_geometry():
    assert segment_angle_degrees([[0, 0, 0], [1, 0, 0], [5, 5, 5], [5, 6, 5]]) == 90
    assert segment_angle_degrees([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 5, 5]]) == 0
    assert segment_angle_degrees([[0, 0, 0], [1, 0, 0], [5, 5, 5], [4, 5, 5]]) == 180
    with pytest.raises(ValueError):
        segment_angle_degrees([[0, 0, 0], [0, 0, 0], [5, 5, 5], [4, 5, 5]])


@pytest.mark.parametrize("first", BONE_KEYS)
def test_pick_sequence_automatically_uses_both_bones(first):
    other = next(key for key in BONE_KEYS if key != first)
    assert bone_pick_keys("length", []) == BONE_KEYS
    assert bone_pick_keys("length", [{"model": first}]) == (other,)
    assert bone_pick_keys("angle", [{"model": first}]) == (first,)
    assert bone_pick_keys("angle", [{"model": first}] * 2) == (other,)
    assert bone_pick_keys("angle", [{"model": first}] * 2 + [{"model": other}]) == (other,)
    assert not bone_pick_keys("browse", [])
