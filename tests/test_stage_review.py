import json
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from mandible_registration.exporter import write_mesh, write_json, sha256_file
from mandible_registration.stage_review import export_stage_review, read_stage_reviews, show_stage_review, prepare_project_stage
from auto_alignment.integration.review import load_viewer_data


@pytest.mark.parametrize("status,accepted,expected", (("success", True, "success"), ("warning", True, "warning"), ("success", False, "failed")))
def test_stage_adapter_loads_in_real_general_viewer(tmp_path, status, accepted, expected):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    target_path = write_mesh(tmp_path / "input.stl", mesh)
    result = SimpleNamespace(transformation=np.eye(4), status=status, confidence="高", warnings=())
    path = export_stage_review(tmp_path, "T_CT", target_path, mesh, result, accepted=accepted)
    reviews, states, _ = read_stage_reviews(tmp_path)
    assert reviews["T_CT"] == path
    assert states["T_CT"] == expected
    loaded = load_viewer_data(path)
    assert loaded.registration_status == expected
    assert loaded.review_only == (not accepted)
    assert np.allclose(loaded.signed_distances_mm, 0)
    assert loaded.annotations_path.parent == path.parent


def test_rejects_changed_stage_mesh_before_opening_viewer(tmp_path, monkeypatch):
    from mandible_registration import stage_review

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    target = write_mesh(tmp_path / "input.stl", mesh)
    result = SimpleNamespace(transformation=np.eye(4), status="success", confidence="高", warnings=())
    path = export_stage_review(tmp_path, "T_CT", target, mesh, result)
    opened = []
    monkeypatch.setattr(stage_review, "run_general_result_viewer", lambda path, **_kwargs: opened.append(path))
    show_stage_review(path)
    assert opened == [path]
    payload = json.loads(path.read_text(encoding="utf-8"))
    (path.parent / payload["outputs"]["aligned_current_stl"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="模型内容已改变"):
        show_stage_review(path)
    assert opened == [path]


def test_historical_project_adapter_preserves_stage_direction_after_move(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    inputs, outputs = {}, {}
    for key in ("baseline_lower", "baseline_upper"):
        path = write_mesh(tmp_path / f"{key}.stl", mesh)
        inputs[key] = {"path": path.name, "sha256": sha256_file(path)}
    for key in ("ct_dentition_t0", "followup_upper_in_t0", "followup_lower_in_t0", "baseline_lower_at_t1"):
        path = write_mesh(tmp_path / "meshes" / f"{key}.stl", mesh)
        outputs[key] = {"path": str(tmp_path / "old_location" / path.name), "sha256": sha256_file(path)}
    project = write_json(tmp_path / "project.json", {
        "workflow": "mandibular_pose_transfer", "inputs": inputs, "outputs": outputs,
        "stages": [{"key": key, "status": "success", "confidence": "高"} for key in ("T_CT", "T_UPPER", "T_DELTA")],
    })
    for key, fixed, moving in (("T_CT", "baseline_lower", "ct_dentition_t0"), ("T_UPPER", "baseline_upper", "followup_upper_in_t0"), ("T_DELTA", "followup_lower_in_t0", "baseline_lower_at_t1")):
        adapter = prepare_project_stage(project, key)
        data = load_viewer_data(adapter)
        assert data.target_path.stem == fixed
        assert data.aligned_path.stem == moving
        assert not data.review_only
        assert prepare_project_stage(project, key) == adapter
