from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from mandible_registration.models import INPUT_SPECS, StudyInputs
from mandible_registration import workflow
from auto_alignment.integration.core import RegistrationMetrics, RegistrationResult


def _make_inputs(root: Path) -> StudyInputs:
    paths: dict[str, Path] = {}
    for index, spec in enumerate(INPUT_SPECS):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(
                np.array(
                    [
                        [index, 0.0, 0.0],
                        [index + 1.0, 0.0, 0.0],
                        [index, 1.0, 0.0],
                    ]
                )
            ),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
        )
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        path = root / f"{index}.stl"
        assert o3d.io.write_triangle_mesh(str(path), mesh)
        paths[spec.key] = path
    return StudyInputs.from_mapping(paths)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def _result(matrix: np.ndarray, *, status: str = "success") -> RegistrationResult:
    return RegistrationResult(
        transformation=matrix,
        status=status,
        confidence="高" if status == "success" else "低",
        metrics=RegistrationMetrics(
            fitness=1.0,
            inlier_rmse_mm=0.01,
            correspondence_count=3,
            overlap_ratio=1.0,
            rotation_degrees=0.0,
            translation_mm=float(np.linalg.norm(matrix[:3, 3])),
        ),
        warnings=() if status == "success" else ("测试失败",),
        elapsed_seconds=0.01,
        quality=None,
    )


def _facts(bounds_min, bounds_max, path="mesh.stl"):
    from auto_alignment.integration.core import MeshFacts
    extent = np.asarray(bounds_max, dtype=float) - np.asarray(bounds_min, dtype=float)
    return MeshFacts(path, 1000, 2000, float(np.linalg.norm(extent)),
                     tuple(bounds_min), tuple(bounds_max), ())


def test_ct_frame_guard_distinguishes_shared_export_from_aligned_intermediate():
    mandible = _facts((24.98, 36.48, 10.23), (157.50, 120.02, 101.27), "bone.stl")
    original = _facts((52.23, 29.23, 29.73), (123.67, 88.49, 90.02), "teeth.stl")
    aligned = _facts((-28.38, -23.52, -30.94), (41.74, 41.74, 29.50), "aligned_current.stl")
    assert workflow.ct_frame_compatibility(original, mandible)["compatible"]
    report = workflow.ct_frame_compatibility(aligned, mandible)
    assert not report["compatible"]
    with pytest.raises(workflow.WorkflowError, match="aligned_current.*未经单独移动"):
        workflow._require_shared_ct_frame(aligned, mandible)


def test_workflow_composes_and_audits_outputs(tmp_path: Path, monkeypatch) -> None:
    inputs = _make_inputs(tmp_path)
    from mandible_registration import condyles
    mandible = o3d.geometry.TriangleMesh.create_box()
    mandible.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(inputs.ct_mandible), mandible)
    mesh, _ = workflow.load_mesh(inputs.ct_mandible)
    selection_path = tmp_path / "selection.json"
    selection = condyles.build_profile(mesh, inputs.ct_mandible, workflow.sha256_file(inputs.ct_mandible),
                                      {"left": np.arange(len(mesh.triangles)) < 2,
                                       "right": np.arange(len(mesh.triangles)) >= len(mesh.triangles) - 2})
    workflow.write_json(selection_path, selection)
    monkeypatch.setattr(condyles, "profile_path", lambda path: selection_path)
    transforms = iter(
        (
            _translation(1.0, 0.0, 0.0),
            _translation(1.0, 0.0, 0.0),
            _translation(0.0, 2.0, 0.0),
            _translation(0.0, 0.0, 3.0),
        )
    )
    monkeypatch.setattr(workflow, "_ct_attempt_is_strong", lambda result: result.succeeded)
    monkeypatch.setattr(
        workflow,
        "register_meshes",
        lambda *_args, **_kwargs: _result(next(transforms)),
    )

    outcome = workflow.run_study(inputs, tmp_path / "outputs")

    assert np.allclose(outcome.t_mandible_t1, outcome.t_delta @ outcome.t_ct)
    assert (outcome.run_directory / "stages" / "01_T_CT.json").is_file()
    project = json.loads((outcome.run_directory / "project.json").read_text("utf-8"))
    assert project["schema_version"] == 3
    assert project["inputs"]["ct_mandible"]["mesh_facts"]["triangles"] == 12
    assert len(project["outputs"]["ct_mandible_t1"]["sha256"]) == 64
    assert outcome.output_files["ct_dentition_t1"].is_file()
    assert len(project["outputs"]["ct_dentition_t1"]["sha256"]) == 64
    assert outcome.output_files["ct_maxilla_t0"].is_file()
    assert "ct_maxilla_t1" not in outcome.output_files
    np.testing.assert_allclose(
        project["transforms"]["T_MAXILLA_T0"]["matrix"],
        project["transforms"]["T_CT"]["matrix"],
    )
    assert project["condyle_selection"] == selection
    assert condyles.project_analysis(outcome.output_files["project"])["regions"]["left"]["distance_mm"] == pytest.approx(3)


def test_optional_priority_faces_apply_only_to_ct_stage(tmp_path, monkeypatch):
    from mandible_registration import registration_selection

    inputs = _make_inputs(tmp_path)
    masks = {
        "baseline_lower": np.array([True]),
        "ct_dentition": np.array([True]),
    }
    profiles = {
        key: {"role": key, "selected_triangle_count": 1}
        for key in masks
    }
    monkeypatch.setattr(
        registration_selection,
        "load_priority_mask",
        lambda _path, role, _count: (masks[role], profiles[role]),
    )
    calls = []

    def register(*_args, **kwargs):
        calls.append(kwargs)
        return _result(np.eye(4))

    monkeypatch.setattr(workflow, "_ct_attempt_is_strong", lambda result: result.succeeded)
    monkeypatch.setattr(workflow, "register_meshes", register)
    outcome = workflow.run_study(inputs, tmp_path / "outputs")
    assert len(calls) == 4
    for kwargs in calls[:2]:
        np.testing.assert_array_equal(kwargs["target_priority_faces"], masks["baseline_lower"])
        np.testing.assert_array_equal(kwargs["source_priority_faces"], masks["ct_dentition"])
    assert all("target_priority_faces" not in kwargs and "source_priority_faces" not in kwargs
               for kwargs in calls[2:])
    project = json.loads(outcome.output_files["project"].read_text("utf-8"))
    assert project["registration_selections"] == profiles


def test_workflow_without_optional_maxilla_keeps_original_six_mesh_behavior(tmp_path, monkeypatch):
    values = _make_inputs(tmp_path).as_mapping()
    values.pop("ct_maxilla")
    inputs = StudyInputs.from_mapping(values)
    transforms = iter((np.eye(4), np.eye(4), np.eye(4), np.eye(4)))
    monkeypatch.setattr(workflow, "_ct_attempt_is_strong", lambda result: result.succeeded)
    monkeypatch.setattr(
        workflow,
        "register_meshes",
        lambda *_args, **_kwargs: _result(next(transforms)),
    )

    outcome = workflow.run_study(inputs, tmp_path / "outputs")
    project = json.loads(outcome.output_files["project"].read_text("utf-8"))

    assert "ct_maxilla" not in project["inputs"]
    assert "ct_maxilla_t0" not in project["outputs"]
    assert "T_MAXILLA_T0" not in project["transforms"]


def test_failed_stage_leaves_a_diagnostic_run(tmp_path: Path, monkeypatch) -> None:
    inputs = _make_inputs(tmp_path)
    monkeypatch.setattr(
        workflow,
        "register_meshes",
        lambda *_args, **_kwargs: _result(np.eye(4), status="failed"),
    )

    updates = []
    with pytest.raises(workflow.WorkflowError, match="诊断记录"):
        workflow.run_study(inputs, tmp_path / "outputs", stage_changed=updates.append)

    assert [(event.key, event.status) for event in updates] == [("T_CT", "running")]
    assert not any(event.ready_outputs for event in updates)

    runs = list((tmp_path / "outputs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "failure.json").is_file()
    assert (runs[0] / "stages" / "01_T_CT_attempt_1.json").is_file()
    assert (runs[0] / "stages" / "01_T_CT_consensus.json").is_file()
    preview = json.loads((runs[0] / "stage_views" / "T_CT" / "results.json").read_text("utf-8"))
    assert preview["review_only"] and preview["registration"]["status"] == "failed"
    assert not (runs[0] / "meshes").exists()


@pytest.mark.parametrize("ct_status", ["success", "warning"])
def test_stage_events_publish_quality_and_ready_models_before_next_stage(tmp_path, monkeypatch, ct_status):
    inputs = _make_inputs(tmp_path)
    updates = []
    calls = []

    def register(*args):
        calls.append(len(calls) + 1)
        expected = {
            1: [("T_CT", "running")],
            2: [("T_CT", "running")],  # One attempt is not a consensus.
            3: [("T_CT", "running"), ("T_CT", ct_status), ("T_UPPER", "running")],
            4: [("T_CT", "running"), ("T_CT", ct_status), ("T_UPPER", "running"),
                ("T_UPPER", "success"), ("T_DELTA", "running")],
        }
        assert [(event.key, event.status) for event in updates] == expected[len(calls)]
        return _result(np.eye(4), status=ct_status if len(calls) <= 2 else "success")

    def changed(event):
        if event.status != "running":
            assert event.review_path.is_file()
            payload = json.loads(event.review_path.read_text("utf-8"))
            assert payload["registration"]["status"] == event.status
            assert not payload["review_only"]
            assert (event.review_path.parent / "aligned.stl").is_file()
        updates.append(event)

    monkeypatch.setattr(workflow, "_ct_attempt_is_strong", lambda result: result.succeeded)
    monkeypatch.setattr(workflow, "register_meshes", register)
    workflow.run_study(inputs, tmp_path / "outputs", stage_changed=changed)
    assert len(updates) == 6
    assert "ct_mandible_t0" in updates[1].ready_outputs
    assert "followup_lower_in_t0" in updates[3].ready_outputs
    assert updates[-1].key == "T_DELTA" and updates[-1].status == "success"
    # Final comparison is enabled only once the complete project is saved.
    assert not any("ct_mandible_t1" in event.ready_outputs for event in updates)


@pytest.mark.parametrize("failed_stage", ["T_UPPER", "T_DELTA"])
@pytest.mark.parametrize("status", ["warning", "failed"])
def test_rejected_stages_never_publish_ready_outputs(tmp_path, monkeypatch, failed_stage, status):
    inputs = _make_inputs(tmp_path)
    results = [_result(np.eye(4)), _result(np.eye(4))]
    if failed_stage == "T_DELTA":
        results.append(_result(np.eye(4)))
    results.append(_result(np.eye(4), status=status))
    iterator = iter(results)
    updates = []
    monkeypatch.setattr(workflow, "_ct_attempt_is_strong", lambda result: result.succeeded)
    monkeypatch.setattr(workflow, "register_meshes", lambda *args: next(iterator))
    with pytest.raises(workflow.WorkflowError):
        workflow.run_study(inputs, tmp_path / "outputs", stage_changed=updates.append)
    assert updates[-1].key == failed_stage and updates[-1].status == "running"
    assert not updates[-1].ready_outputs
    assert updates[1].key == "T_CT" and updates[1].status == "success"
    report = updates[-1].run_directory / "stage_views" / failed_stage / "results.json"
    assert json.loads(report.read_text("utf-8"))["review_only"]
