import json

import numpy as np
import open3d as o3d
import pytest

from mandible_registration.condyles import analyze_motion, build_profile, project_analysis, surface_centroid, validate_profile


def sample_profile(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box(2, 3, 4)
    masks = {"left": np.arange(12) < 2, "right": np.arange(12) >= 10}
    return build_profile(mesh, tmp_path / "bone.stl", "a" * 64, masks)


def test_surface_center_uses_area_not_vertex_count():
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0, 0, 0], [2, 0, 0], [0, 2, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]]),
                                    o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
    center, area = surface_centroid(mesh, [True, True])
    expected = (np.array([2 / 3, 2 / 3, 0]) * 2 + np.array([31 / 3, 1 / 3, 0]) * .5) / 2.5
    np.testing.assert_allclose(center, expected)
    assert area == 2.5
    with pytest.raises(ValueError):
        surface_centroid(mesh, [False, False])


def test_region_profile_rejects_overlap_and_wrong_bone(tmp_path):
    profile = sample_profile(tmp_path)
    assert set(profile["regions"]) == {"left", "right"}
    validate_profile(profile, "a" * 64)
    with pytest.raises(ValueError, match="不属于"):
        validate_profile(profile, "b" * 64)
    mesh = o3d.geometry.TriangleMesh.create_box()
    with pytest.raises(ValueError, match="相同面片"):
        build_profile(mesh, tmp_path / "bone.stl", "a" * 64, {key: np.ones(12, dtype=bool) for key in ("left", "right")})


def test_center_motion_applies_both_matrices_in_original_ct_frame(tmp_path):
    profile = sample_profile(tmp_path)
    t0 = np.eye(4)
    t0[:3, 3] = [20, 30, 40]
    delta = np.eye(4)
    delta[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    delta[:3, 3] = [3, 4, 5]
    report = analyze_motion(profile, t0, delta @ t0)
    assert report["rigid_rotation_degrees"] == pytest.approx(90)
    for key, region in report["regions"].items():
        first = np.asarray(profile["regions"][key]["center_ct_mm"]) + [20, 30, 40]
        second = delta[:3, :3] @ first + delta[:3, 3]
        np.testing.assert_allclose(region["center_t0_mm"], first)
        np.testing.assert_allclose(region["center_t1_mm"], second)
        np.testing.assert_allclose(region["displacement_xyz_mm"], second - first)
        assert region["distance_mm"] == pytest.approx(np.linalg.norm(second - first))
    still = analyze_motion(profile, t0, t0)
    assert still["rigid_rotation_degrees"] == 0
    assert all(region["direction_unit_xyz"] is None for region in still["regions"].values())


def test_archived_selection_survives_missing_original_and_stale_selection_rejected(tmp_path, monkeypatch):
    from mandible_registration import condyles
    selection_path = tmp_path / "current.json"
    monkeypatch.setattr(condyles, "profile_path", lambda path: selection_path)
    profile = sample_profile(tmp_path)
    identity = {"matrix": np.eye(4).tolist()}
    payload = {"inputs": {"ct_mandible": {"path": "missing.stl", "sha256": "a" * 64}},
               "transforms": {"T_CT": identity, "T_DELTA": identity}, "condyle_selection": profile}
    project = tmp_path / "project.json"
    project.write_text(json.dumps(payload), encoding="utf-8")
    assert len(project_analysis(project)["regions"]) == 2
    profile["mesh_sha256"] = "b" * 64
    selection_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="不属于"):
        project_analysis(project)


def test_project_analysis_accepts_a_single_selected_condyle(tmp_path, monkeypatch):
    from mandible_registration import condyles

    selection_path = tmp_path / "current.json"
    monkeypatch.setattr(condyles, "profile_path", lambda path: selection_path)
    profile = sample_profile(tmp_path)
    del profile["regions"]["right"]
    selection_path.write_text(json.dumps(profile), encoding="utf-8")
    identity = {"matrix": np.eye(4).tolist()}
    project = tmp_path / "project.json"
    project.write_text(json.dumps({
        "inputs": {"ct_mandible": {"path": "missing.stl", "sha256": "a" * 64}},
        "transforms": {"T_CT": identity, "T_DELTA": identity},
    }), encoding="utf-8")
    report = project_analysis(project)
    assert set(report["regions"]) == {"left"}


def test_condyle_editor_undo_redo_and_separate_sides_without_writing_stl(tmp_path):
    from auto_alignment.integration.selection import RegionSelectionSession
    from mandible_registration.condyle_selection import CondyleSelectionContext, REGION_SPECS
    from mandible_registration.exporter import write_mesh

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    mesh_path = write_mesh(tmp_path / "bone.stl", mesh)
    context = CondyleSelectionContext.load(mesh_path, tmp_path / "selection.json")
    session = RegionSelectionSession(
        mesh_path,
        context.triangle_count,
        REGION_SPECS,
        initial_masks=context.initial_masks,
        allow_overlap=False,
        on_change=None,
        on_save=context.persist_profile,
    )
    original = mesh_path.read_bytes()
    session.set_mask("left", np.arange(context.triangle_count) < 2)
    session.select_region("right")
    session.set_mask("right", np.arange(context.triangle_count) >= context.triangle_count - 2)
    session.undo()
    snapshot = session.snapshot()
    assert not snapshot.masks["right"].any() and snapshot.masks["left"].sum() == 2
    session.redo()
    assert session.snapshot().masks["right"].sum() == 2
    with pytest.raises(ValueError, match="重叠"):
        session.set_mask("right", np.ones(context.triangle_count, dtype=bool))
    assert session.snapshot().masks["right"].sum() == 2
    session.save(context.state_path)
    assert len(json.loads(context.selection_path.read_text("utf-8"))["regions"]) == 2
    restored = CondyleSelectionContext.load(mesh_path, context.selection_path)
    assert restored.initial_masks["left"].sum() == 2
    assert restored.initial_masks["right"].sum() == 2
    assert mesh_path.read_bytes() == original


def test_condyle_save_accepts_one_side_and_requests_gui_confirmation(tmp_path):
    from auto_alignment.integration.selection import RegionSelectionSession
    from mandible_registration.condyle_selection import (
        CondyleSelectionContext,
        REGION_SPECS,
        needs_single_side_confirmation,
        selected_condyle_sides,
    )
    from mandible_registration.exporter import write_mesh

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    mesh_path = write_mesh(tmp_path / "bone.stl", mesh)
    context = CondyleSelectionContext.load(mesh_path, tmp_path / "selection.json")
    session = RegionSelectionSession(
        mesh_path,
        context.triangle_count,
        REGION_SPECS,
        allow_overlap=False,
        on_change=None,
        on_save=context.persist_profile,
    )
    session.set_mask("left", np.arange(context.triangle_count) < 2)
    assert selected_condyle_sides(session.snapshot()) == ("left",)
    assert needs_single_side_confirmation(session.snapshot())
    session.save(context.state_path)
    assert not session.dirty
    assert set(json.loads(context.selection_path.read_text("utf-8"))["regions"]) == {"left"}
    assert "右侧髁突：未选择（0 面）" in context.summary(session.snapshot())
