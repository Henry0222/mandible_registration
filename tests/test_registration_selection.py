import json

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.integration.selection import RegionSelectionSession
from mandible_registration.exporter import sha256_file, write_mesh
from mandible_registration.registration_selection import (
    REGION_SPECS,
    RegistrationSelectionContext,
    build_profile,
    read_profile,
)


def _masks(triangle_count):
    return {
        "priority": np.arange(triangle_count) < 2,
    }


def _write_box(path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    return write_mesh(path, mesh)


def test_priority_profile_round_trip_and_validation(tmp_path):
    mesh_path = _write_box(tmp_path / "lower.stl")
    triangle_count = 12
    digest = sha256_file(mesh_path)
    profile = build_profile(mesh_path, digest, triangle_count, "baseline_lower", _masks(triangle_count))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    restored, union = read_profile(path, digest, triangle_count, "baseline_lower")
    assert restored["selected_triangle_count"] == 2
    assert union.sum() == 2
    with pytest.raises(ValueError, match="用途"):
        read_profile(path, digest, triangle_count, "ct_dentition")
    with pytest.raises(ValueError, match="不属于"):
        read_profile(path, "0" * 64, triangle_count, "baseline_lower")


def test_public_selection_session_saves_one_region_without_modifying_stl(tmp_path):
    mesh_path = _write_box(tmp_path / "teeth.stl")
    selection_path = tmp_path / "priority.json"
    context = RegistrationSelectionContext.load(mesh_path, "ct_dentition", selection_path)
    session = RegionSelectionSession(
        mesh_path,
        context.triangle_count,
        REGION_SPECS,
        initial_masks=context.initial_masks,
        allow_overlap=False,
        on_change=None,
        on_save=context.persist_complete_profile,
    )
    original = mesh_path.read_bytes()
    assert [spec.key for spec in REGION_SPECS] == ["priority"]
    session.set_mask("priority", _masks(context.triangle_count)["priority"])
    session.save(context.state_path)
    payload = json.loads(selection_path.read_text("utf-8"))
    assert payload["role"] == "ct_dentition"
    assert payload["selected_triangle_count"] == 2
    assert mesh_path.read_bytes() == original
    restored = RegistrationSelectionContext.load(mesh_path, "ct_dentition", selection_path)
    assert restored.initial_masks["priority"].sum() == 2


def test_old_a_b_profile_migrates_to_single_union(tmp_path):
    mesh_path = _write_box(tmp_path / "teeth.stl")
    digest = sha256_file(mesh_path)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "registration_priority_faces",
        "role": "ct_dentition",
        "mesh_sha256": digest,
        "mesh_triangle_count": 12,
        "selected_ranges": [[0, 1], [10, 11]],
        "selected_triangle_count": 4,
        "regions": {
            "posterior": {"selected_ranges": [[0, 1]]},
            "anterior": {"selected_ranges": [[10, 11]]},
        },
    }), encoding="utf-8")
    restored = RegistrationSelectionContext.load(mesh_path, "ct_dentition", path)
    assert set(restored.initial_masks) == {"priority"}
    assert restored.initial_masks["priority"].sum() == 4


def test_priority_selection_refuses_empty_save(tmp_path):
    mesh_path = _write_box(tmp_path / "lower.stl")
    context = RegistrationSelectionContext.load(mesh_path, "baseline_lower", tmp_path / "empty.json")
    session = RegionSelectionSession(
        mesh_path,
        context.triangle_count,
        REGION_SPECS,
        on_save=context.persist_complete_profile,
    )
    with pytest.raises(ValueError, match="至少选择"):
        session.save(context.state_path)
