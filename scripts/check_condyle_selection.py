"""Headless QA for the public multi-region selection contract."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import open3d as o3d

from auto_alignment.integration.selection import RegionSelectionSession
from mandible_registration.condyle_selection import CondyleSelectionContext, REGION_SPECS
from mandible_registration.exporter import write_mesh


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh.create_sphere(8, 24).translate((-14, 0, 0))
    mesh += o3d.geometry.TriangleMesh.create_sphere(8, 24).translate((14, 0, 0))
    mesh.compute_vertex_normals()
    mesh_path = write_mesh(output / "synthetic_qa.stl", mesh)
    context = CondyleSelectionContext.load(mesh_path, output / "synthetic_selection.json")
    triangles = np.asarray(context.mesh.triangles)
    centers_x = np.asarray(context.mesh.vertices)[triangles].mean(axis=1)[:, 0]
    session = RegionSelectionSession(
        mesh_path,
        context.triangle_count,
        REGION_SPECS,
        allow_overlap=False,
        on_change=context.persist_profile,
        on_save=context.persist_profile,
    )
    session.set_mask("left", centers_x < 0)
    session.select_region("right")
    session.set_mask("right", centers_x > 0)
    before = session.snapshot().masks["right"]
    session.undo()
    assert not session.snapshot().masks["right"].any()
    session.redo()
    assert np.array_equal(before, session.snapshot().masks["right"])
    try:
        session.set_mask("right", np.ones(context.triangle_count, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping masks must be rejected")
    session.save(context.state_path)
    assert context.selection_path.is_file() and context.state_path.is_file()
    print("PASS: public region session, separate masks, undo/redo and profile persistence.")


if __name__ == "__main__":
    main()
