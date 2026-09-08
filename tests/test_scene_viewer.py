from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import vtk
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtTest import QTest

from mandible_registration.scene_data import BONE_KEYS, COMPARISON_KEYS, CT_INSPECTION_KEYS, SceneData, SceneModel
from mandible_registration.scene_viewer import SceneViewer, mesh_polydata


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    return instance


@pytest.fixture
def viewer(app, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window = SceneViewer(offscreen=True)
    window.show()
    window.vtk_widget.Initialize()
    first = o3d.geometry.TriangleMesh.create_box(10, 10, 2)
    first.compute_vertex_normals()
    second = o3d.geometry.TriangleMesh(first).translate((0, 0, 3))
    delta = np.eye(4)
    delta[2, 3] = 3
    scene = SceneData(tmp_path / "project.json", [
        SceneModel("ct_mandible_t0", "Bone T0", tmp_path / "t0.stl", first, (0.2, 0.6, 0.9)),
        SceneModel("ct_mandible_t1", "Bone T1", tmp_path / "t1.stl", second, (1, 0.5, 0.1)),
        SceneModel("baseline_lower", "Lower IOS", tmp_path / "ios.stl", first, (0.2, 0.9, 0.4)),
        SceneModel("followup_upper_in_t0", "Upper IOS T1", tmp_path / "upper1.stl", first, (0.7, 0.9, 0.9)),
        SceneModel("followup_lower_in_t0", "Lower IOS T1", tmp_path / "lower1.stl", first, (0.7, 0.9, 0.9)),
        SceneModel("ct_dentition_t0", "CT Teeth", tmp_path / "teeth.stl", first, (0.7, 0.6, 0.9)),
        SceneModel("ct_dentition_t1", "CT Teeth T1", tmp_path / "teeth1.stl", second, (0.9, 0.5, 0.8)),
        SceneModel("baseline_lower_at_t1", "Unneeded transformed IOS", tmp_path / "extra.stl", first, (1, 1, 1)),
    ], delta, deferred_keys={"baseline_upper"})
    window.set_scene(scene)
    app.processEvents()
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_polydata_preserves_geometry():
    mesh = o3d.geometry.TriangleMesh.create_box()
    data = mesh_polydata(mesh)
    assert data.GetNumberOfPoints() == len(mesh.vertices)
    assert data.GetNumberOfCells() == len(mesh.triangles)
    assert data.GetPoint(1) == tuple(np.asarray(mesh.vertices)[1])

def test_viewer_starts_with_only_bones_visible_and_has_no_rainbow_controls(viewer):
    assert viewer.actors["ct_mandible_t0"].GetVisibility()
    assert viewer.actors["ct_mandible_t1"].GetVisibility()
    assert not viewer.actors["baseline_lower"].GetVisibility()
    viewer.model_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not viewer.actors["ct_mandible_t0"].GetVisibility()
    assert not hasattr(viewer, "map_mode")
    assert not hasattr(viewer, "scalar_bar")
    assert viewer.orientation_widget.GetEnabled()
    assert not viewer.orientation_widget.GetInteractive()
    assert viewer.orientation_axes.GetShaftType() == vtk.vtkAxesActor.LINE_SHAFT
    assert viewer.orientation_widget.GetViewport() == pytest.approx((0.84, 0.02, 0.99, 0.18))


@pytest.mark.parametrize(
    ("axis", "expected"),
    (("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
     ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)), ("+Z", (0, 0, 1))),
)
def test_signed_axis_views_place_camera_on_requested_side(viewer, axis, expected):
    viewer._axis_view(axis)
    camera = viewer.renderer.GetActiveCamera()
    direction = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(direction, expected, atol=1e-8)


def test_each_model_option_has_a_color_swatch_that_tracks_color_changes(viewer, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    assert all(not viewer.model_list.item(row).icon().isNull()
               for row in range(viewer.model_list.count()))
    item = viewer.model_list.item(0)
    viewer.model_list.setCurrentItem(item)
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#12ab34"))
    viewer._change_color()
    pixel = item.icon().pixmap(QSize(14, 14)).toImage().pixelColor(7, 7)
    assert pixel.name() == "#12ab34"


def test_automatic_picker_enforces_cross_bone_endpoints_despite_overlap(viewer):
    assert not hasattr(viewer, "pick_target")
    viewer.measure_mode.setCurrentIndex(2)
    viewer.actors["baseline_lower"].SetVisibility(True)
    viewer.actors["baseline_lower"].SetPosition(0, 0, 20)  # oral scan in front must never be picked
    viewer._axis_view("Z")
    renderer = viewer.renderer
    renderer.SetWorldPoint(5, 5, 2, 1)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    assert viewer.pick_at(x, y)
    assert viewer.pending[0]["model"] == "ct_mandible_t1"  # first: nearest bone to camera
    assert "Bone T0" in viewer.measure_hint.text()
    viewer.model_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not viewer.pick_at(x, y)  # do not silently take another point on T1
    assert len(viewer.pending) == 1
    viewer.model_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert viewer.pick_at(x, y)  # next: automatically pick T0 behind the overlapping T1
    record = viewer.measurements[0]
    assert [a["model"] for a in record["anchors"]] == ["ct_mandible_t1", "ct_mandible_t0"]
    np.testing.assert_allclose(record["anchors"][0]["xyz_mm"], [5, 5, 5], atol=1e-5)
    np.testing.assert_allclose(record["anchors"][1]["xyz_mm"], [5, 5, 2], atol=1e-5)
    assert record["value"] == pytest.approx(3)


def test_qt_mouse_events_reach_picker_once_and_ignore_drag(viewer, app):
    viewer.measure_mode.setCurrentIndex(2)
    viewer._axis_view("Z")
    viewer.renderer.SetWorldPoint(5, 5, 5, 1)
    viewer.renderer.WorldToDisplay()
    x, y, _ = viewer.renderer.GetDisplayPoint()
    width, height = viewer.render_window.GetSize()
    point = QPoint(round(x * viewer.vtk_widget.width() / width),
                   round(viewer.vtk_widget.height() - 1 - y * viewer.vtk_widget.height() / height))
    QTest.mouseClick(viewer.vtk_widget, Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert len(viewer.pending) == 1
    QTest.mouseClick(viewer.vtk_widget, Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert len(viewer.measurements) == 1
    assert viewer.measurements[0]["value"] == pytest.approx(3)
    viewer.measure_mode.setCurrentIndex(1)
    QTest.mousePress(viewer.vtk_widget, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(viewer.vtk_widget, point + QPoint(20, 10))
    QTest.mouseMove(viewer.vtk_widget, point)
    QTest.mouseRelease(viewer.vtk_widget, Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()
    assert len(viewer.measurements) == 1  # returning to the start after a drag is not a click


def test_measurements_cross_models_angle_undo_and_payload(viewer):
    viewer.measure_mode.setCurrentIndex(2)
    assert not viewer.add_anchor("baseline_lower", [0, 0, 0], 0)
    assert viewer.add_anchor("ct_mandible_t0", [0, 0, 0], 0)
    assert not viewer.add_anchor("ct_mandible_t0", [1, 0, 0], 0)
    assert viewer.add_anchor("ct_mandible_t1", [3, 4, 0], 1)
    assert viewer.measurements[0]["value"] == 5
    viewer.measure_mode.setCurrentIndex(3)
    assert viewer.add_anchor("ct_mandible_t0", [0, 0, 0], 0)
    assert not viewer.add_anchor("ct_mandible_t0", [0, 0, 0], 0)  # zero-length edge
    assert viewer.add_anchor("ct_mandible_t0", [1, 0, 0], 0)
    assert not viewer.add_anchor("ct_mandible_t0", [0, 1, 0], 0)
    assert viewer.add_anchor("ct_mandible_t1", [0, 0, 3], 0)
    assert viewer.add_anchor("ct_mandible_t1", [0, 1, 3], 0)
    assert viewer.measurements[1]["value"] == 90
    assert viewer.measurements[1]["segments"] == [[0, 1], [2, 3]]
    payload = viewer.measurement_payload()
    assert payload["coordinate_units"] == "mm"
    assert len(payload["measurements"]) == 2
    assert set(payload["models"]) == set(BONE_KEYS)
    assert payload["schema_version"] == 2
    viewer._undo()
    assert len(viewer.measurements) == 1
    viewer.add_anchor("ct_mandible_t0", [0, 0, 0], 0)
    viewer._undo()
    assert not viewer.pending


def test_model_options_ct_inspection_and_return_do_not_leak_teeth(viewer):
    def keys_in(widget):
        return {widget.item(row).data(Qt.ItemDataRole.UserRole) for row in range(widget.count())}

    assert keys_in(viewer.model_list) == set(COMPARISON_KEYS)
    assert {"ct_dentition_t0", "ct_dentition_t1"} <= keys_in(viewer.model_list)
    assert "baseline_lower_at_t1" not in viewer.models
    assert not viewer.actors["ct_dentition_t0"].GetVisibility()
    viewer.measure_mode.setCurrentIndex(2)
    viewer.add_anchor(BONE_KEYS[0], [0, 0, 0], 0)
    viewer._set_preset("ct")
    assert not viewer.pending
    assert keys_in(viewer.model_list) == set(CT_INSPECTION_KEYS)
    assert not viewer.measure_group.isEnabled()
    assert not viewer.annotation_renderer.GetDraw()
    assert viewer.actors["ct_dentition_t0"].GetVisibility()
    assert not viewer.pick_at(100, 100)
    viewer._set_preset("bones")
    assert keys_in(viewer.model_list) == set(COMPARISON_KEYS)
    assert viewer.measure_group.isEnabled()
    assert not viewer.actors["ct_dentition_t0"].GetVisibility()
    viewer._set_preset("none")
    assert not any(actor.GetVisibility() for actor in viewer.actors.values())
    viewer.measure_mode.setCurrentIndex(2)
    assert all(viewer.actors[key].GetVisibility() for key in BONE_KEYS)


def test_undo_angle_restores_automatic_target(viewer):
    viewer.measure_mode.setCurrentIndex(3)
    viewer.add_anchor(BONE_KEYS[1], [0, 0, 0], 0)
    viewer.add_anchor(BONE_KEYS[1], [1, 0, 0], 0)
    assert "Bone T0" in viewer.measure_hint.text()
    viewer._undo()
    assert "Bone T1" in viewer.measure_hint.text()


def test_measurement_export_new_file_only_and_preserves_project(viewer, tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QFileDialog

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a[2]))
    viewer.measure_mode.setCurrentIndex(1)
    viewer.add_anchor("ct_mandible_t0", [1, 2, 3], 0)
    destination = tmp_path / "measures.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
    viewer._save_measurements()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["measurements"][0]["anchors"][0]["xyz_mm"] == [1, 2, 3]
    viewer._save_measurements()
    assert warnings  # exclusive creation prevents overwriting any existing artifact
    project = viewer.scene.project_path
    project.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(project), ""))
    viewer._save_measurements()
    assert project.read_text(encoding="utf-8") == "untouched"


def test_endpoints_are_fixed_surface_circles_and_value_is_only_3d_label(viewer):
    import vtk
    viewer.measure_mode.setCurrentIndex(2)
    viewer.add_anchor(BONE_KEYS[0], [1, 2, 2], 0)
    viewer.add_anchor(BONE_KEYS[1], [1, 2, 5], 0)
    texts = [actor.GetInput() for actor in viewer._overlays if isinstance(actor, vtk.vtkBillboardTextActor3D)]
    assert texts == ["3.000 mm"]
    circles = [actor.GetMapper().GetInputAlgorithm() for actor in viewer._overlays
               if isinstance(actor, vtk.vtkActor) and isinstance(actor.GetMapper().GetInputAlgorithm(), vtk.vtkRegularPolygonSource)]
    assert len(circles) == 2
    assert all(circle.GetRadius() == pytest.approx(.1) for circle in circles)
    assert all(circle.GetNumberOfSides() >= 32 for circle in circles)
    assert "M001" not in viewer.measure_list.item(0).text()


def test_lazy_model_toggle_loads_only_requested_geometry(viewer, app, monkeypatch):
    from mandible_registration import scene_viewer
    from time import monotonic
    key = "baseline_upper"
    assert key not in viewer.models and key in viewer.scene.deferred_keys
    source_mesh = viewer.models["baseline_lower"].mesh
    model = SceneModel(key, "Upper IOS", viewer.scene.project_path.parent / "upper.stl", None, (0.7, 0.9, 0.9))
    requested = []
    vertices = np.array(source_mesh.vertices, copy=True)
    triangles = np.array(source_mesh.triangles, dtype=np.int32, copy=True)
    normals = np.array(source_mesh.vertex_normals, copy=True)

    def load(path, *, keys, array_meshes):
        requested.append(set(keys))
        assert array_meshes
        loaded_model = SceneModel(
            model.key, model.title, model.path,
            (vertices.copy(), triangles.copy(), normals.copy()), model.color,
        )
        return SceneData(path, [loaded_model])

    monkeypatch.setattr(scene_viewer, "load_scene", load)
    # VTK's Windows offscreen window can abort when an actor is added after
    # the first render; exercise the asynchronous loading/state path here and
    # keep actor construction covered by the synchronous polydata tests.
    monkeypatch.setattr(viewer, "_add_model_actors", lambda _models: None)
    completed = []
    viewer._ensure_models((key,), lambda: completed.append(True))
    deadline = monotonic() + 5
    while viewer._thread is not None and monotonic() < deadline:
        QTest.qWait(10)
    assert viewer._thread is None
    assert requested == [{key}]
    assert completed == [True]
    assert key in viewer.models
    assert key not in viewer.scene.deferred_keys


def test_reusable_window_preserves_measurements_when_hidden(viewer, app):
    viewer._reusable = True
    viewer.measure_mode.setCurrentIndex(1)
    viewer.add_anchor(BONE_KEYS[0], [1, 2, 2], 0)
    viewer.close()
    app.processEvents()
    assert not viewer.isVisible() and len(viewer.measurements) == 1
    viewer.show()
    app.processEvents()
    assert viewer.isVisible() and len(viewer.measurements) == 1
    viewer._reusable = False


def test_condyle_analysis_updates_ui_and_exports_without_manual_measurement(viewer, monkeypatch, tmp_path):
    import json
    from PySide6.QtWidgets import QFileDialog
    from mandible_registration import condyles
    mesh = viewer.models[BONE_KEYS[0]].mesh
    profile = condyles.build_profile(mesh, tmp_path / "original.stl", "a" * 64,
                                    {"left": np.arange(12) < 2, "right": np.arange(12) >= 10})
    project = viewer.scene.project_path
    project.write_text(json.dumps({"inputs": {"ct_mandible": {"path": "original.stl", "sha256": "a" * 64}},
                                  "transforms": {"T_CT": {"matrix": np.eye(4).tolist()}, "T_DELTA": {"matrix": viewer.scene.delta.tolist()}},
                                  "condyle_selection": profile}), encoding="utf-8")
    monkeypatch.setattr(condyles, "profile_path", lambda path: tmp_path / "no-current-profile.json")
    viewer.reload_condyles()
    assert len(viewer._condyle_measurements) == 2
    assert "位移 3.000 mm" in viewer.condyle_info.text()
    assert "总旋转角：0.00°" in viewer.condyle_info.text()
    assert all(record["value"] == pytest.approx(3) for record in viewer._condyle_measurements)
    target = tmp_path / "condyles_only.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    viewer._save_measurements()
    payload = json.loads(target.read_text("utf-8"))
    assert not payload["measurements"]
    assert payload["condyle_analysis"]["selection"] == profile
