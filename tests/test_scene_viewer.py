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
        SceneModel("ct_maxilla_t0", "Maxilla T0", tmp_path / "maxilla.stl", first, (0.86, 0.79, 0.62)),
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
    assert viewer.renderer.GetActiveCamera().GetParallelProjection()
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
    assert camera.GetParallelProjection()
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











def test_model_options_ct_inspection_and_return_do_not_leak_teeth(viewer):
    viewer._set_preset("ct")
    assert viewer.model_list.count() == 2
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == set(CT_INSPECTION_KEYS)
    viewer._set_preset("bones")
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == set(BONE_KEYS)
    assert not hasattr(viewer, "measure_mode")
    assert not hasattr(viewer, "measure_group")
    assert not hasattr(viewer, "picker")






def test_measurement_export_new_file_only_and_preserves_project(viewer, app, tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QFileDialog
    view = enable_test_sections(viewer, app, monkeypatch)
    view.measure_toggle.setChecked(True)
    for point in ([5, 0, 1], [5, 10, 1]):
        assert view.pick_at(*section_display(view, point))
    target = tmp_path / "export.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    viewer._save_measurements()
    before = target.read_bytes()
    payload = json.loads(before)
    assert payload["condyle_sections"]["sides"]["left"]["measurements"][0]["value"] == pytest.approx(10)
    assert payload["measurements"] == []
    viewer._save_measurements()
    assert warnings and target.read_bytes() == before
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(viewer.scene.project_path), ""))
    viewer._save_measurements()
    assert len(warnings) == 2






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
    np.testing.assert_allclose(viewer._rotation_center, np.mean([r["center_t0_mm"] for r in viewer.condyle_report["regions"].values()], axis=0))
    assert "位移 3.000 mm" in viewer.condyle_info.text()
    assert "三维总旋转：0.00°" in viewer.condyle_info.text()
    assert all(record["value"] == pytest.approx(3) for record in viewer._condyle_measurements)
    target = tmp_path / "condyles_only.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    viewer._save_measurements()
    payload = json.loads(target.read_text("utf-8"))
    assert not payload["measurements"]
    assert payload["condyle_analysis"]["selection"] == profile



def enable_test_sections(viewer, app, monkeypatch, *, bilateral=True):
    from time import monotonic
    from mandible_registration import scene_data
    # No section may use a file reader: these in-memory models are already loaded.
    monkeypatch.setattr(scene_data, "load_mesh_arrays", lambda *a: pytest.fail("duplicate STL read"))
    viewer.condyle_report = {"regions": {"left": {"center_t0_mm": [5., 5., 2.]}}}
    if bilateral:
        viewer.condyle_report["regions"]["right"] = {"center_t0_mm": [7., 5., 2.]}
    viewer.enable_sections()
    deadline = monotonic() + 10
    while viewer._thread is not None and monotonic() < deadline:
        QTest.qWait(10)
    assert viewer._thread is None
    assert viewer._sections_enabled
    app.processEvents()
    return viewer.section_views["left"]


def test_sections_are_lazy_and_legacy_empty_states_are_independent(viewer):
    assert not viewer.section_views
    assert not viewer.section_rois
    assert not viewer.section_sources
    assert "未保存" in viewer.section_labels["left"].text()
    assert "未保存" in viewer.section_labels["right"].text()
    assert not viewer.section_button.isEnabled()
    assert not hasattr(viewer, "joint_space_button")
    assert "joint_space_analysis" not in viewer.measurement_payload()


def test_single_side_roi_cache_shared_sources_no_duplicate_stl_reads(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch, bilateral=False)
    assert set(viewer.section_views) == {"left"}
    assert "未保存" in viewer.section_labels["right"].text()
    assert not viewer.section_labels["right"].isHidden()
    assert viewer.section_labels["left"].isHidden()
    for key in view.rois:
        assert viewer.section_sources[key] is viewer.polydata[key]
        assert view.cutters[key].GetInput() is view.roi_polydata[key]
    cache = viewer.section_rois
    viewer.enable_sections()
    assert viewer.section_rois is cache
    for key, segments in view.segments.items():
        assert len(segments)
        assert np.max(np.linalg.norm(segments - view.state.center, axis=2)) <= 20 + 1e-7
        assert np.max(np.abs((segments - view.state.origin) @ view.state.normal)) < 1e-7
        assert view.locators[key].GetDataSet() is view.roi_polydata[key]


def test_independent_sections_camera_wheel_pan_and_synchronized_planes(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch)
    right = viewer.section_views["right"]
    right_snapshot = right.state.snapshot()
    view.rotate(37, 18)
    view.scroll(3)
    view.finish_interaction()
    assert view.state.offset_mm == pytest.approx(1.5)
    camera = view.renderer.GetActiveCamera()
    np.testing.assert_allclose(camera.GetDirectionOfProjection(), -view.state.normal)
    disk, actor = viewer._section_planes["left"]
    np.testing.assert_allclose(disk.GetCenter(), view.state.origin)
    np.testing.assert_allclose(disk.GetNormal(), view.state.normal)
    assert disk.GetRadius() == pytest.approx(np.sqrt(400 - 1.5 ** 2))
    assert actor.GetVisibility() and not actor.GetPickable()
    assert right.state.snapshot() == right_snapshot
    before = view.state.snapshot()
    focal = np.array(camera.GetFocalPoint())
    view.pan(20, -15)
    view.finish_interaction()
    assert view.state.snapshot() == before
    assert not np.allclose(camera.GetFocalPoint(), focal)
    assert not hasattr(viewer, "show_section_planes")
    assert actor.GetVisibility()
    assert len(view.segments[BONE_KEYS[0]])


def section_display(view, point):
    view.renderer.SetWorldPoint(*point, 1)
    view.renderer.WorldToDisplay()
    return view.renderer.GetDisplayPoint()[:2]


def section_qt_point(view, point):
    x, y = section_display(view, point)
    width, height = view.render_window.GetSize()
    return QPoint(round(x * view.vtk_widget.width() / width),
                  round(view.vtk_widget.height() - 1 - y * view.vtk_widget.height() / height))


def test_section_real_contour_picking_distance_only_and_restore(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch)
    view.measure_toggle.setChecked(True)
    # Initial x=5 plane intersects box y=0 and y=10 edges at z=1.
    for point in ([5, 0, 1], [5, 10, 1]):
        assert view.pick_at(*section_display(view, point))
    record = view.measurements[0]
    assert record["value"] == pytest.approx(10, abs=1e-7)
    labels = [a.GetInput() for a in view._annotations if isinstance(a, vtk.vtkBillboardTextActor3D)]
    assert labels == ["10.000 mm"]
    assert record["plane"]["roi_radius_mm"] == 20
    assert record["definition"] == "manual_section_contour_distance"
    assert not view.pick_at(*section_display(view, [5, 5, -15]))
    assert view.pick_at(*section_display(view, [5, 0, 1]))
    view.scroll(1)
    view.finish_interaction()
    assert not view.pending
    assert len(view.measurements) == 1
    assert not view._annotations  # No stale points shown on the new plane.
    view.restore_measurement(1)
    assert view.state.offset_mm == 0
    assert view._annotations
    payload = viewer.measurement_payload()
    assert payload["condyle_sections"]["sides"]["left"]["measurements"][0]["value"] == pytest.approx(10)
    assert "joint_space_analysis" not in payload
    view.undo()
    assert not view.measurements


def test_section_qt_gestures_use_right_zoom_middle_pan_without_drag_picks(viewer, app, monkeypatch):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    view = enable_test_sections(viewer, app, monkeypatch)
    view.measure_toggle.setChecked(True)
    widget = view.vtk_widget
    start = QPointF(100, 80)
    end = QPointF(150, 105)
    def event(kind, pos, button, buttons):
        QApplication.sendEvent(widget, QMouseEvent(kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier))
    old_normal = view.state.normal.copy()
    event(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    event(QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    event(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert not np.allclose(view.state.normal, old_normal)
    assert not view.pending and not view.measurements
    before = view.state.snapshot()
    event(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton)
    event(QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton)
    event(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton)
    assert view.state.snapshot() == before
    assert np.linalg.norm(view.state.pan_mm) > 0
    old_scale = view.renderer.GetActiveCamera().GetParallelScale()
    old_pan = view.state.pan_mm.copy()
    event(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    event(QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    event(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert view.renderer.GetActiveCamera().GetParallelScale() > old_scale
    np.testing.assert_allclose(view.state.pan_mm, old_pan)
    scale = view.renderer.GetActiveCamera().GetParallelScale()
    QApplication.sendEvent(widget, QWheelEvent(start, start, QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
    view.finish_interaction()
    assert view.state.offset_mm == .5
    assert view.renderer.GetActiveCamera().GetParallelScale() == scale
    view.reset()
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=section_qt_point(view, [5, 0, 1]))
    app.processEvents()
    assert len(view.pending) == 1


def test_section_visibility_blocks_picks_and_world_color_follows(viewer, app, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    view = enable_test_sections(viewer, app, monkeypatch)
    view.measure_toggle.setChecked(True)
    for check in view.model_checks.values():
        check.setChecked(False)
    assert not view.pick_at(*section_display(view, [5, 0, 1]))
    view.model_checks[BONE_KEYS[0]].setChecked(True)
    assert view.pick_at(*section_display(view, [5, 0, 1]))
    viewer.model_list.setCurrentRow(0)
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#12ab34"))
    viewer._change_color()
    assert view.actors[BONE_KEYS[0]].GetProperty().GetColor() == viewer.actors[BONE_KEYS[0]].GetProperty().GetColor()


def test_section_zoom_and_pan_preserve_world_distance_and_roi(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch)
    view.measure_toggle.setChecked(True)
    cache = view.roi_polydata.copy()
    view.zoom(.5)
    view.pan(30, 15)
    view.finish_interaction()
    for point in ([5, 0, 1], [5, 10, 1]):
        assert view.pick_at(*section_display(view, point))
    assert view.measurements[0]["value"] == pytest.approx(10, abs=1e-7)
    assert view.roi_polydata == cache
    assert view.state.offset_mm == 0


def test_section_interaction_coalesces_updates_then_restores_quality(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch)
    updates = []
    view.plane_changed.connect(lambda side, final: updates.append(final))
    for _ in range(20):
        view.rotate(1, .5)
    assert updates == []
    assert view.timer.isActive() and view.timer.interval() == 33
    assert not view.renderer.GetUseFXAA()
    view.flush()
    assert updates == [False]
    assert not viewer.renderer.GetUseDepthPeeling()
    view.finish_interaction()
    assert updates[-1] is True
    assert view.renderer.GetUseFXAA()
    assert viewer.renderer.GetUseDepthPeeling()
    assert not view.timer.isActive() and not view.finish_timer.isActive()


def test_selected_side_can_have_empty_roi_without_blocking_other_side(viewer, app, monkeypatch):
    # A moved T1 or a distant/missing reference is allowed to have no local faces.
    view = enable_test_sections(viewer, app, monkeypatch)
    from mandible_registration.section_geometry import sphere_roi
    from mandible_registration.section_viewer import SectionView
    empty = {key: sphere_roi(np.array([[100., 0, 0]]), [[0, 0, 0]], [0, 0, 0]) for key in BONE_KEYS}
    window = SectionView("left", [0, 0, 0], empty, viewer.actors, offscreen=True)
    try:
        window.measure_toggle.setChecked(True)
        assert "没有可见截线" in window.info.text()
        assert not window.pick_at(200, 200)
        assert sum(len(s) for s in view.segments.values()) > 0
    finally:
        window.dispose()
        window.deleteLater()


def test_reopen_and_selection_change_lifecycle(viewer, app, monkeypatch):
    view = enable_test_sections(viewer, app, monkeypatch)
    viewer._reusable = True
    viewer.close()
    viewer.show()
    assert viewer.section_views["left"] is view
    viewer._reusable = False
    viewer.condyle_report["regions"]["left"]["center_t0_mm"][0] += 1
    viewer._update_section_availability()
    assert not viewer.section_views
    assert not viewer._section_planes
    assert viewer.section_button.isEnabled()


def test_main_view_orbits_condyle_midpoint_and_preserves_it_after_pan(viewer):
    viewer._rotation_center = np.array([5., 5., 2.])
    viewer._reset_camera()
    camera = viewer.renderer.GetActiveCamera()
    np.testing.assert_allclose(camera.GetFocalPoint(), viewer._rotation_center)
    viewer.pan(30, 12)
    position = np.array(camera.GetPosition())
    focal = np.array(camera.GetFocalPoint())
    pivot = viewer._rotation_center.copy()
    viewer.rotate(40, 25)
    viewer.finish_interaction()
    assert np.linalg.norm(np.array(camera.GetPosition()) - pivot) == pytest.approx(np.linalg.norm(position - pivot))
    assert np.linalg.norm(np.array(camera.GetFocalPoint()) - pivot) == pytest.approx(np.linalg.norm(focal - pivot))
    np.testing.assert_allclose(viewer._rotation_center, pivot)


def test_main_view_mouse_mapping_and_no_manual_picker(viewer):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    viewer._rotation_center = np.array([5., 5., 2.])
    viewer._reset_camera()
    camera = viewer.renderer.GetActiveCamera()
    def drag(button):
        for kind, pos, changed, held in (
            (QEvent.Type.MouseButtonPress, QPointF(100, 80), button, button),
            (QEvent.Type.MouseMove, QPointF(130, 100), Qt.MouseButton.NoButton, button),
            (QEvent.Type.MouseButtonRelease, QPointF(130, 100), button, Qt.MouseButton.NoButton),
        ):
            QApplication.sendEvent(viewer.vtk_widget, QMouseEvent(kind, pos, pos, changed, held, Qt.KeyboardModifier.NoModifier))
    distance = camera.GetDistance()
    scale = camera.GetParallelScale()
    focal = np.array(camera.GetFocalPoint())
    drag(Qt.MouseButton.RightButton)
    assert camera.GetParallelScale() > scale
    assert camera.GetDistance() == pytest.approx(distance)
    np.testing.assert_allclose(camera.GetFocalPoint(), focal)
    scale = camera.GetParallelScale()
    drag(Qt.MouseButton.MiddleButton)
    assert camera.GetDistance() == pytest.approx(distance)
    assert camera.GetParallelScale() == pytest.approx(scale)
    assert not np.allclose(camera.GetFocalPoint(), focal)
    assert not hasattr(viewer, "measure_mode") and not hasattr(viewer, "add_anchor")
    assert not hasattr(viewer, "section_info") and not hasattr(viewer, "show_section_planes")


def test_parallel_projection_keeps_equal_lengths_at_different_depths(viewer):
    viewer._axis_view("+Z")
    camera = viewer.renderer.GetActiveCamera()
    focal = np.array(camera.GetFocalPoint())

    def display(point):
        viewer.renderer.SetWorldPoint(*point, 1)
        viewer.renderer.WorldToDisplay()
        return np.array(viewer.renderer.GetDisplayPoint()[:2])

    lengths = []
    for depth in (-10, 10):
        start = focal + [0, 0, depth]
        lengths.append(np.linalg.norm(display(start + [5, 0, 0]) - display(start)))
    assert lengths[0] == pytest.approx(lengths[1])


def test_parallel_reset_fits_bone_around_off_center_condyle_pivot(viewer):
    viewer._rotation_center = np.array([30., -25., 60.])
    viewer.rotate(50, 25)
    viewer._reset_camera()
    camera = viewer.renderer.GetActiveCamera()
    np.testing.assert_allclose(camera.GetFocalPoint(), viewer._rotation_center)
    bounds = viewer.renderer.ComputeVisiblePropBounds()
    width, height = viewer.render_window.GetSize()
    for x in bounds[:2]:
        for y in bounds[2:4]:
            for z in bounds[4:]:
                viewer.renderer.SetWorldPoint(x, y, z, 1)
                viewer.renderer.WorldToDisplay()
                px, py, pz = viewer.renderer.GetDisplayPoint()
                assert 0 < px < width and 0 < py < height
                assert 0 <= pz <= 1


def test_default_view_ratios_survive_section_activation_and_allow_manual_resize(viewer, app, monkeypatch):
    left, right = viewer.centralWidget().sizes()
    top, bottom = viewer.view_splitter.sizes()
    assert left / right == pytest.approx(1 / 4, abs=.005)
    assert top / bottom == pytest.approx(3 / 2, abs=.01)
    assert viewer.centralWidget().widget(0).horizontalScrollBar().maximum() == 0
    enable_test_sections(viewer, app, monkeypatch)
    app.processEvents()
    top, bottom = viewer.view_splitter.sizes()
    assert top / bottom == pytest.approx(3 / 2, abs=.01)
    assert viewer.centralWidget().widget(0).horizontalScrollBar().maximum() == 0
    viewer.centralWidget().setSizes([400, 1000])
    viewer.view_splitter.setSizes([450, 400])
    custom = (viewer.centralWidget().sizes(), viewer.view_splitter.sizes())
    viewer.hide()
    viewer.show()
    app.processEvents()
    assert (viewer.centralWidget().sizes(), viewer.view_splitter.sizes()) == custom


def test_comparison_window_is_independent_from_main_window(viewer, app):
    from mandible_registration.gui import MainWindow
    main = MainWindow()
    main.show()
    main._measurement_windows["fixture"] = viewer
    app.processEvents()
    assert viewer.parentWidget() is None
    assert viewer.isWindow() and main.isWindow()
    assert viewer.winId() != main.winId()
    main.showMinimized()
    app.processEvents()
    assert not viewer.isMinimized() and viewer.isVisible()
    main.close()
    app.processEvents()
    assert viewer.isVisible()
    assert not viewer._reusable
    main.deleteLater()


def test_missing_one_center_uses_the_available_center(viewer, app, monkeypatch, tmp_path):
    import json
    from mandible_registration import condyles
    mesh = viewer.models[BONE_KEYS[0]].mesh
    profile = condyles.build_profile(mesh, tmp_path / "original.stl", "b" * 64,
                                    {"left": np.arange(12) < 2, "right": np.zeros(12, dtype=bool)})
    viewer.scene.project_path.write_text(json.dumps({
        "inputs": {"ct_mandible": {"path": "original.stl", "sha256": "b" * 64}},
        "transforms": {"T_CT": {"matrix": np.eye(4).tolist()}, "T_DELTA": {"matrix": viewer.scene.delta.tolist()}},
        "condyle_selection": profile,
    }), encoding="utf-8")
    monkeypatch.setattr(condyles, "profile_path", lambda path: tmp_path / "no-current-profile.json")
    viewer.reload_condyles()
    np.testing.assert_allclose(viewer._rotation_center, profile["regions"]["left"]["center_ct_mm"])
    assert "font-size:15px" in viewer.condyle_info.text()
    assert "color:#7a818b" in viewer.condyle_info.text()
    assert "上移 3.000 mm" in viewer.condyle_info.text()
    assert "球拟合" not in viewer.condyle_info.text()
