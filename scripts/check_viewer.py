"""Local non-interactive rendering QA; no input or registration files are changed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import vtk
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandible_registration.gui import MainWindow
from mandible_registration.scene_viewer import SceneViewer
from mandible_registration.input_dialog import SequentialStlDialog
from mandible_registration.drop_import import AssignStlDialog
from mandible_registration.scene_data import BONE_KEYS


def screenshot(window, path):
    QApplication.processEvents()
    window.render_window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window.render_window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()
    data = capture.GetOutput()
    width, height, _ = data.GetDimensions()
    pixels = vtk_to_numpy(data.GetPointData().GetScalars()).reshape(height, width, 3)
    pixels = np.ascontiguousarray(pixels[::-1])
    assert pixels.std() > 15, "Render unexpectedly empty"
    image = QImage(pixels.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    # Native VTK owns its framebuffer; combine that buffer with the real Qt widget grab.
    pixmap = window.grab()
    painter = QPainter(pixmap)
    rect = window.vtk_widget.rect()
    rect.moveTopLeft(window.vtk_widget.mapTo(window, rect.topLeft()))
    painter.drawImage(rect, image)
    painter.end()
    assert pixmap.save(str(path))


def wait_for_task(app, window):
    deadline = time.monotonic() + 120
    while window._thread is not None:
        app.processEvents()
        if time.monotonic() > deadline:
            raise RuntimeError("Background task did not finish in 120 seconds")
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--dark-system", action="store_true", help="Simulate dark Qt palette before each top-level window")
    args = parser.parse_args()
    app = QApplication(["viewer-qa"] if args.native else ["viewer-qa", "-platform", "offscreen"])
    app.setQuitOnLastWindowClosed(False)
    if not QFontDatabase.families():
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    warnings = []
    QMessageBox.warning = lambda *a: warnings.append(a[2])
    QMessageBox.question = lambda *a: QMessageBox.StandardButton.Yes
    args.output.mkdir(parents=True, exist_ok=True)

    def dark_palette():
        if args.dark_system:
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
            palette = QPalette()
            for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
                palette.setColor(role, QColor("#202020"))
            for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
                palette.setColor(role, QColor("#eeeeee"))
            app.setPalette(palette)

    dark_palette()
    gui = MainWindow()
    gui.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    gui.show()
    app.processEvents()
    gui.grab().save(str(args.output / "main_window.png"))
    records = json.loads(args.project.read_text(encoding="utf-8"))["inputs"]
    paths = {key: Path(record["path"]) for key, record in records.items()}
    gui._apply_input_paths({key: paths[key] for key in ("baseline_lower", "ct_mandible", "followup_upper")})
    app.processEvents()
    assert not gui.run_button.isEnabled()
    gui.grab().save(str(args.output / "main_partial.png"))
    gui._load_project(args.project)
    app.processEvents()
    assert gui.run_button.isEnabled()
    assert len(gui._review_paths) == 3
    gui.grab().save(str(args.output / "main_complete.png"))
    diagram_image = QImage(1380, 560, QImage.Format.Format_ARGB32)
    diagram_image.fill(QColor("#f4f7fb"))
    diagram_painter = QPainter(diagram_image)
    gui.flow._renderer.render(diagram_painter)
    diagram_painter.end()
    assert diagram_image.save(str(args.output / "workflow.preview.png"))
    dark_palette()
    dialog = AssignStlDialog(list(paths.values()), {}, gui)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(args.output / "assign_files_light.png"))
    dialog.reject()
    dialog.deleteLater()
    dark_palette()
    dialog = SequentialStlDialog(gui, directory=str(next(iter(paths.values())).parent))
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    dialog.show()
    for _ in range(5):
        app.processEvents()
        time.sleep(0.05)
    dialog.grab().save(str(args.output / "sequential_files_light.png"))
    dialog.reject()
    dialog.deleteLater()
    gui.close()
    dark_palette()
    viewer = SceneViewer(offscreen=not args.native)
    viewer.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    viewer.show()
    app.processEvents()
    viewer.vtk_widget.Initialize()
    print("Qt/VTK initialized", flush=True)
    viewer.load_project(args.project)
    wait_for_task(app, viewer)
    assert viewer.scene is not None, warnings
    assert set(viewer.models) == set(BONE_KEYS), (list(viewer.models), warnings)
    assert len(viewer.scene.deferred_keys) == 5
    assert viewer.model_list.count() == 6
    assert viewer.map_source.count() == 6
    assert not hasattr(viewer, "pick_target")
    assert not warnings, warnings
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == {"ct_mandible_t0", "ct_mandible_t1"}
    print("Loaded bones first; 5 other models remain deferred", flush=True)
    viewer._axis_view("Y")
    screenshot(viewer, args.output / "viewer_bones.png")

    def click_surface(x, y):
        width, height = viewer.render_window.GetSize()
        point = QPoint(round(x * viewer.vtk_widget.width() / width),
                       round(viewer.vtk_widget.height() - 1 - y * viewer.vtk_widget.height() / height))
        QTest.mouseClick(viewer.vtk_widget, Qt.MouseButton.LeftButton, pos=point)
        app.processEvents()

    viewer.map_mode.setCurrentIndex(1)
    viewer._calculate_map()
    wait_for_task(app, viewer)
    assert viewer._map_key == "ct_mandible_t1", warnings
    print("Motion mm min/mean/max:", np.min(viewer._map_values), np.mean(viewer._map_values), np.max(viewer._map_values), flush=True)
    viewer.measure_mode.setCurrentIndex(2)
    # Exercise actual world-to-display surface picks, not synthetic anchor injection.
    vertices = np.asarray(viewer.models["ct_mandible_t1"].mesh.vertices)
    for index in np.linspace(0, len(vertices) - 1, 30, dtype=int):
        viewer.renderer.SetWorldPoint(*vertices[index], 1)
        viewer.renderer.WorldToDisplay()
        x, y, _ = viewer.renderer.GetDisplayPoint()
        click_surface(x, y)
        if viewer.measurements:
            break
    assert viewer.measurements and viewer.measurements[0]["type"] == "length"
    assert {anchor["model"] for anchor in viewer.measurements[0]["anchors"]} == set(BONE_KEYS)
    screenshot(viewer, args.output / "viewer_motion_measurement.png")
    viewer.measure_mode.setCurrentIndex(3)
    for index in np.linspace(0, len(vertices) - 1, 60, dtype=int):
        viewer.renderer.SetWorldPoint(*vertices[index], 1)
        viewer.renderer.WorldToDisplay()
        x, y, _ = viewer.renderer.GetDisplayPoint()
        click_surface(x, y)
        if len(viewer.measurements) == 2:
            break
    assert len(viewer.measurements) == 2
    angle = viewer.measurements[1]
    assert angle["type"] == "angle" and len(angle["anchors"]) == 4
    keys = [anchor["model"] for anchor in angle["anchors"]]
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
    screenshot(viewer, args.output / "viewer_bone_angle.png")
    viewer._set_preset("ct")
    wait_for_task(app, viewer)
    assert viewer.model_list.count() == 2
    viewer.map_mode.setCurrentIndex(0)
    viewer._set_combo(viewer.map_source, "baseline_lower")
    viewer._set_combo(viewer.map_reference, "ct_dentition_t0")
    viewer._calculate_map()
    wait_for_task(app, viewer)
    assert viewer._map_key == "baseline_lower", warnings
    print("IOS to CT surface mm min/mean/max:", np.min(viewer._map_values), np.mean(viewer._map_values), np.max(viewer._map_values), flush=True)
    viewer.measurements.clear()
    viewer.pending.clear()
    viewer._refresh_measurement_list()
    viewer._redraw_measurements()
    viewer._set_preset("ct")
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == {"ct_dentition_t0", "baseline_lower"}
    viewer._axis_view("Z")
    viewer.map_max.setValue(1)
    screenshot(viewer, args.output / "viewer_ct_surface.png")
    viewer.close()
    app.processEvents()
    assert not warnings, warnings
    print("PASS: theme/dialogs, six model options, CT inspection, both maps, automatic cross-bone distance and 4-point angle, clean shutdown.", flush=True)


if __name__ == "__main__":
    main()
