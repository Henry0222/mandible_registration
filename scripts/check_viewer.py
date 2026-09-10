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
    # Native VTK owns each framebuffer; composite all three onto the Qt grab.
    pixmap = window.grab()
    painter = QPainter(pixmap)
    for view in (window, *getattr(window, "section_views", {}).values()):
        view.render_window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(view.render_window)
        capture.SetInputBufferTypeToRGB()
        capture.ReadFrontBufferOff()
        capture.Update()
        data = capture.GetOutput()
        width, height, _ = data.GetDimensions()
        pixels = vtk_to_numpy(data.GetPointData().GetScalars()).reshape(height, width, 3)
        pixels = np.ascontiguousarray(pixels[::-1])
        if view is window:
            assert pixels.std() > 15, "Render unexpectedly empty"
        image = QImage(pixels.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        rect = view.vtk_widget.rect()
        rect.moveTopLeft(view.vtk_widget.mapTo(window, rect.topLeft()))
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
    assert len(gui._review_paths) == 3
    assert gui._project_path == args.project.resolve()
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
    warnings.clear()
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
    assert set(BONE_KEYS) <= set(viewer.models), (list(viewer.models), warnings)
    assert viewer.scene.deferred_keys
    assert not hasattr(viewer, "pick_target")
    assert not hasattr(viewer, "map_mode")
    assert not warnings, warnings
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == {"ct_mandible_t0", "ct_mandible_t1"}
    print(f"Loaded bones first; {len(viewer.scene.deferred_keys)} other models remain deferred", flush=True)
    viewer._axis_view("Y")
    screenshot(viewer, args.output / "viewer_bones.png")

    assert not hasattr(viewer, "measure_mode")
    assert not hasattr(viewer, "picker")
    assert not hasattr(viewer, "section_info")
    assert not hasattr(viewer, "show_section_planes")
    viewer._set_preset("ct")
    wait_for_task(app, viewer)
    assert viewer.model_list.count() == 2
    viewer._set_preset("ct")
    assert {key for key, actor in viewer.actors.items() if actor.GetVisibility()} == {"ct_dentition_t0", "baseline_lower"}
    viewer._axis_view("Z")
    screenshot(viewer, args.output / "viewer_ct_surface.png")
    viewer.close()
    app.processEvents()
    assert not warnings, warnings
    print("PASS: theme/dialogs, model options, CT inspection, removed 3D point tools, clean shutdown.", flush=True)


if __name__ == "__main__":
    main()
