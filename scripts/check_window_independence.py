"""Exercise the real main-window launch path and native top-level ownership."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox
from mandible_registration.gui import MainWindow
from mandible_registration import scene_viewer
from check_viewer import wait_for_task, screenshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication(["window-independence-qa"])
    app.setQuitOnLastWindowClosed(False)
    errors = []
    QMessageBox.warning = lambda *a: errors.append(a[2])
    QMessageBox.question = lambda *a: QMessageBox.StandardButton.Yes
    original = scene_viewer.SceneViewer
    def hidden_native(*a, **kw):
        window = original(*a, **kw)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        return window
    scene_viewer.SceneViewer = hidden_native
    main_window = MainWindow()
    main_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    main_window.show()
    main_window._load_project(args.project)
    main_window._view_result()
    viewer = next(iter(main_window._measurement_windows.values()))
    try:
        wait_for_task(app, viewer)
        assert not errors, errors
        assert viewer.parent() is None
        assert viewer.windowHandle().transientParent() is None
        assert main_window.winId() != viewer.winId()
        main_window.showMinimized()
        app.processEvents()
        assert viewer.isVisible() and not viewer.isMinimized()
        main_window.showNormal()
        app.processEvents()
        main_window.grab().save(str(args.output / "main-window.png"))
        screenshot(viewer, args.output / "comparison-window.png")
        main_window.close()
        app.processEvents()
        assert viewer.isVisible() and not viewer._reusable
        assert viewer.scene is not None
        result = {"parentless": True, "distinct_native_handles": True,
                  "main_minimize_does_not_minimize_viewer": True,
                  "main_close_keeps_viewer_open": True, "warnings": errors}
        (args.output / "window_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("PASS: independent native windows, main minimize/close, real result-launch path")
    finally:
        viewer._reusable = False
        viewer.close()
        main_window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
