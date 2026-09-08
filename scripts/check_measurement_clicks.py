"""Native Qt mouse-event regression: never calls pick_at/add_anchor directly."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mandible_registration.scene_viewer import SceneViewer
from mandible_registration.scene_data import BONE_KEYS, SceneData, SceneModel

import numpy as np
import open3d as o3d
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox


def main():
    app = QApplication(["native-measurement-qa"])
    QMessageBox.question = lambda *_args: QMessageBox.StandardButton.Yes
    window = SceneViewer()
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    app.processEvents()
    window.vtk_widget.Initialize()
    mesh = o3d.geometry.TriangleMesh.create_box(10, 10, 2)
    mesh.compute_vertex_normals()
    shifted = o3d.geometry.TriangleMesh(mesh).translate((0, 0, 3))
    window.set_scene(SceneData(Path("qa/project.json"), [
        SceneModel(BONE_KEYS[0], "Bone 1", Path("bone1.stl"), mesh, (.2, .6, .9)),
        SceneModel(BONE_KEYS[1], "Bone 2", Path("bone2.stl"), shifted, (1, .5, .1)),
    ]))
    window._axis_view("Z")
    app.processEvents()

    def qt_point(x, y, z=5):
        window.renderer.SetWorldPoint(x, y, z, 1)
        window.renderer.WorldToDisplay()
        dx, dy, _ = window.renderer.GetDisplayPoint()
        width, height = window.render_window.GetSize()
        return QPoint(round(dx * window.vtk_widget.width() / width),
                      round((height - 1 - dy) * window.vtk_widget.height() / height))

    def click(x, y):
        QTest.mouseClick(window.vtk_widget, Qt.MouseButton.LeftButton, pos=qt_point(x, y))
        app.processEvents()

    try:
        window.measure_mode.setCurrentIndex(2)
        click(5, 5)
        print("After first native click:", len(window.pending), window.statusBar().currentMessage(), flush=True)
        assert len(window.pending) == 1, "Native Qt click did not reach the bone picker"
        click(5, 5)
        assert len(window.measurements) == 1
        assert np.isclose(window.measurements[0]["value"], 3, atol=1e-6)
        assert {point["model"] for point in window.measurements[0]["anchors"]} == set(BONE_KEYS)
        window.measure_mode.setCurrentIndex(3)
        for x, y in ((3, 3), (7, 3), (3, 3), (3, 7)):
            click(x, y)
        assert len(window.measurements) == 2
        assert np.isclose(window.measurements[1]["value"], 90, atol=.3)
        window.measure_mode.setCurrentIndex(1)
        start, end = qt_point(5, 5), qt_point(7, 7)
        QTest.mousePress(window.vtk_widget, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(window.vtk_widget, end)
        QTest.mouseRelease(window.vtk_widget, Qt.MouseButton.LeftButton, pos=end)
        app.processEvents()
        assert len(window.measurements) == 2, "Camera drag must not add a marker"
        window.measure_mode.setCurrentIndex(0)
        click(5, 5)
        assert len(window.measurements) == 2
        print("PASS: native Qt clicks, cross-bone distance, four-click angle, drag suppression, browse mode", flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
