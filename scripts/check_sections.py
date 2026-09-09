"""Reproducible synthetic and historical-case Qt/VTK section smoke; no case edits.

Run with the project's .venv-integration Python. --project optionally opens an
existing six-STL project read-only; the default fixture includes open bone shells
and an optional fixed maxilla, and is explicitly synthetic, not a clinical case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import open3d as o3d
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPalette, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from check_viewer import screenshot, wait_for_task
from mandible_registration import scene_data
from mandible_registration.condyles import build_profile
from mandible_registration.scene_viewer import SceneViewer


def synthetic_project(root, *, single_side=False):
    root.mkdir(parents=True, exist_ok=True)
    left = o3d.geometry.TriangleMesh.create_sphere(8, resolution=80).translate((-25, 0, 0))
    right = o3d.geometry.TriangleMesh.create_sphere(8, resolution=80).translate((25, 0, 0))
    mesh = left + right
    second = o3d.geometry.TriangleMesh(mesh).translate((0, 2, -2))
    shell = o3d.geometry.TriangleMesh.create_sphere(12, resolution=80)
    corners = np.asarray(shell.vertices)[np.asarray(shell.triangles)]
    shell.remove_triangles_by_mask(corners[:, :, 2].mean(axis=1) < 1)
    shell.remove_unreferenced_vertices()
    maxilla = o3d.geometry.TriangleMesh(shell).translate((-25, 0, 0)) + o3d.geometry.TriangleMesh(shell).translate((25, 0, 0))
    outputs = {}
    for key, geometry in (("ct_mandible_t0", mesh), ("ct_mandible_t1", second), ("ct_maxilla_t0", maxilla)):
        geometry.compute_vertex_normals()
        path = root / f"{key}.stl"
        assert o3d.io.write_triangle_mesh(str(path), geometry)
        outputs[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    count = len(mesh.triangles)
    profile = build_profile(mesh, outputs["ct_mandible_t0"]["path"], outputs["ct_mandible_t0"]["sha256"],
                            {"left": np.arange(count) < len(left.triangles), "right": np.arange(count) >= len(left.triangles)})
    if single_side:
        profile["regions"].pop("right")
    delta = np.eye(4)
    delta[:3, 3] = [0, 2, -2]
    payload = {"schema_version": "1.1.0", "workflow": "mandibular_pose_transfer",
               "inputs": {"ct_mandible": outputs["ct_mandible_t0"]}, "outputs": outputs,
               "transforms": {"T_CT": {"matrix": np.eye(4).tolist()}, "T_DELTA": {"matrix": delta.tolist()}},
               "condyle_selection": profile,
               "joint_space_analysis": {"obsolete_fixture_field": True}}
    path = root / "project.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def drag(view, button, start, end):
    for kind, pos, changed, held in (
        (QEvent.Type.MouseButtonPress, start, button, button),
        (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button),
        (QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(view.vtk_widget, QMouseEvent(kind, pos, pos, changed, held, Qt.KeyboardModifier.NoModifier))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--single-side", action="store_true", help="Only select the synthetic left condyle")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.project and args.single_side:
        parser.error("--single-side only applies to the synthetic fixture")
    project = args.project.resolve() if args.project else synthetic_project(root / "synthetic", single_side=args.single_side)
    original_project = project.read_bytes()
    original_files = {p.name for p in project.parent.iterdir()}
    app = QApplication(["section-smoke"] if args.native else ["section-smoke", "-platform", "offscreen"])
    app.setQuitOnLastWindowClosed(False)
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor("#202020"))
    app.setPalette(palette)
    warnings = []
    QMessageBox.warning = lambda *a: warnings.append(a[2])
    QMessageBox.question = lambda *a: QMessageBox.StandardButton.Yes
    reads = []
    reader = scene_data.load_mesh_arrays
    def counted(path):
        reads.append(str(path))
        return reader(path)
    scene_data.load_mesh_arrays = counted
    window = SceneViewer(offscreen=not args.native)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    app.processEvents()
    window.vtk_widget.Initialize()
    try:
        start = time.perf_counter()
        window.load_project(project)
        wait_for_task(app, window)
        load_seconds = time.perf_counter() - start
        assert window.scene and not window.section_views
        assert window.renderer.GetActiveCamera().GetParallelProjection()
        assert window.centralWidget().widget(0).horizontalScrollBar().maximum() == 0
        assert window.palette().color(QPalette.ColorRole.Window).lightness() > 200
        print(f"Loaded existing meshes in {load_seconds:.3f}s; sections remain lazy", flush=True)
        start = time.perf_counter()
        window.enable_sections()
        wait_for_task(app, window)
        enable_seconds = time.perf_counter() - start
        assert window.section_views, warnings
        if args.single_side:
            assert set(window.section_views) == {"left"}
            assert "未保存" in window.section_labels["right"].text()
        assert len(reads) == len(set(reads))
        for view in window.section_views.values():
            assert sum(len(s) for s in view.segments.values())
        screenshot(window, root / "sections_initial.png")
        print(f"Enabled {len(window.section_views)} sides in {enable_seconds:.3f}s; {len(reads)} unique STL reads", flush=True)
        timings, counts = {}, {}
        for side, view in window.section_views.items():
            initial_count = len(reads)
            counts[side] = {key: {"source_faces": roi.source_triangle_count, "roi_faces": len(roi.triangles)} for key, roi in view.rois.items()}
            view.measure_toggle.setChecked(True)
            points = next(segments.mean(axis=1) for segments in view.segments.values() if len(segments) > 2)
            for point in (points[0], points[len(points) // 2]):
                view.renderer.SetWorldPoint(*point, 1)
                view.renderer.WorldToDisplay()
                x, y, _ = view.renderer.GetDisplayPoint()
                width, height = view.render_window.GetSize()
                pos = QPoint(round(x * view.vtk_widget.width() / width),
                             round(view.vtk_widget.height() - 1 - y * view.vtk_widget.height() / height))
                QTest.mouseClick(view.vtk_widget, Qt.MouseButton.LeftButton, pos=pos)
                app.processEvents()
            assert view.measurements, side
            values = []
            for _ in range(12):
                start = time.perf_counter()
                view.rotate(2, 1)
                view.flush()
                window.render_sections_in_3d()
                values.append(time.perf_counter() - start)
            timings[side] = {"median_frame_ms": float(np.median(values) * 1000), "max_frame_ms": float(max(values) * 1000)}
            drag(view, Qt.MouseButton.LeftButton, QPointF(100, 70), QPointF(130, 90))
            before = view.state.snapshot()
            drag(view, Qt.MouseButton.MiddleButton, QPointF(100, 70), QPointF(120, 85))
            assert view.state.snapshot() == before
            scale = view.renderer.GetActiveCamera().GetParallelScale()
            drag(view, Qt.MouseButton.RightButton, QPointF(100, 70), QPointF(100, 90))
            assert view.renderer.GetActiveCamera().GetParallelScale() > scale
            view.zoom(scale / view.renderer.GetActiveCamera().GetParallelScale())
            pos = QPointF(100, 80)
            QApplication.sendEvent(view.vtk_widget, QWheelEvent(pos, pos, QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
            view.finish_interaction()
            disk, _ = window._section_planes[side]
            np.testing.assert_allclose(disk.GetCenter(), view.state.origin)
            np.testing.assert_allclose(disk.GetNormal(), view.state.normal)
            assert view.state.offset_mm == .5
            assert initial_count == len(reads)
        screenshot(window, root / "sections_rotated.png")
        for view in window.section_views.values():
            view.restore_measurement(1)
        window._axis_view("+Y")
        screenshot(window, root / "sections_measured.png")
        payload = window.measurement_payload()
        assert "joint_space_analysis" not in payload
        (root / "measurements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {"native": args.native, "synthetic": args.project is None, "load_seconds": load_seconds,
                  "enable_seconds": enable_seconds, "stl_read_count": len(reads), "roi": counts,
                  "parallel_projection": bool(window.renderer.GetActiveCamera().GetParallelProjection()),
                  "sidebar_and_view_widths": window.centralWidget().sizes(),
                  "upper_and_lower_heights": window.view_splitter.sizes(),
                  "sidebar_horizontal_scroll_max": window.centralWidget().widget(0).horizontalScrollBar().maximum(),
                  "timing": timings, "warnings": warnings}
        assert not warnings, warnings
        assert project.read_bytes() == original_project
        assert {p.name for p in project.parent.iterdir()} == original_files
        window._reusable = True
        window.close()
        assert window.section_views
        window.show()
        app.processEvents()
        window._reusable = False
        window.close()
        app.processEvents()
        assert not window.section_views
        (root / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print("PASS: sections, native gestures, contour picks, export, light theme, shared reads, clean shutdown", flush=True)
    finally:
        if window._thread is not None:
            wait_for_task(app, window)
        if window.section_views:
            window._reusable = False
            window.close()
            app.processEvents()



if __name__ == "__main__":
    main()
