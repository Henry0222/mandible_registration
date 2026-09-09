"""Lazily constructed VTK contour views; the owner supplies cached ROI meshes."""
from __future__ import annotations

import numpy as np
import warnings
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from vtkmodules.vtkCommonCore import vtkPoints, reference
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPlane, vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkFiltersCore import vtkPlaneCutter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from vtkmodules.vtkRenderingCore import vtkActor, vtkBillboardTextActor3D, vtkPolyDataMapper, vtkRenderer
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from .view_interaction import ViewGestures
from .section_geometry import INITIAL_ORIENTATION, SectionState, clip_segments_to_sphere

SIDE_COLORS = {"left": (0.15, 0.90, 0.60), "right": (0.95, 0.35, 0.75)}


def vtk_array(data):
    # VTK 9.6 uses the deprecated ndarray.shape setter with NumPy 2.5.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated")
        return vtk_to_numpy(data)


def segment_polydata(segments):
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(segments.reshape(-1, 3), deep=True))
    lines = vtkCellArray()
    lines.SetData(numpy_to_vtkIdTypeArray(np.arange(0, len(segments) * 2 + 1, 2, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(segments) * 2, dtype=np.int64), deep=True))
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetLines(lines)
    return data


class SectionView(QWidget):
    plane_changed = Signal(str, bool)

    def __init__(self, side, center, rois, sources, *, offscreen=False, parent=None):
        super().__init__(parent)
        # Reuse the existing Qt/VTK adapter and mesh converter, without STL I/O.
        from .scene_viewer import OffscreenVtkWidget, QVTKRenderWindowInteractor, mesh_polydata

        self.side = side
        self.state = SectionState(center)
        self.rois = rois
        self.measurements, self.pending = [], []
        self._annotations = []
        self.cutters, self.roi_polydata, self.actors = {}, {}, {}
        self.segments, self.locators = {}, {}
        self._plane_dirty = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.offset_label = QLabel()
        bar.addWidget(self.offset_label)
        self.step = QComboBox()
        for value in (.1, .5, 1.):
            self.step.addItem(f"{value:g} mm / 格", value)
        self.step.setCurrentIndex(1)
        self.step.setMaximumWidth(125)
        bar.addWidget(self.step)
        for title, factor in (("+", .8), ("−", 1.25)):
            zoom = QPushButton(title)
            zoom.setFixedWidth(28)
            zoom.setToolTip("放大剖面视野" if factor < 1 else "缩小剖面视野")
            zoom.clicked.connect(lambda checked=False, f=factor: self.zoom(f))
            bar.addWidget(zoom)
        reset = QPushButton("复位")
        reset.setFixedWidth(52)
        reset.clicked.connect(self.reset)
        bar.addWidget(reset)
        layout.addLayout(bar)
        controls = QHBoxLayout()
        self.model_checks = {}
        for key, roi in rois.items():
            check = QCheckBox({"ct_mandible_t0": "颌骨 T0", "ct_mandible_t1": "颌骨 T1",
                               "ct_maxilla_t0": "固定上颌骨"}[key])
            check.setChecked(True)
            check.toggled.connect(lambda checked, k=key: self.set_model_visible(k, checked))
            controls.addWidget(check)
            self.model_checks[key] = check
        controls.addStretch(1)
        layout.addLayout(controls)
        self.vtk_widget = OffscreenVtkWidget(self) if offscreen else QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMinimumHeight(220)
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.SetMultiSamples(0)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.075, .10, .145)
        self.renderer.SetUseFXAA(True)
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleUser())
        self.gestures = ViewGestures(self)
        self.vtk_widget.installEventFilter(self.gestures)
        layout.addWidget(self.vtk_widget, 1)
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetParallelScale(23)
        self.plane = vtkPlane()
        for key, roi in rois.items():
            data = mesh_polydata((roi.vertices, roi.triangles, None))
            self.roi_polydata[key] = data
            locator = vtkStaticCellLocator()
            locator.SetDataSet(data)
            self.locators[key] = locator
            cutter = vtkPlaneCutter()
            cutter.SetInputData(data)
            cutter.SetPlane(self.plane)
            cutter.ComputeNormalsOff()
            cutter.InterpolateAttributesOn()
            cutter.BuildTreeOn()
            self.cutters[key] = cutter
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(segment_polydata([]))
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*sources[key].GetProperty().GetColor())
            actor.GetProperty().SetLineWidth(2)
            actor.GetProperty().LightingOff()
            self.actors[key] = actor
            self.renderer.AddActor(actor)
        actions = QHBoxLayout()
        self.measure_toggle = QCheckBox("距离取点")
        self.measure_toggle.toggled.connect(self._mode_changed)
        actions.addWidget(self.measure_toggle)
        self.records = QComboBox()
        self.records.setMinimumContentsLength(9)
        self.records.activated.connect(self.restore_measurement)
        actions.addWidget(self.records, 1)
        for title, callback in (("撤销", self.undo), ("清空", self.clear_measurements)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.flush)
        self.finish_timer = QTimer(self)
        self.finish_timer.setSingleShot(True)
        self.finish_timer.setInterval(140)
        self.finish_timer.timeout.connect(self.finish_interaction)
        self.vtk_widget.Initialize()
        self._refresh_records()
        self.flush(final=True)

    def _camera(self):
        state = self.state
        focal = state.origin + state.right * state.pan_mm[0] + state.up * state.pan_mm[1]
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*focal)
        camera.SetPosition(*(focal + state.normal * 80))
        camera.SetViewUp(*state.up)
        self.renderer.ResetCameraClippingRange()

    def rotate(self, dx, dy):
        self.state.rotate(dx, dy)
        self._changed()

    def scroll(self, steps):
        self.state.scroll(steps, self.step.currentData())
        self._changed()

    def pan(self, dx, dy):
        scale = 2 * self.renderer.GetActiveCamera().GetParallelScale() / max(self.vtk_widget.height(), 1)
        self.state.pan_mm += [-dx * scale, dy * scale]
        self._schedule()

    def zoom(self, factor, *, interactive=False):
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelScale(float(np.clip(camera.GetParallelScale() * factor, .5, 100)))
        self._schedule() if interactive else self.finish_interaction()

    def _changed(self):
        self.pending.clear()  # Never join anchors taken on different planes.
        self._plane_dirty = True
        self._schedule()

    def _schedule(self):
        self.render_window.SetDesiredUpdateRate(20)
        self.renderer.SetUseFXAA(False)
        if not self.timer.isActive():
            self.timer.start()
        self.finish_timer.start()

    def flush(self, final=False):
        self.timer.stop()
        self._camera()
        if self._plane_dirty:
            self.plane.SetOrigin(*self.state.origin)
            self.plane.SetNormal(*self.state.normal)
            for key, cutter in self.cutters.items():
                if not self.rois[key].triangles.size:
                    self.segments[key] = np.empty((0, 2, 3))
                    continue
                cutter.Update()
                output = cutter.GetOutput()
                count = output.GetNumberOfLines()
                if count:
                    points = vtk_array(output.GetPoints().GetData())
                    lines = vtk_array(output.GetLines().GetConnectivityArray()).reshape(-1, 2)
                    segments, _ = clip_segments_to_sphere(points[lines], self.state.center)
                else:
                    segments = np.empty((0, 2, 3))
                self.segments[key] = segments
                self.actors[key].GetMapper().SetInputData(segment_polydata(segments))
            self._plane_dirty = False
        self.offset_label.setText(f"沿视线 {self.state.offset_mm:+.2f} mm")
        self.offset_label.setToolTip(
            INITIAL_ORIENTATION + "\n当前法向 XYZ：" + " / ".join(f"{v:+.3f}" for v in self.state.normal)
            + "\n正偏移沿相机视线向场景内部；ROI 固定在 T0 中心。"
        )
        visible_count = sum(len(lines) for key, lines in self.segments.items() if self.actors[key].GetVisibility())
        self.info.setText("左拖旋转 · 中拖平移 · 右拖缩放 · 滚轮移剖面" if visible_count else "当前平面在 20 mm ROI 内没有可见截线；可滚动、旋转或复位。")
        self._draw_measurements()
        self.render_window.Render()
        self.plane_changed.emit(self.side, final)

    def finish_interaction(self):
        self.finish_timer.stop()
        self.render_window.SetDesiredUpdateRate(.01)
        self.renderer.SetUseFXAA(True)
        self.flush(final=True)

    def reset(self):
        self.state = SectionState(self.state.center)
        self.renderer.GetActiveCamera().SetParallelScale(23)
        self._changed()
        self.finish_interaction()

    def set_model_visible(self, key, visible):
        self.actors[key].SetVisibility(visible)
        self.pending.clear()
        self.flush(final=True)

    def sync_colors(self, sources):
        for key, actor in self.actors.items():
            actor.GetProperty().SetColor(*sources[key].GetProperty().GetColor())
        self.render_window.Render()

    def _mode_changed(self):
        self.pending.clear()
        self.flush(final=True)

    def pick_qt_position(self, position):
        width, height = self.render_window.GetSize()
        return self.pick_at(position.x() * width / max(self.vtk_widget.width(), 1),
                            (self.vtk_widget.height() - 1 - position.y()) * height / max(self.vtk_widget.height(), 1))

    def pick_at(self, x, y):
        """Snap within 6 logical pixels to an actual visible cut segment, in world mm."""
        if not self.measure_toggle.isChecked():
            return False
        self.flush(final=True)  # Pick the latest plane, even before the 33 ms timer.
        width, height = self.render_window.GetSize()
        camera = self.renderer.GetActiveCamera()
        units_per_pixel = 2 * camera.GetParallelScale() / max(height, 1)
        focal = np.asarray(camera.GetFocalPoint())
        world = focal + self.state.right * (x - width / 2) * units_per_pixel + self.state.up * (y - height / 2) * units_per_pixel
        tolerance = 6 * max(width / max(self.vtk_widget.width(), 1), height / max(self.vtk_widget.height(), 1)) * units_per_pixel
        best = None
        for key, segments in self.segments.items():
            if not len(segments) or not self.actors[key].GetVisibility():
                continue
            start, edge = segments[:, 0], segments[:, 1] - segments[:, 0]
            t = np.clip(np.einsum("ij,ij->i", world - start, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
            points = start + t[:, None] * edge
            distance = np.linalg.norm(points - world, axis=1)
            index = int(np.argmin(distance))
            if distance[index] <= tolerance and (best is None or distance[index] < best[0]):
                best = (distance[index], key, points[index])
        if best is None:
            self.info.setText("未命中可见截线；请靠近轮廓点击（6 像素吸附）。")
            return False
        _, key, point = best
        # VTK's optimized plane cutter does not preserve cell attributes. Resolve
        # provenance only on a click, using the cached ROI locator (no full scan).
        nearest, local_id, sub_id, distance2 = [0., 0., 0.], reference(0), reference(0), reference(0.)
        self.locators[key].FindClosestPoint(point, nearest, local_id, sub_id, distance2)
        if int(local_id) < 0 or float(distance2) > 1e-8:
            return False
        cell_id = int(self.rois[key].source_face_ids[int(local_id)])
        if self.pending and np.linalg.norm(point - self.pending[0]["xyz_mm"]) < 1e-6:
            self.info.setText("距离不能为零，请选择另一个轮廓点。")
            return False
        self.pending.append({"model": key, "xyz_mm": point.tolist(), "triangle_id": cell_id})
        if len(self.pending) == 2:
            value = float(np.linalg.norm(np.subtract(self.pending[1]["xyz_mm"], self.pending[0]["xyz_mm"])))
            self.measurements.append({"type": "length", "unit": "mm", "value": value,
                                      "definition": "manual_section_contour_distance", "side": self.side,
                                      "anchors": self.pending.copy(), "plane": self.state.snapshot()})
            self.pending.clear()
            self._refresh_records()
        self._draw_measurements()
        self.render_window.Render()
        self.info.setText("已取第 1 点，请点击另一截线点。" if self.pending else "距离已记录；可继续取点，选记录可返回原剖面。")
        return True

    def _same_plane(self, record):
        plane = record["plane"]
        return (np.allclose(plane["normal_xyz"], self.state.normal, atol=1e-7, rtol=0)
                and np.allclose(plane["origin_mm"], self.state.origin, atol=1e-7, rtol=0))

    def _draw_measurements(self):
        for actor in self._annotations:
            self.renderer.RemoveActor(actor)
        self._annotations.clear()
        records = [r for r in self.measurements if self._same_plane(r)]
        if self.pending:
            records.append({"anchors": self.pending})
        # Point primitives and simple lines avoid sphere/tube meshes per anchor.
        for record in records:
            coordinates = np.array([a["xyz_mm"] for a in record["anchors"]])
            points = vtkPoints()
            points.SetData(numpy_to_vtk(coordinates, deep=True))
            cells = vtkCellArray()
            for index in range(len(coordinates)):
                cells.InsertNextCell(1)
                cells.InsertCellPoint(index)
            data = vtkPolyData()
            data.SetPoints(points)
            data.SetVerts(cells)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(data)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(7)
            actor.GetProperty().SetColor(1, .95, .35)
            actor.GetProperty().RenderPointsAsSpheresOn()
            self._annotation(actor)
            if "value" in record:
                source = vtkLineSource()
                source.SetPoint1(*coordinates[0])
                source.SetPoint2(*coordinates[1])
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
                line = vtkActor()
                line.SetMapper(mapper)
                line.GetProperty().SetColor(1, .95, .35)
                line.GetProperty().SetLineWidth(2)
                self._annotation(line)
                text = vtkBillboardTextActor3D()
                text.SetInput(f"{record['value']:.3f} mm")
                text.SetPosition(*coordinates.mean(axis=0))
                text.SetDisplayOffset(6, 8)
                text.GetTextProperty().SetFontSize(14)
                text.GetTextProperty().SetColor(1, 1, .85)
                text.GetTextProperty().SetBackgroundColor(.05, .06, .08)
                text.GetTextProperty().SetBackgroundOpacity(.85)
                self._annotation(text)

    def _annotation(self, actor):
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self._annotations.append(actor)

    def _refresh_records(self):
        self.records.clear()
        self.records.addItem(f"测量记录（{len(self.measurements)}）")
        for record in self.measurements:
            self.records.addItem(f"{record['value']:.3f} mm")

    def restore_measurement(self, index):
        if index <= 0:
            return
        plane = self.measurements[index - 1]["plane"]
        self.state = SectionState(plane["roi_center_t0_mm"], np.array(plane["normal_xyz"]),
                                  np.array(plane["view_up_xyz"]), plane["offset_along_view_mm"])
        self._changed()
        self.finish_interaction()

    def undo(self):
        if self.pending:
            self.pending.pop()
        elif self.measurements:
            self.measurements.pop()
        self._refresh_records()
        self.flush(final=True)

    def clear_measurements(self):
        self.pending.clear()
        self.measurements.clear()
        self._refresh_records()
        self.flush(final=True)

    def dispose(self):
        self.timer.stop()
        self.finish_timer.stop()
        self.vtk_widget.removeEventFilter(self.gestures)
        self.vtk_widget.Finalize()
