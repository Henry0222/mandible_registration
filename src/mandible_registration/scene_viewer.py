from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget, QColorDialog,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkRegularPolygonSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkBillboardTextActor3D, vtkFollower,
    vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,
)
from vtkmodules.vtkRenderingUI import vtkGenericRenderWindowInteractor
# Register the OpenGL rendering backend in frozen as well as source runs.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from .scene_data import (
    BONE_KEYS, COMPARISON_KEYS, CT_INSPECTION_KEYS, JOINT_VIEW_KEYS, MODEL_SPECS,
    SceneData, SceneModel,
    load_scene,
)
from .theme import apply_light_theme
from .view_interaction import ViewGestures
from .section_geometry import _rotate, unit


def color_swatch_icon(rgb, size: int = 14) -> QIcon:
    """Small bordered square used as an exact visual key for model colors."""
    color = QColor.fromRgbF(*(float(value) for value in rgb))
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor(72, 84, 101), 1))
    painter.setBrush(color)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.end()
    return QIcon(pixmap)


def _mesh_arrays(mesh):
    if isinstance(mesh, tuple):
        return mesh
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), normals


def mesh_polydata(mesh) -> vtkPolyData:
    """Preserve mesh coordinates and face IDs for reproducible picking."""
    vertices, triangles, mesh_normals = _mesh_arrays(mesh)
    faces = np.asarray(triangles, dtype=np.int64)
    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(faces.ravel(), deep=True),
    )
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    if mesh_normals is not None and len(mesh_normals) == len(vertices):
        normals = numpy_to_vtk(np.asarray(mesh_normals), deep=True)
        normals.SetName("Normals")
        data.GetPointData().SetNormals(normals)
    return data


class OffscreenVtkWidget(QWidget):
    """Test-only hidden VTK window: Qt's offscreen plugin has no valid HWND."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(800, 700)
        self.interactor = vtkGenericRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

    def GetRenderWindow(self):
        return self.window

    def Initialize(self):
        self.interactor.Initialize()

    def Finalize(self):
        self.window.Finalize()


class TaskThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.completed.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))


class SceneViewer(QMainWindow):
    def __init__(self, parent=None, *, preset: str = "bones", offscreen: bool = False, reusable=False):
        apply_light_theme()
        super().__init__(None, Qt.WindowType.Window)
        self._reusable = reusable
        self.setWindowTitle("颌骨对比 · 髁突剖面")
        self.resize(1440, 940)
        self.scene: SceneData | None = None
        self.models: dict[str, SceneModel] = {}
        self.actors: dict[str, vtkActor] = {}
        self.polydata: dict[str, vtkPolyData] = {}
        self._rotation_center = None
        self._overlays = []
        self._thread = None
        self._preset = preset
        self._inspection_active = None
        self._marker_radius = 0.1  # physical millimetres, independent of mesh/camera size
        self._task_result = None
        self._task_callback = None
        self.condyle_report = None
        self._condyle_measurements = []
        self._offscreen = offscreen
        self.section_views = {}
        self.section_rois = {}
        self.section_sources = {}
        self._section_planes = {}
        self._sections_enabled = False
        self._default_layout_applied = False
        self._section_render_timer = QTimer(self)
        self._section_render_timer.setSingleShot(True)
        self._section_render_timer.setInterval(33)
        self._section_render_timer.timeout.connect(self.render_sections_in_3d)

        splitter = QSplitter()
        self.setCentralWidget(splitter)
        self.sidebar = QWidget()
        side = QVBoxLayout(self.sidebar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sidebar)
        scroll.setMinimumWidth(240)
        splitter.addWidget(scroll)

        self.models_group = models_group = QGroupBox("模型显隐")
        model_layout = QVBoxLayout(models_group)
        self.model_list = QListWidget()
        self.model_list.setIconSize(QSize(14, 14))
        self.model_list.setMinimumHeight(210)
        self.model_list.itemChanged.connect(self._visibility_changed)
        self.model_list.currentItemChanged.connect(self._selected_model)
        model_layout.addWidget(self.model_list)
        buttons = QGridLayout()
        for index, (text, callback) in enumerate((
            ("下颌骨", lambda: self._set_preset("bones")),
            ("关节", lambda: self._set_preset("joint")),
            ("CT 配准", lambda: self._set_preset("ct")),
            ("隐藏", lambda: self._set_preset("none")),
        )):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button, index // 2, index % 2)
        model_layout.addLayout(buttons)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self._change_opacity)
        opacity_hint = QLabel("选中模型透明度（左侧透明，右侧不透明）")
        opacity_hint.setWordWrap(True)
        model_layout.addWidget(opacity_hint)
        model_layout.addWidget(self.opacity)
        color = QPushButton("更改选中模型的颜色")
        color.clicked.connect(self._change_color)
        model_layout.addWidget(color)
        side.addWidget(models_group)

        condyles = QGroupBox("髁突中心 · 自动位移")
        condyle_layout = QVBoxLayout(condyles)
        self.condyle_info = QLabel("可在主界面“颌骨”的选区按钮中指定左右髁突。")
        self.condyle_info.setWordWrap(True)
        info_policy = self.condyle_info.sizePolicy()
        info_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.condyle_info.setSizePolicy(info_policy)
        self.condyle_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        condyle_layout.addWidget(self.condyle_info)
        self.show_condyles = QCheckBox("显示中心位移线")
        self.show_condyles.setChecked(True)
        self.show_condyles.toggled.connect(self._redraw_measurements)
        condyle_layout.addWidget(self.show_condyles)
        side.addWidget(condyles)

        sections = QGroupBox("双侧髁突剖面")
        section_controls = QVBoxLayout(sections)
        self.section_button = QPushButton("启用双侧剖面")
        self.section_button.clicked.connect(self.enable_sections)
        section_controls.addWidget(self.section_button)
        side.addWidget(sections)
        self.export_button = QPushButton("导出髁突位移 / 剖面测量 JSON")
        self.export_button.clicked.connect(self._save_measurements)
        side.addWidget(self.export_button)
        side.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        toolbar = QHBoxLayout()
        reset = QPushButton("重置视角")
        reset.clicked.connect(self._reset_camera)
        toolbar.addWidget(reset)
        for axis in ("+X", "-X", "+Y", "-Y", "+Z"):
            button = QPushButton(f"从 {axis} 查看")
            button.clicked.connect(lambda checked=False, value=axis: self._axis_view(value))
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        right_layout.addLayout(toolbar)
        self.vtk_widget = OffscreenVtkWidget(right) if offscreen else QVTKRenderWindowInteractor(right)
        self.render_window = self.vtk_widget.GetRenderWindow()
        if offscreen:
            self.render_window.SetOffScreenRendering(1)
        self.render_window.SetMultiSamples(0)
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetNumberOfLayers(2)
        self.renderer = vtkRenderer()
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.renderer.SetBackground(0.08, 0.105, 0.15)
        self.renderer.SetBackground2(0.19, 0.23, 0.30)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.1)
        self.render_window.AddRenderer(self.renderer)
        # Measurements are an overlay, so a line crossing the bone does not hide
        # its value or endpoints. Picking still uses only the real surface layer.
        self.annotation_renderer = vtkRenderer()
        self.annotation_renderer.SetLayer(1)
        self.annotation_renderer.SetInteractive(False)
        self.annotation_renderer.SetPreserveDepthBuffer(False)
        self.annotation_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        self.render_window.AddRenderer(self.annotation_renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleUser())
        self.orientation_axes = vtkAxesActor()
        self.orientation_axes.SetShaftTypeToLine()
        self.orientation_axes.SetTotalLength(1.0, 1.0, 1.0)
        self.orientation_axes.SetNormalizedShaftLength(0.82, 0.82, 0.82)
        self.orientation_axes.SetNormalizedTipLength(0.18, 0.18, 0.18)
        for actor in (
            self.orientation_axes.GetXAxisCaptionActor2D(),
            self.orientation_axes.GetYAxisCaptionActor2D(),
            self.orientation_axes.GetZAxisCaptionActor2D(),
        ):
            actor.GetCaptionTextProperty().BoldOn()
            actor.GetCaptionTextProperty().SetFontSize(16)
        self.orientation_widget = vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(self.orientation_axes)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.84, 0.02, 0.99, 0.18)
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()
        self.gestures = ViewGestures(self)
        self.vtk_widget.installEventFilter(self.gestures)
        self._camera_timer = QTimer(self)
        self._camera_timer.setSingleShot(True)
        self._camera_timer.setInterval(16)
        self._camera_timer.timeout.connect(self.render_window.Render)
        self._camera_finish_timer = QTimer(self)
        self._camera_finish_timer.setSingleShot(True)
        self._camera_finish_timer.setInterval(140)
        self._camera_finish_timer.timeout.connect(self.finish_interaction)
        self.view_splitter = QSplitter(Qt.Orientation.Vertical)
        self.view_splitter.addWidget(self.vtk_widget)
        lower = QSplitter(Qt.Orientation.Horizontal)
        self.section_cards, self.section_labels, self.section_layouts = {}, {}, {}
        for key, title in (("left", "左侧髁突剖面"), ("right", "右侧髁突剖面")):
            card = QGroupBox(title)
            card.setMinimumWidth(300)
            content = QVBoxLayout(card)
            label = QLabel("等待加载项目…")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            content.addWidget(label)
            self.section_cards[key], self.section_labels[key], self.section_layouts[key] = card, label, content
            lower.addWidget(card)
        self.view_splitter.addWidget(lower)
        self.view_splitter.setStretchFactor(0, 1)
        self.view_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(self.view_splitter, 1)
        legend = QLabel("左拖旋转 · 中拖平移 · 右拖缩放 · 三维滚轮缩放 / 剖面滚轮移层")
        legend.setWordWrap(True)
        right_layout.addWidget(legend)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage("等待加载配准结果…")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._default_layout_applied:
            self._default_layout_applied = True
            # Apply after the first layout has its actual window dimensions.
            # Reopening a cached window preserves the user's divider positions.
            QTimer.singleShot(0, self._set_default_view_sizes)

    def _set_default_view_sizes(self):
        splitter = self.centralWidget()
        width = max(splitter.width() - splitter.handleWidth(), 1)
        splitter.setSizes([width // 5, width - width // 5])
        height = max(self.view_splitter.height() - self.view_splitter.handleWidth(), 1)
        self.view_splitter.setSizes([height * 3 // 5, height - height * 3 // 5])

    def _run_task(self, operation, callback, message):
        if self._thread is not None:
            return
        self.sidebar.setEnabled(False)
        self.statusBar().showMessage(message)
        self._thread = TaskThread(operation, self)
        self._task_result = None
        self._task_callback = callback
        self._thread.completed.connect(self._store_task_result)
        self._thread.failed.connect(self._task_failed)
        self._thread.finished.connect(self._task_finished)
        self._thread.start()

    @Slot(str)
    def _task_failed(self, message):
        self.statusBar().showMessage(f"操作失败：{message}")
        QMessageBox.warning(self, "无法完成操作", message)

    @Slot(object)
    def _store_task_result(self, result):
        self._task_result = result

    @Slot()
    def _task_finished(self):
        # Do not create/render VTK actors while Qt is still unwinding the
        # QThread.finished signal. Some Windows OpenGL drivers abort there.
        QTimer.singleShot(0, self, self._deliver_task_result)

    @Slot()
    def _deliver_task_result(self):
        thread = self._thread
        self.sidebar.setEnabled(True)
        result, callback = self._task_result, self._task_callback
        self._thread = None
        self._task_result = self._task_callback = None
        try:
            if result is not None and callback is not None:
                callback(result)
        except Exception as exc:
            self._task_failed(str(exc))
        finally:
            if thread is not None:
                thread.deleteLater()

    def load_project(self, path):
        keys = CT_INSPECTION_KEYS if self._preset == "ct" else BONE_KEYS
        self._run_task(
            lambda: load_scene(path, keys=keys, array_meshes=True),
            self.set_scene,
            "正在读取并校验两个颌骨；其余模型按需加载…",
        )

    @Slot(object)
    def set_scene(self, scene: SceneData):
        self._dispose_sections()
        for actor in self.actors.values():
            self.renderer.RemoveActor(actor)
        self.actors.clear()
        self.polydata.clear()
        self.condyle_report = None
        self._rotation_center = None
        self._inspection_active = None
        self.scene = scene
        allowed = set(COMPARISON_KEYS) | set(CT_INSPECTION_KEYS)
        self.models = {model.key: model for model in scene.models if model.key in allowed}
        self._add_model_actors(self.models.values())
        self._set_preset(self._preset)
        self.reload_condyles()
        self.statusBar().showMessage(f"已加载 {len(scene.models)} 个模型；其余模型勾选后加载。单位 mm。")
        if scene.warnings:
            QMessageBox.warning(self, "部分数据不可用", "\n".join(scene.warnings))

    def reload_condyles(self):
        if not self.scene:
            return
        from .condyles import project_analysis
        self.condyle_report = None
        self._condyle_measurements = []
        try:
            self.condyle_report = project_analysis(self.scene.project_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.condyle_info.setText(f"髁突中心不可用：{exc}")
        else:
            if not self.condyle_report or not self.condyle_report["regions"]:
                self.condyle_info.setText("尚未选取髁突面片。可在主界面“颌骨”的选区按钮中指定左右髁突。")
            else:
                from .condyles import displacement_description
                lines = []
                for key, region in self.condyle_report["regions"].items():
                    vector = region["displacement_xyz_mm"]
                    angles = region["direction_angles_to_positive_xyz_deg"]
                    lines.append(f'<p style="color:#111111;font-size:15px"><b>{region["name"]} · 位移 {region["distance_mm"]:.3f} mm</b><br/>'
                                 f'{displacement_description(key, vector)}</p>')
                    lines.append('<p style="color:#7a818b;font-size:11px">'
                                 f'ΔX {vector[0]:+.3f} / ΔY {vector[1]:+.3f} / ΔZ {vector[2]:+.3f} mm<br/>'
                                 '与 +X / +Y / +Z 夹角：' + (' / '.join(f'{angle:.2f}°' for angle in angles) if angles else '零位移') + '</p>')
                    self._condyle_measurements.append({"id": key, "type": "length", "value": region["distance_mm"], "unit": "mm",
                        "anchors": [{"model": bone, "xyz_mm": region[f"center_t{index}_mm"], "triangle_id": -1}
                                    for index, bone in enumerate(BONE_KEYS)]})
                rotation = self.condyle_report['rotation_minus_x_view']
                angle = rotation['signed_degrees']
                value = f' {abs(angle):.2f}°' if angle is not None and abs(angle) >= .005 else ''
                lines.append(f'<p style="color:#111111;font-size:15px"><b>{rotation["label"]}{value}</b></p>')
                lines.append(f'<p style="color:#7a818b;font-size:11px">三维总旋转：{self.condyle_report["rigid_rotation_degrees"]:.2f}°</p>')
                self.condyle_info.setText(''.join(lines))
        regions = (self.condyle_report or {}).get("regions", {})
        center = np.mean([r["center_t0_mm"] for r in regions.values()], axis=0) if regions else None
        if center is not None and (self._rotation_center is None or not np.allclose(center, self._rotation_center)):
            self._rotation_center = center
            self._reset_camera()
        self._redraw_measurements()
        self._update_section_availability()

    def _update_section_availability(self):
        # Invalidate ROI caches only when the saved centers change.
        regions = (self.condyle_report or {}).get("regions", {})
        centers = {key: region["center_t0_mm"] for key, region in regions.items()}
        previous = {key: view.state.center.tolist() for key, view in self.section_views.items()}
        if self.section_views and centers != previous:
            self._dispose_sections()
        self.section_button.setEnabled(bool(regions) and not self._sections_enabled)
        self.section_button.setText("剖面已启用" if self._sections_enabled else "启用双侧剖面")
        for key, label in self.section_labels.items():
            if key not in regions:
                label.setText("此侧未保存髁突选区。\n可在主界面选择此侧面片；另一侧仍可使用。")
            elif key not in self.section_views:
                label.setText("已保存此侧髁突选区。\n点击左侧“启用双侧剖面”加载 20 mm 局部截线。")

    def enable_sections(self):
        if self._sections_enabled or self._thread is not None or not self.scene:
            return
        regions = (self.condyle_report or {}).get("regions", {})
        if not regions:
            self._update_section_availability()
            return
        # Optional maxilla is requested only if actually present in this project.
        available = self.models.keys() | self.scene.deferred_keys
        keys = tuple(key for key in JOINT_VIEW_KEYS if key in available)
        if set(keys) - self.models.keys():
            self._ensure_models(keys, self.enable_sections)
            return
        from .section_geometry import sphere_roi
        arrays = {key: _mesh_arrays(self.models[key].mesh) for key in keys}
        centers = {key: region["center_t0_mm"] for key, region in regions.items()}
        self.section_sources = {key: self.polydata[key] for key in keys}

        def build():
            # NumPy only in worker: no render windows or actors cross Qt threads.
            return {side: {key: sphere_roi(vertices, triangles, center)
                           for key, (vertices, triangles, _) in arrays.items()}
                    for side, center in centers.items()}

        def ready(rois):
            from .section_viewer import SectionView, SIDE_COLORS
            self.section_rois = rois
            for side, side_rois in rois.items():
                view = SectionView(side, centers[side], side_rois, self.actors,
                                   offscreen=self._offscreen, parent=self.section_cards[side])
                self.section_layouts[side].addWidget(view)
                self.section_labels[side].hide()
                self.section_views[side] = view
                disk = vtkRegularPolygonSource()
                disk.SetNumberOfSides(64)
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(disk.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*SIDE_COLORS[side])
                actor.GetProperty().SetOpacity(.20)
                actor.GetProperty().LightingOff()
                actor.GetProperty().EdgeVisibilityOn()
                actor.GetProperty().SetEdgeColor(*SIDE_COLORS[side])
                actor.GetProperty().SetLineWidth(2)
                actor.PickableOff()
                # Exclude the planes from camera-fit bounds and from model picking.
                actor.UseBoundsOff()
                self.renderer.AddActor(actor)
                self._section_planes[side] = (disk, actor)
                view.plane_changed.connect(self._section_changed)
                self._section_changed(side, False)
            self.finish_interaction()
            self._sections_enabled = True
            self._update_section_availability()
            if "ct_maxilla_t0" not in keys:
                self.statusBar().showMessage("剖面已启用；此项目未提供固定上颌骨，仅显示两次颌骨截线。")
            else:
                self.statusBar().showMessage("剖面已启用。中键平移视野，右键拖动缩放；滚轮沿视线移动剖面。")

        self._run_task(build, ready, "正在从已加载网格缓存双侧 20 mm ROI…")

    def _section_changed(self, side, final=False):
        view = self.section_views.get(side)
        if view is None or side not in self._section_planes:
            return
        disk, actor = self._section_planes[side]
        disk.SetCenter(*view.state.origin)
        disk.SetNormal(*view.state.normal)
        disk.SetRadius(max(view.state.disk_radius, 1e-6))
        self.renderer.SetUseDepthPeeling(bool(final))
        if final:
            self._section_render_timer.stop()
            self.render_sections_in_3d()
        elif not self._section_render_timer.isActive():
            self._section_render_timer.start()

    def render_sections_in_3d(self):
        if not hasattr(self, "render_window"):
            return
        for side, (_, actor) in self._section_planes.items():
            actor.SetVisibility(self.section_views[side].state.disk_radius > 1e-6)
        self.render_window.Render()

    def _dispose_sections(self):
        self._section_render_timer.stop()
        for view in self.section_views.values():
            view.dispose()
            view.setParent(None)
            view.deleteLater()
        self.section_views.clear()
        self.section_rois.clear()
        self.section_sources.clear()
        for _, actor in self._section_planes.values():
            self.renderer.RemoveActor(actor)
        self._section_planes.clear()
        self._sections_enabled = False
        for label in self.section_labels.values():
            label.show()

    def _add_model_actors(self, models):
        for model in models:
            data = mesh_polydata(model.mesh)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(data)
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            # Unchecked items may never emit itemChanged during the initial preset.
            actor.SetVisibility(False)
            actor.GetProperty().SetColor(*model.color)
            actor.GetProperty().SetAmbient(0.20)
            actor.GetProperty().SetDiffuse(0.75)
            actor.GetProperty().SetSpecular(0.12)
            actor.GetProperty().SetSpecularPower(24)
            self.renderer.AddActor(actor)
            self.actors[model.key] = actor
            self.polydata[model.key] = data

    def _ensure_models(self, keys, callback):
        missing = set(keys) - self.models.keys()
        if not missing:
            callback()
            return
        if not self.scene or self._thread is not None:
            return

        def loaded(scene):
            self._add_model_actors(scene.models)
            self.models.update({model.key: model for model in scene.models})
            self.scene.models.extend(scene.models)
            self.scene.deferred_keys.difference_update(model.key for model in scene.models)
            if scene.warnings:
                QMessageBox.warning(self, "部分数据不可用", "\n".join(scene.warnings))
            if missing <= self.models.keys():
                callback()

        self._run_task(
            lambda: load_scene(self.scene.project_path, keys=missing, array_meshes=True),
            loaded,
            "正在按需读取所选模型…",
        )

    def _set_preset(self, preset):
        self._preset = preset
        required = (
            CT_INSPECTION_KEYS if preset == "ct"
            else JOINT_VIEW_KEYS if preset == "joint"
            else BONE_KEYS if preset == "bones"
            else ()
        )
        if self.scene and set(required) & self.scene.deferred_keys:
            self._ensure_models(required, lambda: self._set_preset(preset))
            return
        inspection = (preset == "ct") if preset != "none" else bool(self._inspection_active)
        if inspection != self._inspection_active:
            self._inspection_active = inspection
            for actor in self.actors.values():
                actor.SetVisibility(False)
            self.model_list.blockSignals(True)
            self.model_list.clear()
            keys = CT_INSPECTION_KEYS if inspection else COMPARISON_KEYS
            for key in keys:
                model = self.models.get(key)
                if model is None and (not self.scene or key not in self.scene.deferred_keys):
                    continue
                title = model.title if model else next(title for name, title, _ in MODEL_SPECS if name == key)
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(str(model.path) if model else "勾选后加载此模型")
                color = model.color if model else next(
                    value for name, _title, value in MODEL_SPECS if name == key
                )
                item.setIcon(color_swatch_icon(color))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.model_list.addItem(item)
            self.model_list.blockSignals(False)
            height = max(self.model_list.sizeHintForRow(0), 22) * self.model_list.count() + 12
            self.model_list.setFixedHeight(max(height, 60))
            self.model_list.setCurrentRow(0)
            self.models_group.setTitle(
                "CT 配准检查"
                if inspection
                else "模型显隐"
            )
            self.annotation_renderer.SetDraw(not inspection)
            self._redraw_measurements()
        keys = (
            set(BONE_KEYS) if preset == "bones"
            else set(JOINT_VIEW_KEYS) & self.models.keys() if preset == "joint"
            else {"baseline_lower", "ct_dentition_t0"} if preset == "ct"
            else set()
        )
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            visible = item.data(Qt.ItemDataRole.UserRole) in keys
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self.render_sections_in_3d()
        self._reset_camera()

    def _visibility_changed(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key not in self.actors and item.checkState() == Qt.CheckState.Checked:
            self._ensure_models((key,), lambda: self._visibility_changed(item))
            return
        if key in self.actors:
            self.actors[key].SetVisibility(item.checkState() == Qt.CheckState.Checked)
            self.render_window.Render()

    def _selected_model(self, current, previous):
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        if key in self.actors:
            self.opacity.blockSignals(True)
            self.opacity.setValue(round(self.actors[key].GetProperty().GetOpacity() * 100))
            self.opacity.blockSignals(False)

    def _change_opacity(self, value):
        item = self.model_list.currentItem()
        if item and item.data(Qt.ItemDataRole.UserRole) in self.actors:
            self.actors[item.data(Qt.ItemDataRole.UserRole)].GetProperty().SetOpacity(value / 100)
            self.render_window.Render()

    def _change_color(self):
        item = self.model_list.currentItem()
        if not item or item.data(Qt.ItemDataRole.UserRole) not in self.actors:
            return
        actor = self.actors[item.data(Qt.ItemDataRole.UserRole)]
        color = QColorDialog.getColor(QColor.fromRgbF(*actor.GetProperty().GetColor()), self, "模型颜色")
        if color.isValid():
            actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())
            item.setIcon(color_swatch_icon((color.redF(), color.greenF(), color.blueF())))
            for view in self.section_views.values():
                view.sync_colors(self.actors)
            self.render_window.Render()

    def _reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera()
        if self._rotation_center is not None and not self._inspection_active:
            camera.OrthogonalizeViewUp()
            normal = -np.array(camera.GetDirectionOfProjection())
            bounds = self.renderer.ComputeVisiblePropBounds()
            if bounds[0] <= bounds[1]:
                corners = np.array([[x, y, z] for x in bounds[:2] for y in bounds[2:4] for z in bounds[4:]])
                up = unit(camera.GetViewUp())
                right = unit(np.cross(up, normal))
                relative = corners - self._rotation_center
                aspect = max(self.vtk_widget.width(), 1) / max(self.vtk_widget.height(), 1)
                # ParallelScale is the viewport's half-height in world units.
                # Fit around the condyle pivot, which can be far above the bone.
                extent = max(float(np.max(np.abs(relative @ up))),
                             float(np.max(np.abs(relative @ right))) / aspect)
                camera.SetParallelScale(max(extent * 1.06, .5))
                distance = max(camera.GetDistance(), float(np.max(relative @ normal)) + 1, 1)
                camera.SetFocalPoint(*self._rotation_center)
                camera.SetPosition(*(self._rotation_center + normal * distance))
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def rotate(self, dx, dy):
        camera = self.renderer.GetActiveCamera()
        pivot = self._rotation_center if self._rotation_center is not None and not self._inspection_active else np.array(camera.GetFocalPoint())
        up = unit(camera.GetViewUp())
        normal = -np.array(camera.GetDirectionOfProjection())
        right = unit(np.cross(up, normal))
        yaw, pitch = np.deg2rad([-dx * .4, -dy * .4])
        def turn(vector):
            return _rotate(_rotate(vector, up, yaw), _rotate(right, up, yaw), pitch)
        camera.SetPosition(*(pivot + turn(np.array(camera.GetPosition()) - pivot)))
        camera.SetFocalPoint(*(pivot + turn(np.array(camera.GetFocalPoint()) - pivot)))
        camera.SetViewUp(*turn(up))
        camera.OrthogonalizeViewUp()
        self._schedule_camera()

    def pan(self, dx, dy):
        camera = self.renderer.GetActiveCamera()
        up = unit(camera.GetViewUp())
        right = unit(np.cross(up, -np.array(camera.GetDirectionOfProjection())))
        height_mm = 2 * camera.GetParallelScale()
        shift = (-dx * right + dy * up) * height_mm / max(self.vtk_widget.height(), 1)
        camera.SetPosition(*(np.array(camera.GetPosition()) + shift))
        camera.SetFocalPoint(*(np.array(camera.GetFocalPoint()) + shift))
        self._schedule_camera()

    def zoom(self, factor, *, interactive=False):
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelScale(float(np.clip(camera.GetParallelScale() * factor, .5, 1000)))
        self._schedule_camera() if interactive else self.finish_interaction()

    def scroll(self, steps):
        self.zoom(float(np.exp(np.clip(-steps * .12, -2, 2))), interactive=True)

    def pick_qt_position(self, position):
        return False  # The upper 3D view no longer offers manual point picking.

    def _schedule_camera(self):
        self.renderer.SetUseDepthPeeling(False)
        self.renderer.ResetCameraClippingRange()
        if not self._camera_timer.isActive():
            self._camera_timer.start()
        self._camera_finish_timer.start()

    def finish_interaction(self):
        self._camera_timer.stop()
        self._camera_finish_timer.stop()
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def has_section_measurements(self):
        return any(view.measurements or view.pending for view in self.section_views.values())

    def _axis_view(self, axis):
        direction = str(axis).strip().upper()
        sign = -1.0 if direction.startswith("-") else 1.0
        name = direction.lstrip("+-")
        if name not in "XYZ" or len(name) != 1:
            raise ValueError(f"未知查看方向：{axis}")
        camera = self.renderer.GetActiveCamera()
        center = np.asarray(camera.GetFocalPoint())
        vector = sign * np.eye(3)["XYZ".index(name)]
        camera.SetPosition(*(center + vector * max(camera.GetDistance(), 1)))
        camera.SetViewUp(*((0, 1, 0) if name == "Z" else (0, 0, 1)))
        self._reset_camera()

    def _surface_circle(self, anchor, color):
        mesh = self.models[anchor["model"]].mesh
        vertices, triangles, _ = _mesh_arrays(mesh)
        face_id = anchor["triangle_id"]
        normal = np.array([0., 0., 1.])
        if 0 <= face_id < len(triangles):
            a, b, c = np.asarray(vertices)[np.asarray(triangles)[face_id]]
            cross = np.cross(b - a, c - a)
            if np.linalg.norm(cross) > 1e-12:
                normal = cross / np.linalg.norm(cross)
        circle = vtkRegularPolygonSource()
        circle.SetNumberOfSides(48)
        circle.SetCenter(*(anchor["xyz_mm"] if face_id >= 0 else (0, 0, 0)))
        circle.SetNormal(*normal)
        circle.SetRadius(self._marker_radius)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(circle.GetOutputPort())
        actor = vtkActor() if face_id >= 0 else vtkFollower()
        actor.SetMapper(mapper)
        if face_id < 0:  # A surface centroid can lie inside the selected patch.
            actor.SetCamera(self.renderer.GetActiveCamera())
            actor.SetPosition(*anchor["xyz_mm"])
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().LightingOff()
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(1, 1, 1)
        self._add_overlay(actor)

    def _text(self, point, text, offset=(8, 8), font_size=13):
        actor = vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(*point)
        actor.SetDisplayOffset(*offset)
        actor.GetTextProperty().SetFontSize(font_size)
        actor.GetTextProperty().SetColor(1, 1, 0.88)
        actor.GetTextProperty().SetBackgroundColor(0.06, 0.07, 0.10)
        actor.GetTextProperty().SetBackgroundOpacity(0.7)
        self._add_overlay(actor)

    def _line(self, first, second):
        source = vtkLineSource()
        source.SetPoint1(*first)
        source.SetPoint2(*second)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0.96, 0.36)
        actor.GetProperty().SetLineWidth(3)
        self._add_overlay(actor)

    def _add_overlay(self, actor):
        actor.PickableOff()
        self.annotation_renderer.AddActor(actor)
        self._overlays.append(actor)

    def _redraw_measurements(self):
        if not hasattr(self, "renderer"):
            return
        for actor in self._overlays:
            self.annotation_renderer.RemoveActor(actor)
        self._overlays.clear()
        records = []
        if self.show_condyles.isChecked() and set(BONE_KEYS) <= self.models.keys():
            records.extend(self._condyle_measurements)
        for record in records:
            points = [np.asarray(anchor["xyz_mm"]) for anchor in record["anchors"]]
            for index, point in enumerate(points):
                key = record["anchors"][index]["model"]
                bone = "T0" if key == BONE_KEYS[0] else "T1"
                self._surface_circle(record["anchors"][index], (0.22, 0.58, 0.90) if bone == "T0" else (1.0, 0.58, 0.20))
            segments = ((0, 1), (2, 3)) if record.get("type") == "angle" else ((0, 1),)
            for first, second in segments:
                if second < len(points):
                    self._line(points[first], points[second])
            if record.get("value") is not None:
                center = np.mean(points, axis=0)
                unit = "°" if record["unit"] == "deg" else record["unit"]
                self._text(center, f"{record['value']:.3f} {unit}", (8, 12), 15)
        self.render_window.Render()

    def measurement_payload(self):
        return {
            "schema_version": 4, "created_at": datetime.now().astimezone().isoformat(),
            "project": str(self.scene.project_path) if self.scene else None,
            "coordinate_reference": self.scene.coordinate_reference if self.scene else None,
            "coordinate_units": "mm", "length_definition": "Euclidean distance between manually selected visible section contour points",
            "models": {key: {"path": str(self.models[key].path), "sha256": self.models[key].sha256}
                       for key in JOINT_VIEW_KEYS if key in self.models},
            "measurements": [],  # Reserved legacy 3D measurement field.
            "condyle_analysis": self.condyle_report,
            "condyle_sections": {
                "definition": "manual distances on real mesh contours; no inferred fossa surface",
                "sides": {key: {"plane": view.state.snapshot(), "measurements": view.measurements}
                          for key, view in self.section_views.items()},
            },
        }

    def _save_measurements(self):
        if (not (self.condyle_report and self.condyle_report["regions"])
                and not any(v.measurements for v in self.section_views.values())):
            QMessageBox.information(self, "没有测量记录", "请先选择髁突或完成剖面距离测量。")
            return
        directory = self.scene.project_path.parent if self.scene else Path.cwd()
        default = directory / f"measurements_{datetime.now():%Y%m%d_%H%M%S}.json"
        filename, _ = QFileDialog.getSaveFileName(self, "导出测量记录", str(default), "测量记录 (*.json)")
        if not filename:
            return
        path = Path(filename).resolve()
        # Never allow a measurement export to overwrite registration data or mesh inputs.
        protected = {model.path.resolve() for model in self.models.values()}
        if self.scene:
            protected.add(self.scene.project_path.resolve())
        if path in protected or path.name in ("summary.json", "project.json") or path.suffix.lower() != ".json":
            QMessageBox.warning(self, "无法保存到此位置", "请使用独立的测量 JSON 文件，不要覆盖模型或配准项目。")
            return
        try:
            # Exclusive creation also protects matrices and other existing audit files.
            with path.open("x", encoding="utf-8") as handle:
                json.dump(self.measurement_payload(), handle, ensure_ascii=False, indent=2, allow_nan=False)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", f"{exc}\n请使用一个尚不存在的新文件名。")
            return
        self.statusBar().showMessage(f"测量已导出：{path}")

    def closeEvent(self, event):
        if self._thread is not None:
            QMessageBox.information(self, "正在处理", "请等待模型加载完成后再关闭。")
            event.ignore()
            return
        if self._reusable:
            self.hide()
            event.ignore()
            return
        if self.has_section_measurements():
            answer = QMessageBox.question(self, "关闭查看器", "关闭后当前标记不会自动保存。确认关闭？如需保存请先导出测量记录。")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._camera_timer.stop()
        self._camera_finish_timer.stop()
        self._dispose_sections()
        self.vtk_widget.Finalize()
        super().closeEvent(event)


def view_project(path: str | Path, *, preset: str = "bones") -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SceneViewer(preset=preset)
    window.show()
    window.vtk_widget.Initialize()
    window.load_project(path)
    return app.exec()
