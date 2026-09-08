from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget, QColorDialog,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkRegularPolygonSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkBillboardTextActor3D, vtkCellPicker, vtkFollower,
    vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,
)
from vtkmodules.vtkRenderingUI import vtkGenericRenderWindowInteractor
# Register the OpenGL rendering backend in frozen as well as source runs.
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from .scene_data import (
    BONE_KEYS, COMPARISON_KEYS, CT_INSPECTION_KEYS, MODEL_SPECS, SceneData, SceneModel,
    bone_pick_keys, segment_angle_degrees, length_mm, load_scene, point_array,
)
from .theme import apply_light_theme


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


class SurfaceClickFilter(QObject):
    """Observe Qt clicks without stealing VTK camera gestures or release focus."""

    clicked = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self._press = None
        self._dragged = False

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick) and event.button() == Qt.MouseButton.LeftButton:
            self._press = QPointF(event.position())
            self._dragged = False
        elif kind == QEvent.Type.MouseMove and self._press is not None:
            delta = event.position() - self._press
            self._dragged |= delta.x() ** 2 + delta.y() ** 2 > 16
        elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            position = QPointF(event.position())
            if self._press is not None:
                delta = position - self._press
                is_click = not self._dragged and delta.x() ** 2 + delta.y() ** 2 <= 16
                if is_click:
                    # Let QVTK finish its camera release before querying the surface.
                    QTimer.singleShot(0, self, lambda: self.clicked.emit(position))
            self._press = None
        elif kind in (QEvent.Type.Hide, QEvent.Type.Leave):
            self._press = None
        return False


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
        super().__init__(parent, Qt.WindowType.Window)
        self._reusable = reusable
        self.setWindowTitle("三维模型查看 · 标记测量")
        self.resize(1440, 940)
        self.scene: SceneData | None = None
        self.models: dict[str, SceneModel] = {}
        self.actors: dict[str, vtkActor] = {}
        self.polydata: dict[str, vtkPolyData] = {}
        self.measurements: list[dict] = []
        self.pending: list[dict] = []
        self._overlays = []
        self._next_measurement = 1
        self._thread = None
        self._preset = preset
        self._inspection_active = None
        self._marker_radius = 0.1  # physical millimetres, independent of mesh/camera size
        self._task_result = None
        self._task_callback = None
        self.condyle_report = None
        self._condyle_measurements = []

        splitter = QSplitter()
        self.setCentralWidget(splitter)
        self.sidebar = QWidget()
        side = QVBoxLayout(self.sidebar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sidebar)
        scroll.setMinimumWidth(345)
        splitter.addWidget(scroll)

        self.models_group = models_group = QGroupBox("模型显隐（四个口扫 + 两个全牙列 + 两个颌骨）")
        model_layout = QVBoxLayout(models_group)
        self.model_list = QListWidget()
        self.model_list.setIconSize(QSize(14, 14))
        self.model_list.setMinimumHeight(210)
        self.model_list.itemChanged.connect(self._visibility_changed)
        self.model_list.currentItemChanged.connect(self._selected_model)
        model_layout.addWidget(self.model_list)
        buttons = QHBoxLayout()
        for text, callback in (("只看下颌骨", lambda: self._set_preset("bones")),
                               ("只看 CT 配准", lambda: self._set_preset("ct")),
                               ("全部隐藏", lambda: self._set_preset("none"))):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        model_layout.addLayout(buttons)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self._change_opacity)
        model_layout.addWidget(QLabel("选中模型透明度（左侧透明，右侧不透明）"))
        model_layout.addWidget(self.opacity)
        color = QPushButton("更改选中模型的颜色")
        color.clicked.connect(self._change_color)
        model_layout.addWidget(color)
        side.addWidget(models_group)

        condyles = QGroupBox("髁突中心 · 自动位移")
        condyle_layout = QVBoxLayout(condyles)
        self.condyle_info = QLabel("可在主界面“颌骨”的选区按钮中指定左右髁突。")
        self.condyle_info.setWordWrap(True)
        self.condyle_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        condyle_layout.addWidget(self.condyle_info)
        self.show_condyles = QCheckBox("显示中心位移线")
        self.show_condyles.setChecked(True)
        self.show_condyles.toggled.connect(self._redraw_measurements)
        condyle_layout.addWidget(self.show_condyles)
        side.addWidget(condyles)

        self.measure_group = measure = QGroupBox("两次下颌骨 · 标记与测量")
        measure_layout = QVBoxLayout(measure)
        measure_form = QFormLayout()
        self.measure_mode = QComboBox()
        for text, mode in (("浏览（不取点）", "browse"), ("下颌骨表面标记", "point"),
                           ("骨间距离：两次骨面各 1 点", "length"), ("骨间角度：两次骨面各 2 点", "angle")):
            self.measure_mode.addItem(text, mode)
        self.measure_mode.currentIndexChanged.connect(self._mode_changed)
        measure_form.addRow("操作模式", self.measure_mode)
        measure_layout.addLayout(measure_form)
        self.measure_hint = QLabel("只在两次下颌骨上取点，无需选择模型。")
        self.measure_hint.setWordWrap(True)
        self.measure_hint.setObjectName("measureHint")
        self.measure_hint.setMinimumHeight(90)
        measure_layout.addWidget(self.measure_hint)
        self.measure_list = QListWidget()
        self.measure_list.setMinimumHeight(115)
        measure_layout.addWidget(self.measure_list)
        actions = QHBoxLayout()
        for text, callback in (("撤销", self._undo), ("删除选中", self._delete_measurement),
                               ("清空", self._clear_measurements)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button)
        measure_layout.addLayout(actions)
        save = QPushButton("导出测量记录 JSON（含点坐标和模型来源）")
        save.clicked.connect(self._save_measurements)
        measure_layout.addWidget(save)
        side.addWidget(measure)
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
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
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
        # TrackballCamera grabs VTK focus on press, so an interactor release
        # observer is not reliable. Observe the real Qt event stream instead.
        self.click_filter = SurfaceClickFilter(self.vtk_widget)
        self.vtk_widget.installEventFilter(self.click_filter)
        self.click_filter.clicked.connect(self._pick_qt_position)
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.picker.PickFromListOn()
        right_layout.addWidget(self.vtk_widget, 1)
        legend = QLabel("左键拖动旋转 · 中键平移 · 滚轮缩放。测量只命中两次下颌骨，按步骤自动切换；请按提示选点。\n"
                        "模型单位按 mm 解释；这里显示的是刚性几何结果，不代表新的 CT 影像或临床诊断。")
        legend.setWordWrap(True)
        right_layout.addWidget(legend)
        splitter.addWidget(right)
        splitter.setSizes([410, 1030])
        self.statusBar().showMessage("等待加载配准结果…")

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
        if result is not None and callback is not None:
            callback(result)
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
                lines = ["所选面片的面积加权中心（不是球拟合中心）"]
                for key, region in self.condyle_report["regions"].items():
                    vector = region["displacement_xyz_mm"]
                    angles = region["direction_angles_to_positive_xyz_deg"]
                    lines.append(f"\n{region['name']}：位移 {region['distance_mm']:.3f} mm")
                    lines.append(f"ΔX {vector[0]:+.3f} / ΔY {vector[1]:+.3f} / ΔZ {vector[2]:+.3f} mm")
                    lines.append("与 +X / +Y / +Z 的方向角：" + (" / ".join(f"{angle:.2f}°" for angle in angles) if angles else "零位移，无方向"))
                    self._condyle_measurements.append({"id": key, "type": "length", "value": region["distance_mm"], "unit": "mm",
                        "anchors": [{"model": bone, "xyz_mm": region[f"center_t{index}_mm"], "triangle_id": -1}
                                    for index, bone in enumerate(BONE_KEYS)]})
                lines.append(f"\n整个颌骨的刚体总旋转角：{self.condyle_report['rigid_rotation_degrees']:.2f}°")
                lines.append("方向参考上颌口扫.1 的 XYZ，不等同于解剖方向；此旋转角不是单侧髁突独立旋转。")
                self.condyle_info.setText("\n".join(lines))
        self._redraw_measurements()

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
            locator = vtkStaticCellLocator()
            locator.SetDataSet(data)
            locator.BuildLocator()
            self.picker.AddLocator(locator)

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
        required = CT_INSPECTION_KEYS if preset == "ct" else BONE_KEYS if preset == "bones" else ()
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
                "CT 配准检查（全牙列.1 + 下颌口扫.1）"
                if inspection
                else "模型显隐（四个口扫 + 两个全牙列 + 两个颌骨）"
            )
            self.measure_mode.setCurrentIndex(0)
            self.pending.clear()
            self.measure_group.setEnabled(not inspection and set(BONE_KEYS) <= self.models.keys())
            self.annotation_renderer.SetDraw(not inspection)
            self._redraw_measurements()
            self._update_measure_hint()
        keys = {"ct_mandible_t0", "ct_mandible_t1"} if preset == "bones" else (
            {"baseline_lower", "ct_dentition_t0"} if preset == "ct" else set())
        for row in range(self.model_list.count()):
            item = self.model_list.item(row)
            visible = item.data(Qt.ItemDataRole.UserRole) in keys
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
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
            self.render_window.Render()

    def _reset_camera(self):
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

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

    def _mode_changed(self):
        self.pending.clear()
        if not self._inspection_active and self.measure_mode.currentData() != "browse":
            # Starting a bone measurement makes both bones available without a target selector.
            for row in range(self.model_list.count()):
                item = self.model_list.item(row)
                key = item.data(Qt.ItemDataRole.UserRole)
                if key in BONE_KEYS:
                    item.setCheckState(Qt.CheckState.Checked)
                    if self.actors[key].GetProperty().GetOpacity() <= 0:
                        self.actors[key].GetProperty().SetOpacity(1)
            self._selected_model(self.model_list.currentItem(), None)
        self._redraw_measurements()
        self._update_measure_hint()
        self.statusBar().showMessage("已切换操作模式，未完成的取点已清除。取点会在两次下颌骨间自动切换。")

    def _update_measure_hint(self):
        if self._inspection_active:
            text = "CT 配准检查中不取测量点。点击“只看下颌骨”返回骨间测量。"
        elif not set(BONE_KEYS) <= self.models.keys():
            text = "需要两次下颌骨模型，当前数据不足，不能测量骨间差距。"
        else:
            mode = self.measure_mode.currentData()
            targets = bone_pick_keys(mode, self.pending)
            if mode == "browse":
                text = "仅测量两个下颌骨，无需选择取点模型。距离在两次骨面各取 1 点；角度在两次骨面各画 1 条线。"
            elif mode == "point":
                text = "点击任一下颌骨表面放置标记。重叠处首点命中靠近相机的骨面，记录会注明 T0/T1。"
            elif not self.pending:
                text = ("第 1/2 点：点击任一下颌骨；下一点自动锁定另一骨面。距离为手选两点的直线长度，不会自动寻找对应解剖点。"
                        if mode == "length" else
                        "第 1/4 点：在任一下颌骨开始画线。先在该骨取 2 点，再自动切到另一骨取 2 点；两条线须按同一解剖方向取点。")
            else:
                title = self.models[targets[0]].title
                count = 2 if mode == "length" else 4
                text = f"第 {len(self.pending) + 1}/{count} 点：已自动锁定「{title}」。"
                text += "请点击另一骨面上的比较位置。" if mode == "length" else "请按两条线相同的解剖方向继续取点。"
                text += "即使两骨重叠，也只取当前这一次骨面。"
        self.measure_hint.setText(text)

    def _pick_qt_position(self, position):
        width, height = self.render_window.GetSize()
        # Use this widget's actual framebuffer size (including per-monitor DPI),
        # not the global cursor screen or the stale VTK event position.
        x = position.x() * width / max(self.vtk_widget.width(), 1)
        y = (self.vtk_widget.height() - 1 - position.y()) * height / max(self.vtk_widget.height(), 1)
        self.pick_at(x, y)

    def pick_at(self, x, y):
        """VTK display coordinates have their origin at the lower left."""
        if self._thread is not None or self._inspection_active or self.measure_mode.currentData() == "browse":
            return False
        if not set(BONE_KEYS) <= self.models.keys():
            self.statusBar().showMessage("需要两次下颌骨模型才能测量。")
            return False
        keys = bone_pick_keys(self.measure_mode.currentData(), self.pending)
        self.picker.InitializePickList()
        for key in keys:
            actor = self.actors[key]
            if actor.GetVisibility() and actor.GetProperty().GetOpacity() > 0:
                self.picker.AddPickList(actor)
        if not self.picker.GetPickList().GetNumberOfItems():
            self.statusBar().showMessage("当前步骤需要的下颌骨已隐藏或完全透明，请恢复显示后继续。")
            return False
        if not self.picker.Pick(float(x), float(y), 0, self.renderer):
            self.statusBar().showMessage("未点中当前步骤的下颌骨，请按提示旋转或放大后再试。")
            return False
        picked_actor = self.picker.GetActor()
        key = next((key for key in keys if self.actors[key] == picked_actor), None)
        if key is None:
            return False
        return self.add_anchor(key, self.picker.GetPickPosition(), self.picker.GetCellId())

    def add_anchor(self, key, xyz, cell_id):
        mode = self.measure_mode.currentData()
        if (self._inspection_active or not set(BONE_KEYS) <= self.models.keys()
                or key not in bone_pick_keys(mode, self.pending)):
            self.statusBar().showMessage("该点不属于当前步骤要求的下颌骨，未添加。")
            return False
        try:
            point_array([xyz], 1)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return False
        if mode == "angle" and len(self.pending) in (1, 3):
            if length_mm([self.pending[-1]["xyz_mm"], xyz]) < 1e-6:
                self.statusBar().showMessage("线段不能为零长度，请重新选择终点。")
                return False
        anchor = {"model": key, "xyz_mm": [float(value) for value in xyz], "triangle_id": int(cell_id)}
        self.pending.append(anchor)
        required = {"point": 1, "length": 2, "angle": 4}[mode]
        if len(self.pending) == required:
            points = [point["xyz_mm"] for point in self.pending]
            try:
                value = length_mm(points) if mode == "length" else segment_angle_degrees(points) if mode == "angle" else None
            except ValueError as exc:
                self.pending.pop()
                self.statusBar().showMessage(str(exc))
                return False
            self.measurements.append({
                "id": f"M{self._next_measurement:03d}", "type": mode,
                "anchors": list(self.pending), "value": value,
                "unit": "mm" if mode == "length" else "deg" if mode == "angle" else None,
                "segments": [[0, 1], [2, 3]] if mode == "angle" else [[0, 1]] if mode == "length" else [],
                "definition": "cross_bone_ordered_segments" if mode == "angle" else "manual_cross_bone_distance" if mode == "length" else "bone_surface_marker",
            })
            self._next_measurement += 1
            self.pending.clear()
            self._refresh_measurement_list()
            self.statusBar().showMessage("测量已记录。可继续取点，或切回浏览模式。")
        else:
            self.statusBar().showMessage(f"已取 {len(self.pending)}/{required} 点；请按提示继续，程序自动切换骨面。")
        self._update_measure_hint()
        self._redraw_measurements()
        return True

    def _refresh_measurement_list(self):
        self.measure_list.clear()
        for record in self.measurements:
            if record["type"] == "point":
                xyz = record["anchors"][0]["xyz_mm"]
                bone = "T0" if record["anchors"][0]["model"] == BONE_KEYS[0] else "T1"
                text = f"{bone} 标记：({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) mm"
            else:
                label = "骨间距离" if record["type"] == "length" else "两骨线段夹角"
                text = f"{label}：{record['value']:.3f} {record['unit']}"
            item = QListWidgetItem(text)
            item.setToolTip("\n".join(f"P{i + 1} {self.models[p['model']].title}: {p['xyz_mm']} mm" for i, p in enumerate(record["anchors"])))
            self.measure_list.addItem(item)

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
        records = [*self.measurements]
        if self.show_condyles.isChecked() and set(BONE_KEYS) <= self.models.keys():
            records.extend(self._condyle_measurements)
        if self.pending:
            records.append({"id": f"M{self._next_measurement:03d}", "anchors": self.pending, "value": None, "type": self.measure_mode.currentData()})
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

    def _undo(self):
        if self.pending:
            self.pending.pop()
        elif self.measurements:
            self.measurements.pop()
        self._refresh_measurement_list()
        self._redraw_measurements()
        self._update_measure_hint()

    def _delete_measurement(self):
        row = self.measure_list.currentRow()
        if 0 <= row < len(self.measurements):
            self.measurements.pop(row)
            self._refresh_measurement_list()
            self._redraw_measurements()

    def _clear_measurements(self):
        if not self.measurements and not self.pending:
            return
        if QMessageBox.question(self, "清空测量", "清空当前窗口的全部标记与测量？已导出的文件不会被删除。") != QMessageBox.StandardButton.Yes:
            return
        self.measurements.clear()
        self.pending.clear()
        self._refresh_measurement_list()
        self._redraw_measurements()
        self._update_measure_hint()

    def measurement_payload(self):
        return {
            "schema_version": 2, "created_at": datetime.now().astimezone().isoformat(),
            "project": str(self.scene.project_path) if self.scene else None,
            "coordinate_reference": self.scene.coordinate_reference if self.scene else None,
            "coordinate_units": "mm", "length_definition": "Euclidean distance between manually selected points on different mandibles; not automatic correspondence",
            "angle_definition": "3D angle between ordered segments P1->P2 and P3->P4 on different mandibles, range 0 to 180 deg",
            "models": {key: {"path": str(self.models[key].path), "sha256": self.models[key].sha256}
                       for key in BONE_KEYS if key in self.models},
            "measurements": self.measurements,
            "condyle_analysis": self.condyle_report,
        }

    def _save_measurements(self):
        if not self.measurements and not (self.condyle_report and self.condyle_report["regions"]):
            QMessageBox.information(self, "没有测量记录", "请先完成标记、长度或角度测量。")
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
        if self.measurements:
            answer = QMessageBox.question(self, "关闭查看器", "关闭后当前标记不会自动保存。确认关闭？如需保存请先导出测量记录。")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.vtk_widget.Finalize()
        super().closeEvent(event)


def view_project(path: str | Path, *, preset: str = "bones") -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SceneViewer(preset=preset)
    window.show()
    window.vtk_widget.Initialize()
    window.load_project(path)
    return app.exec()
