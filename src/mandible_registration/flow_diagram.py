from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPathStroker
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolButton, QToolTip, QWidget

from .drop_import import stl_drop_paths
from .svg_support import set_svg_properties, svg_shape_path


STAGE_COLORS = {"waiting": "#c6cdd6", "running": "#287db8", "success": "#199958", "warning": "#e29022", "failed": "#d44545"}
SELECTION_BUTTON_STYLE = """
QToolButton {
    color: #287db8; background: #f0f7ff; border: 1px solid #c8dbea;
    border-radius: 4px; padding: 1px;
}
QToolButton[selectionSaved="true"] {
    color: #117444; background: #e9f8ef; border-color: #72bf91; font-weight: 600;
}
QToolButton:disabled {
    color: #a9b3be; background: #f4f6f8; border-color: #e1e6ec;
}
"""


@dataclass(frozen=True)
class FlowNode:
    key: str
    title: str
    rect: QRectF
    is_input: bool = True


class WorkflowDiagram(QWidget):
    """Clickable, drop-enabled dependency diagram; all coordinates are logical."""

    choose_input = Signal(str)
    clear_input = Signal(str)
    files_dropped = Signal(str, object)
    stage_clicked = Signal(str)
    compare_clicked = Signal()
    condyle_requested = Signal()
    registration_selection_requested = Signal(str)

    def __init__(self, parent=None, *, asset_path=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumHeight(400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.import_enabled = True
        self.paths: dict[str, Path] = {}
        self.outputs: set[str] = set()
        self.stage_states: dict[str, str] = {}
        self.stage_tooltips: dict[str, str] = {}
        self.review_paths: dict[str, Path] = {}
        self.selection_statuses: set[str] = set()
        self.asset_path = (
            Path(asset_path) if asset_path
            else Path(__file__).with_name("assets") / "workflow.maxilla.editable.svg"
        )
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self._document = ET.parse(self.asset_path).getroot()
        self._elements = {element.get("id"): element for element in self._document.iter() if element.get("id")}
        self._renderer = QSvgRenderer(self)
        self._rebuild_svg()
        self._view_box = self._renderer.viewBoxF()
        self.nodes = []
        for element in self._document.iter():
            if element.get("data-key"):
                key = element.get("data-key")
                title = "".join(self._elements[f"label-{key}"].itertext())
                self.nodes.append(FlowNode(key, title, self._element_rect(f"box-{key}"), element.get("data-kind") == "input"))
        self.stage_paths = {}
        self.stage_labels = {}
        for key in ("T_CT", "T_UPPER", "T_DELTA"):
            element_id = f"stage-{key}"
            element = self._elements.get(element_id)
            if element is None:
                raise ValueError(f"流程图缺少必要连线 ID：{element_id}")
            # Inkscape can replace a polyline with a path and remove data-stage.
            # The stable ID still identifies the stage without silently losing it.
            path = svg_shape_path(element)
            self.stage_paths[key] = self._renderer.transformForElement(element_id).map(path)
            self.stage_labels[key] = self._element_rect(f"hit-label-{key}")
        self.condyle_button = QToolButton(self)
        self.condyle_button.setText("选区")
        self.condyle_button.setToolTip("导入颌骨 STL 后，可分别选取左右髁突面片。")
        self.condyle_button.setProperty("emptySelectionTooltip", self.condyle_button.toolTip())
        self.condyle_button.setEnabled(False)
        self.condyle_button.clicked.connect(self.condyle_requested.emit)
        self.condyle_button.setStyleSheet(SELECTION_BUTTON_STYLE)
        self.registration_buttons = {}
        for key, title in (("baseline_lower", "下颌口扫.1"), ("ct_dentition", "全牙列")):
            button = QToolButton(self)
            button.setText("选区")
            button.setToolTip(
                f"为{title}选择一个 T_CT 配准重点区域；未选择时仍使用自动配准。"
            )
            button.setProperty("emptySelectionTooltip", button.toolTip())
            button.setEnabled(False)
            button.clicked.connect(
                lambda checked=False, value=key: self.registration_selection_requested.emit(value)
            )
            button.setStyleSheet(self.condyle_button.styleSheet())
            self.registration_buttons[key] = button

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for key, button in {"ct_mandible": self.condyle_button, **self.registration_buttons}.items():
            node = next(node for node in self.nodes if node.key == key)
            rect = QRectF(self.widget_point(node.rect.topLeft()), self.widget_point(node.rect.bottomRight()))
            width, height = min(42, rect.width() * .3), min(24, rect.height() - 6)
            button.setGeometry(
                round(rect.right() - width - 5),
                round(rect.center().y() - height / 2),
                round(width),
                round(height),
            )

    def _element_rect(self, element_id):
        return self._renderer.transformForElement(element_id).mapRect(self._renderer.boundsOnElement(element_id))

    def _rebuild_svg(self):
        self.svg_bytes = ET.tostring(self._document, encoding="utf-8")
        if not self._renderer.load(QByteArray(self.svg_bytes)):
            raise ValueError("流程图 SVG 无法加载。")

    def _refresh_svg_states(self):
        for node in self.nodes:
            ready = self.node_ready(node)
            label = self._elements[f"label-{node.key}"]
            set_svg_properties(label, fill="#15191f" if ready else "#8995a4", opacity="1" if ready else "0.55")
            set_svg_properties(self._elements[f"box-{node.key}"], stroke="#bcd9c9" if ready else "#dce4ed")
            for prefix, visible in (("ready", ready), ("missing", not ready)):
                icon = self._elements.get(f"{prefix}-{node.key}")
                if icon is not None:
                    set_svg_properties(icon, opacity="1" if visible else "0")
        status_text = {"waiting": "等待配准", "running": "配准中…", "success": "通过", "warning": "需复核", "failed": "未通过"}
        for key in self.stage_paths:
            state = self.stage_states.get(key, "waiting")
            color = STAGE_COLORS.get(state, STAGE_COLORS["waiting"])
            set_svg_properties(self._elements[f"stage-{key}"], stroke=color)
            set_svg_properties(self._elements[f"arrow-{key}"], fill=color)
            label = self._elements[f"state-{key}"]
            label.text = status_text.get(state, "等待配准") + (" · 点击彩虹图" if key in self.review_paths else "")
            set_svg_properties(label, fill=color if state != 'waiting' else '#8592a3')
        self._rebuild_svg()
        self.update()

    def node_ready(self, node):
        if node.key == "comparison":
            return {"ct_mandible_t0", "ct_mandible_t1"} <= self.outputs
        return node.key in (self.paths if node.is_input else self.outputs)

    def set_inputs(self, paths):
        self.paths = dict(paths)
        self._refresh_action_buttons()
        self._refresh_svg_states()

    def set_selection_statuses(self, keys):
        self.selection_statuses = set(keys)
        self._refresh_action_buttons()

    def set_import_enabled(self, enabled):
        self.import_enabled = bool(enabled)
        self._refresh_action_buttons()

    def _refresh_action_buttons(self):
        self.condyle_button.setEnabled("ct_mandible" in self.paths and self.import_enabled)
        for key, button in self.registration_buttons.items():
            button.setEnabled(key in self.paths and self.import_enabled)
        labels = {
            "ct_mandible": (self.condyle_button, "颌骨髁突选区"),
            "baseline_lower": (self.registration_buttons["baseline_lower"], "下颌口扫.1配准区"),
            "ct_dentition": (self.registration_buttons["ct_dentition"], "全牙列配准区"),
        }
        for key, (button, label) in labels.items():
            saved = key in self.selection_statuses and key in self.paths
            button.setProperty("selectionSaved", saved)
            button.setText("已选" if saved else "选区")
            button.setToolTip(
                f"已有{label}，点击可重新编辑；配准时仍会核对 STL 哈希。"
                if saved else button.property("emptySelectionTooltip") or button.toolTip()
            )
            button.style().unpolish(button)
            button.style().polish(button)

    def set_results(self, states=None, reviews=None, outputs=(), tooltips=None):
        self.stage_states = dict(states or {})
        self.review_paths = dict(reviews or {})
        self.outputs = set(outputs)
        self.stage_tooltips = dict(tooltips or {})
        self._refresh_svg_states()

    def _scale(self):
        return min(self.width() / self._view_box.width(), self.height() / self._view_box.height())

    def _offset(self):
        scale = self._scale()
        return QPointF((self.width() - self._view_box.width() * scale) / 2,
                       (self.height() - self._view_box.height() * scale) / 2)

    def logical_point(self, point):
        return (QPointF(point) - self._offset()) / self._scale() + self._view_box.topLeft()

    def widget_point(self, point):
        return (QPointF(point) - self._view_box.topLeft()) * self._scale() + self._offset()

    def node_at(self, point):
        return next((node for node in self.nodes if node.rect.contains(point)), None)

    def stage_at(self, point):
        stroker = QPainterPathStroker()
        stroker.setWidth(25)
        for key, path in self.stage_paths.items():
            label = self.stage_labels[key]
            if stroker.createStroke(path).contains(point) or (key in self.review_paths and label.contains(point)):
                return key
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = self._view_box.size() * self._scale()
        self._renderer.render(painter, QRectF(self._offset(), size))

    def mousePressEvent(self, event):
        point = self.logical_point(event.position())
        node = self.node_at(point)
        if node and node.is_input and self.import_enabled:
            if event.button() == Qt.MouseButton.LeftButton:
                self.choose_input.emit(node.key)
            elif event.button() == Qt.MouseButton.RightButton:
                menu = QMenu(self)
                choose = menu.addAction("选择文件")
                clear = menu.addAction("清除此项")
                clear.setEnabled(node.key in self.paths)
                selected = menu.exec(event.globalPosition().toPoint())
                if selected == choose:
                    self.choose_input.emit(node.key)
                elif selected == clear:
                    self.clear_input.emit(node.key)
        elif event.button() == Qt.MouseButton.LeftButton:
            if node and node.key == "comparison" and self.node_ready(node):
                self.compare_clicked.emit()
            else:
                key = self.stage_at(point)
                if key in self.review_paths:
                    self.stage_clicked.emit(key)

    def mouseMoveEvent(self, event):
        point = self.logical_point(event.position())
        node = self.node_at(point)
        stage = self.stage_at(point)
        clickable = bool(node and (node.is_input and self.import_enabled or node.key == "comparison" and self.node_ready(node)) or stage in self.review_paths)
        self.setCursor(Qt.CursorShape.PointingHandCursor if clickable else Qt.CursorShape.ArrowCursor)
        if node:
            if node.key == "comparison":
                text = "点击打开颌骨对比和测量" if self.node_ready(node) else "完成配准后可对比测量"
            else:
                text = str(self.paths.get(node.key, "点击导入或拖入 STL")) if node.is_input else "由配准矩阵生成，不需要导入"
        elif stage:
            text = self.stage_tooltips.get(stage, "完成配准后，可点击粗线查看对应彩虹图")
        else:
            text = ""
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def dragEnterEvent(self, event):
        if self.import_enabled and stl_drop_paths(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = stl_drop_paths(event.mimeData())
        if not self.import_enabled or not paths:
            event.ignore()
            return
        node = self.node_at(self.logical_point(event.position()))
        key = node.key if node and node.is_input and len(paths) == 1 else ""
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        self.files_dropped.emit(key, paths)
