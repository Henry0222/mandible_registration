from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .models import INPUT_SPECS, StudyInputs
from .drop_import import StlDropZone, validate_assignments
from .theme import apply_light_theme


class SequentialStlDialog(QFileDialog):
    """Keep one file browser open while assigning the six ordered STL roles.

    A native QFileDialog closes when a file is accepted. The widget-based
    dialog lets us keep its folder/navigation state and advance the prompt
    without closing the window. Nothing is committed to the caller on cancel.
    """

    def __init__(self, parent: QWidget | None = None, *, directory: str = "", initial_paths: dict[str, Path] | None = None) -> None:
        apply_light_theme()
        super().__init__(parent)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        self.setViewMode(QFileDialog.ViewMode.Detail)
        self.setNameFilter("STL 三角网格 (*.stl *.STL)")
        self.setLabelText(QFileDialog.DialogLabel.LookIn, "查找范围：")
        self.setLabelText(QFileDialog.DialogLabel.FileName, "文件名：")
        self.setLabelText(QFileDialog.DialogLabel.FileType, "文件类型：")
        self.setLabelText(QFileDialog.DialogLabel.Reject, "取消导入")
        if directory:
            self.setDirectory(directory)
        self.resize(1100, 760)
        tree = self.findChild(QTreeView, "treeView")
        if tree is not None:
            tree.header().setStretchLastSection(False)
            tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for column in range(1, tree.model().columnCount()):
                tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self._step = 0
        self._paths: dict[str, Path] = dict(initial_paths or {})
        self.paths: dict[str, Path] | None = None
        self.inputs: StudyInputs | None = None

        progress_group = QGroupBox("模型导入（暂缺的可以跳过）", self)
        progress_layout = QVBoxLayout(progress_group)
        self.stage_label = QLabel()
        self.stage_label.setTextFormat(Qt.TextFormat.PlainText)
        self.stage_label.setWordWrap(True)
        self.stage_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.selection_summary = QLabel()
        self.selection_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.selection_summary.setWordWrap(True)
        progress_layout.addWidget(self.stage_label)
        progress_layout.addWidget(self.selection_summary)
        self.drop_zone = StlDropZone()
        self.drop_zone.file_dropped.connect(self.accept_path)
        progress_layout.addWidget(self.drop_zone)
        # Drops import through our dedicated zone; the browser must never move files.
        for view in self.findChildren(QAbstractItemView):
            view.setAcceptDrops(False)
            view.setDragEnabled(False)
        bottom = QHBoxLayout()
        self.back_button = QPushButton("上一步")
        self.back_button.clicked.connect(self._go_back)
        bottom.addWidget(self.back_button)
        self.skip_button = QPushButton("暂缺，跳过")
        self.skip_button.clicked.connect(self._advance)
        bottom.addWidget(self.skip_button)
        self.save_button = QPushButton("保存已选项")
        self.save_button.clicked.connect(self._finish_import)
        bottom.addWidget(self.save_button)
        bottom.addWidget(QLabel("取消不改变已有选择。"), 1)
        progress_layout.addLayout(bottom)

        grid = self.layout()
        grid.addWidget(progress_group, grid.rowCount(), 0, 1, grid.columnCount())
        self._show_step()

    def _show_step(self) -> None:
        spec = INPUT_SPECS[self._step]
        self.setWindowTitle(f"连续选择 STL — 第 {spec.sequence}/6 项：{spec.title}")
        self.stage_label.setText(f"{spec.sequence}/6　{spec.title}")
        self.setLabelText(
            QFileDialog.DialogLabel.Accept,
            "确认并返回" if self._step == len(INPUT_SPECS) - 1 else "确认，下一项",
        )
        self.back_button.setEnabled(self._step > 0)
        lines = []
        for item in INPUT_SPECS:
            selected = self._paths.get(item.key)
            name = selected.name if selected is not None else "未选择"
            marker = "→" if item.sequence == spec.sequence else "  "
            lines.append(f"{marker} {item.sequence}. {item.title}：{name}")
        self.selection_summary.setText("\n".join(lines))
        current = self._paths.get(spec.key)
        if current is not None:
            self.setDirectory(str(current.parent))
            self.selectFile(str(current))
        else:
            # Clearing only the filename preserves the browser directory.
            for view in self.findChildren(QAbstractItemView):
                if view.objectName() in ("listView", "treeView"):
                    view.selectionModel().clear()
            filename_edit = self.findChild(QLineEdit, "fileNameEdit")
            if filename_edit is not None:
                filename_edit.clear()

    def _go_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._show_step()

    def accept(self) -> None:
        filenames = self.selectedFiles()
        if len(filenames) != 1:
            QMessageBox.warning(self, "请选择一个 STL", "当前步骤只能指定一个 STL 文件。")
            return
        path = Path(filenames[0]).resolve()
        if path.is_dir():
            self.setDirectory(str(path))
            return
        self.accept_path(path)

    def accept_path(self, path: Path) -> None:
        path = Path(path).resolve()
        if not path.is_file() or path.suffix.lower() != ".stl":
            QMessageBox.warning(self, "文件无效", "请选择存在的 STL 文件。")
            return
        spec = INPUT_SPECS[self._step]
        for key, existing in self._paths.items():
            if key != spec.key and existing == path:
                title = next(item.title for item in INPUT_SPECS if item.key == key)
                QMessageBox.warning(self, "文件重复", f"这个文件已经用于「{title}」，请另选一个文件。")
                return
        self._paths[spec.key] = path
        self._advance()

    def _advance(self) -> None:
        if self._step < len(INPUT_SPECS) - 1:
            self._step += 1
            self._show_step()
            return
        self._finish_import()

    def _finish_import(self) -> None:
        try:
            paths = validate_assignments(self._paths)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "文件校验未通过", str(exc))
            return
        self.paths = paths
        self.inputs = StudyInputs.from_mapping(paths) if len(paths) == len(INPUT_SPECS) else None
        QDialog.accept(self)
