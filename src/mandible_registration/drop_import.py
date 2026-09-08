from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QMessageBox, QTableWidget, QVBoxLayout,
)

from .models import INPUT_SPECS
from .theme import apply_light_theme


def stl_drop_paths(mime) -> list[Path]:
    """Accept local files only; never interpret a drop as a filesystem move."""
    if not mime.hasUrls():
        return []
    urls = mime.urls()
    if not urls or any(not url.isLocalFile() for url in urls):
        return []
    paths = [Path(url.toLocalFile()).resolve() for url in urls]
    if any(not path.is_file() or path.suffix.lower() != ".stl" for path in paths):
        return []
    return list(dict.fromkeys(paths))


def validate_assignments(paths: dict[str, Path]) -> dict[str, Path]:
    result = {key: Path(value).resolve() for key, value in paths.items()}
    if set(result) - {spec.key for spec in INPUT_SPECS}:
        raise ValueError("包含未知模型类型。")
    for path in result.values():
        if not path.is_file() or path.suffix.lower() != ".stl":
            raise ValueError(f"请选择存在的 STL 文件：{path}")
    if len(set(result.values())) != len(result):
        raise ValueError("同一个 STL 不能同时用于两个输入角色，请检查文件分配。")
    return result


class StlDropTable(QTableWidget):
    files_dropped = Signal(int, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.import_enabled = True
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

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
        if self.import_enabled and paths:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self.files_dropped.emit(self.rowAt(int(event.position().y())), paths)
        else:
            event.ignore()


class StlDropZone(QLabel):
    file_dropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__("也可将当前这一步的 STL 拖到这里（每次一个）", parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(42)
        self.setStyleSheet("border: 2px dashed #719ec5; padding: 8px; background: #edf5ff;")

    def dragEnterEvent(self, event):
        if len(stl_drop_paths(event.mimeData())) == 1:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = stl_drop_paths(event.mimeData())
        if len(paths) == 1:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self.file_dropped.emit(paths[0])
        else:
            event.ignore()


class DirectoryDropEdit(QLineEdit):
    """Accept a local directory URL, or the containing directory of a file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    @staticmethod
    def directory_from_mime(mime):
        if not mime.hasUrls() or len(mime.urls()) != 1 or not mime.urls()[0].isLocalFile():
            return None
        path = Path(mime.urls()[0].toLocalFile()).resolve()
        if path.is_dir():
            return path
        return path.parent if path.is_file() else None

    def dragEnterEvent(self, event):
        if self.isEnabled() and self.directory_from_mime(event.mimeData()) is not None:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        directory = self.directory_from_mime(event.mimeData())
        if self.isEnabled() and directory is not None:
            self.setText(str(directory))
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

class AssignStlDialog(QDialog):
    """Explicit role mapping: Explorer's multi-selection order is not meaningful."""

    def __init__(self, paths: list[Path], existing: dict[str, Path], parent=None):
        apply_light_theme()
        super().__init__(parent)
        self.setWindowTitle("确认拖入文件对应的模型")
        self.resize(850, 370)
        self.paths: dict[str, Path] | None = None
        layout = QVBoxLayout(self)
        hint = QLabel("请为文件指定用途，不按拖入顺序猜测上下颌。只自动匹配约定文件名；空白表示暂不选择。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        layout.addLayout(form)
        choices = list(dict.fromkeys([*existing.values(), *paths]))
        self.combos = {}
        for spec in INPUT_SPECS:
            combo = QComboBox()
            combo.addItem("尚未选择", None)
            for path in choices:
                combo.addItem(str(path), path)
            matches = [p for p in paths if p.name.casefold() == spec.expected_filename.casefold() or p.stem.casefold() == spec.title.casefold()]
            selected = matches[0] if len(matches) == 1 else existing.get(spec.key)
            if selected in choices:
                combo.setCurrentIndex(choices.index(selected) + 1)
            self.combos[spec.key] = combo
            form.addRow(f"{spec.sequence}. {spec.title}", combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        values = {key: combo.currentData() for key, combo in self.combos.items() if combo.currentData()}
        try:
            self.paths = validate_assignments(values)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "请检查文件分配", str(exc))
            return
        super().accept()
