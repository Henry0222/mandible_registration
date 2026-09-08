from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from mandible_registration.drop_import import AssignStlDialog, DirectoryDropEdit, stl_drop_paths
from mandible_registration.gui import MainWindow
from mandible_registration.input_dialog import SequentialStlDialog
from mandible_registration.models import INPUT_SPECS


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def mime_for(paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def send_drop(widget, mime, point):
    enter = QDragEnterEvent(point, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, enter)
    drop = QDropEvent(QPointF(point), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, drop)
    return enter.isAccepted(), drop.isAccepted()


def test_drop_is_local_stl_only(tmp_path):
    good = tmp_path / "扫描.STL"
    good.write_bytes(b"test")
    bad = tmp_path / "other.txt"
    bad.write_bytes(b"test")
    assert stl_drop_paths(mime_for([good])) == [good]
    assert not stl_drop_paths(mime_for([good, bad]))
    assert not stl_drop_paths(mime_for([tmp_path]))
    remote = QMimeData()
    remote.setUrls([QUrl("https://example.com/test.stl")])
    assert not stl_drop_paths(remote)


def test_flow_drop_targets_role_and_locks_during_run(app, tmp_path, monkeypatch):
    file = tmp_path / "lower.stl"
    file.write_bytes(b"test")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window = MainWindow()
    window.show()
    app.processEvents()
    diagram = window.flow
    node = next(node for node in diagram.nodes if node.key == "baseline_upper")
    point = diagram.widget_point(node.rect.center()).toPoint()
    try:
        assert send_drop(diagram, mime_for([file]), point) == (True, True)
        assert window._paths == {"baseline_upper": file}
        assert not window.run_button.isEnabled()
        # A second role cannot use the same file.
        window._drop_inputs(0, [file])
        assert len(warnings) == 1
        assert "baseline_lower" not in window._paths
        window._thread = object()
        window._refresh_sequence()
        assert send_drop(diagram, mime_for([file]), point) == (False, False)
        assert window._paths == {"baseline_upper": file}
    finally:
        window._thread = None
        window.close()
        window.deleteLater()


def test_batch_mapping_never_uses_explorer_order(app, tmp_path, monkeypatch):
    files = []
    for name in ("z.stl", "a.stl", "c.stl"):
        path = tmp_path / name
        path.write_bytes(b"test")
        files.append(path)
    dialog = AssignStlDialog(files, {})
    assert all(combo.currentData() is None for combo in dialog.combos.values())
    dialog.combos["ct_mandible"].setCurrentIndex(1)
    dialog.accept()
    assert dialog.paths == {"ct_mandible": files[0]}
    dialog.deleteLater()


def test_directory_drop_and_file_parent_are_copy_only(app, tmp_path):
    edit = DirectoryDropEdit("unchanged")
    edit.show()
    app.processEvents()
    file = tmp_path / "test.stl"
    file.write_bytes(b"test")
    try:
        for target in (tmp_path, file):
            assert send_drop(edit, mime_for([target]), QPoint(5, 5)) == (True, True)
            assert Path(edit.text()) == tmp_path
            assert file.is_file()
        assert send_drop(edit, mime_for([tmp_path, file]), QPoint(5, 5)) == (False, False)
        remote = QMimeData()
        remote.setUrls([QUrl("https://example.com/file")])
        assert send_drop(edit, remote, QPoint(5, 5)) == (False, False)
        edit.setEnabled(False)
        edit.setText("locked")
        assert send_drop(edit, mime_for([tmp_path]), QPoint(5, 5)) == (False, False)
        assert edit.text() == "locked"
    finally:
        edit.close()
        edit.deleteLater()


def test_short_names_are_matched_when_batch_importing(app, tmp_path):
    paths = []
    for spec in INPUT_SPECS:
        path = tmp_path / f"{spec.title}.stl"
        path.write_bytes(b"test")
        paths.append(path)
    dialog = AssignStlDialog(list(reversed(paths)), {})
    try:
        dialog.accept()
        assert dialog.paths == {spec.key: path for spec, path in zip(INPUT_SPECS, paths)}
    finally:
        dialog.deleteLater()
    named = tmp_path / INPUT_SPECS[0].expected_filename
    named.write_bytes(b"test")
    dialog = AssignStlDialog([named], {})
    assert dialog.combos["baseline_lower"].currentData() == named
    dialog.reject()
    assert dialog.paths is None
    dialog.deleteLater()


def test_drop_six_in_same_dialog_and_cancel_does_not_commit(app, tmp_path):
    dialog = SequentialStlDialog(directory=str(tmp_path))
    dialog.show()
    app.processEvents()
    files = []
    for index in range(6):
        path = tmp_path / f"{index}.stl"
        path.write_bytes(b"test")
        files.append(path)
        assert send_drop(dialog.drop_zone, mime_for([path]), QPoint(10, 10)) == (True, True)
        app.processEvents()
        assert dialog.isVisible() == (index < 5)
    assert list(dialog.inputs.as_mapping().values()) == files
    dialog.deleteLater()
    dialog = SequentialStlDialog(directory=str(tmp_path))
    dialog.accept_path(files[0])
    dialog.reject()
    assert dialog.inputs is None
    dialog.deleteLater()
