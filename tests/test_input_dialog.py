from pathlib import Path

import pytest

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QTreeView,
)

from mandible_registration.input_dialog import SequentialStlDialog
from mandible_registration.models import INPUT_SPECS


@pytest.fixture(scope="module")
def app():
    # No desktop window or native file chooser is needed for widget tests.
    instance = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    if not QFontDatabase.families():
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        if font_id >= 0:
            instance.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
    yield instance


@pytest.fixture
def files(tmp_path):
    result = []
    for name in ("z", "b", "d", "a", "f", "c"):
        path = tmp_path / f"{name}.stl"
        path.write_bytes(b"selection test")
        result.append(path)
    return result


def test_six_accept_clicks_keep_one_window_and_preserve_role_order(app, files, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = SequentialStlDialog(directory=str(files[0].parent))
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    app.processEvents()
    filename_edit = dialog.findChild(QLineEdit, "fileNameEdit")
    button_box = dialog.findChild(QDialogButtonBox)
    open_button = button_box.button(QDialogButtonBox.StandardButton.Open)
    try:
        for index, path in enumerate(files):
            filename_edit.setFocus()
            filename_edit.selectAll()
            QTest.keyClicks(filename_edit, path.as_posix())
            app.processEvents()
            assert open_button.isEnabled()
            assert dialog.selectedFiles() == [str(path).replace("\\", "/")]
            QTest.mouseClick(open_button, Qt.MouseButton.LeftButton)
            app.processEvents()
            assert not warnings
            if index < 5:
                assert dialog.isVisible()
                assert not accepted
                assert dialog._step == index + 1
                assert filename_edit.text() == ""
                assert dialog.directory().absolutePath() == files[0].parent.as_posix()
        assert accepted == [True]
        assert not dialog.isVisible()
        assert list(dialog.inputs.as_mapping().values()) == files
    finally:
        dialog.close()
        dialog.deleteLater()


def test_duplicate_stays_on_current_step_and_back_can_replace(app, files, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = SequentialStlDialog(directory=str(files[0].parent))
    try:
        dialog.selectFile(str(files[0]))
        dialog.accept()
        dialog.selectFile(str(files[0]))
        dialog.accept()
        assert dialog._step == 1
        assert len(warnings) == 1
        dialog._go_back()
        dialog.selectFile(str(files[1]))
        dialog.accept()
        assert dialog._paths[INPUT_SPECS[0].key] == files[1]
        dialog.reject()
        assert dialog.inputs is None
        assert dialog.result() == QFileDialog.DialogCode.Rejected
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("confirm_with", ("button", "double_click"))
def test_selecting_from_file_list_advances_in_the_same_window(app, files, monkeypatch, confirm_with):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = SequentialStlDialog(directory=str(files[0].parent))
    dialog.show()
    tree = dialog.findChild(QTreeView, "treeView")
    button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Open)
    try:
        for step, path in enumerate(files):
            for _ in range(40):
                app.processEvents()
                index = QModelIndex()
                for row in range(tree.model().rowCount(tree.rootIndex())):
                    candidate = tree.model().index(row, 0, tree.rootIndex())
                    if Path(tree.model().filePath(candidate)) == path:
                        index = candidate
                        break
                tree.doItemsLayout()
                if tree.visualRect(index).isValid():
                    break
                QTest.qWait(25)
            tree.scrollTo(index)
            app.processEvents()
            rect = tree.visualRect(index)
            assert rect.isValid(), str((
                step, index.isValid(), tree.isVisible(),
                tree.model().filePath(tree.rootIndex()),
                tree.model().filePath(index), tree.model().rowCount(tree.rootIndex()),
                index.row(), index.column(), index.parent() == tree.rootIndex(),
                tree.columnWidth(0), tree.size(), tree.isRowHidden(index.row(), index.parent()),
            ))
            QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
            app.processEvents()
            assert [Path(value) for value in dialog.selectedFiles()] == [path]
            if confirm_with == "double_click":
                QTest.mouseDClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
            else:
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            app.processEvents()
            assert not warnings
            assert dialog.isVisible() == (step < 5)
        assert list(dialog.inputs.as_mapping().values()) == files
    finally:
        dialog.close()
        dialog.deleteLater()


def test_main_window_commits_only_accepted_sequence(app, files, monkeypatch):
    from mandible_registration import gui
    from mandible_registration.models import StudyInputs

    window = gui.MainWindow()
    original = {INPUT_SPECS[0].key: files[0]}
    window._apply_input_paths(original)
    new_inputs = StudyInputs.from_mapping(
        {spec.key: path for spec, path in zip(INPUT_SPECS, reversed(files))}
    )

    class StubDialog:
        response = QFileDialog.DialogCode.Rejected
        inputs = new_inputs
        paths = new_inputs.as_mapping()

        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return self.response

        def deleteLater(self):
            pass

    monkeypatch.setattr(gui, "SequentialStlDialog", StubDialog)
    try:
        window._choose_six_inputs()
        assert window._paths == original
        StubDialog.response = QFileDialog.DialogCode.Accepted
        window._choose_six_inputs()
        assert window._paths == new_inputs.as_mapping()
        assert window.run_button.isEnabled()
        assert window.flow.paths[INPUT_SPECS[0].key] == files[-1]
        window._thread = object()
        window._refresh_sequence()
        assert not window.sequence_button.isEnabled()
        window._thread = None
    finally:
        window.close()
        window.deleteLater()


def test_skip_missing_roles_and_save_partial_selection(app, files):
    dialog = SequentialStlDialog(directory=str(files[0].parent))
    dialog.show()
    app.processEvents()
    try:
        QTest.mouseClick(dialog.skip_button, Qt.MouseButton.LeftButton)
        assert dialog._step == 1
        dialog.accept_path(files[1])
        for _ in range(4):
            QTest.mouseClick(dialog.skip_button, Qt.MouseButton.LeftButton)
        assert dialog.result() == QFileDialog.DialogCode.Accepted
        assert dialog.paths == {"baseline_upper": files[1]}
        assert dialog.inputs is None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_partial_save_and_cancel_preserve_initial_mapping(app, files):
    original = {"baseline_lower": files[0]}
    for commit in (False, True):
        dialog = SequentialStlDialog(directory=str(files[0].parent), initial_paths=original)
        try:
            dialog._advance()
            dialog.accept_path(files[1])
            if commit:
                dialog._finish_import()
                assert dialog.paths == {**original, "baseline_upper": files[1]}
                assert dialog.inputs is None
            else:
                dialog.reject()
                assert dialog.paths is None
            assert original == {"baseline_lower": files[0]}
        finally:
            dialog.deleteLater()


def test_registration_requires_all_six_and_clearing_invalidates_results(app, files, monkeypatch):
    from mandible_registration.gui import MainWindow

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window = MainWindow()
    try:
        for count in range(7):
            window._apply_input_paths({spec.key: path for spec, path in zip(INPUT_SPECS[:count], files)})
            assert window.run_button.isEnabled() == (count == 6)
            if count < 6:
                window._start()  # Guard is also enforced when called programmatically.
                assert window._thread is None
        assert len(warnings) == 6
        window._project_path = Path("project.json")
        window._review_paths = {"T_CT": Path("results.json")}
        window.flow.set_results({"T_CT": "success"}, window._review_paths, ["ct_mandible_t0"])
        window._clear_input("ct_dentition")
        assert len(window._paths) == 5
        assert not window.run_button.isEnabled()
        assert not window.view_button.isEnabled()
        assert not window.flow.outputs and not window.flow.review_paths
    finally:
        window.close()
        window.deleteLater()
